"""Evaluate the triage decision layer against labeled alerts.

    # Jev + policy only (cheap, no Bedrock): tier/action agreement, under-triage, latency
    python -m triage.evaluate --dry-run

    # Also run the investigation twice per alert: routed tier vs always-L3 (Sonnet)
    python -m triage.evaluate --limit 4

Under-triage (final tier below the label) is reported separately because it is
the error that matters operationally; over-triage only costs money.
Results are written to results/eval-<timestamp>.json.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

from .agent import investigate, tier_specs
from .alerts import normalize
from .config import load_settings
from .credentials import get_jev_api_key
from .decision import decide
from .jev_client import JevClient
from .runbook import Runbook, Tier


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=Path, default=Path("samples/labeled_alerts.json"))
    ap.add_argument("--dry-run", action="store_true", help="decision layer only")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    s = load_settings()
    runbook = Runbook.load()
    jev = JevClient(api_key=get_jev_api_key(s), url=s.jev_api_url, model=s.jev_model)
    rows = json.loads(a.dataset.read_text())
    if a.limit:
        rows = rows[: a.limit]

    results, latencies = [], []
    routed_cost = frontier_cost = jev_cost = 0.0
    frontier = tier_specs(s)[Tier.L3]

    print(f"{'alert':32} {'label':>6} {'jev':>4} {'final':>5} {'conf':>5} {'action':>11} {'ok':>3}")
    for row in rows:
        alert = normalize(row["payload"])
        d = decide(alert, runbook, s, jev)
        want = Tier[row["expected_tier"]]
        tier_ok = d.tier == want
        under = d.tier < want
        action_ok = d.action == row["expected_action"]
        if d.jev_ok:
            latencies.append(d.jev_latency_ms)
        jev_cost += d.jev_cost_usd
        rec = {"name": row["name"], "expected_tier": want.name, "expected_action": row["expected_action"],
               "decision": d.to_dict(), "tier_ok": tier_ok, "under_triage": under, "action_ok": action_ok}

        if not a.dry_run and d.action != "record_only":
            rule = runbook.get(d.rule_id)
            # Baseline: an always-frontier agent investigates every firing alert.
            always = investigate(alert, d, rule, [], s, spec_override=frontier)
            frontier_cost += always.model_cost_usd
            rec["frontier"] = {"model": always.model_label, "cost": always.model_cost_usd}
            if d.action == "watch":  # routed mode skips the LLM for watch alerts
                rec["routed"] = {"model": None, "cost": 0.0}
            else:
                routed = always if d.tier == Tier.L3 else investigate(alert, d, rule, [], s)
                routed_cost += routed.model_cost_usd
                rec["routed"] = {"model": routed.model_label, "cost": routed.model_cost_usd}
        results.append(rec)
        mark = "✓" if tier_ok and action_ok else ("✗!" if under else "✗")
        print(f"{row['name'][:32]:32} {want.name:>6} {d.jev_tier or '-':>4} {d.tier.name:>5} "
              f"{(d.jev_tier_confidence or 0):5.2f} {d.action:>11} {mark:>3}")

    n = len(results)
    summary = {
        "n": n,
        "tier_accuracy": sum(r["tier_ok"] for r in results) / n,
        "action_accuracy": sum(r["action_ok"] for r in results) / n,
        "under_triage": sum(r["under_triage"] for r in results),
        "jev_latency_ms_p50": statistics.median(latencies) if latencies else None,
        "jev_latency_ms_max": max(latencies) if latencies else None,
        "jev_cost_usd": round(jev_cost, 6),
    }
    if not a.dry_run:
        summary |= {
            "routed_model_cost_usd": round(routed_cost, 6),
            "routed_total_cost_usd": round(routed_cost + jev_cost, 6),
            "always_frontier_cost_usd": round(frontier_cost, 6),
            "note": "frontier baseline investigates every firing alert with L3; routed skips 'watch' alerts. Prices are approximate.",
        }
    print("\n" + json.dumps(summary, indent=2))
    out = Path("results")
    out.mkdir(exist_ok=True)
    path = out / f"eval-{time.strftime('%Y%m%d-%H%M%S')}.json"
    path.write_text(json.dumps({"summary": summary, "results": results}, indent=2, default=str))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
