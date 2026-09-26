"""Local CLI: triage one alert file without deploying.

    python -m triage.cli samples/alerts/cluster_red_prod.json
    python -m triage.cli samples/alerts/cpu_spike_nonprod.json --decide-only
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .config import load_settings
from .pipeline import TriageOptions, TriagePipeline


def main() -> None:
    ap = argparse.ArgumentParser(description="Triage an OpenSearch alert locally.")
    ap.add_argument("alert_file", type=Path)
    ap.add_argument("--source", choices=["grafana", "pagerduty", "generic"])
    ap.add_argument("--decide-only", action="store_true", help="Jev + policy only (no Bedrock calls)")
    ap.add_argument("--investigate-watch", action="store_true")
    ap.add_argument("--no-memory", action="store_true")
    ap.add_argument("--json", action="store_true", help="print full JSON result")
    a = ap.parse_args()

    result = TriagePipeline(load_settings()).run(
        json.loads(a.alert_file.read_text()),
        a.source,
        TriageOptions(decide_only=a.decide_only, investigate_watch=a.investigate_watch, use_memory=not a.no_memory),
    )
    if a.json:
        json.dump(result, sys.stdout, indent=2, default=str)
        print()
        return

    d = result["decision"]
    print(f"\n[jev]      tier={d['jev_tier']} conf={d['jev_tier_confidence']} blast={d['blast_radius']} "
          f"transient={d['likely_transient']} human_now={d['needs_human_now']} "
          f"({d['jev_latency_ms']} ms, ${d['jev_cost_usd']:.6f})")
    print(f"[policy]   tier={d['tier']} action={d['action']} rule={d['rule_id']} ({d['rule_match']})")
    for r in d["reasons"]:
        print(f"           - {r}")
    inv = result["investigation"]
    if inv:
        print(f"[bedrock]  {inv['model']} in={inv['input_tokens']} out={inv['output_tokens']} "
              f"~${inv['model_cost_usd']:.6f} tools={inv['tools_used']}")
        print("\n" + inv["report_markdown"])
    print(f"[total]    ~${result['cost_usd_total']:.6f} in {result['elapsed_ms']} ms\n")


if __name__ == "__main__":
    main()
