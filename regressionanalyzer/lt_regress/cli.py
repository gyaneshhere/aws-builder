"""Local CLI.

  python -m lt_regress.cli samples/psu_sinewave_vs_prod.json --local-stats --decide-only
  python -m lt_regress.cli samples/psu_sinewave_vs_prod.json                 # Code Interpreter + Claude
  python -m lt_regress.cli --csv lt.csv prod.csv meta.json
"""

from __future__ import annotations

import argparse
import json
import sys

from .config import load_settings
from .models import load_csv, load_json
from .pipeline import Analyzer, RunOptions

ICON = {"regression": "✗", "watch": "!", "explained": "~", "improvement": "↑", "noise": "·", "insufficient_data": "?"}


def _fmt(v, spec: str) -> str:
    return "-" if v is None else format(v, spec)


def main() -> None:
    ap = argparse.ArgumentParser(description="Compare an LT run against Production.")
    ap.add_argument("input", nargs="?", help="comparison JSON")
    ap.add_argument("--csv", nargs=3, metavar=("LT_CSV", "PROD_CSV", "META_JSON"))
    ap.add_argument("--local-stats", action="store_true", help="run the kernel in-process, not in Code Interpreter")
    ap.add_argument("--decide-only", action="store_true", help="no narrative LLM, no memory writes")
    ap.add_argument("--accept-baseline", action="store_true")
    ap.add_argument("--no-memory", action="store_true")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()
    if not a.input and not a.csv:
        ap.error("give a comparison JSON or --csv")

    comp = load_csv(*a.csv) if a.csv else load_json(a.input)
    res = Analyzer(load_settings()).run(comp, RunOptions(
        decide_only=a.decide_only, accept_baseline=a.accept_baseline, use_memory=not a.no_memory,
        stats_engine="local" if a.local_stats else None))
    if a.json:
        json.dump(res, sys.stdout, indent=2, default=str)
        print()
        return

    d = res["decision"]
    stats = {m["name"]: m for m in res["statistics"]}
    print(f"\n{comp.test_name} · {comp.run_id}   engine={res['stats_engine']}   "
          f"jev_calls={d['jev_calls']} ({d['jev_latency_ms']} ms max, ${d['jev_cost_usd']:.6f})")
    print(f"{'':2}{'metric':30} {'Δ median':>10} {'Holm p':>9} {'effect':>10} {'sev':>5} {'artif':>6}  status")
    for m in d["metrics"]:
        s = stats[m["name"]]
        pct = s.get("median_delta_pct")
        print(f"{ICON[m['status']]:2}{m['name'][:30]:30} "
              f"{(_fmt(pct, '+.1f') + '%') if pct is not None else '-':>10} "
              f"{_fmt(m['holm_p'], '.2g'):>9} {s.get('effect_size', '-'):>10} "
              f"{_fmt(m['jev_severity'], '.2f'):>5} {_fmt(m['jev_artifact'], '.2f'):>6}  {m['status']}")
    for n in d["notes"]:
        print(f"  note: {n}")
    print(f"\nVERDICT: {res['verdict'].upper()}   baseline_used={res['baseline_used']} "
          f"baseline_updated={res['baseline_updated']}")
    if res["narrative"]:
        nv = res["narrative"]
        if nv["used_llm"]:
            print(f"narrative: {nv['model']} in={nv['input_tokens']} out={nv['output_tokens']} "
                  f"~${nv['cost_usd']:.4f} analysis_calls={nv['analysis_calls']}")
        print("\n" + nv["markdown"])
    print(f"total ~${res['cost_usd_total']:.6f} in {res['elapsed_ms']} ms\n")


if __name__ == "__main__":
    main()
