"""Generate a SYNTHETIC LT-vs-Prod comparison for demos and tests (deterministic).

Scenario (shaped like a 60K PSU sine-wave run; numbers are made up):
  * api_latency_p99_ms       +~19% in LT                 -> expected regression
  * db_timeouts_per_min      ~3x in LT                   -> expected regression
  * aurora_db_load           +~33%, LT runs smaller Aurora instances (noted) -> expected explained
  * aurora_writer_cpu_pct    lower in LT                  -> expected improvement
  * aurora_reader_connections_max: 1500-3000 spike in the first 5 LT minutes
    (restart reconnect burst) that the warm-up window removes -> expected noise
  * everything else                                       -> expected noise
"""

import json
import math
import random
from pathlib import Path

R = random.Random(27)
N = 60


def series(mean, sd, n=N, floor=0.0, wave=0.0):
    return [round(max(floor, R.gauss(mean, sd) * (1 + wave * math.sin(2 * math.pi * i / 30))), 3) for i in range(n)]


def poisson(lam, n=N):
    out = []
    for _ in range(n):
        L, k, p = math.exp(-lam), 0, 1.0
        while p > L:
            k += 1
            p *= R.random()
        out.append(k - 1)
    return out


def main():
    readers_lt = series(205, 12)
    for i in range(5):                      # warm-up reconnect burst on a single reader
        readers_lt[i] = round(R.uniform(1500, 3000))
    metrics = [
        dict(name="api_latency_p50_ms", unit="ms", direction="higher_is_worse", slo=120, min_pct=5,
             owner="game-server", lt=series(42.5, 3, wave=0.05), prod=series(42, 3, wave=0.05)),
        dict(name="api_latency_p99_ms", unit="ms", direction="higher_is_worse", slo=250, min_pct=5,
             owner="game-server", lt=series(214, 14, wave=0.05), prod=series(180, 12, wave=0.05)),
        dict(name="error_rate_pct", unit="%", direction="higher_is_worse", slo=0.5, min_pct=10,
             owner="game-server", lt=series(0.082, 0.02), prod=series(0.08, 0.02)),
        dict(name="db_timeouts_per_min", unit="count/min", direction="higher_is_worse", min_pct=20,
             owner="dba", lt=poisson(6.0), prod=poisson(2.0)),
        dict(name="db_stalls_per_min", unit="count/min", direction="higher_is_worse", min_pct=20,
             owner="dba", lt=poisson(1.1), prod=poisson(1.0)),
        dict(name="aurora_reader_connections_max", unit="connections", direction="higher_is_worse", min_pct=15,
             owner="dba", lt=readers_lt, prod=series(210, 10)),
        dict(name="aurora_db_load", unit="AAS", direction="higher_is_worse", min_pct=10,
             owner="dba", lt=series(24, 4), prod=series(18, 3)),
        dict(name="aurora_writer_cpu_pct", unit="%", direction="higher_is_worse", min_pct=5,
             owner="dba", lt=series(47, 5), prod=series(55, 5)),
        dict(name="aurora_slow_queries_per_min", unit="count/min", direction="higher_is_worse", min_pct=20,
             owner="dba", lt=poisson(5.3), prod=poisson(5.0)),
    ]
    comp = {
        "test_name": "psu-60k-sinewave",
        "run_id": "lt-2026-09-24-a",
        "build": "server 27.0.412 (synthetic sample)",
        "notes": ("SYNTHETIC DATA. LT uses 2XL I/O-Optimized Aurora with reader autoscaling (min 2, max 6); "
                  "Prod runs fixed 4XL. Game servers were restarted at LT start, causing a reconnect burst "
                  "in the first minutes."),
        "lt": {"label": "LT sine-wave 60K PSU", "interval_seconds": 60, "warmup_minutes": 5},
        "prod": {"label": "Prod peak hour", "interval_seconds": 60, "warmup_minutes": 0},
        "metrics": metrics,
        "expected": {"api_latency_p99_ms": "regression", "db_timeouts_per_min": "regression",
                     "aurora_db_load": "explained", "aurora_writer_cpu_pct": "improvement",
                     "aurora_reader_connections_max": "noise", "api_latency_p50_ms": "noise",
                     "error_rate_pct": "noise", "db_stalls_per_min": "noise",
                     "aurora_slow_queries_per_min": "noise"},
    }
    out = Path(__file__).with_name("psu_sinewave_vs_prod.json")
    out.write_text(json.dumps(comp, indent=1))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
