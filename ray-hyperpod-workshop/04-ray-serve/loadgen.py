"""Step load generator for the sentiment API (standard library only).

Runs a sequence of stages, each with a fixed number of concurrent clients, and prints
throughput, latency percentiles, errors and the number of Serve replicas per stage,
so you can see autoscaling react to load.

    python loadgen.py                                   # default stages
    python loadgen.py --stages 2:30,16:60,48:90,2:60     # concurrency:seconds,...
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import threading
import time
import urllib.error
import urllib.request
from collections import Counter

# Talk to the local tunnel directly, even if HTTP(S)_PROXY is set in the shell.
_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))

PHRASES = [
    "the new patch is great and matchmaking feels fast",
    "servers were slow tonight and I kept getting lag",
    "love the new stadium, the crowd sounds awesome",
    "the game crashed twice during overtime, really bad",
    "smooth gameplay and quick load times",
    "I hate losing to lag, fix the servers",
]


def pct(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    s = sorted(values)
    k = (len(s) - 1) * q
    lo, hi = int(k), min(int(k) + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def replica_count(dashboard: str) -> str:
    """Running replicas of the 'sentiment' app, read from the Ray dashboard's Serve API."""
    try:
        with _OPENER.open(f"{dashboard}/api/serve/applications/", timeout=3) as r:
            data = json.load(r)
        deps = data["applications"]["sentiment"]["deployments"]
        return str(sum(1 for d in deps.values() for rep in d.get("replicas", []) if rep.get("state") == "RUNNING"))
    except Exception:  # noqa: BLE001 - replica count is informational only
        return "-"


def run_stage(url: str, concurrency: int, seconds: float, timeout: float):
    lat: list[float] = []
    errors: Counter = Counter()
    first_error: list[str] = []
    replicas: Counter = Counter()
    lock = threading.Lock()
    stop_at = time.time() + seconds

    def client() -> None:
        while time.time() < stop_at:
            body = json.dumps({"text": random.choice(PHRASES)}).encode()
            req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
            t0 = time.perf_counter()
            try:
                with _OPENER.open(req, timeout=timeout) as r:
                    payload = json.load(r)
                dt = (time.perf_counter() - t0) * 1000
                with lock:
                    lat.append(dt)
                    replicas[payload.get("replica", "?")] += 1
            except urllib.error.HTTPError as e:
                with lock:
                    errors[f"HTTP {e.code}"] += 1
            except Exception as e:  # noqa: BLE001
                with lock:
                    errors[type(e).__name__] += 1
                    if not first_error:
                        first_error.append(f"{type(e).__name__}: {e}")
                time.sleep(0.2)

    threads = [threading.Thread(target=client, daemon=True) for _ in range(concurrency)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return lat, errors, replicas, first_error


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8000/classify")
    ap.add_argument("--dashboard", default="http://127.0.0.1:8265")
    ap.add_argument("--stages", default="2:30,16:60,48:90,4:60", help="concurrency:seconds,...")
    ap.add_argument("--timeout", type=float, default=30.0)
    a = ap.parse_args()

    stages = [(int(c), float(s)) for c, s in (x.split(":") for x in a.stages.split(","))]
    print(f"Target {a.url}   stages {a.stages}\n")
    print(f"{'stage':>5} {'clients':>7} {'req/s':>7} {'p50 ms':>8} {'p95 ms':>8} {'p99 ms':>8} "
          f"{'errors':>7} {'replicas seen':>13} {'replicas now':>12}")
    for i, (conc, secs) in enumerate(stages, 1):
        lat, errors, replicas, first_error = run_stage(a.url, conc, secs, a.timeout)
        rps = len(lat) / secs
        print(f"{i:>5} {conc:>7} {rps:>7.1f} {pct(lat, .5):>8.1f} {pct(lat, .95):>8.1f} {pct(lat, .99):>8.1f} "
              f"{sum(errors.values()):>7} {len(replicas):>13} {replica_count(a.dashboard):>12}", flush=True)
        if errors:
            print(f"      errors: {dict(errors)}  first: {first_error[0] if first_error else ''}")
    print("\n'replicas seen' = distinct replicas that answered during the stage;"
          " 'replicas now' = RUNNING replicas at the end of the stage.")
    if lat:
        print(f"last stage mean latency {statistics.fmean(lat):.1f} ms")


if __name__ == "__main__":
    main()
