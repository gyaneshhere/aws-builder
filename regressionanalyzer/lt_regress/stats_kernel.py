"""Exact statistics for LT-vs-Prod comparison. Pure standard library.

This file is executed inside AgentCore Code Interpreter (the sandbox runs its
source plus a call to analyze()); the local engine imports the same file, so
results are identical in both places. Jev never sees raw samples or does
arithmetic: it only reads the numbers this kernel produces.

Per metric:
  * steady-state series after dropping each side's warm-up window
  * summaries (n, mean, stdev, p50, p95, p99, min, max)
  * median delta (abs, %), direction-aware "worse" flag
  * Mann-Whitney U (normal approximation, tie-corrected) two-sided p-value
  * Cliff's delta effect size + magnitude label
  * bootstrap 95% CI of the median difference (seeded, reproducible)
  * SLO check on the LT steady-state p95 (or p05 for lower-is-worse)
  * warm-up segment summary (so start-of-run bursts are visible, not averaged in)
  * shift vs the last accepted LT baseline, if one is supplied
"""

import math
import random
import statistics
from bisect import bisect_left, bisect_right

KERNEL_VERSION = "1.0.0"
MIN_SAMPLES = 8


def _pct(xs, q):
    if not xs:
        return None
    s = sorted(xs)
    if len(s) == 1:
        return s[0]
    pos = (len(s) - 1) * q
    lo, hi = math.floor(pos), math.ceil(pos)
    return s[lo] + (s[hi] - s[lo]) * (pos - lo)


def summarize(xs):
    if not xs:
        return {"n": 0}
    return {
        "n": len(xs),
        "mean": statistics.fmean(xs),
        "stdev": statistics.stdev(xs) if len(xs) > 1 else 0.0,
        "p05": _pct(xs, 0.05),
        "p50": _pct(xs, 0.50),
        "p95": _pct(xs, 0.95),
        "p99": _pct(xs, 0.99),
        "min": min(xs),
        "max": max(xs),
    }


def mann_whitney(a, b):
    """Two-sided Mann-Whitney U with tie correction and continuity correction."""
    n1, n2 = len(a), len(b)
    combined = sorted([(v, 0) for v in a] + [(v, 1) for v in b])
    ranks = [0.0] * len(combined)
    tie_term = 0.0
    i = 0
    while i < len(combined):
        j = i
        while j + 1 < len(combined) and combined[j + 1][0] == combined[i][0]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[k] = avg
        t = j - i + 1
        tie_term += t ** 3 - t
        i = j + 1
    r1 = sum(r for r, (_, g) in zip(ranks, combined) if g == 0)
    u1 = r1 - n1 * (n1 + 1) / 2.0
    mu = n1 * n2 / 2.0
    n = n1 + n2
    var = n1 * n2 / 12.0 * ((n + 1) - tie_term / (n * (n - 1)))
    if var <= 0:
        return {"u": u1, "z": 0.0, "p": 1.0}
    z = (u1 - mu - math.copysign(0.5, u1 - mu)) / math.sqrt(var) if u1 != mu else 0.0
    p = math.erfc(abs(z) / math.sqrt(2.0))
    return {"u": u1, "z": z, "p": min(1.0, p)}


def cliffs_delta(a, b):
    """P(a > b) - P(a < b); |d| < .147 negligible, < .33 small, < .474 medium, else large."""
    sb = sorted(b)
    gt = lt = 0
    for x in a:
        lt += len(sb) - bisect_right(sb, x)   # b values greater than x
        gt += bisect_left(sb, x)              # b values less than x
    d = (gt - lt) / (len(a) * len(b))
    ad = abs(d)
    mag = "negligible" if ad < 0.147 else "small" if ad < 0.33 else "medium" if ad < 0.474 else "large"
    return d, mag


def bootstrap_median_diff(a, b, iters=2000, seed=7):
    rng = random.Random(seed)
    diffs = []
    for _ in range(iters):
        ra = [a[rng.randrange(len(a))] for _ in a]
        rb = [b[rng.randrange(len(b))] for _ in b]
        diffs.append(statistics.median(ra) - statistics.median(rb))
    return _pct(diffs, 0.025), _pct(diffs, 0.975)


def _warmup_count(side, n):
    interval = float(side.get("interval_seconds") or 60)
    minutes = float(side.get("warmup_minutes") or 0)
    return min(n, int(round(minutes * 60 / interval)))


def _clean(xs):
    return [float(x) for x in xs if x is not None and not (isinstance(x, float) and math.isnan(x))]


def analyze_metric(m, lt_side, prod_side, baseline=None):
    lt_all, prod_all = _clean(m["lt"]), _clean(m["prod"])
    wl, wp = _warmup_count(lt_side, len(lt_all)), _warmup_count(prod_side, len(prod_all))
    lt, prod = lt_all[wl:], prod_all[wp:]
    direction = m.get("direction", "higher_is_worse")
    out = {
        "name": m["name"], "unit": m.get("unit", ""), "direction": direction,
        "slo": m.get("slo"), "min_pct": m.get("min_pct", 5.0),
        "lt": summarize(lt), "prod": summarize(prod),
        "lt_warmup": summarize(lt_all[:wl]) if wl else None,
        "warmup_samples_dropped": {"lt": wl, "prod": wp},
    }
    if len(lt) < MIN_SAMPLES or len(prod) < MIN_SAMPLES:
        out["status"] = "insufficient_data"
        return out

    lt_med, prod_med = out["lt"]["p50"], out["prod"]["p50"]
    delta = lt_med - prod_med
    pct = (delta / abs(prod_med) * 100.0) if prod_med else None
    if direction == "higher_is_worse":
        worse = delta > 0
    elif direction == "lower_is_worse":
        worse = delta < 0
    else:
        worse = delta != 0
    mw = mann_whitney(lt, prod)
    d, mag = cliffs_delta(lt, prod)
    ci = bootstrap_median_diff(lt, prod)

    slo = m.get("slo")
    slo_breach = None
    if slo is not None:
        slo_breach = (out["lt"]["p95"] > slo) if direction != "lower_is_worse" else (out["lt"]["p05"] < slo)

    out.update({
        "status": "ok",
        "median_delta": delta,
        "median_delta_pct": pct,
        "worse": worse,
        "mann_whitney_p": mw["p"],
        "mann_whitney_z": mw["z"],
        "cliffs_delta": d,
        "effect_size": mag,
        "median_diff_ci95": list(ci),
        "ci_excludes_zero": (ci[0] > 0) or (ci[1] < 0),
        "slo_breach": slo_breach,
    })
    if out["lt_warmup"] and out["lt_warmup"]["n"]:
        out["warmup_peak_vs_steady_pct"] = (
            (out["lt_warmup"]["max"] - out["lt"]["max"]) / abs(out["lt"]["max"]) * 100.0
            if out["lt"]["max"] else None
        )
    if baseline and baseline.get("p50") is not None:
        b = baseline["p50"]
        out["vs_baseline_pct"] = ((lt_med - b) / abs(b) * 100.0) if b else None
        out["baseline_run_id"] = baseline.get("run_id")
    return out


def analyze(payload):
    lt_side, prod_side = payload.get("lt", {}), payload.get("prod", {})
    baselines = payload.get("baselines") or {}
    return {
        "kernel_version": KERNEL_VERSION,
        "metrics": [analyze_metric(m, lt_side, prod_side, baselines.get(m["name"])) for m in payload["metrics"]],
    }
