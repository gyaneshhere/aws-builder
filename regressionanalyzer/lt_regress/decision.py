"""Decision layer: statistics decide what is *real*, Jev decides what *matters*,
code decides what gets *flagged*.

Per metric, Jev answers (batched, several metrics per call):
    sev__<m>       score 0..3   noise | minor drift | notable deviation | real regression
    artifact__<m>  noul         P(deviation is explained by test setup, not the build)

Jev reads the kernel's numbers (deltas, p-values, effect sizes, SLO, warm-up,
baseline shift) plus run notes; it never computes anything itself.

Policy (code, per metric):
  * SLO breach in the worse direction                          -> regression (always)
  * worse + Holm-significant + |median shift| >= min_pct:
        artifact >= ARTIFACT_MIN                               -> explained
        Jev severity >= SEVERITY_FLAG_MIN                      -> regression
        Jev severity confidence < SEVERITY_CONF_MIN            -> regression (fail toward review)
        otherwise                                              -> watch
  * better + significant + practical                           -> improvement
  * everything else                                            -> noise (or insufficient_data)
  * Jev unavailable: significant+practical+worse               -> regression
Run verdict: fail if any regression, watch if any watch, else pass.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any

from .config import Settings
from .jev_client import JevClient, JevError, JevResult
from .models import Comparison

SEVERITY_LEVELS = [
    "noise: within normal run-to-run variation, nothing to act on",
    "minor drift: measurable but inside SLO and unlikely to be felt by players or capacity",
    "notable deviation: meaningful change an owner should look at before sign-off",
    "real regression: likely player-visible (latency, errors, timeouts) or threatens DB/infra capacity",
]


def qkey(metric: str) -> str:
    return re.sub(r"[^a-z0-9_]", "_", metric.lower())[:60]


def holm(pvals: dict[str, float]) -> dict[str, float]:
    """Holm-Bonferroni adjusted p-values (monotone)."""
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m, running, out = len(items), 0.0, {}
    for i, (k, p) in enumerate(items):
        running = max(running, min(1.0, (m - i) * p))
        out[k] = running
    return out


def _r(x: Any, nd: int = 4) -> Any:
    return round(x, nd) if isinstance(x, float) else x


def jev_state_for(metric_stats: dict[str, Any], adj_p: float | None, meta: dict[str, Any]) -> dict[str, Any]:
    s = metric_stats
    view = {
        "unit": s.get("unit"), "direction": s.get("direction"), "slo": s.get("slo"),
        "owner": meta.get("owner"), "practical_threshold_pct": s.get("min_pct"),
        "prod_median": _r(s["prod"].get("p50")), "lt_median": _r(s["lt"].get("p50")),
        "prod_p95": _r(s["prod"].get("p95")), "lt_p95": _r(s["lt"].get("p95")),
        "median_change_pct": _r(s.get("median_delta_pct"), 2),
        "change_is_in_worse_direction": s.get("worse"),
        "holm_adjusted_p": _r(adj_p), "effect_size": s.get("effect_size"),
        "cliffs_delta": _r(s.get("cliffs_delta"), 3),
        "ci95_of_median_diff": [_r(x) for x in s.get("median_diff_ci95", [])],
        "lt_slo_breach": s.get("slo_breach"),
        "change_vs_last_accepted_lt_run_pct": _r(s.get("vs_baseline_pct"), 2),
    }
    if s.get("lt_warmup"):
        view["lt_warmup_max"] = _r(s["lt_warmup"].get("max"))
        view["lt_steady_max"] = _r(s["lt"].get("max"))
    return {k: v for k, v in view.items() if v is not None}


def build_questions(metric_names: list[str]) -> dict[str, Any]:
    q: dict[str, Any] = {}
    for name in metric_names:
        k = qkey(name)
        q[f"sev__{k}"] = {
            "type": "score",
            "instructions": f"How serious is the LT-vs-Production deviation for metric '{name}' "
                            f"(state.metrics['{name}']) for a release sign-off?",
            "criteria": SEVERITY_LEVELS,
        }
        q[f"artifact__{k}"] = {
            "type": "noul",
            "instructions": f"The deviation in '{name}' is explained by the load-test setup "
                            "(warm-up, restart reconnect burst, synthetic traffic shape, different "
                            "instance sizing or config noted in the run) rather than by the build under test.",
        }
    return q


@dataclass
class MetricVerdict:
    name: str
    status: str                  # regression | watch | explained | improvement | noise | insufficient_data
    reasons: list[str] = field(default_factory=list)
    holm_p: float | None = None
    practical: bool | None = None
    jev_severity: float | None = None
    jev_severity_conf: float | None = None
    jev_artifact: float | None = None


@dataclass
class RunDecision:
    verdict: str                 # pass | watch | fail
    metrics: list[MetricVerdict]
    jev_ok: bool
    jev_calls: int = 0
    jev_cost_usd: float = 0.0
    jev_latency_ms: float = 0.0
    notes: list[str] = field(default_factory=list)

    def by_status(self, *status: str) -> list[MetricVerdict]:
        return [m for m in self.metrics if m.status in status]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def classify(s: dict[str, Any], adj_p: float | None, jev: JevResult | None, cfg: Settings) -> MetricVerdict:
    v = MetricVerdict(name=s["name"], status="noise", holm_p=adj_p)
    if s.get("status") == "insufficient_data":
        v.status, v.reasons = "insufficient_data", ["fewer than 8 steady-state samples on one side"]
        return v

    pct = s.get("median_delta_pct")
    v.practical = abs(pct) >= float(s.get("min_pct", 5.0)) if pct is not None else abs(s["median_delta"]) > 0
    significant = adj_p is not None and adj_p < cfg.alpha and s.get("ci_excludes_zero", False)
    worse = bool(s.get("worse"))

    if jev is not None:
        k = qkey(s["name"])
        v.jev_severity = round(jev.score(f"sev__{k}"), 3)
        v.jev_severity_conf = jev.score_confidence(f"sev__{k}")
        v.jev_artifact = round(jev.noul(f"artifact__{k}"), 3)

    if s.get("slo_breach") and worse:
        v.status = "regression"
        v.reasons.append(f"LT breaches SLO {s['slo']} {s.get('unit', '')}".strip())
        return v

    if significant and v.practical and worse:
        v.reasons.append(f"significant (Holm p={adj_p:.4g}) and {pct:+.1f}% vs prod" if pct is not None
                         else f"significant (Holm p={adj_p:.4g})")
        if jev is None:
            v.status = "regression"
            v.reasons.append("Jev unavailable: statistics-only rule")
        elif v.jev_artifact >= cfg.artifact_min:
            v.status = "explained"
            v.reasons.append(f"Jev: likely test-setup artifact ({v.jev_artifact:.2f})")
        elif v.jev_severity >= cfg.severity_flag_min:
            v.status = "regression"
            v.reasons.append(f"Jev severity {v.jev_severity:.2f}")
        elif v.jev_severity_conf is not None and v.jev_severity_conf < cfg.severity_conf_min:
            v.status = "regression"
            v.reasons.append(f"Jev severity {v.jev_severity:.2f} at low confidence {v.jev_severity_conf:.2f}: flagged for review")
        else:
            v.status = "watch"
            v.reasons.append(f"Jev severity {v.jev_severity:.2f} below flag threshold")
        return v

    if significant and v.practical and not worse:
        v.status = "improvement"
        v.reasons.append(f"significant improvement ({pct:+.1f}%)" if pct is not None else "significant improvement")
        return v

    why = []
    if not significant:
        why.append("not significant after Holm correction" if adj_p is not None else "no test")
    if v.practical is False:
        why.append(f"below practical threshold {s.get('min_pct')}%")
    v.reasons.append("; ".join(why) or "no meaningful change")
    return v


def decide(comp: Comparison, stats: dict[str, Any], jev_client: JevClient | None, cfg: Settings) -> RunDecision:
    metrics = [m for m in stats["metrics"]]
    testable = {m["name"]: m["mann_whitney_p"] for m in metrics if m.get("status") == "ok"}
    adj = holm(testable) if testable else {}

    jev_results: dict[str, JevResult] = {}
    notes, calls, cost, latency, jev_ok = [], 0, 0.0, 0.0, jev_client is not None
    if jev_client is not None:
        names = list(testable)
        for i in range(0, len(names), max(1, cfg.jev_metrics_per_call)):
            chunk = names[i:i + cfg.jev_metrics_per_call]
            state = {
                "run": comp.meta(),
                "metrics": {n: jev_state_for(next(m for m in metrics if m["name"] == n), adj.get(n),
                                             comp.metric_meta(n)) for n in chunk},
            }
            try:
                res = jev_client.decide(state=state, questions=build_questions(chunk))
            except JevError as exc:
                jev_ok = False
                notes.append(f"Jev failed for {chunk}: {str(exc)[:160]}")
                continue
            calls += 1
            cost += res.cost_usd
            latency = max(latency, res.latency_ms)
            for n in chunk:
                jev_results[n] = res

    verdicts = [classify(m, adj.get(m["name"]), jev_results.get(m["name"]), cfg) for m in metrics]
    statuses = {v.status for v in verdicts}
    verdict = "fail" if "regression" in statuses else "watch" if "watch" in statuses else "pass"
    return RunDecision(verdict=verdict, metrics=verdicts, jev_ok=jev_ok, jev_calls=calls,
                       jev_cost_usd=round(cost, 6), jev_latency_ms=round(latency, 1), notes=notes)
