"""Offline tests: kernel math, Holm, policy, fail-safes, loaders. No AWS or Jev calls."""

from __future__ import annotations

import json
import random
from pathlib import Path

import pytest

from lt_regress import stats_kernel as K
from lt_regress.config import Settings
from lt_regress.decision import decide, holm, qkey
from lt_regress.engine import LocalEngine, _parse_stream
from lt_regress.jev_client import JevError, JevResult
from lt_regress.models import Comparison, load_csv, load_json

ROOT = Path(__file__).resolve().parent.parent
SAMPLE = ROOT / "samples/psu_sinewave_vs_prod.json"
CFG = Settings()


class FakeJev:
    """Returns severity from a per-metric table; artifact high only where told."""

    def __init__(self, sev=None, artifact=None, conf=0.9, fail=False):
        self.sev, self.artifact, self.conf, self.fail = sev or {}, artifact or {}, conf, fail
        self.calls = []

    def decide(self, state, questions):
        self.calls.append(questions)
        if self.fail:
            raise JevError("boom")
        ans = {}
        for q in questions:
            kind, k = q.split("__", 1)
            if kind == "sev":
                ans[q] = {"type": "score", "score": self.sev.get(k, 2.5), "confidence": self.conf}
            else:
                ans[q] = {"type": "noul", "noul": self.artifact.get(k, 0.1)}
        return JevResult(model="fake", answers=ans, cost_usd=0.00002, latency_ms=90)


def run(jev, comp=None):
    comp = comp or load_json(SAMPLE)
    stats = LocalEngine().analyze(comp.kernel_payload())
    return comp, stats, decide(comp, stats, jev, CFG)


# ---- kernel math ---------------------------------------------------------------
def test_percentiles_and_summary():
    s = K.summarize([1, 2, 3, 4, 5])
    assert (s["p50"], s["min"], s["max"], s["n"]) == (3, 1, 5, 5)


def test_mann_whitney_matches_scipy():
    scipy = pytest.importorskip("scipy.stats")
    rng = random.Random(1)
    for shift in (0.0, 0.3, 1.0):
        a = [round(rng.gauss(10 + shift, 1), 1) for _ in range(40)]   # rounding creates ties
        b = [round(rng.gauss(10, 1), 1) for _ in range(55)]
        ours = K.mann_whitney(a, b)["p"]
        ref = scipy.mannwhitneyu(a, b, alternative="two-sided", method="asymptotic", use_continuity=True).pvalue
        assert ours == pytest.approx(ref, rel=1e-6, abs=1e-12)


def test_cliffs_delta_extremes():
    assert K.cliffs_delta([10, 11, 12], [1, 2, 3]) == (1.0, "large")
    assert K.cliffs_delta([1, 2, 3], [1, 2, 3])[0] == 0.0


def test_bootstrap_is_reproducible():
    a, b = list(range(20, 40)), list(range(0, 20))
    assert K.bootstrap_median_diff(a, b) == K.bootstrap_median_diff(a, b)


def test_warmup_excluded_and_reported():
    comp, stats, _ = run(FakeJev())
    rc = next(m for m in stats["metrics"] if m["name"] == "aurora_reader_connections_max")
    assert rc["warmup_samples_dropped"]["lt"] == 5
    assert rc["lt_warmup"]["max"] > 1000 > rc["lt"]["max"]


def test_insufficient_data():
    d = {"test_name": "t", "run_id": "r", "lt": {}, "prod": {},
         "metrics": [{"name": "x", "lt": [1, 2, 3], "prod": [1, 2, 3]}]}
    comp, stats, dec = run(FakeJev(), Comparison.from_dict(d))
    assert dec.metrics[0].status == "insufficient_data"


def test_holm_monotone_and_bounded():
    adj = holm({"a": 0.01, "b": 0.02, "c": 0.5})
    assert adj["a"] == pytest.approx(0.03) and adj["b"] == pytest.approx(0.04) and adj["c"] == 0.5


# ---- policy -------------------------------------------------------------------
def test_sample_matches_expected_with_reasonable_jev():
    exp = json.loads(SAMPLE.read_text())["expected"]
    jev = FakeJev(sev={qkey("api_latency_p99_ms"): 2.6, qkey("db_timeouts_per_min"): 2.4,
                       qkey("aurora_db_load"): 2.1},
                  artifact={qkey("aurora_db_load"): 0.8})
    _, _, dec = run(jev)
    got = {m.name: m.status for m in dec.metrics}
    assert got == exp
    assert dec.verdict == "fail"


def test_low_severity_becomes_watch_and_low_conf_flags():
    _, _, dec = run(FakeJev(sev={qkey("api_latency_p99_ms"): 1.2}, artifact={qkey("aurora_db_load"): 0.8}))
    assert next(m for m in dec.metrics if m.name == "api_latency_p99_ms").status == "watch"
    _, _, dec = run(FakeJev(sev={qkey("api_latency_p99_ms"): 1.2}, conf=0.3))
    assert next(m for m in dec.metrics if m.name == "api_latency_p99_ms").status == "regression"


def test_jev_cannot_hide_slo_breach():
    d = json.loads(SAMPLE.read_text())
    m = next(x for x in d["metrics"] if x["name"] == "api_latency_p99_ms")
    m["slo"] = 200
    _, _, dec = run(FakeJev(sev={qkey("api_latency_p99_ms"): 0.0}, artifact={qkey("api_latency_p99_ms"): 0.99}),
                    Comparison.from_dict(d))
    v = next(x for x in dec.metrics if x.name == "api_latency_p99_ms")
    assert v.status == "regression" and "SLO" in v.reasons[0]


def test_jev_cannot_flag_noise():
    _, _, dec = run(FakeJev(sev={k: 3.0 for k in map(qkey, ["api_latency_p50_ms", "error_rate_pct"])}))
    got = {m.name: m.status for m in dec.metrics}
    assert got["api_latency_p50_ms"] == "noise" and got["error_rate_pct"] == "noise"


def test_jev_failure_falls_back_to_statistics():
    _, _, dec = run(FakeJev(fail=True))
    got = {m.name: m.status for m in dec.metrics}
    assert not dec.jev_ok and dec.notes
    assert got["aurora_db_load"] == "regression"          # no artifact judgment without Jev
    assert got["aurora_writer_cpu_pct"] == "improvement"
    _, _, dec2 = run(None)
    assert dec2.verdict == "fail"


def test_metrics_are_batched():
    jev = FakeJev()
    run(jev)
    assert len(jev.calls) == 2 and all(len(q) <= 2 * CFG.jev_metrics_per_call for q in jev.calls)


# ---- engine / loaders ---------------------------------------------------------
def test_parse_code_interpreter_stream():
    resp = {"stream": [{"result": {"structuredContent": {"stdout": "<<<RESULT>>>{}<<<END>>>", "stderr": ""},
                                   "isError": False}}]}
    out, err, is_err = _parse_stream(resp)
    assert out.startswith("<<<RESULT>>>") and not is_err


def test_kernel_source_is_self_contained():
    # The sandbox executes the kernel's source text; it must only use the standard library.
    src = Path(K.__file__).read_text()
    imports = {l.split()[1].split(".")[0] for l in src.splitlines() if l.startswith(("import ", "from "))}
    assert imports <= {"math", "random", "statistics", "bisect"}


def test_csv_loader(tmp_path):
    (tmp_path / "lt.csv").write_text("timestamp,lat\n" + "\n".join(f"{i},{100 + i % 3}" for i in range(20)))
    (tmp_path / "prod.csv").write_text("timestamp,lat\n" + "\n".join(f"{i},{90 + i % 3}" for i in range(20)))
    (tmp_path / "meta.json").write_text(json.dumps({"test_name": "t", "run_id": "r", "lt": {}, "prod": {},
                                                    "metrics": [{"name": "lat", "unit": "ms"}]}))
    comp = load_csv(tmp_path / "lt.csv", tmp_path / "prod.csv", tmp_path / "meta.json")
    assert len(comp.metrics[0]["lt"]) == 20


def test_validation_errors():
    with pytest.raises(ValueError):
        Comparison.from_dict({"test_name": "t", "run_id": "r", "metrics": [{"name": "a", "lt": [], "prod": [],
                                                                             "direction": "sideways"}]})
