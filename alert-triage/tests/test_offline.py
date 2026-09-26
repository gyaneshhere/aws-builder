"""Offline tests: normalization, policy, fail-safe, Lambda tool routing. No AWS or Jev calls."""

from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

os.environ.setdefault("AWS_DEFAULT_REGION", "us-east-1")
os.environ.setdefault("AWS_REGION", "us-east-1")

from triage.alerts import normalize  # noqa: E402
from triage.config import Settings  # noqa: E402
from triage.decision import build_questions, decide  # noqa: E402
from triage.jev_client import JevError, JevResult  # noqa: E402
from triage.runbook import Runbook, Tier  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RB = Runbook.load()
S = Settings()
LABELED = json.loads((ROOT / "samples/labeled_alerts.json").read_text())


def sample(name: str) -> dict:
    return next(r["payload"] for r in LABELED if r["name"] == name)


class FakeJev:
    def __init__(self, tier="L1", conf=0.9, blast=0.3, transient=0.2, human=0.1, rule=None, rule_conf=0.9,
                 fail=False):
        self.a = dict(tier=tier, conf=conf, blast=blast, transient=transient, human=human, rule=rule,
                      rule_conf=rule_conf)
        self.fail = fail
        self.last_questions = None

    def decide(self, state, questions):
        self.last_questions = questions
        if self.fail:
            raise JevError("boom")
        a = self.a
        answers = {
            "runbook_tier": {"type": "choice", "choice": a["tier"], "confidence": a["conf"],
                             "probabilities": {a["tier"]: a["conf"]}},
            "blast_radius": {"type": "score", "score": a["blast"]},
            "likely_transient": {"type": "noul", "noul": a["transient"]},
            "needs_human_now": {"type": "noul", "noul": a["human"]},
        }
        if "matched_rule" in questions:
            answers["matched_rule"] = {"type": "choice", "choice": a["rule"] or "unknown",
                                       "confidence": a["rule_conf"]}
        return JevResult(model="fake", answers=answers, cost_usd=0.00003, latency_ms=120)


# ---- normalization ---------------------------------------------------------
def test_grafana_normalization():
    al = normalize(sample("cluster_red_prod"))
    assert (al.source, al.rule, al.domain, al.environment, al.status) == (
        "grafana", "ClusterStatusRed", "player-search-prod", "prod", "firing")


def test_pagerduty_normalization():
    al = normalize(sample("kms_key_prod_pagerduty"))
    assert (al.source, al.rule, al.domain, al.environment) == (
        "pagerduty", "KMSKeyInaccessible", "player-search-prod", "prod")


def test_generic_and_nonprod():
    assert normalize(sample("unknown_rule_prod")).source == "generic"
    assert normalize(sample("cpu_spike_nonprod")).environment == "nonprod"


# ---- runbook / questions --------------------------------------------------
def test_runbook_loads_and_caps_options():
    assert len(RB.jev_criteria()) <= 20
    assert RB.exact_match("clusterstatusred").id == "ClusterStatusRed"
    assert RB.exact_match("nope", metric="FreeStorageSpace").id == "FreeStorageSpaceLow"


def test_rule_question_only_for_unknown_alerts():
    j = FakeJev()
    decide(normalize(sample("cluster_red_prod")), RB, S, j)
    assert "matched_rule" not in j.last_questions
    decide(normalize(sample("unknown_rule_prod")), RB, S, j)
    assert "matched_rule" in j.last_questions
    assert set(build_questions(RB, True)) == {
        "runbook_tier", "blast_radius", "likely_transient", "needs_human_now", "matched_rule"}


# ---- policy -------------------------------------------------------------------
def test_runbook_floor_overrides_jev():
    d = decide(normalize(sample("cluster_red_prod")), RB, S, FakeJev(tier="L1"))
    assert d.tier == Tier.L3 and d.action == "page"
    assert any("runbook floor" in r for r in d.reasons)


def test_low_confidence_escalates():
    d = decide(normalize(sample("cpu_spike_nonprod")), RB, S, FakeJev(tier="L1", conf=0.4))
    assert d.tier == Tier.L2 and d.action == "notify"


def test_blast_radius_floor():
    d = decide(normalize(sample("jvm_pressure_prod")), RB, S, FakeJev(tier="L2", blast=2.7))
    assert d.tier == Tier.L3 and d.action == "page"


def test_transient_prod_l1_watches():
    d = decide(normalize(sample("yellow_during_bluegreen_prod")), RB, S,
               FakeJev(tier="L1", transient=0.85, blast=0.4, human=0.1))
    assert d.tier == Tier.L1 and d.action == "watch"


def test_needs_human_now_pages_even_at_l1():
    d = decide(normalize(sample("yellow_during_bluegreen_prod")), RB, S, FakeJev(tier="L1", human=0.8))
    assert d.action == "page"


def test_nonprod_never_pages():
    d = decide(normalize(sample("cpu_spike_nonprod")), RB, S, FakeJev(tier="L3", human=0.99, blast=3))
    assert d.tier == Tier.L3 and d.action == "notify"


def test_unknown_rule_floor_and_jev_mapping():
    d = decide(normalize(sample("unknown_rule_prod")), RB, S, FakeJev(tier="L1"))
    assert d.rule_match == "none" and d.tier == Tier.L2
    d2 = decide(normalize(sample("unknown_rule_prod")), RB, S, FakeJev(tier="L1", rule="CPUUtilizationHigh"))
    assert d2.rule_match == "jev" and d2.rule_id == "CPUUtilizationHigh"
    d3 = decide(normalize(sample("unknown_rule_prod")), RB, S,
                FakeJev(tier="L1", rule="CPUUtilizationHigh", rule_conf=0.3))
    assert d3.rule_match == "none"


def test_jev_failure_fails_safe():
    d = decide(normalize(sample("cpu_spike_nonprod")), RB, S, FakeJev(fail=True))
    assert d.tier == Tier.L3 and not d.jev_ok and d.action == "notify"
    d = decide(normalize(sample("jvm_pressure_prod")), RB, S, None)
    assert d.tier == Tier.L3 and d.action == "page"


def test_resolved_skips_jev():
    j = FakeJev()
    d = decide(normalize(sample("red_resolved_prod")), RB, S, j)
    assert d.action == "record_only" and j.last_questions is None


def test_all_labeled_samples_route_with_ideal_answers():
    """If Jev returns the labeled tier with high confidence, policy must reproduce the labels."""
    for row in LABELED:
        al = normalize(row["payload"])
        want = row["expected_tier"]
        j = FakeJev(tier=want, conf=0.9,
                    blast={"L1": 0.3, "L2": 1.2, "L3": 2.8}[want],
                    transient=0.85 if row["expected_action"] == "watch" else 0.1,
                    human={"page": 0.9}.get(row["expected_action"], 0.1))
        d = decide(al, RB, S, j)
        assert (d.tier.name, d.action) == (want, row["expected_action"]), row["name"]


# ---- Lambda tools ---------------------------------------------------------------
def test_lambda_routing_and_allowlist(monkeypatch):
    monkeypatch.setenv("DOMAIN_ENDPOINTS", json.dumps({"metrictestopenai": "https://example"}))
    from lambda_tools import handler as h

    monkeypatch.setattr(h, "_get", lambda d, p: {"status": "green", "path": p})
    ctx = SimpleNamespace(client_context=SimpleNamespace(
        custom={"bedrockAgentCoreToolName": "opensearch-diag___get_cluster_health"}))
    assert h.lambda_handler({"domain": "metrictestopenai"}, ctx)["path"] == "/_cluster/health"

    ctx.client_context.custom["bedrockAgentCoreToolName"] = "opensearch-diag___delete_index"
    assert "unknown tool" in h.lambda_handler({"domain": "x"}, ctx)["error"]

    ctx.client_context.custom["bedrockAgentCoreToolName"] = "opensearch-diag___get_cloudwatch_metric"
    assert "not allowed" in h.lambda_handler({"domain": "metrictestopenai", "metric": "Foo"}, ctx)["error"]


def test_lambda_rejects_unlisted_domain(monkeypatch):
    monkeypatch.setenv("DOMAIN_ENDPOINTS", json.dumps({"metrictestopenai": "https://example"}))
    from lambda_tools import handler as h

    with pytest.raises(ValueError):
        h._get("some-other-domain", "/_cluster/health")


def test_tool_schemas_match_functions():
    from lambda_tools.handler import TOOL_SCHEMAS, TOOLS

    assert {t["name"] for t in TOOL_SCHEMAS} == set(TOOLS)
