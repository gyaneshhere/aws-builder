"""The decision layer: one Jev call, then deterministic policy in code.

Jev answers four typed questions in a single forward pass:
    runbook_tier    choice  L1 | L2 | L3
    blast_radius    score   0..3
    likely_transient noul   P(self-resolves / flapping)
    needs_human_now  noul   P(a human must act within 15 minutes)
    matched_rule     choice (only when the alert name is not in the runbook)

Jev supplies judgment; this module owns consequences. Hard floors from the
runbook, confidence-based escalation and the paging decision are plain code so
they are auditable and cannot be talked out of by the alert text. If Jev is
unavailable the policy fails safe to L3.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from .alerts import NormalizedAlert
from .config import Settings
from .jev_client import JevClient, JevError, JevResult
from .runbook import UNKNOWN_RULE, Runbook, RunbookRule, Tier

TIER_CRITERIA = {
    "L1": (
        "Known alert with a documented, low-risk first response that on-call can follow "
        "from the runbook; no customer impact and no data at risk."
    ),
    "L2": (
        "Needs a DBA engineer to investigate: degraded performance, capacity trend, "
        "replica/allocation or heap problems; customers not yet impacted, no data loss."
    ),
    "L3": (
        "Service-impacting or data-at-risk: red cluster, writes blocked or rejected in "
        "production, master unreachable, storage exhausted, encryption key unavailable; "
        "needs a senior engineer or AWS Support now."
    ),
}

BLAST_RADIUS_LEVELS = [
    "none: non-production, test traffic, or a single noisy datapoint",
    "limited: one index or one node degraded, redundancy still intact",
    "cluster-wide degradation: latency or error rate raised across the cluster",
    "outage or data at risk: player-facing reads/writes failing or primaries unavailable",
]


def build_questions(runbook: Runbook, need_rule_match: bool) -> dict[str, Any]:
    q: dict[str, Any] = {
        "runbook_tier": {
            "type": "choice",
            "instructions": (
                "Which runbook response tier does this Amazon OpenSearch Service alert need? "
                "Use the environment, the rule meaning and any prior incidents in the state."
            ),
            "criteria": TIER_CRITERIA,
        },
        "blast_radius": {
            "type": "score",
            "instructions": "How large is the potential impact of this alert right now?",
            "criteria": BLAST_RADIUS_LEVELS,
        },
        "likely_transient": {
            "type": "noul",
            "instructions": "This alert is likely transient or flapping and will self-resolve without action.",
            "criteria": {
                "true": "short spike, known deployment/blue-green, or repeated flapping with no lasting effect",
                "false": "sustained condition or one that worsens without intervention",
            },
        },
        "needs_human_now": {
            "type": "noul",
            "instructions": "A human must act within 15 minutes to prevent or limit customer impact.",
        },
    }
    if need_rule_match:
        q["matched_rule"] = {
            "type": "choice",
            "instructions": "Which runbook rule does this alert correspond to?",
            "criteria": runbook.jev_criteria(),
        }
    return q


@dataclass
class TriageDecision:
    tier: Tier
    action: str                      # page | notify | watch | record_only
    rule_id: str
    rule_match: str                  # exact | jev | none
    jev_ok: bool
    jev_tier: str | None = None
    jev_tier_confidence: float | None = None
    tier_probabilities: dict[str, float] = field(default_factory=dict)
    blast_radius: float | None = None
    likely_transient: float | None = None
    needs_human_now: float | None = None
    reasons: list[str] = field(default_factory=list)
    jev_cost_usd: float = 0.0
    jev_latency_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["tier"] = self.tier.name
        return d


def apply_policy(
    alert: NormalizedAlert,
    runbook: Runbook,
    settings: Settings,
    jev: JevResult | None,
    exact_rule: RunbookRule | None,
) -> TriageDecision:
    """Pure function: Jev answers (or None on failure) + alert -> decision."""
    reasons: list[str] = []

    # Resolved alerts: record only, no LLM spend.
    if alert.status == "resolved":
        return TriageDecision(
            tier=Tier.L1, action="record_only", rule_id=exact_rule.id if exact_rule else UNKNOWN_RULE,
            rule_match="exact" if exact_rule else "none", jev_ok=False,
            reasons=["alert resolved: recorded for history, Jev and LLM skipped"],
        )

    # --- Fail safe: Jev unavailable ---------------------------------------
    if jev is None:
        rule = exact_rule
        return TriageDecision(
            tier=Tier.L3,
            action="page" if alert.environment != "nonprod" else "notify",
            rule_id=rule.id if rule else UNKNOWN_RULE,
            rule_match="exact" if rule else "none",
            jev_ok=False,
            reasons=["Jev unavailable: failing safe to L3"],
        )

    # --- Rule resolution ---------------------------------------------------
    rule, match = exact_rule, "exact" if exact_rule else "none"
    if rule is None and "matched_rule" in jev.answers:
        picked = jev.choice("matched_rule")
        if picked != UNKNOWN_RULE and jev.choice_confidence("matched_rule") >= settings.tier_confidence_min:
            rule, match = runbook.get(picked), "jev"
            reasons.append(f"rule mapped by Jev to {picked} (conf {jev.choice_confidence('matched_rule'):.2f})")

    # --- Tier: Jev judgment, then code-enforced floors ------------------------
    jev_tier = Tier.parse(jev.choice("runbook_tier"))
    conf = jev.choice_confidence("runbook_tier")
    blast = jev.score("blast_radius")
    transient = jev.noul("likely_transient")
    human_now = jev.noul("needs_human_now")
    tier = jev_tier
    reasons.append(f"Jev tier {jev_tier.name} (conf {conf:.2f})")

    if conf < settings.tier_confidence_min:
        tier = tier.bump()
        reasons.append(f"low confidence < {settings.tier_confidence_min}: escalated to {tier.name}")

    if blast >= 2.5 and tier < Tier.L3:
        tier = Tier.L3
        reasons.append(f"blast radius {blast:.2f} >= 2.5: floor L3")
    elif blast >= 1.5 and tier < Tier.L2:
        tier = Tier.L2
        reasons.append(f"blast radius {blast:.2f} >= 1.5: floor L2")

    if rule is not None:
        floor = rule.min_tier(alert.is_prod or alert.environment == "unknown")
        if tier < floor:
            tier = floor
            reasons.append(f"runbook floor for {rule.id}: {floor.name}")
    else:
        if tier < Tier.L2:
            tier = Tier.L2
            reasons.append("alert not in runbook: floor L2")

    # --- Action ------------------------------------------------------------
    if alert.environment == "nonprod":
        action = "notify" if tier >= Tier.L2 else "watch"
    elif tier == Tier.L3 or human_now >= settings.page_now_min:
        action = "page"
    elif tier == Tier.L2:
        action = "notify"
    elif transient >= settings.transient_watch_min and blast < 1.0:
        action = "watch"
    else:
        action = "notify"

    return TriageDecision(
        tier=tier,
        action=action,
        rule_id=rule.id if rule else UNKNOWN_RULE,
        rule_match=match,
        jev_ok=True,
        jev_tier=jev_tier.name,
        jev_tier_confidence=round(conf, 4),
        tier_probabilities=jev.probabilities("runbook_tier"),
        blast_radius=round(blast, 3),
        likely_transient=round(transient, 3),
        needs_human_now=round(human_now, 3),
        reasons=reasons,
        jev_cost_usd=jev.cost_usd,
        jev_latency_ms=round(jev.latency_ms, 1),
    )


def decide(
    alert: NormalizedAlert,
    runbook: Runbook,
    settings: Settings,
    jev_client: JevClient | None,
    prior_incidents: list[str] | None = None,
) -> TriageDecision:
    exact = runbook.exact_match(alert.rule, alert.metric)
    if alert.status == "resolved":  # no Jev call, no LLM call
        return apply_policy(alert, runbook, settings, None, exact)

    state: dict[str, Any] = {"alert": alert.to_state()}
    if exact:
        state["runbook_rule"] = {"id": exact.id, "meaning": exact.summary, "escalation": exact.escalation}
    if prior_incidents:
        state["prior_incidents_same_domain"] = prior_incidents[:3]

    result: JevResult | None = None
    if jev_client is not None:
        try:
            result = jev_client.decide(state=state, questions=build_questions(runbook, exact is None))
        except JevError as exc:
            decision = apply_policy(alert, runbook, settings, None, exact)
            decision.reasons.append(str(exc)[:200])
            return decision
    return apply_policy(alert, runbook, settings, result, exact)
