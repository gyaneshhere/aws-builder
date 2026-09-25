"""Investigation agent: Bedrock model + tool allowlist chosen by the Jev tier.

    L1 -> Nova 2 Lite        tools: cluster health, CloudWatch metric
    L2 -> Nova Pro           + nodes, thread pools, unassigned shards
    L3 -> Claude Sonnet 4.6  + pending tasks, allocation explain

All tools are read-only and live behind AgentCore Gateway (Lambda target).
The tool allowlist is enforced client-side here AND the Lambda only implements
GET calls, so a model cannot mutate a cluster regardless of what it asks for.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass

from pydantic import BaseModel, Field

from .alerts import NormalizedAlert
from .config import Settings
from .decision import TriageDecision
from .runbook import RunbookRule, Tier

log = logging.getLogger(__name__)

GATEWAY_TARGET = "opensearch-diag"

L1_TOOLS = ["get_cluster_health", "get_cloudwatch_metric"]
L2_TOOLS = L1_TOOLS + ["get_nodes", "get_thread_pools", "get_unassigned_shards"]
L3_TOOLS = L2_TOOLS + ["get_pending_tasks", "get_allocation_explain"]


@dataclass(frozen=True)
class TierSpec:
    tier: Tier
    label: str
    model_id: str
    tools: list[str]
    max_tokens: int
    # Approximate USD per 1M tokens (same caveat as the article: verify Bedrock pricing).
    in_per_mtok: float
    out_per_mtok: float

    def cost(self, in_tok: int, out_tok: int) -> float:
        return in_tok / 1e6 * self.in_per_mtok + out_tok / 1e6 * self.out_per_mtok


def tier_specs(s: Settings) -> dict[Tier, TierSpec]:
    return {
        Tier.L1: TierSpec(Tier.L1, "Nova 2 Lite", s.model_l1, L1_TOOLS, 1500, 0.06, 0.24),
        Tier.L2: TierSpec(Tier.L2, "Nova Pro", s.model_l2, L2_TOOLS, 2500, 0.80, 3.20),
        Tier.L3: TierSpec(Tier.L3, "Claude Sonnet 4.6", s.model_l3, L3_TOOLS, 4000, 3.00, 15.00),
    }


class TriageReport(BaseModel):
    summary: str = Field(description="One or two sentences an on-call engineer reads first.")
    likely_cause: str = Field(description="Most likely cause, stated with appropriate uncertainty.")
    evidence: list[str] = Field(description="Facts observed from tool output; cite the tool used.")
    recommended_steps: list[str] = Field(description="Ordered next steps; runbook steps first.")
    escalate_to: str = Field(description="Who to escalate to and when, or 'none'.")
    data_gaps: list[str] = Field(default_factory=list, description="What could not be checked and why.")

    def to_markdown(self, alert: NormalizedAlert, d: TriageDecision) -> str:
        ev = "\n".join(f"- {e}" for e in self.evidence) or "- (no tool evidence)"
        steps = "\n".join(f"{i}. {s}" for i, s in enumerate(self.recommended_steps, 1))
        gaps = "\n".join(f"- {g}" for g in self.data_gaps)
        return (
            f"### {alert.rule} on {alert.domain or 'unknown domain'} [{alert.environment}]\n"
            f"**Tier:** {d.tier.name} | **Action:** {d.action} | **Rule:** {d.rule_id} ({d.rule_match})\n\n"
            f"**Summary:** {self.summary}\n\n**Likely cause:** {self.likely_cause}\n\n"
            f"**Evidence**\n{ev}\n\n**Recommended steps**\n{steps}\n\n"
            f"**Escalate to:** {self.escalate_to}\n" + (f"\n**Data gaps**\n{gaps}\n" if gaps else "")
        )


SYSTEM_PROMPT = """You are the triage assistant for EA's Amazon OpenSearch Service fleet.
A typed decision model has already classified this alert; do not re-litigate the
tier. Your job: gather evidence with the read-only diagnostic tools you have,
apply the runbook's documented first response, and produce a concise report.

Rules:
- Only state facts you observed in tool output or the alert payload. If a tool
  fails or you lack a tool, list it under data_gaps instead of guessing.
- Never recommend deleting indices, restarting nodes, or changing cluster
  settings as a first step; those go through change management (TOBOR).
- Call each tool at most twice. Prefer the documented runbook steps.
- Keep the report short enough to paste into a PagerDuty note."""


def _prompt(alert: NormalizedAlert, d: TriageDecision, rule: RunbookRule | None, prior: list[str], tools_on: bool) -> str:
    parts = [
        "ALERT:\n" + json.dumps(alert.to_state(), indent=2, default=str),
        "DECISION:\n" + json.dumps(
            {k: d.to_dict()[k] for k in ("tier", "action", "blast_radius", "likely_transient", "reasons")},
            indent=2,
        ),
        "RUNBOOK:\n" + (rule.as_prompt() if rule else "No runbook entry matched this alert."),
    ]
    if prior:
        parts.append("PRIOR INCIDENTS ON THIS DOMAIN:\n" + "\n---\n".join(prior))
    if not tools_on:
        parts.append("TOOLS: none available in this run; base the report on the alert and runbook only.")
    elif alert.domain:
        parts.append(f"Use domain='{alert.domain}' for every tool call.")
    else:
        parts.append("The alert carries no domain name, so tools cannot be targeted; note this in data_gaps.")
    return "\n\n".join(parts)


@dataclass
class InvestigationResult:
    report: TriageReport | None
    report_markdown: str
    model_label: str
    model_id: str
    input_tokens: int
    output_tokens: int
    model_cost_usd: float
    tools_used: list[str]


def investigate(
    alert: NormalizedAlert,
    decision: TriageDecision,
    rule: RunbookRule | None,
    prior: list[str],
    settings: Settings,
    spec_override: TierSpec | None = None,
) -> InvestigationResult:
    from strands import Agent
    from strands.models import BedrockModel

    spec = spec_override or tier_specs(settings)[decision.tier]
    model = BedrockModel(
        model_id=spec.model_id, region_name=settings.region, temperature=0.2, max_tokens=spec.max_tokens
    )

    tools, mcp = [], None
    if settings.tools_enabled and alert.domain:
        from strands.tools.mcp import MCPClient

        from .credentials import get_gateway_token

        allowed = re.compile(rf"^(?:{re.escape(GATEWAY_TARGET)}___)?(?:{'|'.join(spec.tools)})$")
        mcp = MCPClient(
            url=settings.gateway_url,
            headers={"Authorization": f"Bearer {get_gateway_token(settings)}"},
            tool_filters={"allowed": [allowed]},
            continue_on_error=True,
        )
        tools = [mcp]

    agent = Agent(
        model=model,
        tools=tools,
        system_prompt=SYSTEM_PROMPT,
        structured_output_model=TriageReport,
        callback_handler=None,
        trace_attributes={"triage.tier": decision.tier.name, "triage.rule": decision.rule_id},
    )
    try:
        result = agent(_prompt(alert, decision, rule, prior, bool(tools)))
    finally:
        if mcp is not None:
            try:
                mcp.stop(None, None, None)
            except Exception:  # noqa: BLE001
                pass

    usage = getattr(result.metrics, "accumulated_usage", {}) or {}
    in_tok, out_tok = int(usage.get("inputTokens", 0)), int(usage.get("outputTokens", 0))
    tools_used = sorted(getattr(result.metrics, "tool_metrics", {}) or {})
    report = result.structured_output if isinstance(result.structured_output, TriageReport) else None
    md = report.to_markdown(alert, decision) if report else str(result)
    return InvestigationResult(
        report=report,
        report_markdown=md,
        model_label=spec.label,
        model_id=spec.model_id,
        input_tokens=in_tok,
        output_tokens=out_tok,
        model_cost_usd=spec.cost(in_tok, out_tok),
        tools_used=tools_used,
    )
