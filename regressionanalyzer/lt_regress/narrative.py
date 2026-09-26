"""Stakeholder narrative.

* Nothing flagged (verdict pass) -> deterministic template, no LLM call.
* Otherwise -> Claude on Bedrock writes a structured StakeholderReport about the
  flagged metrics only. It gets one tool, `run_analysis`, which executes Python
  in the SAME AgentCore Code Interpreter session that produced the statistics
  (INPUT and RESULT are already loaded there), so it can check things like
  "does the p99 regression line up with the reader-connection spike?" without
  ever doing arithmetic in its head. Calls are capped.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, Field

from .config import Settings
from .decision import RunDecision
from .models import Comparison

PRICE_IN, PRICE_OUT = 3.00, 15.00   # approx USD / 1M tokens for Claude Sonnet 4.6; verify current pricing


class Finding(BaseModel):
    metric: str
    status: str = Field(description="regression or watch")
    what_changed: str = Field(description="Plain-language change with numbers copied from the statistics.")
    evidence: list[str] = Field(description="Specific numbers from RESULT or run_analysis output, with their source.")
    likely_cause: str = Field(description="Hypothesis, clearly labeled as such, or 'unknown'.")
    recommended_action: str
    owner: str = Field(default="", description="Owner from metric metadata if given.")


class StakeholderReport(BaseModel):
    headline: str = Field(description="One sentence a producer or director can read in five seconds.")
    verdict_statement: str = Field(description="Pass / watch / fail and what it means for sign-off.")
    findings: list[Finding]
    explained_or_improved: list[str] = Field(default_factory=list,
                                             description="Short lines for explained deviations and improvements.")
    caveats: list[str] = Field(default_factory=list, description="Data or method limits a reviewer should know.")

    def to_markdown(self, comp: Comparison, decision: RunDecision) -> str:
        lines = [f"## {comp.test_name} · {comp.run_id} vs {comp.prod.get('label', 'Production')}",
                 f"**Verdict: {decision.verdict.upper()}** — {self.verdict_statement}", "", self.headline, ""]
        for f in self.findings:
            lines += [f"### {f.metric} ({f.status})", f.what_changed, ""]
            lines += [f"- {e}" for e in f.evidence]
            lines += ["", f"**Likely cause (hypothesis):** {f.likely_cause}",
                      f"**Action:** {f.recommended_action}" + (f" · **Owner:** {f.owner}" if f.owner else ""), ""]
        if self.explained_or_improved:
            lines += ["### Explained or improved"] + [f"- {x}" for x in self.explained_or_improved] + [""]
        if self.caveats:
            lines += ["### Caveats"] + [f"- {x}" for x in self.caveats]
        return "\n".join(lines).strip() + "\n"


SYSTEM_PROMPT = """You write release sign-off notes comparing a load test (LT) against Production.
Audience: producers, engineering managers and the DBA team. Be brief and precise.

Hard rules:
- Every number you write must be copied from STATISTICS or from run_analysis output. Never compute,
  estimate or round differently in your head; if you need a new number, call run_analysis.
- Write findings ONLY for metrics listed under FLAGGED. Mention explained/improved metrics in one line each.
- Label causes as hypotheses. Say 'unknown' rather than guess.
- Do not restate the statistical method; reviewers can read the appendix."""


def _compact_stats(stats: dict[str, Any], names: set[str]) -> list[dict[str, Any]]:
    keep = ("name", "unit", "direction", "slo", "lt", "prod", "median_delta", "median_delta_pct",
            "median_diff_ci95", "effect_size", "cliffs_delta", "slo_breach", "vs_baseline_pct",
            "lt_warmup", "warmup_samples_dropped")
    out = []
    for m in stats["metrics"]:
        if m["name"] in names:
            out.append({k: m.get(k) for k in keep if m.get(k) is not None})
    return out


@dataclass
class NarrativeResult:
    markdown: str
    report: StakeholderReport | None
    used_llm: bool
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    analysis_calls: int = 0


def template_pass(comp: Comparison, decision: RunDecision) -> NarrativeResult:
    improved = [m.name for m in decision.by_status("improvement")]
    explained = [m.name for m in decision.by_status("explained")]
    md = (f"## {comp.test_name} · {comp.run_id} vs {comp.prod.get('label', 'Production')}\n"
          f"**Verdict: PASS** — no metric shows a significant, practically relevant regression "
          f"({len(decision.metrics)} metrics compared).\n")
    if improved:
        md += f"\nImproved: {', '.join(improved)}.\n"
    if explained:
        md += f"\nDeviations attributed to test setup: {', '.join(explained)}.\n"
    return NarrativeResult(markdown=md, report=None, used_llm=False)


def write_narrative(comp: Comparison, stats: dict[str, Any], decision: RunDecision, engine,
                    history: list[str], cfg: Settings) -> NarrativeResult:
    flagged = decision.by_status("regression", "watch")
    if not flagged:
        return template_pass(comp, decision)

    from strands import Agent, tool
    from strands.models import BedrockModel

    calls = {"n": 0}
    tools = []
    if engine.supports_followup:
        @tool
        def run_analysis(code: str) -> str:
            """Run Python in the analysis sandbox. INPUT (the raw comparison with per-metric
            'lt'/'prod' sample lists) and RESULT (the statistics) are already loaded.
            Print what you need; output is truncated to 6000 characters."""
            if calls["n"] >= cfg.narrative_max_tool_calls:
                return "analysis call limit reached; write the report with what you have"
            calls["n"] += 1
            try:
                return engine.run_python(code)
            except Exception as exc:  # noqa: BLE001
                return f"error: {exc}"

        tools.append(run_analysis)

    names = {m.name for m in decision.metrics if m.status in ("regression", "watch", "explained", "improvement")}
    prompt = "\n\n".join([
        "RUN:\n" + json.dumps(comp.meta(), indent=2),
        "FLAGGED:\n" + json.dumps([{"metric": m.name, "status": m.status, "reasons": m.reasons,
                                    "owner": comp.metric_meta(m.name).get("owner", "")} for m in flagged], indent=2),
        "OTHER NOTABLE:\n" + json.dumps([{"metric": m.name, "status": m.status, "reasons": m.reasons}
                                         for m in decision.by_status("explained", "improvement")], indent=2),
        "STATISTICS (steady state, warm-up excluded):\n" + json.dumps(_compact_stats(stats, names), default=str),
        "PREVIOUS FINDINGS FOR THIS TEST:\n" + ("\n---\n".join(history) if history else "none"),
        ("You may call run_analysis to check correlations or time alignment between flagged metrics."
         if tools else "No analysis tool is available in this run."),
    ])
    agent = Agent(
        model=BedrockModel(model_id=cfg.narrative_model, region_name=cfg.region, temperature=0.2, max_tokens=4000),
        tools=tools, system_prompt=SYSTEM_PROMPT, structured_output_model=StakeholderReport,
        callback_handler=None, trace_attributes={"lt.verdict": decision.verdict, "lt.test": comp.test_name},
    )
    result = agent(prompt)
    usage = getattr(result.metrics, "accumulated_usage", {}) or {}
    tin, tout = int(usage.get("inputTokens", 0)), int(usage.get("outputTokens", 0))
    report = result.structured_output if isinstance(result.structured_output, StakeholderReport) else None
    return NarrativeResult(
        markdown=report.to_markdown(comp, decision) if report else str(result),
        report=report, used_llm=True, input_tokens=tin, output_tokens=tout,
        cost_usd=round(tin / 1e6 * PRICE_IN + tout / 1e6 * PRICE_OUT, 6), analysis_calls=calls["n"],
    )
