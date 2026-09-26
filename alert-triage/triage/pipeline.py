"""End-to-end triage: normalize -> recall -> Jev decide -> investigate -> record."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any

from .alerts import normalize
from .config import Settings
from .decision import decide
from .jev_client import JevClient, JevError
from .runbook import Runbook

log = logging.getLogger(__name__)

try:  # present on AgentCore Runtime via aws-opentelemetry-distro
    from opentelemetry import trace

    _tracer = trace.get_tracer("os-alert-triage")
except ImportError:  # pragma: no cover
    _tracer = None


@dataclass
class TriageOptions:
    decide_only: bool = False          # Jev + policy only; no Bedrock, no tools
    investigate_watch: bool = False    # also run the LLM for 'watch' alerts
    use_memory: bool = True


class TriagePipeline:
    def __init__(self, settings: Settings, runbook: Runbook | None = None):
        self.s = settings
        self.runbook = runbook or Runbook.load()
        self._jev: JevClient | None = None
        self._memory = None
        if settings.memory_enabled:
            from .memory import IncidentMemory

            self._memory = IncidentMemory(settings.memory_id, settings.region)

    def _jev_client(self) -> JevClient | None:
        if self._jev is None:
            try:
                from .credentials import get_jev_api_key

                self._jev = JevClient(api_key=get_jev_api_key(self.s), url=self.s.jev_api_url, model=self.s.jev_model)
            except (JevError, Exception) as exc:  # noqa: BLE001 - fail safe, policy handles None
                log.error("Jev client unavailable: %s", exc)
                return None
        return self._jev

    def run(self, payload: dict[str, Any], source: str | None = None, opts: TriageOptions | None = None) -> dict[str, Any]:
        opts = opts or TriageOptions()
        t0 = time.perf_counter()
        alert = normalize(payload, source)

        prior: list[str] = []
        if self._memory and opts.use_memory and alert.status == "firing":
            prior = self._memory.recall(alert)

        decision = decide(alert, self.runbook, self.s, self._jev_client(), prior)
        rule = self.runbook.get(decision.rule_id)
        self._annotate_span(decision)

        out: dict[str, Any] = {
            "alert": alert.to_state(),
            "decision": decision.to_dict(),
            "prior_incidents": len(prior),
            "investigation": None,
        }

        skip = opts.decide_only or decision.action == "record_only" or (
            decision.action == "watch" and not opts.investigate_watch
        )
        report_md = None
        if not skip:
            from .agent import investigate

            inv = investigate(alert, decision, rule, prior, self.s)
            report_md = inv.report_markdown
            out["investigation"] = {
                "model": inv.model_label,
                "model_id": inv.model_id,
                "input_tokens": inv.input_tokens,
                "output_tokens": inv.output_tokens,
                "model_cost_usd": round(inv.model_cost_usd, 6),
                "tools_used": inv.tools_used,
                "report": inv.report.model_dump() if inv.report else None,
                "report_markdown": report_md,
            }

        if self._memory and opts.use_memory and not opts.decide_only:
            self._memory.record(alert, out["decision"], report_md)

        inv_cost = (out["investigation"] or {}).get("model_cost_usd", 0.0)
        out["cost_usd_total"] = round(decision.jev_cost_usd + inv_cost, 6)
        out["elapsed_ms"] = round((time.perf_counter() - t0) * 1000, 1)
        return out

    @staticmethod
    def _annotate_span(d) -> None:
        if _tracer is None:
            return
        span = trace.get_current_span()
        if not span.is_recording():
            return
        span.set_attribute("triage.tier", d.tier.name)
        span.set_attribute("triage.action", d.action)
        span.set_attribute("triage.rule", d.rule_id)
        span.set_attribute("triage.jev_ok", d.jev_ok)
        if d.jev_tier_confidence is not None:
            span.set_attribute("triage.jev_tier_confidence", d.jev_tier_confidence)
            span.set_attribute("triage.jev_latency_ms", d.jev_latency_ms)
