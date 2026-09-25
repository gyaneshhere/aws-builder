"""Incident history in AgentCore Memory.

actor_id   = OpenSearch domain  -> semantic records land in /incidents/<domain>/
session_id = alert id           -> one short-term session per alert instance

recall() feeds prior incidents to both Jev (as state) and the LLM (as context),
so a domain that flapped three times this week is judged with that history.
Memory failures never block triage: they are logged and skipped.
"""

from __future__ import annotations

import json
import logging
import re

from .alerts import NormalizedAlert

log = logging.getLogger(__name__)
NAMESPACE_TEMPLATE = "/incidents/{actorId}/"


def _safe_id(s: str, default: str) -> str:
    s = re.sub(r"[^A-Za-z0-9_-]", "-", s or "")[:80].strip("-")
    return s or default


class IncidentMemory:
    def __init__(self, memory_id: str, region: str):
        from bedrock_agentcore.memory import MemoryClient

        self.memory_id = memory_id
        self.client = MemoryClient(region_name=region)

    @staticmethod
    def actor(alert: NormalizedAlert) -> str:
        return _safe_id(alert.domain, "unknown-domain")

    def recall(self, alert: NormalizedAlert, top_k: int = 3) -> list[str]:
        try:
            records = self.client.retrieve_memories(
                memory_id=self.memory_id,
                namespace=NAMESPACE_TEMPLATE.format(actorId=self.actor(alert)),
                query=f"{alert.rule} {alert.title} {alert.description}"[:900],
                top_k=top_k,
            )
        except Exception as exc:  # noqa: BLE001
            log.warning("memory recall failed: %s", exc)
            return []
        out = []
        for r in records:
            text = (r.get("content") or {}).get("text")
            if text:
                out.append(text[:500])
        return out

    def record(self, alert: NormalizedAlert, decision: dict, report_md: str | None) -> None:
        user_msg = json.dumps({"alert": alert.to_state()}, default=str)[:4000]
        summary = (
            f"Triage of {alert.rule} on {alert.domain or 'unknown domain'} ({alert.environment}): "
            f"tier {decision['tier']}, action {decision['action']}. "
            f"Reasons: {'; '.join(decision.get('reasons', []))}."
        )
        if report_md:
            summary += "\n\n" + report_md[:3500]
        try:
            self.client.create_event(
                memory_id=self.memory_id,
                actor_id=self.actor(alert),
                session_id=_safe_id(alert.alert_id, "alert"),
                messages=[(user_msg, "USER"), (summary, "ASSISTANT")],
            )
        except Exception as exc:  # noqa: BLE001
            log.warning("memory record failed: %s", exc)
