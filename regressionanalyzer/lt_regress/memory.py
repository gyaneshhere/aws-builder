"""Baselines and history in AgentCore Memory.

Two different kinds of memory, used for two different jobs:

* Exact baselines (short-term events). Numbers must round-trip exactly, so the
  steady-state LT summary of each *accepted* run is stored as JSON in a fixed
  session `baselines` under actor = test name, and read back with
  get_last_k_turns. Semantic extraction would paraphrase numbers, so it is not
  used for these.
* Findings (long-term semantic strategy). Every run's verdict + narrative is
  recorded in session = run id; the semantic strategy extracts facts into
  /lt-findings/{actorId}/ ("reader connections regressed in the 09-12 run"),
  which the narrative step retrieves as context.

A run only becomes the new baseline when its verdict is `pass`, or when a
human accepts it explicitly, so a regression can never quietly become normal.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from .models import Comparison

log = logging.getLogger(__name__)
FINDINGS_NAMESPACE = "/lt-findings/{actorId}/"
BASELINE_SESSION = "baselines"
_TAG = "LT_BASELINE_V1 "


def _text(msg: dict[str, Any]) -> str:
    c = msg.get("content")
    return c.get("text", "") if isinstance(c, dict) else str(c or "")


class RunMemory:
    def __init__(self, memory_id: str, region: str):
        from bedrock_agentcore.memory import MemoryClient

        self.memory_id = memory_id
        self.client = MemoryClient(region_name=region)

    # ---- exact baselines ----------------------------------------------------
    def latest_baseline(self, comp: Comparison) -> dict[str, dict[str, Any]]:
        try:
            turns = self.client.get_last_k_turns(
                memory_id=self.memory_id, actor_id=comp.actor_id, session_id=BASELINE_SESSION, k=3)
        except Exception as exc:  # noqa: BLE001
            log.warning("baseline read failed: %s", exc)
            return {}
        # newest first; take the first message carrying our tag
        for turn in turns:
            for msg in turn:
                t = _text(msg)
                if t.startswith(_TAG):
                    try:
                        return json.loads(t[len(_TAG):])["metrics"]
                    except (ValueError, KeyError):
                        continue
        return {}

    def save_baseline(self, comp: Comparison, stats: dict[str, Any]) -> None:
        metrics = {
            m["name"]: {"run_id": comp.run_id, "p50": m["lt"].get("p50"), "p95": m["lt"].get("p95"),
                        "mean": m["lt"].get("mean"), "n": m["lt"].get("n")}
            for m in stats["metrics"] if m.get("status") == "ok"
        }
        body = _TAG + json.dumps({"run_id": comp.run_id, "build": comp.build, "metrics": metrics})
        try:
            self.client.create_event(
                memory_id=self.memory_id, actor_id=comp.actor_id, session_id=BASELINE_SESSION,
                messages=[(f"accept baseline {comp.run_id}", "USER"), (body[:90000], "ASSISTANT")])
        except Exception as exc:  # noqa: BLE001
            log.warning("baseline save failed: %s", exc)

    # ---- findings -----------------------------------------------------------
    def recall_findings(self, comp: Comparison, query: str, top_k: int = 4) -> list[str]:
        try:
            recs = self.client.retrieve_memories(
                memory_id=self.memory_id, namespace=FINDINGS_NAMESPACE.format(actorId=comp.actor_id),
                query=query[:900], top_k=top_k)
        except Exception as exc:  # noqa: BLE001
            log.warning("findings recall failed: %s", exc)
            return []
        return [(r.get("content") or {}).get("text", "")[:400] for r in recs if (r.get("content") or {}).get("text")]

    def record_run(self, comp: Comparison, decision: dict[str, Any], narrative_md: str) -> None:
        flagged = [m["name"] + ":" + m["status"] for m in decision["metrics"]
                   if m["status"] in ("regression", "watch", "explained", "improvement")]
        user = f"LT run {comp.run_id} of {comp.test_name} (build {comp.build or 'n/a'}) vs {comp.prod.get('label', 'prod')}"
        summary = f"Verdict {decision['verdict'].upper()}. Notable metrics: {', '.join(flagged) or 'none'}.\n\n{narrative_md}"
        try:
            self.client.create_event(
                memory_id=self.memory_id, actor_id=comp.actor_id,
                session_id=re.sub(r"[^A-Za-z0-9_-]", "-", f"run-{comp.actor_id}-{comp.run_id}")[:100],
                messages=[(user, "USER"), (summary[:9000], "ASSISTANT")])
        except Exception as exc:  # noqa: BLE001
            log.warning("run record failed: %s", exc)
