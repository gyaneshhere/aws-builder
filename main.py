"""AgentCore Runtime entrypoint.

Invoke payload:
    {"alert": {...grafana|pagerduty|generic payload...},
     "source": "grafana" | "pagerduty" | "generic"   (optional, auto-detected),
     "options": {"decide_only": false, "investigate_watch": false, "use_memory": true}}
"""

from __future__ import annotations

import logging

from bedrock_agentcore.runtime import BedrockAgentCoreApp

from triage.config import load_settings
from triage.pipeline import TriageOptions, TriagePipeline

logging.basicConfig(level=logging.INFO)
app = BedrockAgentCoreApp()
_pipeline: TriagePipeline | None = None


def pipeline() -> TriagePipeline:
    global _pipeline
    if _pipeline is None:  # built once per microVM session, reused across invocations
        _pipeline = TriagePipeline(load_settings())
    return _pipeline


@app.entrypoint
def invoke(payload: dict, context=None) -> dict:
    alert = payload.get("alert")
    if not isinstance(alert, dict):
        return {"error": "payload must contain an 'alert' object"}
    opts = TriageOptions(**{k: v for k, v in (payload.get("options") or {}).items()
                            if k in TriageOptions.__dataclass_fields__})
    try:
        return pipeline().run(alert, payload.get("source"), opts)
    except ValueError as exc:
        return {"error": f"bad alert payload: {exc}"}


if __name__ == "__main__":
    app.run()
