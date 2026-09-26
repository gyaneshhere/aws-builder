"""AgentCore Runtime entrypoint.

Payload:
  {"comparison": {...see lt_regress/models.py...},
   "options": {"decide_only": false, "accept_baseline": false, "use_memory": true}}
"""

from __future__ import annotations

import logging

from bedrock_agentcore.runtime import BedrockAgentCoreApp

from lt_regress.config import load_settings
from lt_regress.models import Comparison
from lt_regress.pipeline import Analyzer, RunOptions

logging.basicConfig(level=logging.INFO)
app = BedrockAgentCoreApp()
_analyzer: Analyzer | None = None


@app.entrypoint
def invoke(payload: dict, context=None) -> dict:
    global _analyzer
    if not isinstance(payload.get("comparison"), dict):
        return {"error": "payload must contain a 'comparison' object"}
    try:
        comp = Comparison.from_dict(payload["comparison"])
    except (ValueError, KeyError, TypeError) as exc:
        return {"error": f"bad comparison: {exc}"}
    opts = RunOptions(**{k: v for k, v in (payload.get("options") or {}).items()
                         if k in RunOptions.__dataclass_fields__ and k != "stats_engine"})
    _analyzer = _analyzer or Analyzer(load_settings())
    return _analyzer.run(comp, opts)


if __name__ == "__main__":
    app.run()
