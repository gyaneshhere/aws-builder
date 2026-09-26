"""Settings from environment variables (.env locally, runtime env vars on AgentCore)."""

from __future__ import annotations

import os
from dataclasses import dataclass, field


def _f(name: str, default: float) -> float:
    return float(os.environ.get(name, default))


def _e(name: str, default: str = "") -> str:
    return os.environ.get(name, default)


@dataclass(frozen=True)
class Settings:
    region: str = field(default_factory=lambda: _e("AWS_REGION", "us-east-1"))

    # Jev
    jev_api_url: str = field(default_factory=lambda: _e("JEV_API_URL", "https://jevtypesafeai.com/api/v1/decide"))
    jev_model: str = field(default_factory=lambda: _e("JEV_MODEL", "jev-latest"))
    jev_api_key: str = field(default_factory=lambda: _e("JEV_API_KEY"))
    jev_api_key_provider: str = field(default_factory=lambda: _e("JEV_API_KEY_PROVIDER", "jev-api-key"))
    jev_metrics_per_call: int = field(default_factory=lambda: int(_e("JEV_METRICS_PER_CALL", "6")))

    # Statistics engine: "code_interpreter" (AgentCore sandbox) or "local" (same kernel in-process)
    stats_engine: str = field(default_factory=lambda: _e("STATS_ENGINE", "code_interpreter"))
    code_interpreter_id: str = field(default_factory=lambda: _e("CODE_INTERPRETER_ID", "aws.codeinterpreter.v1"))

    # Policy thresholds (code-enforced)
    alpha: float = field(default_factory=lambda: _f("ALPHA", 0.05))                 # family-wise, Holm-adjusted
    severity_flag_min: float = field(default_factory=lambda: _f("SEVERITY_FLAG_MIN", 2.0))
    severity_conf_min: float = field(default_factory=lambda: _f("SEVERITY_CONF_MIN", 0.50))
    artifact_min: float = field(default_factory=lambda: _f("ARTIFACT_MIN", 0.60))

    # Narrative model (only runs when something is flagged)
    narrative_model: str = field(default_factory=lambda: _e("NARRATIVE_MODEL", "us.anthropic.claude-sonnet-4-6"))
    narrative_max_tool_calls: int = field(default_factory=lambda: int(_e("NARRATIVE_MAX_TOOL_CALLS", "4")))

    # Memory
    memory_id: str = field(default_factory=lambda: _e("MEMORY_ID"))

    @property
    def memory_enabled(self) -> bool:
        return bool(self.memory_id)


def load_settings() -> Settings:
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass
    return Settings()
