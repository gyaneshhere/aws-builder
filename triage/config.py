"""Settings for the OpenSearch alert triage agent, read from environment variables.

Local runs read a .env file (see .env.example). On AgentCore Runtime the deploy
script injects these as runtime environment variables; secrets never go here:
the Jev API key and the Gateway OAuth client live in AgentCore Identity.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field


def _f(name: str, default: float) -> float:
    return float(os.environ.get(name, default))


@dataclass(frozen=True)
class Settings:
    region: str = field(default_factory=lambda: os.environ.get("AWS_REGION", "us-east-1"))

    # --- Jev -------------------------------------------------------------
    jev_api_url: str = field(
        default_factory=lambda: os.environ.get("JEV_API_URL", "https://jevtypesafeai.com/api/v1/decide")
    )
    # Pin a concrete version for production so decision boundaries don't drift.
    jev_model: str = field(default_factory=lambda: os.environ.get("JEV_MODEL", "jev-latest"))
    # Local dev: plain env var. Runtime: AgentCore Identity API-key provider.
    jev_api_key: str = field(default_factory=lambda: os.environ.get("JEV_API_KEY", ""))
    jev_api_key_provider: str = field(
        default_factory=lambda: os.environ.get("JEV_API_KEY_PROVIDER", "jev-api-key")
    )

    # --- Policy thresholds (deterministic code, not Jev) ----------------------
    tier_confidence_min: float = field(default_factory=lambda: _f("TIER_CONFIDENCE_MIN", 0.60))
    transient_watch_min: float = field(default_factory=lambda: _f("TRANSIENT_WATCH_MIN", 0.70))
    page_now_min: float = field(default_factory=lambda: _f("PAGE_NOW_MIN", 0.50))

    # --- Bedrock model per tier (inference-profile IDs) -----------------------
    model_l1: str = field(default_factory=lambda: os.environ.get("MODEL_L1", "us.amazon.nova-2-lite-v1:0"))
    model_l2: str = field(default_factory=lambda: os.environ.get("MODEL_L2", "us.amazon.nova-pro-v1:0"))
    model_l3: str = field(default_factory=lambda: os.environ.get("MODEL_L3", "us.anthropic.claude-sonnet-4-6"))

    # --- AgentCore Gateway (diagnostic tools over MCP) ------------------------
    gateway_url: str = field(default_factory=lambda: os.environ.get("GATEWAY_URL", ""))
    gateway_oauth_provider: str = field(
        default_factory=lambda: os.environ.get("GATEWAY_OAUTH_PROVIDER", "os-triage-gateway-oauth")
    )
    gateway_scope: str = field(default_factory=lambda: os.environ.get("GATEWAY_SCOPE", ""))
    # Local-dev fallback: direct Cognito client-credentials (never set on Runtime).
    gateway_token_url: str = field(default_factory=lambda: os.environ.get("GATEWAY_TOKEN_URL", ""))
    gateway_client_id: str = field(default_factory=lambda: os.environ.get("GATEWAY_CLIENT_ID", ""))
    gateway_client_secret: str = field(default_factory=lambda: os.environ.get("GATEWAY_CLIENT_SECRET", ""))

    # --- AgentCore Memory ------------------------------------------------------
    memory_id: str = field(default_factory=lambda: os.environ.get("MEMORY_ID", ""))

    @property
    def tools_enabled(self) -> bool:
        return bool(self.gateway_url)

    @property
    def memory_enabled(self) -> bool:
        return bool(self.memory_id)


def load_settings() -> Settings:
    try:  # optional in Runtime
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass
    return Settings()
