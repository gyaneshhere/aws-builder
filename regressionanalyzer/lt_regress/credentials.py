"""Jev API key: env var locally, AgentCore Identity API-key provider on Runtime."""

from __future__ import annotations

from .config import Settings


def get_jev_api_key(s: Settings) -> str:
    if s.jev_api_key:
        return s.jev_api_key
    from bedrock_agentcore.identity.auth import requires_api_key

    @requires_api_key(provider_name=s.jev_api_key_provider)
    def _fetch(*, api_key: str) -> str:
        return api_key

    return _fetch()
