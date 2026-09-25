"""Credentials via AgentCore Identity, with local-dev fallbacks.

On AgentCore Runtime:
  * Jev API key     -> Identity API-key credential provider (requires_api_key)
  * Gateway token   -> Identity OAuth2 provider, M2M client-credentials (requires_access_token)
Locally: JEV_API_KEY and GATEWAY_TOKEN_URL/CLIENT_ID/CLIENT_SECRET env vars.
"""

from __future__ import annotations

import time

import httpx

from .config import Settings

_cache: dict[str, tuple[str, float]] = {}


def get_jev_api_key(s: Settings) -> str:
    if s.jev_api_key:
        return s.jev_api_key
    from bedrock_agentcore.identity.auth import requires_api_key

    @requires_api_key(provider_name=s.jev_api_key_provider)
    def _fetch(*, api_key: str) -> str:
        return api_key

    return _fetch()


def get_gateway_token(s: Settings) -> str:
    cached = _cache.get("gw")
    if cached and cached[1] > time.time() + 60:
        return cached[0]

    if s.gateway_token_url and s.gateway_client_id:
        resp = httpx.post(
            s.gateway_token_url,
            data={
                "grant_type": "client_credentials",
                "client_id": s.gateway_client_id,
                "client_secret": s.gateway_client_secret,
                "scope": s.gateway_scope,
            },
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            timeout=10,
        )
        resp.raise_for_status()
        body = resp.json()
        token, ttl = body["access_token"], float(body.get("expires_in", 3600))
    else:
        from bedrock_agentcore.identity.auth import requires_access_token

        @requires_access_token(
            provider_name=s.gateway_oauth_provider,
            scopes=[s.gateway_scope] if s.gateway_scope else [],
            auth_flow="M2M",
        )
        def _fetch(*, access_token: str) -> str:
            return access_token

        token, ttl = _fetch(), 3000.0

    _cache["gw"] = (token, time.time() + ttl)
    return token
