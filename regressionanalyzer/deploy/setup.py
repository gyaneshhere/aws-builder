"""Step 1: AgentCore Memory (baselines + findings) and the Identity provider for the Jev key.

    python -m deploy.setup

Code Interpreter needs no setup: the analyzer uses the AWS-managed
`aws.codeinterpreter.v1` sandbox (override with CODE_INTERPRETER_ID).
"""

from __future__ import annotations

import os

from botocore.exceptions import ClientError
from bedrock_agentcore.memory import MemoryClient

from deploy.common import REGION, load_state, save_state, session
from lt_regress.memory import FINDINGS_NAMESPACE

MEMORY_NAME = "LtRegressionMemory"
JEV_PROVIDER = os.environ.get("JEV_API_KEY_PROVIDER", "jev-api-key")


def ensure_memory() -> str:
    st = load_state()
    if st.get("memory_id"):
        return st["memory_id"]
    client = MemoryClient(region_name=REGION)
    for m in client.list_memories():
        mid = m.get("id") or m.get("memoryId") or ""
        if mid.startswith(MEMORY_NAME):
            print(f"  reusing memory {mid}")
            return mid
    print("  creating memory (short-term events for baselines + semantic strategy for findings) ...")
    mem = client.create_memory_and_wait(
        name=MEMORY_NAME,
        description="LT vs Prod baselines (exact, per test) and findings (semantic)",
        strategies=[{"semanticMemoryStrategy": {
            "name": "LtFindings",
            "description": "Facts about regressions and verdicts from past LT comparisons",
            "namespaces": [FINDINGS_NAMESPACE],
        }}],
        event_expiry_days=365,   # baselines are events: keep them for a release cycle
    )
    return mem.get("id") or mem["memoryId"]


def ensure_jev_provider() -> str | None:
    key = os.environ.get("JEV_API_KEY", "")
    if not key or key.startswith("jv_live_your"):
        print("  JEV_API_KEY not set: skipping Identity provider (runtime falls back to statistics-only)")
        return None
    ctl = session().client("bedrock-agentcore-control")
    try:
        ctl.create_api_key_credential_provider(name=JEV_PROVIDER, apiKey=key)
        print(f"  created Identity API-key provider {JEV_PROVIDER}")
    except ClientError as e:
        if "ConflictException" not in str(e) and "already exists" not in str(e):
            raise
        ctl.update_api_key_credential_provider(name=JEV_PROVIDER, apiKey=key)
        print(f"  updated Identity API-key provider {JEV_PROVIDER}")
    return JEV_PROVIDER


def main() -> None:
    print("[1/2] AgentCore Memory")
    mem_id = ensure_memory()
    print("[2/2] AgentCore Identity")
    provider = ensure_jev_provider()
    save_state(memory_id=mem_id, **({"jev_api_key_provider": provider} if provider else {}))
    print(f"\nMEMORY_ID={mem_id}   (add to .env for local runs)")


if __name__ == "__main__":
    main()
