"""Step 1: create the AgentCore Memory used for incident history.

    python -m deploy.setup_memory
"""

from __future__ import annotations

from bedrock_agentcore.memory import MemoryClient

from deploy.common import REGION, load_state, save_state
from triage.memory import NAMESPACE_TEMPLATE

NAME = "OsAlertTriageMemory"


def main() -> None:
    st = load_state()
    if st.get("memory_id"):
        print(f"memory already created: {st['memory_id']}")
        return
    client = MemoryClient(region_name=REGION)
    existing = [m for m in client.list_memories() if (m.get("id") or m.get("memoryId") or "").startswith(NAME)]
    if existing:
        mem_id = existing[0].get("id") or existing[0]["memoryId"]
        print(f"reusing memory {mem_id}")
    else:
        print("creating memory (semantic strategy, 90-day event retention) ...")
        mem = client.create_memory_and_wait(
            name=NAME,
            description="OpenSearch alert triage incident history, one actor per domain",
            strategies=[{"semanticMemoryStrategy": {
                "name": "IncidentFacts",
                "description": "Facts about past OpenSearch incidents per domain",
                "namespaces": [NAMESPACE_TEMPLATE],
            }}],
            event_expiry_days=90,
        )
        mem_id = mem.get("id") or mem["memoryId"]
    save_state(memory_id=mem_id)
    print(f"MEMORY_ID={mem_id}")


if __name__ == "__main__":
    main()
