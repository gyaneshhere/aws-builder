"""Step 4: invoke the deployed runtime with an alert file.

    python -m deploy.invoke samples/alerts/cluster_red_prod.json
    python -m deploy.invoke samples/alerts/cpu_spike_nonprod.json --decide-only
"""

from __future__ import annotations

import argparse
import json
import uuid
from pathlib import Path

from botocore.config import Config

from deploy.common import die, load_state, session


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("alert_file", type=Path)
    ap.add_argument("--decide-only", action="store_true")
    ap.add_argument("--investigate-watch", action="store_true")
    a = ap.parse_args()

    st = load_state()
    if not st.get("runtime_arn"):
        die("no runtime_arn in .triage_state.json; run python -m deploy.deploy_runtime first")
    payload = {
        "alert": json.loads(a.alert_file.read_text()),
        "options": {"decide_only": a.decide_only, "investigate_watch": a.investigate_watch},
    }
    client = session().client("bedrock-agentcore", config=Config(read_timeout=300))
    resp = client.invoke_agent_runtime(
        agentRuntimeArn=st["runtime_arn"],
        runtimeSessionId=f"triage-{uuid.uuid4()}",  # >= 33 chars; one session per alert
        payload=json.dumps(payload).encode(),
        contentType="application/json",
        accept="application/json",
    )
    body = resp["response"].read().decode()
    try:
        result = json.loads(body)
    except json.JSONDecodeError:
        print(body)
        return
    inv = result.get("investigation") or {}
    print(json.dumps({k: v for k, v in result.items() if k != "investigation"}, indent=2))
    if inv:
        print(f"\nmodel={inv['model']} cost~=${inv['model_cost_usd']} tools={inv['tools_used']}\n")
        print(inv["report_markdown"])


if __name__ == "__main__":
    main()
