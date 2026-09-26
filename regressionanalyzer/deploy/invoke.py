"""Step 3: send a comparison to the deployed runtime.

    python -m deploy.invoke samples/psu_sinewave_vs_prod.json
    python -m deploy.invoke samples/psu_sinewave_vs_prod.json --decide-only
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
    ap.add_argument("comparison", type=Path)
    ap.add_argument("--decide-only", action="store_true")
    ap.add_argument("--accept-baseline", action="store_true")
    a = ap.parse_args()
    st = load_state()
    if not st.get("runtime_arn"):
        die("no runtime_arn in .ltr_state.json; run python -m deploy.deploy_runtime first")
    payload = {"comparison": json.loads(a.comparison.read_text()),
               "options": {"decide_only": a.decide_only, "accept_baseline": a.accept_baseline}}
    client = session().client("bedrock-agentcore", config=Config(read_timeout=600))
    resp = client.invoke_agent_runtime(
        agentRuntimeArn=st["runtime_arn"], runtimeSessionId=f"ltr-{uuid.uuid4()}-{uuid.uuid4().hex[:6]}",
        payload=json.dumps(payload).encode(), contentType="application/json", accept="application/json")
    body = resp["response"].read().decode()
    try:
        res = json.loads(body)
    except json.JSONDecodeError:
        print(body)
        return
    if "error" in res:
        die(res["error"])
    print(json.dumps({"verdict": res["verdict"], "engine": res["stats_engine"],
                      "metrics": {m["name"]: m["status"] for m in res["decision"]["metrics"]},
                      "cost_usd_total": res["cost_usd_total"], "elapsed_ms": res["elapsed_ms"]}, indent=2))
    if res.get("narrative"):
        print("\n" + res["narrative"]["markdown"])


if __name__ == "__main__":
    main()
