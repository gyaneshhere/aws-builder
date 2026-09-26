"""Step 3: package for linux/arm64 and create or update the AgentCore Runtime.

    python -m deploy.deploy_runtime            # build + upload + create/update
    python -m deploy.deploy_runtime --build-only

Direct code deploy (no Docker): dependencies are installed as manylinux aarch64
wheels for Python 3.12, zipped with main.py + triage/, uploaded to DEPLOY_BUCKET,
and the runtime is started with `opentelemetry-instrument main.py` so traces
flow to AgentCore Observability.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path

from botocore.exceptions import ClientError

from deploy.common import REGION, ROOT, account_id, die, ensure_role, load_state, save_state, session

RUNTIME_NAME = "os_alert_triage"
BUILD = ROOT / "build"
PLATFORMS = ["manylinux2014_aarch64", "manylinux_2_17_aarch64", "manylinux_2_28_aarch64"]


def build_package() -> Path:
    pkg = BUILD / "pkg"
    shutil.rmtree(BUILD, ignore_errors=True)
    pkg.mkdir(parents=True)
    cmd = [sys.executable, "-m", "pip", "install", "--quiet", "--target", str(pkg),
           "--python-version", "3.12", "--implementation", "cp", "--only-binary=:all:",
           "-r", str(ROOT / "requirements.txt")]
    for p in PLATFORMS:
        cmd += ["--platform", p]
    print("  installing arm64 wheels ...")
    subprocess.run(cmd, check=True)
    shutil.copy(ROOT / "main.py", pkg / "main.py")
    shutil.copytree(ROOT / "triage", pkg / "triage", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))

    zpath = BUILD / "package.zip"
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        for f in pkg.rglob("*"):
            if f.is_file() and "__pycache__" not in f.parts:
                z.write(f, f.relative_to(pkg))
    print(f"  package: {zpath} ({zpath.stat().st_size / 1e6:.1f} MB)")
    return zpath


def runtime_env(st: dict) -> dict[str, str]:
    env = {
        "AWS_REGION": REGION,
        "JEV_API_URL": os.environ.get("JEV_API_URL", "https://jevtypesafeai.com/api/v1/decide"),
        "JEV_MODEL": os.environ.get("JEV_MODEL", "jev-latest"),
        "JEV_API_KEY_PROVIDER": st.get("jev_api_key_provider", "jev-api-key"),
        "GATEWAY_URL": st.get("gateway_url", ""),
        "GATEWAY_SCOPE": st.get("gateway_scope", ""),
        "GATEWAY_OAUTH_PROVIDER": st.get("gateway_oauth_provider", "os-triage-gateway-oauth"),
        "MEMORY_ID": st.get("memory_id", ""),
    }
    for k in ("TIER_CONFIDENCE_MIN", "TRANSIENT_WATCH_MIN", "PAGE_NOW_MIN", "MODEL_L1", "MODEL_L2", "MODEL_L3"):
        if os.environ.get(k):
            env[k] = os.environ[k]
    # Never ship local-dev secrets to the runtime.
    return {k: v for k, v in env.items() if v}


def find_runtime(ctl) -> dict | None:
    token = None
    while True:
        resp = ctl.list_agent_runtimes(**({"nextToken": token} if token else {}))
        for r in resp.get("agentRuntimes", []):
            if r["agentRuntimeName"] == RUNTIME_NAME:
                return r
        token = resp.get("nextToken")
        if not token:
            return None


def wait_ready(ctl, runtime_id: str) -> str:
    for _ in range(90):
        status = ctl.get_agent_runtime(agentRuntimeId=runtime_id)["status"]
        if status in ("READY", "CREATE_FAILED", "UPDATE_FAILED"):
            return status
        time.sleep(10)
    return "TIMEOUT"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--build-only", action="store_true")
    a = ap.parse_args()

    print("[1/3] build")
    zpath = build_package()
    if a.build_only:
        return

    bucket = os.environ.get("DEPLOY_BUCKET")
    if not bucket or bucket == "your-artifact-bucket":
        die("set DEPLOY_BUCKET in .env")
    st = load_state()
    if not st.get("gateway_url"):
        print("  WARNING: no gateway in state; runtime will triage without diagnostic tools")
    if not st.get("memory_id"):
        print("  WARNING: no memory in state; runtime will triage without incident history")

    sess = session()
    acct = account_id(sess)
    key = f"os-alert-triage/{time.strftime('%Y%m%d-%H%M%S')}/package.zip"
    print(f"[2/3] upload s3://{bucket}/{key}")
    sess.client("s3").upload_file(str(zpath), bucket, key)

    role = ensure_role(
        sess, "RUNTIME_ROLE_ARN", "os-alert-triage-runtime-role", "runtime-trust.json", "runtime-policy.json",
        REGION=REGION, ACCOUNT_ID=acct, DEPLOY_BUCKET=bucket,
    )
    ctl = sess.client("bedrock-agentcore-control")
    spec = dict(
        agentRuntimeArtifact={"codeConfiguration": {
            "code": {"s3": {"bucket": bucket, "prefix": key}},
            "runtime": "PYTHON_3_12",
            "entryPoint": ["opentelemetry-instrument", "main.py"],
        }},
        roleArn=role,
        networkConfiguration={"networkMode": "PUBLIC"},  # needs egress to the Jev API
        protocolConfiguration={"serverProtocol": "HTTP"},
        environmentVariables=runtime_env(st),
    )

    print("[3/3] create/update runtime")
    existing = find_runtime(ctl)
    try:
        if existing:
            resp = ctl.update_agent_runtime(agentRuntimeId=existing["agentRuntimeId"], **spec)
        else:
            resp = ctl.create_agent_runtime(agentRuntimeName=RUNTIME_NAME,
                                            description="Jev-routed OpenSearch alert triage", **spec)
    except ClientError as e:
        die(f"runtime deploy failed: {e}")
    rid, arn = resp["agentRuntimeId"], resp["agentRuntimeArn"]
    status = wait_ready(ctl, rid)
    save_state(runtime_id=rid, runtime_arn=arn, runtime_version=resp.get("agentRuntimeVersion"))
    print(f"  {RUNTIME_NAME} -> {status}\n  ARN: {arn}")
    if status != "READY":
        die("runtime not READY; check CloudWatch logs /aws/bedrock-agentcore/runtimes/")


if __name__ == "__main__":
    main()
