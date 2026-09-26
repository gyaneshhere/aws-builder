"""Remove everything the deploy scripts created (roles you supplied are left alone).

    python -m deploy.teardown            # asks for confirmation
    python -m deploy.teardown --yes
"""

from __future__ import annotations

import argparse
import os

from botocore.exceptions import ClientError

from deploy.common import REGION, STATE, load_state, session


def _try(label: str, fn, *args, **kw) -> None:
    try:
        fn(*args, **kw)
        print(f"  deleted {label}")
    except ClientError as e:
        print(f"  skip {label}: {e.response['Error']['Code']}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--yes", action="store_true")
    a = ap.parse_args()
    st = load_state()
    if not st:
        print("nothing to tear down")
        return
    if not a.yes and input(f"Delete runtime, gateway, lambda, identity providers and memory in {REGION}? [y/N] ") != "y":
        return
    sess = session()
    ctl = sess.client("bedrock-agentcore-control")

    if st.get("runtime_id"):
        _try("runtime", ctl.delete_agent_runtime, agentRuntimeId=st["runtime_id"])
    if st.get("gateway_id"):
        from bedrock_agentcore_starter_toolkit.operations.gateway.client import GatewayClient

        try:
            GatewayClient(region_name=REGION).cleanup_gateway(st["gateway_id"], st.get("cognito", {}).get("client_info"))
            print("  deleted gateway + targets + cognito")
        except Exception as e:  # noqa: BLE001
            print(f"  gateway cleanup incomplete: {e}")
    if st.get("gateway_oauth_provider"):
        _try("oauth provider", ctl.delete_oauth2_credential_provider, name=st["gateway_oauth_provider"])
    if st.get("jev_api_key_provider"):
        _try("jev api-key provider", ctl.delete_api_key_credential_provider, name=st["jev_api_key_provider"])
    if st.get("lambda_arn"):
        _try("lambda", sess.client("lambda").delete_function, FunctionName="os-triage-diagnostics")
    if st.get("memory_id"):
        _try("memory", ctl.delete_memory, memoryId=st["memory_id"])

    iam = sess.client("iam")
    for env_var, name in [("LAMBDA_ROLE_ARN", "os-triage-diagnostics-role"),
                          ("RUNTIME_ROLE_ARN", "os-alert-triage-runtime-role")]:
        if os.environ.get(env_var):
            continue
        try:
            for p in iam.list_role_policies(RoleName=name)["PolicyNames"]:
                iam.delete_role_policy(RoleName=name, PolicyName=p)
            _try(f"role {name}", iam.delete_role, RoleName=name)
        except ClientError:
            pass
    STATE.unlink(missing_ok=True)
    print("done")


if __name__ == "__main__":
    main()
