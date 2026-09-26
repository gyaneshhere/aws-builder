"""Delete the runtime, memory (baselines included!) and Identity provider.

    python -m deploy.teardown [--keep-memory] [--yes]
"""

from __future__ import annotations

import argparse
import os

from botocore.exceptions import ClientError

from deploy.common import REGION, STATE, load_state, session


def _try(label, fn, **kw):
    try:
        fn(**kw)
        print(f"  deleted {label}")
    except ClientError as e:
        print(f"  skip {label}: {e.response['Error']['Code']}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--keep-memory", action="store_true", help="keep baselines and findings")
    ap.add_argument("--yes", action="store_true")
    a = ap.parse_args()
    st = load_state()
    if not st:
        print("nothing to tear down")
        return
    what = "runtime, Identity provider" + ("" if a.keep_memory else " and MEMORY (all baselines)")
    if not a.yes and input(f"Delete {what} in {REGION}? [y/N] ") != "y":
        return
    sess = session()
    ctl = sess.client("bedrock-agentcore-control")
    if st.get("runtime_id"):
        _try("runtime", ctl.delete_agent_runtime, agentRuntimeId=st["runtime_id"])
    if st.get("jev_api_key_provider"):
        _try("jev api-key provider", ctl.delete_api_key_credential_provider, name=st["jev_api_key_provider"])
    if st.get("memory_id") and not a.keep_memory:
        _try("memory", ctl.delete_memory, memoryId=st["memory_id"])
    if not os.environ.get("RUNTIME_ROLE_ARN"):
        iam, name = sess.client("iam"), "lt-regression-analyzer-runtime-role"
        try:
            for p in iam.list_role_policies(RoleName=name)["PolicyNames"]:
                iam.delete_role_policy(RoleName=name, PolicyName=p)
            _try(f"role {name}", iam.delete_role, RoleName=name)
        except ClientError:
            pass
    if a.keep_memory:
        mem = st.get("memory_id")
        STATE.write_text(__import__("json").dumps({"memory_id": mem}, indent=2))
    else:
        STATE.unlink(missing_ok=True)
    print("done")


if __name__ == "__main__":
    main()
