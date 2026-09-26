"""Shared helpers for deploy scripts: env, boto3 session, state file, IAM roles."""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import boto3
from botocore.exceptions import ClientError

ROOT = Path(__file__).resolve().parent.parent
STATE = ROOT / ".triage_state.json"
IAM_DIR = ROOT / "iam"

try:
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
except ImportError:
    pass

REGION = os.environ.get("AWS_REGION", "us-east-1")


def session() -> boto3.Session:
    return boto3.Session(profile_name=os.environ.get("AWS_PROFILE") or None, region_name=REGION)


def account_id(sess: boto3.Session) -> str:
    return sess.client("sts").get_caller_identity()["Account"]


def load_state() -> dict:
    return json.loads(STATE.read_text()) if STATE.exists() else {}


def save_state(**kv) -> dict:
    st = load_state() | kv
    STATE.write_text(json.dumps(st, indent=2, default=str))
    return st


def die(msg: str) -> None:
    print(f"ERROR: {msg}", file=sys.stderr)
    sys.exit(1)


def render_policy(name: str, **subs: str) -> str:
    text = (IAM_DIR / name).read_text()
    for k, v in subs.items():
        text = text.replace("${" + k + "}", v)
    return text


def ensure_role(sess: boto3.Session, env_var: str, role_name: str, trust_file: str,
                policy_file: str, managed: list[str] | None = None, **subs: str) -> str:
    """Use a pre-created role from env, else try to create it (needs IAM rights)."""
    if os.environ.get(env_var):
        return os.environ[env_var]
    iam = sess.client("iam")
    try:
        return iam.get_role(RoleName=role_name)["Role"]["Arn"]
    except ClientError as e:
        if e.response["Error"]["Code"] != "NoSuchEntity":
            raise
    try:
        arn = iam.create_role(
            RoleName=role_name, AssumeRolePolicyDocument=render_policy(trust_file, **subs)
        )["Role"]["Arn"]
        iam.put_role_policy(RoleName=role_name, PolicyName=f"{role_name}-inline",
                            PolicyDocument=render_policy(policy_file, **subs))
        for m in managed or []:
            iam.attach_role_policy(RoleName=role_name, PolicyArn=m)
        print(f"  created IAM role {role_name}; waiting for propagation...")
        time.sleep(12)
        return arn
    except ClientError as e:
        die(
            f"cannot create IAM role {role_name} ({e.response['Error']['Code']}). "
            f"Ask an admin to create it from iam/{trust_file} + iam/{policy_file} "
            f"(placeholders: {subs}) and set {env_var}=<arn> in .env"
        )
        raise  # unreachable
