"""Step 2: diagnostics Lambda + AgentCore Gateway + AgentCore Identity providers.

    python -m deploy.setup_gateway

Creates / reuses:
  1. Lambda `os-triage-diagnostics` (read-only tools). Domain endpoints and VPC
     settings are discovered from TRIAGE_DOMAINS via the OpenSearch API, so a
     VPC domain gets a Lambda in the same subnets/security groups.
  2. Cognito M2M authorizer + MCP Gateway `os-triage-gw` + Lambda target `opensearch-diag`.
  3. Identity OAuth2 provider (Runtime -> Gateway token) and API-key provider (Jev key).
"""

from __future__ import annotations

import io
import json
import os
import time
import zipfile

from botocore.exceptions import ClientError

from deploy.common import REGION, ROOT, account_id, die, ensure_role, load_state, save_state, session
from lambda_tools.handler import TOOL_SCHEMAS

LAMBDA_NAME = "os-triage-diagnostics"
GATEWAY_NAME = "os-triage-gw"
TARGET_NAME = "opensearch-diag"
OAUTH_PROVIDER = os.environ.get("GATEWAY_OAUTH_PROVIDER", "os-triage-gateway-oauth")
JEV_PROVIDER = os.environ.get("JEV_API_KEY_PROVIDER", "jev-api-key")


def discover_domains(sess, names: list[str]) -> tuple[dict[str, str], dict | None]:
    os_client = sess.client("opensearch")
    info = os_client.describe_domains(DomainNames=names)["DomainStatusList"]
    found = {d["DomainName"] for d in info}
    missing = set(names) - found
    if missing:
        die(f"domains not found in {REGION}: {sorted(missing)}")
    endpoints, vpc = {}, None
    for d in info:
        host = d.get("Endpoint") or (d.get("Endpoints") or {}).get("vpc")
        endpoints[d["DomainName"]] = f"https://{host}"
        v = d.get("VPCOptions")
        if v and v.get("SubnetIds"):
            if vpc and set(vpc["SubnetIds"]) != set(v["SubnetIds"]):
                print(f"  WARNING: {d['DomainName']} is in different subnets; deploy one Lambda per VPC")
            vpc = vpc or {"SubnetIds": v["SubnetIds"], "SecurityGroupIds": v["SecurityGroupIds"]}
    return endpoints, vpc


def lambda_zip() -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(ROOT / "lambda_tools" / "handler.py", "handler.py")
    return buf.getvalue()


def deploy_lambda(sess, acct: str, endpoints: dict[str, str], vpc: dict | None) -> str:
    lam = sess.client("lambda")
    role = ensure_role(
        sess, "LAMBDA_ROLE_ARN", "os-triage-diagnostics-role", "lambda-trust.json", "lambda-policy.json",
        REGION=REGION, ACCOUNT_ID=acct,
    )
    cfg = dict(
        FunctionName=LAMBDA_NAME, Role=role, Handler="handler.lambda_handler", Runtime="python3.12",
        Timeout=30, MemorySize=256, Architectures=["arm64"],
        Environment={"Variables": {"DOMAIN_ENDPOINTS": json.dumps(endpoints)}},
    )
    if vpc:
        cfg["VpcConfig"] = vpc
    try:
        arn = lam.get_function(FunctionName=LAMBDA_NAME)["Configuration"]["FunctionArn"]
        lam.update_function_code(FunctionName=LAMBDA_NAME, ZipFile=lambda_zip())
        lam.get_waiter("function_updated_v2").wait(FunctionName=LAMBDA_NAME)
        cfg.pop("FunctionName"), cfg.pop("Architectures")
        lam.update_function_configuration(FunctionName=LAMBDA_NAME, **cfg)
        print(f"  updated Lambda {arn}")
    except ClientError as e:
        if e.response["Error"]["Code"] != "ResourceNotFoundException":
            raise
        arn = lam.create_function(Code={"ZipFile": lambda_zip()}, **cfg)["FunctionArn"]
        print(f"  created Lambda {arn}")
    lam.get_waiter("function_active_v2").wait(FunctionName=LAMBDA_NAME)
    return arn


def upsert_oauth_provider(ctl, discovery_url: str, client_id: str, client_secret: str) -> None:
    body = {"customOauth2ProviderConfig": {
        "oauthDiscovery": {"discoveryUrl": discovery_url}, "clientId": client_id, "clientSecret": client_secret,
    }}
    try:
        ctl.create_oauth2_credential_provider(
            name=OAUTH_PROVIDER, credentialProviderVendor="CustomOauth2", oauth2ProviderConfigInput=body
        )
        print(f"  created Identity OAuth2 provider {OAUTH_PROVIDER}")
    except ClientError as e:
        if "ConflictException" not in str(e) and "already exists" not in str(e):
            raise
        ctl.update_oauth2_credential_provider(
            name=OAUTH_PROVIDER, credentialProviderVendor="CustomOauth2", oauth2ProviderConfigInput=body
        )
        print(f"  updated Identity OAuth2 provider {OAUTH_PROVIDER}")


def upsert_jev_provider(ctl) -> None:
    key = os.environ.get("JEV_API_KEY", "")
    if not key or key.startswith("jv_live_your"):
        print("  JEV_API_KEY not set: skipping Identity API-key provider (Runtime will fail safe to L3)")
        return
    try:
        ctl.create_api_key_credential_provider(name=JEV_PROVIDER, apiKey=key)
        print(f"  created Identity API-key provider {JEV_PROVIDER}")
    except ClientError as e:
        if "ConflictException" not in str(e) and "already exists" not in str(e):
            raise
        ctl.update_api_key_credential_provider(name=JEV_PROVIDER, apiKey=key)
        print(f"  updated Identity API-key provider {JEV_PROVIDER}")


def main() -> None:
    from bedrock_agentcore_starter_toolkit.operations.gateway.client import GatewayClient

    domains = [d.strip() for d in os.environ.get("TRIAGE_DOMAINS", "").split(",") if d.strip()]
    if not domains:
        die("set TRIAGE_DOMAINS=<domain1>,<domain2> in .env")
    sess = session()
    acct = account_id(sess)
    st = load_state()

    print("[1/3] diagnostics Lambda")
    endpoints, vpc = discover_domains(sess, domains)
    print(f"  domains: {json.dumps(endpoints)}  vpc: {bool(vpc)}")
    lambda_arn = deploy_lambda(sess, acct, endpoints, vpc)

    print("[2/3] AgentCore Gateway")
    gw_client = GatewayClient(region_name=REGION)
    if st.get("gateway_id"):
        gateway = sess.client("bedrock-agentcore-control").get_gateway(gatewayIdentifier=st["gateway_id"])
        cognito = st["cognito"]
        print(f"  reusing gateway {st['gateway_id']}")
    else:
        cognito = gw_client.create_oauth_authorizer_with_cognito(GATEWAY_NAME)
        role = None
        if os.environ.get("GATEWAY_ROLE_ARN"):
            role = os.environ["GATEWAY_ROLE_ARN"]
        gateway = gw_client.create_mcp_gateway(
            name=GATEWAY_NAME, role_arn=role, authorizer_config=cognito["authorizer_config"],
            enable_semantic_search=False,
        )
        save_state(gateway_id=gateway["gatewayId"], gateway_url=gateway["gatewayUrl"], cognito=cognito)

    ctl = sess.client("bedrock-agentcore-control")
    targets = ctl.list_gateway_targets(gatewayIdentifier=gateway["gatewayId"]).get("items", [])
    existing = next((t for t in targets if t["name"] == TARGET_NAME), None)
    payload = {"lambdaArn": lambda_arn, "toolSchema": {"inlinePayload": TOOL_SCHEMAS}}
    if existing:
        ctl.update_gateway_target(
            gatewayIdentifier=gateway["gatewayId"], targetId=existing["targetId"], name=TARGET_NAME,
            targetConfiguration={"mcp": {"lambda": payload}},
            credentialProviderConfigurations=[{"credentialProviderType": "GATEWAY_IAM_ROLE"}],
        )
        print(f"  updated target {TARGET_NAME}")
    else:
        gw_client.create_mcp_gateway_target(gateway=gateway, name=TARGET_NAME, target_type="lambda",
                                            target_payload=payload)
    # Gateway role must be able to invoke the Lambda (skipped if you supplied an admin-made role).
    if not os.environ.get("GATEWAY_ROLE_ARN"):
        from bedrock_agentcore_starter_toolkit.operations.gateway.create_role import append_lambda_target_permission

        append_lambda_target_permission(sess, gw_client.logger, gateway["roleArn"], lambda_arn, REGION)

    print("[3/3] AgentCore Identity providers")
    ci = cognito["client_info"]
    discovery = cognito["authorizer_config"]["customJWTAuthorizer"]["discoveryUrl"]
    upsert_oauth_provider(ctl, discovery, ci["client_id"], ci["client_secret"])
    upsert_jev_provider(ctl)

    st = save_state(
        lambda_arn=lambda_arn, gateway_scope=ci["scope"], gateway_oauth_provider=OAUTH_PROVIDER,
        jev_api_key_provider=JEV_PROVIDER, domain_endpoints=endpoints, updated=time.ctime(),
    )
    print("\nFor local runs add to .env:")
    print(f"GATEWAY_URL={st['gateway_url']}\nGATEWAY_SCOPE={ci['scope']}")
    print(f"GATEWAY_TOKEN_URL={ci['token_endpoint']}\nGATEWAY_CLIENT_ID={ci['client_id']}")
    print("GATEWAY_CLIENT_SECRET=<see .triage_state.json -> cognito.client_info.client_secret>")


if __name__ == "__main__":
    main()
