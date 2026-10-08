#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; TMP="$(mktemp -d)"; trap 'rm -rf "$TMP"' EXIT
: "${AWS_REGION:=us-east-1}"; export AWS_REGION
command -v agentcore >/dev/null 2>&1 || { echo 'Install AgentCore CLI: npm install -g @aws/agentcore' >&2; exit 1; }
ARGS=(create --name DBOpsAgent --framework BedrockManagedAgents --session-storage-mount-path /mnt/home)
if [[ -n "${VPC_ID:-}" ]]; then
  [[ -n "${SUBNET_IDS:-}" && -n "${SECURITY_GROUP_IDS:-}" ]] || { echo 'VPC_ID requires SUBNET_IDS and SECURITY_GROUP_IDS' >&2; exit 1; }
  IFS=',' read -r -a SUBNET_ARRAY <<< "$SUBNET_IDS"; IFS=',' read -r -a SG_ARRAY <<< "$SECURITY_GROUP_IDS"
  ARGS+=(--network-mode VPC --vpc-id "$VPC_ID" --subnets "${SUBNET_ARRAY[@]}" --security-groups "${SG_ARRAY[@]}")
fi
(cd "$TMP" && agentcore "${ARGS[@]}")
G="$TMP/DBOpsAgent"
[[ -f "$G/app/DBOpsAgent/lifecycle/server.py" ]] || { echo 'AWS CLI did not generate lifecycle/server.py' >&2; exit 1; }
cp "$G/app/DBOpsAgent/lifecycle/server.py" "$ROOT/app/DBOpsAgent/lifecycle/server.py"
cp "$G/app/DBOpsAgent/Dockerfile" "$ROOT/app/DBOpsAgent/Dockerfile"
cp "$G/app/DBOpsAgent/otel/collector.yaml" "$ROOT/app/DBOpsAgent/otel/collector.yaml"
cp "$G/app/DBOpsAgent/pyproject.toml" "$ROOT/app/DBOpsAgent/pyproject.aws-generated.toml"
cp "$G/agentcore/agentcore.json" "$ROOT/agentcore/agentcore.json"
rm -rf "$ROOT/agentcore/cdk"; cp -R "$G/agentcore/cdk" "$ROOT/agentcore/cdk"
for f in bma-acr-policy.json bma-session-trust.json bma-session-policy.json; do [[ -f "$G/app/DBOpsAgent/policies/$f" ]] && cp "$G/app/DBOpsAgent/policies/$f" "$ROOT/app/DBOpsAgent/policies/aws-generated-$f"; done
echo 'Bootstrap complete. Merge the DBOps read-only IAM permissions into the generated runtime policy. Do not edit lifecycle/server.py.'
