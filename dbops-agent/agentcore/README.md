# AgentCore configuration

`agentcore.json` is generated and managed by the current AgentCore CLI. Because the Bedrock Managed Agents integration is preview and the CLI owns the exact schema, this repository intentionally keeps a template instead of guessing the schema.

Run `../scripts/bootstrap_from_aws.sh` from the repository root. It invokes:

```bash
agentcore create --name DBOpsAgent --framework BedrockManagedAgents --session-storage-mount-path /mnt/home
```

If VPC variables are supplied, the script adds `--network-mode VPC`, `--vpc-id`, `--subnets`, and `--security-groups`.
