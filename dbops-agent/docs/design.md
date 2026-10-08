# DBOps Agent design

Engineer/PagerDuty -> Bedrock Managed Agent -> AgentCore Runtime microVM -> dbops skills -> OpenSearch GET + CloudWatch read -> evidence/correlation -> RCA.

Security is enforced twice: application-level read-only checks and IAM read-only permissions. The application check is not a substitute for IAM.

Before production: scope IAM resources, use VPC Runtime for private domains, add MCP Gateway for PagerDuty/runbooks, add resource allowlists and audit IDs, enable GenAI Observability, test prompt injection and RCA accuracy, and require human approval for any future mutation capability.
