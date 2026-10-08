# DBOpsAgent Runtime application

AWS's generated `lifecycle/server.py`, Dockerfile, and ADOT configuration are authoritative. Run `../../scripts/bootstrap_from_aws.sh` before deployment.

Runtime environment:
- `BMA_HOME_DIR=/mnt/home`
- `DBOPS_READ_ONLY=true`
- `AWS_REGION=<runtime region>`
- `OPENSEARCH_ENDPOINT=https://...`

The diagnostic layer only issues OpenSearch GET/HEAD and CloudWatch read operations. IAM must independently enforce the same boundary.
