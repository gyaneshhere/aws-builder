---
name: cloudwatch-analysis
description: Analyze CloudWatch metrics and logs read-only to establish baselines and correlate incident signals.
---
# CloudWatch Analysis

Use bounded read-only queries:

```bash
python -m dbops.cli cloudwatch-metric --namespace AWS/ES --metric-name CPUUtilization --dimension DomainName=<DOMAIN> --dimension ClientId=<ACCOUNT_ID>
```

Use the actual namespace, metric, and dimensions; never invent them. Establish a baseline, inspect the incident window, compare trend/peak/average, correlate independent signals, and explicitly report missing data.
