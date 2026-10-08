---
name: opensearch-diagnostics
description: Read-only diagnostics for Amazon OpenSearch Service domains: health, nodes, JVM, thread pools, shards, and indices.
---
# OpenSearch Diagnostics

Use the repository diagnostic CLI:

```bash
python -m dbops.cli opensearch-snapshot
```

Requires `OPENSEARCH_ENDPOINT` and AWS IAM credentials.

Inspect cluster health, node CPU/heap/load, JVM, search/write thread pools and rejections, shard states/distribution, and index health/size/document counts.

Interpretation guardrails:
- Green health does not prove performance is healthy.
- Search queue + increased workload + CPU is stronger saturation evidence than queue alone.
- High heap without GC/latency correlation is insufficient to call JVM pressure root cause.
- Unassigned shards require investigation but do not automatically explain search latency.
