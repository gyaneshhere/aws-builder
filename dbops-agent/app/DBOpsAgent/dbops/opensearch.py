"""Read-only Amazon OpenSearch Service diagnostics."""

from __future__ import annotations

import os
from typing import Any
from urllib.parse import urlparse

import boto3
from opensearchpy import AWSV4SignerAuth, OpenSearch

from .safety import validate_read_only_request


class OpenSearchDiagnostics:
    def __init__(self, endpoint: str | None = None, region: str | None = None):
        self.endpoint = endpoint or os.environ.get("OPENSEARCH_ENDPOINT")
        if not self.endpoint:
            raise ValueError("OPENSEARCH_ENDPOINT must be set")
        parsed = urlparse(self.endpoint)
        if parsed.scheme != "https" or not parsed.hostname:
            raise ValueError("OPENSEARCH_ENDPOINT must be an https URL")
        self.host = parsed.hostname
        self.region = region or os.environ.get("AWS_REGION") or boto3.session.Session().region_name
        if not self.region:
            raise ValueError("AWS_REGION must be set")
        credentials = boto3.Session().get_credentials()
        if credentials is None:
            raise RuntimeError("No AWS credentials available")
        auth = AWSV4SignerAuth(credentials, self.region, "es")
        self.client = OpenSearch(
            hosts=[{"host": self.host, "port": 443}],
            http_auth=auth,
            use_ssl=True,
            verify_certs=True,
            timeout=10,
            max_retries=1,
            retry_on_timeout=False,
        )

    def _get(self, path: str) -> Any:
        validate_read_only_request("GET", path)
        return self.client.transport.perform_request("GET", path)

    def snapshot(self) -> dict[str, Any]:
        """Collect a bounded set of safe health/performance signals."""
        return {
            "cluster_health": self._get("/_cluster/health"),
            "nodes": self._get("/_cat/nodes?format=json&h=name,ip,node.role,cpu,heap.percent,ram.percent,load_1m"),
            "thread_pools": self._get("/_nodes/stats/thread_pool"),
            "jvm": self._get("/_nodes/stats/jvm"),
            "indices": self._get("/_cat/indices?format=json&h=health,status,index,pri,rep,docs.count,store.size"),
            "shards": self._get("/_cat/shards?format=json&h=index,shard,prirep,state,docs,store,node"),
        }
