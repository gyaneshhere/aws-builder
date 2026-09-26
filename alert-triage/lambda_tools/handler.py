"""Read-only OpenSearch diagnostics, exposed as MCP tools via AgentCore Gateway.

Gateway invokes this Lambda with the tool arguments as `event` and the tool name
in context.client_context.custom["bedrockAgentCoreToolName"] as
"<target>___<tool>".

Safety model
  * Only HTTP GET is ever issued; there is no code path for PUT/POST/DELETE.
  * Domains are resolved from an allowlist (DOMAIN_ENDPOINTS env JSON); a model
    cannot point the tool at an arbitrary host.
  * CloudWatch metric names are allowlisted.
  * Responses are trimmed so tool output can't flood the model context.

Env
  DOMAIN_ENDPOINTS  JSON {"<domain-name>": "https://vpc-...es.amazonaws.com", ...}
  AWS_REGION        set by Lambda
Stdlib + boto3 only (both present in the Lambda Python runtime).
"""

from __future__ import annotations

import datetime as dt
import json
import os
import urllib.error
import urllib.request

import boto3
from botocore.auth import SigV4Auth
from botocore.awsrequest import AWSRequest

REGION = os.environ.get("AWS_REGION", "us-east-1")
MAX_CHARS = 6000
CW_METRICS = {
    "ClusterStatus.red", "ClusterStatus.yellow", "ClusterStatus.green", "FreeStorageSpace",
    "ClusterIndexWritesBlocked", "Nodes", "CPUUtilization", "JVMMemoryPressure",
    "OldGenJVMMemoryPressure", "MasterCPUUtilization", "MasterJVMMemoryPressure",
    "MasterReachableFromNode", "ThreadpoolWriteRejected", "ThreadpoolSearchRejected",
    "ThreadpoolWriteQueue", "ThreadpoolSearchQueue", "SearchLatency", "IndexingLatency",
    "5xx", "AutomatedSnapshotFailure", "KMSKeyError", "KMSKeyInaccessible",
}

_session = boto3.Session()
_cw = _session.client("cloudwatch", region_name=REGION)
_account = None


def _endpoints() -> dict[str, str]:
    return json.loads(os.environ.get("DOMAIN_ENDPOINTS", "{}"))


def _get(domain: str, path: str) -> object:
    eps = _endpoints()
    if domain not in eps:
        raise ValueError(f"domain '{domain}' is not in the allowlist ({sorted(eps)})")
    url = eps[domain].rstrip("/") + path
    req = AWSRequest(method="GET", url=url, headers={"Accept": "application/json"})
    SigV4Auth(_session.get_credentials().get_frozen_credentials(), "es", REGION).add_auth(req)
    r = urllib.request.Request(url, method="GET", headers=dict(req.headers))
    try:
        with urllib.request.urlopen(r, timeout=8) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        return {"error": f"HTTP {e.code}", "body": e.read().decode()[:500]}


def get_cluster_health(domain: str) -> object:
    return _get(domain, "/_cluster/health")


def get_nodes(domain: str) -> object:
    cols = "name,node.role,heap.percent,ram.percent,cpu,load_1m,disk.used_percent,uptime"
    return _get(domain, f"/_cat/nodes?format=json&h={cols}&s=heap.percent:desc")


def get_thread_pools(domain: str) -> object:
    rows = _get(domain, "/_cat/thread_pool/write,search?format=json&h=node_name,name,active,queue,rejected")
    return [r for r in rows if isinstance(r, dict) and (r.get("queue") != "0" or r.get("rejected") != "0")] \
        if isinstance(rows, list) else rows


def get_unassigned_shards(domain: str) -> object:
    rows = _get(domain, "/_cat/shards?format=json&h=index,shard,prirep,state,unassigned.reason")
    if not isinstance(rows, list):
        return rows
    un = [r for r in rows if r.get("state") == "UNASSIGNED"]
    return {"unassigned_count": len(un), "primaries": sum(r.get("prirep") == "p" for r in un), "sample": un[:25]}


def get_pending_tasks(domain: str) -> object:
    data = _get(domain, "/_cluster/pending_tasks")
    tasks = data.get("tasks", []) if isinstance(data, dict) else []
    return {"count": len(tasks), "oldest": tasks[:10]}


def get_allocation_explain(domain: str) -> object:
    # GET without a body explains the first unassigned shard; 400 means none are unassigned.
    return _get(domain, "/_cluster/allocation/explain")


def get_cloudwatch_metric(domain: str, metric: str, minutes: int = 60, statistic: str = "Maximum") -> object:
    global _account
    if metric not in CW_METRICS:
        raise ValueError(f"metric '{metric}' not allowed; choose from {sorted(CW_METRICS)}")
    if domain not in _endpoints():
        raise ValueError(f"domain '{domain}' is not in the allowlist")
    statistic = statistic if statistic in {"Maximum", "Minimum", "Average", "Sum"} else "Maximum"
    minutes = max(5, min(int(minutes), 1440))
    _account = _account or _session.client("sts").get_caller_identity()["Account"]
    end = dt.datetime.now(dt.timezone.utc)
    period = 60 if minutes <= 180 else 300
    resp = _cw.get_metric_statistics(
        Namespace="AWS/ES", MetricName=metric,
        Dimensions=[{"Name": "DomainName", "Value": domain}, {"Name": "ClientId", "Value": _account}],
        StartTime=end - dt.timedelta(minutes=minutes), EndTime=end, Period=period, Statistics=[statistic],
    )
    pts = sorted(resp.get("Datapoints", []), key=lambda p: p["Timestamp"])
    vals = [p[statistic] for p in pts]
    return {
        "metric": metric, "statistic": statistic, "window_minutes": minutes, "points": len(vals),
        "min": min(vals) if vals else None, "max": max(vals) if vals else None,
        "last": vals[-1] if vals else None,
        "series_tail": [{"t": p["Timestamp"].isoformat(), "v": p[statistic]} for p in pts[-15:]],
    }


TOOLS = {f.__name__: f for f in (
    get_cluster_health, get_nodes, get_thread_pools, get_unassigned_shards,
    get_pending_tasks, get_allocation_explain, get_cloudwatch_metric,
)}


def lambda_handler(event, context):
    full = ""
    try:
        full = context.client_context.custom.get("bedrockAgentCoreToolName", "")
    except AttributeError:
        full = (event or {}).pop("__tool_name", "")  # local testing
    name = full.split("___", 1)[-1]
    fn = TOOLS.get(name)
    if fn is None:
        return {"error": f"unknown tool '{full}'"}
    try:
        result = fn(**(event or {}))
    except (TypeError, ValueError) as exc:
        return {"error": str(exc)}
    except Exception as exc:  # noqa: BLE001
        return {"error": f"{type(exc).__name__}: {exc}"[:500]}
    text = json.dumps(result, default=str)
    if len(text) > MAX_CHARS:
        text = text[:MAX_CHARS] + '..."(truncated)"'
        return {"result_truncated": text}
    return result


# ---- Tool schemas registered on the Gateway target (inlinePayload) ----------
_DOMAIN = {"domain": {"type": "string", "description": "OpenSearch domain name from the alert"}}
TOOL_SCHEMAS = [
    {"name": "get_cluster_health", "description": "Cluster health: status, node counts, unassigned/relocating shards.",
     "inputSchema": {"type": "object", "properties": _DOMAIN, "required": ["domain"]}},
    {"name": "get_nodes", "description": "Per-node heap, RAM, CPU, load and disk usage, highest heap first.",
     "inputSchema": {"type": "object", "properties": _DOMAIN, "required": ["domain"]}},
    {"name": "get_thread_pools", "description": "Write/search thread pools with non-zero queue or rejections.",
     "inputSchema": {"type": "object", "properties": _DOMAIN, "required": ["domain"]}},
    {"name": "get_unassigned_shards", "description": "Count and sample of UNASSIGNED shards with reasons.",
     "inputSchema": {"type": "object", "properties": _DOMAIN, "required": ["domain"]}},
    {"name": "get_pending_tasks", "description": "Pending cluster-state tasks on the master.",
     "inputSchema": {"type": "object", "properties": _DOMAIN, "required": ["domain"]}},
    {"name": "get_allocation_explain", "description": "Why the first unassigned shard cannot be allocated.",
     "inputSchema": {"type": "object", "properties": _DOMAIN, "required": ["domain"]}},
    {"name": "get_cloudwatch_metric",
     "description": "AWS/ES CloudWatch metric summary (min/max/last and recent points) for a domain.",
     "inputSchema": {"type": "object", "properties": {
         **_DOMAIN,
         "metric": {"type": "string", "description": "e.g. JVMMemoryPressure, FreeStorageSpace, CPUUtilization"},
         "minutes": {"type": "integer", "description": "look-back window, 5-1440 (default 60)"},
         "statistic": {"type": "string", "description": "Maximum | Minimum | Average | Sum"}},
         "required": ["domain", "metric"]}},
]
