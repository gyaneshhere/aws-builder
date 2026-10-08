"""Read-only CloudWatch diagnostics."""

from __future__ import annotations

import datetime as dt
from typing import Any

import boto3


class CloudWatchDiagnostics:
    def __init__(self, region: str | None = None):
        self.client = boto3.client("cloudwatch", region_name=region)
        self.logs = boto3.client("logs", region_name=region)

    def metric(
        self,
        namespace: str,
        metric_name: str,
        dimensions: list[dict[str, str]] | None = None,
        minutes: int = 30,
        period: int = 60,
        statistic: str = "Average",
    ) -> dict[str, Any]:
        end = dt.datetime.now(dt.timezone.utc)
        start = end - dt.timedelta(minutes=minutes)
        response = self.client.get_metric_statistics(
            Namespace=namespace,
            MetricName=metric_name,
            Dimensions=dimensions or [],
            StartTime=start,
            EndTime=end,
            Period=period,
            Statistics=[statistic],
        )
        response["Datapoints"] = sorted(
            response.get("Datapoints", []), key=lambda x: x.get("Timestamp", dt.datetime.min.replace(tzinfo=dt.timezone.utc))
        )
        return response

    def recent_logs(self, log_group: str, pattern: str | None = None, minutes: int = 30, limit: int = 100) -> list[dict[str, Any]]:
        end_ms = int(dt.datetime.now(dt.timezone.utc).timestamp() * 1000)
        start_ms = end_ms - minutes * 60 * 1000
        kwargs: dict[str, Any] = {
            "logGroupName": log_group,
            "startTime": start_ms,
            "endTime": end_ms,
            "limit": min(limit, 10000),
        }
        if pattern:
            kwargs["filterPattern"] = pattern
        return self.logs.filter_log_events(**kwargs).get("events", [])
