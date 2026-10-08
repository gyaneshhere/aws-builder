"""Command-line diagnostics intended to be invoked by the agent."""

from __future__ import annotations

import argparse
import json
import os

from .cloudwatch import CloudWatchDiagnostics
from .opensearch import OpenSearchDiagnostics


def main() -> None:
    parser = argparse.ArgumentParser(description="DBOps read-only diagnostics")
    sub = parser.add_subparsers(dest="command", required=True)

    os_cmd = sub.add_parser("opensearch-snapshot")
    os_cmd.add_argument("--endpoint", default=os.getenv("OPENSEARCH_ENDPOINT"))

    cw_cmd = sub.add_parser("cloudwatch-metric")
    cw_cmd.add_argument("--namespace", required=True)
    cw_cmd.add_argument("--metric-name", required=True)
    cw_cmd.add_argument("--dimension", action="append", default=[], help="Name=Value")
    cw_cmd.add_argument("--minutes", type=int, default=30)
    cw_cmd.add_argument("--statistic", default="Average")

    args = parser.parse_args()
    if args.command == "opensearch-snapshot":
        print(json.dumps(OpenSearchDiagnostics(args.endpoint).snapshot(), indent=2, default=str))
        return

    dimensions = []
    for item in args.dimension:
        if "=" not in item:
            raise SystemExit(f"Invalid dimension: {item}; expected Name=Value")
        name, value = item.split("=", 1)
        dimensions.append({"Name": name, "Value": value})
    result = CloudWatchDiagnostics().metric(
        args.namespace,
        args.metric_name,
        dimensions,
        minutes=args.minutes,
        statistic=args.statistic,
    )
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
