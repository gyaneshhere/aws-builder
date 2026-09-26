"""Input model for one LT-vs-Prod comparison, plus loaders.

JSON shape (see samples/):
{
  "test_name": "psu-60k-sinewave",            # Memory actor: baselines are kept per test
  "run_id": "lt-2026-09-24-a",
  "build": "blaze 27.0.412", "notes": "free text the reviewers should know",
  "lt":   {"label": "...", "interval_seconds": 60, "warmup_minutes": 5},
  "prod": {"label": "...", "interval_seconds": 60, "warmup_minutes": 0},
  "metrics": [
    {"name": "api_latency_p99_ms", "unit": "ms", "direction": "higher_is_worse",
     "slo": 250, "min_pct": 5, "owner": "game-server", "lt": [..], "prod": [..]}
  ]
}
direction: higher_is_worse | lower_is_worse | two_sided
min_pct:   smallest median change (%) that matters in practice for this metric
"""

from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

DIRECTIONS = {"higher_is_worse", "lower_is_worse", "two_sided"}


@dataclass
class Comparison:
    test_name: str
    run_id: str
    lt: dict[str, Any]
    prod: dict[str, Any]
    metrics: list[dict[str, Any]]
    build: str = ""
    notes: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Comparison":
        for k in ("test_name", "run_id", "metrics"):
            if not d.get(k):
                raise ValueError(f"missing '{k}'")
        names = set()
        for m in d["metrics"]:
            if not m.get("name") or "lt" not in m or "prod" not in m:
                raise ValueError(f"metric entries need name, lt and prod: {str(m)[:80]}")
            if m["name"] in names:
                raise ValueError(f"duplicate metric '{m['name']}'")
            names.add(m["name"])
            m.setdefault("direction", "higher_is_worse")
            if m["direction"] not in DIRECTIONS:
                raise ValueError(f"{m['name']}: direction must be one of {sorted(DIRECTIONS)}")
            m.setdefault("min_pct", 5.0)
        known = {"test_name", "run_id", "lt", "prod", "metrics", "build", "notes"}
        return cls(
            test_name=str(d["test_name"]), run_id=str(d["run_id"]),
            lt=d.get("lt") or {}, prod=d.get("prod") or {}, metrics=d["metrics"],
            build=str(d.get("build", "")), notes=str(d.get("notes", "")),
            extra={k: v for k, v in d.items() if k not in known},
        )

    @property
    def actor_id(self) -> str:
        return re.sub(r"[^A-Za-z0-9_-]", "-", self.test_name)[:80] or "test"

    def kernel_payload(self, baselines: dict[str, dict] | None = None) -> dict[str, Any]:
        return {"lt": self.lt, "prod": self.prod, "metrics": self.metrics, "baselines": baselines or {}}

    def meta(self) -> dict[str, Any]:
        return {"test_name": self.test_name, "run_id": self.run_id, "build": self.build, "notes": self.notes,
                "lt": self.lt.get("label", ""), "prod": self.prod.get("label", "")}

    def metric_meta(self, name: str) -> dict[str, Any]:
        m = next(x for x in self.metrics if x["name"] == name)
        return {k: v for k, v in m.items() if k not in ("lt", "prod")}


def load_json(path: str | Path) -> Comparison:
    return Comparison.from_dict(json.loads(Path(path).read_text()))


def _read_wide_csv(path: Path) -> dict[str, list[float | None]]:
    """timestamp,metricA,metricB,... (Grafana / CloudWatch export shape)."""
    cols: dict[str, list[float | None]] = {}
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            for k, v in row.items():
                if k is None or k.lower() in {"timestamp", "time", "ts"}:
                    continue
                try:
                    cols.setdefault(k, []).append(float(v) if v not in ("", None) else None)
                except ValueError:
                    cols.setdefault(k, []).append(None)
    return cols


def load_csv(lt_csv: str | Path, prod_csv: str | Path, meta_json: str | Path) -> Comparison:
    """meta_json holds everything except the series: test_name, run_id, lt, prod and a
    "metrics" list of {name, unit, direction, slo, min_pct}; series come from the CSVs."""
    meta = json.loads(Path(meta_json).read_text())
    lt, prod = _read_wide_csv(Path(lt_csv)), _read_wide_csv(Path(prod_csv))
    for m in meta["metrics"]:
        if m["name"] not in lt or m["name"] not in prod:
            raise ValueError(f"metric '{m['name']}' missing from one of the CSVs")
        m["lt"], m["prod"] = lt[m["name"]], prod[m["name"]]
    return Comparison.from_dict(meta)
