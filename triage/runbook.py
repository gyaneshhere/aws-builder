"""Runbook catalog + tier definitions."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from pathlib import Path
from typing import Any

import yaml

DEFAULT_PATH = Path(__file__).with_name("runbook.yaml")
UNKNOWN_RULE = "unknown"


class Tier(IntEnum):
    L1 = 1
    L2 = 2
    L3 = 3

    @classmethod
    def parse(cls, s: str) -> "Tier":
        return cls[s.strip().upper()]

    def bump(self) -> "Tier":
        return Tier(min(self + 1, Tier.L3))


@dataclass(frozen=True)
class RunbookRule:
    id: str
    stack: str
    metric: str
    summary: str
    min_tier_prod: Tier
    min_tier_nonprod: Tier
    l1_steps: list[str] = field(default_factory=list)
    escalation: str = ""

    def min_tier(self, is_prod: bool) -> Tier:
        return self.min_tier_prod if is_prod else self.min_tier_nonprod

    def as_prompt(self) -> str:
        steps = "\n".join(f"  - {s}" for s in self.l1_steps)
        return (
            f"Rule: {self.id} (stack: {self.stack}, metric: {self.metric})\n"
            f"Meaning: {self.summary}\nDocumented first response:\n{steps}\n"
            f"Escalation: {self.escalation}"
        )


class Runbook:
    def __init__(self, rules: list[RunbookRule]):
        self.rules = {r.id.lower(): r for r in rules}
        self._by_metric = {r.metric.lower(): r for r in rules if r.metric}

    @classmethod
    def load(cls, path: str | Path = DEFAULT_PATH) -> "Runbook":
        data: dict[str, Any] = yaml.safe_load(Path(path).read_text())
        rules = [
            RunbookRule(
                id=r["id"],
                stack=r.get("stack", ""),
                metric=r.get("metric", ""),
                summary=r["summary"],
                min_tier_prod=Tier.parse(r.get("min_tier_prod", "L1")),
                min_tier_nonprod=Tier.parse(r.get("min_tier_nonprod", "L1")),
                l1_steps=list(r.get("l1_steps", [])),
                escalation=r.get("escalation", ""),
            )
            for r in data["rules"]
        ]
        if len(rules) > 19:
            # Jev choice questions stay well-behaved with <=20 options (19 rules + unknown).
            raise ValueError(f"runbook has {len(rules)} rules; split the catalog or raise the cap deliberately")
        return cls(rules)

    def get(self, rule_id: str) -> RunbookRule | None:
        return self.rules.get(rule_id.lower())

    def exact_match(self, rule: str, metric: str = "") -> RunbookRule | None:
        return self.get(rule) or (self._by_metric.get(metric.lower()) if metric else None)

    def jev_criteria(self) -> dict[str, str]:
        crit = {r.id: r.summary for r in self.rules.values()}
        crit[UNKNOWN_RULE] = "None of the listed rules describes this alert."
        return crit
