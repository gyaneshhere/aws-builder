"""Evidence-backed incident report helpers."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any


@dataclass
class Evidence:
    source: str
    signal: str
    value: Any
    interpretation: str = ""


@dataclass
class Finding:
    hypothesis: str
    confidence: str
    rationale: str
    evidence: list[Evidence] = field(default_factory=list)


@dataclass
class IncidentReport:
    incident_id: str
    resource: str
    time_window: str
    severity: str
    finding: Finding
    recommendations: list[str]
    changes_performed: list[str] = field(default_factory=list)
    generated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, default=str)

    def to_markdown(self) -> str:
        e = self.finding.evidence
        evidence = "\n".join(f"- **{x.source} / {x.signal}:** `{x.value}` — {x.interpretation}" for x in e)
        recommendations = "\n".join(f"{i}. {item}" for i, item in enumerate(self.recommendations, 1))
        changes = ", ".join(self.changes_performed) if self.changes_performed else "None — read-only investigation"
        status = "No production changes were performed." if not self.changes_performed else "Production changes are listed above."
        return f"""# Incident Investigation\n\n**Incident:** {self.incident_id}\n**Resource:** {self.resource}\n**Time window:** {self.time_window}\n**Severity:** {self.severity}\n\n## Finding\n\n**Hypothesis:** {self.finding.hypothesis}\n\n**Confidence:** {self.finding.confidence}\n\n{self.finding.rationale}\n\n## Evidence\n\n{evidence or '- No evidence recorded.'}\n\n## Recommended actions\n\n{recommendations or '- No recommendation.'}\n\n## Production changes\n\n{changes}\n\n{status}\n"""
