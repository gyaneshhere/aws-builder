"""Normalize incoming alert payloads into one shape.

Supported sources (auto-detected, or pass source= explicitly):
  * grafana   - Grafana unified-alerting webhook ({"receiver", "alerts": [...]})
  * pagerduty - PagerDuty v3 webhook ({"event": {"event_type": "incident.*", "data": {...}}})
  * generic   - a flat dict using the NormalizedAlert field names
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class NormalizedAlert:
    alert_id: str
    rule: str                      # alert rule name, e.g. "ClusterStatusRed"
    title: str
    domain: str = ""               # OpenSearch domain name
    environment: str = "unknown"   # prod | nonprod | unknown
    status: str = "firing"         # firing | resolved
    metric: str = ""
    value: float | None = None
    threshold: float | None = None
    started_at: str = ""
    description: str = ""
    source: str = "generic"
    labels: dict[str, str] = field(default_factory=dict)

    @property
    def is_prod(self) -> bool:
        return self.environment == "prod"

    def to_state(self) -> dict[str, Any]:
        """Compact JSON state for Jev and the LLM prompt (drops empty fields)."""
        d = asdict(self)
        d.pop("labels", None)
        extra = {k: v for k, v in self.labels.items() if k not in {"alertname", "domain", "env"}}
        if extra:
            d["labels"] = dict(list(extra.items())[:15])
        return {k: v for k, v in d.items() if v not in ("", None, {}, [])}


_PROD = re.compile(r"\b(prod|production|live)\b", re.I)
_NONPROD = re.compile(r"\b(nonprod|non-prod|dev|test|qa|stage|staging|lt|sandbox)\b", re.I)


def _env(*hints: str) -> str:
    text = " ".join(h for h in hints if h)
    if _NONPROD.search(text):
        return "nonprod"
    if _PROD.search(text):
        return "prod"
    return "unknown"


def _num(v: Any) -> float | None:
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _stable_id(*parts: str) -> str:
    return hashlib.sha1("|".join(parts).encode()).hexdigest()[:16]


def _from_grafana(p: dict[str, Any]) -> NormalizedAlert:
    alerts = p.get("alerts") or []
    if not alerts:
        raise ValueError("Grafana payload has no alerts")
    a = next((x for x in alerts if x.get("status") == "firing"), alerts[0])
    labels = {str(k): str(v) for k, v in (a.get("labels") or {}).items()}
    ann = a.get("annotations") or {}
    values = a.get("values") or {}
    first_val = _num(next(iter(values.values()), None)) if values else None
    domain = labels.get("domain") or labels.get("DomainName") or labels.get("domain_name", "")
    rule = labels.get("alertname", "unknown")
    return NormalizedAlert(
        alert_id=a.get("fingerprint") or _stable_id(rule, domain, a.get("startsAt", "")),
        rule=rule,
        title=ann.get("summary") or p.get("title") or rule,
        domain=domain,
        environment=labels.get("env") or _env(domain, labels.get("stack", ""), p.get("title", "")),
        status=a.get("status", "firing"),
        metric=labels.get("metric") or labels.get("__name__", ""),
        value=_num(labels.get("value")) if first_val is None else first_val,
        threshold=_num(labels.get("threshold") or ann.get("threshold")),
        started_at=a.get("startsAt", ""),
        description=ann.get("description", ""),
        source="grafana",
        labels=labels,
    )


def _from_pagerduty(p: dict[str, Any]) -> NormalizedAlert:
    ev = p["event"]
    d = ev.get("data") or {}
    title = d.get("title", "")
    service = (d.get("service") or {}).get("summary", "")
    # Titles from Grafana->PagerDuty usually carry "[FIRING:1] <alertname> ... domain=<x>"
    m_rule = re.search(r"\]\s*([A-Za-z0-9_.-]+)", title)
    m_dom = re.search(r"domain[=: ]+([a-z0-9-]+)", title, re.I)
    status = "resolved" if ev.get("event_type") == "incident.resolved" else "firing"
    return NormalizedAlert(
        alert_id=d.get("id") or _stable_id(title),
        rule=m_rule.group(1) if m_rule else (title.split()[0] if title else "unknown"),
        title=title,
        domain=m_dom.group(1) if m_dom else "",
        environment=_env(title, service),
        status=status,
        started_at=d.get("created_at", ev.get("occurred_at", "")),
        description=f"PagerDuty service={service} urgency={d.get('urgency', '')}",
        source="pagerduty",
        labels={"pd_service": service, "pd_urgency": str(d.get("urgency", ""))},
    )


def _from_generic(p: dict[str, Any]) -> NormalizedAlert:
    rule = str(p.get("rule") or p.get("alertname") or "unknown")
    domain = str(p.get("domain", ""))
    return NormalizedAlert(
        alert_id=str(p.get("alert_id") or _stable_id(rule, domain, str(p.get("started_at", "")))),
        rule=rule,
        title=str(p.get("title") or rule),
        domain=domain,
        environment=str(p.get("environment") or _env(domain)),
        status=str(p.get("status", "firing")),
        metric=str(p.get("metric", "")),
        value=_num(p.get("value")),
        threshold=_num(p.get("threshold")),
        started_at=str(p.get("started_at", "")),
        description=str(p.get("description", "")),
        source="generic",
        labels={str(k): str(v) for k, v in (p.get("labels") or {}).items()},
    )


def normalize(payload: dict[str, Any] | str, source: str | None = None) -> NormalizedAlert:
    if isinstance(payload, str):
        payload = json.loads(payload)
    if source is None:
        if isinstance(payload.get("alerts"), list):
            source = "grafana"
        elif str((payload.get("event") or {}).get("event_type", "")).startswith("incident."):
            source = "pagerduty"
        else:
            source = "generic"
    return {"grafana": _from_grafana, "pagerduty": _from_pagerduty, "generic": _from_generic}[source](payload)
