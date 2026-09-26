"""Thin client for the Jev Decision API (POST /api/v1/decide).

Adapted from kevinlupera/jev-strands-router (MIT). Additions: typed accessors for
choice confidence/probabilities, wall-clock latency, and an injectable transport
so the policy layer can be unit-tested without network access.

Request:  {"model", "state", "questions": {id: {type, instructions, criteria}}}
  choice: criteria = {key: description}      -> {"choice", "confidence", "probabilities"}
  score:  criteria = [level0, level1, ...]   -> {"score", "confidence", "probabilities"}
  noul:   criteria = {"true": .., "false": ..} (optional) -> {"noul": P(true)}
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import httpx


class JevError(RuntimeError):
    """Any failure talking to Jev or reading its answers."""


@dataclass
class JevResult:
    model: str
    answers: dict[str, Any]
    input_tokens: int = 0
    cost_usd: float = 0.0
    latency_ms: float = 0.0

    def _get(self, name: str, kind: str) -> dict[str, Any]:
        ans = self.answers.get(name)
        if not isinstance(ans, dict):
            raise JevError(f"missing answer '{name}'")
        if ans.get("type", kind) != kind:
            raise JevError(f"answer '{name}' is {ans.get('type')}, expected {kind}")
        return ans

    def choice(self, name: str) -> str:
        return str(self._get(name, "choice")["choice"])

    def choice_confidence(self, name: str) -> float:
        ans = self._get(name, "choice")
        if "confidence" in ans:
            return float(ans["confidence"])
        probs = ans.get("probabilities") or {}
        return float(probs.get(ans["choice"], 0.0))

    def probabilities(self, name: str) -> dict[str, float]:
        ans = self.answers.get(name) or {}
        return {str(k): float(v) for k, v in (ans.get("probabilities") or {}).items()}

    def score_confidence(self, name: str) -> float | None:
        ans = self._get(name, "score")
        return float(ans["confidence"]) if "confidence" in ans else None

    def score(self, name: str) -> float:
        return float(self._get(name, "score")["score"])

    def noul(self, name: str) -> float:
        return float(self._get(name, "noul")["noul"])


@dataclass
class JevClient:
    api_key: str
    url: str = "https://jevtypesafeai.com/api/v1/decide"
    model: str = "jev-latest"
    timeout: float = 10.0
    _http: httpx.Client | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if not self.api_key:
            raise JevError("Jev API key is empty")
        self._http = self._http or httpx.Client(timeout=self.timeout)

    def decide(self, state: Any, questions: dict[str, Any]) -> JevResult:
        payload = {"model": self.model, "state": state, "questions": questions}
        t0 = time.perf_counter()
        try:
            resp = self._http.post(
                self.url,
                json=payload,
                headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
            )
        except httpx.HTTPError as exc:
            raise JevError(f"Jev request failed: {exc}") from exc
        latency_ms = (time.perf_counter() - t0) * 1000

        if resp.status_code >= 400:
            try:
                body = resp.json()
                detail = body.get("error") or body.get("code") or body
            except Exception:  # noqa: BLE001
                detail = resp.text[:300]
            raise JevError(f"Jev API {resp.status_code}: {detail}")

        data = resp.json()
        usage = data.get("usage") or {}
        return JevResult(
            model=data.get("model", self.model),
            answers=data.get("answers") or {},
            input_tokens=int(usage.get("input_tokens", 0)),
            cost_usd=float(usage.get("cost_usd", 0.0)),
            latency_ms=latency_ms,
        )

    def close(self) -> None:
        if self._http:
            self._http.close()
