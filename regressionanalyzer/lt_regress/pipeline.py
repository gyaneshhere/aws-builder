"""baseline recall -> exact stats (Code Interpreter) -> Jev + policy -> narrative -> Memory."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any

from .config import Settings
from .decision import decide
from .engine import make_engine
from .jev_client import JevClient
from .models import Comparison
from .narrative import write_narrative

log = logging.getLogger(__name__)

try:
    from opentelemetry import trace
except ImportError:  # pragma: no cover
    trace = None


@dataclass
class RunOptions:
    decide_only: bool = False       # stats + Jev + policy; no narrative LLM, no memory writes
    accept_baseline: bool = False   # human override: store this run as baseline even if not PASS
    use_memory: bool = True
    stats_engine: str | None = None  # override Settings.stats_engine ("local" | "code_interpreter")


class Analyzer:
    def __init__(self, cfg: Settings):
        self.cfg = cfg
        self.memory = None
        if cfg.memory_enabled:
            from .memory import RunMemory

            self.memory = RunMemory(cfg.memory_id, cfg.region)

    def _jev(self) -> JevClient | None:
        try:
            from .credentials import get_jev_api_key

            return JevClient(api_key=get_jev_api_key(self.cfg), url=self.cfg.jev_api_url, model=self.cfg.jev_model)
        except Exception as exc:  # noqa: BLE001 - policy falls back to statistics-only
            log.error("Jev unavailable: %s", exc)
            return None

    def run(self, comp: Comparison, opts: RunOptions | None = None) -> dict[str, Any]:
        opts = opts or RunOptions()
        t0 = time.perf_counter()
        mem = self.memory if opts.use_memory else None

        baselines = mem.latest_baseline(comp) if mem else {}
        engine = make_engine(opts.stats_engine or self.cfg.stats_engine, self.cfg.region, self.cfg.code_interpreter_id)
        try:
            stats = engine.analyze(comp.kernel_payload(baselines))
            decision = decide(comp, stats, self._jev(), self.cfg)
            self._annotate(comp, decision)

            narrative = None
            if not opts.decide_only:
                history = mem.recall_findings(comp, " ".join(m.name for m in decision.by_status("regression", "watch"))
                                              or comp.test_name) if mem else []
                narrative = write_narrative(comp, stats, decision, engine, history, self.cfg)
        finally:
            engine.close()

        baseline_updated = False
        if mem and not opts.decide_only:
            mem.record_run(comp, decision.to_dict(), narrative.markdown if narrative else "")
            if decision.verdict == "pass" or opts.accept_baseline:
                mem.save_baseline(comp, stats)
                baseline_updated = True

        total = decision.jev_cost_usd + (narrative.cost_usd if narrative else 0.0)
        return {
            "run": comp.meta(),
            "verdict": decision.verdict,
            "decision": decision.to_dict(),
            "stats_engine": stats.get("engine"),
            "kernel_version": stats.get("kernel_version"),
            "baseline_used": sorted({b.get("run_id") for b in baselines.values() if b.get("run_id")}),
            "baseline_updated": baseline_updated,
            "narrative": None if narrative is None else {
                "used_llm": narrative.used_llm, "model": self.cfg.narrative_model if narrative.used_llm else None,
                "input_tokens": narrative.input_tokens, "output_tokens": narrative.output_tokens,
                "cost_usd": narrative.cost_usd, "analysis_calls": narrative.analysis_calls,
                "markdown": narrative.markdown,
                "report": narrative.report.model_dump() if narrative.report else None,
            },
            "statistics": stats["metrics"],
            "cost_usd_total": round(total, 6),
            "elapsed_ms": round((time.perf_counter() - t0) * 1000, 1),
        }

    @staticmethod
    def _annotate(comp: Comparison, d) -> None:
        if trace is None:
            return
        span = trace.get_current_span()
        if not span.is_recording():
            return
        span.set_attribute("lt.test", comp.test_name)
        span.set_attribute("lt.run_id", comp.run_id)
        span.set_attribute("lt.verdict", d.verdict)
        span.set_attribute("lt.regressions", len(d.by_status("regression")))
        span.set_attribute("lt.watch", len(d.by_status("watch")))
        span.set_attribute("lt.jev_ok", d.jev_ok)
        span.set_attribute("lt.jev_calls", d.jev_calls)
