"""Where the exact math runs.

CodeInterpreterEngine: uploads the input as a file into an AgentCore Code
Interpreter session and executes stats_kernel.py there. The same session stays
open for the narrative step, so Claude can run follow-up analysis on the same
data (variables `INPUT` and `RESULT` are already loaded in the sandbox).

LocalEngine: imports the same kernel in-process (dev/tests, no AWS).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from . import stats_kernel

log = logging.getLogger(__name__)
KERNEL_SRC = Path(stats_kernel.__file__).read_text()
_BEGIN, _END = "<<<RESULT>>>", "<<<END>>>"


class StatsError(RuntimeError):
    pass


def _parse_stream(resp: dict[str, Any]) -> tuple[str, str, bool]:
    """Collect stdout/stderr from an invoke_code_interpreter event stream."""
    out, err, is_error = [], [], False
    for event in resp.get("stream", []):
        result = event.get("result") or {}
        is_error = is_error or bool(result.get("isError"))
        sc = result.get("structuredContent") or {}
        if sc.get("stdout") or sc.get("stderr"):
            out.append(sc.get("stdout", ""))
            err.append(sc.get("stderr", ""))
            continue
        for c in result.get("content", []):
            if c.get("type") == "text":
                out.append(c.get("text", ""))
    return "".join(out), "".join(err), is_error


class LocalEngine:
    name = "local"
    supports_followup = False

    def analyze(self, payload: dict[str, Any]) -> dict[str, Any]:
        res = stats_kernel.analyze(json.loads(json.dumps(payload)))
        res["engine"] = self.name
        return res

    def run_python(self, code: str) -> str:
        raise StatsError("follow-up analysis needs the Code Interpreter engine")

    def close(self) -> None:
        pass


class CodeInterpreterEngine:
    name = "agentcore_code_interpreter"
    supports_followup = True

    def __init__(self, region: str, identifier: str = "aws.codeinterpreter.v1"):
        from bedrock_agentcore.tools.code_interpreter_client import CodeInterpreter

        self.ci = CodeInterpreter(region)
        self.ci.start(identifier=identifier, session_timeout_seconds=900)
        self.session_id = self.ci.session_id

    def _exec(self, code: str, clear: bool = False) -> tuple[str, str]:
        out, err, is_error = _parse_stream(self.ci.execute_code(code, clear_context=clear))
        if is_error:
            raise StatsError(f"sandbox error: {(err or out)[-800:]}")
        return out, err

    def analyze(self, payload: dict[str, Any]) -> dict[str, Any]:
        self.ci.upload_file("input.json", json.dumps(payload))
        code = (
            KERNEL_SRC
            + "\nimport json as _json\n"
            + "INPUT = _json.load(open('input.json'))\n"
            + "RESULT = analyze(INPUT)\n"
            + f"print('{_BEGIN}' + _json.dumps(RESULT) + '{_END}')\n"
        )
        out, _ = self._exec(code, clear=True)
        if _BEGIN not in out:
            raise StatsError(f"kernel produced no result: {out[-500:]}")
        res = json.loads(out.split(_BEGIN, 1)[1].split(_END, 1)[0])
        res["engine"] = self.name
        res["code_interpreter_session"] = self.session_id
        return res

    def run_python(self, code: str) -> str:
        out, err = self._exec(code)
        text = out + (f"\n[stderr]\n{err}" if err else "")
        return text[-6000:] if len(text) > 6000 else text

    def close(self) -> None:
        try:
            self.ci.stop()
        except Exception:  # noqa: BLE001
            pass


def make_engine(kind: str, region: str, identifier: str):
    if kind == "local":
        return LocalEngine()
    return CodeInterpreterEngine(region, identifier)
