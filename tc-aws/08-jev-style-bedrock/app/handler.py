"""Lambda behind API Gateway (HTTP API, IAM auth): POST /choice, /noul, /score.

  /choice  {"text", "task", "choices": {"name": "description", ...}}
  /noul    {"text", "question"}
  /score   {"text", "instructions", "criteria": ["level 0 description", "level 1 ...", ...]}
"""

import json
import logging

from decisions import DecisionError, choice, noul, score

log = logging.getLogger()
log.setLevel(logging.INFO)
MAX_TEXT = 20_000


def _resp(code, body):
    return {"statusCode": code, "headers": {"Content-Type": "application/json"}, "body": json.dumps(body)}


def lambda_handler(event, context):
    route = event.get("rawPath", "").rstrip("/").rsplit("/", 1)[-1]
    try:
        body = json.loads(event.get("body") or "{}")
        text = str(body.get("text", ""))
        if not text or len(text) > MAX_TEXT:
            raise DecisionError(f"text is required and at most {MAX_TEXT} characters")
        if route == "choice":
            out = choice(text, str(body.get("task", "Pick the best choice.")), dict(body["choices"]))
        elif route == "noul":
            out = noul(text, str(body["question"]))
        elif route == "score":
            out = score(text, str(body.get("instructions", "Score the text.")), list(body["criteria"]))
        else:
            return _resp(404, {"error": f"unknown route {route}; use /choice, /noul or /score"})
    except (DecisionError, KeyError, TypeError, json.JSONDecodeError) as exc:
        return _resp(400, {"error": str(exc)})
    except Exception:                                   # noqa: BLE001 - never leak internals
        log.exception("decision failed")
        return _resp(502, {"error": "upstream model error"})
    usage = out.pop("usage", {})
    log.info(json.dumps({"route": route, "input_tokens": usage.get("inputTokens"),
                         "output_tokens": usage.get("outputTokens")}))
    return _resp(200, out)
