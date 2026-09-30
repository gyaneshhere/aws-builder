"""Jev-style decision primitives on Amazon Bedrock: choice, noul, score.

Same request/response shape as a typed decision API, built on Bedrock Converse with forced tool use so
every answer is schema-valid. Probabilities are the model's self-report, normalized in code; they are not
calibrated (example 10 shows how to measure and fix that). This mimics the interface, not Jev's model.
"""

from __future__ import annotations

import json
import os

SYSTEM = ("You are a careful decision model. Judge the text and call the provided tool exactly once. "
          "Probabilities must reflect your real uncertainty.")
FULL_DIST_MAX = 12          # up to this many choices: ask for the full distribution; above: top 3


class DecisionError(ValueError):
    pass


def _client():
    import boto3
    from botocore.config import Config

    return boto3.client("bedrock-runtime", region_name=os.environ.get("AWS_REGION", "us-east-1"),
                        config=Config(retries={"max_attempts": 6, "mode": "adaptive"}))


def _call(client, model_id, prompt, schema, name):
    resp = client.converse(
        modelId=model_id, system=[{"text": SYSTEM}],
        messages=[{"role": "user", "content": [{"text": prompt}]}],
        toolConfig={"tools": [{"toolSpec": {"name": name, "description": f"Record the {name} decision.",
                                            "inputSchema": {"json": schema}}}],
                    "toolChoice": {"tool": {"name": name}}},
        inferenceConfig={"maxTokens": 600, "temperature": 0})
    usage = resp.get("usage", {})
    for block in resp["output"]["message"]["content"]:
        if "toolUse" in block:
            return block["toolUse"]["input"], usage
    raise DecisionError("model did not return a tool call")


def _normalize(d: dict[str, float], keys: list[str]) -> dict[str, float]:
    vals = {k: max(0.0, float(d.get(k, 0.0))) for k in keys}
    s = sum(vals.values())
    return {k: (v / s if s > 0 else 1.0 / len(keys)) for k, v in vals.items()}


def choice(text: str, task: str, choices: dict[str, str], *, client=None, model_id=None) -> dict:
    if not choices or len(choices) < 2:
        raise DecisionError("choices needs at least 2 entries: {name: description}")
    client, model_id = client or _client(), model_id or os.environ["BEDROCK_MODEL"]
    names = list(choices)
    listing = "\n".join(f"- {k}: {v}" for k, v in choices.items())
    prompt = f"Task: {task}\n\nChoices:\n{listing}\n\nText:\n{text}"
    if len(names) <= FULL_DIST_MAX:
        schema = {"type": "object", "properties": {"probabilities": {
            "type": "object", "properties": {k: {"type": "number", "minimum": 0, "maximum": 1} for k in names},
            "required": names}}, "required": ["probabilities"]}
        out, usage = _call(client, model_id, prompt + "\n\nGive a probability for every choice (they sum to 1).",
                           schema, "choice")
        probs = _normalize(out.get("probabilities", {}), names)
    else:
        item = {"type": "object", "properties": {"choice": {"type": "string", "enum": names},
                                                 "probability": {"type": "number", "minimum": 0, "maximum": 1}},
                "required": ["choice", "probability"]}
        schema = {"type": "object", "properties": {"top": {"type": "array", "items": item, "minItems": 1,
                                                           "maxItems": 3}}, "required": ["top"]}
        out, usage = _call(client, model_id, prompt + "\n\nReturn your 3 most likely choices with probabilities.",
                           schema, "choice")
        top = {t["choice"]: float(t["probability"]) for t in out.get("top", []) if t.get("choice") in choices}
        if not top:
            raise DecisionError("no valid choice returned")
        mass = min(sum(top.values()), 1.0)
        rest = [k for k in names if k not in top]
        probs = {k: top.get(k, (1 - mass) / max(len(rest), 1)) for k in names}
        probs = _normalize(probs, names)
    best = max(probs, key=probs.get)
    return {"type": "choice", "choice": best, "confidence": round(probs[best], 4),
            "probabilities": {k: round(v, 4) for k, v in probs.items()}, "usage": usage}


def noul(text: str, question: str, *, client=None, model_id=None) -> dict:
    client, model_id = client or _client(), model_id or os.environ["BEDROCK_MODEL"]
    schema = {"type": "object", "properties": {"probability_true": {"type": "number", "minimum": 0, "maximum": 1}},
              "required": ["probability_true"]}
    out, usage = _call(client, model_id, f"Statement to judge: {question}\n\nText:\n{text}", schema, "noul")
    p = min(max(float(out["probability_true"]), 0.0), 1.0)
    return {"type": "noul", "noul": round(p, 4), "usage": usage}


def score(text: str, instructions: str, criteria: list[str], *, client=None, model_id=None) -> dict:
    if len(criteria) < 2:
        raise DecisionError("criteria needs at least 2 ordered levels")
    client, model_id = client or _client(), model_id or os.environ["BEDROCK_MODEL"]
    keys = [str(i) for i in range(len(criteria))]
    levels = "\n".join(f"{i}: {c}" for i, c in enumerate(criteria))
    schema = {"type": "object", "properties": {"probabilities": {
        "type": "object", "properties": {k: {"type": "number", "minimum": 0, "maximum": 1} for k in keys},
        "required": keys}}, "required": ["probabilities"]}
    out, usage = _call(client, model_id, f"{instructions}\n\nLevels:\n{levels}\n\nText:\n{text}\n\n"
                       "Give a probability for every level (they sum to 1).", schema, "score")
    probs = _normalize(out.get("probabilities", {}), keys)
    expected = sum(int(k) * v for k, v in probs.items())
    return {"type": "score", "score": round(expected, 4), "level": int(max(probs, key=probs.get)),
            "probabilities": {k: round(v, 4) for k, v in probs.items()}, "usage": usage}
