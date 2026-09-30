"""Zero-/few-shot classification with Amazon Bedrock (Converse API + forced tool use).

Instead of asking for JSON in free text and hoping it parses, the request defines a tool whose input schema
has an enum of the allowed labels and forces the model to call it. The label can only be one of the enum
values; confidence is the model's self-report (not a calibrated probability; example 10 measures how far off).
"""

from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor

SYSTEM = ("You are a precise text classifier. Read the text and call the classify tool exactly once. "
          "confidence is your probability (0 to 1) that the label is correct.")


def tool_config(labels: list[str], descriptions: dict[str, str] | None = None) -> dict:
    desc = "; ".join(f"{l}: {descriptions[l]}" for l in labels) if descriptions else ", ".join(labels)
    return {
        "tools": [{"toolSpec": {
            "name": "classify",
            "description": f"Record the classification. Allowed labels: {desc}",
            "inputSchema": {"json": {
                "type": "object",
                "properties": {"label": {"type": "string", "enum": labels},
                               "confidence": {"type": "number", "minimum": 0, "maximum": 1}},
                "required": ["label", "confidence"]}},
        }}],
        "toolChoice": {"tool": {"name": "classify"}},
    }


class BedrockClassifier:
    def __init__(self, model_id: str, labels: list[str], task: str, region: str = "us-east-1",
                 examples: list[tuple[str, str]] | None = None, client=None, descriptions=None):
        if client is None:
            import boto3
            from botocore.config import Config

            client = boto3.client("bedrock-runtime", region_name=region,
                                  config=Config(retries={"max_attempts": 8, "mode": "adaptive"}))
        self.client, self.model_id, self.labels, self.task = client, model_id, labels, task
        self.tools = tool_config(labels, descriptions)
        self.shots = "".join(f"Text: {t}\nLabel: {l}\n\n" for t, l in (examples or []))
        self.input_tokens = self.output_tokens = 0

    def classify(self, text: str) -> dict:
        examples = "Examples:\n" + self.shots if self.shots else ""
        prompt = f"Task: {self.task}\n\n{examples}Text: {text}"
        resp = self.client.converse(
            modelId=self.model_id, system=[{"text": SYSTEM}],
            messages=[{"role": "user", "content": [{"text": prompt}]}],
            toolConfig=self.tools, inferenceConfig={"maxTokens": 100, "temperature": 0})
        usage = resp.get("usage", {})
        self.input_tokens += usage.get("inputTokens", 0)
        self.output_tokens += usage.get("outputTokens", 0)
        for block in resp["output"]["message"]["content"]:
            if "toolUse" in block:
                out = block["toolUse"]["input"]
                label = out.get("label")
                if label in self.labels:
                    return {"label": label, "confidence": float(out.get("confidence", 0.0))}
        return {"label": None, "confidence": 0.0}          # counted as an error, never guessed

    def classify_many(self, texts: list[str], workers: int = 8) -> list[dict]:
        t0 = time.time()
        with ThreadPoolExecutor(workers) as pool:
            out = list(pool.map(self.classify, texts))
        print(f"{len(texts)} texts in {time.time() - t0:.0f}s; tokens in={self.input_tokens:,} out={self.output_tokens:,}")
        return out


def score(results: list[dict], y: list[int], labels: list[str], price_in: float, price_out: float,
          clf: BedrockClassifier) -> dict:
    """Accuracy/F1 plus calibration of the self-reported confidence, and the run's token cost."""
    import numpy as np

    from common.metrics import classification_metrics

    preds = np.array([labels.index(r["label"]) if r["label"] in labels else -1 for r in results])
    conf = np.array([r["confidence"] for r in results])
    k = len(labels)
    probs = np.full((len(preds), k), 0.0)
    for i, (p, c) in enumerate(zip(preds, conf)):     # spread the remaining mass evenly over other labels
        if p >= 0:
            probs[i] = (1 - c) / (k - 1)
            probs[i, p] = c
        else:
            probs[i] = 1.0 / k
    m = classification_metrics(np.array(y), np.where(preds < 0, (np.array(y) + 1) % k, preds), probs, labels)
    m["invalid_outputs"] = int((preds < 0).sum())
    m["cost_usd"] = round(clf.input_tokens / 1e6 * price_in + clf.output_tokens / 1e6 * price_out, 4)
    m["tokens"] = {"input": clf.input_tokens, "output": clf.output_tokens}
    m["note"] = "ece/brier use the model's self-reported confidence"
    return m
