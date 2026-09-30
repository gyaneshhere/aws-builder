"""Endpoint handler for the fine-tuned encoder, with optional calibration from example 10.

Env vars (set at deploy time):
  TEMPERATURE       divide logits by T before softmax (1.0 = uncalibrated)
  REVIEW_THRESHOLD  confidence below this -> "needs_review": true
Request:  {"texts": ["I still haven't received my card", ...]}
Response: {"predictions": [{"label", "confidence", "needs_review", "top3"}]}
"""

import json
import os

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def model_fn(model_dir):
    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir).eval()
    if torch.cuda.is_available():
        model = model.cuda()
    return {"tok": tok, "model": model, "T": float(os.environ.get("TEMPERATURE", "1.0")),
            "threshold": float(os.environ.get("REVIEW_THRESHOLD", "0.0"))}


def input_fn(body, content_type):
    data = json.loads(body)
    return [str(t) for t in (data["texts"] if isinstance(data, dict) else data)]


@torch.no_grad()
def predict_fn(texts, b):
    enc = b["tok"](texts, truncation=True, max_length=64, padding=True, return_tensors="pt")
    enc = {k: v.to(b["model"].device) for k, v in enc.items()}
    probs = torch.softmax(b["model"](**enc).logits.float() / b["T"], dim=-1).cpu()
    id2label = b["model"].config.id2label
    out = []
    for p in probs:
        top = torch.topk(p, 3)
        conf = float(top.values[0])
        out.append({"label": id2label[int(top.indices[0])], "confidence": round(conf, 4),
                    "needs_review": conf < b["threshold"],
                    "top3": {id2label[int(i)]: round(float(v), 4) for v, i in zip(top.values, top.indices)}})
    return out


def output_fn(pred, accept):
    return json.dumps({"predictions": pred}), "application/json"
