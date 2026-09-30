"""Endpoint for the scoring-head model with the same request shapes as example 8's Bedrock API.

  {"mode": "choice", "text", "task", "choices": {"label": "description", ...}}
  {"mode": "noul",   "text", "question"}                        -> candidates: statement true / false
  {"mode": "score",  "text", "instructions", "criteria": [...]}  -> softmax over levels, expected value
"""

import json
import os

import torch
from transformers import AutoTokenizer

from scorer_model import Scorer, encode_batch, group_scores


def model_fn(model_dir):
    tok = AutoTokenizer.from_pretrained(model_dir)
    model = Scorer(model_dir)
    model.head.load_state_dict(torch.load(os.path.join(model_dir, "head.pt"), map_location="cpu"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = json.load(open(os.path.join(model_dir, "scorer.json")))
    return {"tok": tok, "model": model.to(device).eval(), "device": device, "max_len": cfg["max_length"]}


def input_fn(body, content_type):
    return json.loads(body)


def _decision(req):
    mode, text = req.get("mode"), str(req["text"])
    if mode == "choice":
        cands = [{"label": k, "description": v} for k, v in req["choices"].items()]
        return {"task": req.get("task", "Pick the best choice."), "text": text, "candidates": cands}
    if mode == "noul":
        q = req["question"]
        return {"task": "Is the statement true for this text?", "text": text,
                "candidates": [{"label": "true", "description": q},
                               {"label": "false", "description": f"not the case that: {q}"}]}
    if mode == "score":
        cands = [{"label": str(i), "description": c} for i, c in enumerate(req["criteria"])]
        return {"task": req.get("instructions", "Score the text."), "text": text, "candidates": cands}
    raise ValueError("mode must be choice, noul or score")


@torch.no_grad()
def predict_fn(req, b):
    reqs = req if isinstance(req, list) else [req]
    decisions = [_decision(r) for r in reqs]
    enc, _ = encode_batch(b["tok"], decisions, b["max_len"], b["device"])
    probs = torch.softmax(group_scores(b["model"](**enc).float(), decisions), -1).cpu()
    out = []
    for r, d, p in zip(reqs, decisions, probs):
        p = p[:len(d["candidates"])].tolist()
        labels = [c["label"] for c in d["candidates"]]
        if r["mode"] == "choice":
            i = max(range(len(p)), key=p.__getitem__)
            out.append({"type": "choice", "choice": labels[i], "confidence": round(p[i], 4),
                        "probabilities": {l: round(x, 4) for l, x in zip(labels, p)}})
        elif r["mode"] == "noul":
            out.append({"type": "noul", "noul": round(p[0], 4)})
        else:
            out.append({"type": "score", "score": round(sum(i * x for i, x in enumerate(p)), 4),
                        "probabilities": {l: round(x, 4) for l, x in zip(labels, p)}})
    return out if isinstance(req, list) else out[0]


def output_fn(pred, accept):
    return json.dumps(pred), "application/json"
