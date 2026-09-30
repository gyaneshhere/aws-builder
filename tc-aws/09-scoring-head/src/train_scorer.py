"""Jev-like scoring-head model: one encoder + one scalar head scores (text, task, candidate) pairs.

For a decision with K candidates the model encodes K pairs
    first  segment: "Task: <instruction>\\nText: <text>"
    second segment: "<label>: <description>"
takes the [CLS] vector of each, maps it to one number with a shared linear head, and applies a softmax
across the K numbers. Because the head is shared, K and the labels can change at inference time.

Loss: cross-entropy over each decision's candidates (padded candidates are masked out).
Outputs: metrics.json with per-task accuracy / ECE on held-in validation and held-out test tasks.
"""

import argparse
import json
import math
import os
import random
import time
from collections import defaultdict

import numpy as np
import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup


class Scorer(nn.Module):
    def __init__(self, name):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(name)
        self.head = nn.Linear(self.encoder.config.hidden_size, 1)

    def forward(self, **enc):
        h = self.encoder(**enc).last_hidden_state[:, 0]
        return self.head(h).squeeze(-1)


def pair_text(d, c):
    return f"Task: {d['task']}\nText: {d['text']}", f"{c['label']}: {c['description']}"


def encode_batch(tok, decisions, max_len, device):
    firsts, seconds, owner = [], [], []
    for i, d in enumerate(decisions):
        for c in d["candidates"]:
            f, s = pair_text(d, c)
            firsts.append(f)
            seconds.append(s)
            owner.append(i)
    enc = tok(firsts, seconds, truncation="only_first", max_length=max_len, padding=True, return_tensors="pt")
    return {k: v.to(device) for k, v in enc.items()}, owner


def group_scores(flat, decisions):
    """[sum K] -> [B, Kmax] with -inf padding."""
    kmax = max(len(d["candidates"]) for d in decisions)
    out = torch.full((len(decisions), kmax), float("-inf"), device=flat.device, dtype=flat.dtype)
    i = 0
    for b, d in enumerate(decisions):
        k = len(d["candidates"])
        out[b, :k] = flat[i:i + k]
        i += k
    return out


@torch.no_grad()
def evaluate(model, tok, rows, max_len, device, bs=16):
    from tc_metrics import brier_score, expected_calibration_error

    model.eval()
    by_task = defaultdict(lambda: {"y": [], "p": []})
    for i in range(0, len(rows), bs):
        batch = rows[i:i + bs]
        enc, _ = encode_batch(tok, batch, max_len, device)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            scores = group_scores(model(**enc).float(), batch)
        probs = torch.softmax(scores, -1).cpu().numpy()
        for d, p in zip(batch, probs):
            by_task[d["task_name"]]["y"].append(d["answer"])
            by_task[d["task_name"]]["p"].append(p[:len(d["candidates"])])
    out = {}
    for task, v in by_task.items():
        k = max(len(p) for p in v["p"])
        probs = np.array([np.pad(p, (0, k - len(p))) for p in v["p"]])
        y = np.array(v["y"])
        out[task] = {"n": len(y), "accuracy": round(float((probs.argmax(1) == y).mean()), 4),
                     "ece": round(expected_calibration_error(probs, y), 4), "brier": round(brier_score(probs, y), 4),
                     "n_candidates": k}
    model.train()
    return out


def read_jsonl(d, name):
    with open(os.path.join(d, name)) as f:
        return [json.loads(line) for line in f]


def main():
    from tc_metrics import write_metrics

    p = argparse.ArgumentParser()
    p.add_argument("--model-name", default="answerdotai/ModernBERT-base")
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=8, help="decisions per step (each has up to 8 candidates)")
    p.add_argument("--learning-rate", type=float, default=3e-5)
    p.add_argument("--max-length", type=int, default=192)
    p.add_argument("--train", default=os.environ.get("SM_CHANNEL_TRAIN"))
    p.add_argument("--validation", default=os.environ.get("SM_CHANNEL_VALIDATION"))
    p.add_argument("--test", default=os.environ.get("SM_CHANNEL_TEST"))
    p.add_argument("--model-dir", default=os.environ.get("SM_MODEL_DIR", "model"))
    p.add_argument("--output-dir", default=os.environ.get("SM_OUTPUT_DATA_DIR", "output"))
    a = p.parse_args()
    random.seed(42)
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train = read_jsonl(a.train, "train.jsonl")
    val = read_jsonl(a.validation, "validation.jsonl")
    test = read_jsonl(a.test, "test.jsonl")
    print(f"train={len(train)} validation={len(val)} held-out test={len(test)} device={device}")

    tok = AutoTokenizer.from_pretrained(a.model_name)
    model = Scorer(a.model_name).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=a.learning_rate, weight_decay=0.01)
    steps = a.epochs * math.ceil(len(train) / a.batch_size)
    sched = get_linear_schedule_with_warmup(opt, int(0.06 * steps), steps)
    loss_fn = nn.CrossEntropyLoss()

    step, t0 = 0, time.time()
    for epoch in range(a.epochs):
        random.shuffle(train)
        for i in range(0, len(train), a.batch_size):
            batch = train[i:i + a.batch_size]
            enc, _ = encode_batch(tok, batch, a.max_length, device)
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                scores = group_scores(model(**enc).float(), batch)
            loss = loss_fn(scores, torch.tensor([d["answer"] for d in batch], device=device))
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            step += 1
            if step % 200 == 0:
                print(f"epoch={epoch} step={step}/{steps} train_loss={loss.item():.4f}; ({time.time() - t0:.0f}s)")
        v = evaluate(model, tok, val, a.max_length, device)
        mean_acc = float(np.mean([x["accuracy"] for x in v.values()]))
        print(f"epoch={epoch} heldin_mean_accuracy={mean_acc:.4f}; " +
              " ".join(f"{k}={x['accuracy']}" for k, x in v.items()))

    metrics = {"held_in_validation": evaluate(model, tok, val, a.max_length, device),
               "held_out_test": evaluate(model, tok, test, a.max_length, device),
               "config": vars(a) | {"train": None, "validation": None, "test": None}}
    for k, x in metrics["held_out_test"].items():
        print(f"heldout_{k}_accuracy={x['accuracy']};")
    os.makedirs(a.output_dir, exist_ok=True)
    write_metrics(os.path.join(a.output_dir, "metrics.json"), metrics)

    os.makedirs(a.model_dir, exist_ok=True)
    model.encoder.save_pretrained(a.model_dir)
    tok.save_pretrained(a.model_dir)
    torch.save(model.head.state_dict(), os.path.join(a.model_dir, "head.pt"))
    json.dump({"max_length": a.max_length}, open(os.path.join(a.model_dir, "scorer.json"), "w"))


if __name__ == "__main__":
    main()
