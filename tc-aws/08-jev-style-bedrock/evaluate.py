"""Run the three decision primitives on datasets used by the trained examples, so results line up:

  choice  Banking77 test sample, 77 intents      (compare with example 05 ModernBERT)
  noul    IMDb test sample, "The review is positive."   (compare with example 03 LSTM)
  score   SST-5 test sample, 5 ordered levels    (compare with example 07 Flan-T5)

    python evaluate.py [N]        # default N=200 per task; writes outputs/08-*.json
Calls Bedrock directly with the same code the Lambda runs (no deployed API needed).
"""

import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(HERE, "app"))
import decisions  # noqa: E402

from common.config import SETTINGS  # noqa: E402
from common.data import load_banking77, load_hf  # noqa: E402
from common.metrics import classification_metrics, write_metrics  # noqa: E402

N = int(sys.argv[1]) if len(sys.argv) > 1 else 200
MODEL = SETTINGS.bedrock_model
os.environ.setdefault("AWS_REGION", SETTINGS.region)
SST5 = ["very negative", "negative", "neutral", "positive", "very positive"]


def pmap(fn, items, workers=8):
    t0 = time.time()
    with ThreadPoolExecutor(workers) as ex:
        out = list(ex.map(fn, items))
    return out, round(time.time() - t0, 1)


def tokens(results):
    return {"input": sum(r["usage"].get("inputTokens", 0) for r in results),
            "output": sum(r["usage"].get("outputTokens", 0) for r in results)}


def run_choice():
    splits, labels = load_banking77()
    test = splits["test"].sample(n=N, random_state=42)
    choices = {l: l.replace("_", " ") for l in labels}
    res, secs = pmap(lambda t: decisions.choice(t, "Which banking support intent does this customer message express?",
                                                choices, model_id=MODEL), test["text"].tolist())
    probs = np.array([[r["probabilities"][l] for l in labels] for r in res])
    m = classification_metrics(test["label"].to_numpy(), probs.argmax(1), probs)
    return {**m, "seconds": secs, "tokens": tokens(res)}


def run_noul():
    splits, _ = load_hf("stanfordnlp/imdb")
    test = splits["test"].sample(n=N, random_state=42)
    res, secs = pmap(lambda t: decisions.noul(t[:6000], "The movie review is positive.", model_id=MODEL),
                     test["text"].tolist())
    p = np.array([r["noul"] for r in res])
    probs = np.column_stack([1 - p, p])
    m = classification_metrics(test["label"].to_numpy(), (p >= 0.5).astype(int), probs)
    return {**m, "seconds": secs, "tokens": tokens(res)}


def run_score():
    splits, _ = load_hf("SetFit/sst5", label_names=SST5)
    test = splits["test"].sample(n=N, random_state=42)
    res, secs = pmap(lambda t: decisions.score(t, "How positive is this movie review?", SST5, model_id=MODEL),
                     test["text"].tolist())
    y = test["label"].to_numpy()
    probs = np.array([[r["probabilities"][str(i)] for i in range(5)] for r in res])
    expected = np.array([r["score"] for r in res])
    m = classification_metrics(y, probs.argmax(1), probs)
    m.update({"off_by_one_accuracy": round(float((np.abs(probs.argmax(1) - y) <= 1).mean()), 4),
              "mae_expected_score": round(float(np.abs(expected - y).mean()), 4)})
    return {**m, "seconds": secs, "tokens": tokens(res)}


if __name__ == "__main__":
    os.makedirs("outputs", exist_ok=True)
    for name, fn in (("choice-banking77", run_choice), ("noul-imdb", run_noul), ("score-sst5", run_score)):
        print(f"\n== {name} (n={N}, model={MODEL})")
        write_metrics(f"outputs/08-{name}.json", {"model": MODEL, **fn()})
