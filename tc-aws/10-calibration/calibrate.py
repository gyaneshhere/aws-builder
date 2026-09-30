"""Temperature scaling + a human-review threshold, from saved logits (numpy/scipy only).

Fits one temperature T on the VALIDATION logits by minimizing negative log-likelihood, then reports on TEST:
accuracy (unchanged by T), NLL, ECE and Brier before/after, and picks the lowest confidence threshold whose
auto-handled predictions reach a target accuracy on validation. Writes:
  calibration.json           temperature, review_threshold, metrics (example 05's endpoint reads this)
  reliability.png            reliability diagram before/after (if matplotlib is available)

    python calibrate.py --input-dir outputs/<05 job> --out outputs [--target-accuracy 0.97]
    python calibrate.py --tar output.tar.gz --out /opt/ml/processing/output      (SageMaker Processing)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tarfile

import numpy as np
from scipy.optimize import minimize_scalar

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    from tc_metrics import brier_score, expected_calibration_error, softmax      # inside SageMaker
except ImportError:
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "common"))
    from metrics import brier_score, expected_calibration_error, softmax


def nll(logits, y, t):
    p = softmax(logits, t)
    return float(-np.log(np.clip(p[np.arange(len(y)), y], 1e-12, 1)).mean())


def fit_temperature(logits, y) -> float:
    res = minimize_scalar(lambda log_t: nll(logits, y, np.exp(log_t)), bounds=(-3, 3), method="bounded")
    return float(np.exp(res.x))


def summarize(logits, y, t) -> dict:
    p = softmax(logits, t)
    return {"accuracy": round(float((p.argmax(1) == y).mean()), 4), "nll": round(nll(logits, y, t), 4),
            "ece": round(expected_calibration_error(p, y), 4), "brier": round(brier_score(p, y), 4),
            "mean_confidence": round(float(p.max(1).mean()), 4)}


def pick_threshold(probs, y, target) -> tuple[float, dict]:
    """Lowest threshold where predictions at/above it are >= target accurate (on validation)."""
    conf, correct = probs.max(1), probs.argmax(1) == y
    for th in np.round(np.arange(0.0, 1.0, 0.01), 2):
        keep = conf >= th
        if keep.any() and correct[keep].mean() >= target:
            return float(th), {"auto_rate": round(float(keep.mean()), 4),
                               "auto_accuracy": round(float(correct[keep].mean()), 4)}
    return 1.0, {"auto_rate": 0.0, "auto_accuracy": None}


def at_threshold(probs, y, th) -> dict:
    conf, correct = probs.max(1), probs.argmax(1) == y
    keep = conf >= th
    return {"auto_rate": round(float(keep.mean()), 4),
            "auto_accuracy": round(float(correct[keep].mean()), 4) if keep.any() else None,
            "review_rate": round(float(1 - keep.mean()), 4)}


def reliability_plot(logits, y, t, path, bins=15):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed: skipping reliability.png")
        return
    fig, ax = plt.subplots(figsize=(5, 5))
    edges = np.linspace(0, 1, bins + 1)
    for temp, name in ((1.0, "before (T=1)"), (t, f"after (T={t:.2f})")):
        p = softmax(logits, temp)
        conf, correct = p.max(1), p.argmax(1) == y
        xs, ys = [], []
        for lo, hi in zip(edges[:-1], edges[1:]):
            m = (conf > lo) & (conf <= hi)
            if m.sum() >= 5:
                xs.append(conf[m].mean())
                ys.append(correct[m].mean())
        ax.plot(xs, ys, marker="o", label=name)
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="perfect calibration")
    ax.set_xlabel("confidence")
    ax.set_ylabel("observed accuracy")
    ax.set_title("Reliability on test set")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    print(f"wrote {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", help="dir with validation/test _logits.npy and _labels.npy")
    ap.add_argument("--tar", help="output.tar.gz from the training job (extracted into --out)")
    ap.add_argument("--out", default="outputs")
    ap.add_argument("--target-accuracy", type=float, default=0.97)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    src = a.input_dir
    if a.tar:
        with tarfile.open(a.tar) as t:
            t.extractall(a.out, filter="data")
        src = a.out
    vl, vy = np.load(os.path.join(src, "validation_logits.npy")), np.load(os.path.join(src, "validation_labels.npy"))
    tl, ty = np.load(os.path.join(src, "test_logits.npy")), np.load(os.path.join(src, "test_labels.npy"))

    t = fit_temperature(vl, vy)
    th, val_auto = pick_threshold(softmax(vl, t), vy, a.target_accuracy)
    result = {
        "temperature": t,
        "review_threshold": th,
        "target_accuracy": a.target_accuracy,
        "validation_at_threshold": val_auto,
        "test_before": summarize(tl, ty, 1.0),
        "test_after": summarize(tl, ty, t),
        "test_at_threshold": at_threshold(softmax(tl, t), ty, th),
        "n_validation": int(len(vy)), "n_test": int(len(ty)),
    }
    json.dump(result, open(os.path.join(a.out, "calibration.json"), "w"), indent=2)
    print(json.dumps(result, indent=2))
    reliability_plot(tl, ty, t, os.path.join(a.out, "reliability.png"))


if __name__ == "__main__":
    main()
