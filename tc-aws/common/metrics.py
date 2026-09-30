"""Metrics shared by every example: accuracy, macro-F1, calibration (ECE, Brier), confusion matrix.

Pure numpy + scikit-learn so the same code runs in training containers and on a laptop.
"""

from __future__ import annotations

import json

import numpy as np


def softmax(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    z = logits / temperature
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def expected_calibration_error(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:
    """Top-label ECE: weighted gap between confidence and accuracy across confidence bins."""
    conf = probs.max(axis=1)
    correct = (probs.argmax(axis=1) == labels).astype(float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            ece += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(ece)


def brier_score(probs: np.ndarray, labels: np.ndarray) -> float:
    """Multi-class Brier score: mean squared error between probabilities and one-hot labels."""
    onehot = np.zeros_like(probs)
    onehot[np.arange(len(labels)), labels] = 1.0
    return float(((probs - onehot) ** 2).sum(axis=1).mean())


def classification_metrics(labels, preds, probs: np.ndarray | None = None,
                           label_names: list[str] | None = None) -> dict:
    from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

    labels, preds = np.asarray(labels), np.asarray(preds)
    out = {
        "n": int(len(labels)),
        "accuracy": round(float(accuracy_score(labels, preds)), 4),
        "macro_f1": round(float(f1_score(labels, preds, average="macro")), 4),
    }
    if probs is not None:
        out["ece"] = round(expected_calibration_error(probs, labels), 4)
        out["brier"] = round(brier_score(probs, labels), 4)
        out["mean_confidence"] = round(float(probs.max(axis=1).mean()), 4)
    if label_names is not None and len(label_names) <= 20:
        out["confusion_matrix"] = {"labels": label_names,
                                   "matrix": confusion_matrix(labels, preds,
                                                              labels=list(range(len(label_names)))).tolist()}
    return out


def write_metrics(path: str, metrics: dict) -> None:
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps({k: v for k, v in metrics.items() if k != "confusion_matrix"}, indent=2))
