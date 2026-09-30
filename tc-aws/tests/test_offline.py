"""Offline tests: no AWS, no GPU, no downloads. Run: python -m pytest -q tests/"""

import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT, os.path.join(ROOT, "06-decoder-llm"), os.path.join(ROOT, "08-jev-style-bedrock", "app"),
                os.path.join(ROOT, "10-calibration")]

from common.metrics import brier_score, classification_metrics, expected_calibration_error, softmax  # noqa: E402


class FakeBedrock:
    def __init__(self, reply):
        self.reply, self.calls = reply, []

    def converse(self, **kw):
        self.calls.append(kw)
        out = self.reply(kw) if callable(self.reply) else self.reply
        return {"output": {"message": {"content": [{"toolUse": {"input": out}}]}},
                "usage": {"inputTokens": 100, "outputTokens": 10}}


# ---- metrics ------------------------------------------------------------------
def test_ece_and_brier_hand_values():
    probs = np.array([[0.9, 0.1], [0.6, 0.4], [0.2, 0.8]])
    y = np.array([0, 1, 1])
    # bins: 0.9 correct (gap .1), 0.6 wrong (gap .6), 0.8 correct (gap .2) -> mean .3
    assert expected_calibration_error(probs, y) == pytest.approx(0.3)
    assert brier_score(probs, y) == pytest.approx((0.02 + 0.72 + 0.08) / 3)


def test_softmax_temperature_flattens():
    z = np.array([[4.0, 0.0]])
    assert softmax(z, 2.0)[0, 0] < softmax(z, 1.0)[0, 0]


# ---- example 1: train + serve on synthetic data ---------------------------------
def test_bow_train_and_inference(tmp_path):
    rng = np.random.default_rng(0)
    words = {0: "vote treaty minister", 1: "goal match league", 2: "shares profit market"}
    for split, n in (("train", 300), ("validation", 60), ("test", 60)):
        d = tmp_path / "data" / split
        d.mkdir(parents=True)
        y = rng.integers(0, 3, n)
        pd.DataFrame({"text": [words[i] + " the" for i in y], "label": y}).to_csv(d / f"{split}.csv", index=False)
        (d / "labels.json").write_text(json.dumps(["World", "Sports", "Business"]))
    src = os.path.join(ROOT, "01-bow-logreg", "src", "train.py")
    env = {**os.environ, "PYTHONPATH": str(tmp_path / "shim")}
    os.makedirs(env["PYTHONPATH"], exist_ok=True)
    open(os.path.join(env["PYTHONPATH"], "tc_metrics.py"), "w").write(open(os.path.join(ROOT, "common", "metrics.py")).read())
    subprocess.run([sys.executable, src, "--train", str(tmp_path / "data/train"), "--validation",
                    str(tmp_path / "data/validation"), "--test", str(tmp_path / "data/test"),
                    "--model-dir", str(tmp_path / "model"), "--output-dir", str(tmp_path / "out")],
                   check=True, env=env, capture_output=True)
    m = json.load(open(tmp_path / "out" / "metrics.json"))
    assert m["test"]["accuracy"] > 0.9
    sys.path.insert(0, os.path.dirname(src))
    import train as t01
    b = t01.model_fn(str(tmp_path / "model"))
    pred = t01.predict_fn(t01.input_fn(json.dumps({"texts": ["league goal"]}), "application/json"), b)
    assert pred[0]["label"] == "Sports"


# ---- data helpers ----------------------------------------------------------------
def test_trec_parser_latin1():
    from common.data import parse_trec

    df = parse_trec(b"NUM:count How many ?\nLOC:city Where is Caf\xe9 X ?\n")
    assert df["label_name"].tolist() == ["NUM", "LOC"] and "Caf" in df["text"][1]


def test_blazingtext_format():
    sys.path.insert(0, os.path.join(ROOT, "02-blazingtext"))
    import importlib.util

    spec = importlib.util.spec_from_file_location("bt", os.path.join(ROOT, "02-blazingtext", "run.py"))
    bt = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bt)
    df = pd.DataFrame({"text": ["Great, LOVED it!"], "label": [1]})
    assert bt.to_blazingtext(df, ["negative", "positive"]) == "__label__positive great , loved it !\n"
    assert json.loads(bt.to_jsonlines(df)) == {"source": "great , loved it !"}


# ---- example 6: Bedrock classifier ---------------------------------------------------
def test_bedrock_classifier_rejects_invalid_label():
    from bedrock_classifier import BedrockClassifier

    replies = {"good": {"label": "joy", "confidence": 0.9}, "odd": {"label": "boredom", "confidence": 0.99}}
    fake = FakeBedrock(lambda kw: replies[kw["messages"][0]["content"][0]["text"].split("Text: ")[-1]])
    clf = BedrockClassifier("m", ["joy", "fear"], "Classify.", client=fake)
    assert clf.classify("good") == {"label": "joy", "confidence": 0.9}
    assert clf.classify("odd")["label"] is None
    assert fake.calls[0]["toolConfig"]["toolChoice"] == {"tool": {"name": "classify"}}


def test_prediction_records_keep_gold_and_pred_separate():
    src = open(os.path.join(ROOT, "06-decoder-llm", "run.py")).read()
    assert '"gold": y, "pred": r["label"]' in src      # regression: prediction once overwrote the gold label


# ---- example 8: decision API -------------------------------------------------------------
def test_decisions_normalize_and_clamp():
    import decisions

    c = decisions.choice("t", "task", {"a": "x", "b": "y"}, client=FakeBedrock({"probabilities": {"a": 3, "b": 1}}),
                         model_id="m")
    assert c["choice"] == "a" and c["probabilities"] == {"a": 0.75, "b": 0.25}
    assert decisions.noul("t", "q", client=FakeBedrock({"probability_true": 1.7}), model_id="m")["noul"] == 1.0
    s = decisions.score("t", "i", ["lo", "mid", "hi"], client=FakeBedrock({"probabilities": {"0": 0, "1": 1, "2": 1}}),
                        model_id="m")
    assert s["score"] == pytest.approx(1.5)


def test_handler_validation_and_routes():
    import handler

    assert handler.lambda_handler({"rawPath": "/choice", "body": json.dumps({"text": ""})}, None)["statusCode"] == 400
    assert handler.lambda_handler({"rawPath": "/other", "body": json.dumps({"text": "x"})}, None)["statusCode"] == 404


# ---- example 10: calibration recovers a known temperature -----------------------------
def test_temperature_recovery():
    import calibrate

    rng = np.random.default_rng(1)
    n, k = 4000, 5
    true_logits = rng.normal(0, 2, (n, k))
    y = np.array([rng.choice(k, p=p) for p in softmax(true_logits)])
    t = calibrate.fit_temperature(true_logits * 2.5, y)      # model is 2.5x overconfident
    assert t == pytest.approx(2.5, rel=0.1)
    before = calibrate.summarize(true_logits * 2.5, y, 1.0)["ece"]
    after = calibrate.summarize(true_logits * 2.5, y, t)["ece"]
    assert after < before
