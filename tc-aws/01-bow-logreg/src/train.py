"""TF-IDF + logistic regression (SageMaker scikit-learn container).

Training: reads train/validation/test CSVs (text,label,label_name), fits, evaluates, and writes
  /opt/ml/model/model.joblib, labels.json                -> model.tar.gz
  /opt/ml/output/data/metrics.json, test_predictions.csv, top_terms.json  -> output.tar.gz
Serving: model_fn / input_fn / predict_fn / output_fn below accept {"texts": ["...", ...]}.
"""

import argparse
import glob
import json
import os

import joblib
import numpy as np
import pandas as pd


def read_split(channel_dir: str) -> pd.DataFrame:
    files = glob.glob(os.path.join(channel_dir, "*.csv"))
    return pd.concat([pd.read_csv(f) for f in files], ignore_index=True)


def top_terms(model, labels, k=15):
    vec, clf = model.named_steps["tfidf"], model.named_steps["clf"]
    vocab = np.array(vec.get_feature_names_out())
    coefs = clf.coef_ if clf.coef_.shape[0] > 1 else np.vstack([-clf.coef_[0], clf.coef_[0]])
    return {labels[i]: vocab[np.argsort(coefs[i])[-k:][::-1]].tolist() for i in range(len(labels))}


def main():
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline

    from tc_metrics import classification_metrics, write_metrics

    p = argparse.ArgumentParser()
    p.add_argument("--max-features", type=int, default=200_000)
    p.add_argument("--ngram-max", type=int, default=2)
    p.add_argument("--C", type=float, default=4.0)
    p.add_argument("--train", default=os.environ.get("SM_CHANNEL_TRAIN", "data/train"))
    p.add_argument("--validation", default=os.environ.get("SM_CHANNEL_VALIDATION", "data/validation"))
    p.add_argument("--test", default=os.environ.get("SM_CHANNEL_TEST", "data/test"))
    p.add_argument("--model-dir", default=os.environ.get("SM_MODEL_DIR", "model"))
    p.add_argument("--output-dir", default=os.environ.get("SM_OUTPUT_DATA_DIR", "output"))
    a = p.parse_args()
    os.makedirs(a.model_dir, exist_ok=True)
    os.makedirs(a.output_dir, exist_ok=True)

    train, val, test = read_split(a.train), read_split(a.validation), read_split(a.test)
    labels = json.load(open(os.path.join(a.train, "labels.json")))
    print(f"train={len(train)} validation={len(val)} test={len(test)} classes={len(labels)}")

    model = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=a.max_features, ngram_range=(1, a.ngram_max),
                                  min_df=2, sublinear_tf=True)),
        ("clf", LogisticRegression(C=a.C, max_iter=2000)),
    ])
    model.fit(train["text"], train["label"])

    metrics = {}
    for name, df in (("validation", val), ("test", test)):
        probs = model.predict_proba(df["text"])
        preds = probs.argmax(axis=1)
        metrics[name] = classification_metrics(df["label"], preds, probs, labels)
        if name == "test":
            out = df[["text", "label"]].copy()
            out["pred"], out["confidence"] = preds, probs.max(axis=1)
            out.to_csv(os.path.join(a.output_dir, "test_predictions.csv"), index=False)
    metrics["hyperparameters"] = {"max_features": a.max_features, "ngram_max": a.ngram_max, "C": a.C}
    # SageMaker picks these up from the log as training-job metrics (see run.py metric_definitions)
    print(f"validation_accuracy={metrics['validation']['accuracy']};")
    print(f"test_accuracy={metrics['test']['accuracy']}; test_macro_f1={metrics['test']['macro_f1']};")
    write_metrics(os.path.join(a.output_dir, "metrics.json"), metrics)
    json.dump(top_terms(model, labels), open(os.path.join(a.output_dir, "top_terms.json"), "w"), indent=1)

    joblib.dump(model, os.path.join(a.model_dir, "model.joblib"))
    json.dump(labels, open(os.path.join(a.model_dir, "labels.json"), "w"))


# ---------------- SageMaker inference handlers ----------------
def model_fn(model_dir):
    return {"model": joblib.load(os.path.join(model_dir, "model.joblib")),
            "labels": json.load(open(os.path.join(model_dir, "labels.json")))}


def input_fn(body, content_type):
    if content_type != "application/json":
        raise ValueError(f"unsupported content type {content_type}")
    data = json.loads(body)
    texts = data["texts"] if isinstance(data, dict) else data
    return [str(t) for t in texts]


def predict_fn(texts, bundle):
    probs = bundle["model"].predict_proba(texts)
    return [{"label": bundle["labels"][int(p.argmax())], "confidence": round(float(p.max()), 4),
             "probabilities": {bundle["labels"][i]: round(float(x), 4) for i, x in enumerate(p)}}
            for p in probs]


def output_fn(prediction, accept):
    return json.dumps({"predictions": prediction}), "application/json"


if __name__ == "__main__":
    main()
