"""Example 2: product-review polarity at scale with SageMaker BlazingText (Amazon Polarity, 3.6M reviews).

BlazingText's supervised mode (fastText-style) trains on millions of rows in minutes on one CPU
instance. It is a built-in algorithm: no training script, just a data format and hyperparameters.

    python run.py prepare           # 1M-row training sample by default (MAX_TRAIN=0 for all 3.6M)
    python run.py train
    python run.py evaluate          # Batch Transform over the test sample -> accuracy / F1
    python run.py deploy && python run.py predict "battery died after two days, very disappointed"
    python run.py cleanup
"""

import json
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from common.cli import run  # noqa: E402
from common.config import SETTINGS  # noqa: E402
from common.data import export_splits, load_hf  # noqa: E402
from common.sm import common_kwargs, latest_job  # noqa: E402

EXAMPLE = "02-blazingtext"
MAX_TRAIN = int(os.environ.get("MAX_TRAIN", 1_000_000)) or None
MAX_TEST = int(os.environ.get("MAX_TEST", 50_000)) or None
_PUNCT = re.compile(r"([.,!?;:()\"'/])")


def tokenize(text: str) -> str:
    """BlazingText expects space-separated tokens; keep this identical at training and inference."""
    return " ".join(_PUNCT.sub(r" \1 ", str(text).lower()).split())


def to_blazingtext(df, labels) -> str:
    return "\n".join(f"__label__{labels[y]} {tokenize(t)}" for t, y in zip(df["text"], df["label"])) + "\n"


def to_jsonlines(df) -> str:
    return "\n".join(json.dumps({"source": tokenize(t)}) for t in df["text"]) + "\n"


def prepare():
    splits, labels = load_hf("fancyzhx/amazon_polarity", text_col="content",
                             max_train=MAX_TRAIN, max_test=MAX_TEST, val_from_train=0.02)
    for s in ("train", "validation"):
        export_splits(EXAMPLE, {s: splits[s]}, labels, formatter=lambda df: to_blazingtext(df, labels), ext="txt")
    export_splits(EXAMPLE, {"test": splits["test"]}, labels, formatter=to_jsonlines, ext="jsonl")
    # keep the test labels for scoring the Batch Transform output
    export_splits(EXAMPLE + "-labels", {"test": splits["test"][["label"]]}, labels)


def channels():
    return {s: SETTINGS.s3(EXAMPLE, "data", s) + "/" for s in ("train", "validation")}


def estimator():
    from sagemaker import image_uris
    from sagemaker.estimator import Estimator

    est = Estimator(image_uri=image_uris.retrieve("blazingtext", SETTINGS.region, version="1"),
                    instance_type="ml.c5.4xlarge", instance_count=1, **common_kwargs(EXAMPLE))
    est.set_hyperparameters(mode="supervised", epochs=10, min_epochs=3, early_stopping=True, patience=2,
                            learning_rate=0.05, vector_dim=100, word_ngrams=2, min_count=2)
    return est


def evaluate():
    """Batch Transform over test.jsonl, then score predictions against the stored labels."""
    import io

    import numpy as np
    import pandas as pd
    from sagemaker.estimator import Estimator

    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "common"))
    from metrics import classification_metrics, write_metrics

    est = Estimator.attach(latest_job(EXAMPLE), sagemaker_session=SETTINGS.sm_session())
    out = SETTINGS.s3(EXAMPLE, "batch-output")
    tr = est.transformer(instance_count=1, instance_type="ml.c5.2xlarge", output_path=out, accept="application/jsonlines",
                         strategy="MultiRecord", assemble_with="Line", max_payload=6)
    tr.transform(SETTINGS.s3(EXAMPLE, "data", "test") + "/test.jsonl", content_type="application/jsonlines",
                 split_type="Line", wait=True)

    s3 = SETTINGS.boto().client("s3")
    bucket, key = (out + "/test.jsonl.out")[5:].split("/", 1)
    lines = s3.get_object(Bucket=bucket, Key=key)["Body"].read().decode().splitlines()
    lb, lk = (SETTINGS.s3(EXAMPLE + "-labels", "data", "test") + "/test.csv")[5:].split("/", 1)
    y = pd.read_csv(io.BytesIO(s3.get_object(Bucket=lb, Key=lk)["Body"].read()))["label"].to_numpy()
    labels = ["negative", "positive"]
    preds, conf = [], []
    for ln in lines:
        r = json.loads(ln)
        preds.append(labels.index(r["label"][0].replace("__label__", "")))
        conf.append(r["prob"][0])
    preds = np.array(preds)
    probs = np.column_stack([np.where(preds == 0, conf, 1 - np.array(conf)),
                             np.where(preds == 1, conf, 1 - np.array(conf))])
    os.makedirs("outputs", exist_ok=True)
    write_metrics("outputs/02-blazingtext-metrics.json", classification_metrics(y, preds, probs, labels))


def deploy(endpoint):
    from sagemaker.estimator import Estimator

    est = Estimator.attach(latest_job(EXAMPLE), sagemaker_session=SETTINGS.sm_session())
    est.deploy(initial_instance_count=1, instance_type="ml.m5.large", endpoint_name=endpoint)


def predict(endpoint, text):
    rt = SETTINGS.boto().client("sagemaker-runtime")
    body = json.dumps({"instances": [tokenize(text)], "configuration": {"k": 2}})
    r = rt.invoke_endpoint(EndpointName=endpoint, ContentType="application/json", Body=body)
    print(json.dumps(json.loads(r["Body"].read()), indent=2))


if __name__ == "__main__":
    run(EXAMPLE, prepare=prepare, estimator=estimator, channels=channels, deploy=deploy, predict=predict,
        extra={"evaluate": evaluate})
