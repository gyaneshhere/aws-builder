"""Example 6: emotion classification with decoder LLMs (dair-ai/emotion: 16k train / 2k test, 6 labels).

  6a  Bedrock, no training:     python run.py bedrock-eval [N] [SHOTS]   (default N=500, SHOTS=0)
  6b  Fine-tuned Qwen2.5-0.5B:  python run.py prepare && python run.py train && python run.py results

Same test set for both, so you can weigh "no labels, pay per call" against "train once, cheap per call".
"""

import json
import os
import random
import sys

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common.cli import run  # noqa: E402
from common.config import SETTINGS  # noqa: E402
from common.data import export_splits, load_hf  # noqa: E402
from common.sm import common_kwargs, latest_job, stage_source  # noqa: E402

EXAMPLE = "06-decoder-llm"
SRC_05 = os.path.join(ROOT, "05-modernbert", "src")
SERVE_05 = os.path.join(ROOT, "05-modernbert", "serve")
# Approximate on-demand USD per 1M tokens for the default Bedrock model; check current Bedrock pricing.
PRICE_IN = float(os.environ.get("BEDROCK_PRICE_IN", "1.0"))
PRICE_OUT = float(os.environ.get("BEDROCK_PRICE_OUT", "5.0"))


def load():
    return load_hf("dair-ai/emotion", "split")


def prepare():
    splits, labels = load()
    export_splits(EXAMPLE, splits, labels)


def channels():
    return {s: SETTINGS.s3(EXAMPLE, "data", s) + "/" for s in ("train", "validation", "test")}


def estimator():
    from sagemaker.huggingface import HuggingFace

    return HuggingFace(entry_point="train.py", source_dir=stage_source(SRC_05),
                       transformers_version="4.56.2", pytorch_version="2.8.0", py_version="py312",
                       instance_type="ml.g5.2xlarge", instance_count=1,
                       hyperparameters={"model-name": "Qwen/Qwen2.5-0.5B", "epochs": 3, "train-batch-size": 16,
                                        "learning-rate": 2e-5, "max-length": 128},
                       metric_definitions=[{"Name": "test:accuracy", "Regex": r"test_accuracy=([0-9.]+);"}],
                       **common_kwargs(EXAMPLE))


def bedrock_eval(n="500", shots="0"):
    from bedrock_classifier import BedrockClassifier, score

    from common.metrics import write_metrics

    splits, labels = load()
    test = splits["test"].sample(n=min(int(n), len(splits["test"])), random_state=42)
    examples = []
    if int(shots):
        rng = random.Random(0)
        tr = splits["train"]
        examples = [(r.text, r.label_name) for r in tr.iloc[rng.sample(range(len(tr)), int(shots))].itertuples()]
    clf = BedrockClassifier(SETTINGS.bedrock_model, labels, "Classify the emotion expressed in this text.",
                            region=SETTINGS.region, examples=examples)
    results = clf.classify_many(test["text"].tolist())
    m = score(results, test["label"].tolist(), labels, PRICE_IN, PRICE_OUT, clf)
    m.update({"model": SETTINGS.bedrock_model, "shots": int(shots)})
    os.makedirs("outputs", exist_ok=True)
    write_metrics(f"outputs/bedrock-{shots}shot-metrics.json", m)
    with open(f"outputs/bedrock-{shots}shot-predictions.jsonl", "w") as f:
        for t, y, r in zip(test["text"], test["label_name"], results):
            f.write(json.dumps({"text": t, "gold": y, "pred": r["label"], "confidence": r["confidence"]}) + "\n")


def deploy(endpoint):
    from sagemaker.huggingface import HuggingFaceModel

    job = SETTINGS.boto().client("sagemaker").describe_training_job(TrainingJobName=latest_job(EXAMPLE))
    HuggingFaceModel(model_data=job["ModelArtifacts"]["S3ModelArtifacts"], role=SETTINGS.role(),
                     entry_point="inference.py", source_dir=SERVE_05, transformers_version="4.51.3",
                     pytorch_version="2.6.0", py_version="py312", sagemaker_session=SETTINGS.sm_session()
                     ).deploy(initial_instance_count=1, instance_type="ml.g5.xlarge", endpoint_name=endpoint)


def predict(endpoint, text):
    rt = SETTINGS.boto().client("sagemaker-runtime")
    r = rt.invoke_endpoint(EndpointName=endpoint, ContentType="application/json", Body=json.dumps({"texts": [text]}))
    print(json.dumps(json.loads(r["Body"].read()), indent=2))


if __name__ == "__main__":
    run(EXAMPLE, prepare=prepare, estimator=estimator, channels=channels, deploy=deploy, predict=predict,
        extra={"bedrock-eval": bedrock_eval})
