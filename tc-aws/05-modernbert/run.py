"""Example 5: banking support intent detection with ModernBERT (Banking77: 77 intents, 10k train / 3k test).

Fine-tunes answerdotai/ModernBERT-base on one ml.g5.2xlarge. Saves validation/test logits so example 10
can calibrate this exact model, and deploys with a handler that applies that calibration.
"""

import json
import os
import sys

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
from common.cli import run  # noqa: E402
from common.config import SETTINGS  # noqa: E402
from common.data import export_splits, load_banking77  # noqa: E402
from common.sm import common_kwargs, latest_job, stage_source  # noqa: E402

EXAMPLE = "05-modernbert"
HERE = os.path.dirname(os.path.abspath(__file__))
METRICS = [{"Name": "eval:accuracy", "Regex": r"'eval_accuracy': ([0-9.]+)"},
           {"Name": "validation:accuracy", "Regex": r"validation_accuracy=([0-9.]+);"},
           {"Name": "test:accuracy", "Regex": r"test_accuracy=([0-9.]+);"},
           {"Name": "test:ece", "Regex": r"test_ece=([0-9.]+);"}]


def prepare():
    splits, labels = load_banking77()
    export_splits(EXAMPLE, splits, labels)


def channels():
    return {s: SETTINGS.s3(EXAMPLE, "data", s) + "/" for s in ("train", "validation", "test")}


def estimator():
    from sagemaker.huggingface import HuggingFace

    return HuggingFace(entry_point="train.py", source_dir=stage_source(os.path.join(HERE, "src")),
                       transformers_version="4.56.2", pytorch_version="2.8.0", py_version="py312",
                       instance_type="ml.g5.2xlarge", instance_count=1,
                       hyperparameters={"model-name": "answerdotai/ModernBERT-base", "epochs": 5,
                                        "train-batch-size": 32, "learning-rate": 5e-5, "max-length": 64},
                       metric_definitions=METRICS, **common_kwargs(EXAMPLE))


def calibration_env() -> dict:
    """Use example 10's temperature and threshold if it has been run."""
    path = os.path.join(ROOT, "10-calibration", "outputs", "calibration.json")
    if not os.path.exists(path):
        print("no calibration.json yet (run example 10): deploying uncalibrated")
        return {"TEMPERATURE": "1.0", "REVIEW_THRESHOLD": "0.0"}
    c = json.load(open(path))
    print(f"calibrated deploy: T={c['temperature']:.3f}, review below {c['review_threshold']:.2f}")
    return {"TEMPERATURE": str(c["temperature"]), "REVIEW_THRESHOLD": str(c["review_threshold"])}


def deploy(endpoint):
    from sagemaker.huggingface import HuggingFaceModel

    job = SETTINGS.boto().client("sagemaker").describe_training_job(TrainingJobName=latest_job(EXAMPLE))
    HuggingFaceModel(model_data=job["ModelArtifacts"]["S3ModelArtifacts"], role=SETTINGS.role(),
                     entry_point="inference.py", source_dir=os.path.join(HERE, "serve"),
                     transformers_version="4.51.3", pytorch_version="2.6.0", py_version="py312",
                     env=calibration_env(), sagemaker_session=SETTINGS.sm_session()
                     ).deploy(initial_instance_count=1, instance_type="ml.g5.xlarge", endpoint_name=endpoint)


def predict(endpoint, text):
    rt = SETTINGS.boto().client("sagemaker-runtime")
    r = rt.invoke_endpoint(EndpointName=endpoint, ContentType="application/json", Body=json.dumps({"texts": [text]}))
    print(json.dumps(json.loads(r["Body"].read()), indent=2))


if __name__ == "__main__":
    run(EXAMPLE, prepare=prepare, estimator=estimator, channels=channels, deploy=deploy, predict=predict)
