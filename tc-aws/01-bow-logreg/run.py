"""Example 1: news topic routing with TF-IDF + logistic regression (AG News, 4 classes)."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from common.cli import run  # noqa: E402
from common.config import SETTINGS  # noqa: E402
from common.data import export_splits, load_hf  # noqa: E402
from common.sm import common_kwargs, stage_source  # noqa: E402

EXAMPLE = "01-bow-logreg"
# Train and serve on the same scikit-learn: 1.2-1 is the newest SageMaker sklearn *inference* container
# (1.4-2 exists for training only), and pickled pipelines are not guaranteed to load across versions.
SKLEARN_VERSION = "1.2-1"
HERE = os.path.dirname(os.path.abspath(__file__))
METRICS = [{"Name": "validation:accuracy", "Regex": r"validation_accuracy=([0-9.]+);"},
           {"Name": "test:accuracy", "Regex": r"test_accuracy=([0-9.]+);"},
           {"Name": "test:macro_f1", "Regex": r"test_macro_f1=([0-9.]+);"}]


def prepare():
    splits, labels = load_hf("fancyzhx/ag_news")          # 120k train / 7.6k test; validation carved from train
    export_splits(EXAMPLE, splits, labels)


def channels():
    return {s: SETTINGS.s3(EXAMPLE, "data", s) + "/" for s in ("train", "validation", "test")}


def estimator():
    from sagemaker.sklearn.estimator import SKLearn

    return SKLearn(entry_point="train.py", source_dir=stage_source(os.path.join(HERE, "src")),
                   framework_version=SKLEARN_VERSION, py_version="py3", instance_type="ml.m5.2xlarge",
                   instance_count=1, hyperparameters={"max-features": 200_000, "ngram-max": 2, "C": 4.0},
                   metric_definitions=METRICS, **common_kwargs(EXAMPLE))


def deploy(endpoint):
    from sagemaker.sklearn.model import SKLearnModel
    from common.sm import latest_job

    sm = SETTINGS.boto().client("sagemaker")
    job = sm.describe_training_job(TrainingJobName=latest_job(EXAMPLE))
    model = SKLearnModel(model_data=job["ModelArtifacts"]["S3ModelArtifacts"], role=SETTINGS.role(),
                         entry_point="train.py", source_dir=stage_source(os.path.join(HERE, "src")),
                         framework_version=SKLEARN_VERSION, py_version="py3", sagemaker_session=SETTINGS.sm_session())
    return model.deploy(initial_instance_count=1, instance_type="ml.m5.large", endpoint_name=endpoint)


def predict(endpoint, text):
    import json

    rt = SETTINGS.boto().client("sagemaker-runtime")
    r = rt.invoke_endpoint(EndpointName=endpoint, ContentType="application/json",
                           Body=json.dumps({"texts": [text]}))
    print(json.dumps(json.loads(r["Body"].read()), indent=2))


if __name__ == "__main__":
    run(EXAMPLE, prepare=prepare, estimator=estimator, channels=channels, deploy=deploy, predict=predict)
