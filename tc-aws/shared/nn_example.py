"""Shared launcher for the PyTorch LSTM (03) and Text CNN (04) examples."""

from __future__ import annotations

import json
import os

from common.config import SETTINGS
from common.sm import common_kwargs, latest_job, stage_source

NN_SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "nn_src")
METRICS = [{"Name": "validation:accuracy", "Regex": r"validation_accuracy=([0-9.]+);"},
           {"Name": "train:loss", "Regex": r"train_loss=([0-9.]+);"},
           {"Name": "test:accuracy", "Regex": r"test_accuracy=([0-9.]+);"}]


def channels(example):
    return {s: SETTINGS.s3(example, "data", s) + "/" for s in ("train", "validation", "test")}


def estimator(example, instance_type, hyperparameters):
    from sagemaker.pytorch import PyTorch

    return PyTorch(entry_point="train.py", source_dir=stage_source(NN_SRC), framework_version="2.8.0",
                   py_version="py312", instance_type=instance_type, instance_count=1,
                   hyperparameters=hyperparameters, metric_definitions=METRICS, **common_kwargs(example))


def deploy(example, endpoint, instance_type="ml.m5.large"):
    from sagemaker.pytorch import PyTorchModel

    job = SETTINGS.boto().client("sagemaker").describe_training_job(TrainingJobName=latest_job(example))
    # Newest PyTorch *inference* container is 2.6.0; the saved state_dict loads fine across versions.
    PyTorchModel(model_data=job["ModelArtifacts"]["S3ModelArtifacts"], role=SETTINGS.role(),
                 entry_point="train.py", source_dir=stage_source(NN_SRC), framework_version="2.6.0",
                 py_version="py312", sagemaker_session=SETTINGS.sm_session()
                 ).deploy(initial_instance_count=1, instance_type=instance_type, endpoint_name=endpoint)


def predict(endpoint, text):
    rt = SETTINGS.boto().client("sagemaker-runtime")
    r = rt.invoke_endpoint(EndpointName=endpoint, ContentType="application/json",
                           Body=json.dumps({"texts": [text]}))
    print(json.dumps(json.loads(r["Body"].read()), indent=2))
