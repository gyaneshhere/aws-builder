"""Example 9: a Jev-like scoring-head decision model trained on SageMaker.

Trains ModernBERT + one scalar head on 4 decision tasks (AG News, Banking77, emotion, TREC) and evaluates on
2 tasks it never saw (Rotten Tomatoes, SST-5). Serves choice / noul / score with example 8's request shapes.
"""

import json
import os
import sys

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
from common.cli import run  # noqa: E402
from common.config import SETTINGS  # noqa: E402
from common.sm import common_kwargs, latest_job, stage_source  # noqa: E402

EXAMPLE = "09-scoring-head"
METRICS = [{"Name": "heldin:mean_accuracy", "Regex": r"heldin_mean_accuracy=([0-9.]+);"},
           {"Name": "heldout:rotten_tomatoes_accuracy", "Regex": r"heldout_rotten_tomatoes_accuracy=([0-9.]+);"},
           {"Name": "heldout:sst5_accuracy", "Regex": r"heldout_sst5_accuracy=([0-9.]+);"},
           {"Name": "train:loss", "Regex": r"train_loss=([0-9.]+);"}]


def prepare():
    from build_tasks import build

    s3 = SETTINGS.boto().client("s3")
    bucket = SETTINGS.resolved_bucket()
    for split, rows in build().items():
        key = f"{SETTINGS.prefix}/{EXAMPLE}/data/{split}/{split}.jsonl"
        s3.put_object(Bucket=bucket, Key=key, Body="\n".join(json.dumps(r) for r in rows).encode())
        tasks = sorted({r["task_name"] for r in rows})
        print(f"  {split:10s} {len(rows):>7,} decisions  tasks={tasks}  -> s3://{bucket}/{key}")


def channels():
    return {s: SETTINGS.s3(EXAMPLE, "data", s) + "/" for s in ("train", "validation", "test")}


def estimator():
    from sagemaker.huggingface import HuggingFace

    return HuggingFace(entry_point="train_scorer.py", source_dir=stage_source(os.path.join(HERE, "src")),
                       transformers_version="4.56.2", pytorch_version="2.8.0", py_version="py312",
                       instance_type="ml.g5.2xlarge", instance_count=1,
                       hyperparameters={"model-name": "answerdotai/ModernBERT-base", "epochs": 2,
                                        "batch-size": 8, "learning-rate": 3e-5, "max-length": 192},
                       metric_definitions=METRICS, **common_kwargs(EXAMPLE))


def serve_source() -> str:
    """serve/inference.py plus the training module (as scorer_model.py) so the model code cannot drift."""
    import shutil
    import tempfile

    tmp = tempfile.mkdtemp(prefix="tc-serve-")
    shutil.copy(os.path.join(HERE, "serve", "inference.py"), tmp)
    shutil.copy(os.path.join(HERE, "src", "train_scorer.py"), os.path.join(tmp, "scorer_model.py"))
    return tmp


def deploy(endpoint):
    from sagemaker.huggingface import HuggingFaceModel

    job = SETTINGS.boto().client("sagemaker").describe_training_job(TrainingJobName=latest_job(EXAMPLE))
    HuggingFaceModel(model_data=job["ModelArtifacts"]["S3ModelArtifacts"], role=SETTINGS.role(),
                     entry_point="inference.py", source_dir=serve_source(),
                     transformers_version="4.51.3", pytorch_version="2.6.0", py_version="py312",
                     sagemaker_session=SETTINGS.sm_session()
                     ).deploy(initial_instance_count=1, instance_type="ml.g5.xlarge", endpoint_name=endpoint)


def predict(endpoint, text):
    rt = SETTINGS.boto().client("sagemaker-runtime")
    req = {"mode": "choice", "text": text, "task": "Route this support ticket.",
           "choices": {"billing": "payment, refund or charge problem", "technical": "bug, outage or malfunction",
                       "account": "login, identity or profile problem"}}
    r = rt.invoke_endpoint(EndpointName=endpoint, ContentType="application/json", Body=json.dumps(req))
    print(json.dumps(json.loads(r["Body"].read()), indent=2))


if __name__ == "__main__":
    run(EXAMPLE, prepare=prepare, estimator=estimator, channels=channels, deploy=deploy, predict=predict)
