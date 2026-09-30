"""SageMaker helpers shared by the examples."""

from __future__ import annotations

import json
import os
import shutil
import tarfile
import tempfile

from .config import SETTINGS

HERE = os.path.dirname(os.path.abspath(__file__))


def stage_source(src_dir: str) -> str:
    """Copy an example's src/ plus the shared metrics module into a temp dir used as source_dir.

    Training containers only receive source_dir, so tc_metrics.py travels with the code.
    """
    tmp = tempfile.mkdtemp(prefix="tc-src-")
    shutil.copytree(src_dir, tmp, dirs_exist_ok=True, ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copy(os.path.join(HERE, "metrics.py"), os.path.join(tmp, "tc_metrics.py"))
    return tmp


def common_kwargs(example: str) -> dict:
    """Role, session, output path and tags every estimator uses."""
    return dict(
        role=SETTINGS.role(),
        sagemaker_session=SETTINGS.sm_session(),
        output_path=SETTINGS.s3(example, "jobs"),
        base_job_name=f"tc-{example}"[:30],
        tags=[{"Key": "project", "Value": "text-classification-on-aws"}, {"Key": "example", "Value": example}],
    )


def latest_job(example: str) -> str:
    sm = SETTINGS.boto().client("sagemaker")
    jobs = sm.list_training_jobs(NameContains=f"tc-{example}"[:30], SortBy="CreationTime",
                                 SortOrder="Descending", MaxResults=5)["TrainingJobSummaries"]
    if not jobs:
        raise SystemExit(f"no training jobs found for {example}; run: python run.py train")
    return jobs[0]["TrainingJobName"]


def fetch_output(job_name: str, dest: str) -> dict:
    """Download a job's output.tar.gz (metrics.json, predictions) and return metrics.json."""
    sm = SETTINGS.boto().client("sagemaker")
    desc = sm.describe_training_job(TrainingJobName=job_name)
    if desc["TrainingJobStatus"] != "Completed":
        raise SystemExit(f"{job_name} is {desc['TrainingJobStatus']}: {desc.get('FailureReason', '')}")
    model_uri = desc["ModelArtifacts"]["S3ModelArtifacts"]
    out_uri = model_uri.rsplit("/", 1)[0] + "/output.tar.gz"
    bucket, key = out_uri[5:].split("/", 1)
    os.makedirs(dest, exist_ok=True)
    local = os.path.join(dest, "output.tar.gz")
    SETTINGS.boto().client("s3").download_file(bucket, key, local)
    with tarfile.open(local) as t:
        t.extractall(dest, filter="data")
    print(f"job        {job_name}\nmodel      {model_uri}\noutputs -> {dest}/")
    mpath = os.path.join(dest, "metrics.json")
    return json.load(open(mpath)) if os.path.exists(mpath) else {}


def delete_endpoint(name: str) -> None:
    sm = SETTINGS.boto().client("sagemaker")
    try:
        cfg = sm.describe_endpoint(EndpointName=name)["EndpointConfigName"]
        sm.delete_endpoint(EndpointName=name)
        sm.delete_endpoint_config(EndpointConfigName=cfg)
        print(f"deleted endpoint {name}")
    except sm.exceptions.ClientError as e:
        print(f"skip: {e.response['Error']['Message']}")


def print_metrics(m: dict) -> None:
    print(json.dumps({k: v for k, v in m.items() if k != "confusion_matrix"}, indent=2))
