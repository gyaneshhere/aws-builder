"""Example 10: calibrate ModernBERT (example 05) with temperature scaling and set a human-review threshold.

    python run.py calibrate [JOB]      # fetch 05's logits, fit T, write outputs/calibration.json, upload to S3
    python run.py processing [JOB]     # same thing as a SageMaker Processing job (auditable, repeatable)
    python run.py bedrock FILE.jsonl   # how calibrated is an LLM's self-reported confidence? (example 06 output)
Then redeploy example 05: its endpoint reads calibration.json and applies T and the threshold.
"""

import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
sys.path.insert(0, ROOT)
from common.config import SETTINGS  # noqa: E402
from common.sm import fetch_output, latest_job, stage_source  # noqa: E402

TARGET = os.environ.get("TARGET_ACCURACY", "0.97")


def calibrate(job=None):
    job = job or latest_job("05-modernbert")
    src = os.path.join(HERE, "outputs", job)
    fetch_output(job, src)
    subprocess.run([sys.executable, os.path.join(HERE, "calibrate.py"), "--input-dir", src,
                    "--out", os.path.join(HERE, "outputs"), "--target-accuracy", TARGET], check=True)
    model_uri = SETTINGS.boto().client("sagemaker").describe_training_job(
        TrainingJobName=job)["ModelArtifacts"]["S3ModelArtifacts"]
    bucket, key = model_uri[5:].rsplit("/", 1)[0].split("/", 1)
    SETTINGS.boto().client("s3").upload_file(os.path.join(HERE, "outputs", "calibration.json"), bucket,
                                             f"{key}/calibration.json")
    print(f"uploaded s3://{bucket}/{key}/calibration.json (next to the model). Redeploy example 05 to apply it.")


def processing(job=None):
    from sagemaker.processing import FrameworkProcessor, ProcessingInput, ProcessingOutput
    from sagemaker.sklearn.estimator import SKLearn

    job = job or latest_job("05-modernbert")
    desc = SETTINGS.boto().client("sagemaker").describe_training_job(TrainingJobName=job)
    out_tar = desc["ModelArtifacts"]["S3ModelArtifacts"].rsplit("/", 1)[0] + "/output.tar.gz"
    proc = FrameworkProcessor(estimator_cls=SKLearn, framework_version="1.4-2", role=SETTINGS.role(),
                              instance_type="ml.m5.large", instance_count=1, base_job_name="tc-10-calibration",
                              sagemaker_session=SETTINGS.sm_session())
    dest = SETTINGS.s3("10-calibration", "processing", job)
    proc.run(code="calibrate.py", source_dir=stage_source(HERE),
             inputs=[ProcessingInput(source=out_tar, destination="/opt/ml/processing/input")],
             outputs=[ProcessingOutput(source="/opt/ml/processing/output", destination=dest)],
             arguments=["--tar", "/opt/ml/processing/input/output.tar.gz", "--out", "/opt/ml/processing/output",
                        "--target-accuracy", TARGET])
    print(f"calibration.json -> {dest}/")


def bedrock(path):
    """Reliability of self-reported LLM confidence, from example 06's predictions jsonl (gold, pred, confidence)."""
    import numpy as np

    rows = [json.loads(line) for line in open(path)]
    conf = np.array([float(r["confidence"]) for r in rows])
    correct = np.array([r["pred"] == r["gold"] for r in rows])
    ece, edges = 0.0, np.linspace(0, 1, 11)
    for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        m = (conf >= lo) & ((conf < hi) if i < 9 else (conf <= hi))
        if m.any():
            ece += m.mean() * abs(correct[m].mean() - conf[m].mean())
    print(f"{len(rows)} predictions: accuracy {correct.mean():.3f}, mean self-reported confidence {conf.mean():.3f}, "
          f"ECE {ece:.3f}")
    print(f"{'confidence bin':>16} {'n':>5} {'mean conf':>10} {'accuracy':>9}")
    for lo, hi in ((0, .5), (.5, .7), (.7, .8), (.8, .9), (.9, .95), (.95, 1.01)):
        m = (conf >= lo) & (conf < hi)
        if m.any():
            print(f"{lo:>7.2f} - {min(hi, 1):<6.2f} {m.sum():>5} {conf[m].mean():>10.3f} {correct[m].mean():>9.3f}")


if __name__ == "__main__":
    cmd, args = (sys.argv[1] if len(sys.argv) > 1 else ""), sys.argv[2:]
    {"calibrate": calibrate, "processing": processing, "bedrock": bedrock}.get(
        cmd, lambda *_: sys.exit(__doc__))(*args)
