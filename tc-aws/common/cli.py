"""One command-line shape for every example:

    python run.py prepare            # dataset -> S3
    python run.py train [--dry-run]  # SageMaker training job (dry run prints the container image and exits)
    python run.py results            # download metrics.json + predictions from the latest job
    python run.py deploy             # real-time endpoint (where the example supports it)
    python run.py predict "some text"
    python run.py cleanup            # delete the endpoint
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Callable

from .sm import delete_endpoint, fetch_output, latest_job, print_metrics


def run(example: str, *, prepare: Callable[[], None], estimator: Callable[[], object],
        channels: Callable[[], dict], deploy: Callable[[str], object] | None = None,
        predict: Callable[[str, str], None] | None = None, extra: dict[str, Callable] | None = None) -> None:
    ap = argparse.ArgumentParser(description=f"{example}: prepare | train | results | deploy | predict | cleanup")
    ap.add_argument("command")
    ap.add_argument("args", nargs="*")
    ap.add_argument("--dry-run", action="store_true", help="build the estimator, print the image, do not launch")
    ap.add_argument("--no-wait", action="store_true")
    a = ap.parse_args()
    endpoint = f"tc-{example}"[:63]

    if a.command == "prepare":
        prepare()
    elif a.command == "train":
        est = estimator()
        print(f"image: {est.training_image_uri()}")
        if a.dry_run:
            print("dry run: estimator built, nothing launched")
            return
        est.fit(channels(), wait=not a.no_wait, logs="All" if not a.no_wait else "None")
        print(f"training job: {est.latest_training_job.name}")
    elif a.command == "results":
        job = a.args[0] if a.args else latest_job(example)
        print_metrics(fetch_output(job, os.path.join("outputs", job)))
    elif a.command == "deploy":
        if not deploy:
            sys.exit("this example has no endpoint; see its README")
        deploy(endpoint)
        print(f"endpoint: {endpoint}")
    elif a.command == "predict":
        if not predict or not a.args:
            sys.exit('usage: python run.py predict "text to classify"')
        predict(endpoint, " ".join(a.args))
    elif a.command == "cleanup":
        delete_endpoint(endpoint)
    elif extra and a.command in extra:
        extra[a.command](*a.args)
    else:
        sys.exit(f"unknown command {a.command}")
