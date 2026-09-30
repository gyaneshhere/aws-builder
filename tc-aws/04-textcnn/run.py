"""Example 4: question-type routing with a Text CNN (TREC, 5,452 train / 500 test, 6 coarse classes).

Short questions whose class hinges on local phrases ("how many", "who was", "where is") suit a CNN's
n-gram filters. Small enough to train on a CPU instance in minutes. TREC is downloaded from the original
CogComp files because the Hugging Face copy still needs a loading script (unsupported in datasets>=4).
"""

import os
import sys

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
from common.cli import run  # noqa: E402
from common.data import export_splits, load_trec  # noqa: E402
from shared import nn_example as nn  # noqa: E402

EXAMPLE = "04-textcnn"


def prepare():
    splits, labels = load_trec()
    export_splits(EXAMPLE, splits, labels)


if __name__ == "__main__":
    run(EXAMPLE, prepare=prepare, channels=lambda: nn.channels(EXAMPLE),
        estimator=lambda: nn.estimator(EXAMPLE, "ml.m5.xlarge", {
            "arch": "cnn", "epochs": 12, "batch-size": 32, "lr": 1e-3, "vocab-size": 10000,
            "max-len": 40, "emb-dim": 128}),
        deploy=lambda ep: nn.deploy(EXAMPLE, ep), predict=nn.predict)
