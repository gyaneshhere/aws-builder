"""Example 3: movie-review sentiment with a bidirectional LSTM (IMDb, 25k train / 25k test).

Same dataset as the article's from-scratch LSTM (~85.7%) and ULMFiT (~95.4%) numbers, so you can place
your result on that line. GPU training on ml.g5.xlarge; the model is small enough to serve on CPU.
"""

import os
import sys

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
from common.cli import run  # noqa: E402
from common.data import export_splits, load_hf  # noqa: E402
from shared import nn_example as nn  # noqa: E402

EXAMPLE = "03-lstm"


def prepare():
    splits, labels = load_hf("stanfordnlp/imdb")        # validation = 10% of train
    export_splits(EXAMPLE, splits, labels)


if __name__ == "__main__":
    run(EXAMPLE, prepare=prepare, channels=lambda: nn.channels(EXAMPLE),
        estimator=lambda: nn.estimator(EXAMPLE, "ml.g5.xlarge", {
            "arch": "lstm", "epochs": 6, "batch-size": 64, "lr": 2e-3, "vocab-size": 30000,
            "max-len": 400, "emb-dim": 128, "hidden": 128}),
        deploy=lambda ep: nn.deploy(EXAMPLE, ep), predict=nn.predict)
