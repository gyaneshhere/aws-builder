"""Example 7: 5-level sentiment as text-to-text with Flan-T5 (SST-5: 8,544 train / 1,101 val / 2,210 test).

SST-5 labels are words on an ordinal scale ("very negative" ... "very positive"), which suits text-to-text.
The job reports accuracy for free generation (with an allowlist) and for constrained label scoring.
"""

import os
import sys

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
from common.cli import run  # noqa: E402
from common.config import SETTINGS  # noqa: E402
from common.data import export_splits, load_hf  # noqa: E402
from common.sm import common_kwargs, stage_source  # noqa: E402

EXAMPLE = "07-t5"
HERE = os.path.dirname(os.path.abspath(__file__))
SST5_LABELS = ["very negative", "negative", "neutral", "positive", "very positive"]


def prepare():
    splits, labels = load_hf("SetFit/sst5", label_names=SST5_LABELS)
    export_splits(EXAMPLE, splits, labels)


def channels():
    return {s: SETTINGS.s3(EXAMPLE, "data", s) + "/" for s in ("train", "validation", "test")}


def estimator():
    from sagemaker.huggingface import HuggingFace

    return HuggingFace(entry_point="train_t5.py", source_dir=stage_source(os.path.join(HERE, "src")),
                       transformers_version="4.56.2", pytorch_version="2.8.0", py_version="py312",
                       instance_type="ml.g5.2xlarge", instance_count=1,
                       hyperparameters={"model-name": "google/flan-t5-base", "epochs": 4, "batch-size": 16,
                                        "learning-rate": 3e-4},
                       metric_definitions=[{"Name": "test:accuracy_scored", "Regex": r"test_accuracy_scored=([0-9.]+);"},
                                           {"Name": "test:accuracy_generated",
                                            "Regex": r"test_accuracy_generated=([0-9.]+);"}],
                       **common_kwargs(EXAMPLE))


if __name__ == "__main__":
    run(EXAMPLE, prepare=prepare, estimator=estimator, channels=channels)
