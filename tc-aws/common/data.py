"""Pull a Hugging Face dataset, make train/validation/test splits, write them to S3.

Every example calls export_splits() so the S3 layout is identical:
    s3://<bucket>/<prefix>/<example>/data/{train,validation,test}/<name>.csv|jsonl
"""

from __future__ import annotations

import json
import os
import tempfile
from typing import Callable

import pandas as pd

from .config import SETTINGS


def load_hf(name: str, config: str | None = None, text_col: str = "text", label_col: str = "label",
            val_from_train: float = 0.1, seed: int = 42, max_train: int | None = None,
            max_test: int | None = None, label_names: list[str] | None = None) -> tuple[dict[str, pd.DataFrame], list[str]]:
    """Returns ({"train","validation","test"} -> DataFrame[text,label,label_name], label_names)."""
    from datasets import load_dataset

    ds = load_dataset(name, config) if config else load_dataset(name)
    feat = ds["train"].features[label_col]
    names = label_names or feat.names                     # label_names for datasets without a ClassLabel column
    if "validation" not in ds:
        split = ds["train"].train_test_split(test_size=val_from_train, seed=seed,
                                             stratify_by_column=label_col if hasattr(feat, "names") else None)
        ds["train"], ds["validation"] = split["train"], split["test"]
    out = {}
    for split in ("train", "validation", "test"):
        d = ds[split]
        if split == "train" and max_train and len(d) > max_train:
            d = d.shuffle(seed=seed).select(range(max_train))
        if split == "test" and max_test and len(d) > max_test:
            d = d.shuffle(seed=seed).select(range(max_test))
        df = pd.DataFrame({"text": d[text_col], "label": d[label_col]})
        df["label_name"] = df["label"].map(lambda i: names[i])
        out[split] = df
    return out, list(names)


def export_splits(example: str, splits: dict[str, pd.DataFrame], labels: list[str], fmt: str = "csv",
                  formatter: Callable[[pd.DataFrame], str] | None = None, ext: str | None = None) -> dict[str, str]:
    """Writes each split (and labels.json) to S3; returns {split: s3_prefix}."""
    s3 = SETTINGS.boto().client("s3")
    bucket = SETTINGS.resolved_bucket()
    uris = {}
    with tempfile.TemporaryDirectory() as tmp:
        for split, df in splits.items():
            fname = f"{split}.{ext or fmt}"
            path = os.path.join(tmp, fname)
            if formatter is not None:
                open(path, "w").write(formatter(df))
            elif fmt == "csv":
                df.to_csv(path, index=False)
            else:
                df.to_json(path, orient="records", lines=True)
            key = f"{SETTINGS.prefix}/{example}/data/{split}/{fname}"
            s3.upload_file(path, bucket, key)
            json_key = f"{SETTINGS.prefix}/{example}/data/{split}/labels.json"
            s3.put_object(Bucket=bucket, Key=json_key, Body=json.dumps(labels).encode())
            uris[split] = f"s3://{bucket}/{SETTINGS.prefix}/{example}/data/{split}/"
            print(f"  {split:10s} {len(df):>9,} rows -> {uris[split]}{fname}")
    return uris


BANKING77_URL = "https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/master/banking_data/{}.csv"


def load_banking77(val_frac: float = 0.1, seed: int = 42) -> tuple[dict[str, pd.DataFrame], list[str]]:
    """Banking77 (PolyAI, CC BY 4.0) from its first-party GitHub CSVs: 10,003 train / 3,080 test, 77 intents.

    Used by examples 05, 08 and 10 so their results are directly comparable.
    """
    from sklearn.model_selection import train_test_split

    train_full = pd.read_csv(BANKING77_URL.format("train"))
    test = pd.read_csv(BANKING77_URL.format("test"))
    labels = sorted(train_full["category"].unique())
    idx = {name: i for i, name in enumerate(labels)}
    for df in (train_full, test):
        df["label_name"] = df.pop("category")
        df["label"] = df["label_name"].map(idx)
    train, val = train_test_split(train_full, test_size=val_frac, random_state=seed, stratify=train_full["label"])
    cols = ["text", "label", "label_name"]
    return {"train": train[cols], "validation": val[cols], "test": test[cols]}, labels


TREC_URLS = {"train": "https://cogcomp.seas.upenn.edu/Data/QA/QC/train_5500.label",
             "test": "https://cogcomp.seas.upenn.edu/Data/QA/QC/TREC_10.label"}
TREC_LABELS = ["ABBR", "ENTY", "DESC", "HUM", "LOC", "NUM"]      # same order as the Hugging Face card


def parse_trec(raw: bytes) -> pd.DataFrame:
    rows = []
    for line in raw.decode("latin-1").splitlines():                # the source files are not UTF-8
        if " " not in line.strip():
            continue
        fine, text = line.strip().split(" ", 1)
        coarse = fine.split(":")[0]
        rows.append({"text": text, "label": TREC_LABELS.index(coarse), "label_name": coarse})
    return pd.DataFrame(rows)


def load_trec(val_frac: float = 0.1, seed: int = 42) -> tuple[dict[str, pd.DataFrame], list[str]]:
    """TREC question classification from the original CogComp files (5,452 train / 500 test, 6 coarse classes).

    The Hugging Face copy (CogComp/trec) still needs a loading script, which datasets>=4 no longer runs.
    """
    import urllib.request

    from sklearn.model_selection import train_test_split

    frames = {k: parse_trec(urllib.request.urlopen(u, timeout=30).read()) for k, u in TREC_URLS.items()}
    train, val = train_test_split(frames["train"], test_size=val_frac, random_state=seed,
                                  stratify=frames["train"]["label"])
    return {"train": train, "validation": val, "test": frames["test"]}, TREC_LABELS
