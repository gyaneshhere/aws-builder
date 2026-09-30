"""Fine-tune ModernBERT or a small decoder LLM (example 6b) for sequence classification with the Hugging Face Trainer.

Runs in the SageMaker Hugging Face container (transformers 4.56.2 / PyTorch 2.8). Writes:
  model dir : model weights, tokenizer, labels.json                       -> model.tar.gz
  output    : metrics.json, {validation,test}_logits.npy + _labels.npy    -> output.tar.gz
The logits feed example 10 (temperature scaling), so calibration uses exactly this model's outputs.
"""

import argparse
import glob
import json
import os

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from transformers import (AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding, Trainer,
                          TrainingArguments)


def read_split(d):
    return pd.concat([pd.read_csv(f) for f in glob.glob(os.path.join(d, "*.csv"))], ignore_index=True)


def main():
    from tc_metrics import classification_metrics, softmax, write_metrics

    p = argparse.ArgumentParser()
    p.add_argument("--model-name", default="answerdotai/ModernBERT-base")
    p.add_argument("--epochs", type=float, default=5)
    p.add_argument("--train-batch-size", type=int, default=32)
    p.add_argument("--learning-rate", type=float, default=5e-5)
    p.add_argument("--max-length", type=int, default=64)
    p.add_argument("--train", default=os.environ.get("SM_CHANNEL_TRAIN"))
    p.add_argument("--validation", default=os.environ.get("SM_CHANNEL_VALIDATION"))
    p.add_argument("--test", default=os.environ.get("SM_CHANNEL_TEST"))
    p.add_argument("--model-dir", default=os.environ.get("SM_MODEL_DIR", "model"))
    p.add_argument("--output-dir", default=os.environ.get("SM_OUTPUT_DATA_DIR", "output"))
    a = p.parse_args()

    labels = json.load(open(os.path.join(a.train, "labels.json")))
    tok = AutoTokenizer.from_pretrained(a.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        a.model_name, num_labels=len(labels), id2label=dict(enumerate(labels)),
        label2id={l: i for i, l in enumerate(labels)})
    if tok.pad_token is None:                 # decoder LLMs (example 6b: Qwen) ship without a pad token
        tok.pad_token = tok.eos_token
    model.config.pad_token_id = tok.pad_token_id

    def to_ds(df):
        ds = Dataset.from_pandas(df[["text", "label"]].rename(columns={"label": "labels"}), preserve_index=False)
        return ds.map(lambda b: tok(b["text"], truncation=True, max_length=a.max_length), batched=True,
                      remove_columns=["text"])

    splits = {k: read_split(getattr(a, k)) for k in ("train", "validation", "test")}
    data = {k: to_ds(v) for k, v in splits.items()}
    bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()

    args = TrainingArguments(
        output_dir="/tmp/checkpoints", eval_strategy="epoch", save_strategy="epoch", save_total_limit=1,
        load_best_model_at_end=True, metric_for_best_model="accuracy", learning_rate=a.learning_rate,
        per_device_train_batch_size=a.train_batch_size, per_device_eval_batch_size=128,
        num_train_epochs=a.epochs, warmup_ratio=0.1, weight_decay=0.01, bf16=bf16,
        logging_steps=50, report_to="none", seed=42)

    def compute(ev):
        return {"accuracy": float((ev.predictions.argmax(-1) == ev.label_ids).mean())}

    trainer = Trainer(model=model, args=args, train_dataset=data["train"], eval_dataset=data["validation"],
                      data_collator=DataCollatorWithPadding(tok), compute_metrics=compute)
    trainer.train()

    os.makedirs(a.output_dir, exist_ok=True)
    metrics = {"model_name": a.model_name, "epochs": a.epochs, "learning_rate": a.learning_rate}
    for name in ("validation", "test"):
        logits = trainer.predict(data[name]).predictions.astype(np.float32)
        y = splits[name]["label"].to_numpy()
        np.save(os.path.join(a.output_dir, f"{name}_logits.npy"), logits)
        np.save(os.path.join(a.output_dir, f"{name}_labels.npy"), y)
        probs = softmax(logits)
        metrics[name] = classification_metrics(y, probs.argmax(1), probs, labels)
        print(f"{name}_accuracy={metrics[name]['accuracy']}; {name}_ece={metrics[name]['ece']};")
    write_metrics(os.path.join(a.output_dir, "metrics.json"), metrics)
    json.dump(labels, open(os.path.join(a.output_dir, "labels.json"), "w"))

    trainer.save_model(a.model_dir)          # config.json carries id2label, used by the endpoint
    tok.save_pretrained(a.model_dir)


if __name__ == "__main__":
    main()
