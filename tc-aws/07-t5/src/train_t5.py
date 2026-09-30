"""Text-to-text classification with Flan-T5 (SageMaker Hugging Face container).

Input  "sst5 sentiment: <review>"  ->  target "very negative" | "negative" | "neutral" | "positive" | "very positive"

Two ways to read a label out of the fine-tuned model, both evaluated on the test set:
  generated  greedy decoding, then an allowlist; anything else becomes "needs_review"
  scored     likelihood of each label string, softmax over the 5 -> always valid, and gives probabilities
Outputs metrics.json with both, plus test_predictions.csv.
"""

import argparse
import glob
import json
import os

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from transformers import (AutoModelForSeq2SeqLM, AutoTokenizer, DataCollatorForSeq2Seq, Seq2SeqTrainer,
                          Seq2SeqTrainingArguments)

PREFIX = "sst5 sentiment: "


def read_split(d):
    return pd.concat([pd.read_csv(f) for f in glob.glob(os.path.join(d, "*.csv"))], ignore_index=True)


@torch.no_grad()
def label_logprobs(model, tok, texts, labels, device, bs=32):
    """Sum of token log-probs of each label string given the input -> [n, n_labels]."""
    lab = tok(labels, padding=True, return_tensors="pt")
    lab_ids = lab.input_ids.to(device)
    lab_ids_masked = lab_ids.masked_fill(lab.attention_mask.to(device) == 0, -100)
    out = []
    for i in range(0, len(texts), bs):
        enc = tok([PREFIX + t for t in texts[i:i + bs]], truncation=True, max_length=256, padding=True,
                  return_tensors="pt").to(device)
        rows = []
        for j in range(len(labels)):
            tgt = lab_ids_masked[j].unsqueeze(0).expand(enc.input_ids.size(0), -1)
            logits = model(**enc, labels=tgt).logits.float()
            lp = torch.log_softmax(logits, -1).gather(-1, tgt.clamp(min=0).unsqueeze(-1)).squeeze(-1)
            rows.append((lp * (tgt != -100)).sum(-1))
        out.append(torch.stack(rows, dim=1).cpu())
    return torch.cat(out).numpy()


def main():
    from tc_metrics import classification_metrics, softmax, write_metrics

    p = argparse.ArgumentParser()
    p.add_argument("--model-name", default="google/flan-t5-base")
    p.add_argument("--epochs", type=float, default=4)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--learning-rate", type=float, default=3e-4)
    p.add_argument("--train", default=os.environ.get("SM_CHANNEL_TRAIN"))
    p.add_argument("--validation", default=os.environ.get("SM_CHANNEL_VALIDATION"))
    p.add_argument("--test", default=os.environ.get("SM_CHANNEL_TEST"))
    p.add_argument("--model-dir", default=os.environ.get("SM_MODEL_DIR", "model"))
    p.add_argument("--output-dir", default=os.environ.get("SM_OUTPUT_DATA_DIR", "output"))
    a = p.parse_args()

    labels = json.load(open(os.path.join(a.train, "labels.json")))
    tok = AutoTokenizer.from_pretrained(a.model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(a.model_name)
    splits = {k: read_split(getattr(a, k)) for k in ("train", "validation", "test")}

    def to_ds(df):
        ds = Dataset.from_pandas(df[["text", "label_name"]], preserve_index=False)

        def enc(b):
            x = tok([PREFIX + t for t in b["text"]], truncation=True, max_length=256)
            x["labels"] = tok(text_target=b["label_name"], max_length=8, truncation=True)["input_ids"]
            return x
        return ds.map(enc, batched=True, remove_columns=["text", "label_name"])

    args = Seq2SeqTrainingArguments(
        output_dir="/tmp/checkpoints", eval_strategy="epoch", save_strategy="no", learning_rate=a.learning_rate,
        per_device_train_batch_size=a.batch_size, per_device_eval_batch_size=64, num_train_epochs=a.epochs,
        predict_with_generate=True, generation_max_length=8, logging_steps=50, report_to="none", seed=42)
    trainer = Seq2SeqTrainer(model=model, args=args, train_dataset=to_ds(splits["train"]),
                             eval_dataset=to_ds(splits["validation"]),
                             data_collator=DataCollatorForSeq2Seq(tok, model=model))
    trainer.train()

    test = splits["test"]
    y = test["label"].to_numpy()
    device = model.device
    model.eval()

    # 1) free generation + allowlist
    gen = []
    for i in range(0, len(test), 64):
        enc = tok([PREFIX + t for t in test["text"][i:i + 64]], truncation=True, max_length=256, padding=True,
                  return_tensors="pt").to(device)
        with torch.no_grad():
            ids = model.generate(**enc, max_new_tokens=8)
        gen += [s.strip().lower() for s in tok.batch_decode(ids, skip_special_tokens=True)]
    gen_idx = np.array([labels.index(g) if g in labels else -1 for g in gen])
    needs_review = int((gen_idx < 0).sum())

    # 2) constrained scoring over the 5 label strings
    probs = softmax(label_logprobs(model, tok, test["text"].tolist(), labels, device))
    scored = probs.argmax(1)

    metrics = {
        "generated": {**classification_metrics(y, np.where(gen_idx < 0, (y + 1) % len(labels), gen_idx),
                                               None, labels),
                      "needs_review": needs_review, "note": "needs_review rows counted as wrong"},
        "scored": classification_metrics(y, scored, probs, labels),
        "scored_off_by_one_accuracy": round(float((np.abs(scored - y) <= 1).mean()), 4),
    }
    print(f"test_accuracy_generated={metrics['generated']['accuracy']}; "
          f"test_accuracy_scored={metrics['scored']['accuracy']};")
    os.makedirs(a.output_dir, exist_ok=True)
    write_metrics(os.path.join(a.output_dir, "metrics.json"), metrics)
    out = test[["text", "label_name"]].copy()
    out["generated"], out["scored"] = gen, [labels[i] for i in scored]
    out["scored_confidence"] = probs.max(1)
    out.to_csv(os.path.join(a.output_dir, "test_predictions.csv"), index=False)

    trainer.save_model(a.model_dir)
    tok.save_pretrained(a.model_dir)


if __name__ == "__main__":
    main()
