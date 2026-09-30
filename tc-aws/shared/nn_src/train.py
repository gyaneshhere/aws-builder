"""LSTM (example 3) and Text CNN (example 4) classifiers in plain PyTorch (SageMaker PyTorch 2.8 container).

    --arch lstm   bidirectional LSTM over word embeddings (ordered, long-range context)
    --arch cnn    Kim-style CNN: parallel 3/4/5-gram filters + max-pool (local phrase patterns)

Vocabulary is built from the training split and saved with the model, so inference needs nothing else.
Outputs: model.pt, vocab.json, labels.json, config.json (model dir); metrics.json, test_predictions.csv (output).
"""

import argparse
import glob
import json
import os
import re
import time
from collections import Counter

import numpy as np
import pandas as pd
import torch
from torch import nn

TOKEN = re.compile(r"[a-z0-9']+|[.,!?;:()]")
PAD, UNK = 0, 1


def tokenize(text: str) -> list[str]:
    return TOKEN.findall(str(text).lower().replace("<br />", " "))


def build_vocab(texts, size):
    c = Counter(t for x in texts for t in tokenize(x))
    itos = ["<pad>", "<unk>"] + [w for w, _ in c.most_common(size - 2)]
    return {w: i for i, w in enumerate(itos)}


def encode(texts, vocab, max_len):
    ids = np.zeros((len(texts), max_len), dtype=np.int64)
    for i, t in enumerate(texts):
        toks = [vocab.get(w, UNK) for w in tokenize(t)][:max_len]
        ids[i, :len(toks)] = toks
    return torch.from_numpy(ids)


class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, emb_dim, hidden, n_classes, dropout=0.3):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=PAD)
        self.lstm = nn.LSTM(emb_dim, hidden, batch_first=True, bidirectional=True)
        self.drop = nn.Dropout(dropout)
        self.out = nn.Linear(hidden * 2, n_classes)

    def forward(self, ids):
        lengths = (ids != PAD).sum(1).clamp(min=1).cpu()
        packed = nn.utils.rnn.pack_padded_sequence(self.emb(ids), lengths, batch_first=True, enforce_sorted=False)
        _, (h, _) = self.lstm(packed)
        return self.out(self.drop(torch.cat([h[-2], h[-1]], dim=1)))


class TextCNN(nn.Module):
    def __init__(self, vocab_size, emb_dim, n_classes, kernels=(3, 4, 5), channels=128, dropout=0.5):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=PAD)
        self.convs = nn.ModuleList([nn.Conv1d(emb_dim, channels, k, padding=k // 2) for k in kernels])
        self.drop = nn.Dropout(dropout)
        self.out = nn.Linear(channels * len(kernels), n_classes)

    def forward(self, ids):
        x = self.emb(ids).transpose(1, 2)
        pooled = [torch.relu(c(x)).max(dim=2).values for c in self.convs]
        return self.out(self.drop(torch.cat(pooled, dim=1)))


def make_model(cfg):
    if cfg["arch"] == "lstm":
        return LSTMClassifier(cfg["vocab_size"], cfg["emb_dim"], cfg["hidden"], cfg["n_classes"])
    return TextCNN(cfg["vocab_size"], cfg["emb_dim"], cfg["n_classes"])


def read_split(d):
    return pd.concat([pd.read_csv(f) for f in glob.glob(os.path.join(d, "*.csv"))], ignore_index=True)


@torch.no_grad()
def predict_logits(model, ids, device, bs=512):
    model.eval()
    return torch.cat([model(ids[i:i + bs].to(device)).float().cpu() for i in range(0, len(ids), bs)]).numpy()


def main():
    from tc_metrics import classification_metrics, softmax, write_metrics

    p = argparse.ArgumentParser()
    p.add_argument("--arch", choices=["lstm", "cnn"], default="lstm")
    p.add_argument("--epochs", type=int, default=6)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--vocab-size", type=int, default=30_000)
    p.add_argument("--max-len", type=int, default=400)
    p.add_argument("--emb-dim", type=int, default=128)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--train", default=os.environ.get("SM_CHANNEL_TRAIN", "data/train"))
    p.add_argument("--validation", default=os.environ.get("SM_CHANNEL_VALIDATION", "data/validation"))
    p.add_argument("--test", default=os.environ.get("SM_CHANNEL_TEST", "data/test"))
    p.add_argument("--model-dir", default=os.environ.get("SM_MODEL_DIR", "model"))
    p.add_argument("--output-dir", default=os.environ.get("SM_OUTPUT_DATA_DIR", "output"))
    a = p.parse_args()
    os.makedirs(a.model_dir, exist_ok=True)
    os.makedirs(a.output_dir, exist_ok=True)
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train, val, test = read_split(a.train), read_split(a.validation), read_split(a.test)
    labels = json.load(open(os.path.join(a.train, "labels.json")))
    vocab = build_vocab(train["text"], a.vocab_size)
    cfg = {"arch": a.arch, "vocab_size": len(vocab), "emb_dim": a.emb_dim, "hidden": a.hidden,
           "n_classes": len(labels), "max_len": a.max_len}
    xtr, ytr = encode(train["text"].tolist(), vocab, a.max_len), torch.tensor(train["label"].values)
    xva, xte = encode(val["text"].tolist(), vocab, a.max_len), encode(test["text"].tolist(), vocab, a.max_len)
    print(f"arch={a.arch} device={device} train={len(train)} vocab={len(vocab)} classes={len(labels)}")

    model = make_model(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss()
    best, best_state = -1.0, None
    for epoch in range(a.epochs):
        model.train()
        t0, perm, total = time.time(), torch.randperm(len(xtr)), 0.0
        for i in range(0, len(xtr), a.batch_size):
            idx = perm[i:i + a.batch_size]
            loss = loss_fn(model(xtr[idx].to(device)), ytr[idx].to(device))
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += loss.item() * len(idx)
        val_acc = float((predict_logits(model, xva, device).argmax(1) == val["label"].values).mean())
        print(f"epoch={epoch} train_loss={total / len(xtr):.4f}; validation_accuracy={val_acc:.4f}; "
              f"({time.time() - t0:.0f}s)")
        if val_acc > best:
            best, best_state = val_acc, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    metrics = {"best_validation_accuracy": round(best, 4), "config": cfg}
    for name, x, df in (("validation", xva, val), ("test", xte, test)):
        logits = predict_logits(model, x, device)
        probs = softmax(logits)
        metrics[name] = classification_metrics(df["label"].values, probs.argmax(1), probs, labels)
        np.save(os.path.join(a.output_dir, f"{name}_logits.npy"), logits)
        np.save(os.path.join(a.output_dir, f"{name}_labels.npy"), df["label"].values)
    print(f"test_accuracy={metrics['test']['accuracy']}; test_macro_f1={metrics['test']['macro_f1']};")
    write_metrics(os.path.join(a.output_dir, "metrics.json"), metrics)

    torch.save(model.state_dict(), os.path.join(a.model_dir, "model.pt"))
    for name, obj in (("vocab.json", vocab), ("labels.json", labels), ("config.json", cfg)):
        json.dump(obj, open(os.path.join(a.model_dir, name), "w"))


# ---------------- SageMaker inference handlers ----------------
def model_fn(model_dir):
    cfg = json.load(open(os.path.join(model_dir, "config.json")))
    model = make_model(cfg)
    model.load_state_dict(torch.load(os.path.join(model_dir, "model.pt"), map_location="cpu"))
    model.eval()
    return {"model": model, "cfg": cfg, "vocab": json.load(open(os.path.join(model_dir, "vocab.json"))),
            "labels": json.load(open(os.path.join(model_dir, "labels.json")))}


def input_fn(body, content_type):
    data = json.loads(body)
    return [str(t) for t in (data["texts"] if isinstance(data, dict) else data)]


@torch.no_grad()
def predict_fn(texts, b):
    probs = torch.softmax(b["model"](encode(texts, b["vocab"], b["cfg"]["max_len"])), dim=1).numpy()
    return [{"label": b["labels"][int(p.argmax())], "confidence": round(float(p.max()), 4)} for p in probs]


def output_fn(pred, accept):
    return json.dumps({"predictions": pred}), "application/json"


if __name__ == "__main__":
    main()
