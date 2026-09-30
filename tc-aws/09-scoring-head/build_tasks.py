"""Builds the multi-task decision dataset for the scoring-head model (example 9).

Every row is one decision:
  {"task_name", "task", "text", "candidates": [{"label", "description"}, ...], "answer": <index>, "split"}

Training tasks (held-in):  AG News topic, Banking77 intent, emotion, TREC question type
Held-out tasks (eval only): Rotten Tomatoes sentiment, SST-5 5-level sentiment
The model never sees a held-out task, its instruction or its labels during training, so held-out accuracy
measures generalization to new decisions, not memorized label sets.
"""

from __future__ import annotations

import random

from common.data import load_banking77, load_hf, load_trec

AG_NEWS = {"World": "international news, politics, conflicts, diplomacy",
           "Sports": "sports teams, matches, athletes, tournaments",
           "Business": "companies, markets, economy, earnings, deals",
           "Sci/Tech": "science, technology, software, internet, space"}
EMOTION = {"sadness": "feeling sad, down, hurt or lonely", "joy": "feeling happy, glad, pleased or content",
           "love": "feeling affection, caring or romantic attachment", "anger": "feeling angry, irritated or annoyed",
           "fear": "feeling afraid, anxious, nervous or scared", "surprise": "feeling surprised, amazed or shocked"}
TREC = {"ABBR": "asks what an abbreviation stands for or its short form", "ENTY": "asks for an entity: a thing, "
        "animal, product, color, term, etc.", "DESC": "asks for a definition, description, reason or manner",
        "HUM": "asks about a person or group of people", "LOC": "asks about a place: city, country, mountain, etc.",
        "NUM": "asks for a number: count, date, amount, distance, money, etc."}
ROTTEN = {"negative": "the reviewer dislikes the movie", "positive": "the reviewer likes the movie"}
SST5 = {"very negative": "strongly dislikes the movie", "negative": "somewhat dislikes the movie",
        "neutral": "neither likes nor dislikes the movie", "positive": "somewhat likes the movie",
        "very positive": "strongly likes the movie"}

TRAIN_TASKS = ("ag_news", "banking77", "emotion", "trec")
HELDOUT_TASKS = ("rotten_tomatoes", "sst5")


def _rows(task_name, task, df, labels, desc, split, max_rows, rng, n_candidates=None):
    df = df.sample(n=min(max_rows, len(df)), random_state=0) if max_rows else df
    out = []
    for text, y in zip(df["text"], df["label"]):
        cand_idx = list(range(len(labels)))
        if n_candidates and len(labels) > n_candidates:          # negative sampling for large label sets
            negatives = rng.sample([i for i in cand_idx if i != y], n_candidates - 1)
            cand_idx = negatives + [y]
            rng.shuffle(cand_idx)
        cands = [{"label": labels[i], "description": desc(labels[i])} for i in cand_idx]
        out.append({"task_name": task_name, "task": task, "text": str(text), "candidates": cands,
                    "answer": cand_idx.index(y), "split": split})
    return out


def build(max_train_per_task: int = 8000, max_eval_per_task: int = 1000, banking_candidates: int = 8,
          seed: int = 13) -> dict[str, list[dict]]:
    rng = random.Random(seed)
    specs = []
    s, l = load_hf("fancyzhx/ag_news")
    specs.append(("ag_news", "What is the topic of this news article?", s, l, AG_NEWS.get, None))
    s, l = load_banking77()
    specs.append(("banking77", "Which banking support request is the customer making?", s, l,
                  lambda x: x.replace("_", " "), banking_candidates))
    s, l = load_hf("dair-ai/emotion", "split")
    specs.append(("emotion", "Which emotion does the writer express?", s, l, EMOTION.get, None))
    s, l = load_trec()
    specs.append(("trec", "What kind of answer is this question asking for?", s, l, TREC.get, None))

    data = {"train": [], "validation": [], "test": []}
    for name, task, splits, labels, desc, k in specs:
        data["train"] += _rows(name, task, splits["train"], labels, desc, "train", max_train_per_task, rng, k)
        data["validation"] += _rows(name, task, splits["validation"], labels, desc, "validation",
                                    max_eval_per_task, rng, k)

    s, l = load_hf("cornell-movie-review-data/rotten_tomatoes")
    data["test"] += _rows("rotten_tomatoes", "Is this movie review positive or negative?", s["test"], l,
                          ROTTEN.get, "test", None, rng)
    sst5_labels = list(SST5)
    s, _ = load_hf("SetFit/sst5", label_names=sst5_labels)
    data["test"] += _rows("sst5", "How much does the reviewer like the movie?", s["test"], sst5_labels,
                          SST5.get, "test", None, rng)
    rng.shuffle(data["train"])
    return data
