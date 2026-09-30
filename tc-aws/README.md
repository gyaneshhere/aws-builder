# Text Classification on AWS — from Bag-of-Words to Jev-style Decision APIs

Concrete, runnable AWS implementations of the ten approaches in *Language Models for Text Classification:
From Bag-of-Words to Jev* (Sebastian Raschka), each on a real public dataset, with the training code,
SageMaker launch code, evaluation output and serving path.

## The ten examples

| # | Approach | Concrete use case | Dataset (size) | AWS services | Train on |
|---|---|---|---|---|---|
| 01 | TF-IDF + logistic regression | News topic routing | AG News (120k / 7.6k, 4 classes) | SageMaker scikit-learn, endpoint | ml.m5.2xlarge |
| 02 | BlazingText (fastText-style) | Product-review polarity at scale | Amazon Polarity (1M sample of 3.6M / 50k test) | Built-in BlazingText, Batch Transform | ml.c5.4xlarge |
| 03 | Bidirectional LSTM | Movie-review sentiment | IMDb (25k / 25k) | SageMaker PyTorch 2.8 | ml.g5.xlarge |
| 04 | Text CNN | Question-type routing | TREC (5,452 / 500, 6 classes) | SageMaker PyTorch 2.8 | ml.m5.xlarge |
| 05 | ModernBERT fine-tune | Banking support intent | Banking77 (10,003 / 3,080, 77 intents) | SageMaker Hugging Face, endpoint | ml.g5.2xlarge |
| 06a | Decoder LLM, zero/few-shot | Emotion detection, no training | dair-ai/emotion (2k test) | Amazon Bedrock Converse (tool use) | — |
| 06b | Decoder LLM + classification head | Emotion detection, fine-tuned | dair-ai/emotion (16k / 2k) | SageMaker Hugging Face (Qwen2.5-0.5B) | ml.g5.2xlarge |
| 07 | Flan-T5 text-to-text | 5-level sentiment as words | SST-5 (8,544 / 2,210) | SageMaker Hugging Face | ml.g5.2xlarge |
| 08 | Jev-style Choice / Noul / Score | Decision API without training | Banking77, IMDb, SST-5 samples | Bedrock + Lambda + API Gateway (SAM) | — |
| 09 | Jev-like scoring-head model | One model, any label set | 4 training tasks, 2 held-out tasks | SageMaker Hugging Face, endpoint | ml.g5.2xlarge |
| 10 | Calibration + review threshold | Trustworthy confidence for automation | Banking77 (05's logits) | SageMaker Processing, S3 | ml.m5.large |

Datasets are deliberately reused so results line up: **Banking77** in 05 / 08 / 10, **IMDb** in 03 / 08,
**SST-5** in 07 / 08 / 09, **emotion** in 06a / 06b / 09.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env            # SM_ROLE_ARN, AWS_REGION (BUCKET blank = SageMaker default bucket)
python -m pytest -q tests/      # 10 offline tests, no AWS calls

cd 05-modernbert
python run.py prepare           # dataset -> s3://<bucket>/text-classification/05-modernbert/data/
python run.py train --dry-run   # resolves the container image, launches nothing
python run.py train             # SageMaker training job
python run.py results           # metrics.json + predictions from the job
python run.py deploy && python run.py predict "I still haven't received my new card"
python run.py cleanup           # delete the endpoint
```

Every SageMaker example uses the same commands. Exceptions: 06a (`python run.py bedrock-eval [N] [SHOTS]`),
08 (`sam build && sam deploy --guided`, and `python evaluate.py [N]`), 02 (`python run.py evaluate` for Batch
Transform) and 10 (`python run.py calibrate`).

## What each example produces

```text
s3://<bucket>/text-classification/<example>/
  data/{train,validation,test}/        split files + labels.json
  jobs/<job>/output/model.tar.gz       model weights (+ tokenizer / vocab / labels)
  jobs/<job>/output/output.tar.gz      metrics.json, test_predictions.csv, *_logits.npy (03, 04, 05)
```

`metrics.json` always has accuracy, macro-F1, ECE, Brier score and (for ≤ 20 classes) the confusion matrix.
Training jobs also publish metrics to SageMaker (visible in the console and CloudWatch) through `metric_definitions`.

## Example notes

**01 TF-IDF + LR.** Also writes `top_terms.json`, the highest-weight n-grams per class, which is the
explainability the article credits to linear models. Serves on a CPU endpoint with `{"texts": [...]}`.

**02 BlazingText.** No training script: a built-in algorithm, a `__label__<name> tokens` file and hyperparameters
(supervised mode, bigrams, early stopping on the validation channel). `evaluate` runs Batch Transform with
`{"source": ...}` JSON Lines (the documented batch format) and scores the output. Set `MAX_TRAIN=0` for all 3.6M rows.

**03 / 04 LSTM and Text CNN.** One PyTorch script (`shared/nn_src/train.py`, `--arch lstm|cnn`) with its own
tokenizer and vocabulary, saved with the model. IMDb matches the article's from-scratch LSTM (~85.7%) and ULMFiT
(~95.4%) reference points. TREC is loaded from the original CogComp files (see corrections below).

**05 ModernBERT.** `answerdotai/ModernBERT-base`, 5 epochs, best checkpoint by validation accuracy. Saves
validation and test logits for example 10. The endpoint handler applies 10's temperature and review threshold
when `10-calibration/outputs/calibration.json` exists, returning `needs_review: true` below the threshold.

**06 Decoder LLMs.** 06a forces Bedrock to answer through a tool whose schema has an **enum of the allowed
labels**, so the answer can only be one of them; anything else is counted as an error, never guessed. It reports
accuracy, token cost and how well the model's self-reported confidence is calibrated. 06b fine-tunes
Qwen2.5-0.5B with a classification head (the article's second decoder approach) using 05's training script.

**07 Flan-T5.** Reports two readings of the same model: free generation with an allowlist (counting
`needs_review` outputs) and constrained scoring of the 5 label strings, which is always valid and gives
probabilities. Also reports off-by-one accuracy, because SST-5 labels are ordinal.

**08 Jev-style API.** `POST /choice | /noul | /score` on API Gateway (IAM auth) → Lambda → Bedrock, with Jev-like
response shapes. Probabilities are normalized in code but remain the model's self-report. `evaluate.py` runs
Choice on Banking77 (vs 05), Noul on IMDb (vs 03) and Score on SST-5 (vs 07).

**09 Scoring-head model.** ModernBERT + one shared scalar head scores each (task, text, candidate label +
description) pair; softmax over candidates. Trained on AG News, Banking77 (8 sampled candidates per example),
emotion and TREC; evaluated on **Rotten Tomatoes and SST-5, which it never sees in training**. The endpoint
accepts example 08's request shapes, so the two can be swapped behind one interface.

**10 Calibration.** Fits one temperature on validation logits (bounded 1-D search on NLL), reports test NLL / ECE /
Brier before and after, picks the lowest confidence threshold whose auto-handled predictions reach
`TARGET_ACCURACY` on validation, and writes `calibration.json` next to the model in S3. `processing` runs the same
script as a SageMaker Processing job; `bedrock FILE` bins 06a's self-reported confidence against accuracy.

## Verified here, and how

| Check | Result |
|---|---|
| All 8 training setups resolve to real SageMaker containers (`run.py train --dry-run`) | scikit-learn 1.2-1, BlazingText 1, PyTorch 2.8.0 (CPU and GPU), Hugging Face transformers 4.56.2 / PyTorch 2.8.0 |
| Serving containers | Hugging Face 4.51.3 / PyTorch 2.6.0, PyTorch 2.6.0, scikit-learn 1.2-1 |
| Offline test suite | 10 passed (metrics vs hand values, example 01 train + serve, formats, Bedrock and decision-API logic with a fake client, temperature recovery) |
| Example 01's training script on **real Banking77** (not its AG News target) | test accuracy **0.8828**, macro-F1 0.8834, ECE 0.2513 (under-confident with 77 classes) |
| Example 10 on **real Banking77** logits from a weakly-regularized TF-IDF + LR | T = 1.26; test ECE 0.0336 → 0.0219, accuracy unchanged at 0.8854; threshold 0.73 auto-handles 82.3% of test at 95.5% accuracy (target 95%) |

**Not run here:** the SageMaker jobs themselves, GPU training (03, 05, 06b, 07, 09), Bedrock calls, the SAM
deployment, and datasets other than Banking77 (Hugging Face was not reachable from the test environment). Run
each example's `train --dry-run` first, then one real job, before relying on it.

## Corrections to the source notes

| Source file | Issue | Here |
|---|---|---|
| Hugging Face estimator `transformers_version="4.36"` | ModernBERT needs transformers ≥ 4.48 | 4.56.2 training, 4.51.3 serving |
| `TrainingArguments(evaluation_strategy=...)` | Removed in transformers 4.46 | `eval_strategy` |
| SKLearn `framework_version="1.2-1"` | Fine for serving; 1.4-2 exists for training only | 1.2-1 for both, so the pickle loads |
| Prompted JSON + `json.loads` for Bedrock | Breaks on any stray text; labels unchecked | Converse tool use with an enum schema |
| `CogComp/trec` on Hugging Face | Needs a loading script; `datasets` ≥ 4 refuses | Original CogComp files |
| SageMaker Python SDK | v3 (`sagemaker-train`) is current and replaces estimators | Pinned to v2.257.6; v2 is on a deprecation path |

## Dataset licenses

Banking77 is CC BY 4.0. TREC's card lists its license as unknown. Check each dataset card (AG News, Amazon
Polarity, IMDb, dair-ai/emotion, SST-5, Rotten Tomatoes) before any commercial use.

## Cleanup

`python run.py cleanup` in each example with an endpoint; `sam delete` for example 08. Training jobs stop on
their own; S3 data stays under `s3://<bucket>/text-classification/` until you delete it.
