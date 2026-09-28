# Module 2 — Distributed training with Ray Data + Ray Train (30 min)

**Goal:** train a PyTorch model across GPU nodes, with a checkpoint on FSx every epoch.

## What the script does

[train.py](train.py) is about 150 lines:

| Piece | Ray API | What it does |
|---|---|---|
| Data | `ray.data.range(...).map_batches(...)` | Generates a synthetic dataset and streams one shard to each worker |
| Training | `TorchTrainer` + `ScalingConfig(num_workers=N, use_gpu=True)` | Runs PyTorch DDP, one worker per GPU node |
| Checkpoints | `ray.train.report(metrics, checkpoint=...)` | Rank 0 saves model + optimizer + epoch to `STORAGE_PATH` on FSx |
| Retries | `FailureConfig(max_failures=10)` | After a worker is lost, Ray Train restarts the workers and calls the training function again |
| Resume | `ray.train.get_checkpoint()` | On a retry, returns the latest checkpoint, so training continues instead of starting over |

The data is synthetic on purpose: no download, no credentials, runs the same everywhere. The label is a non-linear
function of the features, so accuracy climbs visibly. `--epoch-sleep` (default 20s) paces each epoch so you have
time to break something in Module 3.

## 1. Submit the job

```bash
./02-distributed-training/submit-training.sh
```

PyTorch is installed through a Ray **runtime environment** (`{"pip": ["torch==2.7.1"]}`), so there is no custom
image to build. The first job on each node spends a few minutes installing it; later jobs reuse the cached env.

## 2. Follow it

```bash
./02-distributed-training/job.sh logs      # Ctrl-C stops following, not the job
./02-distributed-training/job.sh status
```

Expected shape of the logs:

```text
Run 'train-20260928-101500': 2 workers, 12 epochs, checkpoints in /fsx/ray-workshop/train-20260928-101500
Starting from scratch (no checkpoint yet)
worker rank=0 pod=ray-workshop-gpu-worker-abcde device=cuda:0
worker rank=1 pod=ray-workshop-gpu-worker-fghij device=cuda:0
epoch   0  loss=0.5731  acc=0.694  (23.1s)  checkpoint saved
epoch   1  loss=0.4012  acc=0.812  (21.4s)  checkpoint saved
...
DONE  final metrics: {'epoch': 11, 'loss': ..., 'accuracy': ...}
```

In the dashboard, the **Train** tab shows the run, its workers, and reported metrics; **Cluster** shows both GPUs busy.

## 3. Look at the checkpoints

```bash
source env.sh
kubectl -n $NAMESPACE exec $(kubectl -n $NAMESPACE get pod -l ray.io/node-type=head -o name) -- \
  bash -c "ls -1 $STORAGE_PATH/*/ | tail -5"
```

Only the two newest checkpoints are kept (`CheckpointConfig(num_to_keep=2)`).

## Try it

- `./02-distributed-training/submit-training.sh --epochs 4 --epoch-sleep 0` for a fast run.
- Lower `--workers` to 1 and compare epoch time.

**Keep a long run going for Module 3:** submit one with the defaults (12 epochs × ~20s) and move on while it runs.

✅ **Done when** the logs show epochs completing with `checkpoint saved` and loss decreasing.
