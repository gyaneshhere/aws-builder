"""Distributed training with Ray Data + Ray Train, built to survive node failures.

What it does
  * Ray Data generates a synthetic classification dataset (no download, runs anywhere)
    and streams a shard to each training worker.
  * Ray Train runs PyTorch DDP across GPU workers (one per HyperPod node).
  * Every epoch writes a checkpoint to shared storage (FSx).
  * FailureConfig(max_failures=...) tells Ray Train to retry after a worker is lost;
    the retry calls this same function again and it resumes from the latest checkpoint.

Why synthetic data? The workshop is about the platform, not the model. A label that is a
non-linear function of the features still gives a loss curve you can watch drop, and an
epoch length you control with --epoch-sleep, so there is time to inject a fault.

    python train.py --workers 2 --epochs 12 --epoch-sleep 20
    python train.py --cpu --workers 2 --epochs 3 --epoch-sleep 0 --storage-path /tmp/ckpt   # laptop
"""

from __future__ import annotations

import argparse
import os
import socket
import tempfile
import time

import numpy as np
import ray
import ray.train
import ray.train.torch
import torch
from ray.train import Checkpoint, CheckpointConfig, FailureConfig, RunConfig, ScalingConfig
from ray.train.torch import TorchTrainer

N_FEATURES = 32


def make_batch(batch: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Deterministic features and a non-linear label derived from each row id."""
    ids = batch["id"].astype(np.int64)
    rng = np.random.default_rng(ids[0])
    x = rng.standard_normal((len(ids), N_FEATURES)).astype(np.float32)
    score = np.sin(x[:, 0] * 2) + x[:, 1] * x[:, 2] - 0.5 * np.abs(x[:, 3]) + 0.3 * x[:, 4:8].sum(axis=1)
    return {"x": x, "y": (score > 0).astype(np.int64)}


def build_model() -> torch.nn.Module:
    return torch.nn.Sequential(
        torch.nn.Linear(N_FEATURES, 256), torch.nn.ReLU(),
        torch.nn.Linear(256, 256), torch.nn.ReLU(),
        torch.nn.Linear(256, 2),
    )


def _maybe_fail_for_testing(cfg: dict, epoch: int) -> None:
    """Test hook for the offline smoke test: crash one worker once at a chosen epoch."""
    at = cfg.get("fail_at_epoch", -1)
    if at < 0 or epoch != at or ray.train.get_context().get_world_rank() != 1:
        return
    marker = os.path.join(cfg["marker_dir"], "failure-injected")
    if not os.path.exists(marker):
        os.makedirs(cfg["marker_dir"], exist_ok=True)
        open(marker, "w").close()
        print(f"[test hook] rank 1 crashing at epoch {epoch}", flush=True)
        os._exit(1)


def train_loop_per_worker(cfg: dict) -> None:
    ctx = ray.train.get_context()
    rank = ctx.get_world_rank()
    device = ray.train.torch.get_device()

    model = ray.train.torch.prepare_model(build_model())
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"])
    loss_fn = torch.nn.CrossEntropyLoss()

    # ---- resume -----------------------------------------------------------------
    start_epoch = 0
    ckpt = ray.train.get_checkpoint()
    if ckpt:
        with ckpt.as_directory() as d:
            state = torch.load(os.path.join(d, "state.pt"), map_location=device)
        model.module.load_state_dict(state["model"]) if hasattr(model, "module") else model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        start_epoch = state["epoch"] + 1
        if rank == 0:
            print(f"RESUMED from checkpoint: continuing at epoch {start_epoch}", flush=True)
    elif rank == 0:
        print("Starting from scratch (no checkpoint yet)", flush=True)

    print(f"worker rank={rank} pod={socket.gethostname()} device={device}", flush=True)
    shard = ray.train.get_dataset_shard("train")

    for epoch in range(start_epoch, cfg["epochs"]):
        _maybe_fail_for_testing(cfg, epoch)
        t0 = time.time()
        model.train()
        total_loss, correct, seen = 0.0, 0, 0
        for batch in shard.iter_torch_batches(batch_size=cfg["batch_size"], device=device):
            logits = model(batch["x"])
            loss = loss_fn(logits, batch["y"])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(batch["y"])
            correct += (logits.argmax(dim=1) == batch["y"]).sum().item()
            seen += len(batch["y"])
        time.sleep(cfg["epoch_sleep"])          # workshop pacing: time to inject a fault

        metrics = {"epoch": epoch, "loss": total_loss / max(seen, 1), "accuracy": correct / max(seen, 1),
                   "epoch_seconds": round(time.time() - t0, 1)}
        with tempfile.TemporaryDirectory() as d:
            checkpoint = None
            if rank == 0:                        # one copy of the weights is enough
                sd = model.module.state_dict() if hasattr(model, "module") else model.state_dict()
                torch.save({"model": sd, "optimizer": optimizer.state_dict(), "epoch": epoch},
                           os.path.join(d, "state.pt"))
                checkpoint = Checkpoint.from_directory(d)
            ray.train.report(metrics, checkpoint=checkpoint)
        if rank == 0:
            print(f"epoch {epoch:3d}  loss={metrics['loss']:.4f}  acc={metrics['accuracy']:.3f}  "
                  f"({metrics['epoch_seconds']}s)  checkpoint saved", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=int(os.environ.get("GPU_WORKERS", 2)))
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--rows", type=int, default=400_000)
    ap.add_argument("--batch-size", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--epoch-sleep", type=float, default=20.0)
    ap.add_argument("--max-failures", type=int, default=10)
    ap.add_argument("--storage-path", default=os.environ.get("STORAGE_PATH", "/fsx/ray-workshop"))
    ap.add_argument("--run-name", default=f"train-{time.strftime('%Y%m%d-%H%M%S')}")
    ap.add_argument("--cpu", action="store_true", help="train on CPU (laptop / smoke test)")
    ap.add_argument("--fail-at-epoch", type=int, default=-1, help=argparse.SUPPRESS)
    a = ap.parse_args()

    ray.init()
    ds = ray.data.range(a.rows).map_batches(make_batch, batch_size=4096)

    trainer = TorchTrainer(
        train_loop_per_worker,
        train_loop_config={"epochs": a.epochs, "batch_size": a.batch_size, "lr": a.lr,
                           "epoch_sleep": a.epoch_sleep, "fail_at_epoch": a.fail_at_epoch,
                           "marker_dir": os.path.join(a.storage_path, a.run_name, "_test")},
        scaling_config=ScalingConfig(num_workers=a.workers, use_gpu=not a.cpu),
        datasets={"train": ds},
        run_config=RunConfig(
            name=a.run_name,
            storage_path=a.storage_path,
            failure_config=FailureConfig(max_failures=a.max_failures),   # default is 0 = no retries
            checkpoint_config=CheckpointConfig(num_to_keep=2),
        ),
    )
    print(f"Run '{a.run_name}': {a.workers} workers, {a.epochs} epochs, checkpoints in {a.storage_path}/{a.run_name}",
          flush=True)
    result = trainer.fit()
    print(f"\nDONE  final metrics: {result.metrics}")
    print(f"      latest checkpoint: {result.checkpoint.path if result.checkpoint else None}")


if __name__ == "__main__":
    main()
