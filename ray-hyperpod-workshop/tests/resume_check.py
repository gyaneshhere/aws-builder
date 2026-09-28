"""Checks the Ray Train control flow that train.py relies on, without PyTorch:
dataset sharding, per-epoch checkpoint reports, FailureConfig retries after a worker
crash, and resuming from the latest checkpoint. Mirrors train.py's structure with a
NumPy "model" so it runs on any laptop with Ray installed.
"""

import json
import os
import time
import sys
import tempfile

import numpy as np
import ray
import ray.train
from ray.train import Checkpoint, CheckpointConfig, FailureConfig, RunConfig, ScalingConfig
from ray.train.v2.api.data_parallel_trainer import DataParallelTrainer  # same base class as ray.train.torch.TorchTrainer in Ray 2.58

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "02-distributed-training"))


def make_batch(batch):
    ids = batch["id"].astype(np.int64)
    rng = np.random.default_rng(ids[0])
    x = rng.standard_normal((len(ids), 32)).astype(np.float32)
    return {"x": x, "y": (x[:, 0] > 0).astype(np.int64)}


def loop(cfg):
    rank = ray.train.get_context().get_world_rank()
    start, w = 0, np.zeros(32)
    ck = ray.train.get_checkpoint()
    if ck:
        with ck.as_directory() as d:
            st = json.load(open(os.path.join(d, "state.json")))
        start, w = st["epoch"] + 1, np.array(st["w"])
        if rank == 0:
            print(f"RESUMED from checkpoint: continuing at epoch {start}", flush=True)
            open(os.path.join(cfg["marker_dir"], f"resumed-at-{start}"), "w").close()
    shard = ray.train.get_dataset_shard("train")
    for epoch in range(start, cfg["epochs"]):
        marker = os.path.join(cfg["marker_dir"], "failure-injected")
        if epoch == cfg["fail_at"] and rank == 1 and not os.path.exists(marker):
            os.makedirs(cfg["marker_dir"], exist_ok=True)
            open(marker, "w").close()
            print(f"[test] rank 1 crashing at epoch {epoch}", flush=True)
            os._exit(1)
        rows = 0
        for b in shard.iter_batches(batch_size=512):
            w += 0.001 * (b["x"].T @ (b["y"] - 0.5))
            rows += len(b["y"])
        time.sleep(cfg["epoch_sleep"])   # like train.py --epoch-sleep: realistic epoch length
        with tempfile.TemporaryDirectory() as d:
            ckpt = None
            if rank == 0:
                json.dump({"epoch": epoch, "w": w.tolist()}, open(os.path.join(d, "state.json"), "w"))
                ckpt = Checkpoint.from_directory(d)
            ray.train.report({"epoch": epoch, "rows": rows}, checkpoint=ckpt)


def main():
    storage = tempfile.mkdtemp(prefix="ray-resume-check-")
    ray.init(num_cpus=4, include_dashboard=False)
    ds = ray.data.range(20_000).map_batches(make_batch, batch_size=4096)
    trainer = DataParallelTrainer(
        loop,
        train_loop_config={"epochs": 4, "fail_at": 2, "epoch_sleep": 3, "marker_dir": os.path.join(storage, "_test")},
        scaling_config=ScalingConfig(num_workers=2, use_gpu=False),
        datasets={"train": ds},
        run_config=RunConfig(name="resume-check", storage_path=storage,
                             failure_config=FailureConfig(max_failures=3),
                             checkpoint_config=CheckpointConfig(num_to_keep=2)),
    )
    result = trainer.fit()
    assert os.path.exists(os.path.join(storage, "_test", "failure-injected")), "fault was not injected"
    assert result.metrics["epoch"] == 3, result.metrics
    resumed = [f for f in os.listdir(os.path.join(storage, "_test")) if f.startswith("resumed-at-")]
    assert resumed, "retry started from scratch instead of resuming from the checkpoint"
    print(f"PASS: crashed at epoch 2, {resumed[0]}, finished epoch {result.metrics['epoch']}; "
          f"checkpoint {result.checkpoint.path}")


if __name__ == "__main__":
    main()
