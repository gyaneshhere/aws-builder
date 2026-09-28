# What was verified, and how

| Check | Method | Result |
|---|---|---|
| Shell scripts | `bash -n` and `shellcheck -S warning` on every script | Clean |
| RayCluster manifest | Rendered from `env.sh`, validated with `kubeconform -strict` against the `ray.io/v1` schema from the KubeRay **v1.7.1** CRD | Valid; a misspelled field is rejected |
| `hello_ray.py` | Submitted as a Ray job to a local Ray 2.58.0 cluster | Tasks, actor state (total 6) as expected |
| Ray Train retry + resume | `resume_check.py`: same control flow as `train.py` (Ray Data shards, rank-0 checkpoints, `FailureConfig`, `get_checkpoint`) with a NumPy model; one worker is killed at epoch 2 | Retried and **resumed from a checkpoint**, finished all epochs |
| Serve deploy | `deploy_serve.py` submitted as a Ray job with `MODEL_ID=toy` | App RUNNING after the job exits; 400 on bad input |
| Autoscaling + load generator | `loadgen.py` stages 1 → 48 → 2 clients | 1 → 6 → 1 replicas, 0 errors |

Run them yourself with `./tests/local-smoke.sh` (requires `pip install "ray[default,train,serve]==2.58.0"`).

## Not verified here

- **On a HyperPod cluster**: KubeRay scheduling, the GPU image, FSx mounts, runtime-env installs of PyTorch and
  Transformers, the node reboot in Module 3, and the distilbert model on GPU. Do one full dry run before delivering.
- **`train.py` itself**: needs PyTorch, which was not installed in the test environment. Its Ray Train control flow
  is covered by `resume_check.py`; the PyTorch parts use standard `ray.train.torch` APIs (`prepare_model`,
  `get_device`, `iter_torch_batches`).
- **The `rayproject/ray:2.58.0-py311-gpu` tag**: follows the published naming scheme but was not pulled here.

## Two findings from testing, reflected in the workshop

1. **Ray 2.58 uses Ray Train V2 by default.** `ray.train.torch.TorchTrainer` is V2; mixing in V1 classes (for example
   `ray.train.data_parallel_trainer.DataParallelTrainer`) fails with a `RunConfig` type error.
2. **A checkpoint reported just before a crash can be lost.** With near-instant epochs the retry started from
   scratch; with 3-second epochs it resumed, but from the checkpoint before the newest one. Module 3 tells
   attendees to expect to repeat up to one epoch, and `train.py` paces epochs with `--epoch-sleep`.
