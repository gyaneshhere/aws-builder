# Module 3 — Resilience: break a node, keep training (30 min)

**Goal:** reboot the node under a Ray worker in the middle of training, and watch three layers recover:
HyperPod restores the node, KubeRay restores the pod, and Ray Train resumes from the last checkpoint.

## How recovery works

1. **HyperPod** monitors node health. A node labeled `sagemaker.amazonaws.com/node-health-status=UnschedulablePendingReboot`
   is rebooted by HyperPod when **automatic node recovery** is enabled on the cluster (`NodeRecovery: Automatic`).
   See [manual reboot or replace](https://docs.aws.amazon.com/sagemaker/latest/dg/sagemaker-hyperpod-eks-resiliency-manual.html).
   Health checks apply the same label when real hardware fails.
2. **KubeRay** sees the worker pod disappear and recreates it once the node is schedulable again.
3. **Ray Train** detects the lost worker, restarts the worker group (up to `FailureConfig.max_failures`) and calls
   the training function again; `ray.train.get_checkpoint()` returns the latest checkpoint.

The default for `max_failures` is **0**: without setting it, a single fault ends the run even though HyperPod
recovered the node ([AWS docs](https://docs.aws.amazon.com/sagemaker/latest/dg/sagemaker-hyperpod-ray-node-recovery.html)).
`train.py` sets it to 10.

## 1. Start a training run (if one is not running)

```bash
./02-distributed-training/submit-training.sh
./02-distributed-training/job.sh logs          # wait for 2–3 "checkpoint saved" lines
```

## 2. Open the recovery view (second terminal)

```bash
./03-resilience/watch-recovery.sh
```

## 3. Inject the fault (third terminal)

```bash
./03-resilience/inject-fault.sh
```

The script picks a worker that is **not** on the head's node (losing the head ends the whole Ray cluster), confirms
`NodeRecovery=Automatic`, asks for confirmation, and labels that node for reboot.

**No permission to label nodes, or NodeRecovery is not Automatic?** Use `./03-resilience/inject-fault.sh --mode pod`:
it force-deletes the worker pod instead. HyperPod is not involved, but KubeRay and Ray Train recover the same way.

## 4. What you should see

| When | Where | What |
|---|---|---|
| Seconds | training logs | a worker error, then Ray Train retrying (`Error count: 1 (max allowed: 10)`) |
| Seconds | recovery view | the node's health label changes; the worker pod terminates |
| Minutes | recovery view | the node returns `Schedulable`, KubeRay starts a new worker pod |
| After that | training logs | `RESUMED from checkpoint: continuing at epoch N`, then epochs continue to the end |

Training waits for the replacement worker, because it asks for `GPU_WORKERS` workers.

**Expect to repeat up to one epoch.** Ray Train resumes from the last checkpoint that its controller has
registered. In our local test, a worker that crashed immediately after reporting epoch 1 resumed at epoch 1, not 2:
the newest report had not been registered yet. This is why frequent, cheap checkpoints matter.

## 5. Discuss

- What would have happened with `max_failures=0`?
- Where would a checkpoint on the node's local disk have gone?
- Why is the head started with `num-cpus: 0`, and why does the script avoid the head's node?

✅ **Done when** the training job finishes as SUCCEEDED and its logs contain `RESUMED from checkpoint`.
