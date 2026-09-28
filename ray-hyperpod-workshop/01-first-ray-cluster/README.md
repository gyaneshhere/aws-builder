# Module 1 — Your first Ray cluster (25 min)

**Goal:** create a RayCluster on HyperPod, open the Ray dashboard, and see where Ray places your code.

## 1. Create the cluster

```bash
./01-first-ray-cluster/create-cluster.sh --dry-run    # optional: see the rendered manifest
./01-first-ray-cluster/create-cluster.sh
```

The manifest ([raycluster.yaml.tpl](raycluster.yaml.tpl)) is a standard `ray.io/v1` RayCluster:

- **Head pod**: runs the Ray control plane, dashboard (8265) and Serve proxy (8000). It is started with
  `num-cpus: 0`, so no user work lands on it: when a worker node fails later, the head keeps running.
- **Worker group `gpu`**: `GPU_WORKERS` pods, each requesting 1 GPU; anti-affinity spreads them across nodes.
- **FSx PVC** mounted at `/fsx` on every pod: shared storage for checkpoints in Modules 2–3.

The first start pulls the Ray GPU image (several GB), so allow a few minutes. The script finishes with `ray status`.

## 2. Open the dashboard

```bash
./01-first-ray-cluster/dashboard.sh start
```

Open http://localhost:8265. In SageMaker Code Editor, open forwarded port 8265 from the Ports panel. Look at the
**Cluster** tab: one head and `GPU_WORKERS` worker nodes, each worker with 1 GPU.

> **Managed alternative:** Ray on HyperPod also offers authenticated dashboard access from SageMaker Studio,
> without port-forwarding. See [Ray on SageMaker HyperPod](https://docs.aws.amazon.com/sagemaker/latest/dg/sagemaker-hyperpod-ray.html).

## 3. Run your first Ray program

```bash
./01-first-ray-cluster/run-hello.sh
```

[hello_ray.py](hello_ray.py) runs three things. Expected shape of the output (pod names differ):

```text
Cluster: 8 CPUs, 2 GPUs, 3 Ray nodes (head + workers)

1) 40 tasks finished in 6.2s (40 x 0.5s of work) on these pods:
     ray-workshop-gpu-worker-abcde                  20 tasks
     ray-workshop-gpu-worker-fghij                  20 tasks

2) One task per GPU (2):
     ray-workshop-gpu-worker-abcde                  ray_gpu_ids=[0] CUDA_VISIBLE_DEVICES=0
     ray-workshop-gpu-worker-fghij                  ray_gpu_ids=[0] CUDA_VISIBLE_DEVICES=0

3) Actor on ray-workshop-gpu-worker-abcde kept its state across calls: total = 6 (expected 6)
```

**What to notice**
- Tasks spread across workers and none ran on the head (`num-cpus: 0`).
- Ray assigns each GPU task its own GPU and sets `CUDA_VISIBLE_DEVICES` for it.
- An actor is a long-lived process with state; you will meet actors again as Ray Train workers and Serve replicas.

In the dashboard, open **Jobs** to find this run and its logs.

## Troubleshooting

| Symptom | Fix |
|---|---|
| Worker pods `Pending` | `kubectl describe pod <pod>`: not enough GPU/CPU/memory. Lower `WORKER_CPU`/`WORKER_MEMORY` or `GPU_WORKERS` in `env.sh` and re-run `create-cluster.sh`. |
| Pods `Pending` with an FSx volume error | Check `FSX_PVC` and `NAMESPACE`; the PVC must be in the same namespace as the RayCluster. |
| `ImagePullBackOff` | Nodes need internet access (NAT) to pull `rayproject/ray`; or mirror the image to ECR and set `RAY_IMAGE`. |
| Script times out | `kubectl get pods -l ray.io/cluster=ray-workshop -w` and `kubectl logs <pod>` |

✅ **Done when** `run-hello.sh` prints all three sections and the dashboard shows the job as SUCCEEDED.
