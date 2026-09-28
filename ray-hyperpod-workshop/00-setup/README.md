# Module 0 — Setup (15 min)

**Goal:** confirm your cluster is ready and install the KubeRay operator.

## 1. Tools

You need `kubectl`, `helm`, the AWS CLI v2 and `python3` on the machine you run the workshop from
(SageMaker Code Editor or AWS CloudShell work well). `envsubst` is optional.

Point `kubectl` at the EKS cluster that orchestrates your HyperPod cluster:

```bash
aws eks update-kubeconfig --name <eks-cluster-name> --region <region>
```

## 2. Configure `env.sh`

Open `env.sh` at the repo root and set at least:

| Variable | What to put |
|---|---|
| `AWS_REGION` | Region of your HyperPod cluster |
| `HYPERPOD_CLUSTER_NAME` | HyperPod cluster name (used to confirm automatic node recovery for Module 3) |
| `NAMESPACE` | Namespace that holds your FSx PVC (often `default`) |
| `FSX_PVC` | Name of the FSx for Lustre PVC (`kubectl get pvc -A`) |

The defaults size the Ray pods for `ml.g5.2xlarge` (1 GPU, 8 vCPU, 32 GiB). For other GPU instances adjust
`WORKER_CPU` / `WORKER_MEMORY` so a worker fits on one node with room for system pods.

## 3. Check prerequisites

```bash
./00-setup/check-prereqs.sh
```

It checks tools, cluster access, GPU capacity (at least `GPU_WORKERS` GPUs, and 2 GPU nodes for Module 3),
HyperPod node health labels, the PVC, KubeRay, and whether `NodeRecovery` is `Automatic`. Fix every `✘` before
continuing; `!` items are warnings.

## 4. Install KubeRay

```bash
./00-setup/install-kuberay.sh
```

Installs KubeRay operator `1.7.1` with Helm into the `kuberay` namespace. If your cluster already runs a KubeRay
operator, the script keeps it (HyperPod works with an existing operator); pass `--force` to upgrade.

## 5. Permissions (shared clusters)

If you are not a cluster admin, your IAM principal needs a HyperPod cluster-access policy on its EKS access entry:
`AmazonSagemakerHyperpodTrainingPolicy` for RayCluster and RayJob, `AmazonSagemakerHyperpodInferencePolicy` for
RayService. See [Managing Ray workloads with kubectl](https://docs.aws.amazon.com/sagemaker/latest/dg/sagemaker-hyperpod-ray-manage-kubectl.html).
Module 3's node reboot also needs permission to label nodes.

✅ **Done when** `check-prereqs.sh` shows `FAIL=0` and `kubectl get pods -n kuberay` shows the operator `Running`.
