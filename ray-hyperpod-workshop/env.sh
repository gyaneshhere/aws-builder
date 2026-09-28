# shellcheck shell=bash
# Workshop settings. Edit the values in this block, then: source env.sh
# Every script sources this file, so change a value here once and it applies everywhere.

# ---- Your cluster ------------------------------------------------------------
export AWS_REGION="${AWS_REGION:-us-west-2}"
export HYPERPOD_CLUSTER_NAME="${HYPERPOD_CLUSTER_NAME:-}"   # HyperPod cluster name (used by Module 3 checks)
export NAMESPACE="${NAMESPACE:-default}"                    # must be the namespace of your FSx PVC
export FSX_PVC="${FSX_PVC:-fsx-claim}"                      # shared file system for checkpoints

# ---- Ray cluster shape (defaults fit 2+ x ml.g5.2xlarge: 1 GPU, 8 vCPU, 32 GiB) --
export RAY_CLUSTER_NAME="${RAY_CLUSTER_NAME:-ray-workshop}"
export GPU_WORKERS="${GPU_WORKERS:-2}"          # one Ray worker pod per GPU node
export WORKER_CPU="${WORKER_CPU:-4}"
export WORKER_MEMORY="${WORKER_MEMORY:-16Gi}"
export HEAD_CPU="${HEAD_CPU:-2}"
export HEAD_MEMORY="${HEAD_MEMORY:-6Gi}"

# ---- Versions (pinned so every attendee gets the same behavior) --------------
export RAY_VERSION="${RAY_VERSION:-2.58.0}"
export RAY_IMAGE="${RAY_IMAGE:-rayproject/ray:${RAY_VERSION}-py311-gpu}"
export KUBERAY_VERSION="${KUBERAY_VERSION:-1.7.1}"
export TORCH_VERSION="${TORCH_VERSION:-2.7.1}"
export TRANSFORMERS_VERSION="${TRANSFORMERS_VERSION:-4.56.2}"

# ---- Workload settings ---------------------------------------------------------
export STORAGE_PATH="${STORAGE_PATH:-/fsx/ray-workshop}"    # checkpoints (inside the pods)
export SERVE_MODEL="${SERVE_MODEL:-distilbert/distilbert-base-uncased-finetuned-sst-2-english}"
