#!/usr/bin/env bash
# Submits train.py as a Ray job and returns immediately so you can watch it (and break it in Module 3).
#   ./submit-training.sh                      12 epochs, ~20s each
#   ./submit-training.sh --epochs 4 --epoch-sleep 5   (any train.py flag passes through)
source "$(dirname "$0")/../lib/common.sh"
here="$(cd "$(dirname "$0")" && pwd)"

# PyTorch is installed per job through a Ray runtime environment, so no custom image is needed.
# The first job on each node spends a few minutes installing it; later jobs reuse the cached env.
RENV=$(printf '{"pip": ["torch==%s"], "env_vars": {"STORAGE_PATH": "%s", "GPU_WORKERS": "%s"}}' \
       "$TORCH_VERSION" "$STORAGE_PATH" "$GPU_WORKERS")

info "Submitting train.py (${GPU_WORKERS} GPU workers, checkpoints in ${STORAGE_PATH})"
out=$(submit_job "$here" "$RENV" nowait python train.py "$@")
echo "$out" | tail -n 8
job=$(echo "$out" | grep -oE "raysubmit_[A-Za-z0-9]+" | head -1)
[[ -n "$job" ]] || die "could not read the job id from the output above"
echo "$job" > /tmp/ray-workshop-last-job
ok "Submitted ${job}"
echo "  Follow logs:   ./02-distributed-training/job.sh logs"
echo "  Job status:    ./02-distributed-training/job.sh status"
