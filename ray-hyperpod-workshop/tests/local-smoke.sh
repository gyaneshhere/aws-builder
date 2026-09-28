#!/usr/bin/env bash
# Maintainer check: runs the workshop's Ray code on a LOCAL Ray cluster (no Kubernetes, no GPU).
# Requires:  pip install "ray[default,train,serve]==2.58.0"
# Covers: hello_ray.py via a Ray job, Ray Train retry + resume, Serve deploy via a job (toy model), loadgen autoscaling.
# Does not cover: train.py itself (needs PyTorch), GPUs, KubeRay, HyperPod node recovery.
set -euo pipefail
cd "$(dirname "$0")/.."
ray stop --force >/dev/null 2>&1 || true
ray start --head --num-cpus 6 --dashboard-host 127.0.0.1 --disable-usage-stats >/dev/null
trap 'ray stop --force >/dev/null 2>&1 || true' EXIT
export RAY_ADDRESS=http://127.0.0.1:8265

echo "### 1. hello_ray.py as a Ray job"
ray job submit --working-dir 01-first-ray-cluster -- python hello_ray.py 2>&1 | grep -q "expected 6" && echo PASS

echo "### 2. Ray Train retry + resume"
python tests/resume_check.py 2>&1 | grep -E "^PASS" || { echo FAIL; exit 1; }

echo "### 3. Serve deploy (toy model) as a Ray job"
ray job submit --working-dir 04-ray-serve --runtime-env-json '{"env_vars": {"MODEL_ID": "toy"}}' \
  -- python deploy_serve.py 2>&1 | grep -q "is RUNNING" && echo PASS

echo "### 4. Load generator + autoscaling"
python 04-ray-serve/loadgen.py --stages 1:10,48:50,2:50
