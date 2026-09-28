#!/usr/bin/env bash
# Deploys the sentiment API to the RayCluster.
#   ./deploy-serve.sh              model from env.sh (SERVE_MODEL), GPU replicas
#   ./deploy-serve.sh --toy        tiny keyword model: no download, no GPU
source "$(dirname "$0")/../lib/common.sh"
here="$(cd "$(dirname "$0")" && pwd)"
MODEL="$SERVE_MODEL"
[[ "${1:-}" == "--toy" ]] && MODEL="toy"

if [[ "$MODEL" == "toy" ]]; then
  RENV='{"env_vars": {"MODEL_ID": "toy"}}'
else
  RENV=$(printf '{"pip": ["torch==%s", "transformers==%s"], "env_vars": {"MODEL_ID": "%s"}}' \
         "$TORCH_VERSION" "$TRANSFORMERS_VERSION" "$MODEL")
fi
info "Deploying Ray Serve app 'sentiment' (model: ${MODEL})"
submit_job "$here" "$RENV" wait python deploy_serve.py
echo
ok "Test it (after ./01-first-ray-cluster/dashboard.sh start):"
echo "  curl -s localhost:8000/classify -H 'Content-Type: application/json' -d '{\"text\": \"the new patch is great\"}'"
