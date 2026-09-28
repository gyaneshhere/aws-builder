#!/usr/bin/env bash
# Checks everything the workshop needs and prints a PASS / WARN / FAIL summary. Safe to re-run.
source "$(dirname "$0")/../lib/common.sh"
set +e
PASS=0; WARN=0; FAIL=0
p() { ok "$*"; PASS=$((PASS+1)); }
w() { warn "$*"; WARN=$((WARN+1)); }
f() { fail "$*"; FAIL=$((FAIL+1)); }

info "Local tools"
for t in kubectl helm aws python3; do command -v "$t" >/dev/null && p "$t found" || f "$t missing"; done
command -v envsubst >/dev/null && p "envsubst found" || w "envsubst missing (python3 fallback will be used)"

info "Cluster access"
if kubectl cluster-info >/dev/null 2>&1; then
  p "kubectl can reach $(kubectl config current-context)"
else
  f "kubectl cannot reach a cluster: run  aws eks update-kubeconfig --name <eks-cluster> --region ${AWS_REGION}"
  echo; echo "PASS=${PASS} WARN=${WARN} FAIL=${FAIL}"; exit 1
fi

info "HyperPod nodes"
kubectl get nodes -L node.kubernetes.io/instance-type -L sagemaker.amazonaws.com/node-health-status
gpus=$(kubectl get nodes -o jsonpath='{range .items[*]}{.status.allocatable.nvidia\.com/gpu}{"\n"}{end}' | awk '{s+=$1} END {print s+0}')
gpu_nodes=$(kubectl get nodes -o jsonpath='{range .items[*]}{.status.allocatable.nvidia\.com/gpu}{"\n"}{end}' | awk '$1>0' | wc -l | tr -d ' ')
[[ "$gpus" -ge "$GPU_WORKERS" ]] && p "${gpus} allocatable GPUs on ${gpu_nodes} nodes (need ${GPU_WORKERS})" \
  || f "only ${gpus} allocatable GPUs; need GPU_WORKERS=${GPU_WORKERS} (is the NVIDIA device plugin running?)"
[[ "$gpu_nodes" -ge 2 ]] && p "${gpu_nodes} GPU nodes: Module 3 can reboot one while the other keeps the head" \
  || w "fewer than 2 GPU nodes: Module 3 must use --mode pod"
unhealthy=$(kubectl get nodes -o jsonpath='{range .items[*]}{.metadata.labels.sagemaker\.amazonaws\.com/node-health-status}{"\n"}{end}' \
  | grep -v -e '^Schedulable$' -e '^$' | wc -l | tr -d ' ')
[[ "$unhealthy" -eq 0 ]] && p "no nodes waiting on health checks or recovery" || w "${unhealthy} node(s) not Schedulable yet"

info "Storage"
phase=$(kubectl -n "$NAMESPACE" get pvc "$FSX_PVC" -o jsonpath='{.status.phase}' 2>/dev/null)
[[ "$phase" == "Bound" ]] && p "PVC ${NAMESPACE}/${FSX_PVC} is Bound" \
  || f "PVC ${NAMESPACE}/${FSX_PVC} not found or not Bound (set NAMESPACE / FSX_PVC in env.sh)"

info "KubeRay operator"
if kubectl get crd rayclusters.ray.io >/dev/null 2>&1; then
  v=$(kubectl get deploy -A -l app.kubernetes.io/name=kuberay-operator -o jsonpath='{.items[0].spec.template.spec.containers[0].image}' 2>/dev/null)
  p "KubeRay installed (${v:-image unknown})"
else
  w "KubeRay not installed yet: run 00-setup/install-kuberay.sh"
fi

info "Automatic node recovery (Module 3)"
if [[ -z "$HYPERPOD_CLUSTER_NAME" ]]; then
  w "HYPERPOD_CLUSTER_NAME not set in env.sh; skipping"
else
  rec=$(aws sagemaker describe-cluster --cluster-name "$HYPERPOD_CLUSTER_NAME" --region "$AWS_REGION" \
        --query 'NodeRecovery' --output text 2>/dev/null)
  [[ "$rec" == "Automatic" ]] && p "NodeRecovery=Automatic on ${HYPERPOD_CLUSTER_NAME}" \
    || w "NodeRecovery='${rec:-unknown}': node reboot in Module 3 needs Automatic (use --mode pod otherwise)"
fi

echo; echo "${B}PASS=${PASS} WARN=${WARN} FAIL=${FAIL}${N}"
[[ "$FAIL" -eq 0 ]]
