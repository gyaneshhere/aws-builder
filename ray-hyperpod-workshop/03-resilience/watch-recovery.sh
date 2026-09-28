#!/usr/bin/env bash
# Live view of the three layers recovering: HyperPod node health, KubeRay pods, Ray nodes.
# Refreshes every 10 seconds; Ctrl-C to stop.
source "$(dirname "$0")/../lib/common.sh"
require kubectl
while true; do
  clear 2>/dev/null || true
  echo "${B}$(date -u +%H:%M:%S) UTC${N}   $(cat /tmp/ray-workshop-fault-time 2>/dev/null || echo 'no fault injected yet')"
  echo
  echo "${B}1. HyperPod node health${N}"
  kubectl get nodes -L sagemaker.amazonaws.com/node-health-status -L node.kubernetes.io/instance-type \
    --no-headers 2>/dev/null | awk '{printf "   %-40s %-10s %-28s %s\n", $1, $2, $6, $7}'
  echo
  echo "${B}2. KubeRay pods${N}"
  kc get pods -l "ray.io/cluster=${RAY_CLUSTER_NAME}" -o wide --no-headers 2>/dev/null \
    | awk '{printf "   %-45s %-10s restarts=%-3s node=%s\n", $1, $3, $4, $7}'
  echo
  echo "${B}3. Ray nodes (from the head)${N}"
  h="$(head_pod)"
  if [[ -n "$h" ]]; then
    kc exec "$h" -- ray status 2>/dev/null | sed -n '/Active:/,/Recent failures:/p' | sed 's/^/   /' | head -12
  fi
  echo
  echo "   Training resumes when the logs show:  RESUMED from checkpoint"
  sleep 10
done
