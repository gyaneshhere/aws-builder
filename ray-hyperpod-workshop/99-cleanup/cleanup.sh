#!/usr/bin/env bash
# Removes everything the workshop created.
#   ./cleanup.sh                  Serve app, RayCluster, port-forwards
#   ./cleanup.sh --purge-data     also delete checkpoints under STORAGE_PATH on FSx
#   ./cleanup.sh --all            also uninstall the KubeRay operator (only if this workshop installed it)
source "$(dirname "$0")/../lib/common.sh"
require kubectl
PURGE=0; ALL=0
for a in "$@"; do
  case "$a" in --purge-data) PURGE=1 ;; --all) ALL=1; PURGE=1 ;; *) die "unknown argument $a" ;; esac
done

"$(dirname "$0")/../01-first-ray-cluster/dashboard.sh" stop || true

h="$(head_pod)"
if [[ -n "$h" ]]; then
  info "Stopping Ray Serve"
  kc exec "$h" -- serve shutdown -y --address http://127.0.0.1:8265 >/dev/null 2>&1 && ok "Serve stopped" || ok "no Serve app"
  if [[ "$PURGE" -eq 1 ]]; then
    info "Deleting checkpoints in ${STORAGE_PATH}"
    [[ "$STORAGE_PATH" == /fsx/* && "$STORAGE_PATH" != "/fsx/" ]] || die "refusing to delete STORAGE_PATH='${STORAGE_PATH}'"
    kc exec "$h" -- rm -rf "$STORAGE_PATH" && ok "checkpoints removed"
  fi
fi

info "Deleting RayCluster ${NAMESPACE}/${RAY_CLUSTER_NAME}"
kc delete raycluster "$RAY_CLUSTER_NAME" --ignore-not-found --wait=true
ok "RayCluster deleted"

info "Checking node-health labels set by Module 3"
for n in $(kubectl get nodes -l sagemaker.amazonaws.com/node-health-status=UnschedulablePendingReboot -o name 2>/dev/null); do
  warn "${n} is still pending reboot; HyperPod will clear the label after recovery"
done

if [[ "$ALL" -eq 1 ]]; then
  info "Uninstalling KubeRay operator"
  helm -n kuberay uninstall kuberay-operator 2>/dev/null && ok "KubeRay removed" || warn "helm release kuberay-operator not found in namespace kuberay (installed another way?)"
fi
rm -f /tmp/ray-workshop-last-job /tmp/ray-workshop-fault-time
ok "Cleanup complete. HyperPod nodes are untouched; scale or delete the HyperPod cluster separately if it was created only for this workshop."
