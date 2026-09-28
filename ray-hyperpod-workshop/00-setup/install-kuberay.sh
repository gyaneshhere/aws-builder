#!/usr/bin/env bash
# Installs the open-source KubeRay operator with Helm. Skips if one is already running.
#   ./install-kuberay.sh           install if missing
#   ./install-kuberay.sh --force   install/upgrade to KUBERAY_VERSION even if present
source "$(dirname "$0")/../lib/common.sh"
require kubectl helm

if [[ "${1:-}" != "--force" ]] && kubectl get deploy -A -l app.kubernetes.io/name=kuberay-operator --no-headers 2>/dev/null | grep -q .; then
  ok "A KubeRay operator is already running; keeping it (use --force to upgrade to ${KUBERAY_VERSION})."
  kubectl get deploy -A -l app.kubernetes.io/name=kuberay-operator
  exit 0
fi

info "Installing KubeRay operator ${KUBERAY_VERSION}"
helm repo add kuberay https://ray-project.github.io/kuberay-helm/ >/dev/null 2>&1 || true
helm repo update kuberay >/dev/null
helm upgrade --install kuberay-operator kuberay/kuberay-operator \
  --version "${KUBERAY_VERSION}" --namespace kuberay --create-namespace --wait --timeout 5m
kubectl -n kuberay rollout status deploy/kuberay-operator --timeout=180s
ok "KubeRay operator is running"
kubectl get crd | grep ray.io
