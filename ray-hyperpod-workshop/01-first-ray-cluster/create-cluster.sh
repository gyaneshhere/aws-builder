#!/usr/bin/env bash
# Renders raycluster.yaml.tpl, applies it, and waits until the head and all workers are Ready.
#   ./create-cluster.sh            create or update
#   ./create-cluster.sh --dry-run  print the rendered manifest only
source "$(dirname "$0")/../lib/common.sh"
here="$(dirname "$0")"

if [[ "${1:-}" == "--dry-run" ]]; then render "$here/raycluster.yaml.tpl"; exit 0; fi
require kubectl

info "Applying RayCluster ${NAMESPACE}/${RAY_CLUSTER_NAME} (${GPU_WORKERS} GPU workers, image ${RAY_IMAGE})"
render "$here/raycluster.yaml.tpl" | kubectl apply -f -

info "Waiting for pods (the first image pull takes a few minutes)"
want=$((GPU_WORKERS + 1))
for i in $(seq 1 90); do
  ready=$(kc get pods -l "ray.io/cluster=${RAY_CLUSTER_NAME}" \
      -o jsonpath='{range .items[*]}{.status.containerStatuses[0].ready}{"\n"}{end}' 2>/dev/null | grep -c true || true)
  printf "\r  %s/%s pods ready (%ss)" "$ready" "$want" $((i * 10))
  [[ "$ready" -ge "$want" ]] && { echo; break; }
  sleep 10
done
echo
kc get pods -l "ray.io/cluster=${RAY_CLUSTER_NAME}" -o wide
[[ "${ready:-0}" -ge "$want" ]] || die "Cluster not ready after 15 minutes: see 'Troubleshooting' in 01-first-ray-cluster/README.md"

info "Ray's own view of the cluster"
kc exec "$(head_pod)" -- ray status
ok "RayCluster is ready"
