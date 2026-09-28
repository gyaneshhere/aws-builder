#!/usr/bin/env bash
# Breaks one Ray worker while training runs, so you can watch HyperPod + Ray recover.
#   ./inject-fault.sh                reboot the node under a worker (HyperPod automatic node recovery)
#   ./inject-fault.sh --mode pod     delete a worker pod instead (works on any cluster, no node impact)
#   ./inject-fault.sh --yes          skip the confirmation prompt
#
# It never touches the node that runs the Ray head: losing the head ends the whole cluster.
source "$(dirname "$0")/../lib/common.sh"
require kubectl
MODE=reboot; YES=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode) MODE="$2"; shift 2 ;;
    --yes|-y) YES=1; shift ;;
    *) die "unknown argument $1" ;;
  esac
done
[[ "$MODE" == "reboot" || "$MODE" == "pod" ]] || die "--mode must be reboot or pod"

head="$(require_head)"
head_node=$(kc get pod "$head" -o jsonpath='{.spec.nodeName}')

# Pick a Running worker pod that is NOT on the head's node.
read -r victim_pod victim_node < <(kc get pods -l "ray.io/cluster=${RAY_CLUSTER_NAME},ray.io/node-type=worker" \
  -o jsonpath='{range .items[?(@.status.phase=="Running")]}{.metadata.name}{" "}{.spec.nodeName}{"\n"}{end}' \
  | awk -v h="$head_node" '$2!=h' | head -1) || true
[[ -n "${victim_pod:-}" ]] || die "no worker pod found on a node other than the head's (${head_node}). Use --mode pod, or add a GPU node."

info "Target"
echo "  worker pod : ${victim_pod}"
echo "  node       : ${victim_node}"
echo "  head node  : ${head_node} (left alone)"

if [[ "$MODE" == "reboot" ]]; then
  if [[ -n "$HYPERPOD_CLUSTER_NAME" ]]; then
    rec=$(aws sagemaker describe-cluster --cluster-name "$HYPERPOD_CLUSTER_NAME" --region "$AWS_REGION" \
          --query NodeRecovery --output text 2>/dev/null || echo unknown)
    [[ "$rec" == "Automatic" ]] || die "NodeRecovery is '${rec}'. Enable Automatic node recovery, or use --mode pod."
  else
    warn "HYPERPOD_CLUSTER_NAME not set; cannot confirm NodeRecovery=Automatic"
  fi
  echo
  echo "  This labels the node UnschedulablePendingReboot. HyperPod reboots it in a few minutes;"
  echo "  other pods on that node are interrupted too."
fi

if [[ "$YES" -ne 1 ]]; then
  read -r -p "  Proceed? [y/N] " ans
  [[ "$ans" == "y" || "$ans" == "Y" ]] || { echo "aborted"; exit 0; }
fi

date -u +"%H:%M:%S fault injected" > /tmp/ray-workshop-fault-time
if [[ "$MODE" == "reboot" ]]; then
  kubectl label node "$victim_node" sagemaker.amazonaws.com/node-health-status=UnschedulablePendingReboot --overwrite
  ok "Node ${victim_node} marked for reboot at $(cut -d' ' -f1 /tmp/ray-workshop-fault-time) UTC"
else
  kc delete pod "$victim_pod" --grace-period=0 --force
  ok "Worker pod ${victim_pod} deleted at $(cut -d' ' -f1 /tmp/ray-workshop-fault-time) UTC"
fi
echo
echo "  Watch recovery:      ./03-resilience/watch-recovery.sh"
echo "  Watch training logs: ./02-distributed-training/job.sh logs"
