# shellcheck shell=bash
# Shared helpers. Sourced by every script; not meant to be run directly.
set -euo pipefail

WORKSHOP_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=../env.sh
source "${WORKSHOP_ROOT}/env.sh"

if [[ -t 1 ]]; then G=$'\e[32m'; Y=$'\e[33m'; R=$'\e[31m'; B=$'\e[1m'; N=$'\e[0m'; else G=; Y=; R=; B=; N=; fi
info() { echo "${B}==>${N} $*"; }
ok()   { echo "  ${G}✔${N} $*"; }
warn() { echo "  ${Y}!${N} $*"; }
fail() { echo "  ${R}✘${N} $*" >&2; }
die()  { fail "$*"; exit 1; }

require() { for c in "$@"; do command -v "$c" >/dev/null 2>&1 || die "'$c' is not installed (see 00-setup/README.md)"; done; }

# Render a template: replaces ${VAR} with environment values (envsubst if present, else python3).
render() {
  if command -v envsubst >/dev/null 2>&1; then envsubst < "$1"
  else python3 -c 'import os,sys; sys.stdout.write(os.path.expandvars(open(sys.argv[1]).read()))' "$1"; fi
}

kc() { kubectl -n "${NAMESPACE}" "$@"; }

head_pod() {
  kc get pods -l "ray.io/cluster=${RAY_CLUSTER_NAME},ray.io/node-type=head" \
     -o jsonpath='{.items[0].metadata.name}' 2>/dev/null || true
}

require_head() {
  local p; p="$(head_pod)"
  [[ -n "$p" ]] || die "Ray head pod not found. Run 01-first-ray-cluster/create-cluster.sh first."
  echo "$p"
}

# Copy a local directory into the head pod and submit it as a Ray job from there.
# No local Ray install or port-forward is needed.
#   submit_job <local_dir> <runtime_env_json> <wait|nowait> <entrypoint...>
submit_job() {
  local dir="$1" renv="$2" mode="$3"; shift 3
  local head; head="$(require_head)"
  local dest
  dest="/tmp/workshop-$(basename "$dir")-$(date +%s)"
  kc exec "$head" -- mkdir -p "$dest"
  tar -C "$dir" --exclude='__pycache__' -cf - . | kc exec -i "$head" -- tar -xf - -C "$dest"
  local flags=(--address http://127.0.0.1:8265 --working-dir . --runtime-env-json "$renv")
  [[ "$mode" == "nowait" ]] && flags+=(--no-wait)
  kc exec "$head" -- bash -c "cd '$dest' && ray job submit $(printf '%q ' "${flags[@]}") -- $(printf '%q ' "$@")"
}
