#!/usr/bin/env bash
# Small wrapper around the Ray Jobs CLI, run inside the head pod.
#   ./job.sh logs   [job_id]   follow logs (default: last submitted job)
#   ./job.sh status [job_id]
#   ./job.sh stop   [job_id]
#   ./job.sh list
source "$(dirname "$0")/../lib/common.sh"
cmd="${1:-status}"; job="${2:-$(cat /tmp/ray-workshop-last-job 2>/dev/null || true)}"
head="$(require_head)"
addr=(--address http://127.0.0.1:8265)
case "$cmd" in
  logs)   [[ -n "$job" ]] || die "no job id"; kc exec -it "$head" -- ray job logs "${addr[@]}" -f "$job" ;;
  status) [[ -n "$job" ]] || die "no job id"; kc exec "$head" -- ray job status "${addr[@]}" "$job" ;;
  stop)   [[ -n "$job" ]] || die "no job id"; kc exec "$head" -- ray job stop "${addr[@]}" "$job" ;;
  list)   kc exec "$head" -- ray job list "${addr[@]}" ;;
  *) die "usage: $0 logs|status|stop|list [job_id]" ;;
esac
