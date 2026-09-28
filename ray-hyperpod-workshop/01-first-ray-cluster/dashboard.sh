#!/usr/bin/env bash
# Opens local tunnels to the Ray head: dashboard on :8265 and Ray Serve on :8000.
#   ./dashboard.sh start | stop | status
source "$(dirname "$0")/../lib/common.sh"
require kubectl
PIDFILE="/tmp/ray-workshop-port-forward.pid"
svc="${RAY_CLUSTER_NAME}-head-svc"

case "${1:-start}" in
  start)
    "$0" stop >/dev/null 2>&1 || true
    kc get svc "$svc" >/dev/null || die "Service ${svc} not found. Is the RayCluster running?"
    kc port-forward "svc/${svc}" 8265:8265 8000:8000 >/tmp/ray-workshop-port-forward.log 2>&1 &
    echo $! > "$PIDFILE"
    sleep 3
    kill -0 "$(cat "$PIDFILE")" 2>/dev/null || die "port-forward failed: $(cat /tmp/ray-workshop-port-forward.log)"
    ok "Ray dashboard: http://localhost:8265"
    ok "Ray Serve:     http://localhost:8000  (used in Module 4)"
    echo "  In SageMaker Code Editor, open the dashboard with the Ports panel or the proxy URL .../proxy/8265/" ;;
  stop)
    [[ -f "$PIDFILE" ]] && kill "$(cat "$PIDFILE")" 2>/dev/null && rm -f "$PIDFILE" && ok "tunnels closed" || ok "no tunnels running" ;;
  status)
    [[ -f "$PIDFILE" ]] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null && ok "tunnels running (pid $(cat "$PIDFILE"))" || warn "tunnels not running" ;;
  *) die "usage: $0 start|stop|status" ;;
esac
