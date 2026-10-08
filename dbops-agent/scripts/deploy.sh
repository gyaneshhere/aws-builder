#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$ROOT"
[[ -f agentcore/agentcore.json ]] || { echo 'Run ./scripts/bootstrap_from_aws.sh first.' >&2; exit 1; }
agentcore deploy
agentcore status
