#!/usr/bin/env bash
# Submits hello_ray.py as a Ray job from the head pod and streams its output.
source "$(dirname "$0")/../lib/common.sh"
here="$(cd "$(dirname "$0")" && pwd)"
info "Submitting hello_ray.py"
submit_job "$here" '{}' wait python hello_ray.py
