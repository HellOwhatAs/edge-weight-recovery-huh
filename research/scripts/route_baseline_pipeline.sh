#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
WORKSPACE=$(CDPATH= cd -- "$SCRIPT_DIR/../.." && pwd)
RUNNER="$SCRIPT_DIR/run_route_baseline_pipeline.py"
CONFIG="$WORKSPACE/research/experiments/route_baselines_full_test_20260719/pipeline.json"

usage() {
  printf '%s\n' \
    "Usage: $0 validate|smoke|full|start|status|verify|log|stop [arguments...]" \
    "" \
    "  validate check the frozen pipeline configuration without running tasks" \
    "  smoke    run CUDA preflight and all seven 500-route regression gates" \
    "  full     run/verify smoke, then the sequential full-test pipeline" \
    "  start    launch the full-test pipeline as a detached user service" \
    "  status   show stage, device, completed samples, elapsed time and ETA" \
    "  verify   status plus deep SHA-256 verification of eligible receipts" \
    "  log      tail the current log; pass --task ID or --lines N if desired" \
    "  stop     gracefully stop; start again later to resume from receipts/chunks"
}

if (( $# == 0 )); then
  usage
  exit 2
fi

operation=$1
shift
case "$operation" in
  validate)
    exec python3 "$RUNNER" --config "$CONFIG" validate "$@"
    ;;
  smoke)
    exec python3 "$RUNNER" --config "$CONFIG" run --profile smoke "$@"
    ;;
  full)
    exec python3 "$RUNNER" --config "$CONFIG" run --profile full "$@"
    ;;
  start)
    exec python3 "$RUNNER" --config "$CONFIG" launch --profile full "$@"
    ;;
  status)
    exec python3 "$RUNNER" --config "$CONFIG" status "$@"
    ;;
  verify)
    exec python3 "$RUNNER" --config "$CONFIG" status --verify "$@"
    ;;
  log)
    exec python3 "$RUNNER" --config "$CONFIG" log "$@"
    ;;
  stop)
    exec python3 "$RUNNER" --config "$CONFIG" stop "$@"
    ;;
  -h|--help|help)
    usage
    ;;
  *)
    printf 'Unknown operation: %s\n' "$operation" >&2
    usage >&2
    exit 2
    ;;
esac
