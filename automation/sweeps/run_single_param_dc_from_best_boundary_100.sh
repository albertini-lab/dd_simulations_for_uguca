#!/bin/bash
set -u
set -o pipefail

MODE="${1:-run}"
CONFIG="/Users/joshmcneely/uguca/build/simulations/automation/sweeps/single_param_dc_from_best_boundary_100.json"
TOOL="/Users/joshmcneely/uguca/build/simulations/automation/sweeps/sweep_configurable.py"

case "$MODE" in
  dry-run)
    cd /Users/joshmcneely/uguca/build/simulations
    python3 "$TOOL" --config "$CONFIG" --dry-run
    ;;
  run)
    cd /Users/joshmcneely/uguca/build/simulations
    python3 "$TOOL" --config "$CONFIG" --run
    ;;
  *)
    echo "Usage: $0 [dry-run|run]"
    exit 1
    ;;
esac
