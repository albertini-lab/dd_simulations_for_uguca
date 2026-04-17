#!/bin/bash
set -euo pipefail

# Run all comparison plots for two completed DD runs:
# 1) direct experimental restart input
# 2) spline experimental restart input
#
# By default this runs each DD case against two baselines:
# - nu=0.25 baseline in ./results_baseline
# - nu=0.19 baseline in ./results_baseline_local_compare
#
# Defaults assume this script is run from (or lives in) build/simulations.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

EXP_BASE_DIR="/Users/joshmcneely/introsims/simulation_outputs"
PLOT_DIR="./comparison_plots"

BASELINE_025_DIR="./results_baseline"
BASELINE_019_DIR="./results_baseline_local_compare"
ENABLE_BASELINE_019="true"
DIRECT_DD_DIR="./results_dd_mcklaskey_debug_direct"
SPLINE_DD_DIR="./results_dd_mcklaskey_debug_spline"

DIRECT_RESTART="$EXP_BASE_DIR/mcklaskey_debug_direct-restart"
SPLINE_RESTART="$EXP_BASE_DIR/mcklaskey_debug_spline-restart"

usage() {
    cat <<'EOF'
Usage:
  ./run_direct_spline_plots.sh [options]

Options:
    --baseline-dir <path>      Baseline nu=0.25 results directory (alias)
    --baseline-025-dir <path>  Baseline nu=0.25 results directory
    --baseline-019-dir <path>  Baseline nu=0.19 results directory
    --no-baseline-019          Skip nu=0.19 baseline plots
  --direct-dd-dir <path>     Direct DD results directory
  --spline-dd-dir <path>     Spline DD results directory
  --direct-restart <path>    Direct experiment restart directory
  --spline-restart <path>    Spline experiment restart directory
  --plot-dir <path>          Output plot directory
  -h, --help                 Show this help and exit

Example:
  ./run_direct_spline_plots.sh \
        --baseline-025-dir ./results_baseline \
        --baseline-019-dir ./results_baseline_local_compare \
    --direct-dd-dir ./results_dd_direct \
    --spline-dd-dir ./results_dd_spline
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --baseline-dir)
            BASELINE_025_DIR="$2"
            shift 2
            ;;
        --baseline-025-dir)
            BASELINE_025_DIR="$2"
            shift 2
            ;;
        --baseline-019-dir)
            BASELINE_019_DIR="$2"
            shift 2
            ;;
        --no-baseline-019)
            ENABLE_BASELINE_019="false"
            shift
            ;;
        --direct-dd-dir)
            DIRECT_DD_DIR="$2"
            shift 2
            ;;
        --spline-dd-dir)
            SPLINE_DD_DIR="$2"
            shift 2
            ;;
        --direct-restart)
            DIRECT_RESTART="$2"
            shift 2
            ;;
        --spline-restart)
            SPLINE_RESTART="$2"
            shift 2
            ;;
        --plot-dir)
            PLOT_DIR="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "[ERROR] Unknown option: $1"
            usage
            exit 1
            ;;
    esac
done

check_dir() {
    local path="$1"
    local label="$2"
    if [[ ! -d "$path" ]]; then
        echo "[ERROR] Missing ${label}: ${path}"
        exit 1
    fi
}

run_case() {
    local tag="$1"
    local case_label="$2"
    local dd_dir="$3"
    local exp_restart="$4"
    local baseline_dir="$5"
    local baseline_tag="$6"
    local full_tag="${tag}_${baseline_tag}"
    local full_label="${case_label} (${baseline_tag})"

    echo ""
    echo "=========================================="
    echo "[INFO] Plotting case: ${full_label}"
    echo "[INFO] DD dir: ${dd_dir}"
    echo "[INFO] Baseline dir: ${baseline_dir}"
    echo "[INFO] Exp restart: ${exp_restart}"
    echo "=========================================="

    python3 Plot_DcA_Spacetime.py \
        --dd-dir "$dd_dir" \
        --baseline-dir "$baseline_dir" \
        --exp-restart-dir "$exp_restart" \
        --case-label "$full_label" \
        --output-tag "$full_tag" \
        --plot-dir "$PLOT_DIR"

    python3 Plot_DcA_Slip_Isochrones.py \
        --dd-dir "$dd_dir" \
        --baseline-dir "$baseline_dir" \
        --exp-restart-dir "$exp_restart" \
        --case-label "$full_label" \
        --output-prefix "$full_tag" \
        --plot-experiment-overlay \
        --plot-dir "$PLOT_DIR"

    python3 Plot_DcA_RMSE.py \
        --dd-dir "$dd_dir" \
        --baseline-dir "$baseline_dir" \
        --exp-restart-dir "$exp_restart" \
        --case-label "$full_label" \
        --output-prefix "$full_tag" \
        --plot-dir "$PLOT_DIR"
}

run_all_for_baseline() {
    local baseline_dir="$1"
    local baseline_tag="$2"

    run_case "direct" "mcklaskey_debug_direct" "$DIRECT_DD_DIR" "$DIRECT_RESTART" "$baseline_dir" "$baseline_tag"
    run_case "spline" "mcklaskey_debug_spline" "$SPLINE_DD_DIR" "$SPLINE_RESTART" "$baseline_dir" "$baseline_tag"
}

run_combined_overlay() {
    python3 Plot_Overlay_DirectSpline_Baselines.py \
        --dd-direct-dir "$DIRECT_DD_DIR" \
        --dd-spline-dir "$SPLINE_DD_DIR" \
        --baseline-025-dir "$BASELINE_025_DIR" \
        --baseline-019-dir "$BASELINE_019_DIR" \
        --exp-direct-restart-dir "$DIRECT_RESTART" \
        --exp-spline-restart-dir "$SPLINE_RESTART" \
        --plot-dir "$PLOT_DIR"
}

echo "=========================================="
echo "Direct + Spline plot batch"
echo "=========================================="

check_dir "$BASELINE_025_DIR" "baseline nu=0.25 results directory"
if [[ "$ENABLE_BASELINE_019" == "true" ]]; then
    check_dir "$BASELINE_019_DIR" "baseline nu=0.19 results directory"
fi
check_dir "$DIRECT_DD_DIR" "direct DD results directory"
check_dir "$SPLINE_DD_DIR" "spline DD results directory"
check_dir "$DIRECT_RESTART" "direct experimental restart directory"
check_dir "$SPLINE_RESTART" "spline experimental restart directory"

mkdir -p "$PLOT_DIR"

run_all_for_baseline "$BASELINE_025_DIR" "nu025"
if [[ "$ENABLE_BASELINE_019" == "true" ]]; then
    run_all_for_baseline "$BASELINE_019_DIR" "nu019"
    run_combined_overlay
else
    echo "[INFO] Skipping combined direct/spline overlay because --no-baseline-019 is enabled."
fi

echo ""
echo "=========================================="
echo "[SUCCESS] All plotting complete"
echo "[SUCCESS] Output directory: ${PLOT_DIR}"
echo "=========================================="
