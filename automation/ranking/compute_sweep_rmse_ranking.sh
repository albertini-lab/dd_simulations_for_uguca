#!/bin/bash
# compute_sweep_rmse_ranking.sh
# Compute RMSE for all baseline sweep cases and rank them by error

set -u
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

RESULTS_DIR="./results_sweep"
RMSE_SCRIPT="./compute_baseline_vs_exp_rmse.py"
EXP_RESTART_DIR="/Users/joshmcneely/introsims/simulation_outputs/interface-restart"
OUTPUT_RANKING="./sweep_rmse_ranking.txt"

# Generate list of all baseline sweep directories
echo "Scanning for baseline sweep results..."
BASE_DIRS=($(find "$RESULTS_DIR" -maxdepth 1 -type d -name "base_sweep_run_*" | sort))

if [ ${#BASE_DIRS[@]} -eq 0 ]; then
    echo "[ERROR] No baseline sweep directories found in $RESULTS_DIR"
    exit 1
fi

TOTAL=${#BASE_DIRS[@]}
echo "Found $TOTAL baseline cases"
echo ""

# Temporary file for results
TEMP_RESULTS=$(mktemp)
trap "rm -f $TEMP_RESULTS" EXIT

# Compute RMSE for each case
echo "Computing RMSE for all cases..."
COUNT=0

for BASE_DIR in "${BASE_DIRS[@]}"; do
    COUNT=$((COUNT + 1))
    CASE_NAME=$(basename "$BASE_DIR")
    SIM_NAME="$CASE_NAME"
    
    echo -n "[$COUNT/$TOTAL] $CASE_NAME ... "
    
    # Extract parameters from directory name
    # Example: base_sweep_run_000
    RUN_NUM=$(echo "$CASE_NAME" | sed 's/.*_run_//')
    
    # Run RMSE calculation and capture output
    RMSE_OUTPUT=$(python3 "$RMSE_SCRIPT" \
        --baseline-dir "$BASE_DIR" \
        --baseline-sim-name "$SIM_NAME" \
        --exp-restart-dir "$EXP_RESTART_DIR" \
        --nb-nodes 512 \
        --domain-length 6.0 2>&1)
    
    # Extract mean relative error (look for "Average relative error" line)
    MEAN_ERROR=$(echo "$RMSE_OUTPUT" | grep "Average relative error" | awk '{print $NF}' | tr -d '(%')
    
    if [ -z "$MEAN_ERROR" ]; then
        echo "FAILED"
        echo "$RMSE_OUTPUT" >> "$TEMP_RESULTS.errors"
    else
        echo "RMSE=$MEAN_ERROR"
        echo "$MEAN_ERROR|$CASE_NAME" >> "$TEMP_RESULTS"
    fi
done

echo ""
echo "Ranking results..."
echo ""

if [ ! -f "$TEMP_RESULTS" ] || [ ! -s "$TEMP_RESULTS" ]; then
    echo "[ERROR] No successful RMSE calculations"
    exit 1
fi

# Sort by RMSE value (ascending = best first)
{
    echo "================================================================"
    echo "BASELINE SWEEP RMSE RANKING (Best → Worst)"
    echo "================================================================"
    echo "Rank | RMSE (%)      | Case Name"
    echo "-----+---------------+----------------------------------------"
    
    sort -n "$TEMP_RESULTS" | awk -v count=1 'BEGIN {FS="|"} {printf "%4d | %13s | %s\n", count, $1, $2; count++}'
    
    echo "================================================================"
} | tee "$OUTPUT_RANKING"

echo ""
echo "Ranking saved to: $OUTPUT_RANKING"
