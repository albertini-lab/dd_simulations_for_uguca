#!/bin/bash
# rank_baseline_focused_sweep.sh
# Extracts RMSE from completed focused baseline sweep runs and outputs ranking

set -u

SIMULATIONS_DIR="/Users/joshmcneely/uguca/build/simulations"
RMSE_SCRIPT="${SIMULATIONS_DIR}/automation/ranking/compute_baseline_vs_exp_rmse.py"
SWEEP_ROOT="${SIMULATIONS_DIR}/sweep_outputs/focused_baseline_ndtau_dc"
CASE_BASE_DIR="${SWEEP_ROOT}/cases"
PLOT_DIR="${SWEEP_ROOT}/comparison_plots"
RMSE_RANK_FILE="${PLOT_DIR}/sweep_baseline_param_rmse_ranking_all.txt"

EXP_BASE_DIR="/Users/joshmcneely/introsims/simulation_outputs"
EXP_RESTART_NAME="interface-restart"

TEMP_RMSE_FILE="/tmp/focused_baseline_rmse_$$.tsv"

if [ ! -f "$RMSE_SCRIPT" ]; then
  echo "[FATAL] RMSE script not found: $RMSE_SCRIPT"
  exit 1
fi

cd "$SIMULATIONS_DIR"
mkdir -p "$PLOT_DIR"

# Find all focused baseline run directories
RESULTS_DIRS=($(find "$CASE_BASE_DIR" -maxdepth 1 -type d -name "results_baseline_ndtau*_dc*" | sort))

if [ ${#RESULTS_DIRS[@]} -eq 0 ]; then
  echo "[ERROR] No focused baseline run directories found (pattern: results_baseline_ndtau*_dc*)"
  exit 1
fi

echo "Found ${#RESULTS_DIRS[@]} focused baseline runs"
echo "Extracting RMSE from each run..."
echo ""

for dir in "${RESULTS_DIRS[@]}"; do
  case_name=$(basename "$dir")
  
  echo -n "  $case_name ... "
  
  if rmse_output=$(python3 "$RMSE_SCRIPT" \
    --baseline-dir "$dir" \
    --baseline-sim-name "local_baseline_run" \
    --exp-base-dir "$EXP_BASE_DIR" \
    --exp-restart-dir "$EXP_RESTART_NAME" \
    --nb-nodes 512 \
    --domain-length 6.0 2>&1); then
    
    if rmse_percent=$(printf "%s\n" "$rmse_output" | grep "Average relative error" | sed 's/^.*(\([0-9.eE+\-]*\)%)/\1/' | tail -1); then
      if [ -n "$rmse_percent" ] && [ "$rmse_percent" != "" ]; then
        echo "$rmse_percent"
        echo -e "${rmse_percent}\t${case_name}\t${dir}" >> "$TEMP_RMSE_FILE"
      else
        echo "FAILED (extraction)"
      fi
    else
      echo "FAILED (parsing)"
    fi
  else
    echo "FAILED (computation)"
  fi
done

echo ""
echo "================================================================"
echo "FOCUSED BASELINE SWEEP RMSE RANKING (Best -> Worst)"
echo "================================================================"

if [ -s "$TEMP_RMSE_FILE" ]; then
  {
    echo "Rank | RMSE (%)      | RUN_LABEL                  | OUTPUT_DIR"
    echo "-----+---------------+----------------------------+--------------------------------"
    sort -g "$TEMP_RMSE_FILE" | awk -F '\t' '{printf "%4d | %13s | %-26s | %s\n", NR, $1, $2, $3}'
    echo "================================================================"
  } | tee "$RMSE_RANK_FILE"
  
  echo ""
  echo "Ranking saved to: $RMSE_RANK_FILE"
  rm -f "$TEMP_RMSE_FILE"
else
  echo "[ERROR] No valid RMSE values extracted"
  rm -f "$TEMP_RMSE_FILE"
  exit 1
fi
