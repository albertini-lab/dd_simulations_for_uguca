#!/bin/bash
set -u
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

PLOT_SCRIPT="./Plot_BaselineExp_Spacetime.py"
PLOT_DIR="./comparison_plots_sweep"

if [ -d "/Users/joshmcneely/introsims/simulation_outputs/interface-restart" ]; then
    EXP_RESTART_DIR="/Users/joshmcneely/introsims/simulation_outputs/interface-restart"
elif [ -d "../../introsims/simulation_outputs/interface-restart" ]; then
    EXP_RESTART_DIR="../../introsims/simulation_outputs/interface-restart"
else
    echo "Experimental restart data directory not found."
    exit 1
fi

mkdir -p "$PLOT_DIR"

ASPERITY_CONFIGS=("False_0.0" "True_0.2" "True_0.4")
NUC_CENTERS=("0.65" "1.0")
NUC_TIMES=("0.0025" "0.0029" "0.0033" "0.00375")
NUC_SIZES=("0.1" "0.2")
A_VALS=("0.011" "0.015")
B_VALS=("0.014" "0.018")
DC_VALS=("1.0e-06" "1.6e-06")

declare -a ALL_RUNS
for asp in "${ASPERITY_CONFIGS[@]}"; do
for ncen in "${NUC_CENTERS[@]}"; do
for nt in "${NUC_TIMES[@]}"; do
for ns in "${NUC_SIZES[@]}"; do
for a in "${A_VALS[@]}"; do
for b in "${B_VALS[@]}"; do
for dc in "${DC_VALS[@]}"; do
    ALL_RUNS+=("${asp}|${ncen}|${nt}|${ns}|${a}|${b}|${dc}")
done
done
done
done
done
done
done

TOTAL=${#ALL_RUNS[@]}

echo "Starting to plot spacetime graphs for all available runs in results_sweep..."

for (( i=0; i<TOTAL; i++ )); do
    RUN_PARAMS="${ALL_RUNS[$i]}"
    IFS='|' read -r ASP NCEN NT NS A_VAL B_VAL DC_VAL <<< "$RUN_PARAMS"
    
    ASP_BOOL=$(echo "$ASP" | cut -d'_' -f1)
    
    RUN_ID=$(printf "run_%03d" "$i")
    SIM_NAME="base_sweep_${RUN_ID}"
    OUT_DIR="./results_sweep/${SIM_NAME}"
    
    # Check if the output directory exists
    if [ -d "$OUT_DIR" ]; then
        TAG="r${i}_asp${ASP_BOOL}_ns${NS}_a${A_VAL}_b${B_VAL}_dc${DC_VAL}"
        TITLE="Run ${i}: Asp=${ASP_BOOL}, a=${A_VAL}, b=${B_VAL}, Dc=${DC_VAL}"
        
        echo "Plotting available run: $SIM_NAME"
        python3 "$PLOT_SCRIPT" \
            --baseline-dir "$OUT_DIR" \
            --case-label "$TITLE" \
            --output-tag "$TAG" \
            --plot-dir "$PLOT_DIR" \
            --exp-restart-dir "$EXP_RESTART_DIR" \
            --baseline-sim-name "$SIM_NAME" \
            --nb-nodes 512 \
            --domain-length 6.0 \
            --roi-min 0.0 \
            --roi-max 3.05
    fi
done

echo "Done plotting spacetime available runs."