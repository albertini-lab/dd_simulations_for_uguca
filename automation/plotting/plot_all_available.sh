#!/bin/bash
set -u
set -o pipefail

PLOT_SCRIPT="./Plot_BaselineExp_Spacetime.py"
PLOT_ISOCHRONES_SCRIPT="./Plot_DcA_Slip_Isochrones.py" # Assuming we can adapt or run this
PLOT_DIR="./comparison_plots_sweep"
EXP_DIR="../../introsims/experimental_data_processed"

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

for (( i=0; i<TOTAL; i++ )); do
    RUN_PARAMS="${ALL_RUNS[$i]}"
    IFS='|' read -r ASP NCEN NT NS A_VAL B_VAL DC_VAL <<< "$RUN_PARAMS"
    
    ASP_BOOL=$(echo "$ASP" | cut -d'_' -f1)
    
    RUN_ID=$(printf "run_%03d" "$i")
    SIM_NAME="base_sweep_${RUN_ID}"
    OUT_DIR="./results_sweep/${SIM_NAME}"
    
    # We check if there's any file in the out dir, indicating it ran.
    if [ -d "$OUT_DIR" ]; then
        if [ "$(ls -A $OUT_DIR 2>/dev/null)" ]; then
            TAG="r${i}_asp${ASP_BOOL}_ns${NS}_a${A_VAL}_b${B_VAL}_dc${DC_VAL}"
            TITLE="Run ${i}: Asp=${ASP_BOOL}, a=${A_VAL}, b=${B_VAL}, Dc=${DC_VAL}"
            
            echo "Plotting available run: $SIM_NAME"
            python3 "$PLOT_SCRIPT" \
                --baseline-dir "$OUT_DIR" \
                --exp-dir "$EXP_DIR" \
                --plot-dir "$PLOT_DIR" \
                --case-title "$TITLE" \
                --output-tag "$TAG" \
                --sim-name "$SIM_NAME"
        fi
    fi
done

echo "Done plotting available runs."
