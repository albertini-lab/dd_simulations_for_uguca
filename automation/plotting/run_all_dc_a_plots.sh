#!/bin/bash
set -u
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

if [ -x "$SCRIPT_DIR/../../.venv/bin/python" ]; then
  PYTHON_EXEC="$SCRIPT_DIR/../../.venv/bin/python"
elif command -v python3 >/dev/null 2>&1; then
  PYTHON_EXEC="$(command -v python3)"
else
  echo "[FATAL] No Python interpreter found (.venv/bin/python or python3)."
  exit 1
fi

PLOT_SCRIPTS=(
  "Plot_Overlay_DcA_Viridis.py"
  "Plot_DcA_RMSE.py"
  "Plot_DcA_Slip_Isochrones.py"
)

TOTAL=${#PLOT_SCRIPTS[@]}
FAILED=0

echo "================================================================"
echo "Running Dc/a plotting pipeline"
echo "Directory : $SCRIPT_DIR"
echo "Python    : $PYTHON_EXEC"
echo "Scripts   : $TOTAL"
echo "================================================================"

for i in "${!PLOT_SCRIPTS[@]}"; do
  script_name="${PLOT_SCRIPTS[$i]}"
  script_path="$SCRIPT_DIR/$script_name"
  run_num=$((i + 1))

  echo ""
  echo "[$run_num/$TOTAL] Running $script_name"

  if [ ! -f "$script_path" ]; then
    echo "[ERROR] Missing script: $script_path"
    FAILED=$((FAILED + 1))
    continue
  fi

  "$PYTHON_EXEC" "$script_path"
  exit_code=$?

  if [ "$exit_code" -ne 0 ]; then
    echo "[ERROR] $script_name failed with exit code $exit_code"
    FAILED=$((FAILED + 1))
  else
    echo "[OK] $script_name completed"
  fi
done

echo ""
echo "================================================================"
if [ "$FAILED" -eq 0 ]; then
  echo "All plotting scripts completed successfully."
  echo "Outputs are in: $SCRIPT_DIR/comparison_plots"
  echo "================================================================"
  exit 0
else
  echo "$FAILED plotting script(s) failed."
  echo "Check terminal output above for details."
  echo "================================================================"
  exit 1
fi
