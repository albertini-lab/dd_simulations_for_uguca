#!/bin/bash
# run_all_spacetime_plots.sh
# Generates spacetime heatmap plots for all discovered Dc/a parameter sweep cases

echo "=========================================="
echo "Spacetime Heatmap Batch Generator"
echo "=========================================="
echo ""

# Find all data-driven result directories
dd_dirs=$(ls -d results_dd_Dc*_a* 2>/dev/null)

if [ -z "$dd_dirs" ]; then
    echo "[ERROR] No results_dd_Dc*_a* directories found"
    exit 1
fi

# Count total
total=$(echo "$dd_dirs" | wc -l | tr -d ' ')
echo "[INFO] Found $total Dc/a sweep cases"
echo ""

# Process each case
count=0
success=0
failed=0

for dd_dir in $dd_dirs; do
    count=$((count + 1))
    
    # Extract Dc and a values from directory name
    # Example: results_dd_Dc1e-5_a2e-5 -> Dc=1e-5, a=2e-5
    if [[ $dd_dir =~ results_dd_Dc([0-9eE+\-.]+)_a([0-9eE+\-.]+) ]]; then
        dc_val="${BASH_REMATCH[1]}"
        a_val="${BASH_REMATCH[2]}"
        
        # Check if corresponding baseline directory exists
        baseline_dir="results_baseline_Dc${dc_val}_a${a_val}"
        if [ ! -d "$baseline_dir" ]; then
            echo "[$count/$total] SKIP: $dd_dir (no matching baseline)"
            failed=$((failed + 1))
            continue
        fi
        
        echo "[$count/$total] Processing: Dc=$dc_val, a=$a_val"
        
        # Run the plotter
        if python3 Plot_DcA_Spacetime.py "$dc_val" "$a_val" 2>&1 | grep -q "SUCCESS"; then
            success=$((success + 1))
            echo "           ✓ Success"
        else
            failed=$((failed + 1))
            echo "           ✗ Failed"
        fi
        echo ""
    else
        echo "[$count/$total] SKIP: Could not parse $dd_dir"
        failed=$((failed + 1))
    fi
done

echo "=========================================="
echo "Batch Processing Complete"
echo "=========================================="
echo "Total cases:     $total"
echo "Successful:      $success"
echo "Failed/Skipped:  $failed"
echo "=========================================="
