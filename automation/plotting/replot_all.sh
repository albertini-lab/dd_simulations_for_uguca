#!/bin/bash

cd /Users/joshmcneely/uguca/build/simulations

echo "=========================================="
echo "  REPLOTTING ALL SWEEP RESULTS"
echo "=========================================="

# Loop through all archived result directories
for archived_dir in results_w*; do
    if [ -d "$archived_dir" ]; then
        w_value=$(echo "$archived_dir" | sed 's/results_w//')
        echo ""
        echo "Replotting W = $w_value..."
        
        # Remove old results directory and copy archived data
        rm -rf results
        cp -r "$archived_dir" results
        
        # Run the plot script
        python3 Plot_Local_Debug.py
        
        # Move the generated plot to the archived directory with unique name
        if [ -f "comparison_plots/diagnostic_comparative.png" ]; then
            mkdir -p "$archived_dir/comparison_plots"
            cp "comparison_plots/diagnostic_comparative.png" "$archived_dir/comparison_plots/diagnostic_w${w_value}.png"
            echo "   -> Plot saved to $archived_dir/comparison_plots/diagnostic_w${w_value}.png"
        fi
    fi
done

# Clean up temporary results directory
rm -rf results comparison_plots

echo ""
echo "=========================================="
echo "  REPLOTTING COMPLETE!"
echo "  Baseline included if available."
echo "=========================================="