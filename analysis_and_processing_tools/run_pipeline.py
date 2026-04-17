#!/usr/bin/env python3

import subprocess
import sys
import os
import re

# --- ANSI Color Codes for Terminal Output ---
class Colors:
    RED = '\033[0;31m'
    GREEN = '\033[0;32m'
    BLUE = '\033[0;34m'
    YELLOW = '\033[1;33m'
    NC = '\033[0m'  # No Color

def print_color(color, message):
    """Prints a message in the specified color."""
    print(f"{color}{message}{Colors.NC}")

# --- Main Pipeline Function ---
def main():
    """
    Main function to execute the UGUCA testing and analysis pipeline.
    """
    print_color(Colors.BLUE, "========================================")
    print_color(Colors.BLUE, "    UGUCA Complete Pipeline Test (Python) ")
    print_color(Colors.BLUE, "========================================")

    # --- Configuration ---
    # This script uses the 'os.path' module for robust path handling.
    try:
        # Get the current working directory as a string.
        script_dir = os.getcwd()
        if not script_dir.endswith('analysis_and_processing_tools'):
            print_color(Colors.RED, f"ERROR: This script must be run from the 'analysis_and_processing_tools' directory.")
            print_color(Colors.YELLOW, f"You are currently in: {script_dir}")
            sys.exit(1)

        # Construct paths as strings using os.path.join.
        simulations_dir = os.path.abspath(os.path.join(script_dir, '..'))
        
        # Build directory paths - simulations run FROM build directory
        build_simulations_dir = "/Users/joshmcneely/uguca/build/simulations"
        
        # Define base names for simulations
        gt_sim_base_name = "time_int_opt_gt"
        # CORRECTED: This name must match the 'sim_name' inside the .in file
        dd_noisy_sim_base_name = "time_int_opt_deviated_dd_noisy_scenario"

        # Define input file names
        gt_input_file = "ground_truth.in"
        dd_noisy_input_file = "deviated_params_dd_noisy_scenario.in"

        # Define all paths as strings
        # Simulation executable and working directory
        sim_executable = "./time_int_opt"  # Run from build directory
        input_dir = os.path.join(simulations_dir, "input_files")
        plotting_tools_dir = os.path.join(simulations_dir, "plotting_tools")
        
        plot_script_spacetime = os.path.join(plotting_tools_dir, "time_int_opt_plot.py")
        plot_script_final_state = os.path.join(plotting_tools_dir, "time_int_opt_fig.py")
        process_script = os.path.join(script_dir, "process_ground_truth_displacement.py")
        mse_script = os.path.join(script_dir, "calculate_mse.py")

    except Exception as e:
        print_color(Colors.RED, f"ERROR: Failed to configure paths. Details: {e}")
        sys.exit(1)

    # --- Verification ---
    print_color(Colors.YELLOW, "\nVerifying script setup...")
    paths_to_check = {
        "Build simulations directory": build_simulations_dir,
        "Simulation executable": os.path.join(build_simulations_dir, "time_int_opt"),
        "Input directory": input_dir,
        "Processing script": process_script,
    }
    has_error = False
    for name, path in paths_to_check.items():
        if not os.path.exists(path):
            print_color(Colors.RED, f"ERROR: {name} not found at '{path}'")
            has_error = True
    if has_error:
        print_color(Colors.RED, "Please ensure the project is built and paths are correct.")
        sys.exit(1)
    print_color(Colors.GREEN, "✓ Setup verified. Starting pipeline.\n")

    # --- Step 1: Run Ground Truth Generation ---
    print_color(Colors.BLUE, "--- Step 1: Running Ground Truth Generation ---")
    try:
        # Run from build directory, using relative paths from the build simulations directory
        cmd = [sim_executable, f"{simulations_dir}/input_files/{gt_input_file}"]
        print(f"Executing from '{build_simulations_dir}': {' '.join(cmd)}")
        subprocess.run(cmd, cwd=build_simulations_dir, check=True, capture_output=True, text=True)
        print_color(Colors.GREEN, "✓ Ground truth simulation finished.\n")
    except subprocess.CalledProcessError as e:
        print_color(Colors.RED, f"ERROR: Ground truth simulation failed.")
        print_color(Colors.RED, f"Stderr:\n{e.stderr}")
        sys.exit(1)

    # --- Step 2: Process Ground Truth Data to Create Noisy Input ---
    print_color(Colors.BLUE, "--- Step 2: Processing ground truth data to create noisy input ---")
    try:
        noise_stddev = None
        with open(os.path.join(input_dir, dd_noisy_input_file)) as f:
            content = f.read()
            match = re.search(r"^\s*noise_stddev_sim\s*=\s*(\S+)", content, re.MULTILINE)
            if match:
                noise_stddev = match.group(1)
        
        if not noise_stddev:
            raise ValueError("Could not find 'noise_stddev_sim' in input file.")
            
        print(f"Found required noise standard deviation: {noise_stddev}")

        # The processing script needs to work with the build directory structure  
        # It should look for ground truth data in build/simulations/simulation_outputs/
        # and save noisy data there as well
        cmd = [
            "python3", process_script, gt_sim_base_name,
            "--add-noise", "--noise-stddev", noise_stddev,
            "--output-noisy-data"
        ]
        # Run the processing script from the build directory so it finds the correct simulation_outputs
        print(f"Running from '{build_simulations_dir}': {' '.join(cmd)}")
        print(f"Running from '{build_simulations_dir}': {' '.join(cmd)}")
        subprocess.run(cmd, cwd=build_simulations_dir, check=True, capture_output=True, text=True)
        print_color(Colors.GREEN, "✓ Noisy data generated and saved.\n")
    except (ValueError, subprocess.CalledProcessError) as e:
        print_color(Colors.RED, f"ERROR: Failed to process ground truth data.")
        if isinstance(e, subprocess.CalledProcessError):
            print_color(Colors.RED, f"Stderr:\n{e.stderr}")
        else:
            print_color(Colors.RED, str(e))
        sys.exit(1)

    # --- Step 3: Run Data-Driven Simulation with Noisy Data ---
    print_color(Colors.BLUE, "--- Step 3: Running Data-Driven Simulation with Noisy Data ---")
    try:
        # Run from build directory, using relative paths from the build simulations directory
        cmd = [sim_executable, f"{simulations_dir}/input_files/{dd_noisy_input_file}"]
        print(f"Executing from '{build_simulations_dir}': {' '.join(cmd)}")
        subprocess.run(cmd, cwd=build_simulations_dir, check=True, capture_output=True, text=True)
        print_color(Colors.GREEN, "✓ Data-driven simulation finished.\n")
    except subprocess.CalledProcessError as e:
        print_color(Colors.RED, f"ERROR: Data-driven simulation failed.")
        print_color(Colors.RED, f"Stderr:\n{e.stderr}")
        sys.exit(1)

    # --- Step 4: Generate Comparison Plots ---
    print_color(Colors.BLUE, "--- Step 4: Generating comparison plots ---")
    try:
        # Run plotting tools from build directory so they find the correct simulation_outputs
        cmd_final_state = ["python3", plot_script_final_state, gt_sim_base_name, dd_noisy_sim_base_name]
        print(f"Running from '{build_simulations_dir}': {' '.join(cmd_final_state)}")
        subprocess.run(cmd_final_state, cwd=build_simulations_dir, check=True)

        print_color(Colors.GREEN, "✓ Comparison plots generated in 'figures_and_plots/'.\n")
    except subprocess.CalledProcessError as e:
        print_color(Colors.RED, f"ERROR: Plot generation failed.")
        print_color(Colors.RED, f"Stderr:\n{e.stderr}")
        sys.exit(1)

    # --- Step 5: Calculate and Plot MSE ---
    print_color(Colors.BLUE, "--- Step 5: Calculating and plotting RMSE ---")
    try:
        # Run MSE calculation from build directory so it finds the correct simulation_outputs
        cmd = ["python3", mse_script, gt_sim_base_name, dd_noisy_sim_base_name]
        print(f"Running from '{build_simulations_dir}': {' '.join(cmd)}")
        subprocess.run(cmd, cwd=build_simulations_dir, check=True)
        print_color(Colors.GREEN, "✓ RMSE plot generated in 'figures_and_plots/'.\n")
    except subprocess.CalledProcessError as e:
        print_color(Colors.RED, f"ERROR: MSE calculation failed.")
        print_color(Colors.RED, f"Stderr:\n{e.stderr}")
        sys.exit(1)

    # --- Final Summary ---
    print_color(Colors.BLUE, "========================================")
    print_color(Colors.BLUE, "         Pipeline Test Complete         ")
    print_color(Colors.BLUE, "========================================")
    print_color(Colors.GREEN, "✓ All steps completed successfully!")
    print("Outputs can be found in:")
    print(f"  - {os.path.join(build_simulations_dir, 'simulation_outputs')}/")
    print(f"  - {os.path.join(build_simulations_dir, 'figures_and_plots')}/")
    print("\nPlease review the generated plots for visual verification.")

if __name__ == "__main__":
    main()
