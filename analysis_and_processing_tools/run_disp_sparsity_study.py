#!/usr/bin/env python3

import os
import sys
import subprocess
import re
import numpy as np
import shutil

def design_sparsity_study():
    """
    Design a comprehensive parameter study for spatial data sparsity.
    Generates sparsity fractions from 1% to 100%.
    """
    print("=== Spatial Data Sparsity Parameter Study Design ===")
    
    # Define sparsity range
    min_fraction = 0.01  # 1% sparsity
    max_fraction = 1.00  # 100% sparsity (0% data)
    num_samples = 100
    
    sparsity_fractions = np.linspace(min_fraction, max_fraction, num_samples)
    
    print(f"Sparsity fraction analysis:")
    print(f"  Lower bound (low sparsity): {min_fraction:.2f}")
    print(f"  Upper bound (high sparsity): {max_fraction:.2f}")
    print(f"  Target samples: {len(sparsity_fractions)}")
    print()
    
    print(f"Generated {len(sparsity_fractions)} sparsity fraction values.")
    print("Note: Each simulation requires:")
    print("  1. Ground truth generation and noise application (once)")
    print("  2. Application of spatial sparsity to the noisy ground truth data")
    print("  3. Data-driven simulation with the sparse data")
    
    return sparsity_fractions

def format_sparsity_for_filename(fraction):
    """
    Create a unique, descriptive filename string for sparsity fractions.
    """
    percent = int(fraction * 100)
    return f"sparsity_{percent:03d}_percent"

def generate_ground_truth():
    """
    Generate ground truth data by running the ground truth simulation.
    """
    print("--- Generating Ground Truth Data ---")
    
    # Find project directories
    script_path = os.path.abspath(os.path.dirname(__file__))
    project_root = script_path
    while os.path.basename(project_root) != 'uguca' and project_root != '/':
        project_root = os.path.dirname(project_root)
    
    if os.path.basename(project_root) != 'uguca':
        raise FileNotFoundError("Could not find the project root directory 'uguca'")
    
    build_dir = os.path.join(project_root, "build", "simulations")
    source_sim_dir = os.path.join(project_root, "simulations")
    
    executable_path = os.path.join(build_dir, "time_int_opt")
    gt_input_file = os.path.join(source_sim_dir, "input_files", "ground_truth.in")
    
    if not os.path.exists(executable_path):
        raise FileNotFoundError(f"Executable not found at {executable_path}")
    if not os.path.exists(gt_input_file):
        raise FileNotFoundError(f"Ground truth input file not found at {gt_input_file}")
    
    # Copy input file to build directory to ensure it's found
    input_files_dir = os.path.join(build_dir, "input_files")
    os.makedirs(input_files_dir, exist_ok=True)
    
    gt_input_build = os.path.join(input_files_dir, "ground_truth.in")
    shutil.copy2(gt_input_file, gt_input_build)
    
    try:
        cmd = [executable_path, os.path.join("input_files", "ground_truth.in")]
        print(f"Running command: {' '.join(cmd)} in {build_dir}")
        subprocess.run(cmd, check=True, cwd=build_dir)
        
        # Find the created directory
        simulation_outputs_dir = os.path.join(build_dir, "simulation_outputs")
        gt_pattern = "time_int_opt_gt_ground_truth"
        for dirname in os.listdir(simulation_outputs_dir):
            if gt_pattern in dirname:
                created_dir = os.path.join(simulation_outputs_dir, dirname)
                print(f"--- Ground Truth Data Generated in: {created_dir} ---")
                return created_dir
        return None # Should not happen if simulation succeeded
        
    except subprocess.CalledProcessError as e:
        print(f"Error generating ground truth data: {e}")
        return None
    finally:
        # Clean up copied input file
        if os.path.exists(gt_input_build):
            os.remove(gt_input_build)

def ensure_ground_truth_exists():
    """
    Check if ground truth data exists, generate if needed.
    Returns the path to the ground truth simulation output.
    """
    print("=== Checking Ground Truth Data Availability ===")
    
    # Look for existing ground truth data
    build_dir = "/Users/joshmcneely/uguca/build/simulations"
    simulation_outputs_dir = os.path.join(build_dir, "simulation_outputs")
    gt_pattern = "time_int_opt_gt_ground_truth"
    
    potential_gt_dirs = []
    if os.path.exists(simulation_outputs_dir):
        for dirname in os.listdir(simulation_outputs_dir):
            if gt_pattern in dirname and os.path.isdir(os.path.join(simulation_outputs_dir, dirname)):
                potential_gt_dirs.append(os.path.join(simulation_outputs_dir, dirname))
    
    if potential_gt_dirs:
        latest_gt_dir = max(potential_gt_dirs, key=os.path.getmtime)
        print(f"Found existing ground truth data at: {latest_gt_dir}")
        return latest_gt_dir
    
    # If not found, generate it
    print("Ground truth data not found. Generating it now...")
    return generate_ground_truth()

def ensure_noisy_ground_truth_exists():
    """
    Check if noisy ground truth data exists, generate if needed.
    This will be the baseline data before applying sparsity.
    Returns the path to the noisy ground truth simulation output.
    """
    print("=== Checking Noisy Ground Truth Data Availability ===")
    
    # Path for the processed noisy data
    script_path = os.path.abspath(os.path.dirname(__file__))
    noisy_gt_dir = os.path.join(script_path, "simulation_outputs", "noisy_gt_data_input")

    if os.path.exists(noisy_gt_dir):
        print(f"Found existing noisy ground truth data at: {noisy_gt_dir}")
        return noisy_gt_dir

    print("Noisy ground truth not found. Generating it now...")
    
    # 1. Ensure original (non-noisy) ground truth exists
    original_gt_dir = ensure_ground_truth_exists()
    if not original_gt_dir:
        print("Fatal: Could not find or generate original ground truth data.")
        return None

    # 2. Process it to add noise
    print("--- Applying noise to ground truth data ---")
    
    # Find project directories
    script_path = os.path.abspath(os.path.dirname(__file__))
    project_root = script_path
    while os.path.basename(project_root) != 'uguca' and project_root != '/':
        project_root = os.path.dirname(project_root)
    
    process_script = os.path.join(script_path, "process_ground_truth_displacement.py")
    
    if not os.path.exists(process_script):
        print(f"Error: Processing script not found at {process_script}")
        return None
    
    # We need the base name of the simulation, e.g., 'time_int_opt_gt'
    original_sim_name_base = os.path.basename(original_gt_dir).replace('_ground_truth', '')

    try:
        # This command will add default noise and save the output to 'noisy_gt_data_input'
        cmd = [
            "python", process_script,
            original_sim_name_base,
            "--add-noise",
            "--output-noisy-data"
        ]
        print(f"Running command: {' '.join(cmd)}")
        # Run from the build/simulations directory to ensure correct working directory
        build_dir = os.path.join(project_root, "build", "simulations")
        subprocess.run(cmd, check=True, cwd=build_dir)
        print(f"Successfully created noisy ground truth data at: {noisy_gt_dir}")
        return noisy_gt_dir
    except subprocess.CalledProcessError as e:
        print(f"Error processing ground truth with noise: {e}")
        return None

def process_ground_truth_with_sparsity(base_noisy_gt_dir, sparsity_fraction):
    """
    Process the original ground truth data with noise AND spatial sparsity.
    Returns the path to the processed sparse data.
    """
    print(f"--- Processing Original GT with Noise + Sparsity (fraction={sparsity_fraction:.2f}) ---")
    
    script_path = os.path.abspath(os.path.dirname(__file__))
    process_script = os.path.join(script_path, "process_ground_truth_displacement.py")
    
    if not os.path.exists(process_script):
        print(f"Error: Processing script not found at {process_script}")
        return None

    # The input for processing is the *original* ground truth data
    original_sim_name_base = "time_int_opt_gt"
    
    # Define a unique output directory for this sparsity level
    output_dir_name = format_sparsity_for_filename(sparsity_fraction)
    
    # Find project directories
    project_root = script_path
    while os.path.basename(project_root) != 'uguca' and project_root != '/':
        project_root = os.path.dirname(project_root)
    
    build_dir = os.path.join(project_root, "build", "simulations")
    absolute_output_path = os.path.join(build_dir, "simulation_outputs", output_dir_name)
    
    try:
        cmd = [
            "python", process_script,
            original_sim_name_base,
            "--add-noise",  # Add noise to the original ground truth
            "--enable-spatial-sparsity",  # And apply sparsity
            "--sparsity-type", "random_fraction",
            "--sparsity-fraction", str(sparsity_fraction),
            "--output-noisy-data", # Re-using this flag to mean "output processed data"
            "--output-dir-name", output_dir_name # Custom flag needed in processing script
        ]
        print(f"Running command: {' '.join(cmd)}")
        # We need to run from a directory where it can find "time_int_opt_gt"
        subprocess.run(cmd, check=True, cwd=build_dir)
        print(f"Successfully created sparse data at: {absolute_output_path}")
        return absolute_output_path
    except subprocess.CalledProcessError as e:
        print(f"Error processing ground truth with sparsity: {e}")
        return None

def run_data_driven_simulation(sparsity_fraction, base_input_file, sparse_data_path):
    """
    Run data-driven simulation with specific sparsity parameters.
    """
    print(f"--- Running Data-Driven Simulation (sparsity={sparsity_fraction:.2f}) ---")
    
    # Find project directories
    script_path = os.path.abspath(os.path.dirname(__file__))
    project_root = script_path
    while os.path.basename(project_root) != 'uguca' and project_root != '/':
        project_root = os.path.dirname(project_root)
    
    if os.path.basename(project_root) != 'uguca':
        raise FileNotFoundError("Could not find the project root directory 'uguca'")
    
    build_dir = os.path.join(project_root, "build", "simulations")
    source_sim_dir = os.path.join(project_root, "simulations")
    
    executable_path = os.path.join(build_dir, "time_int_opt")
    base_input_path = os.path.join(source_sim_dir, "input_files", base_input_file)
    
    # Read base input file
    with open(base_input_path, 'r') as f:
        base_input_content = f.read()
    
    # Create unique simulation name
    sim_name = format_sparsity_for_filename(sparsity_fraction)
    
    # Modify input file content
    modified_content = re.sub(r"sim_name\s*=\s*.*", f"sim_name = {sim_name}", base_input_content)
    
    # The path to the sparse data is now relative to the build/simulations dir
    relative_sparse_data_path = os.path.relpath(sparse_data_path, build_dir)
    
    modified_content = re.sub(r"ground_truth_data_path\s*=\s*.*", f"ground_truth_data_path = {relative_sparse_data_path}", modified_content)
    modified_content = re.sub(r"ground_truth_data_name\s*=\s*.*", f"ground_truth_data_name = {os.path.basename(relative_sparse_data_path)}", modified_content)
    
    # Create a temporary input file for this specific run
    input_files_dir = os.path.join(build_dir, "input_files")
    os.makedirs(input_files_dir, exist_ok=True)
    
    run_input_file = os.path.join(input_files_dir, f"{sim_name}.in")
    with open(run_input_file, 'w') as f:
        f.write(modified_content)
        
    print(f"Created temporary input file: {run_input_file}")

    try:
        cmd = [executable_path, os.path.join("input_files", f"{sim_name}.in")]
        print(f"Running command: {' '.join(cmd)} in {build_dir}")
        subprocess.run(cmd, check=True, cwd=build_dir)
        print(f"--- Successfully completed simulation for {sim_name} ---")
    except subprocess.CalledProcessError as e:
        print(f"Error running data-driven simulation for {sim_name}: {e}")
    finally:
        # Clean up the temporary input file
        if os.path.exists(run_input_file):
            os.remove(run_input_file)

def run_sparsity_study(base_input_file, sparsity_fractions):
    """
    Main function to run the entire spatial sparsity study.
    """
    print("\n\n=== Starting Spatial Sparsity Study ===")
    
    # 1. Ensure the baseline noisy ground truth data is ready.
    base_noisy_gt_dir = ensure_noisy_ground_truth_exists()
    if not base_noisy_gt_dir:
        print("Fatal: Could not prepare baseline noisy ground truth data. Aborting study.")
        return
        
    # 2. Loop through each sparsity level
    for i, fraction in enumerate(sparsity_fractions):
        print(f"\n--- Running Study Step {i+1}/{len(sparsity_fractions)}: Sparsity Fraction = {fraction:.2f} ---")
        
        # a. Process the noisy data to introduce sparsity
        sparse_data_path = process_ground_truth_with_sparsity(base_noisy_gt_dir, fraction)
        
        if not sparse_data_path:
            print(f"Skipping simulation for sparsity {fraction:.2f} due to processing error.")
            continue
            
        # b. Run the data-driven simulation with the sparse data
        run_data_driven_simulation(fraction, base_input_file, sparse_data_path)
        
    print("\n=== Spatial Sparsity Study Complete ===")

if __name__ == "__main__":
    # Define the base input file for the data-driven runs
    base_input = "deviated_params_dd_noisy_scenario.in"
    
    # 1. Design the study
    sparsity_levels = design_sparsity_study()
    
    # 2. Run the full study
    run_sparsity_study(base_input, sparsity_levels)
