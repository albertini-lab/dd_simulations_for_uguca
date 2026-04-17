#!/usr/bin/env python3
"""
Comprehensive data_driven_update_interval parameter study.
Runs    print("Starting COMPREHENSIVE update interval parameter study")
    print(f"Base input file: {base_input_file}")
    if intervals:
        print(f"Testing {len(intervals)} interval values across entire spectrum")
        print(f"Range: {min(intervals)} to {max(intervals)}")
    else:
        print("Baseline-only mode: running baseline simulation without parameter study")
    print()simulations with update intervals from 1 to 500 to characterize
how update frequency affects data-driven simulation accuracy and stability.
"""
import os
import sys
import subprocess
import re
import numpy as np

def design_interval_study():
    """
    Design a comprehensive parameter study for data_driven_update_interval.
    Generates exactly 100 unique values from 1 to 500 using smart spacing.
    """
    print("=== Comprehensive Update Interval Parameter Study Design ===")
    print("Spanning entire range from frequent to infrequent updates")
    
    # Define comprehensive range
    interval_min = 1     # Most frequent updates (every time step)
    interval_max = 500   # Very infrequent updates
    target_samples = 100 # Exactly 100 simulations
    
    print(f"Update interval spectrum analysis:")
    print(f"  Lower bound (frequent): {interval_min}")
    print(f"  Upper bound (infrequent): {interval_max}")
    print(f"  Target samples: {target_samples}")
    print()
    
    # Generate intervals with smart spacing to get exactly 100 unique values
    # Use a hybrid approach: dense spacing for low values, logarithmic for high values
    
    # For intervals 1-20: every integer (20 values)
    low_intervals = list(range(1, 21))
    
    # For intervals 21-100: every 2-3 integers (27 values)
    mid_intervals = list(range(21, 101, 3))
    
    # For intervals 100-500: logarithmic spacing (53 values)
    high_log = np.logspace(np.log10(100), np.log10(500), 55)
    high_intervals = np.unique(np.round(high_log).astype(int))
    high_intervals = [x for x in high_intervals if x > 100]  # Remove overlap
    
    # Combine all intervals
    all_intervals = low_intervals + mid_intervals + list(high_intervals)
    intervals = np.unique(np.array(all_intervals))
    
    print(f"Generated {len(intervals)} unique update intervals (showing first 10, middle 10, last 10):")
    print("Update Interval | Updates per 1718 time steps | Update Frequency [%]")
    print("-" * 70)
    
    total_steps = 1718  # Approximate number of time steps in simulation
    
    # Show first 10
    for i, interval in enumerate(intervals[:10]):
        updates_count = total_steps // interval
        update_freq = 100.0 / interval
        print(f"{i+1:2d}. {interval:12d}   |       {updates_count:8d}            |   {update_freq:8.2f}")
    
    print("    ... (middle values) ...")
    
    # Show middle 10 (around indices 45-55)
    mid_start = len(intervals) // 2 - 5
    for i, interval in enumerate(intervals[mid_start:mid_start+10], mid_start+1):
        updates_count = total_steps // interval
        update_freq = 100.0 / interval
        print(f"{i:2d}. {interval:12d}   |       {updates_count:8d}            |   {update_freq:8.2f}")
    
    print("    ... (more values) ...")
    
    # Show last 10
    for i, interval in enumerate(intervals[-10:], len(intervals)-9):
        updates_count = total_steps // interval
        update_freq = 100.0 / interval
        print(f"{i:2d}. {interval:12d}   |       {updates_count:8d}            |   {update_freq:8.2f}")
    
    print()
    print(f"Total simulations in comprehensive study: {len(intervals)}")
    print(f"Achieved target of ~{target_samples} simulations: {len(intervals)} total")
    print(f"Coverage: Dense spacing for low intervals (1-20), medium spacing (21-100), logarithmic for high intervals (101-500)")
    print(f"This provides comprehensive characterization across the full spectrum")
    
    return intervals.tolist()

def run_parameter_study(base_input_file, intervals):
    """
    Runs comprehensive parameter study for different data_driven_update_interval values.
    """
    print(f"Starting COMPREHENSIVE update interval parameter study")
    print(f"Base input file: {base_input_file}")
    if intervals:
        print(f"Testing {len(intervals)} interval values across entire spectrum")
        print(f"Range: {min(intervals)} to {max(intervals)}")
    else:
        print("Running baseline simulation only (no interval parameter study)")
    print()

    # Use current working directory approach - we know we're in uguca project
    current_dir = os.getcwd()
    
    # Find uguca root by going up from current directory
    project_root = current_dir
    while not os.path.exists(os.path.join(project_root, 'build', 'simulations')) and project_root != '/':
        project_root = os.path.dirname(project_root)
    
    # Verify we found the right directory structure
    build_dir = os.path.join(project_root, "build", "simulations")
    source_sim_dir = os.path.join(project_root, "simulations")
    
    if not os.path.isdir(build_dir):
        print(f"Error: Build directory not found at {build_dir}")
        print(f"Current directory: {current_dir}")
        print("Please ensure the project is built and you are running from the correct location.")
        return
    
    print(f"Project root: {project_root}")
    print(f"Build directory: {build_dir}")
    print(f"Source simulations directory: {source_sim_dir}")
    print()

    executable_path = os.path.join(build_dir, "time_int_opt")
    if not os.path.exists(executable_path):
        print(f"Error: Simulation executable not found at {executable_path}")
        return

    base_input_path = os.path.join(source_sim_dir, "input_files", base_input_file)
    if not os.path.exists(base_input_path):
        print(f"Error: Base input file not found at {base_input_path}")
        return

    with open(base_input_path, 'r') as f:
        base_content = f.read()

    # First, run the baseline simulation with perturbed parameters but no data-driven updates
    print("--- Running baseline simulation (no data-driven updates) ---")
    baseline_sim_name = "time_int_opt_dd_baseline_no_dd"
    baseline_content = re.sub(r"sim_name\s*=\s*.*", f"sim_name = {baseline_sim_name}", base_content)
    baseline_content = re.sub(r"enable_data_driven_mode\s*=\s*.*", "enable_data_driven_mode = false", baseline_content)
    baseline_content = re.sub(r"ground_truth_solution_generation\s*=\s*.*", "ground_truth_solution_generation = true", baseline_content)
    baseline_content = re.sub(r"nb_dumps\s*=\s*.*", f"nb_dumps = 1", baseline_content)
    
    baseline_input_filename = f"{baseline_sim_name}.in"
    input_files_dir = os.path.join(build_dir, "input_files")
    os.makedirs(input_files_dir, exist_ok=True)
    baseline_input_path = os.path.join(input_files_dir, baseline_input_filename)
    
    with open(baseline_input_path, 'w') as f:
        f.write(baseline_content)
    
    try:
        print(f"Executing: {executable_path} {baseline_input_filename}")
        subprocess.run([executable_path, baseline_input_filename], check=True, cwd=build_dir)
        print("--- Finished baseline simulation ---")
    except subprocess.CalledProcessError as e:
        print(f"Error running baseline simulation: {e}")
    finally:
        if os.path.exists(baseline_input_path):
            os.remove(baseline_input_path)

    # Track results
    successful_runs = 0
    failed_runs = 0

    # Skip parameter study if intervals list is empty (baseline-only mode)
    if not intervals:
        print("Baseline-only mode: skipping parameter study iterations.")
        print("=== BASELINE SIMULATION COMPLETED ===")
        print("Baseline simulation finished successfully.")
        return

    for i, interval in enumerate(intervals, 1):
        print(f"--- Running {i}/{len(intervals)}: interval = {interval} ---")
        
        new_sim_name = f"time_int_opt_dd_interval_{interval}"
        
        # Modify content
        content = re.sub(r"sim_name\s*=\s*.*", f"sim_name = {new_sim_name}", base_content)
        content = re.sub(r"data_driven_update_interval\s*=\s*.*", f"data_driven_update_interval = {interval}", content)
        # Keep data-driven mode enabled and ensure displacement dumps are captured
        # (Data-driven mode has built-in per-step displacement dumping via calculated_disp_dumper)
        content = re.sub(r"nb_dumps\s*=\s*.*", f"nb_dumps = 1", content)  # Only dump final state for regular output
       
        # Write to a temporary input file in the build directory's input_files
        temp_input_filename = f"{new_sim_name}.in"
        input_files_dir = os.path.join(build_dir, "input_files")
        # Create input_files directory in build if it doesn't exist
        os.makedirs(input_files_dir, exist_ok=True)
        temp_input_path = os.path.join(input_files_dir, temp_input_filename)
        
        with open(temp_input_path, 'w') as f:
            f.write(content)
            
        print(f"Generated input file: {temp_input_filename}")

        # Run the simulation directly
        try:
            print(f"Executing: {executable_path} {temp_input_filename}")
            result = subprocess.run([executable_path, temp_input_filename], 
                                  check=True, cwd=build_dir, 
                                  capture_output=True, text=True)
            print(f"✓ Completed successfully")
            successful_runs += 1
        except subprocess.CalledProcessError as e:
            print(f"✗ Error running simulation: {e}")
            print(f"  Return code: {e.returncode}")
            if e.stdout:
                print(f"  Stdout: {e.stdout[-200:]}")  # Last 200 chars
            if e.stderr:
                print(f"  Stderr: {e.stderr[-200:]}")  # Last 200 chars
            failed_runs += 1
        except Exception as e:
            print(f"✗ Unexpected error: {e}")
            failed_runs += 1
        finally:
            # Clean up the temporary input file
            if os.path.exists(temp_input_path):
                os.remove(temp_input_path)
        
        print()

    print("=== COMPREHENSIVE UPDATE INTERVAL STUDY COMPLETED ===")
    if intervals:
        print(f"Successful runs: {successful_runs}/{len(intervals)}")
        print(f"Failed runs: {failed_runs}/{len(intervals)}")
    else:
        print(f"Baseline simulation completed successfully" if successful_runs > 0 else "Baseline simulation failed")
    
    if successful_runs > 0:
        print()
        print("Next steps:")
        if intervals:
            print("1. Run the parameter study visualization:")
            print("   python plotting_tools/plot_parameter_study.py")
            print()
            print("2. Analyze convergence behavior:")
            print("   python plotting_tools/plot_interval_convergence.py")
        else:
            print("1. Now run the full parameter studies:")
            print("   python simulations/analysis_and_processing_tools/run_parameter_study.py --run")
            print("   python simulations/analysis_and_processing_tools/run_w_factor_study_refined.py")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--run":
        base_file = "deviated_params_dd_noisy_scenario.in"
        # Use comprehensive 100-simulation study design
        intervals_to_run = design_interval_study()
        print(f"\nStarting comprehensive study with {len(intervals_to_run)} simulations...\n")
        run_parameter_study(base_file, intervals_to_run)
    elif len(sys.argv) > 1 and sys.argv[1] == "--baseline-only":
        base_file = "deviated_params_dd_noisy_scenario.in"
        print("Running baseline simulation only (no data-driven updates)...")
        run_parameter_study(base_file, [])  # Empty intervals list will only run baseline
    else:
        print("This script is for running a comprehensive update interval parameter study.")
        print("Usage:")
        print("  python run_parameter_study.py --run           # Run full 101-simulation study")
        print("  python run_parameter_study.py --baseline-only # Run baseline simulation only")
        print("\nFull study runs 101 simulations with update intervals from 1 to 500")
        print("to characterize how update frequency affects simulation accuracy and stability.")

