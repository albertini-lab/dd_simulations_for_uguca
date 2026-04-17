#!/usr/bin/env python3
"""
Parameter study for the no-nucleation nucleation discovery test.
Tests different data_driven_update_interval values (1 to 100, increments of 10)
to see how update frequency affects the ability to discover nucleation from data.
"""
import os
import sys
import subprocess
import re
import numpy as np

def design_no_nucleation_study():
    """
    Design parameter study for no-nucleation case.
    Tests update intervals from 1 to 100 in increments of 10.
    """
    print("=== No-Nucleation Nucleation Discovery Parameter Study Design ===")
    print("Testing how update frequency affects nucleation discovery capability")

    # Simple range: 1 to 101 in increments of 10, plus 1
    intervals = [1] + list(range(1, 101, 10))  # [1, 11, 21, ..., 101]
    intervals = sorted(set(intervals))  # Remove duplicates and sort
    print(f"Update interval analysis:")
    print(f"  Range: {min(intervals)} to {max(intervals)}")
    print(f"  Total simulations: {len(intervals)}")
    print()
    
    print("Update intervals to test:")
    print("Interval | Updates per ~1718 steps | Update Frequency [%]")
    print("-" * 50)
    
    total_steps = 1718  # Approximate number of time steps
    
    for i, interval in enumerate(intervals):
        updates_count = total_steps // interval
        update_freq = 100.0 / interval
        print(f"{i+1:2d}. {interval:8d}   |       {updates_count:8d}       |   {update_freq:8.2f}")
    
    print(f"\nTotal simulations: {len(intervals)}")
    print("This study tests if frequent updates (low intervals) are needed")
    print("for the data-driven method to discover nucleation behavior.")
    
    return intervals

def run_no_nucleation_parameter_study(base_input_file, intervals):
    """
    Runs parameter study for no-nucleation case with different update intervals.
    """
    print(f"Starting NO-NUCLEATION parameter study")
    print(f"Base input file: {base_input_file}")
    print(f"Testing {len(intervals)} interval values")
    print(f"Range: {min(intervals)} to {max(intervals)}")
    print()


    # Since we're running from build/simulations, look for input files here
    current_dir = os.getcwd()
    build_dir = current_dir  # We're already in build/simulations
    
    print(f"Current directory: {current_dir}")
    print(f"Build directory: {build_dir}")
    print()

    executable_path = os.path.join(build_dir, "time_int_opt")
    if not os.path.exists(executable_path):
        print(f"Error: Simulation executable not found at {executable_path}")
        return

    # Look for input file in multiple locations
    possible_input_paths = [
        os.path.join(build_dir, "input_files", base_input_file),  # build/simulations/input_files/
        os.path.join(build_dir, base_input_file),                 # build/simulations/
        base_input_file                                           # current directory
    ]
    
    base_input_path = None
    for path in possible_input_paths:
        if os.path.exists(path):
            base_input_path = path
            break
    
    if base_input_path is None:
        print(f"Error: Base input file '{base_input_file}' not found in any of these locations:")
        for path in possible_input_paths:
            print(f"  - {path}")
        print("\nPlease make sure the dd_perturbed_no_nucleation_test.in file exists.")
        return

    print(f"Found base input file: {base_input_path}")

    with open(base_input_path, 'r') as f:
        base_content = f.read()

    # Track results
    successful_runs = 0
    failed_runs = 0

    input_files_dir = os.path.join(build_dir, "input_files")
    os.makedirs(input_files_dir, exist_ok=True)

    for i, interval in enumerate(intervals, 1):
        print(f"--- Running {i}/{len(intervals)}: update_interval = {interval} ---")

        new_sim_name = f"dd_perturbed_no_nucleation_test_interval_{interval}"

        # Modify content
        content = re.sub(r"sim_name\s*=\s*.*", f"sim_name = {new_sim_name}", base_content)
        content = re.sub(r"data_driven_update_interval\s*=\s*.*", f"data_driven_update_interval = {interval}", content)

       
        # Write temporary input file
        temp_input_filename = f"{new_sim_name}.in"
        temp_input_path = os.path.join(input_files_dir, temp_input_filename)
        
        with open(temp_input_path, 'w') as f:
            f.write(content)
            
        print(f"Generated input file: {temp_input_filename}")

        # Run the simulation
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
                print(f"  Stdout: {e.stdout[-200:]}")
            if e.stderr:
                print(f"  Stderr: {e.stderr[-200:]}")
            failed_runs += 1
        except Exception as e:
            print(f"✗ Unexpected error: {e}")
            failed_runs += 1
        finally:
            # Clean up temporary input file
            if os.path.exists(temp_input_path):
                os.remove(temp_input_path)
        
        print()

    print("=== NO-NUCLEATION PARAMETER STUDY COMPLETED ===")
    print(f"Successful runs: {successful_runs}/{len(intervals)}")
    print(f"Failed runs: {failed_runs}/{len(intervals)}")
    
    if successful_runs > 0:
        print()
        print("Next steps:")
        print("1. Run the visualization:")
        print("   python plotting_tools/plot_no_nucleation_study.py")
        print()
        print("2. Compare with ground truth:")
        print("   python plotting_tools/time_int_opt_plot.py time_int_opt_gt tau_max 0")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--run":
        base_file = "dd_no_nucleation_test.in"
        intervals_to_run = design_no_nucleation_study()
        print(f"\nStarting no-nucleation study with {len(intervals_to_run)} simulations...\n")
        run_no_nucleation_parameter_study(base_file, intervals_to_run)
    else:
        print("No-Nucleation Parameter Study Script")
        print("Usage:")
        print("  python analysis_and_processing_tools/no_nucleation_parameter_study.py --run")
        print()
        print("This tests different update intervals to see how frequency")
        print("affects the ability to discover nucleation from ground truth data.")
        intervals = design_no_nucleation_study()