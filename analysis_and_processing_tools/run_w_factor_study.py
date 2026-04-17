#!/usr/bin/env python3
import os
import sys
import subprocess
import numpy as np
import re
import re

def format_w_factor_for_filename(w_factor):
    """Create a unique filename string for w_factor values."""
    if w_factor == 0:
        return "0p000000en99"
    
    # Use high precision to avoid collisions (6 decimal places)
    sci_str = f"{w_factor:.6e}"
    
    # Parse the scientific notation
    if 'e' in sci_str:
        mantissa_str, exp_str = sci_str.split('e')
        mantissa = float(mantissa_str)
        exponent = int(exp_str)
        
        # Format mantissa to remove decimal point
        # e.g., 4.123456 -> 4p123456
        if '.' in mantissa_str:
            integer_part, decimal_part = mantissa_str.split('.')
            # Pad decimal part to ensure consistent length
            decimal_part = decimal_part.ljust(6, '0')[:6]  # Exactly 6 digits
            formatted_mantissa = f"{integer_part}p{decimal_part}"
        else:
            formatted_mantissa = f"{mantissa_str}p000000"
        
        # Format exponent with consistent width
        if exponent >= 0:
            exp_part = f"e{exponent:02d}"
        else:
            exp_part = f"en{abs(exponent):02d}"
        
        return f"{formatted_mantissa}{exp_part}"
    else:
        # Fallback for non-scientific notation
        return sci_str.replace('.', 'p').replace('-', 'n')

def generate_w_factor_values():
    """Generate 75 w_factor values with more sensible range."""
    # More sensible range: 1e-4 to 1 (focused around typical values)
    log_min = np.log10(1e-4)  # -4
    log_max = np.log10(1.0)   # 0
    
    # Generate 75 logarithmically spaced values
    log_values = np.linspace(log_min, log_max, 75)
    w_factor_values = [10**log_val for log_val in log_values]
    
    # Ensure exact boundary values
    w_factor_values[0] = 1e-4
    w_factor_values[-1] = 1.0
    
    return sorted(w_factor_values)

def run_w_factor_study():
    """Run parameter study for different w_factor_constant values."""
    
    # Generate w_factor values
    w_factor_values = generate_w_factor_values()
    
    print(f"Starting w_factor parameter study")
    print(f"Range: {min(w_factor_values):.2e} to {max(w_factor_values):.2e}")
    print(f"Number of simulations: {len(w_factor_values)}")
    
    # Find project directories
    script_path = os.path.abspath(os.path.dirname(__file__))
    project_root = script_path
    while os.path.basename(project_root) != 'uguca' and project_root != '/':
        project_root = os.path.dirname(project_root)

    if os.path.basename(project_root) != 'uguca':
        print("Error: Could not determine project root directory.")
        return

    build_dir = os.path.join(project_root, "build", "simulations")
    source_sim_dir = os.path.join(project_root, "simulations")
    
    if not os.path.isdir(build_dir):
        print(f"Error: Build directory not found at {build_dir}")
        return

    executable_path = os.path.join(build_dir, "time_int_opt")
    if not os.path.exists(executable_path):
        print(f"Error: Executable not found at {executable_path}")
        return

    # Base input file
    base_input_file = os.path.join(source_sim_dir, "input_files", "deviated_params_dd_noisy_scenario.in")
    if not os.path.exists(base_input_file):
        print(f"Error: Base input file not found at {base_input_file}")
        return

    # Output directory
    output_dir = os.path.join(build_dir, "simulation_outputs")
    os.makedirs(output_dir, exist_ok=True)

    successful_runs = 0
    failed_runs = 0
    
    for i, w_factor in enumerate(w_factor_values):
        print(f"\nRunning simulation {i+1}/{len(w_factor_values)}")
        print(f"w_factor = {w_factor:.6e}")
        
        # Create formatted filename string
        w_factor_str = format_w_factor_for_filename(w_factor)
        sim_name = f"time_int_opt_wfactor_{w_factor_str}"
        
        # Create input file in input_files directory
        input_files_dir = os.path.join(build_dir, "input_files")
        os.makedirs(input_files_dir, exist_ok=True)
        input_file = os.path.join(input_files_dir, f"{sim_name}.in")
        
        try:
            with open(base_input_file, 'r') as f:
                content = f.read()
            
            # Replace w_factor_constant value
            content = content.replace("w_factor_constant = 1e-2", f"w_factor_constant = {w_factor}")
            
            # Also update sim_name to be unique
            content = content.replace("sim_name = time_int_opt_deviated_dd_noisy_scenario", f"sim_name = {sim_name}")
            
            with open(input_file, 'w') as f:
                f.write(content)
                
        except Exception as e:
            print(f"Error creating input file: {e}")
            failed_runs += 1
            continue
        
        # Run simulation with relative path to input file
        try:
            relative_input_file = os.path.join("input_files", f"{sim_name}.in")
            result = subprocess.run(
                [executable_path, relative_input_file],
                cwd=build_dir,
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )
            
            if result.returncode == 0:
                print(f"✓ Simulation completed successfully")
                successful_runs += 1
            else:
                print(f"✗ Simulation failed with return code {result.returncode}")
                print(f"Error: {result.stderr}")
                failed_runs += 1
                
        except subprocess.TimeoutExpired:
            print(f"✗ Simulation timed out after 5 minutes")
            failed_runs += 1
        except Exception as e:
            print(f"✗ Error running simulation: {e}")
            failed_runs += 1
        
        # Clean up input file
        try:
            os.remove(input_file)
        except:
            pass
    
    print(f"\n=== W-factor Study Complete ===")
    print(f"Successful runs: {successful_runs}")
    print(f"Failed runs: {failed_runs}")
    print(f"Total runs: {len(w_factor_values)}")

if __name__ == "__main__":
    run_w_factor_study()
