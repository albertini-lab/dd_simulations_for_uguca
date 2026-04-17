#!/usr/bin/env python3
"""
Comprehensive noise covariance parameter study.
This study varies the noise standard deviation to characterize how
measurement uncertainty affects data-driven simulation accuracy.

Workflow for each noise level:
1. Generate ground truth data (once, if not already available)
2. Process ground truth data with specific noise level
3. Run data-driven simulation with matching noise parameters
4. Analyze results
"""
import os
import sys
import subprocess
import re
import numpy as np
import shutil

def design_noise_study():
    """
    Design a comprehensive parameter study for noise standard deviation.
    Generates noise levels from very low (1e-8) to high (1e-3) noise.
    """
    print("=== Comprehensive Noise Covariance Parameter Study Design ===")
    print("Spanning range from very low to high measurement uncertainty")
    
    # Define comprehensive range - noise standard deviations
    noise_min = 1e-8    # Very low noise (near perfect measurements)
    noise_max = 1e-3    # High noise (significant measurement uncertainty)
    target_samples = 100 # 100 noise levels for good coverage

    print(f"Noise standard deviation spectrum analysis:")
    print(f"  Lower bound (low noise): {noise_min:.2e}")
    print(f"  Upper bound (high noise): {noise_max:.2e}")
    print(f"  Target samples: {target_samples}")
    print()
    
    # Generate noise levels with logarithmic spacing (appropriate for wide range)
    log_min = np.log10(noise_min)
    log_max = np.log10(noise_max)
    log_values = np.linspace(log_min, log_max, target_samples)
    noise_stddevs = [10**log_val for log_val in log_values]
    
    # Ensure boundary values are exact
    noise_stddevs[0] = noise_min
    noise_stddevs[-1] = noise_max
    
    print(f"Generated {len(noise_stddevs)} noise standard deviation values:")
    print("Noise Std Dev | Noise Variance | Signal-to-Noise Ratio (approx)")
    print("-" * 70)
    
    # Approximate displacement magnitude for SNR calculation (typical value)
    typical_displacement = 1e-6  # meters, approximate from previous simulations
    
    # Show sample values
    sample_indices = [0, len(noise_stddevs)//4, len(noise_stddevs)//2, 3*len(noise_stddevs)//4, len(noise_stddevs)-1]
    for i, idx in enumerate(sample_indices):
        noise_std = noise_stddevs[idx]
        noise_var = noise_std ** 2
        snr = typical_displacement / noise_std if noise_std > 0 else float('inf')
        print(f"{idx+1:2d}. {noise_std:12.2e}   |   {noise_var:12.2e}   |       {snr:8.2f}")
    
    print("    ... (intermediate values) ...")
    
    print()
    print(f"Total simulations in noise study: {len(noise_stddevs)}")
    print(f"Coverage: Logarithmic spacing from very low to high noise levels")
    print(f"This provides comprehensive characterization of noise sensitivity")
    print()
    print("Note: Each simulation requires:")
    print("  1. Ground truth generation (once)")
    print("  2. Noise application to ground truth data")
    print("  3. Data-driven simulation with matching noise parameters")
    
    return noise_stddevs

def format_noise_for_filename(noise_stddev):
    """
    Create a unique, high-precision filename string for noise values.
    This prevents duplicates in both input files and output directories.
    """
    if noise_stddev == 0:
        return "0p000000en99"
    
    # Use high precision to avoid collisions (6 decimal places)
    sci_str = f"{noise_stddev:.6e}"
    
    # Parse the scientific notation
    if 'e' in sci_str:
        mantissa_str, exp_str = sci_str.split('e')
        mantissa = float(mantissa_str)
        exponent = int(exp_str)
        
        # Format mantissa to remove decimal point
        # e.g., 1.234567 -> 1p234567
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

def ensure_ground_truth_exists():
    """
    Check if ground truth data exists, generate if needed.
    Returns the path to the ground truth simulation output.
    """
    print("=== Checking Ground Truth Data Availability ===")
    
    # Look for existing ground truth data
    simulation_outputs_dir = "/Users/joshmcneely/uguca/build/simulations/simulation_outputs"
    gt_pattern = "time_int_opt_gt_ground_truth"
    
    potential_gt_dirs = []
    if os.path.exists(simulation_outputs_dir):
        for item in os.listdir(simulation_outputs_dir):
            if gt_pattern in item:
                potential_gt_dirs.append(os.path.join(simulation_outputs_dir, item))
    
    if potential_gt_dirs:
        gt_dir = potential_gt_dirs[0]  # Use first found
        print(f"Found existing ground truth data: {gt_dir}")
        
        # Verify essential files exist
        gt_coords = os.path.join(gt_dir, "time_int_opt_gt.coords")
        gt_time = os.path.join(gt_dir, "time_int_opt_gt.time")
        gt_restart_dir = os.path.join(gt_dir, "time_int_opt_gt_displacement_dump-restart")
        
        if all(os.path.exists(f) for f in [gt_coords, gt_time]) and os.path.isdir(gt_restart_dir):
            print("Ground truth data verified complete.")
            return gt_dir
        else:
            print("Ground truth data incomplete, will regenerate.")
    
    # Generate ground truth data
    print("Generating ground truth data...")
    return generate_ground_truth()

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
        raise RuntimeError("Could not determine project root directory.")
    
    build_dir = os.path.join(project_root, "build", "simulations")
    source_sim_dir = os.path.join(project_root, "simulations")
    
    executable_path = os.path.join(build_dir, "time_int_opt")
    gt_input_file = os.path.join(source_sim_dir, "input_files", "ground_truth.in")
    
    if not os.path.exists(executable_path):
        raise RuntimeError(f"Simulation executable not found: {executable_path}")
    if not os.path.exists(gt_input_file):
        raise RuntimeError(f"Ground truth input file not found: {gt_input_file}")
    
    # Copy input file to build directory
    input_files_dir = os.path.join(build_dir, "input_files")
    os.makedirs(input_files_dir, exist_ok=True)
    
    gt_input_build = os.path.join(input_files_dir, "ground_truth.in")
    shutil.copy2(gt_input_file, gt_input_build)
    
    try:
        print(f"Executing: {executable_path} ground_truth.in")
        subprocess.run([executable_path, "ground_truth.in"], 
                      check=True, cwd=build_dir)
        print("Ground truth generation completed successfully.")
        
        # Find and return the output directory
        simulation_outputs_dir = os.path.join(build_dir, "simulation_outputs")
        for item in os.listdir(simulation_outputs_dir):
            if "time_int_opt_gt_ground_truth" in item:
                return os.path.join(simulation_outputs_dir, item)
        
        raise RuntimeError("Ground truth output directory not found after generation.")
        
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Ground truth generation failed: {e}")
    finally:
        # Clean up temporary input file
        if os.path.exists(gt_input_build):
            os.remove(gt_input_build)

def process_ground_truth_with_noise(noise_stddev):
    """
    Process ground truth data with specific noise level.
    Returns the path to the processed noisy data.
    """
    print(f"--- Processing Ground Truth with Noise (stddev={noise_stddev:.2e}) ---")
    
    script_path = os.path.abspath(os.path.dirname(__file__))
    process_script = os.path.join(script_path, "process_ground_truth_displacement.py")
    
    if not os.path.exists(process_script):
        raise RuntimeError(f"Processing script not found: {process_script}")
    
    try:
        cmd = [
            "python", process_script,
            "time_int_opt_gt",
            "--add-noise",
            "--noise-stddev", str(noise_stddev),
            "--output-noisy-data"
        ]
        print(f"Executing: {' '.join(cmd)}")
        subprocess.run(cmd, check=True, cwd=os.path.dirname(script_path))
        
        # Return path to noisy data
        simulation_outputs_dir = "/Users/joshmcneely/uguca/build/simulations/simulation_outputs"
        noisy_data_dir = os.path.join(simulation_outputs_dir, "noisy_gt_data_input")
        
        if os.path.exists(noisy_data_dir):
            print(f"Noisy data generated: {noisy_data_dir}")
            return noisy_data_dir
        else:
            raise RuntimeError("Noisy data directory not found after processing.")
            
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Noise processing failed: {e}")

def run_data_driven_simulation(noise_stddev, base_input_file):
    """
    Run data-driven simulation with specific noise parameters.
    """
    print(f"--- Running Data-Driven Simulation (noise_stddev={noise_stddev:.2e}) ---")
    
    # Find project directories
    script_path = os.path.abspath(os.path.dirname(__file__))
    project_root = script_path
    while os.path.basename(project_root) != 'uguca' and project_root != '/':
        project_root = os.path.dirname(project_root)
    
    build_dir = os.path.join(project_root, "build", "simulations")
    source_sim_dir = os.path.join(project_root, "simulations")
    
    executable_path = os.path.join(build_dir, "time_int_opt")
    base_input_path = os.path.join(source_sim_dir, "input_files", base_input_file)
    
    # Read base input file
    with open(base_input_path, 'r') as f:
        base_content = f.read()
    
    # Create unique simulation name using high-precision formatting
    noise_str = format_noise_for_filename(noise_stddev)
    sim_name = f"time_int_opt_noise_{noise_str}"
    
    # Modify content for this noise level
    content = re.sub(r"sim_name\s*=\s*.*", f"sim_name = {sim_name}", base_content)
    content = re.sub(r"noise_stddev_sim\s*=\s*.*", f"noise_stddev_sim = {noise_stddev:.12e}", content)  # High precision
    content = re.sub(r"nb_dumps\s*=\s*.*", f"nb_dumps = 1", content)  # Only final dump for speed
    
    # Write temporary input file
    input_files_dir = os.path.join(build_dir, "input_files")
    os.makedirs(input_files_dir, exist_ok=True)
    
    temp_input_filename = f"{sim_name}.in"
    temp_input_path = os.path.join(input_files_dir, temp_input_filename)
    
    with open(temp_input_path, 'w') as f:
        f.write(content)
    
    try:
        print(f"Executing: {executable_path} {temp_input_filename}")
        print(f"Noise precision: {noise_stddev:.12e}")
        subprocess.run([executable_path, temp_input_filename], 
                      check=True, cwd=build_dir)
        print(f"Data-driven simulation completed for noise_stddev={noise_stddev:.2e}")
        return sim_name
        
    except subprocess.CalledProcessError as e:
        print(f"Error in data-driven simulation for noise_stddev {noise_stddev:.2e}: {e}")
        return None
    finally:
        # Clean up temporary input file
        if os.path.exists(temp_input_path):
            os.remove(temp_input_path)

def run_noise_study(base_input_file, noise_stddevs):
    """
    Run comprehensive noise parameter study.
    """
    print(f"Starting COMPREHENSIVE noise covariance parameter study")
    print(f"Base input file: {base_input_file}")
    print(f"Testing {len(noise_stddevs)} noise levels")
    print(f"Range: {min(noise_stddevs):.2e} to {max(noise_stddevs):.2e}")
    print()
    
    # Step 1: Ensure ground truth data exists
    try:
        gt_dir = ensure_ground_truth_exists()
        print(f"Ground truth data available at: {gt_dir}")
    except Exception as e:
        print(f"Failed to ensure ground truth data: {e}")
        return
    
    # Track results and used names to detect duplicates
    successful_runs = 0
    failed_runs = 0
    results = []
    used_names = set()
    duplicate_count = 0
    
    # Step 2: For each noise level, process data and run simulation
    for i, noise_stddev in enumerate(noise_stddevs, 1):
        print(f"\n{'='*60}")
        print(f"NOISE STUDY: {i}/{len(noise_stddevs)} - Noise StdDev = {noise_stddev:.2e}")
        print(f"{'='*60}")
        
        try:
            # Create unique simulation name and check for duplicates
            noise_str = format_noise_for_filename(noise_stddev)
            sim_name = f"time_int_opt_noise_{noise_str}"
            
            # Check for duplicates and add safety counter if needed
            original_sim_name = sim_name
            counter = 1
            while sim_name in used_names:
                duplicate_count += 1
                sim_name = f"{original_sim_name}_dup{counter}"
                counter += 1
                print(f"WARNING: Duplicate name detected, using: {sim_name}")
            
            used_names.add(sim_name)
            
            # Step 2a: Process ground truth with current noise level
            noisy_data_dir = process_ground_truth_with_noise(noise_stddev)
            
            # Step 2b: Run data-driven simulation
            actual_sim_name = run_data_driven_simulation(noise_stddev, base_input_file)
            
            if actual_sim_name:
                successful_runs += 1
                results.append({
                    'noise_stddev': noise_stddev,
                    'noise_variance': noise_stddev ** 2,
                    'sim_name': actual_sim_name,
                    'status': 'completed'
                })
                print(f"✓ Completed noise_stddev = {noise_stddev:.2e}")
            else:
                failed_runs += 1
                results.append({
                    'noise_stddev': noise_stddev,
                    'noise_variance': noise_stddev ** 2,
                    'sim_name': None,
                    'status': 'failed'
                })
                print(f"✗ Failed noise_stddev = {noise_stddev:.2e}")
                
        except Exception as e:
            failed_runs += 1
            results.append({
                'noise_stddev': noise_stddev,
                'noise_variance': noise_stddev ** 2,
                'sim_name': None,
                'status': f'error: {str(e)}'
            })
            print(f"✗ Error for noise_stddev = {noise_stddev:.2e}: {e}")
    
    # Summary
    print("\n" + "="*80)
    print("NOISE COVARIANCE PARAMETER STUDY SUMMARY")
    print("="*80)
    
    if duplicate_count > 0:
        print(f"WARNING: {duplicate_count} duplicate names were detected and resolved.")
    
    print(f"Total simulations attempted: {len(noise_stddevs)}")
    print(f"Successful runs: {successful_runs}")
    print(f"Failed runs: {failed_runs}")
    print(f"Success rate: {100*successful_runs/len(noise_stddevs):.1f}%")
    print()
    
    if successful_runs > 0:
        print("Completed simulations:")
        print("Noise StdDev | Noise Variance | Simulation Name")
        print("-" * 60)
        for result in results:
            if result['status'] == 'completed':
                print(f"{result['noise_stddev']:12.2e} | {result['noise_variance']:14.2e} | {result['sim_name']}")
        
        # Check for any remaining issues with unique naming
        sim_names = [r['sim_name'] for r in results if r['status'] == 'completed']
        unique_names = set(sim_names)
        if len(sim_names) != len(unique_names):
            print(f"WARNING: Found {len(sim_names) - len(unique_names)} duplicate simulation names in completed results!")
        else:
            print("✓ All simulation names are unique.")
        
        print()
        print("Next steps:")
        print("1. Analyze displacement accuracy vs noise level")
        print("2. Plot error metrics as function of noise covariance")
        print("3. Determine optimal noise estimation for real data")
    
    if failed_runs > 0:
        print()
        print("Failed simulations:")
        for result in results:
            if result['status'] != 'completed':
                print(f"  noise_stddev = {result['noise_stddev']:.2e}: {result['status']}")
    
    print("="*80)

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--design":
        design_noise_study()
    elif len(sys.argv) > 1 and sys.argv[1] == "--run":
        base_file = "deviated_params_dd_noisy_scenario.in"
        
        # Generate noise study values
        noise_stddevs = design_noise_study()
        
        run_noise_study(base_file, noise_stddevs)
    else:
        print("Noise Covariance Parameter Study Script")
        print("Usage:")
        print("  python run_noise_study.py --design  # Show noise level design")
        print("  python run_noise_study.py --run     # Run the noise parameter study")
        print()
        print("This study tests different noise levels in measurement data to characterize")
        print("how measurement uncertainty affects data-driven simulation accuracy.")
        print()
        print("Workflow for each noise level:")
        print("  1. Ensure ground truth data exists (generate if needed)")
        print("  2. Process ground truth data with specific noise level")
        print("  3. Run data-driven simulation with matching noise parameters")
        print("  4. Collect results for analysis")
