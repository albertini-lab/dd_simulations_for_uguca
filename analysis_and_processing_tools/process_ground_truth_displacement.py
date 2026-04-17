import numpy as np
import os
import re # Import regular expressions for parsing filenames
import argparse
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation # For animations
import shutil # For copying files
import glob
import struct
import sys
import datetime

def find_simulation_files(sim_name_base):
    """
    Search for simulation files in the organized directory structure.
    Returns a dictionary with the paths to the found files.
    """
    found_files = {
        'coords_file': None,
        'time_file': None,
        'data_dir': None,
        'base_path': None
    }
    
    # Search patterns for different locations
    search_patterns = [
        f"simulation_outputs/{sim_name_base}_ground_truth",
        f"simulation_outputs/{sim_name_base}_*",
        f"simulation_outputs/*{sim_name_base}*",
        f"simulation_outputs/{sim_name_base}",  # For processed data like noisy_gt_data_input
        ".",
        ".."
    ]
    
    for pattern in search_patterns:
        # Use glob to handle wildcards in directory names
        for base_dir in glob.glob(pattern):
            if not os.path.isdir(base_dir):
                continue

            # Look for coords file
            coords_path = os.path.join(base_dir, f"{sim_name_base}.coords")
            if found_files['coords_file'] is None and os.path.exists(coords_path):
                found_files['coords_file'] = coords_path
                found_files['base_path'] = base_dir
            
            # Look for time file
            time_path = os.path.join(base_dir, f"{sim_name_base}.time")
            if found_files['time_file'] is None and os.path.exists(time_path):
                found_files['time_file'] = time_path
                if found_files['base_path'] is None:
                    found_files['base_path'] = base_dir
                    
            # Look for data directory (nested restart directory for ground truth)
            data_dir_path = os.path.join(base_dir, f"{sim_name_base}_displacement_dump-restart")
            if found_files['data_dir'] is None and os.path.isdir(data_dir_path):
                found_files['data_dir'] = data_dir_path
                if found_files['base_path'] is None:
                    found_files['base_path'] = base_dir
            
            # Also look for processed data restart directory (for noisy_gt_data_input style)
            processed_data_dir_path = os.path.join(base_dir, f"{sim_name_base}-restart")
            if found_files['data_dir'] is None and os.path.isdir(processed_data_dir_path):
                found_files['data_dir'] = processed_data_dir_path
                if found_files['base_path'] is None:
                    found_files['base_path'] = base_dir
    
    return found_files

# How to use the script:
# Run from the terminal.
#
# Positional Arguments:
#   original_sim_name_base: Base name of the original simulation (e.g., 'time_int_opt_gt').
#   step_frequency (optional): Load every Nth step. Default is 1 (load all steps found).
#
# --- Basic Usage ---
# Load and plot (no noise, no zeroing, no new output, no animation):
#   `python process_ground_truth_displacement.py time_int_opt_gt`
# Load every 10th step and plot:
#   `python process_ground_truth_displacement.py time_int_opt_gt 10`
#
# --- Adding Noise ---
# Load, add default noise (stddev 1e-6 by default from script), and plot:
#   `python process_ground_truth_displacement.py time_int_opt_gt --add-noise`
# Load, add custom fixed noise (e.g., stddev 0.005), and plot:
#   `python process_ground_truth_displacement.py time_int_opt_gt --add-noise --noise-stddev 0.005`
#
# --- Zeroing Out Data ---
# Load, zero out a default fraction (10%) of data points, and plot:
#   `python process_ground_truth_displacement.py time_int_opt_gt --enable-zero-out`
# Load, zero out a custom fraction (e.g., 5%) of data points, and plot:
#   `python process_ground_truth_displacement.py time_int_opt_gt --enable-zero-out --zero-out-fraction 0.05`
#
# --- Combining Noise and Zeroing ---
# Load, add default noise, then zero out 10% of data, and plot:
#   `python process_ground_truth_displacement.py time_int_opt_gt --add-noise --enable-zero-out`
#
# --- Outputting Processed Data ---
# The output data will be saved in a directory structure named 'noisy_gt_data_input'
# (containing 'noisy_gt_data_input.coords', '.time', '.info', and a '-restart' subdirectory).
# This structure is expected by 'data_driven_noisy.in'.
#
# Load, add noise, output processed data, and plot:
#   `python process_ground_truth_displacement.py time_int_opt_gt --add-noise --output-noisy-data`
#
# --- Generating Animations ---
# Animations are saved as MP4 files (e.g., 'time_int_opt_gt_processed_preview_displacement_evolution_subplots.mp4').
#
# Load and generate displacement animation (no noise, no zeroing):
#   `python process_ground_truth_displacement.py time_int_opt_gt --animate-displacement`
#
# --- Using Direct Input Directory for .out files (for animating custom simulation outputs) ---
# Animate data where .coords and .time are from 'my_sim_output' and .out files are in 'my_sim_output/my_sim_output-restart':
#   `python process_ground_truth_displacement.py my_sim_output --direct-input-dir-for-out-files my_sim_output/my_sim_output-restart --animate-displacement`
#
# General Notes:
# - The plotting part of the script will always show the data *after* any modifications (noise, zeroing) have been applied.
# - The saved files (if --output-noisy-data is used) will contain this modified data.
# - If both noise and zeroing are enabled, noise is applied first, then zeroing.




def animate_displacement_evolution(x_coords, time_data, top_disp, bot_disp, sim_name_base):
    """Creates an animation of the displacement evolution."""
    print(f"--- Animating Displacement for {sim_name_base} ---")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    import numpy as np
    # Example: animate top displacement profile over time
    fig, ax = plt.subplots(figsize=(10, 6))
    line_top, = ax.plot([], [], 'b-', label='Top Disp')
    line_bot, = ax.plot([], [], 'g-', label='Bot Disp')
    ax.set_xlim(np.min(x_coords), np.max(x_coords))
    # Set y-limits based on data range
    y_min = min(np.min(top_disp), np.min(bot_disp))
    y_max = max(np.max(top_disp), np.max(bot_disp))
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel('Position')
    ax.set_ylabel('Displacement')
    ax.set_title(f'Displacement Evolution: {sim_name_base}')
    ax.legend()

    def update(frame):
        line_top.set_data(x_coords, top_disp[frame][0])
        line_bot.set_data(x_coords, bot_disp[frame][0])
        ax.set_title(f'Displacement Evolution: {sim_name_base} (Step {frame})')
        return line_top, line_bot

    num_frames = top_disp.shape[0]
    anim = FuncAnimation(fig, update, frames=num_frames, interval=30, blit=True)

    # Save animation as MP4
    mp4_filename = f"{sim_name_base}_processed_preview_displacement_evolution_subplots.mp4"
    print(f"Saving animation to {mp4_filename} ...")
    try:
        anim.save(mp4_filename, writer='ffmpeg', fps=30)
        print(f"Animation saved: {mp4_filename}")
    except Exception as e:
        print(f"Error saving animation: {e}")
        print("Make sure ffmpeg is installed and available in your PATH.")
    plt.close(fig)

# --- Argument Parsing --- (Example of where to add the new argument)
# In the main section, you would add:
# parser.add_argument("--output-dir-name", type=str, default=None, help="Specify a custom output directory name for processed data.")


def load_coords_from_file(sim_name_base):
    """Loads x-coordinates from the .coords file."""
    # Find simulation files in organized structure
    files = find_simulation_files(sim_name_base)
    
    if files['coords_file'] is None:
        print(f"Error: Could not find coordinates file for simulation: {sim_name_base}")
        print("Searched in: simulation_outputs/, current directory, and parent directory")
        return None
        
    coords_file = files['coords_file']
    x_coords = []
    try:
        with open(coords_file, 'r') as fl:
            lines = fl.readlines()
        for line in lines:
            if line.strip():
                # Assuming coords file has one x-coordinate per line, possibly with other values
                x_coords.append(float(line.strip().split()[0]))
        if not x_coords:
            print(f"Error: No data found in coordinates file: {coords_file}")
            return None
        return np.array(x_coords)
    except FileNotFoundError:
        print(f"Error: Coordinates file not found: {coords_file}")
        return None
    except Exception as e:
        print(f"Error parsing coordinate data in {coords_file}: {e}")
        return None


def load_time_data_from_file(sim_name_base):
    """Loads step numbers and time values from the .time file."""
    # Find simulation files in organized structure
    files = find_simulation_files(sim_name_base)
    
    if files['time_file'] is None:
        print(f"Error: Could not find time file for simulation: {sim_name_base}")
        print("Searched in: simulation_outputs/, current directory, and parent directory")
        return None, None
        
    time_file = files['time_file']
    time_data_raw = []  # List of (step_number, time_value) tuples
    try:
        with open(time_file, 'r') as fl:
            lines = fl.readlines()
        for line in lines:
            if line.strip():
                parts = line.strip().split()
                if len(parts) == 2:
                    time_data_raw.append((int(parts[0]), float(parts[1])))
                else:
                    print(f"Warning: Skipping malformed line in {time_file}: {line.strip()}")
        if not time_data_raw:
            print(f"Error: No data found in time file: {time_file}")
            return None, None

        step_numbers = np.array([item[0] for item in time_data_raw])
        time_values = np.array([item[1] for item in time_data_raw])
        return step_numbers, time_values
    except FileNotFoundError:
        print(f"Error: Time file not found: {time_file}")
        return None, None
    except Exception as e:
        print(f"Error parsing time data in {time_file}: {e}")
        return None, None


def load_displacement_at_step(data_files_path, field_basename, step, num_nodes, num_components=2, format_type='binary'):
    """Loads displacement data for a single field at a single step."""
    file_path = os.path.join(data_files_path, f"{field_basename}.proc0.s{step}.out")
    
    try:
        if format_type.lower() == 'binary':
            # Binary format: read as float32 values
            expected_values = num_nodes * num_components
            with open(file_path, 'rb') as f:
                # Read all data as float32
                data = f.read()
                num_floats = len(data) // 4  # 4 bytes per float32
                if num_floats != expected_values:
                    print(f"Warning: Binary file {file_path} has {num_floats} values, expected {expected_values}")
                    # Try to handle the case where we have the right amount of data
                    if num_floats >= expected_values:
                        print(f"Using first {expected_values} values from file")
                        num_floats = expected_values
                    else:
                        print(f"Error: Not enough data in binary file")
                        return None
                
                # Unpack as float32 values
                values = struct.unpack(f'{num_floats}f', data[:num_floats*4])
                values = np.array(values, dtype=np.float64)  # Convert to double precision
        else:
            # ASCII format (existing code)
            with open(file_path, 'r') as f:
                line = f.readline()
            values = np.array([float(v) for v in line.strip().split()])
            expected_values = num_nodes * num_components
            if len(values) != expected_values:
                print(f"Error: ASCII file {file_path} has {len(values)} values, expected {expected_values}")
                return None
        
        # Reshape to (components, nodes)
        if len(values) == expected_values:
            displacement_data = values.reshape((num_components, num_nodes))
            return displacement_data
        else:
            print(f"Error: Could not reshape data properly for {file_path}")
            return None
            
    except FileNotFoundError:
        print(f"Error: Displacement data file not found: {file_path}")
        return None
    except Exception as e:
        print(f"Error reading displacement file {file_path}: {e}")
        return None


def save_nodal_field_at_step(output_dir_path, field_basename, step, data_for_step, num_components, num_nodes, format_type='binary'):
    """Saves nodal field data for a single step."""
    file_path = os.path.join(output_dir_path, f"{field_basename}.proc0.s{step}.out")
    flat_data = data_for_step.reshape(-1)
    
    try:
        if format_type.lower() == 'binary':
            # Save as binary float32
            with open(file_path, 'wb') as f:
                # Convert to float32 and write
                float32_data = flat_data.astype(np.float32)
                f.write(float32_data.tobytes())
        else:
            # Save as ASCII (existing code)
            with open(file_path, 'w') as f:
                f.write(" ".join(map(str, flat_data)))
    except IOError as e:
        print(f"Error writing data to {file_path}: {e}")


def save_processed_data(output_dir_name, x_coords, time_data, top_disp_data, bot_disp_data, top_weight_data, bot_weight_data, original_sim_name_base, output_format='binary'):
    """Saves the processed (e.g., noisy, sparse) data to a new directory."""
    print(f"--- Saving Processed Data to Directory: {output_dir_name} ---")
    
    # Create directories - ensure output goes to simulation_outputs
    if not os.path.isabs(output_dir_name):
        # If it's a relative path, put it in simulation_outputs
        output_base_dir = os.path.join("simulation_outputs", output_dir_name)
    else:
        output_base_dir = output_dir_name
    
    output_base_dir = os.path.abspath(output_base_dir)
    output_restart_dir = os.path.join(output_base_dir, f"{os.path.basename(output_base_dir)}-restart")
    os.makedirs(output_restart_dir, exist_ok=True)

    # --- Save .coords, .time, and .info files ---
    coords_out_path = os.path.join(output_base_dir, f"{os.path.basename(output_base_dir)}.coords")
    with open(coords_out_path, 'w') as f:
        for x in x_coords:
            f.write(f"{x}\n")

    time_out_path = os.path.join(output_base_dir, f"{os.path.basename(output_base_dir)}.time")
    with open(time_out_path, 'w') as f:
        for step, time_val in time_data:
            f.write(f"{step} {time_val}\n")

    info_out_path = os.path.join(output_base_dir, f"{os.path.basename(output_base_dir)}.info")
    with open(info_out_path, 'w') as f:
        f.write(f"# Ground truth data processed from simulation: {original_sim_name_base}\n")
        f.write(f"# Processed on: {datetime.datetime.now()}\n")

    # --- Save displacement data (.out files) ---
    num_nodes = len(x_coords)
    include_bot_disp = bot_disp_data is not None and getattr(bot_disp_data, "size", 0) > 0
    include_bot_weights = bot_weight_data is not None and getattr(bot_weight_data, "size", 0) > 0

    for i, (step, _) in enumerate(time_data):
        # Save displacement and weight data with correct field names for data-driven mode
        # The simulation expects "interface_*" field names for all data-driven fields
        save_nodal_field_at_step(output_restart_dir, "interface_top_disp", step, top_disp_data[i], 2, num_nodes, output_format)
        if include_bot_disp:
            save_nodal_field_at_step(output_restart_dir, "interface_bot_disp", step, bot_disp_data[i], 2, num_nodes, output_format)
        # Save weight data - this is critical for data-driven simulations to respect sparsity
        save_nodal_field_at_step(output_restart_dir, "interface_top_weights", step, top_weight_data[i], 2, num_nodes, output_format)
        if include_bot_weights:
            save_nodal_field_at_step(output_restart_dir, "interface_bot_weights", step, bot_weight_data[i], 2, num_nodes, output_format)

    print("--- Finished Saving Processed Data ---")

def load_all_ground_truth_displacements(original_sim_name_base, step_frequency=1,
                                        add_noise=False, noise_stddev=1e-6,
                                        enable_spatial_sparsity=False,
                                        sparsity_type="random_fraction",
                                        sparsity_fraction=0.1,
                                        sparsity_n_nodes=10,
                                        direct_input_dir_for_out_files=None,
                                        force_format=None):
    """
    Loads ground truth displacement data with format auto-detection.
    
    Parameters:
    -----------
    force_format : str, optional
        Force a specific format ('ascii' or 'binary'). If None, auto-detects.
    """
    print(f"--- Loading Displacement Data for: {original_sim_name_base} ---")

    x_coordinates = load_coords_from_file(original_sim_name_base)
    if x_coordinates is None:
        return None
    num_nodes = len(x_coordinates)
    print(f"Successfully loaded {num_nodes} nodal coordinates from '{original_sim_name_base}.coords'.")

    sparse_step_numbers, sparse_time_axis = load_time_data_from_file(original_sim_name_base)
    if sparse_time_axis is None or sparse_step_numbers is None or len(sparse_step_numbers) == 0:
        print(f"Warning: Could not load sparse .time file ('{original_sim_name_base}.time'). Time axis will be a placeholder if needed.")
        # Keep them as None or empty arrays to indicate missing sparse time data
        sparse_step_numbers, sparse_time_axis = np.array([]), np.array([])
    else:
        print(f"Sparse '.time' file ('{original_sim_name_base}.time') loaded, indicating simulation ran up to at least step {sparse_step_numbers[-1]} (t={sparse_time_axis[-1]:.2e}s).")

    # --- Path to the .out data files ---
    if direct_input_dir_for_out_files:
        path_to_out_files = direct_input_dir_for_out_files
        print(f"Using direct path for .out files: {os.path.abspath(path_to_out_files)}")
    else:
        # Find simulation files in organized structure
        files = find_simulation_files(original_sim_name_base)
        
        if files['data_dir'] is not None:
            path_to_out_files = files['data_dir']
            print(f"Found displacement .out data files in organized structure: {os.path.abspath(path_to_out_files)}")
        else:
            # Fallback to legacy structure
            gt_dir_name = f"{original_sim_name_base}_displacement_dump"
            gt_restart_subdir = f"{gt_dir_name}-restart" 
            path_to_out_files = os.path.join(gt_dir_name, gt_restart_subdir)
            print(f"Falling back to legacy structure. Expecting displacement .out data files in: {os.path.abspath(path_to_out_files)}")

    if not os.path.isdir(path_to_out_files):
        print(f"Error: Directory for .out files not found: {path_to_out_files}")
        return None

    # --- Discover available steps from the data files in path_to_out_files ---
    actual_steps_in_path_set = set()
    # Use a common field name to discover steps, assuming if one component exists, the step exists.
    # For processed data (like noisy_gt_data_input), use "top_disp" and "bot_disp"
    # For original ground truth data, use "interface_top_disp" and "interface_bot_disp"
    
    # First, check what types of files are actually in the directory
    try:
        files_in_dir = os.listdir(path_to_out_files)
    except FileNotFoundError:
        print(f"Error: Could not list files in {path_to_out_files} to determine available steps.")
        return None
    
    # Determine the field names based on what files are actually present
    has_processed_top = any(f.startswith("top_disp.proc0.s") for f in files_in_dir)
    has_processed_bot = any(f.startswith("bot_disp.proc0.s") for f in files_in_dir)
    has_original_top = any(f.startswith("interface_top_disp.proc0.s") for f in files_in_dir)
    has_original_bot = any(f.startswith("interface_bot_disp.proc0.s") for f in files_in_dir)

    bot_data_available = False

    if has_processed_top:
        # Processed data format
        file_pattern = re.compile(r"top_disp\.proc0\.s(\d+)\.out")
        top_disp_field_basename = "top_disp"
        bot_disp_field_basename = "bot_disp"
        bot_data_available = has_processed_bot
        print(f"Detected processed data format (top_disp/bot_disp) in {path_to_out_files}")
        if not bot_data_available:
            print("Note: No bot_disp files detected. Proceeding with top interface only.")
    elif has_original_top:
        # Original ground truth format
        file_pattern = re.compile(r"interface_top_disp\.proc0\.s(\d+)\.out")
        top_disp_field_basename = "interface_top_disp"
        bot_disp_field_basename = "interface_bot_disp"
        bot_data_available = has_original_bot
        print(f"Detected original ground truth format (interface_top_disp/interface_bot_disp) in {path_to_out_files}")
        if not bot_data_available:
            print("Note: No interface_bot_disp files detected. Proceeding with top interface only.")
    else:
        print(f"Error: No displacement data files found in {path_to_out_files}.")
        print(f"Expected files starting with 'top_disp.proc0.s' or 'interface_top_disp.proc0.s'")
        print(f"Found files: {files_in_dir[:10]}...")  # Show first 10 files for debugging
        return None
    
    # Now scan for step numbers using the determined pattern
    for filename in files_in_dir:
        match = file_pattern.match(filename)
        if match:
            actual_steps_in_path_set.add(int(match.group(1)))
    
    if not actual_steps_in_path_set:
        print(f"Error: No displacement data files found with pattern {file_pattern.pattern}.")
        return None

    sorted_actual_steps_in_path = sorted(list(actual_steps_in_path_set))
    print(f"Discovered {len(sorted_actual_steps_in_path)} available steps in data directory '{os.path.basename(path_to_out_files)}'.")
    if len(sorted_actual_steps_in_path) > 10:
        print(f"  Examples: {sorted_actual_steps_in_path[:5]}...{sorted_actual_steps_in_path[-5:]}")
    else:
        print(f"  Steps: {sorted_actual_steps_in_path}")
    
    min_available_step = sorted_actual_steps_in_path[0]
    max_available_step = sorted_actual_steps_in_path[-1]

    # --- Filter these available steps by step_frequency ---
    if step_frequency > 1:
        print(f"--- Applying step frequency: Loading every {step_frequency}th step from the available {len(sorted_actual_steps_in_path)} steps. ---")
        final_step_numbers_to_load_list = [s for i, s in enumerate(sorted_actual_steps_in_path) if i % step_frequency == 0]
        if not final_step_numbers_to_load_list and sorted_actual_steps_in_path: # Ensure at least one if possible
             final_step_numbers_to_load_list = [sorted_actual_steps_in_path[0]]
        final_step_numbers_to_load = np.array(final_step_numbers_to_load_list)
        print(f"Filtered to {len(final_step_numbers_to_load)} steps due to frequency {step_frequency}.")
    else:
        final_step_numbers_to_load = np.array(sorted_actual_steps_in_path)

    if len(final_step_numbers_to_load) == 0:
        print("Error: No steps to process after filtering available steps by frequency (or no steps found initially).")
        return None

    # --- Generate time axis for the final_step_numbers_to_load using sparse time data ---
    final_time_axis = np.zeros(len(final_step_numbers_to_load))
    if sparse_time_axis.size > 0 and sparse_step_numbers.size > 0:
        if sparse_step_numbers.size > 1:
            # Interpolate, np.interp handles ends by using end values if steps are outside range
            final_time_axis = np.interp(final_step_numbers_to_load, sparse_step_numbers, sparse_time_axis)
            
            # Optional: More explicit extrapolation for steps beyond the sparse data range
            last_sparse_step = sparse_step_numbers[-1]
            if final_step_numbers_to_load[-1] > last_sparse_step:
                # Find where extrapolation should start
                extrap_start_index = np.searchsorted(final_step_numbers_to_load, last_sparse_step, side='right')
                if extrap_start_index < len(final_step_numbers_to_load) and extrap_start_index > 0: # Check if there are steps to extrapolate
                    # Calculate an approximate dt from the last two points of sparse data
                    dt_approx = (sparse_time_axis[-1] - sparse_time_axis[-2]) / (sparse_step_numbers[-1] - sparse_step_numbers[-2]) \
                                if (sparse_step_numbers[-1] - sparse_step_numbers[-2]) != 0 else 0
                    
                    for i in range(extrap_start_index, len(final_step_numbers_to_load)):
                        # Extrapolate linearly from the last interpolated value for a step within sparse range
                        # Find the last step *before* extrap_start_index that was in sparse_step_numbers range for a good base
                        # Or simply from final_time_axis[extrap_start_index-1]
                        time_at_last_known_interp = final_time_axis[extrap_start_index-1]
                        step_at_last_known_interp = final_step_numbers_to_load[extrap_start_index-1]
                        final_time_axis[i] = time_at_last_known_interp + dt_approx * (final_step_numbers_to_load[i] - step_at_last_known_interp)
        elif sparse_step_numbers.size == 1: # Only one point in .time file
            final_time_axis.fill(sparse_time_axis[0])
            print(f"Warning: Sparse .time file has only one entry. Using t={sparse_time_axis[0]} for all loaded steps.")
        # If sparse_time_axis is empty, final_time_axis remains zeros, will be placeholder later
    
    if np.all(final_time_axis == 0) and final_time_axis.size > 0 : # If still all zeros (e.g. no sparse data)
        final_time_axis = final_step_numbers_to_load.astype(float) * 1e-9 # Placeholder based on step number
        print("Warning: Time axis is a placeholder (step_number * 1e-9) due to insufficient or non-overlapping sparse time data.")


    print(f"Attempting to load data for {len(final_step_numbers_to_load)} time points:")
    if len(final_step_numbers_to_load) > 0:
        print(f"  Range: from step {final_step_numbers_to_load[0]} (approx. t={final_time_axis[0]:.2e}s) "
              f"to step {final_step_numbers_to_load[-1]} (approx. t={final_time_axis[-1]:.2e}s).")

    # Field names were determined earlier based on data type
    num_components = 2
    all_top_displacements_list = []
    all_bot_displacements_list = []

    # Auto-detect format if not forced
    if force_format is None:
        # Try to detect format by reading the first file
        test_file = os.path.join(path_to_out_files, f"{top_disp_field_basename}.proc0.s{final_step_numbers_to_load[0]}.out")
        if os.path.exists(test_file):
            with open(test_file, 'rb') as f:
                first_bytes = f.read(100)  # Read more bytes for better detection
            
            # Check if it looks like binary (check for null bytes and non-printable characters)
            null_count = first_bytes.count(0)
            if null_count > 10:  # If we have many null bytes, it's likely binary
                detected_format = 'binary'
            else:
                # Try to decode as ASCII
                try:
                    decoded = first_bytes.decode('ascii')
                    # Check if it looks like ASCII numbers
                    if any(char.isdigit() or char in '.-e+ \n\t' for char in decoded):
                        detected_format = 'ascii'
                    else:
                        detected_format = 'binary'
                except UnicodeDecodeError:
                    detected_format = 'binary'
        else:
            detected_format = 'binary'  # Default to binary since your simulation generates binary
        
        print(f"Auto-detected format: {detected_format}")
    else:
        detected_format = force_format
        print(f"Using forced format: {detected_format}")

    for current_step_to_load in final_step_numbers_to_load:
        top_disp_at_step = load_displacement_at_step(path_to_out_files, top_disp_field_basename, 
                                                   current_step_to_load, num_nodes, num_components, detected_format)
        if top_disp_at_step is None:
            print(f"Failed to load top displacement data for step {current_step_to_load}. Aborting.")
            return None

        bot_disp_at_step = None
        if bot_data_available:
            bot_disp_at_step = load_displacement_at_step(path_to_out_files, bot_disp_field_basename, 
                                                       current_step_to_load, num_nodes, num_components, detected_format)
            if bot_disp_at_step is None:
                print(f"Failed to load bot displacement data for step {current_step_to_load}. Aborting.")
                return None

        all_top_displacements_list.append(top_disp_at_step)
        if bot_data_available:
            all_bot_displacements_list.append(bot_disp_at_step)

    final_top_displacements = np.array(all_top_displacements_list)
    if bot_data_available:
        final_bot_displacements = np.array(all_bot_displacements_list)
    else:
        final_bot_displacements = np.array([])
    
    if final_top_displacements.size == 0 and final_bot_displacements.size == 0 and len(final_step_numbers_to_load) > 0:
        print(f"Warning: Displacement arrays are empty after attempting to load {len(final_step_numbers_to_load)} steps.")

    final_top_weights = np.ones_like(final_top_displacements) if final_top_displacements.size > 0 else np.array([])
    final_bot_weights = np.ones_like(final_bot_displacements) if final_bot_displacements.size > 0 else np.array([])

    if add_noise:
        if noise_stddev <= 0:
            print("--- Noise addition requested but noise_stddev must be positive. No noise added. ---")
        elif final_top_displacements.size == 0 and final_bot_displacements.size == 0:
            print("--- Noise addition requested but no displacement data available. No noise added. ---")
        else:
            print(f"--- Adding random noise (stddev={noise_stddev:.2e}) to available displacement data ---")
            if final_top_displacements.size > 0:
                original_top_mean = np.mean(np.abs(final_top_displacements))
                num_loaded_time_points = final_top_displacements.shape[0]
                total_noise_added_top = 0
                for t_idx in range(num_loaded_time_points):
                    current_data_shape = final_top_displacements[t_idx].shape 
                    top_noise_for_step = np.random.normal(0, noise_stddev, size=current_data_shape)
                    final_top_displacements[t_idx] += top_noise_for_step
                    total_noise_added_top += np.sum(np.abs(top_noise_for_step))
                noisy_top_mean = np.mean(np.abs(final_top_displacements))
                print(f"Top disp mean magnitude: {original_top_mean:.2e} -> {noisy_top_mean:.2e}")
                print(f"Total absolute noise added - Top: {total_noise_added_top:.2e}")
            else:
                print("Skipping top displacement noise injection: no top data loaded.")

            if final_bot_displacements.size > 0:
                original_bot_mean = np.mean(np.abs(final_bot_displacements))
                num_loaded_time_points_bot = final_bot_displacements.shape[0]
                total_noise_added_bot = 0
                for t_idx in range(num_loaded_time_points_bot):
                    current_data_shape = final_bot_displacements[t_idx].shape 
                    bot_noise_for_step = np.random.normal(0, noise_stddev, size=current_data_shape)
                    final_bot_displacements[t_idx] += bot_noise_for_step
                    total_noise_added_bot += np.sum(np.abs(bot_noise_for_step))
                noisy_bot_mean = np.mean(np.abs(final_bot_displacements))
                print(f"Bot disp mean magnitude: {original_bot_mean:.2e} -> {noisy_bot_mean:.2e}")
                print(f"Total absolute noise added - Bot: {total_noise_added_bot:.2e}")
            else:
                print("Skipping bot displacement noise injection: no bot data loaded.")

            print(f"--- Noise addition complete (stddev: {noise_stddev:.2e}) ---")


    if enable_spatial_sparsity and (final_top_weights.size > 0 or final_bot_weights.size > 0):
        if sparsity_type == "random_fraction":
            print(f"--- Applying 'random_fraction' spatial sparsity: zeroing out {sparsity_fraction*100:.2f}% of WEIGHT data points ---")
            if final_top_weights.size > 0:
                top_weight_zero_mask = np.random.rand(*final_top_weights.shape) < sparsity_fraction
                final_top_weights[top_weight_zero_mask] = 0.0
                num_top_weights_zeroed = np.sum(top_weight_zero_mask)
                print(f"    Zeroed out {num_top_weights_zeroed} points in top_weights.")
            if final_bot_weights.size > 0:
                bot_weight_zero_mask = np.random.rand(*final_bot_weights.shape) < sparsity_fraction
                final_bot_weights[bot_weight_zero_mask] = 0.0
                num_bot_weights_zeroed = np.sum(bot_weight_zero_mask)
                print(f"    Zeroed out {num_bot_weights_zeroed} points in bot_weights")
        
        elif sparsity_type == "every_n_nodes":
            print(f"--- Applying 'every_n_nodes' spatial sparsity: keeping every {sparsity_n_nodes}-th node ---")
            num_nodes = final_top_weights.shape[2]  # (time, components, nodes)
            nodes_to_zero_mask = np.ones(num_nodes, dtype=bool)
            nodes_to_zero_mask[::sparsity_n_nodes] = False
            
            if final_top_weights.size > 0:
                final_top_weights[:, :, nodes_to_zero_mask] = 0.0
                num_top_weights_zeroed = np.sum(nodes_to_zero_mask) * final_top_weights.shape[0] * final_top_weights.shape[1]
                print(f"    Zeroed out weights for {np.sum(nodes_to_zero_mask)} nodes in top_weights (total points: {num_top_weights_zeroed}).")

            if final_bot_weights.size > 0:
                final_bot_weights[:, :, nodes_to_zero_mask] = 0.0
                num_bot_weights_zeroed = np.sum(nodes_to_zero_mask) * final_bot_weights.shape[0] * final_bot_weights.shape[1]
                print(f"    Zeroed out weights for {np.sum(nodes_to_zero_mask)} nodes in bot_weights (total points: {num_bot_weights_zeroed}).")

        print("--- Spatial sparsity application complete ---")

    print("--- Successfully loaded all requested displacement and weight data. ---")
    
    return final_time_axis, final_step_numbers_to_load, x_coordinates, final_top_displacements, final_bot_displacements, final_top_weights, final_bot_weights


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process ground truth displacement data. Optionally adds noise, zeros out data, saves modified data, and generates animations.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter 
    )
    
    parser.add_argument(
        "original_sim_name_base",
        help="Base name of the original simulation that generated the ground truth data (e.g., 'time_int_opt_gt'). "
             "Used to find .coords and .time files, and as a base for default data directories."
    )
    parser.add_argument(
        "step_frequency",
        type=int,
        nargs='?', 
        default=1,
        help="Frequency of steps to load (e.g., 1 for every step, 10 for every 10th step from the full list)."
    )

    noise_group = parser.add_argument_group('Noise Options')
    noise_group.add_argument(
        "--add-noise",
        action="store_true",
        help="Flag to add random Gaussian noise to the displacement data."
    )
    noise_group.add_argument(
        "--noise-stddev", 
        type=float,
        default=1e-6, 
        help="Standard deviation for Gaussian noise if --add-noise is used."
    )

    sparsity_group = parser.add_argument_group('Spatial Sparsity Options (for Weights)')
    sparsity_group.add_argument(
        "--enable-spatial-sparsity",
        action="store_true",
        help="Flag to enable spatial sparsity by setting some weights to zero."
    )
    sparsity_group.add_argument(
        "--sparsity-type",
        type=str,
        default="random_fraction",
        choices=["random_fraction", "every_n_nodes"],
        help="Type of spatial sparsity to apply."
    )
    sparsity_group.add_argument(
        "--sparsity-fraction",
        type=float,
        default=0.1,
        help="For 'random_fraction' type: the fraction of WEIGHT data points to randomly set to zero (0.0 to 1.0)."
    )
    sparsity_group.add_argument(
        "--sparsity-n-nodes",
        type=int,
        default=10,
        help="For 'every_n_nodes' type: keep data from every N-th node, zeroing out others."
    )

    output_group = parser.add_argument_group('Output Options for Modified Data')
    output_group.add_argument(
        "--output-noisy-data",
        action="store_true",
        help="Flag to save the (potentially noisy/modified) displacement and weight data to a new set of files "
             "in 'noisy_gt_data_input/' directory structure."
    )
    output_group.add_argument(
        "--noisy-sim-name", # This argument is present but its primary use for naming output is currently fixed.
        type=str,
        default=None,
        help="Base name for the output dataset if --output-noisy-data is used. Currently, output is fixed to 'noisy_gt_data_input'."
    )
    output_group.add_argument( 
        "--direct-input-dir-for-out-files",
        type=str,
        default=None,
        help="ADVANCED: Directly specify the path to the directory containing the .proc0.sX.out files. "
             "If used, original_sim_name_base.coords and .time are still expected relative to original_sim_name_base."
    )
    parser.add_argument(
        "--output-dir-name",
        type=str,
        default=None,
        help="Specify a custom output directory name for processed data."
    )

    parser.add_argument( 
        "--animate-displacement",
        action="store_true",
        help="Flag to generate and save an animation of the displacement profile over time."
    )

    # Add format control
    parser.add_argument(
        "--input-format",
        type=str,
        choices=['ascii', 'binary', 'auto'],
        default='auto',
        help="Format of input displacement files. 'auto' attempts to detect automatically."
    )
    
    parser.add_argument(
        "--output-format", 
        type=str,
        choices=['ascii', 'binary'],
        default='binary',
        help="Format for output displacement files when using --output-noisy-data."
    )

    args = parser.parse_args()

    # Pass format to loading function
    force_format = None if args.input_format == 'auto' else args.input_format
    loaded_data_tuple = load_all_ground_truth_displacements(
        args.original_sim_name_base,
        step_frequency=args.step_frequency,
        add_noise=args.add_noise,
        noise_stddev=args.noise_stddev,
        enable_spatial_sparsity=args.enable_spatial_sparsity,
        sparsity_type=args.sparsity_type,
        sparsity_fraction=args.sparsity_fraction,
        sparsity_n_nodes=args.sparsity_n_nodes,
        direct_input_dir_for_out_files=args.direct_input_dir_for_out_files,
        force_format=force_format
    )

    if loaded_data_tuple is None:
        print("Exiting due to data loading failure.")
        sys.exit(1)

    final_time_axis, final_step_numbers_to_load, x_coords, final_top_displacements, final_bot_displacements, final_top_weights, final_bot_weights = loaded_data_tuple

    # Create time_data as list of tuples for compatibility with save_processed_data
    time_data = list(zip(final_step_numbers_to_load, final_time_axis))

    # --- Output processed data if requested ---
    if args.output_noisy_data:
        output_directory = args.output_dir_name if args.output_dir_name else "noisy_gt_data_input"
        save_processed_data(output_directory, x_coords, time_data, final_top_displacements, final_bot_displacements, final_top_weights, final_bot_weights, args.original_sim_name_base, args.output_format)



    # --- Animation ---
    if args.animate_displacement:
        animate_displacement_evolution(x_coords, time_data, final_top_displacements, final_bot_displacements, args.original_sim_name_base)

