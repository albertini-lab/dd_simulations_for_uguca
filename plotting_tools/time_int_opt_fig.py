#!/usr/bin/env python3
from __future__ import print_function
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import sys
import os
import glob

# --- Matplotlib parameters ---
params = {
    'text.usetex': True,
    'text.latex.preamble': r'\usepackage{lmodern}\usepackage[T1]{fontenc}\usepackage{sansmath}\sansmath',
    'font.family': 'sans-serif',
    'font.sans-serif': ['Latin Modern Sans'],
    'font.size': 11, 'figure.titlesize': 12, 'axes.labelsize': 11,
    'axes.titlesize': 11, 'legend.fontsize': 11, 'legend.frameon': False,
    'legend.columnspacing': 1, 'legend.numpoints': 1, 'legend.scatterpoints': 1,
    'legend.handlelength': 1.5, 'lines.linewidth': 0.75, 'axes.linewidth': 0.5,
    'xtick.labelsize': 11, 'ytick.labelsize': 11, 'xtick.major.size': 2.5,
    'ytick.major.size': 2.5, 'xtick.minor.size': 2, 'ytick.minor.size': 2,
    'xtick.major.width': 0.5, 'ytick.major.width': 0.5, 'xtick.minor.width': 0.5,
    'ytick.minor.width': 0.5,
}
mpl.rcParams.update(params)

def find_simulation_files(sim_name):
    """
    Search for simulation files in the organized directory structure.
    Returns a dictionary with the paths to the found files.
    """
    found_files = {
        'coords_file': None,
        'data_dir': None,
        'base_path': None
    }
    
    # Define search patterns for organized, legacy, and relative paths
    search_patterns = [
        f"simulation_outputs/{sim_name}_*",  # Organized structure e.g., simulation_outputs/sim_name_ground_truth
        f"{sim_name}",                       # Legacy structure where output is in a dir named after sim
        ".",                                 # Current directory
        ".."                                 # Parent directory
    ]

    for pattern in search_patterns:
        # Use glob to handle potential wildcards in directory names
        for base_dir in glob.glob(pattern):
            if not os.path.isdir(base_dir):
                continue

            # --- Look for coords file ---
            coords_path = os.path.join(base_dir, f"{sim_name}.coords")
            if found_files['coords_file'] is None and os.path.exists(coords_path):
                found_files['coords_file'] = coords_path
                # If we find the coords file, we can tentatively set the base path
                if found_files['base_path'] is None:
                    found_files['base_path'] = base_dir
            
            # --- Look for data directory ---
            data_dir_path = os.path.join(base_dir, f"{sim_name}-DataFiles")
            if found_files['data_dir'] is None and os.path.isdir(data_dir_path):
                found_files['data_dir'] = data_dir_path
                # If we find the data dir, we can tentatively set the base path
                if found_files['base_path'] is None:
                    found_files['base_path'] = base_dir
                    
            # If we have found both, we can exit early
            if found_files['coords_file'] and found_files['data_dir']:
                # The base_path should be the common directory.
                # This simple logic assumes they are in the same base_dir, which they should be.
                # A more complex check could verify common paths if needed.
                return found_files

    # If we finish the loop and haven't found both, we return what we have.
    # The calling code will then handle the missing paths.
    return found_files

# --- Function to load final state data ---
def load_final_state(sim_name, field_name, num_nodes, component_index=0, is_scalar_field=False):
    """Loads the last line from the specified output file for a given component."""
    # Find simulation files in organized structure
    files = find_simulation_files(sim_name)
    
    if files['data_dir'] is None:
        print(f"Error: Could not find data directory for simulation: {sim_name}")
        return None
        
    file_path = os.path.join(files['data_dir'], f"{field_name}.out")
    
    try:
        with open(file_path, 'r') as fl:
            lines = fl.readlines()
        if not lines:
            print(f"Error: File {file_path} is empty.")
            return None
        last_line_data = ""
        for line in reversed(lines): # Find the last non-empty line
            if line.strip():
                last_line_data = line.strip()
                break
        if not last_line_data:
            print(f"Error: No data found in the last line of {file_path}.")
            return None
        data_array = np.array([float(i) for i in last_line_data.split()])
        
        if is_scalar_field:
            if data_array.shape[0] != num_nodes:
                print(f"Warning: Scalar field '{field_name}' has {data_array.shape[0]} values, expected {num_nodes}.")
                if data_array.shape[0] < num_nodes: return None # Or handle differently, e.g. pad or error
                final_data_comp = data_array[:num_nodes] # Truncate if too long, might be an issue
            else:
                final_data_comp = data_array
        else: # Vector field
            if num_nodes == 0:
                print(f"Error: num_nodes is zero for field '{field_name}'.")
                return None
            if data_array.shape[0] % num_nodes != 0:
                print(f"Error: Field '{field_name}' in {file_path} has {data_array.shape[0]} values, not a multiple of num_nodes ({num_nodes}).")
                return None
            num_components_in_file = data_array.shape[0] // num_nodes
            if not (0 <= component_index < num_components_in_file):
                print(f"Error: Requested component {component_index} for '{field_name}' is out of range (file has {num_components_in_file} components).")
                return None
            # Data is typically stored as [u1_x, u2_x, ..., un_x, u1_y, u2_y, ..., un_y, ...]
            start_idx = component_index * num_nodes
            end_idx = (component_index + 1) * num_nodes
            final_data_comp = data_array[start_idx:end_idx]
        return final_data_comp
    except IOError:
        print(f"Error: Cannot open file: {file_path}")
        return None
    except ValueError as e:
        print(f"Error parsing data in {file_path}: {e}")
        return None

# --- Function to load coordinates ---
def load_coords(sim_name):
    """Loads coordinates from the .coords file."""
    # Find simulation files in organized structure
    files = find_simulation_files(sim_name)
    
    if files['coords_file'] is None:
        print(f"Error: Could not find coordinates file for simulation: {sim_name}")
        return None
        
    file_path = files['coords_file']
    X = []
    try:
        with open(file_path, 'r') as fl:
            lines = fl.readlines()
        for line in lines:
            if line.strip(): # Ensure line is not empty or just whitespace
                 # Assuming coords file has one x-coordinate per line, possibly followed by other values
                 X.append(float(line.strip().split()[0]))
        return np.array(X)
    except IOError:
        print(f"Error: Cannot open file: {file_path}")
        return None
    except (ValueError, IndexError) as e: # Catch errors if line is empty or parsing fails
         print(f"Error parsing data in {file_path}: {e}")
         return None

def format_ax(ax, title, ylabel, X_coords, data_list):
    """Helper function to format a subplot."""
    if X_coords is not None and len(X_coords) > 0:
        ax.set_xlim([0, np.max(X_coords)]) # Assuming X_coords starts at or near 0
    
    # Calculate y-limits based on all valid data presented in the subplot
    valid_data_for_ylim = []
    for data_series in data_list:
        if data_series is not None and data_series.size > 0:
            valid_data_for_ylim.append(data_series[np.isfinite(data_series)]) # Filter out NaNs and Infs
    
    if valid_data_for_ylim:
        all_valid_points = np.concatenate(valid_data_for_ylim)
        if all_valid_points.size > 0:
            y_min_data = np.min(all_valid_points)
            y_max_data = np.max(all_valid_points)
            padding = (y_max_data - y_min_data) * 0.10 # 10% padding
            if padding == 0: # Handle case where all data points are the same
                padding = 0.1 if y_max_data == 0 else np.abs(y_max_data) * 0.1
            ax.set_ylim([y_min_data - padding, y_max_data + padding])

    ax.set_xlabel(r'Position, $x$ [m]')
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=params['axes.titlesize'])
    ax.legend(loc='best')
    ax.grid(True, linestyle=':', linewidth=0.5, alpha=0.7)


# -----------------------------------------------------------------------------
if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit("Usage: ./time_int_opt_fig.py <simulation_1_name> <simulation_2_name>\n"
                 "  Example: ./time_int_opt_fig.py time_int_opt_gt time_int_opt_dd_noisy")

    sim1_name = sys.argv[1]
    sim2_name = sys.argv[2]

    # --- Figure Setup ---
    fig = plt.figure()
    fig.set_size_inches(10, 6) # Adjusted for 2x2 layout
    fig.suptitle(f'Final State Comparison: {sim1_name} vs. {sim2_name}', fontsize=params['figure.titlesize'])

    gs = gridspec.GridSpec(2, 2, figure=fig) # 2 rows, 2 columns
    gs.update(left=0.08, bottom=0.1, right=0.97, top=0.90, wspace=0.3, hspace=0.35) # Adjusted spacing for 2x2

    ax1 = plt.subplot(gs[0, 0]) # Top Disp Comp 0 (x)
    ax2 = plt.subplot(gs[0, 1]) # Top Disp Comp 1 (y)
    ax3 = plt.subplot(gs[1, 0]) # Bot Disp Comp 0 (x)
    ax4 = plt.subplot(gs[1, 1]) # Bot Disp Comp 1 (y)

    X_sim1 = load_coords(sim1_name)
    X_sim2 = load_coords(sim2_name)

    if X_sim1 is None or X_sim2 is None:
        sys.exit("Error loading coordinates. Exiting.")
    num_nodes_sim1 = len(X_sim1)
    num_nodes_sim2 = len(X_sim2)
    if num_nodes_sim1 == 0 or num_nodes_sim2 == 0:
        sys.exit("Error: Coordinate data is empty for one or both simulations.")
    
    X_ref = X_sim1 # Use sim1 coordinates as the reference by default
    if not np.allclose(X_sim1, X_sim2):
        print(f"Warning: Coordinate data differs between {sim1_name} and {sim2_name}. Using coordinates from {sim1_name} for plotting.")
    
    field_name_top_disp = "top_disp"
    field_name_bot_disp = "bot_disp"

    # --- Load and Plot Data ---

    # Top Displacement Comp 0 (x)
    top_u0_sim1 = load_final_state(sim1_name, field_name_top_disp, num_nodes_sim1, component_index=0)
    top_u0_sim2 = load_final_state(sim2_name, field_name_top_disp, num_nodes_sim2, component_index=0)
    if top_u0_sim1 is not None: ax1.plot(X_sim1, top_u0_sim1, color='green', linestyle='-', label=sim1_name)
    if top_u0_sim2 is not None: ax1.plot(X_sim2, top_u0_sim2, color='darkgreen', linestyle='--', label=sim2_name)
    format_ax(ax1, r'Top Displacement, $u_x$', r'Displacement [m]', X_ref, [top_u0_sim1, top_u0_sim2])

    # Top Displacement Comp 1 (y)
    top_u1_sim1 = load_final_state(sim1_name, field_name_top_disp, num_nodes_sim1, component_index=1)
    top_u1_sim2 = load_final_state(sim2_name, field_name_top_disp, num_nodes_sim2, component_index=1)
    if top_u1_sim1 is not None: ax2.plot(X_sim1, top_u1_sim1, color='cyan', linestyle='-', label=sim1_name)
    if top_u1_sim2 is not None: ax2.plot(X_sim2, top_u1_sim2, color='teal', linestyle='--', label=sim2_name)
    format_ax(ax2, r'Top Displacement, $u_y$', r'Displacement [m]', X_ref, [top_u1_sim1, top_u1_sim2])

    # Bottom Displacement Comp 0 (x)
    bot_u0_sim1 = load_final_state(sim1_name, field_name_bot_disp, num_nodes_sim1, component_index=0)
    bot_u0_sim2 = load_final_state(sim2_name, field_name_bot_disp, num_nodes_sim2, component_index=0)
    if bot_u0_sim1 is not None: ax3.plot(X_sim1, bot_u0_sim1, color='purple', linestyle='-', label=sim1_name)
    if bot_u0_sim2 is not None: ax3.plot(X_sim2, bot_u0_sim2, color='indigo', linestyle='--', label=sim2_name)
    format_ax(ax3, r'Bottom Displacement, $u_x$', r'Displacement [m]', X_ref, [bot_u0_sim1, bot_u0_sim2])

    # Bottom Displacement Comp 1 (y)
    bot_u1_sim1 = load_final_state(sim1_name, field_name_bot_disp, num_nodes_sim1, component_index=1)
    bot_u1_sim2 = load_final_state(sim2_name, field_name_bot_disp, num_nodes_sim2, component_index=1)
    if bot_u1_sim1 is not None: ax4.plot(X_sim1, bot_u1_sim1, color='orange', linestyle='-', label=sim1_name)
    if bot_u1_sim2 is not None: ax4.plot(X_sim2, bot_u1_sim2, color='saddlebrown', linestyle='--', label=sim2_name)
    format_ax(ax4, r'Bottom Displacement, $u_y$', r'Displacement [m]', X_ref, [bot_u1_sim1, bot_u1_sim2])
    
    # ---
    
    # Create figures directory if it doesn't exist and save figure
    figures_dir = "figures_and_plots"
    if not os.path.exists(figures_dir):
        os.makedirs(figures_dir)
    
    fig_filename = os.path.join(figures_dir, f'comparison_subplots_{sim1_name}_vs_{sim2_name}.png')
    # Adjust rect for suptitle and bottom labels to fit within the 2x2 layout
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) 
    fig.savefig(fig_filename, dpi=300)
    print(f"Saved comparison figure with subplots to: {fig_filename}")
    # plt.show() # Uncomment to display interactively

