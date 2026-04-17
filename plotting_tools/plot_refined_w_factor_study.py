#!/usr/bin/env python3
"""
Plotting script for refined w_factor parameter study results.
Similar to the original plot_w_factor_study.py but focused on the stable range.
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.collections import LineCollection
from matplotlib.patches import Rectangle
import sys
import os
import glob
import re

# Add MacTeX to PATH for LaTeX rendering
os.environ['PATH'] = '/usr/local/texlive/2023/bin/universal-darwin:' + os.environ.get('PATH', '')

# --- Matplotlib parameters ---
params = {
    'text.usetex': True,
    'text.latex.preamble': r'\usepackage{lmodern}\usepackage[T1]{fontenc}\usepackage{sansmath}\sansmath',
    'font.family': 'sans-serif',
    'font.sans-serif': ['Latin Modern Sans'],
    'font.size': 11, 'figure.titlesize': 12, 'axes.labelsize': 11,
    'axes.titlesize': 12, 'legend.fontsize': 11, 'legend.frameon': False,
    'lines.linewidth': 1.2, 'axes.linewidth': 0.6,
    'xtick.labelsize': 11, 'ytick.labelsize': 11, 'xtick.major.size': 3,
    'ytick.major.size': 3,
}
mpl.rcParams.update(params)

def find_simulation_files(sim_name):
    """Find simulation output files with suffix handling"""
    base_path = os.path.join("simulation_outputs", sim_name)
    if not os.path.isdir(base_path):
        base_path_dd = os.path.join("simulation_outputs", f"{sim_name}_data_driven")
        if os.path.isdir(base_path_dd):
            base_path = base_path_dd
        else:
            base_path_gt = os.path.join("simulation_outputs", f"{sim_name}_ground_truth")
            if os.path.isdir(base_path_gt):
                base_path = base_path_gt
            else:
                print(f"Warning: Could not find simulation directory for {sim_name}")
                return None, None

    coords_file = os.path.join(base_path, f"{sim_name}.coords")
    data_dir = os.path.join(base_path, f"{sim_name}-DataFiles")

    if not os.path.exists(coords_file) or not os.path.isdir(data_dir):
        return None, None
        
    return coords_file, data_dir

def load_final_state(sim_name, data_dir, field_name, num_nodes, component_index=0):
    """Load final state from simulation output"""
    file_path = os.path.join(data_dir, f"{field_name}.out")
    try:
        with open(file_path, 'r') as fl:
            lines = fl.readlines()
        if not lines: return None
        last_line = lines[-1].strip()
        if not last_line: last_line = lines[-2].strip()
        data_array = np.array([float(i) for i in last_line.split()])
        start_idx = component_index * num_nodes
        end_idx = (component_index + 1) * num_nodes
        return data_array[start_idx:end_idx]
    except (IOError, IndexError, ValueError) as e:
        print(f"Error loading final state for {sim_name} ({field_name}): {e}")
        return None

def load_coords(coords_file):
    """Load coordinate data"""
    try:
        return np.loadtxt(coords_file, usecols=0)
    except (IOError, ValueError) as e:
        print(f"Error loading coordinates from {coords_file}: {e}")
        return None

def format_ax(ax, title, ylabel):
    """Format axis with consistent styling"""
    ax.set_xlabel(r'Position, $x$ [m]')
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, linestyle=':', linewidth=0.5, alpha=0.7)

def parse_w_factor_from_name(sim_name):
    """Extract w_factor value from simulation name"""
    # Try refined naming first
    match = re.search(r"wfactor_refined_(\d+)e(\d+)", sim_name)
    if match:
        mantissa = float(match.group(1))
        exponent = int(match.group(2))
        return mantissa * (10 ** exponent)
    
    match = re.search(r"wfactor_refined_(\d+)en(\d+)", sim_name)
    if match:
        mantissa = float(match.group(1))
        exponent = -int(match.group(2))
        return mantissa * (10 ** exponent)
    
    # Fallback to original naming
    match = re.search(r"wfactor_(\d+)e(\d+)", sim_name)
    if match:
        mantissa = float(match.group(1))
        exponent = int(match.group(2))
        return mantissa * (10 ** exponent)
    
    match = re.search(r"wfactor_(\d+)en(\d+)", sim_name)
    if match:
        mantissa = float(match.group(1))
        exponent = -int(match.group(2))
        return mantissa * (10 ** exponent)
    
    return None

if __name__ == "__main__":
    # --- Find Simulation Results ---
    sim_output_dir = "simulation_outputs"
    if not os.path.isdir(sim_output_dir):
        sys.exit(f"Error: simulation_outputs directory not found. Please run simulations first.")

    # Try refined results first, fall back to original
    sim_dirs = glob.glob(os.path.join(sim_output_dir, "time_int_opt_wfactor_refined_*"))
    study_type = "refined"
    
    if not sim_dirs:
        sim_dirs = glob.glob(os.path.join(sim_output_dir, "time_int_opt_wfactor_*"))
        study_type = "original"
    
    if not sim_dirs:
        sys.exit("Error: No w_factor parameter study simulation results found.")

    w_factors = []
    sim_names = []
    for s_dir in sim_dirs:
        sim_name = os.path.basename(s_dir).replace("_data_driven", "")
        w_factor = parse_w_factor_from_name(sim_name)
        if w_factor is not None:
            w_factors.append(w_factor)
            sim_names.append(sim_name)

    if not w_factors:
        sys.exit("Error: Could not parse w_factor values from simulation names.")

    # Sort by w_factor value
    sorted_pairs = sorted(zip(w_factors, sim_names))
    w_factors, sim_names = zip(*sorted_pairs)

    print(f"Found {len(w_factors)} w_factor simulations ({study_type} study):")
    for w_f, sim_n in zip(w_factors, sim_names):
        print(f"  w_factor = {w_f:8.2e} -> {sim_n}")

    # --- Figure Setup ---
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    # --- Load Ground Truth ---
    gt_sim_name = "time_int_opt_gt"
    gt_coords_file, gt_data_dir = find_simulation_files(gt_sim_name)
    if gt_coords_file and gt_data_dir:
        X_gt = load_coords(gt_coords_file)
        num_nodes_gt = len(X_gt)
        gt_top_u0 = load_final_state(gt_sim_name, gt_data_dir, "top_disp", num_nodes_gt, 0)
        gt_bot_u0 = load_final_state(gt_sim_name, gt_data_dir, "bot_disp", num_nodes_gt, 0)
        if X_gt is not None and gt_top_u0 is not None:
            ax.plot(X_gt, gt_top_u0, color='red', linestyle='-', linewidth=2.5, label=r'Ground Truth Final Displacement $u_{GT}$', zorder=10)
        if X_gt is not None and gt_bot_u0 is not None:
            ax.plot(X_gt, gt_bot_u0, color='red', linestyle='-', linewidth=2.5, zorder=10)

    # --- Load Baseline ---
    baseline_sim_name = "time_int_opt_dd_baseline_no_dd"
    baseline_coords_file, baseline_data_dir = find_simulation_files(baseline_sim_name)
    if baseline_coords_file and baseline_data_dir:
        X_baseline = load_coords(baseline_coords_file)
        if X_baseline is not None:
            num_nodes_baseline = len(X_baseline)
            baseline_top_u0 = load_final_state(baseline_sim_name, baseline_data_dir, "top_disp", num_nodes_baseline, 0)
            baseline_bot_u0 = load_final_state(baseline_sim_name, baseline_data_dir, "bot_disp", num_nodes_baseline, 0)
            if baseline_top_u0 is not None:
                ax.plot(X_baseline, baseline_top_u0, 'k-', linewidth=2, label=r'Baseline Final Displacement $u_{Base}$', alpha=0.8)
            if baseline_bot_u0 is not None:
                ax.plot(X_baseline, baseline_bot_u0, 'k-', linewidth=2, alpha=0.8)

    # --- Color Map for W_factors (log scale) ---
    cmap = plt.cm.viridis  # Use viridis for consistent styling
    norm = mpl.colors.LogNorm(vmin=min(w_factors), vmax=max(w_factors))

    # --- Load and Plot Parameter Study Data ---
    # Plot data-driven results first (behind GT and baseline)
    for i, (w_factor, sim_name) in enumerate(zip(w_factors, sim_names)):
        coords_file, data_dir = find_simulation_files(sim_name)
        if not coords_file or not data_dir:
            continue

        X = load_coords(coords_file)
        if X is None: continue
        num_nodes = len(X)

        color = cmap(norm(w_factor))

        top_u0 = load_final_state(sim_name, data_dir, "top_disp", num_nodes, 0)
        bot_u0 = load_final_state(sim_name, data_dir, "bot_disp", num_nodes, 0)
        
        if top_u0 is not None:
            # Highlight optimal w_factor (2e-2) with thicker line
            lw = 2.0 if abs(w_factor - 2e-2) < 1e-3 else 1.2
            alpha = 0.9 if abs(w_factor - 2e-2) < 1e-3 else 0.7
            ax.plot(X, top_u0, color=color, alpha=alpha, linewidth=lw, zorder=2)

        if bot_u0 is not None:
            # Highlight optimal w_factor (2e-2) with thicker line
            lw = 2.0 if abs(w_factor - 2e-2) < 1e-3 else 1.2
            alpha = 0.9 if abs(w_factor - 2e-2) < 1e-3 else 0.7
            ax.plot(X, bot_u0, color=color, alpha=alpha, linewidth=lw, zorder=2)

    # Plot ground truth and baseline on top (replot with higher zorder)
    if gt_coords_file and gt_data_dir and X_gt is not None and gt_top_u0 is not None:
        ax.plot(X_gt, gt_top_u0, color='red', linestyle='-', linewidth=2.5, label=r'$u_{GT}$', zorder=10)
        ax.plot(X_gt, gt_bot_u0, color='red', linestyle='-', linewidth=2.5, zorder=10)
    
    if baseline_coords_file and baseline_data_dir and X_baseline is not None and baseline_top_u0 is not None:
        ax.plot(X_baseline, baseline_top_u0, 'k-', linewidth=2, label=r'$u_{Base}$', alpha=0.8, zorder=9)
        ax.plot(X_baseline, baseline_bot_u0, 'k-', linewidth=2, alpha=0.8, zorder=9)

    # --- Formatting and Colorbar ---
    ax.set_xlabel(r'Position, $x/L$')
    ax.set_ylabel(r'Displacement $u$ at Final Time-Step')
    ax.grid(True, linestyle=':', linewidth=0.5, alpha=0.7)
    
    # Create custom legend with gradient line for data-driven results
    from matplotlib.legend_handler import HandlerBase
    from matplotlib.patches import Rectangle
    import matplotlib.patches as mpatches
    
    class GradientHandler(HandlerBase):
        def create_artists(self, legend, orig_handle, xdescent, ydescent, width, height, fontsize, trans):
            # Create gradient effect by overlaying multiple thin rectangles
            n_segments = 20
            segment_width = width / n_segments
            artists = []
            
            for i in range(n_segments):
                x_pos = xdescent + i * segment_width
                color_val = i / (n_segments - 1)
                color = cmap(color_val)
                mini_rect = Rectangle((x_pos, ydescent), segment_width, height, 
                                    facecolor=color, edgecolor='none', transform=trans)
                artists.append(mini_rect)
            
            return artists
    
    # Create legend elements
    legend_elements = []
    
    # Add ground truth and baseline if they exist
    if gt_coords_file and gt_data_dir and X_gt is not None and gt_top_u0 is not None:
        legend_elements.append(plt.Line2D([0], [0], color='red', linewidth=2.5, label=r'$u_{GT}$'))
    if baseline_coords_file and baseline_data_dir and X_baseline is not None and baseline_top_u0 is not None:
        legend_elements.append(plt.Line2D([0], [0], color='black', linewidth=2, label=r'$u_{Base}$'))
    
    # Add data-driven with gradient representation
    if len(w_factors) > 0:
        gradient_patch = mpatches.Patch(label=r'$u_{DD}$')
        legend_elements.append(gradient_patch)
    
    # Create legend with custom handler
    if len(w_factors) > 0:
        handler_map = {gradient_patch: GradientHandler()}
        ax.legend(handles=legend_elements, loc='lower left', handler_map=handler_map)
    else:
        ax.legend(handles=legend_elements, loc='lower left')

    # Add colorbar with log scale
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, orientation='vertical', fraction=0.025, pad=0.05, shrink=0.8)
    cbar.set_label(r'Weighting Constant $w$')

    plt.tight_layout()

    # --- Save Figure ---
    figures_dir = "figures_and_plots"
    if not os.path.exists(figures_dir):
        os.makedirs(figures_dir)
    fig_filename = os.path.join(figures_dir, f'parameter_study_w_factor_{study_type}.png')
    fig.savefig(fig_filename, dpi=300)
    print(f"\nSaved w_factor parameter study plot to: {fig_filename}")
    
    # Print summary
    print(f"\n=== PLOTTING SUMMARY ===")
    print(f"Study type: {study_type}")
    print(f"W_factor range: {min(w_factors):.1e} to {max(w_factors):.1e}")
    print(f"Number of simulations: {len(w_factors)}")
    if study_type == "refined":
        print("This refined study focuses on the stable w_factor range identified from divergence analysis.")
    print(f"Optimal w_factor (2e-2) highlighted with thicker lines if present.")
