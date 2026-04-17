#!/usr/bin/env python3
"""
Plot w-factor study results comparing different weighting constants.
"""
from __future__ import print_function
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
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
    # First try the exact name
    base_path = os.path.join("simulation_outputs", sim_name)
    if not os.path.isdir(base_path):
        # Try with _data_driven suffix for parameter study results
        base_path_dd = os.path.join("simulation_outputs", f"{sim_name}_data_driven")
        if os.path.isdir(base_path_dd):
            base_path = base_path_dd
        else:
            # Try with _standard suffix for baseline simulations
            base_path_std = os.path.join("simulation_outputs", f"{sim_name}_standard")
            if os.path.isdir(base_path_std):
                base_path = base_path_std
            else:
                # Fallback for ground truth
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
    try:
        return np.loadtxt(coords_file, usecols=0)
    except (IOError, ValueError) as e:
        print(f"Error loading coordinates from {coords_file}: {e}")
        return None

def format_ax(ax, title, ylabel):
    ax.set_xlabel(r'Position, $x$ [m]')
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, linestyle=':', linewidth=0.5, alpha=0.7)

def parse_w_factor_from_name(sim_name):
    """Extract w_factor value from simulation name (handles both old and new high-precision formats)"""
    
    # HIGH-PRECISION FORMAT: wfactor_1p000000e02 -> 1.000000e+02
    match = re.search(r"wfactor_(\d+)p(\d+)e(\d+)", sim_name)
    if match:
        integer_part = match.group(1)
        decimal_part = match.group(2)
        exponent = int(match.group(3))
        mantissa_str = f"{integer_part}.{decimal_part}"
        mantissa = float(mantissa_str)
        return mantissa * (10 ** exponent)
    
    # HIGH-PRECISION FORMAT: wfactor_1p000000en06 -> 1.000000e-06
    match = re.search(r"wfactor_(\d+)p(\d+)en(\d+)", sim_name)
    if match:
        integer_part = match.group(1)
        decimal_part = match.group(2)
        exponent = int(match.group(3))
        mantissa_str = f"{integer_part}.{decimal_part}"
        mantissa = float(mantissa_str)
        return mantissa * (10 ** (-exponent))
    
    # OLD FORMAT (for backward compatibility): wfactor_1e01 -> 1e1 = 10
    match = re.search(r"wfactor_(\d+)e(\d+)", sim_name)
    if match:
        mantissa = float(match.group(1))
        exponent = int(match.group(2))
        return mantissa * (10 ** exponent)
    
    # OLD FORMAT (for backward compatibility): wfactor_1en06 -> 1e-6
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

    sim_dirs = glob.glob(os.path.join(sim_output_dir, "time_int_opt_wfactor_*"))
    if not sim_dirs:
        sys.exit("Error: No w_factor parameter study simulation results found. Run run_w_factor_study.py first.")

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

    # Sort w_factors and corresponding sim_names
    sorted_pairs = sorted(zip(w_factors, sim_names))
    w_factors, sim_names = zip(*sorted_pairs)
    
    # Find simulations around 1e-2 (0.01) - take 15 on each side
    target_value = 1e-2  # 0.01
    target_index = None
    
    # Find the index closest to target value
    min_diff = float('inf')
    for i, w_val in enumerate(w_factors):
        diff = abs(w_val - target_value)
        if diff < min_diff:
            min_diff = diff
            target_index = i
    
    if target_index is not None:
        # Take ALL simulations from the start up to 12 simulations above target
        start_idx = 0  # Start from the very beginning (lowest w_factor)
        end_idx = min(len(w_factors), target_index + 13)  # +13 to include target + 12 after
        
        w_factors = w_factors[start_idx:end_idx]
        sim_names = sim_names[start_idx:end_idx]
        print(f"Focused on {len(w_factors)} simulations from lowest w_factor to 1e-2+12 (target at position {target_index + 1})")
    
    print(f"Found {len(w_factors)} w_factor simulations:")

    print(f"Using {len(w_factors)} w_factor simulations:")
    for w_f, sim_n in zip(w_factors, sim_names):
        print(f"  w_factor = {w_f:8.2e} -> {sim_n}")

    # --- Figure Setup ---
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    # --- Load Ground Truth ---
    gt_sim_name = "time_int_opt_gt_ground_truth"
    gt_coords_file, gt_data_dir = find_simulation_files(gt_sim_name)
    
    # If not found with _ground_truth suffix, try without
    if gt_coords_file is None:
        gt_sim_name = "time_int_opt_gt"
        gt_coords_file, gt_data_dir = find_simulation_files(gt_sim_name)
    
    X_gt = None
    gt_top_u0 = None
    gt_bot_u0 = None
    
    if gt_coords_file and gt_data_dir:
        X_gt = load_coords(gt_coords_file)
        if X_gt is not None:
            num_nodes_gt = len(X_gt)
            gt_top_u0 = load_final_state(gt_sim_name, gt_data_dir, "top_disp", num_nodes_gt, 0)
            gt_bot_u0 = load_final_state(gt_sim_name, gt_data_dir, "bot_disp", num_nodes_gt, 0)

    # --- Load Baseline (No Data-Driven) ---
    baseline_sim_name = "time_int_opt_dd_baseline_no_dd"
    baseline_coords_file, baseline_data_dir = find_simulation_files(baseline_sim_name)
    
    # If not found, try with different naming variations
    if baseline_coords_file is None:
        baseline_sim_name = "time_int_opt_dd_baseline_no_dd_standard"
        baseline_coords_file, baseline_data_dir = find_simulation_files(baseline_sim_name)
    
    if baseline_coords_file is None:
        baseline_sim_name = "time_int_opt_baseline_deviated"
        baseline_coords_file, baseline_data_dir = find_simulation_files(baseline_sim_name)
    
    X_baseline = None
    baseline_top_u0 = None
    baseline_bot_u0 = None
    
    if baseline_coords_file and baseline_data_dir:
        X_baseline = load_coords(baseline_coords_file)
        if X_baseline is not None:
            num_nodes_baseline = len(X_baseline)
            baseline_top_u0 = load_final_state(baseline_sim_name, baseline_data_dir, "top_disp", num_nodes_baseline, 0)
            baseline_bot_u0 = load_final_state(baseline_sim_name, baseline_data_dir, "bot_disp", num_nodes_baseline, 0)

    # --- Color Map for W_factors (log scale) ---
    cmap = plt.cm.viridis
    norm = mpl.colors.LogNorm(vmin=min(w_factors), vmax=max(w_factors))

    # --- Process Results ---
    all_results = []
    
    # Find indices for positions 0.2 and 0.8 to track displacement evolution (will be set after loading first simulation)
    idx_020 = None
    idx_080 = None
    pos_020 = None
    pos_080 = None
    
    # Collect displacement values at tracking positions for evolution lines
    top_disps_020 = []
    bot_disps_020 = []
    top_disps_080 = []
    bot_disps_080 = []
    
    for w_factor, sim_name in zip(w_factors, sim_names):
        coords_file, data_dir = find_simulation_files(sim_name)
        if not coords_file or not data_dir:
            continue

        X = load_coords(coords_file)
        if X is None:
            continue
        num_nodes = len(X)

        # Load displacement data
        top_u0 = load_final_state(sim_name, data_dir, "top_disp", num_nodes, 0)
        bot_u0 = load_final_state(sim_name, data_dir, "bot_disp", num_nodes, 0)
        
        # Set up position tracking indices on first successful load
        if idx_020 is None and X is not None:
            idx_020 = np.argmin(np.abs(X - 0.2))
            idx_080 = np.argmin(np.abs(X - 0.8))
            pos_020 = X[idx_020]
            pos_080 = X[idx_080]
            print(f"Tracking displacement evolution at positions: {pos_020:.3f} (index {idx_020}), {pos_080:.3f} (index {idx_080})")
        
        # Collect displacement values at tracking positions
        if top_u0 is not None and bot_u0 is not None and idx_020 is not None:
            top_disps_020.append(top_u0[idx_020])
            bot_disps_020.append(bot_u0[idx_020])
            top_disps_080.append(top_u0[idx_080])
            bot_disps_080.append(bot_u0[idx_080])
        else:
            # Handle missing data
            top_disps_020.append(np.nan)
            bot_disps_020.append(np.nan)
            top_disps_080.append(np.nan)
            bot_disps_080.append(np.nan)
        
        result = {
            'w_factor': w_factor,
            'sim_name': sim_name,
            'X': X,
            'top_u0': top_u0,
            'bot_u0': bot_u0
        }
        
        all_results.append(result)

    # --- Plot Results ---
    print(f"Plotting {len(all_results)} simulations around 1e-2")
    
    for i, result in enumerate(all_results):
        color = cmap(norm(result['w_factor']))
        
        if result['top_u0'] is not None:
            ax.plot(result['X'], result['top_u0'], color=color, alpha=0.65, 
                    linewidth=1.0, zorder=1)
        
        if result['bot_u0'] is not None:
            ax.plot(result['X'], result['bot_u0'], color=color, alpha=0.65, 
                    linewidth=1.0, zorder=1)

    # Draw evolution lines connecting displacement values at x=0.2 across all simulations
    if len(top_disps_020) > 1 and pos_020 is not None:
        # Filter out NaN values
        valid_top_020 = [d for d in top_disps_020 if not np.isnan(d)]
        valid_bot_020 = [d for d in bot_disps_020 if not np.isnan(d)]
        
        if len(valid_top_020) > 1 and len(valid_bot_020) > 1:
            # Create x-coordinates for evolution visualization
            n_points = len(valid_top_020)
            
            # Create curved x-coordinates to avoid self-overlap
            curve_width = 0.012
            t = np.linspace(0, 1, n_points)
            # Create a parabolic curve: start at center, curve out, return to center
            x_curve = pos_020 + curve_width * (4 * t * (1 - t))  # Parabolic curve
            
            # Plot evolution curves for top and bottom displacement (on top of everything)
            ax.plot(x_curve, valid_top_020, color='aqua', linewidth=1.5, alpha=1.0, zorder=15)
            ax.plot(x_curve, valid_bot_020, color='aqua', linewidth=1.5, alpha=1.0, zorder=15)
            
            # Add markers: 'o' for start (lowest w_factor), 'x' for end (highest w_factor)
            # Start markers (lowest w_factor - first in list)
            ax.plot(x_curve[0], valid_top_020[0], 'o', color='aqua', markersize=5, zorder=16, 
                    markerfacecolor='aqua', markeredgecolor='darkblue', markeredgewidth=1)
            ax.plot(x_curve[0], valid_bot_020[0], 'o', color='aqua', markersize=5, zorder=16,
                    markerfacecolor='aqua', markeredgecolor='darkblue', markeredgewidth=1)
            
            # End markers (highest w_factor - last in list) 
            ax.plot(x_curve[-1], valid_top_020[-1], 'x', color='aqua', markersize=5, zorder=16, 
                    markeredgecolor='darkblue', markeredgewidth=1.5)
            ax.plot(x_curve[-1], valid_bot_020[-1], 'x', color='aqua', markersize=5, zorder=16,
                    markeredgecolor='darkblue', markeredgewidth=1.5)
            
            print(f"Added evolution curve at position {pos_020:.3f}")
            print(f"Top displacement evolution: {valid_top_020[0]:.2e} → {valid_top_020[-1]:.2e}")
            print(f"Bottom displacement evolution: {valid_bot_020[0]:.2e} → {valid_bot_020[-1]:.2e}")
    
    # Draw evolution lines connecting displacement values at x=0.8 across all simulations
    if len(top_disps_080) > 1 and pos_080 is not None:
        # Filter out NaN values
        valid_top_080 = [d for d in top_disps_080 if not np.isnan(d)]
        valid_bot_080 = [d for d in bot_disps_080 if not np.isnan(d)]
        
        if len(valid_top_080) > 1 and len(valid_bot_080) > 1:
            # Create x-coordinates for evolution visualization
            n_points = len(valid_top_080)
            
            # Create curved x-coordinates to avoid self-overlap
            curve_width = 0.012  # Same width as x=0.2 curve
            t = np.linspace(0, 1, n_points)
            # Create a parabolic curve: start at center, curve out, return to center
            x_curve = pos_080 + curve_width * (4 * t * (1 - t))  # Parabolic curve
            
            # Plot evolution curves for top and bottom displacement (on top of everything)
            ax.plot(x_curve, valid_top_080, color='aqua', linewidth=1.5, alpha=1.0, zorder=15)
            ax.plot(x_curve, valid_bot_080, color='aqua', linewidth=1.5, alpha=1.0, zorder=15)
            
            # Add markers: 'o' for start (lowest w_factor), 'x' for end (highest w_factor)
            # Start markers (lowest w_factor - first in list)
            ax.plot(x_curve[0], valid_top_080[0], 'o', color='aqua', markersize=5, zorder=16, 
                    markerfacecolor='aqua', markeredgecolor='darkblue', markeredgewidth=1)
            ax.plot(x_curve[0], valid_bot_080[0], 'o', color='aqua', markersize=5, zorder=16,
                    markerfacecolor='aqua', markeredgecolor='darkblue', markeredgewidth=1)
            
            # End markers (highest w_factor - last in list) 
            ax.plot(x_curve[-1], valid_top_080[-1], 'x', color='aqua', markersize=5, zorder=16, 
                    markeredgecolor='darkblue', markeredgewidth=1.5)
            ax.plot(x_curve[-1], valid_bot_080[-1], 'x', color='aqua', markersize=5, zorder=16,
                    markeredgecolor='darkblue', markeredgewidth=1.5)
            
            print(f"Added evolution curve at position {pos_080:.3f}")
            print(f"Top displacement evolution: {valid_top_080[0]:.2e} → {valid_top_080[-1]:.2e}")
            print(f"Bottom displacement evolution: {valid_bot_080[0]:.2e} → {valid_bot_080[-1]:.2e}")

    # Plot ground truth and baseline on top (front) with higher zorder
    if gt_top_u0 is not None and X_gt is not None:
        ax.plot(X_gt, gt_top_u0, 'r-', linewidth=1, label=r'$u_{GT}$', alpha=1, zorder=10)
        ax.plot(X_gt, gt_bot_u0, 'r-', linewidth=1, alpha=1, zorder=10)
    
    # Plot baseline if available
    if baseline_top_u0 is not None and X_baseline is not None:
        ax.plot(X_baseline, baseline_top_u0, 'orange', linewidth=1, label=r'$u_{Base}$', alpha=1, zorder=9)
        ax.plot(X_baseline, baseline_bot_u0, 'orange', linewidth=1, alpha=1, zorder=9)
    
    # Update colorbar normalization to include all w_factor values for full visualization
    if all_results:
        all_w_factors = [r['w_factor'] for r in all_results]
        norm = mpl.colors.LogNorm(vmin=min(all_w_factors), vmax=max(all_w_factors))

    # --- Formatting ---
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
    if gt_top_u0 is not None and X_gt is not None:
        legend_elements.append(plt.Line2D([0], [0], color='red', linewidth=2.5, label=r'$u_{GT}$'))
    if baseline_top_u0 is not None and X_baseline is not None:
        legend_elements.append(plt.Line2D([0], [0], color='orange', linewidth=2, label=r'$u_{Base}$'))
    
    # Add data-driven with gradient representation
    if len(all_results) > 0:
        gradient_patch = mpatches.Patch(label=r'$u_{DD}$')
        legend_elements.append(gradient_patch)
    
    # Add evolution line legend entry with start and end indicators
    if len(top_disps_020) > 1 and pos_020 is not None:
        # Create a custom legend line with markers at start and end
        class EvolutionLineHandler(HandlerBase):
            def create_artists(self, legend, orig_handle, xdescent, ydescent, width, height, fontsize, trans):
                # Create the main line
                line = plt.Line2D([xdescent, xdescent + width], [ydescent + height/2, ydescent + height/2],
                                color='aqua', linewidth=1.5, transform=trans)
                
                # Add 'x' marker at start (left side - should show 'x' for end/highest w_factor)
                start_marker = plt.Line2D([xdescent + width*0.1], [ydescent + height/2], 
                                        marker='x', color='aqua', markersize=4, 
                                        markeredgecolor='darkblue', markeredgewidth=1,
                                        linestyle='None', transform=trans)
                
                # Add 'o' marker at end (right side - should show 'o' for start/lowest w_factor)
                end_marker = plt.Line2D([xdescent + width*0.9], [ydescent + height/2], 
                                      marker='o', color='aqua', markersize=4,
                                      markerfacecolor='aqua', markeredgecolor='darkblue',
                                      markeredgewidth=0.5, linestyle='None', transform=trans)
                
                return [line, start_marker, end_marker]
        
        # Create evolution legend patch
        evolution_patch = mpatches.Patch(label=r'Evolution at $x = 0.2, x = 0.8$')
        legend_elements.append(evolution_patch)
    
    # Create legend with custom handler
    handler_map = {gradient_patch: GradientHandler()}
    if len(top_disps_020) > 1 and pos_020 is not None:
        handler_map[evolution_patch] = EvolutionLineHandler()
    
    if len(all_results) > 0:
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
    fig_filename = os.path.join(figures_dir, 'parameter_study_w_factor_constant.png')
    fig.savefig(fig_filename, dpi=300, bbox_inches='tight')
    print(f"Saved w_factor parameter study plot to: {fig_filename}")
