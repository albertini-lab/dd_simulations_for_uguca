#!/usr/bin/env python3
"""
Plot parameter study results comparing different data-driven update intervals.
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
    # Simplified find function for this specific plotting script
    # First try the exact name
    base_path = os.path.join("simulation_outputs", sim_name)
    if not os.path.isdir(base_path):
        # Try with _data_driven suffix for parameter study results
        base_path_dd = os.path.join("simulation_outputs", f"{sim_name}_data_driven")
        if os.path.isdir(base_path_dd):
            base_path = base_path_dd
        else:
            # Try with _standard suffix for standard simulations
            base_path_std = os.path.join("simulation_outputs", f"{sim_name}_standard")
            if os.path.isdir(base_path_std):
                base_path = base_path_std
            else:
                # Fallback for ground truth if it has a different pattern
                base_path_gt = os.path.join("simulation_outputs", f"{sim_name}_ground_truth")
                if os.path.isdir(base_path_gt):
                    base_path = base_path_gt
                else:
                    print(f"Warning: Could not find simulation directory for {sim_name}")
                    return None, None

    coords_file = os.path.join(base_path, f"{sim_name}.coords")
    data_dir = os.path.join(base_path, f"{sim_name}-DataFiles")

    if not os.path.exists(coords_file) or not os.path.isdir(data_dir):
        # Try to find sim_name from info file if needed, but for now, assume it matches dir
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
    ax.set_xlabel(r'Position')
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, linestyle=':', linewidth=0.5, alpha=0.7)

if __name__ == "__main__":
    # --- Find Simulation Results ---
    sim_output_dir = "simulation_outputs"
    if not os.path.isdir(sim_output_dir):
        sys.exit(f"Error: simulation_outputs directory not found. Please run simulations first.")

    sim_dirs = glob.glob(os.path.join(sim_output_dir, "time_int_opt_dd_interval_*"))
    if not sim_dirs:
        sys.exit("Error: No parameter study simulation results found. Run run_parameter_study.py first.")

    intervals = []
    for s_dir in sim_dirs:
        match = re.search(r"interval_(\d+)", s_dir)
        if match:
            intervals.append(int(match.group(1)))
    intervals.sort()

    # --- Create figure with single axes ---
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    # Load ground truth for comparison if available
    X_gt, gt_top_u0, gt_bot_u0 = None, None, None
    gt_sim_name = "time_int_opt_gt"
    gt_coords_file, gt_data_dir = find_simulation_files(gt_sim_name)
    if gt_coords_file and gt_data_dir:
        X_gt = load_coords(gt_coords_file)
        if X_gt is not None:
            num_nodes_gt = len(X_gt)
            gt_top_u0 = load_final_state(gt_sim_name, gt_data_dir, "top_disp", num_nodes_gt, 0)
            gt_bot_u0 = load_final_state(gt_sim_name, gt_data_dir, "bot_disp", num_nodes_gt, 0)

    # --- Load Baseline (No Data-Driven) ---
    baseline_sim_name = "time_int_opt_dd_baseline_no_dd"
    baseline_coords_file, baseline_data_dir = find_simulation_files(baseline_sim_name)
    X_baseline, baseline_top_u0, baseline_bot_u0 = None, None, None
    if baseline_coords_file and baseline_data_dir:
        X_baseline = load_coords(baseline_coords_file)
        if X_baseline is not None:
            num_nodes_baseline = len(X_baseline)
            baseline_top_u0 = load_final_state(baseline_sim_name, baseline_data_dir, "top_disp", num_nodes_baseline, 0)
            baseline_bot_u0 = load_final_state(baseline_sim_name, baseline_data_dir, "bot_disp", num_nodes_baseline, 0)

    # --- Color Map for Intervals ---
    cmap = plt.cm.viridis
    norm = mpl.colors.Normalize(vmin=min(intervals), vmax=max(intervals))

    # Store results for potential RMSE analysis
    results = []
    
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

    # --- Load and Plot Parameter Study Data ---
    for interval in intervals:
        sim_name = f"time_int_opt_dd_interval_{interval}"
        coords_file, data_dir = find_simulation_files(sim_name)
        if not coords_file or not data_dir:
            continue

        X = load_coords(coords_file)
        if X is None: 
            continue
        num_nodes = len(X)

        color = cmap(norm(interval))

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
        
        # Store results
        results.append({
            'interval': interval,
            'X': X,
            'top_u0': top_u0,
            'bot_u0': bot_u0,
            'color': color
        })

    # Plot data-driven results first (behind GT and baseline)
    for i, result in enumerate(results):
        if result['top_u0'] is not None:
            ax.plot(result['X'], result['top_u0'], color=result['color'], alpha=0.65, linewidth=1.0, zorder=1)
        if result['bot_u0'] is not None:
            ax.plot(result['X'], result['bot_u0'], color=result['color'], alpha=0.65, linewidth=1.0, zorder=1)

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
            
            # Add markers: 'o' for start (lowest interval), 'x' for end (highest interval)
            # Start markers (lowest update interval - first in list)
            ax.plot(x_curve[0], valid_top_020[0], 'o', color='aqua', markersize=5, zorder=16, 
                    markerfacecolor='aqua', markeredgecolor='darkblue', markeredgewidth=1)
            ax.plot(x_curve[0], valid_bot_020[0], 'o', color='aqua', markersize=5, zorder=16,
                    markerfacecolor='aqua', markeredgecolor='darkblue', markeredgewidth=1)
            
            # End markers (highest update interval - last in list) 
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
            
            # Add markers: 'o' for start (lowest interval), 'x' for end (highest interval)
            # Start markers (lowest update interval - first in list)
            ax.plot(x_curve[0], valid_top_080[0], 'o', color='aqua', markersize=5, zorder=16, 
                    markerfacecolor='aqua', markeredgecolor='darkblue', markeredgewidth=1)
            ax.plot(x_curve[0], valid_bot_080[0], 'o', color='aqua', markersize=5, zorder=16,
                    markerfacecolor='aqua', markeredgecolor='darkblue', markeredgewidth=1)
            
            # End markers (highest update interval - last in list) 
            ax.plot(x_curve[-1], valid_top_080[-1], 'x', color='aqua', markersize=5, zorder=16, 
                    markeredgecolor='darkblue', markeredgewidth=1.5)
            ax.plot(x_curve[-1], valid_bot_080[-1], 'x', color='aqua', markersize=5, zorder=16,
                    markeredgecolor='darkblue', markeredgewidth=1.5)
            
            print(f"Added evolution curve at position {pos_080:.3f}")
            print(f"Top displacement evolution: {valid_top_080[0]:.2e} → {valid_top_080[-1]:.2e}")
            print(f"Bottom displacement evolution: {valid_bot_080[0]:.2e} → {valid_bot_080[-1]:.2e}")

    # Plot ground truth and baseline on top (front) with higher zorder
    if X_gt is not None and gt_top_u0 is not None:
        ax.plot(X_gt, gt_top_u0, 'r-', linewidth=1, label=r'$u_{GT}$', alpha=1.0, zorder=10)
        ax.plot(X_gt, gt_bot_u0, 'r-', linewidth=1, alpha=1.0, zorder=10)
    
    if X_baseline is not None and baseline_top_u0 is not None:
        ax.plot(X_baseline, baseline_top_u0, color='orange', linewidth=1, label=r'$u_{Base}$', alpha=1.0, zorder=9)
        ax.plot(X_baseline, baseline_bot_u0, color='orange', linewidth=1, alpha=1.0, zorder=9)

    ax.set_xlabel(r'Position, $x/L$')
    ax.set_ylabel(r'Displacement $u$ at Final Time-Step')
    ax.grid(True, alpha=0.3)
    
    # Create custom legend with gradient line for data-driven results
    from matplotlib.legend_handler import HandlerBase
    from matplotlib.patches import Rectangle
    import matplotlib.patches as mpatches
    
    class GradientHandler(HandlerBase):
        def create_artists(self, legend, orig_handle, xdescent, ydescent, width, height, fontsize, trans):
            # Create a rectangle that will be filled with gradient
            rect = Rectangle((xdescent, ydescent), width, height, transform=trans)
            
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
    if X_gt is not None and gt_top_u0 is not None:
        legend_elements.append(plt.Line2D([0], [0], color='red', linewidth=1, label=r'$u_{GT}$'))
    if X_baseline is not None and baseline_top_u0 is not None:
        legend_elements.append(plt.Line2D([0], [0], color='orange', linewidth=1, label=r'$u_{Base}$'))
    
    # Add data-driven with gradient representation
    if len(results) > 0:
        # Create a custom patch for gradient legend
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
                
                # Add 'x' marker at start (left side - should show 'x' for end/highest interval)
                start_marker = plt.Line2D([xdescent + width*0.1], [ydescent + height/2], 
                                        marker='x', color='aqua', markersize=4, 
                                        markeredgecolor='darkblue', markeredgewidth=1,
                                        linestyle='None', transform=trans)
                
                # Add 'o' marker at end (right side - should show 'o' for start/lowest interval)
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
    
    if len(results) > 0:
        ax.legend(handles=legend_elements, loc='lower left', handler_map=handler_map)
    else:
        ax.legend(handles=legend_elements, loc='lower left')

    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, orientation='vertical', fraction=0.025, pad=0.05, shrink=0.8)
    cbar.set_label(r'Data-Driven Update Interval (Time Steps)')

    plt.tight_layout()

    # --- Save Figure ---
    figures_dir = "figures_and_plots"
    if not os.path.exists(figures_dir):
        os.makedirs(figures_dir)
    fig_filename = os.path.join(figures_dir, 'parameter_study_update_interval.png')
    fig.savefig(fig_filename, dpi=300, bbox_inches='tight')
    print(f"Saved parameter study displacement plots to: {fig_filename}")

    # Create separate RMSE figure if ground truth is available
    if X_gt is not None and gt_top_u0 is not None:
        fig_rmse, ax_rmse = plt.subplots(1, 1, figsize=(10, 6))
        
        intervals_list = []
        top_rmse_list = []
        bot_rmse_list = []
        
        for result in results:
            if result['top_u0'] is not None and result['bot_u0'] is not None:
                # Calculate RMSE vs ground truth (interpolate if needed)
                if len(result['X']) == len(X_gt):
                    top_rmse = np.sqrt(np.mean((result['top_u0'] - gt_top_u0)**2))
                    bot_rmse = np.sqrt(np.mean((result['bot_u0'] - gt_bot_u0)**2))
                else:
                    # Interpolate ground truth to match simulation grid
                    gt_top_interp = np.interp(result['X'], X_gt, gt_top_u0)
                    gt_bot_interp = np.interp(result['X'], X_gt, gt_bot_u0)
                    top_rmse = np.sqrt(np.mean((result['top_u0'] - gt_top_interp)**2))
                    bot_rmse = np.sqrt(np.mean((result['bot_u0'] - gt_bot_interp)**2))
                
                intervals_list.append(result['interval'])
                top_rmse_list.append(top_rmse)
                bot_rmse_list.append(bot_rmse)
        
        if intervals_list:
            ax_rmse.semilogy(intervals_list, top_rmse_list, 'bo-', label='Top Surface RMSE', markersize=6)
            ax_rmse.semilogy(intervals_list, bot_rmse_list, 'ro-', label='Bottom Surface RMSE', markersize=6)
            
            # Find optimal interval
            min_top_idx = np.argmin(top_rmse_list)
            optimal_interval = intervals_list[min_top_idx]
            ax_rmse.axvline(optimal_interval, color='green', linestyle='--', alpha=0.7,
                          label=f'Optimal Interval: {optimal_interval}')
            
            ax_rmse.set_xlabel('Data-Driven Update Interval')
            ax_rmse.set_ylabel('RMSE vs Ground Truth [m]')
            ax_rmse.set_title('RMSE vs Data-Driven Update Interval')
            ax_rmse.grid(True, alpha=0.3)
            ax_rmse.legend()
            
            plt.tight_layout()
            
            rmse_filename = os.path.join(figures_dir, 'parameter_study_rmse_analysis.png')
            fig_rmse.savefig(rmse_filename, dpi=300, bbox_inches='tight')
            print(f"Saved parameter study RMSE analysis to: {rmse_filename}")
            
            print(f"\nOptimal data-driven update interval: {optimal_interval} (min top RMSE: {min(top_rmse_list):.2e})")
        else:
            print("No valid RMSE data found for analysis")

