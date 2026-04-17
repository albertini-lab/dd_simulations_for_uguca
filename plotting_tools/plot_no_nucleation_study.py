#!/usr/bin/env python3
"""
Plot displacement at final time step for no-nucleation parameter study.
Shows displacement for each update interval with standard (non-dynamic) ground truth in red.
No nucleation study tests data-driven methods without dynamic nucleation.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.collections import LineCollection
from matplotlib.patches import Rectangle
import os

# Add MacTeX to PATH for LaTeX rendering
os.environ['PATH'] = '/usr/local/texlive/2023/bin/universal-darwin:' + os.environ.get('PATH', '')

# Add MacTeX to PATH for LaTeX rendering
os.environ['PATH'] = '/usr/local/texlive/2023/bin/universal-darwin:' + os.environ.get('PATH', '')

# Matplotlib parameters
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
plt.rcParams.update(params)

def find_simulation_files(sim_name):
    """Find simulation output files."""
    # Look for standard ground truth data (non-dynamic nucleation)
    if sim_name == "time_int_opt_gt":
        base_dir = "simulation_outputs/time_int_opt_gt_ground_truth"
        if os.path.isdir(base_dir):
            coords_path = os.path.join(base_dir, "time_int_opt_gt.coords")
            data_dir_path = os.path.join(base_dir, "time_int_opt_gt-DataFiles")
            if os.path.exists(coords_path) and os.path.isdir(data_dir_path):
                return coords_path, data_dir_path
    
    # Look for data-driven test results
    elif sim_name.startswith("dd_perturbed_no_nucleation_test_interval"):
        base_dir = f"simulation_outputs/{sim_name}_data_driven"
        if os.path.isdir(base_dir):
            coords_path = os.path.join(base_dir, f"{sim_name}.coords")
            data_dir_path = os.path.join(base_dir, f"{sim_name}-DataFiles")
            if os.path.exists(coords_path) and os.path.isdir(data_dir_path):
                return coords_path, data_dir_path
    
    return None, None

def load_coords(coords_file):
    """Load coordinates."""
    try:
        return np.loadtxt(coords_file, usecols=0)
    except (IOError, ValueError) as e:
        print(f"Error loading coordinates from {coords_file}: {e}")
        return None

def load_final_state(sim_name, data_dir, field_name, num_nodes, component_index=0):
    """Load final state from field output file."""
    file_path = os.path.join(data_dir, f"{field_name}.out")
    try:
        with open(file_path, 'r') as fl:
            lines = fl.readlines()
        if not lines: 
            return None
        last_line = lines[-1].strip()
        if not last_line: 
            last_line = lines[-2].strip()
        data_array = np.array([float(i) for i in last_line.split()])
        start_idx = component_index * num_nodes
        end_idx = (component_index + 1) * num_nodes
        return data_array[start_idx:end_idx]
    except (IOError, IndexError, ValueError) as e:
        print(f"Error loading final state for {sim_name} ({field_name}): {e}")
        return None

def plot_no_nucleation_study():
    """Plot displacement at final time step for all simulations."""
    
    # Define the intervals tested
    intervals = [
    1, 11, 21, 31, 41, 51, 61, 71, 81, 91, 101, 151, 201, 251, 301, 351, 401, 451, 501, 551,
    601, 651, 701, 751, 801, 851, 901, 951, 1001, 1051, 1101, 1151, 1201, 1251, 1301, 1351,
    1401, 1451, 1501, 1551, 1601, 1651, 1701
    ]
    intervals = sorted(set(intervals))  # Remove duplicates and sort
    print("=== NO-NUCLEATION STUDY: Final Displacement Comparison ===")
    
    # Load ground truth
    gt_coords_path, gt_data_dir = find_simulation_files("time_int_opt_gt")
    if gt_coords_path is None:
        print("Error: Could not find standard ground truth data")
        return
    
    coords = load_coords(gt_coords_path)
    if coords is None:
        print("Error: Could not load coordinates")
        return
    
    num_nodes = len(coords)
    gt_top_disp = load_final_state("time_int_opt_gt", gt_data_dir, "top_disp", num_nodes, 0)
    gt_bot_disp = load_final_state("time_int_opt_gt", gt_data_dir, "bot_disp", num_nodes, 0)
    
    if gt_top_disp is None or gt_bot_disp is None:
        print("Error: Could not load standard ground truth displacement")
        return
    
    print(f"✓ Loaded standard ground truth: {num_nodes} nodes")
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot data-driven results with colorbar
    top_displacements = []
    bot_displacements = []
    
    for interval in intervals:
        sim_name = f"dd_perturbed_no_nucleation_test_interval_{interval}"
        coords_path, data_dir = find_simulation_files(sim_name)
        top_disp = load_final_state(sim_name, data_dir, "top_disp", num_nodes, 0)
        bot_disp = load_final_state(sim_name, data_dir, "bot_disp", num_nodes, 0)
        top_displacements.append(top_disp)
        bot_displacements.append(bot_disp)
        print(f"✓ Loaded interval {interval}")
    
    # Plot each simulation with color based on update interval (behind GT)
    colors = plt.cm.viridis(np.linspace(0, 1, len(intervals)))
    
    # Find indices for positions 0.2 and 0.8 to track displacement evolution
    idx_020 = np.argmin(np.abs(coords - 0.2))
    idx_080 = np.argmin(np.abs(coords - 0.8))
    pos_020 = coords[idx_020]
    pos_080 = coords[idx_080]
    print(f"Tracking displacement evolution at positions: {pos_020:.3f} (index {idx_020}), {pos_080:.3f} (index {idx_080})")
    
    # Collect displacement values at tracking positions for evolution lines
    top_disps_020 = []
    bot_disps_020 = []
    top_disps_080 = []
    bot_disps_080 = []
    
    for i, (interval, top_disp, bot_disp) in enumerate(zip(intervals, top_displacements, bot_displacements)):
        ax.plot(coords, top_disp, color=colors[i], linewidth=1.0, 
                alpha=0.65, zorder=1)
        ax.plot(coords, bot_disp, color=colors[i], linewidth=1.0, 
                alpha=0.65, zorder=1)
        
        # Collect displacement values at tracking positions
        if top_disp is not None and bot_disp is not None:
            top_disps_020.append(top_disp[idx_020])
            bot_disps_020.append(bot_disp[idx_020])
            top_disps_080.append(top_disp[idx_080])
            bot_disps_080.append(bot_disp[idx_080])
        else:
            # Handle missing data
            top_disps_020.append(np.nan)
            bot_disps_020.append(np.nan)
            top_disps_080.append(np.nan)
            bot_disps_080.append(np.nan)
    
    # Draw evolution lines connecting displacement values at x=0.2 across all simulations
    # The line shows how displacement at this point changes with update interval
    if len(top_disps_020) > 1:
        # Filter out NaN values
        valid_top_020 = [d for d in top_disps_020 if not np.isnan(d)]
        valid_bot_020 = [d for d in bot_disps_020 if not np.isnan(d)]
        
        if len(valid_top_020) > 1 and len(valid_bot_020) > 1:
            # Create x-coordinates for evolution visualization
            # Instead of vertical lines, create a curved path to show evolution clearly
            n_points = len(valid_top_020)
            
            # Create curved x-coordinates to avoid self-overlap
            curve_width = 0.012  # Slightly increased width for less tight curve
            t = np.linspace(0, 1, n_points)
            # Create a parabolic curve: start at center, curve out, return to center
            x_curve = pos_020 + curve_width * (4 * t * (1 - t))  # Parabolic curve
            
            # Plot evolution curves for top and bottom displacement (on top of everything)
            ax.plot(x_curve, valid_top_020, color='aqua', linewidth=1.5, alpha=1.0, zorder=15, 
                    label=r'Evolution at $x = 0.2$')
            ax.plot(x_curve, valid_bot_020, color='aqua', linewidth=1.5, alpha=1.0, zorder=15)
            
            # Add markers: 'o' for start (highest interval), 'x' for end (lowest interval)
            # Start markers (highest update interval - first in list)
            ax.plot(x_curve[0], valid_top_020[0], 'o', color='aqua', markersize=5, zorder=16, 
                    markerfacecolor='aqua', markeredgecolor='darkblue', markeredgewidth=1)
            ax.plot(x_curve[0], valid_bot_020[0], 'o', color='aqua', markersize=5, zorder=16,
                    markerfacecolor='aqua', markeredgecolor='darkblue', markeredgewidth=1)
            
            # End markers (lowest update interval - last in list) 
            ax.plot(x_curve[-1], valid_top_020[-1], 'x', color='aqua', markersize=5, zorder=16, 
                    markeredgecolor='darkblue', markeredgewidth=1.5)
            ax.plot(x_curve[-1], valid_bot_020[-1], 'x', color='aqua', markersize=5, zorder=16,
                    markeredgecolor='darkblue', markeredgewidth=1.5)
            
            print(f"Added evolution curve at position {pos_020:.3f}")
            print(f"Top displacement evolution: {valid_top_020[0]:.2e} → {valid_top_020[-1]:.2e}")
            print(f"Bottom displacement evolution: {valid_bot_020[0]:.2e} → {valid_bot_020[-1]:.2e}")
    
    # Draw evolution lines connecting displacement values at x=0.8 across all simulations
    if len(top_disps_080) > 1:
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
            
            # Add markers: 'o' for start (highest interval), 'x' for end (lowest interval)
            # Start markers (highest update interval - first in list)
            ax.plot(x_curve[0], valid_top_080[0], 'o', color='aqua', markersize=5, zorder=16, 
                    markerfacecolor='aqua', markeredgecolor='darkblue', markeredgewidth=1)
            ax.plot(x_curve[0], valid_bot_080[0], 'o', color='aqua', markersize=5, zorder=16,
                    markerfacecolor='aqua', markeredgecolor='darkblue', markeredgewidth=1)
            
            # End markers (lowest update interval - last in list) 
            ax.plot(x_curve[-1], valid_top_080[-1], 'x', color='aqua', markersize=5, zorder=16, 
                    markeredgecolor='darkblue', markeredgewidth=1.5)
            ax.plot(x_curve[-1], valid_bot_080[-1], 'x', color='aqua', markersize=5, zorder=16,
                    markeredgecolor='darkblue', markeredgewidth=1.5)
            
            print(f"Added evolution curve at position {pos_080:.3f}")
            print(f"Top displacement evolution: {valid_top_080[0]:.2e} → {valid_top_080[-1]:.2e}")
            print(f"Bottom displacement evolution: {valid_bot_080[0]:.2e} → {valid_bot_080[-1]:.2e}")
    
    # Plot ground truth on top (front) with higher zorder
    ax.plot(coords, gt_top_disp, 'r-', linewidth=1, label=r'$u_{GT}$', alpha=1.0, zorder=10)
    ax.plot(coords, gt_bot_disp, 'r-', linewidth=1, alpha=1.0, zorder=10)
    
    ax.set_xlabel(r'Position, $x/L$')
    ax.set_ylabel(r'Displacement $u$ at Final Time-Step')
    ax.grid(True, alpha=0.3)
    
    # Create custom legend with gradient line for data-driven results
    from matplotlib.legend_handler import HandlerBase
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
                color = plt.cm.viridis(color_val)
                mini_rect = Rectangle((x_pos, ydescent), segment_width, height, 
                                    facecolor=color, edgecolor='none', transform=trans)
                artists.append(mini_rect)
            
            return artists
    
    # Create legend elements
    legend_elements = []
    
    # Add ground truth
    legend_elements.append(plt.Line2D([0], [0], color='red', linewidth=1, label=r'$u_{GT}$'))
    
    # Add data-driven with gradient representation
    if len(intervals) > 0:
        gradient_patch = mpatches.Patch(label=r'$u_{DD}$')
        legend_elements.append(gradient_patch)
    
    # Add evolution line legend entry with start and end indicators
    if len(top_disps_020) > 1:
        # Create a custom legend line with markers at start and end
        class EvolutionLineHandler(HandlerBase):
            def create_artists(self, legend, orig_handle, xdescent, ydescent, width, height, fontsize, trans):
                # Create the main line
                line = plt.Line2D([xdescent, xdescent + width], [ydescent + height/2, ydescent + height/2],
                                color='aqua', linewidth=1.5, transform=trans)
                
                # Add 'x' marker at start (left side - should show 'x' for end/lowest interval)
                start_marker = plt.Line2D([xdescent + width*0.1], [ydescent + height/2], 
                                        marker='x', color='aqua', markersize=4, 
                                        markeredgecolor='darkblue', markeredgewidth=1,
                                        linestyle='None', transform=trans)
                
                # Add 'o' marker at end (right side - should show 'o' for start/highest interval)
                end_marker = plt.Line2D([xdescent + width*0.9], [ydescent + height/2], 
                                      marker='o', color='aqua', markersize=4,
                                      markerfacecolor='aqua', markeredgecolor='darkblue',
                                      markeredgewidth=0.5, linestyle='None', transform=trans)
                
                return [line, start_marker, end_marker]
        
        # Create evolution legend patch
        evolution_patch = mpatches.Patch(label=r'Evolution at $x = 0.2, x = 0.8$')
        legend_elements.append(evolution_patch)
    
    # Create legend with custom handlers
    handler_map = {gradient_patch: GradientHandler()}
    if len(top_disps_020) > 1:
        handler_map[evolution_patch] = EvolutionLineHandler()
    
    ax.legend(handles=legend_elements, loc='lower left', handler_map=handler_map)
    
    # Create colorbar
    sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, 
                              norm=plt.Normalize(vmin=min(intervals), 
                                               vmax=max(intervals)))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, orientation='vertical', fraction=0.025, pad=0.05, shrink=0.8)
    cbar.set_label(r'Data-Driven Update Interval (Time Steps)')
    
    plt.tight_layout()
    
    # Save the plot
    os.makedirs('figures_and_plots', exist_ok=True)
    filename = 'figures_and_plots/no_nucleation_perturbed_displacement_study.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"\nSaved plot: {filename}")
    
    plt.show()

if __name__ == "__main__":
    plot_no_nucleation_study()