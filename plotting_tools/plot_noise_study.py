#!/usr/bin/env python3
"""
Noise study parameter analysis and plotting script.
Analyzes how displacement accuracy varies with noise_stddev_sim and plots convergence metrics.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.collections import LineCollection
from matplotlib.patches import Rectangle
import os
import sys
import glob
import re

# Matplotlib parameters for clean plots
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

# Add MacTeX to PATH for LaTeX rendering
os.environ['PATH'] = '/usr/local/texlive/2023/bin/universal-darwin:' + os.environ.get('PATH', '')

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
                base_path_std = os.path.join("simulation_outputs", f"{sim_name}_standard")
                if os.path.isdir(base_path_std):
                    base_path = base_path_std
                else:
                    return None, None

    coords_file = os.path.join(base_path, f"{sim_name}.coords")
    data_dir = os.path.join(base_path, f"{sim_name}-DataFiles")

    if not os.path.exists(coords_file) or not os.path.isdir(data_dir):
        return None, None
        
    return coords_file, data_dir

def load_coords(coords_file):
    """Load coordinate data"""
    try:
        return np.loadtxt(coords_file, usecols=0)
    except (IOError, ValueError) as e:
        return None

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
        return None

def parse_noise_from_name(sim_name):
    """Extract noise_stddev value from simulation name (handles both old and new formats)"""
    
    # NEW HIGH-PRECISION FORMAT: time_int_opt_noise_1p000000en08 -> 1.000000e-08
    match = re.search(r"time_int_opt_noise_(\d+)p(\d+)en(\d+)", sim_name)
    if match:
        integer_part = match.group(1)
        decimal_part = match.group(2)
        exponent = int(match.group(3))
        mantissa_str = f"{integer_part}.{decimal_part}"
        mantissa = float(mantissa_str)
        return mantissa * (10 ** (-exponent))
    
    # NEW HIGH-PRECISION FORMAT: time_int_opt_noise_1p000000e02 -> 1.000000e+02
    match = re.search(r"time_int_opt_noise_(\d+)p(\d+)e(\d+)", sim_name)
    if match:
        integer_part = match.group(1)
        decimal_part = match.group(2)
        exponent = int(match.group(3))
        mantissa_str = f"{integer_part}.{decimal_part}"
        mantissa = float(mantissa_str)
        return mantissa * (10 ** exponent)
    
    # OLD FORMAT (for backward compatibility): time_int_opt_noise_1e01 -> 1e1 = 10
    match = re.search(r"time_int_opt_noise_(\d+)e(\d+)", sim_name)
    if match:
        mantissa = float(match.group(1))
        exponent = int(match.group(2))
        return mantissa * (10 ** exponent)
    
    # OLD FORMAT (for backward compatibility): time_int_opt_noise_1en08 -> 1e-8
    match = re.search(r"time_int_opt_noise_(\d+)en(\d+)", sim_name)
    if match:
        mantissa = float(match.group(1))
        exponent = -int(match.group(2))
        return mantissa * (10 ** exponent)
    
    # OLD DECIMAL FORMAT (for backward compatibility): time_int_opt_noise_1p5en06 -> 1.5e-06
    match = re.search(r"time_int_opt_noise_(\d+)p(\d+)en(\d+)", sim_name)
    if match:
        integer_part = match.group(1)
        decimal_part = match.group(2)
        exponent = int(match.group(3))
        mantissa_str = f"{integer_part}.{decimal_part}"
        mantissa = float(mantissa_str)
        return mantissa * (10 ** (-exponent))
    
    return None

def check_divergence_metrics(displacement_data, noise_stddev, gt_data=None, baseline_data=None):
    """Check if simulation diverged based on distance from ground truth relative to baseline"""
    if displacement_data is None:
        return True
    
    # Check for NaN/Inf
    if np.any(np.isnan(displacement_data)) or np.any(np.isinf(displacement_data)):
        return True
    
    # If we have ground truth and baseline, use relative distance criterion
    if gt_data is not None and baseline_data is not None:
        # Calculate RMSE relative to ground truth
        sim_rmse = np.sqrt(np.mean((displacement_data - gt_data)**2))
        baseline_rmse = np.sqrt(np.mean((baseline_data - gt_data)**2))
        
        # Consider diverged if simulation is significantly worse than baseline
        # Tightened tolerance: 1.5x worse than baseline (was 2.0x)
        tolerance_factor = 1.5
        if sim_rmse > tolerance_factor * baseline_rmse:
            return True
    
    # Fallback criteria for cases without ground truth/baseline
    max_disp = np.max(np.abs(displacement_data))
    
    # Tightened: smaller displacements are considered unstable
    if max_disp > 50:  # Was 100
        return True
    
    # Check for unusual displacement patterns (too small or too large variations)
    disp_std = np.std(displacement_data)
    disp_mean = np.mean(np.abs(displacement_data))
    
    # Flag simulations with extremely small variations (might indicate numerical issues)
    if disp_std < 1e-12 and disp_mean > 1e-6:
        return True
    
    # Flag simulations with excessive variations
    if disp_std > 10 * disp_mean and disp_mean > 1e-8:
        return True
    
    # Check for NaN/Inf again after calculations
    if np.any(np.isnan(displacement_data)) or np.any(np.isinf(displacement_data)):
        return True
    
    return False

def calculate_rmse(data1, data2):
    """Calculate RMSE between two datasets"""
    if data1 is None or data2 is None:
        return np.inf
    if len(data1) != len(data2):
        return np.inf
    return np.sqrt(np.mean((data1 - data2)**2))

def calculate_max_error(data1, data2):
    """Calculate maximum absolute error between two datasets"""
    if data1 is None or data2 is None:
        return np.inf
    if len(data1) != len(data2):
        return np.inf
    return np.max(np.abs(data1 - data2))

def load_ground_truth():
    """Load ground truth data for comparison"""
    gt_sim_name = "time_int_opt_gt"
    gt_coords_file, gt_data_dir = find_simulation_files(gt_sim_name)
    
    if gt_coords_file and gt_data_dir:
        X_coords = load_coords(gt_coords_file)
        if X_coords is not None:
            num_nodes = len(X_coords)
            gt_top_u0 = load_final_state(gt_sim_name, gt_data_dir, "top_disp", num_nodes, 0)
            gt_bot_u0 = load_final_state(gt_sim_name, gt_data_dir, "bot_disp", num_nodes, 0)
            return X_coords, gt_top_u0, gt_bot_u0
    
    return None, None, None

def analyze_noise_convergence():
    """Analyze convergence of noise study results"""
    print("Loading noise simulation results for convergence analysis...")
    
    # Find all noise simulations
    sim_output_dir = "simulation_outputs"
    if not os.path.isdir(sim_output_dir):
        print(f"Error: {sim_output_dir} directory not found")
        return
    
    # Find noise simulations
    noise_sim_dirs = glob.glob(os.path.join(sim_output_dir, "time_int_opt_noise_*_data_driven"))
    
    print(f"Found {len(noise_sim_dirs)} noise simulations")
    
    if not noise_sim_dirs:
        print("No noise simulations found")
        return
    
    # Load ground truth for comparison
    X_gt, gt_top_u0, gt_bot_u0 = load_ground_truth()
    if gt_top_u0 is None:
        print("Warning: No ground truth data found for comparison")
    
    # Load baseline for comparison
    baseline_sim_name = "time_int_opt_dd_baseline_no_dd"
    baseline_coords_file, baseline_data_dir = find_simulation_files(baseline_sim_name)
    baseline_top_u0, baseline_bot_u0 = None, None
    if baseline_coords_file and baseline_data_dir:
        baseline_coords = load_coords(baseline_coords_file)
        if baseline_coords is not None:
            num_nodes_baseline = len(baseline_coords)
            baseline_top_u0 = load_final_state(baseline_sim_name, baseline_data_dir, "top_disp", num_nodes_baseline, 0)
            baseline_bot_u0 = load_final_state(baseline_sim_name, baseline_data_dir, "bot_disp", num_nodes_baseline, 0)
    
    if baseline_top_u0 is None:
        print("Warning: No baseline data found for comparison")
    
    # Process noise results
    results = []
    for s_dir in noise_sim_dirs:
        dir_name = os.path.basename(s_dir)
        sim_name = dir_name.replace("_data_driven", "")
        noise_stddev = parse_noise_from_name(sim_name)
        
        if noise_stddev is None:
            print(f"Warning: Could not parse noise value from {sim_name}")
            continue
        
        coords_file, data_dir = find_simulation_files(sim_name)
        if coords_file and data_dir:
            X_coords = load_coords(coords_file)
            if X_coords is not None:
                num_nodes = len(X_coords)
                top_u0 = load_final_state(sim_name, data_dir, "top_disp", num_nodes, 0)
                bot_u0 = load_final_state(sim_name, data_dir, "bot_disp", num_nodes, 0)
                
                # Interpolate ground truth and baseline to match simulation grid if needed
                gt_top_interp = gt_top_u0
                gt_bot_interp = gt_bot_u0
                baseline_top_interp = baseline_top_u0
                baseline_bot_interp = baseline_bot_u0
                
                if gt_top_u0 is not None and len(X_coords) != len(X_gt):
                    gt_top_interp = np.interp(X_coords, X_gt, gt_top_u0)
                    gt_bot_interp = np.interp(X_coords, X_gt, gt_bot_u0)
                
                if baseline_top_u0 is not None and len(X_coords) != len(baseline_coords):
                    baseline_top_interp = np.interp(X_coords, baseline_coords, baseline_top_u0)
                    baseline_bot_interp = np.interp(X_coords, baseline_coords, baseline_bot_u0)
                
                # Check for divergence using ground truth and baseline comparison
                top_diverged = check_divergence_metrics(top_u0, noise_stddev, gt_top_interp, baseline_top_interp)
                bot_diverged = check_divergence_metrics(bot_u0, noise_stddev, gt_bot_interp, baseline_bot_interp)
                diverged = top_diverged or bot_diverged
                
                # Calculate errors relative to ground truth
                top_rmse = calculate_rmse(top_u0, gt_top_interp) if gt_top_interp is not None else None
                bot_rmse = calculate_rmse(bot_u0, gt_bot_interp) if gt_bot_interp is not None else None
                top_max_error = calculate_max_error(top_u0, gt_top_interp) if gt_top_interp is not None else None
                bot_max_error = calculate_max_error(bot_u0, gt_bot_interp) if gt_bot_interp is not None else None
                
                results.append({
                    'noise_stddev': noise_stddev,
                    'noise_variance': noise_stddev ** 2,
                    'sim_name': sim_name,
                    'diverged': diverged,
                    'X_coords': X_coords,
                    'top_u0': top_u0,
                    'bot_u0': bot_u0,
                    'top_rmse': top_rmse,
                    'bot_rmse': bot_rmse,
                    'top_max_error': top_max_error,
                    'bot_max_error': bot_max_error
                })
    
    # Sort by noise_stddev
    results.sort(key=lambda x: x['noise_stddev'])
    
    print(f"Loaded {len(results)} noise simulations")
    
    converged = [r for r in results if not r['diverged']]
    
    print(f"Converged: {len(converged)}, Diverged: {len(results) - len(converged)}")
    
    # Print some divergence statistics
    diverged_results = [r for r in results if r['diverged']]
    if diverged_results:
        diverged_noise_levels = [r['noise_stddev'] for r in diverged_results]
        print(f"Diverged noise levels range: {min(diverged_noise_levels):.2e} to {max(diverged_noise_levels):.2e}")
    
    if converged:
        converged_noise_levels = [r['noise_stddev'] for r in converged]
        print(f"Converged noise levels range: {min(converged_noise_levels):.2e} to {max(converged_noise_levels):.2e}")
    
    # Create plots
    create_noise_convergence_plots(results, converged, X_gt, gt_top_u0, gt_bot_u0)

def create_noise_convergence_plots(results, converged, X_gt, gt_top_u0, gt_bot_u0):
    """Create noise convergence analysis plots - displacement plots and error metrics"""
    
    # Extract noise_stddevs for all results and converged results
    all_noise_stddevs = [r['noise_stddev'] for r in results]
    converged_noise_stddevs = [r['noise_stddev'] for r in converged]
    
    # Create figure: single displacement plot
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    # Filter out the 3 lowest noise level simulations for plotting and colormap
    if len(converged) > 3:
        converged_to_plot = sorted(converged, key=lambda x: x['noise_stddev'])[3:]  # Skip first 3 (lowest noise)
        plotted_noise_stddevs = [r['noise_stddev'] for r in converged_to_plot]
    else:
        converged_to_plot = converged
        plotted_noise_stddevs = converged_noise_stddevs
    
    # Color map for noise levels - use only plotted simulation range for better contrast
    cmap = plt.cm.viridis
    if plotted_noise_stddevs:
        norm = mcolors.LogNorm(vmin=min(plotted_noise_stddevs), vmax=max(plotted_noise_stddevs))
    else:
        # Fallback if no simulations to plot
        norm = mcolors.LogNorm(vmin=min(all_noise_stddevs), vmax=max(all_noise_stddevs))
    
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
    
    # Plot the filtered converged data-driven results first (behind GT and baseline)
    print(f"Plotting {len(converged_to_plot)} out of {len(converged)} converged simulations (removed 3 lowest noise levels)" if len(converged) > 3 else f"Plotting all {len(converged_to_plot)} converged simulations")
    
    for i, result in enumerate(converged_to_plot):
        color = cmap(norm(result['noise_stddev']))
        
        # Set up position tracking indices on first successful load
        if idx_020 is None and result['X_coords'] is not None:
            idx_020 = np.argmin(np.abs(result['X_coords'] - 0.2))
            idx_080 = np.argmin(np.abs(result['X_coords'] - 0.8))
            pos_020 = result['X_coords'][idx_020]
            pos_080 = result['X_coords'][idx_080]
            print(f"Tracking displacement evolution at positions: {pos_020:.3f} (index {idx_020}), {pos_080:.3f} (index {idx_080})")
        
        # Collect displacement values at tracking positions
        if result['top_u0'] is not None and result['bot_u0'] is not None and idx_020 is not None:
            top_disps_020.append(result['top_u0'][idx_020])
            bot_disps_020.append(result['bot_u0'][idx_020])
            top_disps_080.append(result['top_u0'][idx_080])
            bot_disps_080.append(result['bot_u0'][idx_080])
        else:
            # Handle missing data
            top_disps_020.append(np.nan)
            bot_disps_020.append(np.nan)
            top_disps_080.append(np.nan)
            bot_disps_080.append(np.nan)
        
        ax.plot(result['X_coords'], result['top_u0'], color=color, alpha=0.65, 
                linewidth=1.0, zorder=1)
        ax.plot(result['X_coords'], result['bot_u0'], color=color, alpha=0.65, 
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
            
            # Add markers: 'o' for start (lowest noise), 'x' for end (highest noise)
            # Start markers (lowest noise - first in list since we removed the 3 lowest)
            ax.plot(x_curve[0], valid_top_020[0], 'o', color='aqua', markersize=5, zorder=16, 
                    markerfacecolor='aqua', markeredgecolor='darkblue', markeredgewidth=1)
            ax.plot(x_curve[0], valid_bot_020[0], 'o', color='aqua', markersize=5, zorder=16,
                    markerfacecolor='aqua', markeredgecolor='darkblue', markeredgewidth=1)
            
            # End markers (highest noise - last in list) 
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
            
            # Add markers: 'o' for start (lowest noise), 'x' for end (highest noise)
            # Start markers (lowest noise - first in list since we removed the 3 lowest)
            ax.plot(x_curve[0], valid_top_080[0], 'o', color='aqua', markersize=5, zorder=16, 
                    markerfacecolor='aqua', markeredgecolor='darkblue', markeredgewidth=1)
            ax.plot(x_curve[0], valid_bot_080[0], 'o', color='aqua', markersize=5, zorder=16,
                    markerfacecolor='aqua', markeredgecolor='darkblue', markeredgewidth=1)
            
            # End markers (highest noise - last in list) 
            ax.plot(x_curve[-1], valid_top_080[-1], 'x', color='aqua', markersize=5, zorder=16, 
                    markeredgecolor='darkblue', markeredgewidth=1.5)
            ax.plot(x_curve[-1], valid_bot_080[-1], 'x', color='aqua', markersize=5, zorder=16,
                    markeredgecolor='darkblue', markeredgewidth=1.5)
            
            print(f"Added evolution curve at position {pos_080:.3f}")
            print(f"Top displacement evolution: {valid_top_080[0]:.2e} → {valid_top_080[-1]:.2e}")
            print(f"Bottom displacement evolution: {valid_bot_080[0]:.2e} → {valid_bot_080[-1]:.2e}")

    # Plot ground truth and baseline on top (front) with higher zorder
    if gt_top_u0 is not None and X_gt is not None:
        ax.plot(X_gt, gt_top_u0, 'r-', linewidth=1, label=r'$u_{GT}$', alpha=1.0, zorder=10)
        ax.plot(X_gt, gt_bot_u0, 'r-', linewidth=1, alpha=1.0, zorder=10)
    
    # Load and plot baseline (no data-driven) if available
    baseline_sim_name = "time_int_opt_dd_baseline_no_dd"
    baseline_coords_file, baseline_data_dir = find_simulation_files(baseline_sim_name)
    if baseline_coords_file and baseline_data_dir:
        baseline_coords = load_coords(baseline_coords_file)
        if baseline_coords is not None:
            num_nodes = len(baseline_coords)
            baseline_top_u0 = load_final_state(baseline_sim_name, baseline_data_dir, "top_disp", num_nodes, 0)
            baseline_bot_u0 = load_final_state(baseline_sim_name, baseline_data_dir, "bot_disp", num_nodes, 0)
            if baseline_top_u0 is not None and baseline_bot_u0 is not None:
                ax.plot(baseline_coords, baseline_top_u0, color='orange', linewidth=1, label=r'$u_{Base}$', alpha=1.0, zorder=9)
                ax.plot(baseline_coords, baseline_bot_u0, color='orange', linewidth=1, alpha=1.0, zorder=9)
    
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
        legend_elements.append(plt.Line2D([0], [0], color='red', linewidth=1, label=r'$u_{GT}$'))
    if baseline_coords_file and baseline_data_dir:
        legend_elements.append(plt.Line2D([0], [0], color='orange', linewidth=1, label=r'$u_{Base}$'))
    
    # Add data-driven with gradient representation
    if len(converged) > 0:
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
                
                # Add 'x' marker at start (left side - should show 'x' for end/highest noise)
                start_marker = plt.Line2D([xdescent + width*0.1], [ydescent + height/2], 
                                        marker='x', color='aqua', markersize=4, 
                                        markeredgecolor='darkblue', markeredgewidth=1,
                                        linestyle='None', transform=trans)
                
                # Add 'o' marker at end (right side - should show 'o' for start/lowest noise)
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
    
    if len(converged) > 0:
        ax.legend(handles=legend_elements, loc='lower left', handler_map=handler_map)
    else:
        ax.legend(handles=legend_elements, loc='lower left')
    
    # Add colorbar for displacement plots
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, orientation='vertical', fraction=0.025, pad=0.05, shrink=0.8)
    cbar.set_label(r'Covariance $\sigma^2$')
    
    plt.tight_layout()
    
    # Save displacement plots
    figures_dir = "figures_and_plots"
    os.makedirs(figures_dir, exist_ok=True)
    displacement_filename = os.path.join(figures_dir, "noise_study_displacements.png")
    fig.savefig(displacement_filename, dpi=300, bbox_inches='tight')
    print(f"Displacement plots saved: {displacement_filename}")
    
    plt.close(fig)
    
    # Print summary statistics
    print(f"\n=== NOISE CONVERGENCE ANALYSIS SUMMARY ===")
    print(f"Total simulations analyzed: {len(results)}")
    print(f"Converged simulations: {len(converged)}")
    print(f"Diverged simulations: {len(results) - len(converged)}")
    if converged:
        print(f"Noise range (converged): {min(converged_noise_stddevs):.2e} to {max(converged_noise_stddevs):.2e}")
    else:
        print("No converged simulations found - criteria may be too strict")
    
    if gt_top_u0 is not None:
        print("Ground truth comparison available for error analysis.")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--help":
        print("Noise Study Convergence Analysis Script")
        print("Usage:")
        print("  python plot_noise_study.py")
        print("")
        print("This script analyzes noise study results and creates convergence plots.")
    else:
        analyze_noise_convergence()
