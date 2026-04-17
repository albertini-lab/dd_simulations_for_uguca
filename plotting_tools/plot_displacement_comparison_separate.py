#!/usr/bin/env python3
"""
Displacement Comparison Plotting Script - Separate Figures

Creates two separate figures:
1. Ground truth vs baseline deviated displacement over time (with noisy data points)
2. Ground truth vs baseline deviated vs data-driven displacement over time

Focuses on top interface displacement at x = 0.3.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add MacTeX to PATH for LaTeX rendering
os.environ['PATH'] = '/usr/local/texlive/2023/bin/universal-darwin:' + os.environ.get('PATH', '')

# Add LaTeX formatting support with Aptos (Body) font
plt.rcParams.update({
    'text.usetex': True,
    'text.latex.preamble': r'\usepackage{lmodern}\usepackage[T1]{fontenc}\usepackage{sansmath}\sansmath',
    'font.family': 'sans-serif',
    'font.sans-serif': ['Aptos', 'Arial', 'DejaVu Sans'],  # Fallback to Arial/DejaVu Sans if Aptos not available
    'font.size': 28,
    'axes.labelsize': 28,
    'legend.fontsize': 28,
    'xtick.labelsize': 28,
    'ytick.labelsize': 28
})

# Add the analysis tools directory to path to import the existing data loading functions
sys.path.append('/Users/joshmcneely/uguca/build/simulations/analysis_and_processing_tools')

try:
    from process_ground_truth_displacement import load_all_ground_truth_displacements
except ImportError as e:
    print(f"Error importing data loading functions: {e}")
    print("Make sure the process_ground_truth_displacement.py script is available.")
    sys.exit(1)

def load_displacement_data(sim_name, sim_type_dir):
    """Load displacement data using the existing processing script."""
    print(f"Loading displacement data for {sim_name} from {sim_type_dir}...")
    
    # Use the existing function but specify direct input directory
    data_dir = f"simulation_outputs/{sim_type_dir}/{sim_name}_displacement_dump-restart"
    
    try:
        # Load all displacement data - function returns 7 values
        time_axis, step_numbers, x_coords, top_disp_data, bot_disp_data, top_weights, bot_weights = load_all_ground_truth_displacements(
            sim_name, 
            step_frequency=1,
            direct_input_dir_for_out_files=data_dir,
            force_format='binary'
        )
        
        return x_coords, step_numbers, time_axis, top_disp_data, bot_disp_data
        
    except Exception as e:
        print(f"Error loading data for {sim_name}: {e}")
        return None, None, None, None, None

def load_noisy_data():
    """Load the noisy ground truth data."""
    print("Loading noisy ground truth data...")
    
    try:
        # The noisy data was created with the same structure as ground truth - function returns 7 values
        time_axis, step_numbers, x_coords, top_disp_data, bot_disp_data, top_weights, bot_weights = load_all_ground_truth_displacements(
            "noisy_gt_data_input", 
            step_frequency=1,
            direct_input_dir_for_out_files="simulation_outputs/noisy_gt_data_input/noisy_gt_data_input-restart",
            force_format='binary'
        )
        
        return x_coords, step_numbers, time_axis, top_disp_data, bot_disp_data
        
    except Exception as e:
        print(f"Error loading noisy data: {e}")
        return None, None, None, None, None

def main():
    print("=== Displacement Comparison Analysis - Separate Figures ===")
    
    # Load ground truth data
    gt_x, gt_steps, gt_times, gt_top_disp, gt_bot_disp = load_displacement_data(
        "time_int_opt_gt", "time_int_opt_gt_ground_truth"
    )
    
    # Load baseline deviated data  
    baseline_x, baseline_steps, baseline_times, baseline_top_disp, baseline_bot_disp = load_displacement_data(
        "time_int_opt_baseline_deviated", "time_int_opt_baseline_deviated_standard"
    )
    
    # Load data-driven simulation data
    dd_x, dd_steps, dd_times, dd_top_disp, dd_bot_disp = load_displacement_data(
        "time_int_opt_deviated_dd_noisy_scenario", "time_int_opt_deviated_dd_noisy_scenario_data_driven"
    )
    
    # Load noisy data
    noisy_x, noisy_steps, noisy_times, noisy_top_disp, noisy_bot_disp = load_noisy_data()
    
    # Check if all data loaded successfully
    if any(data is None for data in [gt_x, baseline_x, dd_x, noisy_x]):
        print("Error: Failed to load some data. Exiting.")
        return
    
    print(f"Data loaded successfully:")
    print(f"  Ground truth: {len(gt_steps)} time steps")
    print(f"  Baseline: {len(baseline_steps)} time steps")
    print(f"  Data-driven: {len(dd_steps)} time steps")
    print(f"  Noisy data: {len(noisy_steps)} time steps")
    
    # Find node closest to x = 0.3
    target_position = 0.3
    node_idx = np.argmin(np.abs(gt_x - target_position))
    actual_position = gt_x[node_idx]
    
    print(f"Selected node at position: {actual_position:.3f} (target: {target_position})")
    
    # Data-driven update interval
    dd_update_interval = 40
    
    # Component to plot (0=x, 1=y) - let's plot x-displacement
    component = 0
    
    # Colors as requested
    gt_color = 'red'
    baseline_color = 'orange'
    noisy_color = 'purple'
    dd_color = 'blue'
    
    # Create output directory
    os.makedirs('figures_and_plots', exist_ok=True)
    
    # Figure 1: Ground truth vs baseline deviated (with noisy data points)
    fig1, ax1 = plt.subplots(1, 1, figsize=(12, 9))
    
    # Ground truth
    ax1.plot(gt_times, gt_top_disp[:, component, node_idx], 
            color=gt_color, linestyle='-', linewidth=2, 
            label=r'$u_{GT}$', alpha=0.8)
    
    # Baseline deviated
    ax1.plot(baseline_times, baseline_top_disp[:, component, node_idx], 
            color=baseline_color, linestyle='-', linewidth=2, 
            label=r'$u_{Base}$', alpha=0.8)
    
    # Noisy data points at update intervals
    noisy_indices = np.arange(0, len(noisy_steps), dd_update_interval)
    ax1.scatter(noisy_times[noisy_indices], noisy_top_disp[noisy_indices, component, node_idx], 
              color=noisy_color, marker='x', s=50, alpha=0.9,
              label=r'$\hat{u}$')
    
    ax1.set_xlabel(r'Time, $t$ [s]')
    ax1.set_ylabel(r'Displacement, $u$')
    # ax1.set_title(r'$u_{GT}$ vs $u_{Base}$ at $x = 0.3$')
    #ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Save Figure 1
    plt.tight_layout()
    output_file1 = 'figures_and_plots/displacement_comparison_baseline_x02.png'
    fig1.savefig(output_file1, dpi=300, bbox_inches='tight')
    print(f"Figure 1 saved to: {output_file1}")
    
    # Figure 2: All methods including data-driven
    fig2, ax2 = plt.subplots(1, 1, figsize=(12, 9))
    
    # Ground truth
    ax2.plot(gt_times, gt_top_disp[:, component, node_idx], 
            color=gt_color, linestyle='-', linewidth=2, 
            label=r'$u_{GT}$', alpha=0.8)
    
    # Baseline deviated
    ax2.plot(baseline_times, baseline_top_disp[:, component, node_idx], 
            color=baseline_color, linestyle='-', linewidth=2, 
            label=r'$u_{Base}$', alpha=0.8)
    
    # Data-driven
    ax2.plot(dd_times, dd_top_disp[:, component, node_idx], 
            color=dd_color, linestyle='-', linewidth=2, 
            label=r'$u_{DD}$', alpha=0.8)
    
    # Noisy data points at update intervals (lighter)
    noisy_indices = np.arange(0, len(noisy_steps), dd_update_interval)
    ax2.scatter(noisy_times[noisy_indices], noisy_top_disp[noisy_indices, component, node_idx], 
               color=noisy_color, marker='x', s=50, alpha=0.9,
               label=r'$$\hat{u}$$')
    
    # Add thin dashed red lines connecting noisy data points to data-driven curve
    noisy_times_at_updates = noisy_times[noisy_indices]
    noisy_values_at_updates = noisy_top_disp[noisy_indices, component, node_idx]
    
    # Interpolate data-driven curve at noisy data times
    dd_interpolated = np.interp(noisy_times_at_updates, dd_times, dd_top_disp[:, component, node_idx])
    
    # Draw thin dashed red lines from noisy points to data-driven curve
    for i in range(len(noisy_times_at_updates)):
        ax2.plot([noisy_times_at_updates[i], noisy_times_at_updates[i]], 
                [noisy_values_at_updates[i], dd_interpolated[i]], 
                color='red', linestyle='--', linewidth=0.8, alpha=0.7)
    
    ax2.set_xlabel(r'Time, $t$ [s]')
    ax2.set_ylabel(r'Displacement, $u$')
    # ax2.set_title(r'$u_{GT}$ vs $u_{Base}$ vs $u_{DD}$ at $x = 0.3$')
    #ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Save Figure 2
    plt.tight_layout()
    output_file2 = 'figures_and_plots/displacement_comparison_all_methods_x02.png'
    fig2.savefig(output_file2, dpi=300, bbox_inches='tight')
    print(f"Figure 2 saved to: {output_file2}")
    
    # Show statistics
    print("\n=== Statistics ===")
    print(f"Node at position {actual_position:.3f}:")
    
    gt_final = gt_top_disp[-1, component, node_idx]
    baseline_final = baseline_top_disp[-1, component, node_idx]
    dd_final = dd_top_disp[-1, component, node_idx]
    
    print(f"  Final displacement:")
    print(f"    Ground truth: {gt_final:.2e} m")
    print(f"    Baseline:     {baseline_final:.2e} m")
    print(f"    Data-driven:  {dd_final:.2e} m")
    
    baseline_error = abs(baseline_final - gt_final)
    dd_error = abs(dd_final - gt_final)
    
    print(f"  Absolute error vs ground truth:")
    print(f"    Baseline:     {baseline_error:.2e} m")
    print(f"    Data-driven:  {dd_error:.2e} m")
    
    if baseline_error > 0:
        improvement = (baseline_error - dd_error) / baseline_error * 100
        print(f"    Improvement:  {improvement:.1f}%")
    
    # Display both plots
    plt.show()

if __name__ == "__main__":
    main()
