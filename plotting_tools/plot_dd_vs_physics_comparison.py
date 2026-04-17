#!/usr/bin/env python
"""
Plot comparison between data-driven and physics-only simulations.
Generates waterfall plots, heatmaps, and displacement time series.
"""
import numpy as np
import matplotlib.pyplot as plt
import os

# Configuration
OUTPUT_DIR = '/Users/joshmcneely/uguca/build/simulations/simulation_outputs/comparison_plots'
DATA_DIR = '/Users/joshmcneely/uguca/build/simulations/simulation_outputs'

DD_NAME = 'dd_mcklaskey_debug'
PHYS_NAME = 'dd_mcklaskey_debug_physics_only'

ROI_X_MIN = 0.0
ROI_X_MAX = 3.0  # meters
NUM_NODES = 16  # Number of nodes to plot

def load_simulation_data(sim_name):
    """Load displacement, velocity, and time data for a simulation."""
    data_dir = f'{DATA_DIR}/{sim_name}-interface-DataFiles'
    
    # Load displacement and velocity
    disp = np.loadtxt(f'{data_dir}/top_disp.out')
    velo = np.loadtxt(f'{data_dir}/top_velo.out')
    
    # Load time from .time file
    time_file = f'{DATA_DIR}/{sim_name}-interface.time'
    times = []
    with open(time_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                times.append(float(parts[1]))
    times = np.array(times)
    
    # Load coordinates from .coords file
    coords_file = f'{DATA_DIR}/{sim_name}-interface.coords'
    coords = np.loadtxt(coords_file)
    
    return times, disp, velo, coords

def main():
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("Loading data...")
    times_dd, disp_dd, velo_dd, coords_dd = load_simulation_data(DD_NAME)
    times_phys, disp_phys, velo_phys, coords_phys = load_simulation_data(PHYS_NAME)
    
    print(f"Data-Driven: {disp_dd.shape} timesteps x nodes, time range: {times_dd.min():.6e} to {times_dd.max():.6e} s")
    print(f"Physics-Only: {disp_phys.shape} timesteps x nodes, time range: {times_phys.min():.6e} to {times_phys.max():.6e} s")
    print(f"Coordinates shape: {coords_dd.shape}")
    
    # Get spatial positions (x-coordinates)
    if coords_dd.ndim > 1:
        x_coords = coords_dd[:, 0]
    else:
        x_coords = coords_dd
    
    print(f"X-coordinate range: {x_coords.min():.3f} to {x_coords.max():.3f} m")
    
    # Filter nodes in ROI
    valid_mask = (x_coords >= ROI_X_MIN) & (x_coords <= ROI_X_MAX)
    valid_indices = np.where(valid_mask)[0]
    
    if len(valid_indices) == 0:
        print("WARNING: No nodes in ROI, using all nodes")
        valid_indices = np.arange(len(x_coords))
    
    # Select evenly spaced nodes
    if len(valid_indices) > NUM_NODES:
        selection = np.linspace(0, len(valid_indices)-1, NUM_NODES, dtype=int)
        node_indices = valid_indices[selection]
    else:
        node_indices = valid_indices
    
    print(f"Selected {len(node_indices)} nodes for plotting")
    
    # Common time steps
    n_common = min(len(times_dd), len(times_phys))
    times = times_dd[:n_common]
    
    # Calculate differences
    disp_diff = disp_dd[:n_common] - disp_phys[:n_common]
    velo_diff = velo_dd[:n_common] - velo_phys[:n_common]
    
    print(f"\n=== COMPARISON STATISTICS ===")
    print(f"Max abs displacement diff: {np.max(np.abs(disp_diff)):.6e} m")
    print(f"RMS displacement diff: {np.sqrt(np.mean(disp_diff**2)):.6e} m")
    print(f"Max abs velocity diff: {np.max(np.abs(velo_diff)):.6e} m/s")
    print(f"RMS velocity diff: {np.sqrt(np.mean(velo_diff**2)):.6e} m/s")
    
    # =========================================================================
    # FIGURE 1: Waterfall Plot (Both runs side by side)
    # =========================================================================
    print("\nGenerating waterfall plots...")
    
    fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 10))
    fig1.suptitle('WATERFALL COMPARISON: Data-Driven vs Physics-Only', fontsize=18, fontweight='bold')
    
    # Calculate slip (relative displacement) in microns
    slip_factor = 2.0  # top_disp -> slip
    
    # Build matrices for selected nodes
    U_dd = np.zeros((len(node_indices), n_common))
    U_phys = np.zeros((len(node_indices), n_common))
    y_labels = []
    
    for i, n_idx in enumerate(node_indices):
        # Relative slip in microns
        U_dd[i, :] = (disp_dd[:n_common, n_idx] - disp_dd[0, n_idx]) * slip_factor * 1e6
        U_phys[i, :] = (disp_phys[:n_common, n_idx] - disp_phys[0, n_idx]) * slip_factor * 1e6
        y_labels.append(f"{x_coords[n_idx]:.2f} m")
    
    # Auto-scale
    max_val = max(np.max(np.abs(U_dd)), np.max(np.abs(U_phys)))
    if max_val == 0:
        max_val = 1.0
    scaling = 1.5 / max_val
    
    # Plot Data-Driven
    for i in range(len(node_indices)):
        visual_trace = (U_dd[i, :] * scaling) + i
        ax1.plot(times * 1e3, visual_trace, color='red', linewidth=0.8)
    
    ax1.set_xlabel('Time (ms)', fontsize=12)
    ax1.set_ylabel('Spatial Position (m)', fontsize=12)
    ax1.set_title('Data-Driven (w_factor=10)', fontsize=14, color='red')
    ax1.set_yticks(np.arange(len(node_indices)))
    ax1.set_yticklabels(y_labels)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(-1, len(node_indices))
    
    # Plot Physics-Only
    for i in range(len(node_indices)):
        visual_trace = (U_phys[i, :] * scaling) + i
        ax2.plot(times * 1e3, visual_trace, color='blue', linewidth=0.8)
    
    ax2.set_xlabel('Time (ms)', fontsize=12)
    ax2.set_ylabel('Spatial Position (m)', fontsize=12)
    ax2.set_title('Physics-Only (baseline)', fontsize=14, color='blue')
    ax2.set_yticks(np.arange(len(node_indices)))
    ax2.set_yticklabels(y_labels)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(-1, len(node_indices))
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig1.savefig(f'{OUTPUT_DIR}/waterfall_comparison.png', dpi=300)
    print(f"  Saved: {OUTPUT_DIR}/waterfall_comparison.png")
    plt.close(fig1)
    
    # =========================================================================
    # FIGURE 2: Heatmaps (Both runs side by side)
    # =========================================================================
    print("Generating heatmaps...")
    
    fig2, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig2.suptitle('HEATMAP COMPARISON: Data-Driven vs Physics-Only', fontsize=18, fontweight='bold')
    
    # Displacement heatmaps
    extent = [times.min()*1e3, times.max()*1e3, -0.5, len(node_indices)-0.5]
    
    im1 = axes[0, 0].imshow(U_dd, aspect='auto', origin='lower', cmap='viridis',
                             extent=extent, interpolation='nearest')
    axes[0, 0].set_title('Data-Driven: Slip', fontsize=12, color='red')
    axes[0, 0].set_xlabel('Time (ms)')
    axes[0, 0].set_ylabel('Node Index')
    axes[0, 0].set_yticks(np.arange(len(node_indices)))
    axes[0, 0].set_yticklabels(y_labels)
    cbar1 = plt.colorbar(im1, ax=axes[0, 0])
    cbar1.set_label('Slip (μm)')
    
    im2 = axes[0, 1].imshow(U_phys, aspect='auto', origin='lower', cmap='viridis',
                             extent=extent, interpolation='nearest')
    axes[0, 1].set_title('Physics-Only: Slip', fontsize=12, color='blue')
    axes[0, 1].set_xlabel('Time (ms)')
    axes[0, 1].set_ylabel('Node Index')
    axes[0, 1].set_yticks(np.arange(len(node_indices)))
    axes[0, 1].set_yticklabels(y_labels)
    cbar2 = plt.colorbar(im2, ax=axes[0, 1])
    cbar2.set_label('Slip (μm)')
    
    # Velocity heatmaps (mm/s)
    V_dd = np.zeros((len(node_indices), n_common))
    V_phys = np.zeros((len(node_indices), n_common))
    
    for i, n_idx in enumerate(node_indices):
        V_dd[i, :] = velo_dd[:n_common, n_idx] * 1e3  # to mm/s
        V_phys[i, :] = velo_phys[:n_common, n_idx] * 1e3
    
    vmax = max(np.max(np.abs(V_dd)), np.max(np.abs(V_phys)))
    
    im3 = axes[1, 0].imshow(V_dd, aspect='auto', origin='lower', cmap='hot',
                             extent=extent, interpolation='nearest', vmin=0, vmax=vmax)
    axes[1, 0].set_title('Data-Driven: Velocity', fontsize=12, color='red')
    axes[1, 0].set_xlabel('Time (ms)')
    axes[1, 0].set_ylabel('Node Index')
    axes[1, 0].set_yticks(np.arange(len(node_indices)))
    axes[1, 0].set_yticklabels(y_labels)
    cbar3 = plt.colorbar(im3, ax=axes[1, 0])
    cbar3.set_label('Velocity (mm/s)')
    
    im4 = axes[1, 1].imshow(V_phys, aspect='auto', origin='lower', cmap='hot',
                             extent=extent, interpolation='nearest', vmin=0, vmax=vmax)
    axes[1, 1].set_title('Physics-Only: Velocity', fontsize=12, color='blue')
    axes[1, 1].set_xlabel('Time (ms)')
    axes[1, 1].set_ylabel('Node Index')
    axes[1, 1].set_yticks(np.arange(len(node_indices)))
    axes[1, 1].set_yticklabels(y_labels)
    cbar4 = plt.colorbar(im4, ax=axes[1, 1])
    cbar4.set_label('Velocity (mm/s)')
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig2.savefig(f'{OUTPUT_DIR}/heatmap_comparison.png', dpi=300)
    print(f"  Saved: {OUTPUT_DIR}/heatmap_comparison.png")
    plt.close(fig2)
    
    # =========================================================================
    # FIGURE 3: Difference Heatmaps
    # =========================================================================
    print("Generating difference plots...")
    
    fig3, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    fig3.suptitle('DIFFERENCE: Data-Driven minus Physics-Only', fontsize=18, fontweight='bold')
    
    U_diff = U_dd - U_phys
    V_diff = V_dd - V_phys
    
    # Use symmetric colormap for differences
    slip_max = np.max(np.abs(U_diff))
    velo_max = np.max(np.abs(V_diff))
    
    im1 = ax1.imshow(U_diff, aspect='auto', origin='lower', cmap='RdBu_r',
                      extent=extent, interpolation='nearest', 
                      vmin=-slip_max if slip_max > 0 else -1, vmax=slip_max if slip_max > 0 else 1)
    ax1.set_title(f'Slip Difference (max: {slip_max:.2e} μm)', fontsize=12)
    ax1.set_xlabel('Time (ms)')
    ax1.set_ylabel('Node Index')
    ax1.set_yticks(np.arange(len(node_indices)))
    ax1.set_yticklabels(y_labels)
    cbar1 = plt.colorbar(im1, ax=ax1)
    cbar1.set_label('Δ Slip (μm)')
    
    im2 = ax2.imshow(V_diff, aspect='auto', origin='lower', cmap='RdBu_r',
                      extent=extent, interpolation='nearest',
                      vmin=-velo_max if velo_max > 0 else -1, vmax=velo_max if velo_max > 0 else 1)
    ax2.set_title(f'Velocity Difference (max: {velo_max:.2e} mm/s)', fontsize=12)
    ax2.set_xlabel('Time (ms)')
    ax2.set_ylabel('Node Index')
    ax2.set_yticks(np.arange(len(node_indices)))
    ax2.set_yticklabels(y_labels)
    cbar2 = plt.colorbar(im2, ax=ax2)
    cbar2.set_label('Δ Velocity (mm/s)')
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig3.savefig(f'{OUTPUT_DIR}/difference_heatmaps.png', dpi=300)
    print(f"  Saved: {OUTPUT_DIR}/difference_heatmaps.png")
    plt.close(fig3)
    
    # =========================================================================
    # FIGURE 4: Time series at selected positions
    # =========================================================================
    print("Generating time series plots...")
    
    # Select 4 representative nodes
    n_traces = min(4, len(node_indices))
    trace_indices = np.linspace(0, len(node_indices)-1, n_traces, dtype=int)
    
    fig4, axes = plt.subplots(n_traces, 2, figsize=(14, 3*n_traces))
    fig4.suptitle('TIME SERIES COMPARISON', fontsize=18, fontweight='bold')
    
    for i, trace_idx in enumerate(trace_indices):
        n_idx = node_indices[trace_idx]
        pos = x_coords[n_idx]
        
        # Displacement
        axes[i, 0].plot(times*1e3, U_dd[trace_idx, :], 'r-', label='Data-Driven', linewidth=1.5)
        axes[i, 0].plot(times*1e3, U_phys[trace_idx, :], 'b--', label='Physics-Only', linewidth=1.5)
        axes[i, 0].set_ylabel('Slip (μm)')
        axes[i, 0].set_title(f'Position: {pos:.2f} m')
        axes[i, 0].legend(loc='upper left', fontsize=8)
        axes[i, 0].grid(True, alpha=0.3)
        
        # Velocity
        axes[i, 1].plot(times*1e3, V_dd[trace_idx, :], 'r-', label='Data-Driven', linewidth=1.5)
        axes[i, 1].plot(times*1e3, V_phys[trace_idx, :], 'b--', label='Physics-Only', linewidth=1.5)
        axes[i, 1].set_ylabel('Velocity (mm/s)')
        axes[i, 1].set_title(f'Position: {pos:.2f} m')
        axes[i, 1].legend(loc='upper left', fontsize=8)
        axes[i, 1].grid(True, alpha=0.3)
    
    axes[-1, 0].set_xlabel('Time (ms)')
    axes[-1, 1].set_xlabel('Time (ms)')
    
    plt.tight_layout(rect=[0, 0.02, 1, 0.96])
    fig4.savefig(f'{OUTPUT_DIR}/time_series_comparison.png', dpi=300)
    print(f"  Saved: {OUTPUT_DIR}/time_series_comparison.png")
    plt.close(fig4)
    
    # =========================================================================
    # FIGURE 5: Overlay waterfall (both on same plot)
    # =========================================================================
    print("Generating overlay waterfall...")
    
    fig5, ax = plt.subplots(figsize=(12, 10))
    fig5.suptitle('OVERLAY WATERFALL: Data-Driven (red) vs Physics-Only (blue)', 
                   fontsize=16, fontweight='bold')
    
    for i in range(len(node_indices)):
        visual_dd = (U_dd[i, :] * scaling) + i
        visual_phys = (U_phys[i, :] * scaling) + i
        ax.plot(times * 1e3, visual_phys, color='blue', linewidth=1.0, alpha=0.7)
        ax.plot(times * 1e3, visual_dd, color='red', linewidth=1.0, alpha=0.7)
    
    ax.set_xlabel('Time (ms)', fontsize=12)
    ax.set_ylabel('Spatial Position (m)', fontsize=12)
    ax.set_yticks(np.arange(len(node_indices)))
    ax.set_yticklabels(y_labels)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-1, len(node_indices))
    
    # Add legend
    from matplotlib.lines import Line2D
    legend_elements = [Line2D([0], [0], color='red', label='Data-Driven'),
                       Line2D([0], [0], color='blue', label='Physics-Only')]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=12)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig5.savefig(f'{OUTPUT_DIR}/overlay_waterfall.png', dpi=300)
    print(f"  Saved: {OUTPUT_DIR}/overlay_waterfall.png")
    plt.close(fig5)
    
    print(f"\n=== All plots saved to: {OUTPUT_DIR} ===")
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY: DATA-DRIVEN vs PHYSICS-ONLY COMPARISON")
    print("="*60)
    print(f"Number of timesteps compared: {n_common}")
    print(f"Time range: {times.min()*1e3:.3f} to {times.max()*1e3:.3f} ms")
    print(f"")
    print(f"Displacement difference:")
    print(f"  Max absolute: {np.max(np.abs(disp_diff)):.6e} m")
    print(f"  RMS: {np.sqrt(np.mean(disp_diff**2)):.6e} m")
    print(f"  Relative to max displacement: {np.max(np.abs(disp_diff))/np.max(np.abs(disp_dd))*100:.6e} %")
    print(f"")
    print(f"Velocity difference:")
    print(f"  Max absolute: {np.max(np.abs(velo_diff)):.6e} m/s")
    print(f"  RMS: {np.sqrt(np.mean(velo_diff**2)):.6e} m/s")
    print(f"  Relative to max velocity: {np.max(np.abs(velo_diff))/np.max(np.abs(velo_dd))*100:.6e} %")
    print("="*60)
    
    if np.max(np.abs(disp_diff)) < 1e-15:
        print("\n⚠️  WARNING: Differences are at numerical noise level!")
        print("    The data-driven method appears to have NO EFFECT.")
        print("    This confirms the bug we identified.")

if __name__ == '__main__':
    main()
