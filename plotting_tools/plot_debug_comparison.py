#!/usr/bin/env python3
"""
Plot displacement comparison: Data-Driven vs Physics-Only
Reads .out format from uguca dumps
"""

import numpy as np
import matplotlib.pyplot as plt
import struct
import os
from pathlib import Path

def read_out_file(filepath, time_file, nb_nodes):
    """Read .out format file from uguca simulation (all timesteps).
    Supports both ASCII and binary formats."""
    if not os.path.exists(filepath):
        print(f"File not found: {filepath}")
        return None, None
    
    # Read time file to get number of timesteps
    with open(time_file, 'r') as f:
        num_timesteps = len(f.readlines())
    
    print(f"Reading {num_timesteps} timesteps from {filepath}")
    
    # Detect format: try reading as ASCII first (check if it's text)
    with open(filepath, 'rb') as f:
        first_bytes = f.read(100)
    
    is_ascii = all(b < 128 and (b >= 32 or b in (9, 10, 13)) for b in first_bytes)
    
    if is_ascii:
        # ASCII format: one line per timestep, space-separated values
        print("Detected ASCII format")
        data_list = []
        with open(filepath, 'r') as f:
            for line in f:
                values = [float(v) for v in line.split()]
                data_list.append(values)
        data = np.array(data_list)
        actual_timesteps = data.shape[0]
        print(f"Loaded {actual_timesteps} timesteps, {data.shape[1]} values per timestep")
        return data, actual_timesteps
    else:
        # Binary format
        print("Detected binary format")
        with open(filepath, 'rb') as f:
            data = np.fromfile(f, dtype=np.float64)
        
        values_per_timestep = nb_nodes
        print(f"Total values: {len(data)}, Expected: {values_per_timestep * num_timesteps}")
        
        try:
            data = data.reshape((num_timesteps, nb_nodes))
            return data, num_timesteps
        except ValueError as e:
            print(f"Reshape error: {e}")
            actual_timesteps = len(data) // values_per_timestep
            print(f"Actual timesteps that fit: {actual_timesteps}")
            if actual_timesteps > 0:
                data = data[:actual_timesteps * values_per_timestep]
                data = data.reshape((actual_timesteps, nb_nodes))
                return data, actual_timesteps
            return None, None

def load_simulation_data(base_path, sim_name, nb_nodes, field_name):
    """Load a specific simulation field from .out files."""
    filepath = os.path.join(base_path, f"{sim_name}-interface-DataFiles", f"{field_name}.out")
    timefile = os.path.join(base_path, f"{sim_name}-interface.time")
    return read_out_file(filepath, timefile, nb_nodes)

def choose_data_field(base_path, sim_name, nb_nodes, disp_threshold=1e-10):
    """Prefer displacement if non-trivial, otherwise fall back to velocity."""
    disp_data, disp_steps = load_simulation_data(base_path, sim_name, nb_nodes, "top_disp")
    if disp_data is not None:
        max_disp = np.max(np.abs(disp_data)) if disp_data.size else 0.0
        if max_disp >= disp_threshold:
            return "top_disp", disp_data, disp_steps

    velo_data, velo_steps = load_simulation_data(base_path, sim_name, nb_nodes, "top_velo")
    return "top_velo", velo_data, velo_steps

def read_experimental_restart(restart_dir, nb_nodes, num_timesteps, sensor_nodes):
    """Read experimental displacement from binary restart files at sensor nodes."""
    import glob
    import re

    # Find all files matching the pattern for top interface displacement
    files = glob.glob(os.path.join(restart_dir, "interface_top_disp.0.s*.bin"))
    # Extract step numbers and sort
    step_file_pairs = []
    for f in files:
        m = re.search(r'\.s(\d+)\.bin$', f)
        if m:
            step = int(m.group(1))
            step_file_pairs.append((step, f))
    step_file_pairs.sort()
    # Only keep up to num_timesteps
    step_file_pairs = step_file_pairs[:num_timesteps]
    # Preallocate array
    data = np.zeros((len(step_file_pairs), len(sensor_nodes)))
    for i, (step, f) in enumerate(step_file_pairs):
        arr = np.fromfile(f, dtype=np.float64)
        # arr shape: nb_nodes * 2 (x and y), but we want x only
        arr = arr.reshape(nb_nodes, 2)
        data[i, :] = arr[sensor_nodes, 0]  # x component
    return data, len(step_file_pairs)

def plot_comparison(output_dir="/Users/joshmcneely/uguca/build/simulations/simulation_outputs", length=6.0, nb_nodes=1024):
    # Create x-coordinate array
    x = np.linspace(0, length, nb_nodes)
    dx = length / nb_nodes

    # Define sensor locations (matching McKlaskey experimental setup)
    sensor_positions = np.arange(0.05, 3.05 + 0.01, 0.2)  # 0.05m to 3.05m, every 0.2m
    sensor_nodes = np.round(sensor_positions / dx).astype(int)
    sensor_nodes = sensor_nodes[sensor_nodes < nb_nodes]  # Keep only valid nodes

    print(f"Plotting at {len(sensor_nodes)} sensor locations: {sensor_positions[:len(sensor_nodes)]}")

    # Load data from both simulations (prefer displacement if valid)
    print("\nLoading data-driven simulation...")
    dd_field, dd_data, dd_steps = choose_data_field(output_dir, "dd_mcklaskey_debug", nb_nodes)

    print("Loading physics-only simulation...")
    po_field, po_data, po_steps = choose_data_field(output_dir, "dd_mcklaskey_debug_physics_only", nb_nodes)

    if dd_data is None or po_data is None:
        print("Error: Could not load simulation data")
        return

    print(f"Loaded {dd_steps} steps from data-driven simulation ({dd_field} data)")
    print(f"Loaded {po_steps} steps from physics-only simulation ({po_field} data)")

    # Use minimum number of steps
    num_steps = min(dd_steps, po_steps)

    # Load time data
    time_file = os.path.join(output_dir, "dd_mcklaskey_debug-interface.time")
    with open(time_file, 'r') as f:
        time_data = np.array([float(line.split()[0]) for line in f.readlines()[:num_steps]])

    # Plot all sensor locations
    num_sensors_to_plot = len(sensor_nodes)

    # Calculate L2 and L-infinity norms at sensor locations over time
    l2_norms = np.zeros(num_steps)
    linf_norms = np.zeros(num_steps)

    print("\n" + "="*60)
    print("Computing norms at sensor locations...")
    print("="*60)

    for step in range(num_steps):
        # Extract displacements at all sensor locations for this timestep
        dd_sensors = dd_data[step, sensor_nodes]  # Data-driven at sensor nodes (1D array)
        po_sensors = po_data[step, sensor_nodes]  # Physics-only at sensor nodes (1D array)

        # Compute difference
        diff = dd_sensors - po_sensors

        # L2 norm: sqrt(sum of squared differences)
        l2_norms[step] = np.linalg.norm(diff)

        # L-infinity norm: maximum absolute difference
        linf_norms[step] = np.max(np.abs(diff))

    # Print overall statistics
    print(f"\nNorm Statistics over {num_steps} timesteps at {len(sensor_nodes)} sensor locations:")
    print(f"  L2 norm:   min={np.min(l2_norms):.3e}, max={np.max(l2_norms):.3e}, mean={np.mean(l2_norms):.3e}")
    print(f"  L∞ norm:   min={np.min(linf_norms):.3e}, max={np.max(linf_norms):.3e}, mean={np.mean(linf_norms):.3e}")

    # Create figure with all displacement time series + norms
    # Layout: 4 rows x 4 cols for sensors, + 1 row for norms
    ncols = 4
    nrows_sensors = int(np.ceil(num_sensors_to_plot / ncols))
    total_rows = nrows_sensors + 1  # +1 for norm plots

    fig = plt.figure(figsize=(20, 4 * total_rows))
    gs = fig.add_gridspec(total_rows, ncols, hspace=0.35, wspace=0.3)

    # Plot all sensor time series
    for sensor_idx in range(num_sensors_to_plot):
        row = sensor_idx // ncols
        col = sensor_idx % ncols
        ax = fig.add_subplot(gs[row, col])

        node = sensor_nodes[sensor_idx]
        pos = sensor_positions[sensor_idx]

        # Extract time series for this sensor
        dd_timeseries = dd_data[:num_steps, node]
        po_timeseries = po_data[:num_steps, node]

        # Plot both time series
        ax.plot(time_data * 1e3, dd_timeseries, 'b-', linewidth=1.5, label='Data-Driven')
        ax.plot(time_data * 1e3, po_timeseries, 'r--', linewidth=1.5, label='Physics-Only')

        ax.set_xlabel('Time (ms)', fontsize=9)
        y_label = 'Displacement (m)' if dd_field == 'top_disp' else 'Velocity (m/s)'
        ax.set_ylabel(y_label, fontsize=9)
        ax.set_title(f'x = {pos:.2f} m', fontsize=10, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
        ax.tick_params(labelsize=8)

        # Print statistics
        max_dd = np.max(np.abs(dd_timeseries))
        max_po = np.max(np.abs(po_timeseries))
        diff = dd_timeseries - po_timeseries
        rms_diff = np.sqrt(np.mean(diff**2))

        print(f"\nSensor at x={pos:.2f}m (node {node}):")
        if dd_field == 'top_disp':
            print(f"  Max DD disp: {max_dd:.3e} m")
            print(f"  Max PO disp: {max_po:.3e} m")
            print(f"  RMS disp diff: {rms_diff:.3e} m")
        else:
            print(f"  Max DD velocity: {max_dd:.3e} m/s")
            print(f"  Max PO velocity: {max_po:.3e} m/s")
            print(f"  RMS velocity diff: {rms_diff:.3e} m/s")

    # Last row: norm plots spanning full width
    ax_l2 = fig.add_subplot(gs[nrows_sensors, :2])  # L2 norm spans 2 columns
    ax_linf = fig.add_subplot(gs[nrows_sensors, 2:])  # L-infinity norm spans 2 columns

    # Plot L2 norm
    ax_l2.plot(time_data * 1e3, l2_norms, 'g-', linewidth=2)
    ax_l2.set_xlabel('Time (ms)', fontsize=10)
    l2_label = 'L2 Norm (m)' if dd_field == 'top_disp' else 'L2 Norm (m/s)'
    ax_l2.set_ylabel(l2_label, fontsize=10)
    ax_l2.set_title(f'L2 Norm of Difference (at {len(sensor_nodes)} sensors)', fontsize=11, fontweight='bold')
    ax_l2.grid(True, alpha=0.3)
    ax_l2.set_xlim(time_data[0] * 1e3, time_data[-1] * 1e3)

    # Plot L-infinity norm
    ax_linf.plot(time_data * 1e3, linf_norms, 'm-', linewidth=2)
    ax_linf.set_xlabel('Time (ms)', fontsize=10)
    linf_label = 'L∞ Norm (m)' if dd_field == 'top_disp' else 'L∞ Norm (m/s)'
    ax_linf.set_ylabel(linf_label, fontsize=10)
    ax_linf.set_title('L∞ Norm (max abs diff)', fontsize=11, fontweight='bold')
    ax_linf.grid(True, alpha=0.3)
    ax_linf.set_xlim(time_data[0] * 1e3, time_data[-1] * 1e3)

    # Save figure
    output_file = os.path.join(output_dir, "debug_comparison_timeseries.png")
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\nSaved figure to: {output_file}")

    plt.show()

if __name__ == "__main__":
    # Change to simulations directory if needed
    if os.path.exists("/Users/joshmcneely/uguca/simulations"):
        os.chdir("/Users/joshmcneely/uguca/simulations")

    plot_comparison()
