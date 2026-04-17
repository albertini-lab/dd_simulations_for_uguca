#!/usr/bin/env python3
"""
Compare w-factor study results to verify data-driven effect.
Shows displacement comparison across different w_factors.
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path

def read_out_file(filepath, time_file):
    """Read ASCII .out format file from uguca simulation."""
    if not os.path.exists(filepath):
        print(f"  File not found: {filepath}")
        return None, None
    
    with open(time_file, 'r') as f:
        num_timesteps = len(f.readlines())
    
    data_list = []
    with open(filepath, 'r') as f:
        for line in f:
            values = [float(v) for v in line.split()]
            data_list.append(values)
    
    data = np.array(data_list)
    return data, data.shape[0]

def plot_w_factor_comparison(output_dir="simulation_outputs", nb_nodes=1024):
    """Compare displacements across different w_factors."""
    
    # W-factors to look for
    w_factors = [
        ("w0", 0.0),
        ("w1", 1.0),
        ("w100", 100.0),
        ("w10000", 10000.0),
        ("w1e6", 1e6),
        ("w1e9", 1e9),
    ]
    
    # Colors for each w_factor
    colors = plt.cm.viridis(np.linspace(0, 1, len(w_factors)))
    
    # Find available results
    results = {}
    build_output = "/Users/joshmcneely/uguca/build/simulations/simulation_outputs"
    
    for w_str, w_val in w_factors:
        sim_name = f"w_study_{w_str}"
        disp_file = os.path.join(build_output, f"{sim_name}-interface-DataFiles/top_disp.out")
        time_file = os.path.join(build_output, f"{sim_name}-interface.time")
        
        if os.path.exists(disp_file) and os.path.exists(time_file):
            data, steps = read_out_file(disp_file, time_file)
            if data is not None:
                with open(time_file, 'r') as f:
                    times = np.array([float(line.split()[0]) for line in f.readlines()[:steps]])
                results[w_str] = {
                    'w_val': w_val,
                    'data': data,
                    'times': times,
                    'steps': steps,
                }
                print(f"  ✓ Loaded {sim_name}: {steps} timesteps")
        else:
            print(f"  ✗ Not found: {sim_name}")
    
    if len(results) < 2:
        print("\nERROR: Need at least 2 simulations to compare!")
        print("Run simulations first: ./run_w_study_local.sh")
        return
    
    # Get baseline (w=0)
    baseline_key = "w0" if "w0" in results else list(results.keys())[0]
    baseline = results[baseline_key]
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Sensor locations to plot
    sensor_nodes = [256, 512, 768]  # Different positions along interface
    sensor_x = [n * 6.0 / nb_nodes for n in sensor_nodes]
    
    # Plot 1: Displacement time series at x=3m (middle)
    ax1 = axes[0, 0]
    node = 512
    for i, (w_str, res) in enumerate(results.items()):
        label = f"w={res['w_val']:.0e}" if res['w_val'] >= 1000 else f"w={res['w_val']}"
        ax1.plot(res['times'] * 1e3, res['data'][:, node], 
                 color=colors[i % len(colors)], linewidth=1.5, label=label)
    ax1.set_xlabel('Time (ms)')
    ax1.set_ylabel('Displacement (m)')
    ax1.set_title(f'Displacement at x = {node * 6.0 / nb_nodes:.2f} m')
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Difference from baseline at middle
    ax2 = axes[0, 1]
    for i, (w_str, res) in enumerate(results.items()):
        if w_str == baseline_key:
            continue
        min_steps = min(res['steps'], baseline['steps'])
        diff = res['data'][:min_steps, node] - baseline['data'][:min_steps, node]
        label = f"w={res['w_val']:.0e}" if res['w_val'] >= 1000 else f"w={res['w_val']}"
        ax2.plot(res['times'][:min_steps] * 1e3, diff,
                 color=colors[i % len(colors)], linewidth=1.5, label=f"{label} - baseline")
    ax2.set_xlabel('Time (ms)')
    ax2.set_ylabel('Displacement Difference (m)')
    ax2.set_title(f'Difference from w=0 baseline at x = {node * 6.0 / nb_nodes:.2f} m')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Max displacement difference vs w_factor
    ax3 = axes[1, 0]
    w_vals = []
    max_diffs = []
    for w_str, res in results.items():
        if w_str == baseline_key:
            continue
        min_steps = min(res['steps'], baseline['steps'])
        diff = res['data'][:min_steps, :] - baseline['data'][:min_steps, :]
        max_diff = np.max(np.abs(diff))
        w_vals.append(res['w_val'])
        max_diffs.append(max_diff)
    
    if w_vals:
        ax3.semilogx(w_vals, max_diffs, 'bo-', markersize=10, linewidth=2)
        ax3.set_xlabel('w_factor')
        ax3.set_ylabel('Max |displacement difference| (m)')
        ax3.set_title('Effect of w_factor on Displacement')
        ax3.grid(True, alpha=0.3)
    
    # Plot 4: L2 norm of difference over time
    ax4 = axes[1, 1]
    for i, (w_str, res) in enumerate(results.items()):
        if w_str == baseline_key:
            continue
        min_steps = min(res['steps'], baseline['steps'])
        l2_norms = np.linalg.norm(res['data'][:min_steps, :] - baseline['data'][:min_steps, :], axis=1)
        label = f"w={res['w_val']:.0e}" if res['w_val'] >= 1000 else f"w={res['w_val']}"
        ax4.plot(res['times'][:min_steps] * 1e3, l2_norms,
                 color=colors[i % len(colors)], linewidth=1.5, label=label)
    ax4.set_xlabel('Time (ms)')
    ax4.set_ylabel('L2 Norm of Difference')
    ax4.set_title('Total Difference from Baseline Over Time')
    ax4.legend(fontsize=8)
    ax4.grid(True, alpha=0.3)
    
    plt.suptitle('W-Factor Study: Does Data-Driven Mode Have Any Effect?', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    # Save
    output_file = os.path.join(build_output, "w_factor_comparison.png")
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\nSaved: {output_file}")
    
    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY: Effect of w_factor on simulation")
    print("=" * 60)
    if max_diffs:
        print(f"  Max displacement difference from baseline:")
        for w, d in sorted(zip(w_vals, max_diffs)):
            w_str = f"{w:.0e}" if w >= 1000 else f"{w}"
            print(f"    w = {w_str:>10}: {d:.3e} m")
        
        if max(max_diffs) < 1e-15:
            print("\n  ⚠️  WARNING: No significant effect detected!")
            print("      The data-driven method may not be working.")
        elif max(max_diffs) < 1e-10:
            print("\n  ⚠️  CAUTION: Very small effect detected.")
            print("      May need to check data loading or w_factor scaling.")
        else:
            print("\n  ✓  Data-driven method IS having an effect!")
    
    plt.show()

if __name__ == "__main__":
    print("=" * 60)
    print("W-Factor Study Comparison")
    print("=" * 60)
    plot_w_factor_comparison()
