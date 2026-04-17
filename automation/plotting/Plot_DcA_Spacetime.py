#!/usr/bin/env python3
"""
Plot_DcA_Spacetime.py
Creates spacetime heatmap visualization comparing baseline, data-driven,
and experimental data as position-vs-time colormaps.

Input modes:
1) Legacy Dc/a mode:
   python Plot_DcA_Spacetime.py 1e-6 0.012
2) Explicit directory mode (fully generic):
   python Plot_DcA_Spacetime.py --dd-dir path/to/dd_case --baseline-dir path/to/base_case
"""

import argparse
import glob
import os
import re
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# --- CONFIGURATION ---
SIM_DIR = Path("/Users/joshmcneely/uguca/build/simulations")
EXP_BASE_DIR = Path("/Users/joshmcneely/introsims/simulation_outputs")
PLOT_DIR = SIM_DIR / "comparison_plots"

DOMAIN_LENGTH = 6.0
NB_NODES = 2048
DD_SIM_NAME = "local_debug_run"
BASE_SIM_NAME = "local_baseline_run"

# Sensor positions at 0.2m spacing from 0.05m to 3.05m (16 sensors)
SENSOR_POSITIONS = np.arange(0.05, 3.05 + 0.01, 0.2)
SENSOR_POSITIONS = np.round(SENSOR_POSITIONS, 12)
SENSOR_POSITIONS = SENSOR_POSITIONS[SENSOR_POSITIONS <= 3.05]

MIN_TIME_STEPS = 10
MIN_SIM_TIME = 1e-6


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate spacetime comparison plots for DD/baseline runs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("dc", nargs="?", help="Dc value (legacy mode)")
    parser.add_argument("a", nargs="?", help="a value (legacy mode)")

    parser.add_argument("--sim-dir", type=Path, default=SIM_DIR, help="Root directory containing simulation case directories")
    parser.add_argument("--plot-dir", type=Path, default=PLOT_DIR, help="Directory to write output plots")
    parser.add_argument("--exp-base-dir", type=Path, default=EXP_BASE_DIR, help="Experimental data base directory")
    parser.add_argument("--exp-restart-dir", type=Path, default=None, help="Direct path to experimental restart directory")

    parser.add_argument("--dd-dir", type=Path, default=None, help="Explicit data-driven case directory")
    parser.add_argument("--baseline-dir", type=Path, default=None, help="Explicit baseline case directory")
    parser.add_argument("--case-label", type=str, default=None, help="Custom case label used in plot title")
    parser.add_argument("--output-tag", type=str, default=None, help="Output filename tag (without extension)")

    parser.add_argument(
        "--dd-template",
        type=str,
        default="results_dd_Dc{dc}_a{a}",
        help="Template used in legacy mode to locate DD case directory",
    )
    parser.add_argument(
        "--baseline-template",
        type=str,
        default="results_baseline_Dc{dc}_a{a}",
        help="Template used in legacy mode to locate baseline case directory",
    )

    parser.add_argument("--dd-sim-name", type=str, default=DD_SIM_NAME, help="DD simulation run prefix")
    parser.add_argument("--baseline-sim-name", type=str, default=BASE_SIM_NAME, help="Baseline simulation run prefix")
    parser.add_argument("--nb-nodes", type=int, default=NB_NODES, help="Number of interface nodes")
    parser.add_argument("--domain-length", type=float, default=DOMAIN_LENGTH, help="Domain length in meters")
    parser.add_argument("--roi-min", type=float, default=0.0, help="Minimum x-position for spacetime ROI")
    parser.add_argument("--roi-max", type=float, default=3.05, help="Maximum x-position for spacetime ROI")

    args = parser.parse_args()

    explicit_mode = args.dd_dir is not None or args.baseline_dir is not None
    legacy_mode = args.dc is not None or args.a is not None

    if explicit_mode and legacy_mode:
        parser.error("Use either explicit dirs (--dd-dir/--baseline-dir) or legacy positional Dc/a, not both.")

    if explicit_mode and (args.dd_dir is None or args.baseline_dir is None):
        parser.error("Both --dd-dir and --baseline-dir are required in explicit mode.")

    if not explicit_mode and not (args.dc is not None and args.a is not None):
        parser.error("Provide either positional <dc> <a> or both --dd-dir and --baseline-dir.")

    return args


def sanitize_tag(text):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("_")


def resolve_case_inputs(args):
    if args.dd_dir is not None and args.baseline_dir is not None:
        dd_dir = args.dd_dir
        baseline_dir = args.baseline_dir

        label = args.case_label or f"{dd_dir.name} vs {baseline_dir.name}"
        tag = args.output_tag or sanitize_tag(label)
        title = label
        return dd_dir, baseline_dir, title, tag

    dd_dir = args.sim_dir / args.dd_template.format(dc=args.dc, a=args.a)
    baseline_dir = args.sim_dir / args.baseline_template.format(dc=args.dc, a=args.a)

    try:
        dc_val = float(args.dc)
        a_val = float(args.a)
        title = args.case_label or f"Dc={dc_val:.2e}, a={a_val:.2e}"
    except ValueError:
        title = args.case_label or f"Dc={args.dc}, a={args.a}"

    default_tag = f"Dc{args.dc}_a{args.a}"
    tag = args.output_tag or sanitize_tag(default_tag)

    return dd_dir, baseline_dir, title, tag


def is_simulation_complete(base_path, sim_name, nb_nodes):
    """Check if simulation has completed successfully."""
    _ = nb_nodes
    try:
        data_dir = base_path / f"{sim_name}-interface-DataFiles"
        path = data_dir / "top_disp.out"
        t_path = base_path / f"{sim_name}-interface.time"

        if not path.exists() or not t_path.exists():
            return False, "missing top_disp/time file"
        if path.stat().st_size == 0 or t_path.stat().st_size == 0:
            return False, "empty top_disp/time file"

        with open(t_path, "r") as file_obj:
            time_lines = [line for line in file_obj.readlines() if line.strip()]
        if len(time_lines) < MIN_TIME_STEPS:
            return False, f"only {len(time_lines)} time rows"

        try:
            times = np.array([float(line.split()[1]) for line in time_lines], dtype=float)
        except (IndexError, ValueError):
            return False, "malformed time file"

        if times[-1] < MIN_SIM_TIME:
            return False, f"t_end={times[-1]:.3e}s below minimum"

        return True, f"n_steps={len(times)}, t_end={times[-1]:.3e}s"

    except Exception as exc:
        return False, f"check exception: {exc}"


def load_sim(base_path, sim_name, nb_nodes):
    """Load simulation data (time and displacement for all nodes)."""
    complete, reason = is_simulation_complete(base_path, sim_name, nb_nodes)
    if not complete:
        print(f"[WARNING] Simulation incomplete ({reason})")
        return None, None

    try:
        data_dir = base_path / f"{sim_name}-interface-DataFiles"
        disp_path = data_dir / "top_disp.out"
        time_path = base_path / f"{sim_name}-interface.time"

        # Load times
        with open(time_path, "r") as file_obj:
            time_lines = [line for line in file_obj.readlines() if line.strip()]
        times = np.array([float(line.split()[1]) for line in time_lines], dtype=float)

        # Load displacement data
        data = np.loadtxt(disp_path, dtype=float)
        if data.ndim == 1:
            data = data.reshape(1, -1)

        return times, data

    except Exception as exc:
        print(f"[ERROR] Failed to load {base_path / sim_name}: {exc}")
        return None, None


def load_exp(restart_dir, sensor_nodes, nb_nodes):
    """Load experimental data from restart files."""
    restart_name = restart_dir.name
    exp_name = restart_name[:-8] if restart_name.endswith("-restart") else restart_name
    info_file = restart_dir.parent / f"{exp_name}.info"
    if not info_file.exists():
        raise FileNotFoundError(f"Experimental info file not found: {info_file}")

    with open(info_file, "r") as file_obj:
        dt_sim = float(re.search(r"time_step\s*=\s*([\d.eE+-]+)", file_obj.read()).group(1))

    files = glob.glob(str(restart_dir / "top_disp.proc0.s*.out"))
    files.sort(key=lambda x: int(re.search(r"\.s(\d+)\.out$", x).group(1)))

    data, times = [], []
    for filename in files:
        step = int(re.search(r"\.s(\d+)\.out$", filename).group(1))
        times.append(step * dt_sim)
        raw_bin = np.fromfile(filename, dtype=np.float32)
        u_x = raw_bin[:nb_nodes]
        data.append(u_x[sensor_nodes])

    return np.array(times), np.array(data)


def extract_full_spatial_data(times, data, roi_min=0.0, roi_max=3.05, dx=None, domain_length=DOMAIN_LENGTH):
    """Extract full spatial resolution displacement data within ROI and convert to relative slip."""
    # data shape: (n_timesteps, n_nodes) or (n_nodes,) for single timestep
    if data.ndim == 1:
        data = data.reshape(1, -1)
    
    _ = len(times)
    n_nodes = data.shape[1]
    
    # Calculate node positions
    if dx is None:
        dx = domain_length / n_nodes
    positions = np.arange(n_nodes) * dx
    
    # Select nodes within ROI [roi_min, roi_max]
    roi_mask = (positions >= roi_min) & (positions <= roi_max)
    roi_indices = np.where(roi_mask)[0]
    roi_positions = positions[roi_indices]
    
    print(f"    Using {len(roi_indices)} nodes in ROI [{roi_min}, {roi_max}] m (spatial resolution: {dx*1000:.2f} mm)")
    
    # Extract ROI data
    roi_data = data[:, roi_indices]
    
    # Convert to relative slip (subtract initial value, multiply by 2 for top_disp, convert to microns)
    slip_relative = np.zeros_like(roi_data.T)  # Transpose to (n_nodes, n_timesteps)
    for i in range(len(roi_indices)):
        slip_relative[i, :] = (roi_data[:, i] - roi_data[0, i]) * 2.0 * 1e6  # microns
    
    return slip_relative, roi_positions


def plot_spacetime_comparison(
    dd_dir,
    baseline_dir,
    case_title,
    output_tag,
    *,
    plot_dir,
    exp_restart_dir,
    dd_sim_name,
    baseline_sim_name,
    nb_nodes,
    domain_length,
    roi_min,
    roi_max,
):
    """
    Create spacetime heatmap comparison for given Dc and a values.
    Shows three subplots: Data-Driven, Baseline, and Experimental.
    """
    if not dd_dir.exists():
        print(f"[ERROR] Data-driven directory not found: {dd_dir}")
        return False
    if not baseline_dir.exists():
        print(f"[ERROR] Baseline directory not found: {baseline_dir}")
        return False
    
    # Calculate spatial resolution
    dx = domain_length / nb_nodes
    
    # Calculate sensor node indices (for experimental data only)
    sensor_nodes_raw = np.round(SENSOR_POSITIONS / dx).astype(int)
    valid_sensors = sensor_nodes_raw < nb_nodes
    sensor_nodes = sensor_nodes_raw[valid_sensors]

    print(f"[INFO] Loading simulations for case: {case_title}")
    
    # Load data-driven simulation
    print("  Loading data-driven simulation...")
    dd_times, dd_data = load_sim(dd_dir, dd_sim_name, nb_nodes)
    if dd_times is None or dd_data is None:
        print("[ERROR] Failed to load data-driven simulation")
        return False
    
    # Load baseline simulation
    print("  Loading baseline simulation...")
    base_times, base_data = load_sim(baseline_dir, baseline_sim_name, nb_nodes)
    if base_times is None or base_data is None:
        print("[ERROR] Failed to load baseline simulation")
        return False
    
    # Load experimental data
    print("  Loading experimental data...")
    exp_times, exp_data = load_exp(exp_restart_dir, sensor_nodes, nb_nodes)
    if exp_data is None or len(exp_data) == 0:
        print("[ERROR] Failed to load experimental data")
        return False
    
    # Convert experimental data to relative slip in microns
    exp_slip = np.zeros_like(exp_data)
    for i in range(exp_data.shape[1]):
        exp_slip[:, i] = (exp_data[:, i] - exp_data[0, i]) * 2.0 * 1e6
    exp_slip = exp_slip.T  # Transpose to (n_sensors, n_timesteps)
    
    # Get sensor positions for experimental data
    sensor_positions = sensor_nodes * dx
    
    # Extract FULL SPATIAL RESOLUTION data from simulations
    print("  Processing simulation data at full spatial resolution...")
    dd_slip, dd_positions = extract_full_spatial_data(
        dd_times,
        dd_data,
        roi_min=roi_min,
        roi_max=roi_max,
        dx=dx,
        domain_length=domain_length,
    )
    base_slip, base_positions = extract_full_spatial_data(
        base_times,
        base_data,
        roi_min=roi_min,
        roi_max=roi_max,
        dx=dx,
        domain_length=domain_length,
    )
    
    print(f"  Baseline slip range: [{base_slip.min():.2f}, {base_slip.max():.2f}] microns")
    print(f"  DD slip range: [{dd_slip.min():.2f}, {dd_slip.max():.2f}] microns")
    print(f"  Experimental slip range: [{exp_slip.min():.2f}, {exp_slip.max():.2f}] microns")
    
    # Unified colorbar range across all three panels
    # Set vmax = 2x experimental peak so experimental peak sits at coolwarm midpoint (white)
    exp_vmin = float(exp_slip.min())
    exp_vmax = float(exp_slip.max())
    unified_vmin = exp_vmin
    unified_vmax = 2.0 * exp_vmax
    
    print(f"  Unified colorbar range: [{unified_vmin:.2f}, {unified_vmax:.2f}] microns")
    print(f"  Experimental peak ({exp_vmax:.2f}) at coolwarm midpoint")
    print(f"  Baseline: {len(base_positions)} nodes x {len(base_times)} timesteps")
    print(f"  DD: {len(dd_positions)} nodes x {len(dd_times)} timesteps")
    print(f"  Exp: {len(sensor_positions)} sensors x {len(exp_times)} timesteps")
    
    # Create figure with three subplots - REORDERED: Baseline -> DD -> Experimental
    fig, axes = plt.subplots(1, 3, figsize=(21, 6), sharey=False, constrained_layout=True)
    fig.suptitle(f"Spacetime Comparison: {case_title}", fontsize=16, fontweight='bold')
    
    # Plot 1: Baseline (HIGH RESOLUTION)
    # Interpolation options: 'nearest', 'bilinear', 'bicubic', 'spline16', 'spline36', 'lanczos', etc.
    im1 = axes[0].imshow(base_slip, aspect='auto', origin='lower', cmap='coolwarm',
                         vmin=unified_vmin, vmax=unified_vmax,
                         interpolation='nearest',
                         extent=[base_times.min(), base_times.max(), base_positions.min(), base_positions.max()])
    axes[0].set_xlabel('Time (s)', fontsize=12)
    axes[0].set_ylabel('Position (m)', fontsize=12)
    axes[0].set_title('Baseline Simulation\n(Full Resolution)', fontsize=13, fontweight='bold')
    axes[0].grid(True, alpha=0.3, linestyle='--')
    
    # Plot 2: Data-Driven (HIGH RESOLUTION)
    im2 = axes[1].imshow(dd_slip, aspect='auto', origin='lower', cmap='coolwarm',
                         vmin=unified_vmin, vmax=unified_vmax,
                         interpolation='nearest',
                         extent=[dd_times.min(), dd_times.max(), dd_positions.min(), dd_positions.max()])
    axes[1].set_xlabel('Time (s)', fontsize=12)
    axes[1].set_ylabel('Position (m)', fontsize=12)
    axes[1].set_title('Data-Driven Simulation\n(Full Resolution)', fontsize=13, fontweight='bold')
    axes[1].grid(True, alpha=0.3, linestyle='--')
    
    # Plot 3: Experimental (16 SENSORS) - uses 'nearest' to preserve discrete sensor measurements
    im3 = axes[2].imshow(exp_slip, aspect='auto', origin='lower', cmap='coolwarm',
                         vmin=unified_vmin, vmax=unified_vmax,
                         interpolation='nearest',
                         extent=[exp_times.min(), exp_times.max(), sensor_positions.min(), sensor_positions.max()])
    axes[2].set_xlabel('Time (s)', fontsize=12)
    axes[2].set_ylabel('Position (m)', fontsize=12)
    axes[2].set_title('Experimental Data\n(16 Sensors)', fontsize=13, fontweight='bold')
    axes[2].grid(True, alpha=0.3, linestyle='--')
    
    # Add one colorbar per panel (independent ranges)
    cbar1 = fig.colorbar(im1, ax=axes[0], pad=0.02)
    cbar1.set_label(r'Slip ($\mu$m)', fontsize=10)
    cbar1.ax.tick_params(labelsize=9)

    cbar2 = fig.colorbar(im2, ax=axes[1], pad=0.02)
    cbar2.set_label(r'Slip ($\mu$m)', fontsize=10)
    cbar2.ax.tick_params(labelsize=9)

    cbar3 = fig.colorbar(im3, ax=axes[2], pad=0.02)
    cbar3.set_label(r'Slip ($\mu$m)', fontsize=10)
    cbar3.ax.tick_params(labelsize=9)
    
    # Save figure
    plot_dir.mkdir(exist_ok=True)
    output_filename = f"spacetime_{output_tag}.png"
    output_path = plot_dir / output_filename
    
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n[SUCCESS] Saved spacetime plot to: {output_path}")
    plt.close(fig)
    
    return True


def main():
    args = parse_args()
    dd_dir, baseline_dir, case_title, output_tag = resolve_case_inputs(args)
    exp_restart_dir = args.exp_restart_dir or (args.exp_base_dir / "mcklaskey_debug-restart")

    print(f"\n{'='*60}")
    print(f"Spacetime Heatmap Plot Generator")
    print(f"{'='*60}")
    print(f"DD dir:        {dd_dir}")
    print(f"Baseline dir:  {baseline_dir}")
    print(f"Case title:    {case_title}")
    print(f"Output tag:    {output_tag}")
    print(f"Exp restart:   {exp_restart_dir}")
    print(f"{'='*60}\n")

    success = plot_spacetime_comparison(
        dd_dir,
        baseline_dir,
        case_title,
        output_tag,
        plot_dir=args.plot_dir,
        exp_restart_dir=exp_restart_dir,
        dd_sim_name=args.dd_sim_name,
        baseline_sim_name=args.baseline_sim_name,
        nb_nodes=args.nb_nodes,
        domain_length=args.domain_length,
        roi_min=args.roi_min,
        roi_max=args.roi_max,
    )

    if success:
        print(f"\n{'='*60}")
        print("Spacetime plot generation complete!")
        print(f"{'='*60}\n")
    else:
        print(f"\n{'='*60}")
        print("Spacetime plot generation FAILED")
        print(f"{'='*60}\n")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
