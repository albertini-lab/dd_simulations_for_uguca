#!/usr/bin/env python3
"""
Create one combined overlay figure containing:
- Data-driven direct run
- Data-driven spline run
- Baseline nu=0.25 run
- Baseline nu=0.19 run
- Experimental direct and spline datasets

The output figure uses a 4x4 sensor grid plus a global RMSE panel.
"""

import argparse
import glob
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

# --- Defaults ---
SIM_DIR = Path(".")
PLOT_DIR = SIM_DIR / "comparison_plots"
EXP_BASE_DIR = Path("/Users/joshmcneely/introsims/simulation_outputs")

DOMAIN_LENGTH = 6.0
NB_NODES = 2048
SENSOR_POSITIONS = np.arange(0.05, 3.05 + 0.01, 0.2)
SENSOR_POSITIONS = np.round(SENSOR_POSITIONS, 12)
SENSOR_POSITIONS = SENSOR_POSITIONS[SENSOR_POSITIONS <= 3.05]
SIM_DENSITY_MULTIPLIER = 30
COMPLETION_TIME_FRACTION = 0.999

DD_SIM_NAME = "local_debug_run"
BASE_SIM_NAME = "local_baseline_run"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create a single overlay for direct+spline DD and both baselines.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--plot-dir", type=Path, default=PLOT_DIR, help="Directory to write plots")
    parser.add_argument(
        "--output-name",
        type=str,
        default="overlay_direct_spline_both_baselines.png",
        help="Output filename",
    )

    parser.add_argument(
        "--dd-direct-dir",
        type=Path,
        default=SIM_DIR / "results_dd_mcklaskey_debug_direct",
        help="Direct DD results directory",
    )
    parser.add_argument(
        "--dd-spline-dir",
        type=Path,
        default=SIM_DIR / "results_dd_mcklaskey_debug_spline",
        help="Spline DD results directory",
    )
    parser.add_argument(
        "--baseline-025-dir",
        type=Path,
        default=SIM_DIR / "results_baseline",
        help="Baseline nu=0.25 results directory",
    )
    parser.add_argument(
        "--baseline-019-dir",
        type=Path,
        default=SIM_DIR / "results_baseline_local_compare",
        help="Baseline nu=0.19 results directory",
    )

    parser.add_argument(
        "--exp-direct-restart-dir",
        type=Path,
        default=EXP_BASE_DIR / "mcklaskey_debug_direct-restart",
        help="Experimental direct restart directory",
    )
    parser.add_argument(
        "--exp-spline-restart-dir",
        type=Path,
        default=EXP_BASE_DIR / "mcklaskey_debug_spline-restart",
        help="Experimental spline restart directory",
    )

    parser.add_argument("--dd-sim-name", type=str, default=DD_SIM_NAME, help="DD simulation run prefix")
    parser.add_argument("--baseline-sim-name", type=str, default=BASE_SIM_NAME, help="Baseline simulation run prefix")
    parser.add_argument("--nb-nodes", type=int, default=NB_NODES, help="Number of interface nodes")
    parser.add_argument("--domain-length", type=float, default=DOMAIN_LENGTH, help="Domain length in meters")
    parser.add_argument(
        "--sim-density-multiplier",
        type=int,
        default=SIM_DENSITY_MULTIPLIER,
        help="Simulation plotting density multiplier relative to experimental sampling",
    )
    parser.add_argument(
        "--completion-time-fraction",
        type=float,
        default=COMPLETION_TIME_FRACTION,
        help="Minimum fraction of final experimental time required for accepted runs",
    )

    return parser.parse_args()


def exp_name_from_restart_dir(restart_dir):
    name = restart_dir.name
    suffix = "-restart"
    if name.endswith(suffix):
        return name[:-len(suffix)]
    return name


def is_simulation_complete(base_path, sim_name):
    try:
        data_dir = base_path / f"{sim_name}-interface-DataFiles"
        disp_path = data_dir / "top_disp.out"
        time_path = base_path / f"{sim_name}-interface.time"

        if not disp_path.exists() or not time_path.exists():
            return False, "missing top_disp/time file"
        if disp_path.stat().st_size == 0 or time_path.stat().st_size == 0:
            return False, "empty top_disp/time file"

        with open(time_path, "r") as file_obj:
            lines = [line for line in file_obj.readlines() if line.strip()]

        if len(lines) < 10:
            return False, f"only {len(lines)} time rows"

        times = np.array([float(line.split()[1]) for line in lines], dtype=float)
        if times[-1] < 1e-6:
            return False, f"t_end={times[-1]:.3e}s below minimum"

        return True, f"n_steps={len(times)}, t_end={times[-1]:.3e}s"
    except Exception as exc:
        return False, f"check exception: {exc}"


def load_sim(base_path, sim_name, nb_nodes):
    complete, reason = is_simulation_complete(base_path, sim_name)
    if not complete:
        raise RuntimeError(f"Simulation incomplete in {base_path}: {reason}")

    data_dir = base_path / f"{sim_name}-interface-DataFiles"
    disp_path = data_dir / "top_disp.out"
    time_path = base_path / f"{sim_name}-interface.time"

    with open(time_path, "r") as file_obj:
        times = np.array([float(line.split()[1]) for line in file_obj.readlines() if line.strip()], dtype=float)

    raw_data = np.loadtxt(disp_path)
    if raw_data.ndim == 1:
        raw_data = raw_data.reshape(1, -1)

    data = raw_data[:, :nb_nodes]
    n = min(len(times), data.shape[0])
    return times[:n], data[:n, :]


def load_exp(restart_dir, sensor_nodes, nb_nodes):
    exp_name = exp_name_from_restart_dir(restart_dir)
    info_file = restart_dir.parent / f"{exp_name}.info"
    if not info_file.exists():
        raise FileNotFoundError(f"Experimental info file not found: {info_file}")

    with open(info_file, "r") as file_obj:
        dt_sim = float(re.search(r"time_step\s*=\s*([\d.eE+-]+)", file_obj.read()).group(1))

    files = glob.glob(str(restart_dir / "top_disp.proc0.s*.out"))
    files.sort(key=lambda name: int(re.search(r"\.s(\d+)\.out$", name).group(1)))

    data = []
    times = []
    for file_path in files:
        step = int(re.search(r"\.s(\d+)\.out$", file_path).group(1))
        times.append(step * dt_sim)
        raw_bin = np.fromfile(file_path, dtype=np.float32)
        u_x = raw_bin[:nb_nodes]
        data.append(u_x[sensor_nodes])

    return np.array(times), np.array(data)


def compute_rmse_per_time(sim_t, sim_d, exp_t, exp_d, sensor_nodes):
    idx = np.searchsorted(sim_t, exp_t).clip(0, len(sim_t) - 1)
    sim_sensors = sim_d[idx][:, sensor_nodes]
    return np.sqrt(np.mean((sim_sensors - exp_d) ** 2, axis=1))


def decimate_series(times, data, target_points):
    stride = max(1, len(times) // max(1, target_points))
    return times[::stride], data[::stride]


def average_rmse_on_direct_timeline(direct_t, direct_rmse, other_t, other_rmse):
    other_interp = np.interp(direct_t, other_t, other_rmse, left=np.nan, right=np.nan)
    combined = direct_rmse.copy()
    valid = ~np.isnan(other_interp)
    combined[valid] = 0.5 * (direct_rmse[valid] + other_interp[valid])
    return combined


def require_dir(path, label):
    if not path.exists():
        raise FileNotFoundError(f"Missing {label}: {path}")


def main():
    args = parse_args()
    args.plot_dir.mkdir(exist_ok=True)

    require_dir(args.dd_direct_dir, "direct DD directory")
    require_dir(args.dd_spline_dir, "spline DD directory")
    require_dir(args.baseline_025_dir, "baseline nu=0.25 directory")
    require_dir(args.baseline_019_dir, "baseline nu=0.19 directory")
    require_dir(args.exp_direct_restart_dir, "experimental direct restart directory")
    require_dir(args.exp_spline_restart_dir, "experimental spline restart directory")

    dx = args.domain_length / args.nb_nodes
    sensor_nodes_raw = np.round(SENSOR_POSITIONS / dx).astype(int)
    valid_sensors = sensor_nodes_raw < args.nb_nodes
    sensor_nodes = sensor_nodes_raw[valid_sensors]
    sensor_positions = sensor_nodes * dx

    print("[INFO] Loading experimental datasets...")
    exp_direct_t, exp_direct_d = load_exp(args.exp_direct_restart_dir, sensor_nodes, args.nb_nodes)
    exp_spline_t, exp_spline_d = load_exp(args.exp_spline_restart_dir, sensor_nodes, args.nb_nodes)

    print("[INFO] Loading simulation datasets...")
    dd_direct_t, dd_direct_d = load_sim(args.dd_direct_dir, args.dd_sim_name, args.nb_nodes)
    dd_spline_t, dd_spline_d = load_sim(args.dd_spline_dir, args.dd_sim_name, args.nb_nodes)
    base_025_t, base_025_d = load_sim(args.baseline_025_dir, args.baseline_sim_name, args.nb_nodes)
    base_019_t, base_019_d = load_sim(args.baseline_019_dir, args.baseline_sim_name, args.nb_nodes)

    required_direct = exp_direct_t[-1] * args.completion_time_fraction
    required_spline = exp_spline_t[-1] * args.completion_time_fraction

    if dd_direct_t[-1] < required_direct:
        raise RuntimeError(
            f"Direct DD run too short: t_end={dd_direct_t[-1]:.6e}s, required>={required_direct:.6e}s"
        )
    if dd_spline_t[-1] < required_spline:
        raise RuntimeError(
            f"Spline DD run too short: t_end={dd_spline_t[-1]:.6e}s, required>={required_spline:.6e}s"
        )

    required_baseline = max(required_direct, required_spline)
    if base_025_t[-1] < required_baseline:
        raise RuntimeError(
            f"Baseline nu=0.25 run too short: t_end={base_025_t[-1]:.6e}s, required>={required_baseline:.6e}s"
        )
    if base_019_t[-1] < required_baseline:
        raise RuntimeError(
            f"Baseline nu=0.19 run too short: t_end={base_019_t[-1]:.6e}s, required>={required_baseline:.6e}s"
        )

    dd_direct_rmse = compute_rmse_per_time(dd_direct_t, dd_direct_d, exp_direct_t, exp_direct_d, sensor_nodes)
    dd_spline_rmse = compute_rmse_per_time(dd_spline_t, dd_spline_d, exp_spline_t, exp_spline_d, sensor_nodes)

    base025_rmse_direct = compute_rmse_per_time(base_025_t, base_025_d, exp_direct_t, exp_direct_d, sensor_nodes)
    base025_rmse_spline = compute_rmse_per_time(base_025_t, base_025_d, exp_spline_t, exp_spline_d, sensor_nodes)
    base019_rmse_direct = compute_rmse_per_time(base_019_t, base_019_d, exp_direct_t, exp_direct_d, sensor_nodes)
    base019_rmse_spline = compute_rmse_per_time(base_019_t, base_019_d, exp_spline_t, exp_spline_d, sensor_nodes)

    # Use direct experimental timeline as RMSE reference and average baseline fit quality
    # across direct and spline datasets for a single baseline curve.
    base025_rmse = average_rmse_on_direct_timeline(
        exp_direct_t,
        base025_rmse_direct,
        exp_spline_t,
        base025_rmse_spline,
    )
    base019_rmse = average_rmse_on_direct_timeline(
        exp_direct_t,
        base019_rmse_direct,
        exp_spline_t,
        base019_rmse_spline,
    )

    print("[INFO] Mean RMSE summary:")
    print(f"  DD direct:        {np.mean(dd_direct_rmse):.3e}")
    print(f"  DD spline:        {np.mean(dd_spline_rmse):.3e}")
    print(f"  Baseline nu=0.25: {np.mean(base025_rmse):.3e} (avg over direct+spline data)")
    print(f"  Baseline nu=0.19: {np.mean(base019_rmse):.3e} (avg over direct+spline data)")

    target_plot_points = max(len(exp_direct_t), len(exp_spline_t)) * args.sim_density_multiplier

    dd_direct_t_plot, dd_direct_d_plot = decimate_series(dd_direct_t, dd_direct_d, target_plot_points)
    dd_spline_t_plot, dd_spline_d_plot = decimate_series(dd_spline_t, dd_spline_d, target_plot_points)
    base025_t_plot, base025_d_plot = decimate_series(base_025_t, base_025_d, target_plot_points)
    base019_t_plot, base019_d_plot = decimate_series(base_019_t, base_019_d, target_plot_points)

    exp_direct_stride = max(1, len(exp_direct_t) // 400)
    exp_spline_stride = max(1, len(exp_spline_t) // 400)

    y_min = min(np.min(exp_direct_d), np.min(exp_spline_d))
    y_max = max(np.max(exp_direct_d), np.max(exp_spline_d))
    y_margin = (y_max - y_min) * 0.10
    if y_margin == 0:
        y_margin = 1e-8
    global_y_lim = (y_min - y_margin, y_max + y_margin)

    max_plot_time = min(
        exp_direct_t[-1],
        exp_spline_t[-1],
        dd_direct_t[-1],
        dd_spline_t[-1],
        base_025_t[-1],
        base_019_t[-1],
    )

    styles = {
        "dd_direct": {"color": "#0B5FFF", "linestyle": "-", "linewidth": 1.3, "label": "DD direct"},
        "dd_spline": {"color": "#00A3A3", "linestyle": "-", "linewidth": 1.3, "label": "DD spline"},
        "base_025": {"color": "#F08C00", "linestyle": "--", "linewidth": 1.5, "label": "Baseline nu=0.25"},
        "base_019": {"color": "#2F9E44", "linestyle": "--", "linewidth": 1.5, "label": "Baseline nu=0.19"},
    }

    fig = plt.figure(figsize=(22, 19))
    gs = fig.add_gridspec(5, 4, hspace=0.4, wspace=0.3, height_ratios=[1, 1, 1, 1, 0.9])

    num_sensors = len(sensor_nodes)
    for sensor_idx in range(num_sensors):
        row = sensor_idx // 4
        col = sensor_idx % 4
        axis = fig.add_subplot(gs[row, col])
        sensor_node = sensor_nodes[sensor_idx]

        exp_direct_mask = exp_direct_t[::exp_direct_stride] <= max_plot_time
        exp_spline_mask = exp_spline_t[::exp_spline_stride] <= max_plot_time

        axis.plot(
            exp_direct_t[::exp_direct_stride][exp_direct_mask] * 1e3,
            exp_direct_d[::exp_direct_stride][exp_direct_mask, sensor_idx],
            marker="x",
            color="#111111",
            linestyle="none",
            markersize=3,
            alpha=0.85,
            zorder=30,
        )
        axis.plot(
            exp_spline_t[::exp_spline_stride][exp_spline_mask] * 1e3,
            exp_spline_d[::exp_spline_stride][exp_spline_mask, sensor_idx],
            marker="+",
            color="#C92A2A",
            linestyle="none",
            markersize=3,
            alpha=0.85,
            zorder=30,
        )

        mask_direct = dd_direct_t_plot <= max_plot_time
        mask_spline = dd_spline_t_plot <= max_plot_time
        mask_base025 = base025_t_plot <= max_plot_time
        mask_base019 = base019_t_plot <= max_plot_time

        axis.plot(
            dd_direct_t_plot[mask_direct] * 1e3,
            dd_direct_d_plot[mask_direct, sensor_node],
            color=styles["dd_direct"]["color"],
            linestyle=styles["dd_direct"]["linestyle"],
            linewidth=styles["dd_direct"]["linewidth"],
            zorder=14,
        )
        axis.plot(
            dd_spline_t_plot[mask_spline] * 1e3,
            dd_spline_d_plot[mask_spline, sensor_node],
            color=styles["dd_spline"]["color"],
            linestyle=styles["dd_spline"]["linestyle"],
            linewidth=styles["dd_spline"]["linewidth"],
            zorder=14,
        )
        axis.plot(
            base025_t_plot[mask_base025] * 1e3,
            base025_d_plot[mask_base025, sensor_node],
            color=styles["base_025"]["color"],
            linestyle=styles["base_025"]["linestyle"],
            linewidth=styles["base_025"]["linewidth"],
            zorder=16,
        )
        axis.plot(
            base019_t_plot[mask_base019] * 1e3,
            base019_d_plot[mask_base019, sensor_node],
            color=styles["base_019"]["color"],
            linestyle=styles["base_019"]["linestyle"],
            linewidth=styles["base_019"]["linewidth"],
            zorder=16,
        )

        axis.set_ylim(global_y_lim)
        axis.set_xlabel("Time (ms)", fontsize=9)
        axis.set_ylabel("Displacement (m)", fontsize=9)
        axis.set_title(f"x = {sensor_positions[sensor_idx]:.2f}m", fontsize=10)
        axis.grid(True, alpha=0.3)
        axis.tick_params(labelsize=8)

    ax_rmse = fig.add_subplot(gs[4, :])

    rmse_direct_mask = exp_direct_t <= max_plot_time
    rmse_spline_mask = exp_spline_t <= max_plot_time

    ax_rmse.plot(
        exp_direct_t[rmse_direct_mask] * 1e3,
        dd_direct_rmse[rmse_direct_mask],
        color=styles["dd_direct"]["color"],
        linestyle=styles["dd_direct"]["linestyle"],
        linewidth=1.4,
        marker="o",
        markersize=2.2,
    )
    ax_rmse.plot(
        exp_spline_t[rmse_spline_mask] * 1e3,
        dd_spline_rmse[rmse_spline_mask],
        color=styles["dd_spline"]["color"],
        linestyle=styles["dd_spline"]["linestyle"],
        linewidth=1.4,
        marker="o",
        markersize=2.2,
    )
    ax_rmse.plot(
        exp_direct_t[rmse_direct_mask] * 1e3,
        base025_rmse[rmse_direct_mask],
        color=styles["base_025"]["color"],
        linestyle=styles["base_025"]["linestyle"],
        linewidth=1.6,
        marker="o",
        markersize=2.2,
    )
    ax_rmse.plot(
        exp_direct_t[rmse_direct_mask] * 1e3,
        base019_rmse[rmse_direct_mask],
        color=styles["base_019"]["color"],
        linestyle=styles["base_019"]["linestyle"],
        linewidth=1.6,
        marker="o",
        markersize=2.2,
    )

    ax_rmse.set_xlabel("Time (ms)", fontsize=10)
    ax_rmse.set_ylabel("RMSE (m)", fontsize=10)
    ax_rmse.set_title(
        "Global RMSE (baselines averaged over direct+spline data)",
        fontsize=11,
        fontweight="bold",
    )
    ax_rmse.grid(True, alpha=0.3)

    legend_handles = [
        Line2D([0], [0], marker="x", color="#111111", linestyle="none", markersize=5, label="Exp direct"),
        Line2D([0], [0], marker="+", color="#C92A2A", linestyle="none", markersize=6, label="Exp spline"),
        Line2D([0], [0], color=styles["dd_direct"]["color"], linestyle=styles["dd_direct"]["linestyle"], linewidth=1.6, label=styles["dd_direct"]["label"]),
        Line2D([0], [0], color=styles["dd_spline"]["color"], linestyle=styles["dd_spline"]["linestyle"], linewidth=1.6, label=styles["dd_spline"]["label"]),
        Line2D([0], [0], color=styles["base_025"]["color"], linestyle=styles["base_025"]["linestyle"], linewidth=1.8, label=styles["base_025"]["label"]),
        Line2D([0], [0], color=styles["base_019"]["color"], linestyle=styles["base_019"]["linestyle"], linewidth=1.8, label=styles["base_019"]["label"]),
    ]
    ax_rmse.legend(handles=legend_handles, fontsize=9, ncol=3, loc="upper right")

    fig.suptitle(
        "Combined Overlay: DD Direct + DD Spline + Baselines nu=0.25/0.19 + Experimental Data",
        fontsize=14,
        fontweight="bold",
    )

    output_path = args.plot_dir / args.output_name
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)

    print(f"[SUCCESS] Combined overlay saved: {output_path}")


if __name__ == "__main__":
    main()
