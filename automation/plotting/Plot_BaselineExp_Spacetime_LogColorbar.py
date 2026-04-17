#!/usr/bin/env python3
"""
Plot_BaselineExp_Spacetime_LogColorbar.py

Generate a two-panel spacetime comparison (same behavior as Plot_BaselineExp_Spacetime.py)
but with a logarithmic colorbar scaling using symmetric log normalization.
"""

import argparse
import glob
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import SymLogNorm
import numpy as np


SIM_DIR = Path("/Users/joshmcneely/uguca/build/simulations")
EXP_BASE_DIR = Path("/Users/joshmcneely/introsims/simulation_outputs")
PLOT_DIR = SIM_DIR / "comparison_plots"

DOMAIN_LENGTH = 6.0
NB_NODES = 512
BASE_SIM_NAME = "local_baseline_run"

SENSOR_POSITIONS = np.arange(0.05, 3.05 + 0.01, 0.2)
SENSOR_POSITIONS = np.round(SENSOR_POSITIONS, 12)
SENSOR_POSITIONS = SENSOR_POSITIONS[SENSOR_POSITIONS <= 3.05]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate baseline-vs-experimental spacetime plot (log color scale).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--baseline-dir", type=Path, required=True, help="Baseline case directory")
    parser.add_argument("--case-label", type=str, default=None, help="Case label used in title")
    parser.add_argument("--output-tag", type=str, default=None, help="Output filename tag")
    parser.add_argument("--plot-dir", type=Path, default=PLOT_DIR, help="Directory to write output plot")
    parser.add_argument("--exp-restart-dir", type=Path, default=None, help="Direct path to experimental restart directory")
    parser.add_argument("--exp-base-dir", type=Path, default=EXP_BASE_DIR, help="Experimental data base directory")
    parser.add_argument("--baseline-sim-name", type=str, default=BASE_SIM_NAME, help="Baseline simulation run prefix")
    parser.add_argument("--nb-nodes", type=int, default=NB_NODES, help="Number of interface nodes")
    parser.add_argument("--domain-length", type=float, default=DOMAIN_LENGTH, help="Domain length in meters")
    parser.add_argument("--roi-min", type=float, default=0.0, help="Minimum x-position for spacetime ROI")
    parser.add_argument("--roi-max", type=float, default=3.05, help="Maximum x-position for spacetime ROI")
    parser.add_argument("--best-shift-m", type=float, default=None, help="Deprecated: accepted for backward compatibility, ignored")
    parser.add_argument("--write-translated-plot", action="store_true", help="Deprecated: accepted for backward compatibility, ignored")
    args = parser.parse_args()

    if args.output_tag is None:
        args.output_tag = args.baseline_dir.name
    if args.case_label is None:
        args.case_label = args.output_tag
    if args.exp_restart_dir is None:
        args.exp_restart_dir = args.exp_base_dir / "interface-restart"

    return args


def load_sim(base_path: Path, sim_name: str, nb_nodes: int):
    data_dir = base_path / f"{sim_name}-interface-DataFiles"
    disp_path = data_dir / "top_disp.out"
    time_path = base_path / f"{sim_name}-interface.time"

    if not disp_path.exists() or not time_path.exists():
        print(f"[ERROR] Missing baseline output files in {base_path}")
        return None, None

    with open(time_path, "r") as file_obj:
        times = np.array([float(line.split()[1]) for line in file_obj.readlines() if line.strip()], dtype=float)

    data = np.loadtxt(disp_path, dtype=float)
    if data.ndim == 1:
        data = data.reshape(1, -1)

    data = data[:, :nb_nodes]
    n = min(len(times), data.shape[0])
    return times[:n], data[:n, :]


def load_exp(restart_dir: Path, sensor_nodes: np.ndarray, nb_nodes: int):
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


def extract_full_spatial_data(data, roi_min: float, roi_max: float, dx: float):
    if data.ndim == 1:
        data = data.reshape(1, -1)

    n_nodes = data.shape[1]
    positions = np.arange(n_nodes) * dx
    roi_mask = (positions >= roi_min) & (positions <= roi_max)
    roi_indices = np.where(roi_mask)[0]
    roi_positions = positions[roi_indices]

    roi_data = data[:, roi_indices]
    slip_relative = np.zeros_like(roi_data.T)
    for i in range(len(roi_indices)):
        slip_relative[i, :] = (roi_data[:, i] - roi_data[0, i]) * 2.0 * 1e6

    return slip_relative, roi_positions


def _build_symlog_norm(base_slip: np.ndarray, exp_slip: np.ndarray) -> SymLogNorm:
    all_values = np.concatenate([base_slip.ravel(), exp_slip.ravel()])
    abs_values = np.abs(all_values)
    vmax = float(np.nanmax(abs_values)) if abs_values.size else 1.0
    if not np.isfinite(vmax) or vmax <= 0.0:
        vmax = 1.0

    nonzero = abs_values[abs_values > 0.0]
    if nonzero.size == 0:
        linthresh = 1e-6
    else:
        min_nonzero = float(np.nanmin(nonzero))
        linthresh = max(min_nonzero, vmax * 1e-3)

    return SymLogNorm(linthresh=linthresh, linscale=1.0, vmin=-vmax, vmax=vmax, base=10)


def save_plot(base_slip, base_positions, base_times, exp_slip, exp_times, sensor_positions, case_label, output_path):
    norm = _build_symlog_norm(base_slip, exp_slip)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)
    fig.suptitle(f"Baseline vs Experimental (Log Colorbar): {case_label}", fontsize=14, fontweight="bold")

    im1 = axes[0].imshow(
        base_slip,
        aspect="auto",
        origin="lower",
        cmap="coolwarm",
        norm=norm,
        interpolation="nearest",
        extent=[base_times.min(), base_times.max(), base_positions.min(), base_positions.max()],
    )
    axes[0].set_xlabel("Time (s)")
    axes[0].set_ylabel("Position (m)")
    axes[0].set_title("Baseline Simulation (Full Resolution)")
    axes[0].grid(True, alpha=0.3, linestyle="--")

    im2 = axes[1].imshow(
        exp_slip,
        aspect="auto",
        origin="lower",
        cmap="coolwarm",
        norm=norm,
        interpolation="nearest",
        extent=[exp_times.min(), exp_times.max(), sensor_positions.min(), sensor_positions.max()],
    )
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Position (m)")
    axes[1].set_title("Experimental Data (16 Sensors)")
    axes[1].grid(True, alpha=0.3, linestyle="--")

    cbar1 = fig.colorbar(im1, ax=axes[0], pad=0.02)
    cbar1.set_label(r"Slip ($\mu$m), symmetric log scale")
    cbar2 = fig.colorbar(im2, ax=axes[1], pad=0.02)
    cbar2.set_label(r"Slip ($\mu$m), symmetric log scale")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[SUCCESS] Saved log-scaled spacetime plot to: {output_path}")


def main():
    args = parse_args()

    dx = args.domain_length / args.nb_nodes
    sensor_nodes = np.round(SENSOR_POSITIONS / dx).astype(int)
    sensor_nodes = sensor_nodes[sensor_nodes < args.nb_nodes]
    sensor_positions = sensor_nodes * dx

    print(f"[INFO] baseline_dir={args.baseline_dir}")
    print(f"[INFO] exp_restart_dir={args.exp_restart_dir}")

    base_times, base_data = load_sim(args.baseline_dir, args.baseline_sim_name, args.nb_nodes)
    if base_times is None or base_data is None:
        raise SystemExit(1)

    exp_times, exp_data = load_exp(args.exp_restart_dir, sensor_nodes, args.nb_nodes)
    if exp_data is None or len(exp_data) == 0:
        print("[ERROR] Failed to load experimental data")
        raise SystemExit(1)

    base_slip, base_positions = extract_full_spatial_data(base_data, args.roi_min, args.roi_max, dx)
    if args.write_translated_plot or args.best_shift_m is not None:
        print("[INFO] Translated plotting is currently disabled; ignoring translation options.")

    exp_slip = np.zeros_like(exp_data)
    for i in range(exp_data.shape[1]):
        exp_slip[:, i] = (exp_data[:, i] - exp_data[0, i]) * 2.0 * 1e6
    exp_slip = exp_slip.T

    save_plot(base_slip, base_positions, base_times, exp_slip, exp_times, sensor_positions, args.case_label, args.plot_dir / f"spacetime_{args.output_tag}.png")
    # Translated plot generation is intentionally disabled for now.


if __name__ == "__main__":
    main()
