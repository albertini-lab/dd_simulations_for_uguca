#!/usr/bin/env python3
"""
compute_baseline_vs_exp_rmse.py

Computes RMSE between the local_baseline_run simulation and experimental data,
using the same sensor positions, loading routines, and time-alignment strategy
as Plot_W_Factor_RMSE.py.

Reports:
  - Mean RMSE (across all sensors and timesteps)
  - Per-sensor RMSE
  - Per-timestep mean RMSE (min / median / max summary)

Usage:
  python compute_baseline_vs_exp_rmse.py
  python compute_baseline_vs_exp_rmse.py \
    --baseline-dir /path/to/results_baseline \
    --baseline-sim-name local_baseline_run \
    --exp-base-dir /path/to/experimental_data_processed \
    --nb-nodes 2048 \
    --domain-length 6.0
"""

from __future__ import annotations

import argparse
import glob
import re
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
_SIM_DIR = Path("/Users/joshmcneely/uguca/build/simulations")
_DEFAULT_BASELINE_DIR = _SIM_DIR / "results_sweep"
_DEFAULT_EXP_BASE_DIR = Path("/Users/joshmcneely/introsims/simulation_outputs")
_SENSOR_POSITIONS = np.arange(0.05, 3.05 + 0.01, 0.2)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Compute baseline-vs-experimental RMSE.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--baseline-dir", type=Path, default=_DEFAULT_BASELINE_DIR)
    p.add_argument("--baseline-sim-name", type=str, default="local_baseline_run")
    p.add_argument("--exp-base-dir", type=Path, default=_DEFAULT_EXP_BASE_DIR)
    p.add_argument("--exp-restart-dir", type=str, default="interface-restart",
                   help="Subdirectory inside --exp-base-dir with restart snapshots")
    p.add_argument("--nb-nodes", type=int, default=512)
    p.add_argument("--domain-length", type=float, default=6.0)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Data loaders  (identical logic to Plot_W_Factor_RMSE.py)
# ---------------------------------------------------------------------------
def load_sim(base_path: Path, sim_name: str, nb_nodes: int):
    data_dir = base_path / f"{sim_name}-interface-DataFiles"
    path = data_dir / "top_disp.out"
    t_path = base_path / f"{sim_name}-interface.time"
    if not path.exists() or not t_path.exists():
        return None, None
    with open(t_path, "r") as f:
        times = np.array([float(line.split()[1]) for line in f.readlines()])
    raw_data = np.loadtxt(path)
    if raw_data.ndim == 1:
        raw_data = raw_data.reshape(1, -1)
    data = raw_data[:, :nb_nodes]
    n = min(len(times), data.shape[0])
    return times[:n], data[:n, :]


def load_exp(restart_dir: Path, sensor_nodes: np.ndarray, nb_nodes: int):
    prefix = restart_dir.name.split('-')[0]
    info_file = restart_dir.parent / f"{prefix}.info"
    with open(info_file, "r") as f:
        dt_sim = float(re.search(r"time_step\s*=\s*([\d.eE+-]+)", f.read()).group(1))
    files = sorted(
        glob.glob(str(restart_dir / "top_disp.proc0.s*.out")),
        key=lambda x: int(re.search(r"\.s(\d+)\.out$", x).group(1)),
    )
    data, times = [], []
    for fp in files:
        step = int(re.search(r"\.s(\d+)\.out$", fp).group(1))
        times.append(step * dt_sim)
        raw_bin = np.fromfile(fp, dtype=np.float32)
        u_x = raw_bin[:nb_nodes]
        data.append(u_x[sensor_nodes])
    return np.array(times), np.array(data)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    args = parse_args()

    dx = args.domain_length / args.nb_nodes
    sensor_nodes = np.round(_SENSOR_POSITIONS / dx).astype(int)
    sensor_nodes = sensor_nodes[sensor_nodes < args.nb_nodes]
    n_sensors = len(sensor_nodes)

    # Load baseline
    base_t, base_d = load_sim(args.baseline_dir, args.baseline_sim_name, args.nb_nodes)
    if base_t is None:
        print(f"[ERROR] Could not load baseline from: {args.baseline_dir}")
        print(f"        Expected: {args.baseline_dir}/{args.baseline_sim_name}-interface-DataFiles/top_disp.out")
        return
    print(f"Baseline: {len(base_t)} timesteps, t=[{base_t[0]:.4e}, {base_t[-1]:.4e}] s")

    # Load experimental data
    restart_dir = args.exp_base_dir / args.exp_restart_dir
    exp_times, exp_data = load_exp(restart_dir, sensor_nodes, args.nb_nodes)
    print(f"Experimental: {len(exp_times)} snapshots, {n_sensors} sensors")

    # Restrict to time range covered by the baseline
    base_t_max = base_t[-1]
    common_mask = exp_times <= base_t_max
    common_exp_times = exp_times[common_mask]
    common_exp_data = exp_data[common_mask]          # shape: (T_common, n_sensors)
    print(f"Common time range: {len(common_exp_times)}/{len(exp_times)} exp snapshots "
          f"(t <= {base_t_max:.4e} s)")

    if len(common_exp_times) == 0:
        print("[ERROR] No experimental snapshots fall within the baseline time range.")
        return

    # Align baseline to experimental time points (nearest-neighbor)
    idx = np.searchsorted(base_t, common_exp_times).clip(0, len(base_t) - 1)
    base_at_exp = base_d[idx][:, sensor_nodes]       # shape: (T_common, n_sensors)

    # Relative L2 error per timestep: ||u_base(t) - u_data(t)||_2 / ||u_data(t)||_2
    residuals = base_at_exp - common_exp_data         # (T_common, n_sensors)

    numerator   = np.linalg.norm(residuals, axis=1)          # (T_common,)
    denominator = np.linalg.norm(common_exp_data, axis=1)    # (T_common,)

    rel = np.full_like(numerator, np.nan)
    valid = denominator > 0.0
    rel[valid] = numerator[valid] / denominator[valid]

    n_valid      = int(np.sum(valid))
    mean_rel     = float(np.nanmean(rel))
    final_rel    = float(rel[-1])

    # ---------------------------------------------------------------------------
    # Report
    # ---------------------------------------------------------------------------
    print()
    print("=" * 60)
    print("  Baseline vs Experimental  —  Relative L2 error")
    print("  r(t) = ||u_baseline(t) - u_exp(t)||_2 / ||u_exp(t)||_2")
    print(f"  Baseline dir : {args.baseline_dir}")
    print(f"  Sim name     : {args.baseline_sim_name}")
    print(f"  Timesteps    : {len(common_exp_times)}  ({n_valid} valid, i.e. non-zero exp norm)")
    print(f"  Sensors      : {n_sensors}")
    print("=" * 60)
    print(f"  Average relative error (valid timesteps) : {mean_rel:.10e}  ({mean_rel*100:.5f}%)")
    print(f"  Final  timestep relative error           : {final_rel:.10e}  ({final_rel*100:.5f}%)")
    print("=" * 60)


if __name__ == "__main__":
    main()
