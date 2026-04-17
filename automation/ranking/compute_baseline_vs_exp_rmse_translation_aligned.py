#!/usr/bin/env python3
"""
compute_baseline_vs_exp_rmse_translation_aligned.py

Computes baseline-vs-experimental relative error after searching for a single
global spatial translation that minimizes the error between the simulation and
experimental displacement profiles.

This is a translation-aware companion to compute_baseline_vs_exp_rmse.py.

Reports:
  - Best global spatial shift (m)
  - Mean relative error at the best shift
  - Final timestep relative error at the best shift
"""

from __future__ import annotations

import argparse
import glob
import re
from pathlib import Path

import numpy as np


_SIM_DIR = Path("/Users/joshmcneely/uguca/build/simulations")
_DEFAULT_BASELINE_DIR = _SIM_DIR / "results_sweep"
_DEFAULT_EXP_BASE_DIR = Path("/Users/joshmcneely/introsims/simulation_outputs")
_SENSOR_POSITIONS = np.arange(0.05, 3.05 + 0.01, 0.2)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Compute baseline-vs-experimental error with translation alignment.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--baseline-dir", type=Path, default=_DEFAULT_BASELINE_DIR)
    p.add_argument("--baseline-sim-name", type=str, default="local_baseline_run")
    p.add_argument("--exp-base-dir", type=Path, default=_DEFAULT_EXP_BASE_DIR)
    p.add_argument(
        "--exp-restart-dir",
        type=str,
        default="interface-restart",
        help="Subdirectory inside --exp-base-dir with restart snapshots",
    )
    p.add_argument("--nb-nodes", type=int, default=512)
    p.add_argument("--domain-length", type=float, default=6.0)
    p.add_argument("--shift-min", type=float, default=-0.5)
    p.add_argument("--shift-max", type=float, default=0.5)
    p.add_argument("--shift-step", type=float, default=0.01)
    return p.parse_args()


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


def _build_shift_grid(shift_min: float, shift_max: float, shift_step: float) -> np.ndarray:
    if shift_step <= 0.0:
        raise ValueError("--shift-step must be > 0")
    if shift_max < shift_min:
        raise ValueError("--shift-max must be >= --shift-min")
    return np.arange(shift_min, shift_max + shift_step * 0.5, shift_step, dtype=float)


def _error_for_shift(
    baseline_rows: np.ndarray,
    exp_rows: np.ndarray,
    x_grid: np.ndarray,
    sensor_positions: np.ndarray,
    shift: float,
) -> tuple[float, float, float, int]:
    query_positions = sensor_positions - shift
    valid_mask = (query_positions >= x_grid[0]) & (query_positions <= x_grid[-1])
    valid_fraction = float(np.count_nonzero(valid_mask)) / float(len(sensor_positions))
    if not np.any(valid_mask):
        return np.nan, np.nan, valid_fraction, 0

    query_positions = query_positions[valid_mask]
    exp_valid = exp_rows[:, valid_mask]

    rel_errors = np.full(exp_valid.shape[0], np.nan, dtype=float)
    for i, row in enumerate(baseline_rows):
        sim_valid = np.interp(query_positions, x_grid, row)
        denom = np.linalg.norm(exp_valid[i])
        if denom > 0.0:
            rel_errors[i] = np.linalg.norm(sim_valid - exp_valid[i]) / denom

    valid_steps = int(np.sum(np.isfinite(rel_errors)))
    if valid_steps == 0:
        return np.nan, np.nan, valid_fraction, 0

    return float(np.nanmean(rel_errors)), float(rel_errors[-1]), valid_fraction, valid_steps


def main() -> None:
    args = parse_args()

    dx = args.domain_length / args.nb_nodes
    sensor_nodes = np.round(_SENSOR_POSITIONS / dx).astype(int)
    sensor_nodes = sensor_nodes[sensor_nodes < args.nb_nodes]
    sensor_positions = sensor_nodes.astype(float) * dx
    n_sensors = len(sensor_nodes)
    x_grid = np.arange(args.nb_nodes, dtype=float) * dx

    shift_grid = _build_shift_grid(args.shift_min, args.shift_max, args.shift_step)

    base_t, base_d = load_sim(args.baseline_dir, args.baseline_sim_name, args.nb_nodes)
    if base_t is None:
        print(f"[ERROR] Could not load baseline from: {args.baseline_dir}")
        print(f"        Expected: {args.baseline_dir}/{args.baseline_sim_name}-interface-DataFiles/top_disp.out")
        return
    print(f"Baseline: {len(base_t)} timesteps, t=[{base_t[0]:.4e}, {base_t[-1]:.4e}] s")

    restart_dir = args.exp_base_dir / args.exp_restart_dir
    exp_times, exp_data = load_exp(restart_dir, sensor_nodes, args.nb_nodes)
    print(f"Experimental: {len(exp_times)} snapshots, {n_sensors} sensors")

    base_t_max = base_t[-1]
    common_mask = exp_times <= base_t_max
    common_exp_times = exp_times[common_mask]
    common_exp_data = exp_data[common_mask]
    print(
        f"Common time range: {len(common_exp_times)}/{len(exp_times)} exp snapshots "
        f"(t <= {base_t_max:.4e} s)"
    )

    if len(common_exp_times) == 0:
        print("[ERROR] No experimental snapshots fall within the baseline time range.")
        return

    idx = np.searchsorted(base_t, common_exp_times).clip(0, len(base_t) - 1)
    base_at_exp = base_d[idx]

    best_shift = None
    best_mean_rel = np.inf
    best_final_rel = np.inf
    best_overlap = 0.0
    best_valid_steps = 0

    for shift in shift_grid:
        mean_rel, final_rel, overlap_fraction, valid_steps = _error_for_shift(
            base_at_exp,
            common_exp_data,
            x_grid,
            sensor_positions,
            shift,
        )
        if not np.isfinite(mean_rel):
            continue
        if mean_rel < best_mean_rel or (np.isclose(mean_rel, best_mean_rel) and final_rel < best_final_rel):
            best_shift = float(shift)
            best_mean_rel = float(mean_rel)
            best_final_rel = float(final_rel)
            best_overlap = float(overlap_fraction)
            best_valid_steps = int(valid_steps)

    if best_shift is None:
        print("[ERROR] No valid shifts produced a finite error.")
        return

    print()
    print("=" * 72)
    print("  Baseline vs Experimental  -  Translation-aligned relative L2 error")
    print("  r_shift(t) = ||u_sim(x - shift, t) - u_exp(x, t)||_2 / ||u_exp(x, t)||_2")
    print(f"  Baseline dir : {args.baseline_dir}")
    print(f"  Sim name     : {args.baseline_sim_name}")
    print(f"  Search range  : [{args.shift_min:+.3f}, {args.shift_max:+.3f}] m step {args.shift_step:.3f} m")
    print(f"  Best shift found (m) : {best_shift:+.6f}")
    print(f"  Overlap frac  : {best_overlap:.5f}")
    print(f"  Timesteps     : {len(common_exp_times)}  ({best_valid_steps} valid, i.e. non-zero exp norm)")
    print(f"  Sensors       : {n_sensors}")
    print("=" * 72)
    print(f"  Average relative error (valid timesteps) : {best_mean_rel:.10e}  ({best_mean_rel * 100:.5f}%)")
    print(f"  Final timestep relative error           : {best_final_rel:.10e}  ({best_final_rel * 100:.5f}%)")
    print("=" * 72)


if __name__ == "__main__":
    main()