#!/usr/bin/env python3
"""
Plot_W_Factor_RMSE.py
Plots the average RMSE vs w-factor across all archived sweep results.
One point per simulation — x-axis is w-factor (log scale), y-axis is mean RMSE.
"""
import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
import glob, re, os
from pathlib import Path

# --- CONFIGURATION ---
SIM_DIR = Path("/Users/joshmcneely/uguca/build/simulations")
EXP_BASE_DIR = Path("/Users/joshmcneely/introsims/simulation_outputs")
BASELINE_DIR = SIM_DIR / "results_baseline"
PLOT_DIR = SIM_DIR / "comparison_plots"

DOMAIN_LENGTH = 6.0
NB_NODES = 2048
DD_SIM_NAME = "local_debug_run"
BASE_SIM_NAME = "local_baseline_run"
SENSOR_POSITIONS = np.arange(0.05, 3.05 + 0.01, 0.2)

# --- HELPERS ---
def load_sim(base_path, sim_name, nb_nodes):
    data_dir = base_path / f"{sim_name}-interface-DataFiles"
    path = data_dir / "top_disp.out"
    t_path = base_path / f"{sim_name}-interface.time"
    if not path.exists() or not t_path.exists():
        return None, None
    with open(t_path, 'r') as f:
        times = np.array([float(line.split()[1]) for line in f.readlines()])
    raw_data = np.loadtxt(path)
    if raw_data.ndim == 1:
        raw_data = raw_data.reshape(1, -1)
    data = raw_data[:, :nb_nodes]
    n = min(len(times), data.shape[0])
    return times[:n], data[:n, :]

def load_exp(restart_dir, sensor_nodes, nb_nodes):
    info_file = restart_dir.parent / "mcklaskey_debug.info"
    with open(info_file, 'r') as f:
        dt_sim = float(re.search(r'time_step\s*=\s*([\d.eE+-]+)', f.read()).group(1))
    files = sorted(
        glob.glob(str(restart_dir / "top_disp.proc0.s*.out")),
        key=lambda x: int(re.search(r'\.s(\d+)\.out$', x).group(1))
    )
    data, times = [], []
    for fp in files:
        step = int(re.search(r'\.s(\d+)\.out$', fp).group(1))
        times.append(step * dt_sim)
        raw_bin = np.fromfile(fp, dtype=np.float32)
        u_x = raw_bin[:nb_nodes]
        data.append(u_x[sensor_nodes])
    return np.array(times), np.array(data)

# --- MAIN ---
def main():
    dx = DOMAIN_LENGTH / NB_NODES
    sensor_nodes = np.round(SENSOR_POSITIONS / dx).astype(int)
    sensor_nodes = sensor_nodes[sensor_nodes < NB_NODES]

    # Load experimental data once
    restart_dir = EXP_BASE_DIR / "mcklaskey_debug-restart"
    exp_times, exp_data = load_exp(restart_dir, sensor_nodes, NB_NODES)
    print(f"Loaded {len(exp_times)} experimental snapshots, {len(sensor_nodes)} sensors")

    # Load baseline (may be partial — use only available timesteps)
    base_t, base_d = load_sim(BASELINE_DIR, BASE_SIM_NAME, NB_NODES)
    has_baseline = base_t is not None
    if has_baseline:
        base_t_max = base_t[-1]
        print(f"Baseline available up to t={base_t_max:.6f}s ({len(base_t)} dumps)")
        # Restrict experimental times to what the baseline covers
        common_mask = exp_times <= base_t_max
        common_exp_times = exp_times[common_mask]
        common_exp_data = exp_data[common_mask]
        print(f"Using {len(common_exp_times)}/{len(exp_times)} experimental snapshots within baseline range")
    else:
        print("[WARNING] Baseline not found — plotting DD only, using all exp times")
        common_exp_times = exp_times
        common_exp_data = exp_data

    # Find all results_w* directories
    result_dirs = sorted(glob.glob(str(SIM_DIR / "results_w*")))
    if not result_dirs:
        print("No results_w* directories found!")
        return

    w_factors = []
    mean_rmses = []

    for rdir in result_dirs:
        rdir = Path(rdir)
        w_str = rdir.name.replace("results_w", "")
        try:
            w_val = float(w_str)
        except ValueError:
            print(f"  Skipping {rdir.name} (cannot parse w-factor)")
            continue

        sim_t, sim_d = load_sim(rdir, DD_SIM_NAME, NB_NODES)
        if sim_t is None:
            print(f"  Skipping w={w_str} (missing data)")
            continue

        # Use the common time range for fair comparison
        idx = np.searchsorted(sim_t, common_exp_times).clip(0, len(sim_t) - 1)
        sim_at_exp = sim_d[idx][:, sensor_nodes]

        rmse_per_time = np.sqrt(np.mean((sim_at_exp - common_exp_data) ** 2, axis=1))
        mean_rmse = np.mean(rmse_per_time)

        w_factors.append(w_val)
        mean_rmses.append(mean_rmse)
        print(f"  w={w_str:>8s}  ->  mean RMSE = {mean_rmse:.4e}")

    if not w_factors:
        print("No valid results to plot.")
        return

    # Compute baseline RMSE over the same common time range
    base_mean_rmse = None
    if has_baseline:
        idx_base = np.searchsorted(base_t, common_exp_times).clip(0, len(base_t) - 1)
        base_at_exp = base_d[idx_base][:, sensor_nodes]
        base_rmse_per_time = np.sqrt(np.mean((base_at_exp - common_exp_data) ** 2, axis=1))
        base_mean_rmse = np.mean(base_rmse_per_time)
        print(f"  Baseline  ->  mean RMSE = {base_mean_rmse:.4e}")

    # Sort by w-factor
    order = np.argsort(w_factors)
    w_factors = np.array(w_factors)[order]
    mean_rmses = np.array(mean_rmses)[order]

    # --- PLOT ---
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(w_factors, mean_rmses, 'ro-', markersize=8, linewidth=2, label='DD Simulation')

    if base_mean_rmse is not None:
        ax.axhline(y=base_mean_rmse, color='k', linestyle='--', linewidth=2, alpha=0.7,
                    label=f'Baseline (t≤{base_t_max*1e3:.1f}ms)')

    ax.set_xscale('log')
    ax.set_xlabel('w-factor', fontsize=14)
    ax.set_ylabel('Mean RMSE (m)', fontsize=14)
    ax.set_title('Average RMSE vs W-Factor', fontsize=16)
    ax.grid(True, alpha=0.3, which='both')
    ax.tick_params(labelsize=12)

    ax.legend(fontsize=12)
    os.makedirs(PLOT_DIR, exist_ok=True)
    out_path = PLOT_DIR / "rmse_vs_w_factor.png"
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    print(f"\n[SUCCESS] Plot saved to: {out_path}")

if __name__ == "__main__":
    main()
