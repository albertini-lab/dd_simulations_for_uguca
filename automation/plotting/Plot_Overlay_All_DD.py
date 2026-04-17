#!/usr/bin/env python3

import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import matplotlib.patches as mpatches
import matplotlib
import glob
import re
from matplotlib.lines import Line2D

matplotlib.use('Agg')

# --- PATH CONFIGURATION ---
SWEEP_RESULTS_DIR = Path(".")
BASELINE_RESULTS_DIR = Path("/Users/joshmcneely/uguca/build/simulations/results_baseline")
EXP_BASE_DIR = Path("/Users/joshmcneely/introsims/simulation_outputs")
PLOT_DIR = Path("./comparison_plots")

# --- GEOMETRY & SETTINGS ---
DOMAIN_LENGTH = 6.0
NB_NODES = 2048
SENSOR_POSITIONS = np.arange(0.05, 3.05 + 0.01, 0.2)
SIM_DENSITY_MULTIPLIER = 200

# --- SIMULATION NAMES ---
DD_SIM_NAME = "local_debug_run"
BASE_SIM_NAME = "local_baseline_run"
EXP_SIM_NAME = "mcklaskey_debug"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create overlay plots in the Plot_Overlay_All_DD style.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--sweep-results-dir", type=Path, default=SWEEP_RESULTS_DIR, help="Directory containing DD sweep case folders")
    parser.add_argument("--sweep-glob", type=str, default="results_w*", help="Glob used to discover DD sweep folders")
    parser.add_argument("--dd-dir", type=Path, default=None, help="Explicit DD case directory or DD -DataFiles directory")
    parser.add_argument("--case-label", type=str, default=None, help="Custom label when --dd-dir is used")

    parser.add_argument("--baseline-results-dir", type=Path, default=BASELINE_RESULTS_DIR, help="Baseline case directory")
    parser.add_argument("--baseline-dir", type=Path, default=None, help="Explicit baseline case directory or baseline -DataFiles directory")

    parser.add_argument("--exp-base-dir", type=Path, default=EXP_BASE_DIR, help="Experimental data base directory")
    parser.add_argument("--exp-name", type=str, default=EXP_SIM_NAME, help="Experimental dataset base name (without -restart/.info)")
    parser.add_argument("--exp-restart-dir", type=Path, default=None, help="Direct path to experimental restart directory")

    parser.add_argument("--dd-sim-name", type=str, default=DD_SIM_NAME, help="DD simulation run prefix")
    parser.add_argument("--baseline-sim-name", type=str, default=BASE_SIM_NAME, help="Baseline simulation run prefix")
    parser.add_argument("--plot-dir", type=Path, default=PLOT_DIR, help="Output plot directory")
    parser.add_argument("--output-name", type=str, default="overlay_all_dd.png", help="Output filename")

    parser.add_argument("--domain-length", type=float, default=DOMAIN_LENGTH, help="Domain length")
    parser.add_argument("--nb-nodes", type=int, default=NB_NODES, help="Number of interface nodes")
    parser.add_argument(
        "--sim-density-multiplier",
        type=int,
        default=SIM_DENSITY_MULTIPLIER,
        help="Simulation plotting density multiplier relative to experimental sampling",
    )

    return parser.parse_args()


def resolve_sim_io_paths(base_path, sim_name):
    base_path = Path(base_path)
    datafiles_suffix = "-interface-DataFiles"

    if base_path.name.endswith(datafiles_suffix):
        data_dir = base_path
        prefix = base_path.name[:-len(datafiles_suffix)]
        t_path = base_path.parent / f"{prefix}-interface.time"
        if not t_path.exists():
            t_path = base_path.parent / f"{sim_name}-interface.time"
        return data_dir, t_path

    data_dir = base_path / f"{sim_name}-interface-DataFiles"
    t_path = base_path / f"{sim_name}-interface.time"
    return data_dir, t_path

def load_sim_robust(base_path, sim_name, nb_nodes):
    """Load simulation data (using Plot_Local_Debug logic)."""
    try:
        data_dir, t_path = resolve_sim_io_paths(base_path, sim_name)
        path = data_dir / "top_disp.out"
        
        if not path.exists() or not t_path.exists():
            return None, None
        
        with open(t_path, 'r') as f:
            times = np.array([float(line.split()[1]) for line in f.readlines()])
        
        raw_data = np.loadtxt(path)
        if raw_data.ndim == 1: 
            raw_data = raw_data.reshape(1, -1)
        
        # Extract X-shear (first nb_nodes columns)
        if raw_data.shape[1] >= nb_nodes:
            data = raw_data[:, :nb_nodes]
        else:
            data = raw_data
        
        final_count = min(len(times), data.shape[0])
        return times[:final_count], data[:final_count, :]
        
    except Exception as e:
        print(f"[ERROR] Could not load {sim_name} from {base_path}: {e}")
        return None, None

def load_exp(restart_dir, sensor_nodes, nb_nodes, exp_name):
    """Load experimental data (using Plot_Local_Debug logic)."""
    info_file = restart_dir.parent / f"{exp_name}.info"
    with open(info_file, 'r') as f:
        dt_sim = float(re.search(r'time_step\s*=\s*([\d.eE+-]+)', f.read()).group(1))

    files = glob.glob(str(restart_dir / "top_disp.proc0.s*.out"))
    files.sort(key=lambda x: int(re.search(r'\.s(\d+)\.out$', x).group(1)))
    
    data, times = [], []
    for f in files:
        step = int(re.search(r'\.s(\d+)\.out$', f).group(1))
        times.append(step * dt_sim) 
        raw_bin = np.fromfile(f, dtype=np.float32)
        u_x = raw_bin[:nb_nodes]
        data.append(u_x[sensor_nodes])
        
    return np.array(times), np.array(data)

def main():
    args = parse_args()
    args.plot_dir.mkdir(exist_ok=True)
    
    # Compute sensor nodes
    dx = args.domain_length / args.nb_nodes
    sensor_nodes_raw = np.round(SENSOR_POSITIONS / dx).astype(int)
    valid_sensors = sensor_nodes_raw < args.nb_nodes
    sensor_nodes = sensor_nodes_raw[valid_sensors]
    actual_positions = sensor_nodes * dx
    
    print("[INFO] Loading baseline...")
    baseline_source = args.baseline_dir or args.baseline_results_dir
    base_t, base_d = load_sim_robust(baseline_source, args.baseline_sim_name, args.nb_nodes)
    
    print("[INFO] Loading experimental data...")
    restart_dir = args.exp_restart_dir or (args.exp_base_dir / f"{args.exp_name}-restart")
    exp_times, exp_data = load_exp(restart_dir, sensor_nodes, args.nb_nodes, args.exp_name)
    
    if exp_data is None:
        print("[CRITICAL ERROR] No experimental data.")
        return
    
    # Find all DD results and compute RMSE for each
    print("[INFO] Scanning DD results and computing RMSE...")
    dd_results = {}

    if args.dd_dir is not None:
        label = args.case_label or Path(args.dd_dir).name
        dd_t, dd_d = load_sim_robust(args.dd_dir, args.dd_sim_name, args.nb_nodes)
        if dd_d is None:
            print(f"[CRITICAL ERROR] Could not load DD run from: {args.dd_dir}")
            return

        idx_dd = np.searchsorted(dd_t, exp_times).clip(0, len(dd_t)-1)
        dd_sensors_exact = dd_d[idx_dd][:, sensor_nodes]
        rmse_per_time = np.sqrt(np.mean((dd_sensors_exact - exp_data)**2, axis=1))
        mean_rmse = np.mean(rmse_per_time)

        dd_results[label] = {
            "time": dd_t,
            "data": dd_d,
            "rmse": mean_rmse,
            "rmse_per_time": rmse_per_time
        }
        print(f"  case={label}: Mean RMSE={mean_rmse:.3e}")
    else:
        for result_dir in sorted(args.sweep_results_dir.glob(args.sweep_glob)):
            if result_dir.name.startswith("results_w"):
                case_key = result_dir.name.replace("results_w", "")
            else:
                case_key = result_dir.name

            dd_t, dd_d = load_sim_robust(result_dir, args.dd_sim_name, args.nb_nodes)

            if dd_d is None:
                continue

            # Match simulation times to experimental times (Plot_Local_Debug logic)
            idx_dd = np.searchsorted(dd_t, exp_times).clip(0, len(dd_t)-1)
            dd_sensors_exact = dd_d[idx_dd][:, sensor_nodes]

            # Compute RMSE per time snapshot (across sensors)
            rmse_per_time = np.sqrt(np.mean((dd_sensors_exact - exp_data)**2, axis=1))
            mean_rmse = np.mean(rmse_per_time)

            dd_results[case_key] = {
                "time": dd_t,
                "data": dd_d,
                "rmse": mean_rmse,
                "rmse_per_time": rmse_per_time
            }
            print(f"  case={case_key}: Mean RMSE={mean_rmse:.3e}")
    
    if not dd_results:
        print("[CRITICAL ERROR] No DD results found.")
        return
    
    # Compute baseline RMSE (Plot_Local_Debug logic)
    base_rmse_per_time = None
    if base_d is not None:
        idx_base = np.searchsorted(base_t, exp_times).clip(0, len(base_t)-1)
        base_sensors_exact = base_d[idx_base][:, sensor_nodes]
        base_rmse_per_time = np.sqrt(np.mean((base_sensors_exact - exp_data)**2, axis=1))
        print(f"  Baseline: Mean RMSE={np.mean(base_rmse_per_time):.3e}")
    
    # Sort by RMSE (best first) and assign ranks
    sorted_results = sorted(dd_results.items(), key=lambda x: x[1]["rmse"])
    n_runs = len(sorted_results)
    
    print(f"\n[INFO] Ranking {n_runs} runs by RMSE:")
    for rank, (w_str, result) in enumerate(sorted_results, start=1):
        # VERY DRAMATIC: Best (rank 1) → alpha = 1.0, Worst (rank N) → alpha = 0.05
        alpha = 1.0 - 0.5 * (rank - 1) / max(1, n_runs - 1)
        result["alpha"] = alpha
        result["rank"] = rank
        print(f"  Rank {rank}: w={w_str}, RMSE={result['rmse']:.3e}, alpha={alpha:.2f}")
    
    # Decimate simulation data for plotting (Plot_Local_Debug logic)
    target_plot_points = len(exp_times) * args.sim_density_multiplier
    
    # Decimate baseline
    base_t_plot, base_d_plot = None, None
    if base_d is not None:
        stride_base = max(1, len(base_t) // target_plot_points)
        base_t_plot = base_t[::stride_base]
        base_d_plot = base_d[::stride_base]
    
    # Decimate all DD results
    dd_results_decimated = {}
    for w_str, result_dict in dd_results.items():
        dd_t = result_dict["time"]
        dd_d = result_dict["data"]
        stride_dd = max(1, len(dd_t) // target_plot_points)
        dd_results_decimated[w_str] = {
            "time": dd_t[::stride_dd],
            "data": dd_d[::stride_dd],
            "rmse": result_dict["rmse"],
            "rmse_per_time": result_dict["rmse_per_time"],
            "alpha": result_dict["alpha"],
            "rank": result_dict["rank"]
        }

    # Rank-ordered list (best -> worst) for consistent plotting/legend
    dd_items_ranked = sorted(dd_results_decimated.items(), key=lambda kv: kv[1]["rank"])
    single_dd_mode = len(dd_items_ranked) == 1

    dd_linewidth = 2.4 if single_dd_mode else 1.5
    dd_linestyle = '-'
    dd_zorder = 18 if single_dd_mode else 5
    base_linewidth = 2.0 if single_dd_mode else 2.2
    base_alpha = 0.9 if single_dd_mode else 1.0
    
    # Global Y-Axis Scaling (Plot_Local_Debug logic)
    y_min = np.min(exp_data)
    y_max = np.max(exp_data)
    y_margin = (y_max - y_min) * 0.10
    if y_margin == 0: 
        y_margin = 1e-8
    global_y_lim = (y_min - y_margin, y_max + y_margin)
    
    # Create figure with GridSpec (proper spacing)
    num_sensors = len(sensor_nodes)
    fig = plt.figure(figsize=(20, 18))
    
    # GridSpec: 5 rows (4 for sensors, 1 for RMSE), 4 columns
    gs = fig.add_gridspec(5, 4, hspace=0.4, wspace=0.3, 
                          height_ratios=[1, 1, 1, 1, 0.8])
    
    # Limit plotting time
    max_plot_time = min(exp_times[-1], base_t[-1] if base_t is not None else exp_times[-1])
    
    # Plot each sensor (4x4 grid)
    for i in range(num_sensors):
        row = i // 4
        col = i % 4
        ax = fig.add_subplot(gs[row, col])
        
        sensor_node = sensor_nodes[i]
        
        # LAYER 1: Experimental (purple crosses) - BOTTOM
        exp_mask = exp_times <= max_plot_time
        exp_t_plot = exp_times[exp_mask]
        exp_d_plot = exp_data[exp_mask]
        ax.plot(exp_t_plot * 1e3, exp_d_plot[:, i], marker='x', color='purple', 
               linestyle='none', markersize=6, label='Experiment', zorder=1)
        
        # LAYER 2: DD simulations (worst first, best last)
        for w_str, result_dict in reversed(dd_items_ranked):
            dd_t = result_dict["time"]
            dd_d = result_dict["data"]
            alpha = result_dict["alpha"]

            dd_mask = dd_t <= max_plot_time
            dd_t_plot = dd_t[dd_mask]
            dd_d_plot = dd_d[dd_mask]

            ax.plot(dd_t_plot * 1e3, dd_d_plot[:, sensor_node], color='blue',
                    linewidth=dd_linewidth, alpha=alpha, linestyle=dd_linestyle, zorder=dd_zorder)

        # LAYER 3: Baseline (solid yellow line) - TOP
        if base_d_plot is not None:
            base_mask = base_t_plot <= max_plot_time
            base_t_plot_masked = base_t_plot[base_mask]
            base_d_plot_masked = base_d_plot[base_mask]
            ax.plot(base_t_plot_masked * 1e3, base_d_plot_masked[:, sensor_node],
                    color='orange', linewidth=base_linewidth, alpha=base_alpha, linestyle='-',
                    label='Baseline', zorder=15)
        
        ax.set_ylim(global_y_lim)
        ax.set_xlabel('Time (ms)', fontsize=9)
        ax.set_ylabel('Displacement (m)', fontsize=9)
        ax.set_title(f'x = {actual_positions[i]:.2f}m', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=8)
        
        if i == 0:
            ax.legend(fontsize=8, loc='upper left')
    
    # RMSE plot (spans all 4 columns in row 5)
    ax_rmse = fig.add_subplot(gs[4, :])
    
    # LAYER 1: Experimental marker (faint baseline)
    exp_mask = exp_times <= max_plot_time
    exp_t_plot = exp_times[exp_mask]
    ax_rmse.axhline(y=0, color='purple', linestyle='--', alpha=0.2, linewidth=1, zorder=1)
    
    # LAYER 2: DD RMSE traces
    for w_str, result_dict in reversed(dd_items_ranked):
        rmse_per_time = result_dict["rmse_per_time"]
        alpha = result_dict["alpha"]

        exp_mask = exp_times <= max_plot_time
        exp_t_plot = exp_times[exp_mask]
        rmse_plot = rmse_per_time[exp_mask]

        ax_rmse.plot(exp_t_plot * 1e3, rmse_plot, color='blue', marker='o',
                linewidth=dd_linewidth, alpha=alpha, markersize=4, linestyle=dd_linestyle, zorder=dd_zorder)

    # LAYER 3: Baseline RMSE - TOP (solid yellow)
    if base_rmse_per_time is not None:
        exp_mask = exp_times <= max_plot_time
        exp_t_plot = exp_times[exp_mask]
        base_rmse_plot = base_rmse_per_time[exp_mask]
        ax_rmse.plot(exp_t_plot * 1e3, base_rmse_plot, color='orange', marker='o',
                linewidth=base_linewidth, alpha=base_alpha, label='Baseline', zorder=15)

    ax_rmse.set_xlabel('Time (ms)', fontsize=10)
    ax_rmse.set_ylabel('RMSE (m)', fontsize=10)
    ax_rmse.set_title('Global Root Mean Square Error', fontsize=11, fontweight='bold')
    ax_rmse.grid(True, alpha=0.3)

    # Full legend including DD runs (best -> worst)
    legend_handles = [
        Line2D([0], [0], marker='x', color='purple', linestyle='none',
               markersize=6, label='Experiment'),
        Line2D([0], [0], color='blue', linewidth=1.8, alpha=0.9,
               label='Data-Driven'),
        Line2D([0], [0], color='orange', linewidth=2.2, linestyle='-',
               label='Baseline'),
    ]
    ax_rmse.legend(handles=legend_handles, fontsize=9, ncol=3, loc='upper right')

    output_path = args.plot_dir / args.output_name
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    print(f"\n[SUCCESS] Overlay plot saved: {output_path}")
    plt.close()

if __name__ == "__main__":
    main()