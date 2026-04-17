#!/usr/bin/env python3
"""
Plot_DcA_Slip_Isochrones.py

Creates two plots for the Dc/a data-driven sweep:
1) Final-time slip vs node overlay for all runs.
2) Best-run slip vs node isochrones at multiple times.

Best run is selected by minimum mean DD RMSE vs experiment.
Slip is computed as top displacement minus its initial snapshot.
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
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from matplotlib.legend_handler import HandlerBase
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches

# --- CONFIGURATION ---
SIM_DIR = Path("/Users/joshmcneely/uguca/build/simulations")
EXP_BASE_DIR = Path("/Users/joshmcneely/introsims/simulation_outputs")
PLOT_DIR = SIM_DIR / "comparison_plots"

DOMAIN_LENGTH = 6.0
NB_NODES = 2048
PLOT_NODES = NB_NODES // 2
DD_SIM_NAME = "local_debug_run"
BASE_SIM_NAME = "local_baseline_run"
SENSOR_POSITIONS = np.arange(0.05, 3.05 + 0.01, 0.2)
SENSOR_POSITIONS = np.round(SENSOR_POSITIONS, 12)
SENSOR_POSITIONS = SENSOR_POSITIONS[SENSOR_POSITIONS <= 3.05]

MIN_TIME_STEPS = 10  # Minimum timesteps for a complete simulation
MIN_SIM_TIME = 1e-6  # Minimum simulation time in seconds
COMPLETION_TIME_FRACTION = 0.999  # Require runs to reach ~full experimental time
DEBUG_LOG_PROGRESS = os.getenv("DCA_PLOT_DEBUG", "1") != "0"
ISOCHRONE_COUNT = 150  # Higher temporal frequency for best-run isochrones
PLOT_EXPERIMENT_OVERLAY = False  # Temporarily disable experimental point overlays
PLOT_ALL_EVOLUTIONS = False


def debug_log(message):
    if DEBUG_LOG_PROGRESS:
        print(f"[DEBUG] {message}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create slip isochrone plots for DD/baseline simulation outputs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--sim-dir", type=Path, default=SIM_DIR, help="Root directory containing case folders")
    parser.add_argument("--plot-dir", type=Path, default=PLOT_DIR, help="Directory to write plots")
    parser.add_argument("--exp-base-dir", type=Path, default=EXP_BASE_DIR, help="Experimental data base directory")
    parser.add_argument("--exp-restart-dir", type=Path, default=None, help="Direct path to experimental restart directory")
    parser.add_argument("--exp-name", type=str, default="mcklaskey_debug", help="Experimental dataset base name (without -restart/.info)")
    parser.add_argument(
        "--exp-name-from-case-key",
        action="store_true",
        help="Use discovered case key as experimental dataset base name for each pair",
    )

    parser.add_argument("--dd-sim-name", type=str, default=DD_SIM_NAME, help="DD simulation run prefix")
    parser.add_argument("--baseline-sim-name", type=str, default=BASE_SIM_NAME, help="Baseline simulation run prefix")

    parser.add_argument("--dd-glob", type=str, default="results_dd_*", help="Glob for DD case folders")
    parser.add_argument("--baseline-glob", type=str, default="results_baseline_*", help="Glob for baseline case folders")
    parser.add_argument("--dd-prefix", type=str, default="results_dd_", help="Prefix stripped from DD folder names for pairing")
    parser.add_argument(
        "--baseline-prefix",
        type=str,
        default="results_baseline_",
        help="Prefix stripped from baseline folder names for pairing",
    )

    parser.add_argument("--dd-dir", type=Path, default=None, help="Explicit DD case directory")
    parser.add_argument("--baseline-dir", type=Path, default=None, help="Explicit baseline case directory")
    parser.add_argument("--case-label", type=str, default=None, help="Label for explicit DD/baseline pair")

    parser.add_argument("--nb-nodes", type=int, default=NB_NODES, help="Number of interface nodes")
    parser.add_argument("--plot-nodes", type=int, default=PLOT_NODES, help="Number of nodes to plot from x=0")
    parser.add_argument("--domain-length", type=float, default=DOMAIN_LENGTH, help="Domain length in meters")
    parser.add_argument("--isochrone-count", type=int, default=ISOCHRONE_COUNT, help="Number of time isochrones for best-run evolution")
    parser.add_argument(
        "--completion-time-fraction",
        type=float,
        default=COMPLETION_TIME_FRACTION,
        help="Minimum fraction of final experimental time required for accepted runs",
    )
    parser.add_argument(
        "--plot-experiment-overlay",
        action="store_true",
        default=PLOT_EXPERIMENT_OVERLAY,
        help="Overlay experimental points on slip isochrone plots",
    )
    parser.add_argument(
        "--plot-all-evolutions",
        action="store_true",
        default=PLOT_ALL_EVOLUTIONS,
        help="Generate temporal DD isochrone plot for every run (not only best run)",
    )
    parser.add_argument(
        "--output-prefix",
        type=str,
        default="dc_a",
        help="Output filename prefix tag",
    )

    args = parser.parse_args()

    explicit_mode = args.dd_dir is not None or args.baseline_dir is not None
    if explicit_mode and (args.dd_dir is None or args.baseline_dir is None):
        parser.error("Both --dd-dir and --baseline-dir are required in explicit mode.")

    if args.exp_restart_dir is not None and args.exp_name_from_case_key:
        parser.error("--exp-restart-dir cannot be combined with --exp-name-from-case-key.")

    if args.plot_nodes <= 0 or args.plot_nodes > args.nb_nodes:
        parser.error("--plot-nodes must be between 1 and --nb-nodes.")

    return args


def strip_prefix(name, prefix):
    if prefix and name.startswith(prefix):
        return name[len(prefix):]
    return name


def sanitize_tag(text):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("_")


def exp_name_from_restart_dir(restart_dir):
    name = restart_dir.name
    suffix = "-restart"
    if name.endswith(suffix):
        return name[:-len(suffix)]
    return name


class GradientHandler(HandlerBase):
    """Custom legend handler that creates a gradient effect for viridis colormap."""
    def __init__(self, cmap, norm):
        super().__init__()
        self.cmap = cmap
        self.norm = norm
    
    def create_artists(self, legend, orig_handle, xdescent, ydescent, width, height, fontsize, trans):
        # Create gradient effect by overlaying multiple thin rectangles
        n_segments = 20
        segment_width = width / n_segments
        artists = []
        
        for i in range(n_segments):
            x_pos = xdescent + i * segment_width
            color_val = i / (n_segments - 1)
            color = self.cmap(color_val)
            mini_rect = Rectangle((x_pos, ydescent), segment_width, height, 
                                facecolor=color, edgecolor='none', transform=trans)
            artists.append(mini_rect)
        
        return artists


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
    # First check if simulation is complete
    is_complete, reason = is_simulation_complete(base_path, sim_name, nb_nodes)
    if not is_complete:
        debug_log(f"{base_path.name} [{sim_name}] skipped in load: {reason}")
        return None, None

    try:
        data_dir = base_path / f"{sim_name}-interface-DataFiles"
        path = data_dir / "top_disp.out"
        t_path = base_path / f"{sim_name}-interface.time"

        with open(t_path, "r") as file_obj:
            times = np.array([float(line.split()[1]) for line in file_obj.readlines()])

        raw_data = np.loadtxt(path)
        if raw_data.ndim == 1:
            raw_data = raw_data.reshape(1, -1)

        data = raw_data[:, :nb_nodes]
        n = min(len(times), data.shape[0])
        return times[:n], data[:n, :]
    except Exception:
        return None, None


def load_exp(restart_dir, sensor_nodes, nb_nodes, exp_name=None):
    if exp_name is None:
        exp_name = exp_name_from_restart_dir(restart_dir)

    info_file = restart_dir.parent / f"{exp_name}.info"
    if not info_file.exists():
        raise FileNotFoundError(f"Experimental info file not found: {info_file}")

    with open(info_file, "r") as file_obj:
        dt_sim = float(re.search(r"time_step\s*=\s*([\d.eE+-]+)", file_obj.read()).group(1))

    files = sorted(
        glob.glob(str(restart_dir / "top_disp.proc0.s*.out")),
        key=lambda x: int(re.search(r"\.s(\d+)\.out$", x).group(1)),
    )

    data, times = [], []
    for filepath in files:
        step = int(re.search(r"\.s(\d+)\.out$", filepath).group(1))
        times.append(step * dt_sim)
        raw_bin = np.fromfile(filepath, dtype=np.float32)
        u_x = raw_bin[:nb_nodes]
        data.append(u_x[sensor_nodes])

    return np.array(times), np.array(data)


def resolve_exp_inputs_for_case(args, case_key):
    if args.exp_restart_dir is not None:
        restart_dir = args.exp_restart_dir
        exp_name = exp_name_from_restart_dir(restart_dir)
        return exp_name, restart_dir

    exp_name = case_key if args.exp_name_from_case_key else args.exp_name
    restart_dir = args.exp_base_dir / f"{exp_name}-restart"
    return exp_name, restart_dir


def discover_paired_cases(
    sim_dir,
    dd_glob,
    baseline_glob,
    dd_prefix,
    baseline_prefix,
    dd_sim_name,
    baseline_sim_name,
    nb_nodes,
    explicit_dd_dir=None,
    explicit_baseline_dir=None,
    explicit_label=None,
):
    cases = []
    skipped_incomplete = []

    if explicit_dd_dir is not None and explicit_baseline_dir is not None:
        dd_complete, dd_reason = is_simulation_complete(explicit_dd_dir, dd_sim_name, nb_nodes)
        base_complete, base_reason = is_simulation_complete(explicit_baseline_dir, baseline_sim_name, nb_nodes)

        if not dd_complete or not base_complete:
            status = []
            if not dd_complete:
                status.append(f"DD incomplete ({dd_reason})")
            if not base_complete:
                status.append(f"Baseline incomplete ({base_reason})")
            print(f"[CRITICAL ERROR] Explicit pair invalid: {', '.join(status)}")
            return []

        label = explicit_label or f"{explicit_dd_dir.name} vs {explicit_baseline_dir.name}"
        return [
            {
                "label": label,
                "key": label,
                "dd_dir": explicit_dd_dir,
                "baseline_dir": explicit_baseline_dir,
            }
        ]

    baseline_dirs = sorted(sim_dir.glob(baseline_glob))
    baseline_by_key = {
        strip_prefix(folder.name, baseline_prefix): folder for folder in baseline_dirs
    }

    dd_dirs = sorted(sim_dir.glob(dd_glob))
    total_cases = len(dd_dirs)
    for dd_idx, dd_dir in enumerate(dd_dirs, start=1):
        debug_log(f"Discover {dd_idx}/{total_cases}: {dd_dir.name}")

        key = strip_prefix(dd_dir.name, dd_prefix)
        baseline_dir = baseline_by_key.get(key)

        if baseline_dir is None:
            print(f"[WARNING] Missing baseline pair for {dd_dir.name}; skipping.")
            continue
        if not baseline_dir.exists():
            print(f"[WARNING] Baseline path missing on disk for {dd_dir.name}; skipping.")
            continue
        
        # Check if both DD and baseline simulations are complete
        dd_complete, dd_reason = is_simulation_complete(dd_dir, dd_sim_name, nb_nodes)
        base_complete, base_reason = is_simulation_complete(baseline_dir, baseline_sim_name, nb_nodes)

        if not dd_complete or not base_complete:
            status = []
            if not dd_complete:
                status.append(f"DD incomplete ({dd_reason})")
            if not base_complete:
                status.append(f"Baseline incomplete ({base_reason})")
            skipped_incomplete.append(f"{key} ({', '.join(status)})")
            continue

        cases.append(
            {
                "label": key,
                "key": key,
                "dd_dir": dd_dir,
                "baseline_dir": baseline_dir,
            }
        )
    
    if skipped_incomplete:
        print(f"\n[INFO] Skipped {len(skipped_incomplete)} incomplete simulation(s):")
        for skip_msg in skipped_incomplete:
            print(f"  - {skip_msg}")

    cases.sort(key=lambda item: item["key"])
    return cases


def mean_rmse_vs_exp(sim_t, sim_d, exp_times, exp_data, sensor_nodes):
    idx = np.searchsorted(sim_t, exp_times).clip(0, len(sim_t) - 1)
    sim_at_exp = sim_d[idx][:, sensor_nodes]
    rmse_per_time = np.sqrt(np.mean((sim_at_exp - exp_data) ** 2, axis=1))
    return float(np.mean(rmse_per_time))


def make_safe_norm(values):
    vmin = float(np.min(values))
    vmax = float(np.max(values))
    if np.isclose(vmin, vmax):
        vmax = vmin + max(1e-16, abs(vmin) * 1e-6)
    return Normalize(vmin=vmin, vmax=vmax)


def main():
    args = parse_args()
    args.plot_dir.mkdir(exist_ok=True)

    dx = args.domain_length / args.nb_nodes
    sensor_nodes = np.round(SENSOR_POSITIONS / dx).astype(int)
    sensor_nodes = sensor_nodes[sensor_nodes < args.nb_nodes]
    sensor_nodes_half_mask = sensor_nodes < args.plot_nodes
    sensor_nodes_half = sensor_nodes[sensor_nodes_half_mask]

    cases = discover_paired_cases(
        sim_dir=args.sim_dir,
        dd_glob=args.dd_glob,
        baseline_glob=args.baseline_glob,
        dd_prefix=args.dd_prefix,
        baseline_prefix=args.baseline_prefix,
        dd_sim_name=args.dd_sim_name,
        baseline_sim_name=args.baseline_sim_name,
        nb_nodes=args.nb_nodes,
        explicit_dd_dir=args.dd_dir,
        explicit_baseline_dir=args.baseline_dir,
        explicit_label=args.case_label,
    )
    if not cases:
        print("[CRITICAL ERROR] No valid DD/baseline pairs found.")
        return

    exp_cache = {}

    results = []
    for case in cases:
        exp_name, restart_dir = resolve_exp_inputs_for_case(args, case["key"])
        cache_key = (str(restart_dir), exp_name)
        if cache_key not in exp_cache:
            if not restart_dir.exists():
                print(f"[WARNING] Missing experimental restart dir for {case['label']}: {restart_dir}")
                continue
            try:
                exp_times_case, exp_data_case = load_exp(restart_dir, sensor_nodes, args.nb_nodes, exp_name=exp_name)
            except Exception as exc:
                print(f"[WARNING] Failed loading experimental data for {case['label']} ({exp_name}): {exc}")
                continue

            if exp_data_case is None or len(exp_data_case) == 0:
                print(f"[WARNING] Experimental data empty for {case['label']} ({exp_name}); skipping.")
                continue
            exp_cache[cache_key] = (exp_times_case, exp_data_case)

        exp_times_case, exp_data_case = exp_cache[cache_key]
        required_end_time = exp_times_case[-1] * args.completion_time_fraction

        sim_t, sim_d = load_sim(case["dd_dir"], args.dd_sim_name, args.nb_nodes)
        base_t, base_d = load_sim(case["baseline_dir"], args.baseline_sim_name, args.nb_nodes)
        if sim_t is None or base_t is None:
            print(f"[WARNING] Missing DD/baseline files for {case['label']}; skipping.")
            continue

        if sim_t[-1] < required_end_time or base_t[-1] < required_end_time:
            print(
                f"[WARNING] Incomplete run duration for {case['label']}; "
                f"DD t_end={sim_t[-1]:.6e}s, Baseline t_end={base_t[-1]:.6e}s, "
                f"required >= {required_end_time:.6e}s. Skipping."
            )
            continue

        common_mask = exp_times_case <= min(exp_times_case[-1], sim_t[-1], base_t[-1])
        if np.count_nonzero(common_mask) == 0:
            print(f"[WARNING] No common time range for {case['label']}; skipping.")
            continue

        common_exp_times = exp_times_case[common_mask]
        common_exp_data = exp_data_case[common_mask]
        mean_rmse = mean_rmse_vs_exp(sim_t, sim_d, common_exp_times, common_exp_data, sensor_nodes)

        results.append(
            {
                "case": case,
                "exp_name": exp_name,
                "exp_times": exp_times_case,
                "exp_data": exp_data_case,
                "sim_t": sim_t,
                "sim_d": sim_d,
                "base_t": base_t,
                "base_d": base_d,
                "mean_rmse": mean_rmse,
            }
        )

    if not results:
        print("[CRITICAL ERROR] No valid DD cases to plot.")
        return

    node_index = np.arange(args.plot_nodes)
    rmse_values = np.array([entry["mean_rmse"] for entry in results], dtype=float)
    best_entry = min(results, key=lambda entry: entry["mean_rmse"])

    unique_exp_names = sorted({entry["exp_name"] for entry in results})
    if args.plot_experiment_overlay and len(unique_exp_names) > 1:
        print(
            "[INFO] Multiple experimental datasets detected; final-time experiment overlay "
            "uses only the best-case experimental profile."
        )

    # ------------------------------------------------------------------
    # 1) Final-time slip-vs-node overlay: Data-Driven runs
    # ------------------------------------------------------------------
    dd_fig, dd_ax = plt.subplots(figsize=(12, 6.5))
    cmap = plt.get_cmap("viridis")
    rmse_norm = make_safe_norm(rmse_values)

    for entry in results:
        color = cmap(rmse_norm(entry["mean_rmse"]))
        dd_final_slip = entry["sim_d"][-1, :args.plot_nodes] - entry["sim_d"][0, :args.plot_nodes]
        dd_ax.plot(node_index, dd_final_slip, color=color, linewidth=1.0, linestyle="-")

    exp_final_slip = best_entry["exp_data"][-1] - best_entry["exp_data"][0]
    if args.plot_experiment_overlay:
        dd_ax.plot(
            sensor_nodes_half,
            exp_final_slip[sensor_nodes_half_mask],
            color="red",
            marker="x",
            linestyle="none",
            markersize=3,
            zorder=40,
        )

    dd_ax.set_xlabel("Node index", fontsize=12)
    dd_ax.set_ylabel("Slip (m)", fontsize=12)
    dd_ax.set_title("Final-time slip isochrones: Data-Driven runs", fontsize=14)
    dd_ax.grid(True, alpha=0.3)

    # Create legend with gradient effect
    gradient_patch_dd = mpatches.Patch(label="Data-Driven")
    dd_legend = [gradient_patch_dd]
    if args.plot_experiment_overlay:
        dd_legend.append(
            Line2D([0], [0], color="red", marker="x", linestyle="none", markersize=6, label="Experiment")
        )
    handler_map_dd = {gradient_patch_dd: GradientHandler(cmap, rmse_norm)}
    dd_ax.legend(handles=dd_legend, handler_map=handler_map_dd, fontsize=10, loc="best")

    dd_sm = ScalarMappable(norm=rmse_norm, cmap=cmap)
    dd_sm.set_array([])
    dd_cbar = dd_fig.colorbar(dd_sm, ax=dd_ax, pad=0.02)
    dd_cbar.set_label("Mean DD RMSE (m)", fontsize=10)

    dd_out_path = args.plot_dir / f"slip_vs_node_final_isochrones_{args.output_prefix}_dd.png"
    dd_fig.tight_layout()
    dd_fig.savefig(dd_out_path, dpi=220, bbox_inches="tight")
    plt.close(dd_fig)

    # ------------------------------------------------------------------
    # 2) Final-time slip-vs-node overlay: Baseline runs
    # ------------------------------------------------------------------
    base_fig, base_ax = plt.subplots(figsize=(12, 6.5))

    for entry in results:
        color = cmap(rmse_norm(entry["mean_rmse"]))
        base_final_slip = entry["base_d"][-1, :args.plot_nodes] - entry["base_d"][0, :args.plot_nodes]
        base_ax.plot(node_index, base_final_slip, color=color, linewidth=1.0, linestyle="-")

    if args.plot_experiment_overlay:
        base_ax.plot(
            sensor_nodes_half,
            exp_final_slip[sensor_nodes_half_mask],
            color="red",
            marker="x",
            linestyle="none",
            markersize=6,
            zorder=40,
        )

    base_ax.set_xlabel("Node index", fontsize=12)
    base_ax.set_ylabel("Slip (m)", fontsize=12)
    base_ax.set_title("Final-time slip isochrones: Baseline runs", fontsize=14)
    base_ax.grid(True, alpha=0.3)

    # Create legend with gradient effect
    gradient_patch_base = mpatches.Patch(label="Baseline")
    base_legend = [gradient_patch_base]
    if args.plot_experiment_overlay:
        base_legend.append(
            Line2D([0], [0], color="red", marker="x", linestyle="none", markersize=6, label="Experiment")
        )
    handler_map_base = {gradient_patch_base: GradientHandler(cmap, rmse_norm)}
    base_ax.legend(handles=base_legend, handler_map=handler_map_base, fontsize=10, loc="best")

    base_sm = ScalarMappable(norm=rmse_norm, cmap=cmap)
    base_sm.set_array([])
    base_cbar = base_fig.colorbar(base_sm, ax=base_ax, pad=0.02)
    base_cbar.set_label("Mean DD RMSE (m)", fontsize=10)

    base_out_path = args.plot_dir / f"slip_vs_node_final_isochrones_{args.output_prefix}_baseline.png"
    base_fig.tight_layout()
    base_fig.savefig(base_out_path, dpi=220, bbox_inches="tight")
    plt.close(base_fig)

    def plot_temporal_isochrones(entry, title, output_path):
        run_t = entry["sim_t"]
        run_d = entry["sim_d"][:, :args.plot_nodes]
        run_exp_times = entry["exp_times"]
        run_exp_data = entry["exp_data"]

        n_isochrones = min(args.isochrone_count, len(run_t))
        sim_iso_indices = np.linspace(0, len(run_t) - 1, num=n_isochrones, dtype=int)
        sim_iso_indices = np.unique(sim_iso_indices)
        sim_iso_times = run_t[sim_iso_indices]

        evo_fig, evo_ax = plt.subplots(figsize=(12, 6.5))
        time_norm = make_safe_norm(sim_iso_times)

        for sim_idx, sim_time in zip(sim_iso_indices, sim_iso_times):
            color = cmap(time_norm(sim_time))
            slip_profile = run_d[sim_idx] - run_d[0]
            evo_ax.plot(node_index, slip_profile, color=color, linewidth=1.1, zorder=10)

            if args.plot_experiment_overlay:
                exp_idx = int(np.clip(np.searchsorted(run_exp_times, sim_time), 0, len(run_exp_times) - 1))
                exp_slip_profile = run_exp_data[exp_idx] - run_exp_data[0]
                evo_ax.plot(
                    sensor_nodes_half,
                    exp_slip_profile[sensor_nodes_half_mask],
                    color="red",
                    marker="x",
                    linestyle="none",
                    markersize=4.5,
                    alpha=0.9,
                    zorder=35,
                )

        evo_ax.set_xlabel("Node index", fontsize=12)
        evo_ax.set_ylabel("Slip (m)", fontsize=12)
        evo_ax.set_title(title, fontsize=14)
        evo_ax.grid(True, alpha=0.3)

        gradient_patch_evo = mpatches.Patch(label="Simulation (DD)")
        evo_legend = [gradient_patch_evo]
        if args.plot_experiment_overlay:
            evo_legend.append(
                Line2D([0], [0], color="red", marker="x", linestyle="none", markersize=5, label="Experiment")
            )
        handler_map_evo = {gradient_patch_evo: GradientHandler(cmap, time_norm)}
        evo_ax.legend(handles=evo_legend, handler_map=handler_map_evo, fontsize=10, loc="best")

        evo_sm = ScalarMappable(norm=time_norm, cmap=cmap)
        evo_sm.set_array([])
        evo_cbar = evo_fig.colorbar(evo_sm, ax=evo_ax, pad=0.02)
        evo_cbar.set_label("Time (s)", fontsize=10)

        evo_fig.tight_layout()
        evo_fig.savefig(output_path, dpi=220, bbox_inches="tight")
        plt.close(evo_fig)

    # ------------------------------------------------------------------
    # 3) Best-run slip-vs-node isochrones over time
    # ------------------------------------------------------------------
    best_case = best_entry["case"]
    evo_out_path = args.plot_dir / f"slip_vs_node_best_run_isochrones_{args.output_prefix}.png"
    plot_temporal_isochrones(
        best_entry,
        f"Best run evolution isochrones: {best_case['label']}",
        evo_out_path,
    )

    per_run_count = 0
    if args.plot_all_evolutions:
        all_runs_dir = args.plot_dir / f"slip_vs_node_all_run_isochrones_{args.output_prefix}"
        all_runs_dir.mkdir(exist_ok=True)

        for entry in results:
            case = entry["case"]
            case_tag = sanitize_tag(case["key"])
            run_out = all_runs_dir / f"slip_vs_node_isochrones_{args.output_prefix}_{case_tag}.png"
            plot_temporal_isochrones(
                entry,
                f"Run evolution isochrones: {case['label']}",
                run_out,
            )
            per_run_count += 1

    print(f"Best run by mean DD RMSE: {best_case['label']}")
    print(f"[SUCCESS] Saved: {dd_out_path}")
    print(f"[SUCCESS] Saved: {base_out_path}")
    print(f"[SUCCESS] Saved: {evo_out_path}")
    if args.plot_all_evolutions:
        print(
            f"[SUCCESS] Saved per-run temporal isochrones: {per_run_count} file(s) in "
            f"{args.plot_dir / f'slip_vs_node_all_run_isochrones_{args.output_prefix}'}"
        )


if __name__ == "__main__":
    main()
