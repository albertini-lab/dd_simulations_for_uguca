#!/usr/bin/env python3

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

# --- PATH CONFIGURATION ---
SWEEP_RESULTS_DIR = Path(".")
EXP_BASE_DIR = Path("/Users/joshmcneely/introsims/simulation_outputs")
PLOT_DIR = Path("./comparison_plots")

# --- GEOMETRY & SETTINGS ---
DOMAIN_LENGTH = 6.0
NB_NODES = 2048
SENSOR_POSITIONS = np.arange(0.05, 3.05 + 0.01, 0.2)
SENSOR_POSITIONS = np.round(SENSOR_POSITIONS, 12)
SENSOR_POSITIONS = SENSOR_POSITIONS[SENSOR_POSITIONS <= 3.05]
SIM_DENSITY_MULTIPLIER = 30

# --- SIMULATION NAMES ---
DD_SIM_NAME = "local_debug_run"
BASE_SIM_NAME = "local_baseline_run"

MIN_TIME_STEPS = 10  # Minimum timesteps for a complete simulation
MIN_SIM_TIME = 1e-6  # Minimum simulation time in seconds
COMPLETION_TIME_FRACTION = 0.999  # Require runs to reach ~full experimental time
DEBUG_LOG_PROGRESS = os.getenv("DCA_PLOT_DEBUG", "1") != "0"


def debug_log(message):
    if DEBUG_LOG_PROGRESS:
        print(f"[DEBUG] {message}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create DD/baseline overlay plots with experimental comparisons.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--sim-dir", type=Path, default=SWEEP_RESULTS_DIR, help="Root directory containing case folders")
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

    return args


def strip_prefix(name, prefix):
    if prefix and name.startswith(prefix):
        return name[len(prefix):]
    return name


def exp_name_from_restart_dir(restart_dir):
    name = restart_dir.name
    suffix = "-restart"
    if name.endswith(suffix):
        return name[:-len(suffix)]
    return name


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


def load_sim_robust(base_path, sim_name, nb_nodes):
    try:
        # First check if simulation is complete
        is_complete, reason = is_simulation_complete(base_path, sim_name, nb_nodes)
        if not is_complete:
            debug_log(f"{base_path.name} [{sim_name}] skipped in load: {reason}")
            return None, None

        data_dir = base_path / f"{sim_name}-interface-DataFiles"
        path = data_dir / "top_disp.out"
        t_path = base_path / f"{sim_name}-interface.time"

        with open(t_path, "r") as file_obj:
            times = np.array([float(line.split()[1]) for line in file_obj.readlines()])

        raw_data = np.loadtxt(path)
        if raw_data.ndim == 1:
            raw_data = raw_data.reshape(1, -1)

        if raw_data.shape[1] >= nb_nodes:
            data = raw_data[:, :nb_nodes]
        else:
            data = raw_data

        final_count = min(len(times), data.shape[0])
        return times[:final_count], data[:final_count, :]

    except Exception as exc:
        print(f"[ERROR] Could not load {sim_name} from {base_path}: {exc}")
        return None, None


def load_exp(restart_dir, sensor_nodes, nb_nodes, exp_name=None):
    if exp_name is None:
        exp_name = exp_name_from_restart_dir(restart_dir)

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


def resolve_exp_inputs_for_case(args, case_key):
    if args.exp_restart_dir is not None:
        restart_dir = args.exp_restart_dir
        exp_name = exp_name_from_restart_dir(restart_dir)
        return exp_name, restart_dir

    exp_name = case_key if args.exp_name_from_case_key else args.exp_name
    restart_dir = args.exp_base_dir / f"{exp_name}-restart"
    return exp_name, restart_dir


def discover_paired_cases(
    base_dir,
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

    baseline_dirs = sorted(base_dir.glob(baseline_glob))
    baseline_by_key = {
        strip_prefix(folder.name, baseline_prefix): folder for folder in baseline_dirs
    }

    dd_dirs = sorted(base_dir.glob(dd_glob))
    total_cases = len(dd_dirs)
    for dd_idx, dd_dir in enumerate(dd_dirs, start=1):
        debug_log(f"Discover {dd_idx}/{total_cases}: {dd_dir.name}")

        key = strip_prefix(dd_dir.name, dd_prefix)
        baseline_dir = baseline_by_key.get(key)
        if baseline_dir is None:
            print(f"[WARNING] Missing baseline pair for {dd_dir.name}. Skipping.")
            continue
        if not baseline_dir.exists():
            print(f"[WARNING] Baseline path missing on disk for {dd_dir.name}. Skipping.")
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


def main():
    args = parse_args()
    args.plot_dir.mkdir(exist_ok=True)

    dx = args.domain_length / args.nb_nodes
    sensor_nodes_raw = np.round(SENSOR_POSITIONS / dx).astype(int)
    valid_sensors = sensor_nodes_raw < args.nb_nodes
    sensor_nodes = sensor_nodes_raw[valid_sensors]
    actual_positions = sensor_nodes * dx

    print("[INFO] Discovering paired DD/baseline cases...")
    cases = discover_paired_cases(
        base_dir=args.sim_dir,
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

    print(f"[INFO] Found {len(cases)} case(s).")

    exp_cache = {}

    case_results = []
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
                print(f"[WARNING] Experimental data empty for {case['label']} ({exp_name}).")
                continue
            exp_cache[cache_key] = (exp_times_case, exp_data_case)

        exp_times_case, exp_data_case = exp_cache[cache_key]
        required_end_time = exp_times_case[-1] * args.completion_time_fraction

        dd_t, dd_d = load_sim_robust(case["dd_dir"], args.dd_sim_name, args.nb_nodes)
        base_t, base_d = load_sim_robust(case["baseline_dir"], args.baseline_sim_name, args.nb_nodes)

        if dd_d is None or base_d is None:
            print(f"[WARNING] Missing sim data for case {case['label']}. Skipping.")
            continue

        if dd_t[-1] < required_end_time or base_t[-1] < required_end_time:
            print(
                f"[WARNING] Incomplete run duration for case {case['label']}. "
                f"DD t_end={dd_t[-1]:.6e}s, Baseline t_end={base_t[-1]:.6e}s, "
                f"required >= {required_end_time:.6e}s. Skipping."
            )
            continue

        idx_dd = np.searchsorted(dd_t, exp_times_case).clip(0, len(dd_t) - 1)
        dd_sensors_exact = dd_d[idx_dd][:, sensor_nodes]
        dd_rmse_per_time = np.sqrt(np.mean((dd_sensors_exact - exp_data_case) ** 2, axis=1))

        idx_base = np.searchsorted(base_t, exp_times_case).clip(0, len(base_t) - 1)
        base_sensors_exact = base_d[idx_base][:, sensor_nodes]
        base_rmse_per_time = np.sqrt(np.mean((base_sensors_exact - exp_data_case) ** 2, axis=1))

        case_results.append(
            {
                "case": case,
                "exp_name": exp_name,
                "exp_times": exp_times_case,
                "exp_data": exp_data_case,
                "dd_t": dd_t,
                "dd_d": dd_d,
                "base_t": base_t,
                "base_d": base_d,
                "dd_rmse_per_time": dd_rmse_per_time,
                "base_rmse_per_time": base_rmse_per_time,
                "dd_rmse_mean": float(np.mean(dd_rmse_per_time)),
                "base_rmse_mean": float(np.mean(base_rmse_per_time)),
            }
        )

    if not case_results:
        print("[CRITICAL ERROR] No complete Dc/a cases could be loaded.")
        return

    for result in case_results:
        label = result["case"]["label"]
        print(
            f"  {label}: DD mean RMSE={result['dd_rmse_mean']:.3e}, "
            f"Baseline mean RMSE={result['base_rmse_mean']:.3e}"
        )

    target_plot_points = max(len(result["exp_times"]) for result in case_results) * args.sim_density_multiplier
    for result in case_results:
        stride_dd = max(1, len(result["dd_t"]) // target_plot_points)
        stride_base = max(1, len(result["base_t"]) // target_plot_points)

        result["dd_t_plot"] = result["dd_t"][::stride_dd]
        result["dd_d_plot"] = result["dd_d"][::stride_dd]
        result["base_t_plot"] = result["base_t"][::stride_base]
        result["base_d_plot"] = result["base_d"][::stride_base]

    y_min = min(np.min(result["exp_data"]) for result in case_results)
    y_max = max(np.max(result["exp_data"]) for result in case_results)
    y_margin = (y_max - y_min) * 0.10
    if y_margin == 0:
        y_margin = 1e-8
    global_y_lim = (y_min - y_margin, y_max + y_margin)

    unique_exp_names = sorted({result["exp_name"] for result in case_results})
    show_single_exp_markers = len(unique_exp_names) == 1
    if show_single_exp_markers:
        reference_exp_times = case_results[0]["exp_times"]
        reference_exp_data = case_results[0]["exp_data"]
    else:
        reference_exp_times = None
        reference_exp_data = None
        print(
            "[INFO] Multiple experimental datasets detected; sensor markers are omitted "
            "to avoid overlay ambiguity."
        )

    num_sensors = len(sensor_nodes)
    cmap = plt.get_cmap("viridis")

    displacement_linewidth = 1.0
    rmse_linewidth = 1.1
    rmse_marker_size = 1.8
    rmse_floor = 1e-16

    def build_norm(values):
        vmin = float(np.min(values))
        vmax = float(np.max(values))
        if np.isclose(vmin, vmax):
            vmax = vmin + max(1e-16, abs(vmin) * 1e-6)
        return Normalize(vmin=vmin, vmax=vmax)

    def plot_family(
        family_key,
        time_key,
        data_key,
        rmse_key,
        rmse_mean_key,
        line_style,
        title,
        colorbar_label,
        output_filename,
    ):
        fig = plt.figure(figsize=(22, 19))
        gs = fig.add_gridspec(5, 4, hspace=0.4, wspace=0.3, height_ratios=[1, 1, 1, 1, 0.9])

        mean_rmses = np.array([result[rmse_mean_key] for result in case_results], dtype=float)
        norm = build_norm(mean_rmses)

        for sensor_idx in range(num_sensors):
            row = sensor_idx // 4
            col = sensor_idx % 4
            axis = fig.add_subplot(gs[row, col])

            sensor_node = sensor_nodes[sensor_idx]
            if show_single_exp_markers:
                axis.plot(
                    reference_exp_times * 1e3,
                    reference_exp_data[:, sensor_idx],
                    marker="x",
                    color="red",
                    linestyle="none",
                    markersize=3,
                    zorder=30,
                )

            for result in case_results:
                color = cmap(norm(result[rmse_mean_key]))
                mask = result[time_key] <= result["exp_times"][-1]

                axis.plot(
                    result[time_key][mask] * 1e3,
                    result[data_key][mask, sensor_node],
                    color=color,
                    linewidth=displacement_linewidth,
                    linestyle=line_style,
                    zorder=12,
                )

            axis.set_ylim(global_y_lim)
            axis.set_xlabel("Time (ms)", fontsize=9)
            axis.set_ylabel("Displacement (m)", fontsize=9)
            axis.set_title(f"x = {actual_positions[sensor_idx]:.2f}m", fontsize=10)
            axis.grid(True, alpha=0.3)
            axis.tick_params(labelsize=8)

        ax_rmse = fig.add_subplot(gs[4, :])
        for result in case_results:
            color = cmap(norm(result[rmse_mean_key]))
            ax_rmse.plot(
                result["exp_times"] * 1e3,
                np.clip(result[rmse_key], rmse_floor, None),
                color=color,
                linewidth=rmse_linewidth,
                linestyle=line_style,
                marker="o",
                markersize=rmse_marker_size,
            )

        ax_rmse.set_xlabel("Time (ms)", fontsize=10)
        ax_rmse.set_ylabel("RMSE (m)", fontsize=10)
        ax_rmse.set_title(title, fontsize=11, fontweight="bold")
        ax_rmse.set_yscale("linear")
        ax_rmse.grid(True, alpha=0.3)

        scalar_mappable = ScalarMappable(norm=norm, cmap=cmap)
        scalar_mappable.set_array([])
        colorbar = fig.colorbar(
            scalar_mappable,
            ax=fig.axes,
            fraction=0.018,
            pad=0.02,
        )
        colorbar.set_label(colorbar_label, fontsize=9)

        output_path = args.plot_dir / output_filename
        plt.savefig(output_path, dpi=200, bbox_inches="tight")
        plt.close()
        print(f"[SUCCESS] Saved {family_key} overlay: {output_path}")

    plot_family(
        family_key="dd",
        time_key="dd_t_plot",
        data_key="dd_d_plot",
        rmse_key="dd_rmse_per_time",
        rmse_mean_key="dd_rmse_mean",
        line_style="-",
        title="Global RMSE (Data-Driven only)",
        colorbar_label="Mean DD RMSE (m)",
        output_filename=f"overlay_{args.output_prefix}_viridis_dd.png",
    )

    plot_family(
        family_key="baseline",
        time_key="base_t_plot",
        data_key="base_d_plot",
        rmse_key="base_rmse_per_time",
        rmse_mean_key="base_rmse_mean",
        line_style="-",
        title="Global RMSE (Baseline only)",
        colorbar_label="Mean Baseline RMSE (m)",
        output_filename=f"overlay_{args.output_prefix}_viridis_baseline.png",
    )


if __name__ == "__main__":
    main()
