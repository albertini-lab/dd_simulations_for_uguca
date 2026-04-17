#!/usr/bin/env python3
"""
Plot_DcA_RMSE.py

Generic DD vs baseline RMSE sweep plotter.

Default behavior matches the historical Dc/a workflow, but this script now supports
arbitrary paired naming and per-case experimental datasets.
"""

import argparse
import glob
import os
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


# --- CONFIGURATION ---
SIM_DIR = Path("/Users/joshmcneely/uguca/build/simulations")
EXP_BASE_DIR = Path("/Users/joshmcneely/introsims/simulation_outputs")
PLOT_DIR = SIM_DIR / "comparison_plots"

DOMAIN_LENGTH = 6.0
NB_NODES = 2048
DD_SIM_NAME = "local_debug_run"
BASE_SIM_NAME = "local_baseline_run"
SENSOR_POSITIONS = np.arange(0.05, 3.05 + 0.01, 0.2)
SENSOR_POSITIONS = np.round(SENSOR_POSITIONS, 12)
SENSOR_POSITIONS = SENSOR_POSITIONS[SENSOR_POSITIONS <= 3.05]

MIN_TIME_STEPS = 10
MIN_SIM_TIME = 1e-6
COMPLETION_TIME_FRACTION = 0.999
DEBUG_LOG_PROGRESS = os.getenv("DCA_PLOT_DEBUG", "1") != "0"


def debug_log(message):
    if DEBUG_LOG_PROGRESS:
        print(f"[DEBUG] {message}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create DD vs baseline RMSE comparison for paired sweep cases.",
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
    parser.add_argument("--domain-length", type=float, default=DOMAIN_LENGTH, help="Domain length in meters")
    parser.add_argument(
        "--completion-time-fraction",
        type=float,
        default=COMPLETION_TIME_FRACTION,
        help="Minimum fraction of final experimental time required for accepted runs",
    )
    parser.add_argument("--output-prefix", type=str, default="dc_a", help="Output filename prefix tag")

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


def resolve_exp_inputs_for_case(args, case_key):
    if args.exp_restart_dir is not None:
        restart_dir = args.exp_restart_dir
        exp_name = exp_name_from_restart_dir(restart_dir)
        return exp_name, restart_dir

    exp_name = case_key if args.exp_name_from_case_key else args.exp_name
    restart_dir = args.exp_base_dir / f"{exp_name}-restart"
    return exp_name, restart_dir


def is_simulation_complete(base_path, sim_name, nb_nodes):
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
    except Exception as exc:
        debug_log(f"Failed to load simulation {base_path}: {exc}")
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

    data = []
    times = []
    for filepath in files:
        step = int(re.search(r"\.s(\d+)\.out$", filepath).group(1))
        times.append(step * dt_sim)
        raw_bin = np.fromfile(filepath, dtype=np.float32)
        u_x = raw_bin[:nb_nodes]
        data.append(u_x[sensor_nodes])

    return np.array(times), np.array(data)


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
            print(f"[WARNING] Missing baseline pair for {dd_dir.name}. Skipping.")
            continue
        if not baseline_dir.exists():
            print(f"[WARNING] Baseline path missing on disk for {dd_dir.name}. Skipping.")
            continue

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


def main():
    args = parse_args()

    dx = args.domain_length / args.nb_nodes
    sensor_nodes = np.round(SENSOR_POSITIONS / dx).astype(int)
    sensor_nodes = sensor_nodes[sensor_nodes < args.nb_nodes]

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
        print("[CRITICAL ERROR] No paired DD/baseline cases found.")
        return

    exp_cache = {}

    labels = []
    dd_means = []
    base_means = []

    print(f"Found {len(cases)} paired case(s):")
    for case in cases:
        exp_name, restart_dir = resolve_exp_inputs_for_case(args, case["key"])
        cache_key = (str(restart_dir), exp_name)
        if cache_key not in exp_cache:
            if not restart_dir.exists():
                print(f"  Skipping {case['label']} (missing experimental restart: {restart_dir})")
                continue
            try:
                exp_times, exp_data = load_exp(restart_dir, sensor_nodes, args.nb_nodes, exp_name=exp_name)
            except Exception as exc:
                print(f"  Skipping {case['label']} (experimental load failed: {exc})")
                continue
            if exp_data is None or len(exp_data) == 0:
                print(f"  Skipping {case['label']} (empty experimental data)")
                continue
            exp_cache[cache_key] = (exp_times, exp_data)

        exp_times, exp_data = exp_cache[cache_key]
        required_end_time = exp_times[-1] * args.completion_time_fraction

        dd_t, dd_d = load_sim(case["dd_dir"], args.dd_sim_name, args.nb_nodes)
        base_t, base_d = load_sim(case["baseline_dir"], args.baseline_sim_name, args.nb_nodes)
        if dd_t is None or base_t is None:
            print(f"  Skipping {case['label']} (missing simulation files)")
            continue

        if dd_t[-1] < required_end_time or base_t[-1] < required_end_time:
            print(
                f"  Skipping {case['label']} (incomplete duration: "
                f"DD={dd_t[-1]:.6e}s, Baseline={base_t[-1]:.6e}s, "
                f"required>={required_end_time:.6e}s)"
            )
            continue

        common_t_max = min(dd_t[-1], base_t[-1], exp_times[-1])
        common_mask = exp_times <= common_t_max
        if np.count_nonzero(common_mask) == 0:
            print(f"  Skipping {case['label']} (no common time window)")
            continue

        common_exp_times = exp_times[common_mask]
        common_exp_data = exp_data[common_mask]

        dd_mean = mean_rmse_vs_exp(dd_t, dd_d, common_exp_times, common_exp_data, sensor_nodes)
        base_mean = mean_rmse_vs_exp(base_t, base_d, common_exp_times, common_exp_data, sensor_nodes)

        labels.append(case["label"])
        dd_means.append(dd_mean)
        base_means.append(base_mean)

        print(
            f"  {case['label']}: DD={dd_mean:.4e}, Baseline={base_mean:.4e} "
            f"(exp={exp_name})"
        )

    if not labels:
        print("No valid paired cases to plot.")
        return

    sorted_indices = np.argsort(dd_means)[::-1]
    labels = [labels[i] for i in sorted_indices]
    dd_means = [dd_means[i] for i in sorted_indices]
    base_means = [base_means[i] for i in sorted_indices]

    x = np.arange(len(labels))

    fig_width = max(11, 1.2 * len(labels))
    fig, ax = plt.subplots(figsize=(fig_width, 6.5))

    for i in range(len(x)):
        ax.plot(
            [x[i], x[i]],
            [dd_means[i], base_means[i]],
            color="gray",
            linewidth=1.5,
            alpha=0.6,
            zorder=1,
        )

        if base_means[i] > 0:
            improvement = 100.0 * (base_means[i] - dd_means[i]) / base_means[i]
            mid_y = np.sqrt(max(dd_means[i], 1e-16) * max(base_means[i], 1e-16))
            label_text = f"{improvement:.1f}%"
        else:
            mid_y = max(dd_means[i], 1e-16)
            label_text = "n/a"

        ax.text(
            x[i] + 0.12,
            mid_y,
            label_text,
            fontsize=8,
            color="darkgreen",
            weight="bold",
            va="center",
            rotation=0,
            zorder=2,
        )

    ax.scatter(
        x,
        dd_means,
        s=100,
        marker="o",
        color="tab:red",
        label="Data-Driven",
        zorder=3,
        edgecolors="darkred",
        linewidths=1.5,
    )
    ax.scatter(
        x,
        base_means,
        s=100,
        marker="s",
        color="tab:blue",
        label="Baseline",
        zorder=3,
        edgecolors="darkblue",
        linewidths=1.5,
    )

    positive_floor = 1e-16
    lower_bound = max(min(np.min(dd_means), np.min(base_means)) * 0.7, positive_floor)
    upper_bound = max(np.max(dd_means), np.max(base_means)) * 1.4

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
    ax.set_xlabel("Sweep cases", fontsize=13, weight="bold")
    ax.set_ylabel("Average RMSE (m)", fontsize=13, weight="bold")
    ax.set_title("Data-Driven vs Baseline: RMSE Comparison", fontsize=15, weight="bold")
    ax.set_yscale("log")
    ax.set_ylim(lower_bound, upper_bound)
    ax.set_xlim(-0.5, len(x) - 0.5 + 0.3)
    ax.grid(True, axis="y", alpha=0.3, linestyle="--")
    ax.grid(True, axis="x", alpha=0.15, linestyle=":")

    legend_elements = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="tab:red",
            markeredgecolor="darkred",
            markersize=10,
            linewidth=0,
            label="Data-Driven",
        ),
        Line2D(
            [0],
            [0],
            marker="s",
            color="w",
            markerfacecolor="tab:blue",
            markeredgecolor="darkblue",
            markersize=10,
            linewidth=0,
            label="Baseline",
        ),
        Line2D(
            [0],
            [0],
            color="darkgreen",
            linewidth=0,
            marker="$\\%$",
            markersize=12,
            label="% RMSE Improvement\n(DD vs Baseline)",
        ),
    ]
    ax.legend(handles=legend_elements, fontsize=10, loc="upper right", framealpha=0.9)

    args.plot_dir.mkdir(exist_ok=True)
    out_path = args.plot_dir / f"rmse_vs_{args.output_prefix}.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close()

    print(f"\n[SUCCESS] Plot saved to: {out_path}")


if __name__ == "__main__":
    main()
