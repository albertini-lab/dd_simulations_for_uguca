#!/usr/bin/env python3

"""
Compute relative error metrics between baseline-deviated and ground-truth outputs.

For each timestep t, computes:
    r(t) = ||u_baseline(t) - u_gt(t)||_2 / ||u_gt(t)||_2

Reports two values for each component:
  1) Average over all timesteps with non-zero denominator
  2) Final timestep value

Components reported:
  - top_disp
  - bot_disp
  - combined (top + bot concatenated)

Example:
  python compute_baseline_vs_gt_relative_error.py
  python compute_baseline_vs_gt_relative_error.py \
    --baseline-dir simulation_outputs/time_int_opt_baseline_deviated_standard \
    --gt-dir simulation_outputs/time_int_opt_gt_ground_truth
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute average and final relative L2 errors for top/bot/combined displacement.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--baseline-dir",
        type=Path,
        default=Path("simulation_outputs/time_int_opt_baseline_deviated_standard"),
        help="Directory containing baseline-deviated outputs",
    )
    parser.add_argument(
        "--gt-dir",
        type=Path,
        default=Path("simulation_outputs/time_int_opt_gt_ground_truth"),
        help="Directory containing ground-truth outputs",
    )
    parser.add_argument(
        "--baseline-run-name",
        type=str,
        default="time_int_opt_baseline_deviated",
        help="Baseline run name prefix used in .time and -DataFiles paths",
    )
    parser.add_argument(
        "--gt-run-name",
        type=str,
        default="time_int_opt_gt",
        help="Ground-truth run name prefix used in .time and -DataFiles paths",
    )
    parser.add_argument(
        "--alignment",
        choices=["exact", "nearest"],
        default="nearest",
        help="How to align GT data to baseline timesteps",
    )
    return parser.parse_args()


def load_time_file(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"Missing time file: {path}")

    values = []
    with path.open("r") as file_obj:
        for line in file_obj:
            parts = line.strip().split()
            if len(parts) >= 2:
                values.append(float(parts[1]))

    if not values:
        raise ValueError(f"No time rows parsed from: {path}")
    return np.asarray(values, dtype=float)


def load_displacement_file(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"Missing displacement file: {path}")

    data = np.loadtxt(path)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    return data


def trim_rows_to_time(data: np.ndarray, times: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = min(data.shape[0], len(times))
    return data[:n, :], times[:n]


def nearest_indices(target_t: np.ndarray, ref_t: np.ndarray) -> np.ndarray:
    right = np.searchsorted(ref_t, target_t)
    right = np.clip(right, 0, len(ref_t) - 1)
    left = np.clip(right - 1, 0, len(ref_t) - 1)

    right_dist = np.abs(ref_t[right] - target_t)
    left_dist = np.abs(target_t - ref_t[left])
    choose_left = left_dist <= right_dist

    idx = right.copy()
    idx[choose_left] = left[choose_left]
    return idx


def align_gt_to_baseline_times(
    baseline_t: np.ndarray,
    gt_t: np.ndarray,
    gt_data: np.ndarray,
    alignment: str,
) -> np.ndarray:
    if alignment == "exact":
        if len(baseline_t) != len(gt_t) or not np.allclose(baseline_t, gt_t, rtol=0.0, atol=1e-15):
            raise ValueError(
                "Exact alignment requested, but time vectors differ. "
                "Use --alignment nearest for nearest-neighbor time matching."
            )
        return gt_data

    idx = nearest_indices(baseline_t, gt_t)
    return gt_data[idx, :]


def relative_l2_per_timestep(baseline_u: np.ndarray, gt_u: np.ndarray) -> np.ndarray:
    diff = baseline_u - gt_u
    numerator = np.linalg.norm(diff, axis=1)
    denominator = np.linalg.norm(gt_u, axis=1)

    rel = np.full_like(numerator, np.nan, dtype=float)
    valid = denominator > 0.0
    rel[valid] = numerator[valid] / denominator[valid]
    return rel


def summarize(rel: np.ndarray) -> tuple[float, float, int]:
    valid = np.isfinite(rel)
    if not np.any(valid):
        return float("nan"), float("nan"), 0

    avg = float(np.mean(rel[valid]))
    final = float(rel[-1])
    return avg, final, int(np.count_nonzero(valid))


def clip_to_common_columns(a: np.ndarray, b: np.ndarray, label: str) -> tuple[np.ndarray, np.ndarray]:
    cols = min(a.shape[1], b.shape[1])
    if a.shape[1] != b.shape[1]:
        print(
            f"[WARNING] {label}: column mismatch ({a.shape[1]} vs {b.shape[1]}). "
            f"Using first {cols} columns."
        )
    return a[:, :cols], b[:, :cols]


def main() -> None:
    args = parse_args()

    baseline_time_path = args.baseline_dir / f"{args.baseline_run_name}.time"
    gt_time_path = args.gt_dir / f"{args.gt_run_name}.time"

    baseline_top_path = args.baseline_dir / f"{args.baseline_run_name}-DataFiles" / "top_disp.out"
    gt_top_path = args.gt_dir / f"{args.gt_run_name}-DataFiles" / "top_disp.out"
    baseline_bot_path = args.baseline_dir / f"{args.baseline_run_name}-DataFiles" / "bot_disp.out"
    gt_bot_path = args.gt_dir / f"{args.gt_run_name}-DataFiles" / "bot_disp.out"

    baseline_t = load_time_file(baseline_time_path)
    gt_t = load_time_file(gt_time_path)

    baseline_top = load_displacement_file(baseline_top_path)
    gt_top = load_displacement_file(gt_top_path)
    baseline_bot = load_displacement_file(baseline_bot_path)
    gt_bot = load_displacement_file(gt_bot_path)

    baseline_top, baseline_t = trim_rows_to_time(baseline_top, baseline_t)
    baseline_bot, _ = trim_rows_to_time(baseline_bot, baseline_t)
    gt_top, gt_t = trim_rows_to_time(gt_top, gt_t)
    gt_bot, _ = trim_rows_to_time(gt_bot, gt_t)

    shared_end = min(baseline_t[-1], gt_t[-1])
    baseline_mask = baseline_t <= shared_end

    baseline_t = baseline_t[baseline_mask]
    baseline_top = baseline_top[baseline_mask, :]
    baseline_bot = baseline_bot[baseline_mask, :]

    gt_mask = gt_t <= shared_end
    gt_t_shared = gt_t[gt_mask]
    gt_top_shared = gt_top[gt_mask, :]
    gt_bot_shared = gt_bot[gt_mask, :]

    gt_top_aligned = align_gt_to_baseline_times(baseline_t, gt_t_shared, gt_top_shared, args.alignment)
    gt_bot_aligned = align_gt_to_baseline_times(baseline_t, gt_t_shared, gt_bot_shared, args.alignment)

    baseline_top, gt_top_aligned = clip_to_common_columns(baseline_top, gt_top_aligned, "top_disp")
    baseline_bot, gt_bot_aligned = clip_to_common_columns(baseline_bot, gt_bot_aligned, "bot_disp")

    rel_top = relative_l2_per_timestep(baseline_top, gt_top_aligned)
    rel_bot = relative_l2_per_timestep(baseline_bot, gt_bot_aligned)

    baseline_combined = np.concatenate([baseline_top, baseline_bot], axis=1)
    gt_combined = np.concatenate([gt_top_aligned, gt_bot_aligned], axis=1)
    rel_combined = relative_l2_per_timestep(baseline_combined, gt_combined)

    avg_top, final_top, n_top = summarize(rel_top)
    avg_bot, final_bot, n_bot = summarize(rel_bot)
    avg_combined, final_combined, n_combined = summarize(rel_combined)

    print("Relative L2 error: r(t) = ||u_baseline(t) - u_gt(t)||_2 / ||u_gt(t)||_2")
    print(f"Alignment mode: {args.alignment}")
    print(f"Shared timesteps considered: {len(baseline_t)}")
    print(f"Final shared time: {baseline_t[-1]:.10e} s")
    print("")

    print("Component      Average over timesteps             Final timestep")
    print("-----------------------------------------------------------------------")
    print(f"top_disp       {avg_top: .10e} ({avg_top*100:8.5f}%)    {final_top: .10e} ({final_top*100:8.5f}%)")
    print(f"bot_disp       {avg_bot: .10e} ({avg_bot*100:8.5f}%)    {final_bot: .10e} ({final_bot*100:8.5f}%)")
    print(
        f"combined       {avg_combined: .10e} ({avg_combined*100:8.5f}%)    "
        f"{final_combined: .10e} ({final_combined*100:8.5f}%)"
    )
    print("")
    print(f"Valid timestep count used in averages: top={n_top}, bot={n_bot}, combined={n_combined}")


if __name__ == "__main__":
    main()
