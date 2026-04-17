#!/usr/bin/env python3
"""Recompute translation-aligned (cross-correlated) RMSE for all available outputs.

This script consumes all rows from both combined ranking sources (cross-correlated
and other), recomputes translation-aligned RMSE for every unique output directory,
and rewrites:

- combined_cross_correlated_rankings.txt (readable table)
- combined_cross_correlated_rankings.tsv (machine-readable)
"""

from __future__ import annotations

import argparse
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path


DEFAULT_COMBINED_DIR = Path("/Users/joshmcneely/uguca/build/simulations/sweep_outputs/combined_rankings")
DEFAULT_SWEEP_ROOT = Path("/Users/joshmcneely/uguca/build/simulations/sweep_outputs")
DEFAULT_RANKING_NAME = "sweep_rmse_ranking_all.txt"
DEFAULT_EXP_BASE = Path("/Users/joshmcneely/introsims/simulation_outputs")
DEFAULT_SCRIPT = Path("/Users/joshmcneely/uguca/build/simulations/automation/ranking/compute_baseline_vs_exp_rmse_translation_aligned.py")

MEAN_RE = re.compile(r"Average relative error .*?\(([-+0-9.eE]+)%\)")
SHIFT_RE = re.compile(r"Best shift found \(m\)\s*:\s*([-+0-9.eE]+)")


@dataclass(frozen=True)
class CaseRecord:
    sweep_name: str
    run_label: str
    output_dir: str


@dataclass(frozen=True)
class RecomputedRecord:
    overall_rank: int
    rmse_percent: float
    best_shift_m: float
    sweep_name: str
    run_label: str
    output_dir: str


def detect_sim_name(output_dir: str, fallback: str) -> str:
    path = Path(output_dir)
    if not path.exists() or not path.is_dir():
        return fallback

    candidates = sorted(path.glob("*-interface.time"))
    if candidates:
        return candidates[0].name.replace("-interface.time", "")

    info_candidates = sorted(path.glob("*-interface.info"))
    if info_candidates:
        return info_candidates[0].name.replace("-interface.info", "")

    return fallback


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Recompute cross-correlated RMSE for all available combined outputs")
    p.add_argument("--combined-dir", type=Path, default=DEFAULT_COMBINED_DIR)
    p.add_argument("--sweep-root", type=Path, default=DEFAULT_SWEEP_ROOT)
    p.add_argument("--ranking-name", type=str, default=DEFAULT_RANKING_NAME)
    p.add_argument("--exp-base-dir", type=Path, default=DEFAULT_EXP_BASE)
    p.add_argument("--exp-restart-dir", type=str, default="interface-restart")
    p.add_argument("--nb-nodes", type=int, default=512)
    p.add_argument("--domain-length", type=float, default=6.0)
    p.add_argument("--shift-min", type=float, default=-0.5)
    p.add_argument("--shift-max", type=float, default=0.5)
    p.add_argument("--shift-step", type=float, default=0.01)
    return p.parse_args()


def _parse_per_sweep_ranking(path: Path, sweep_name: str) -> list[CaseRecord]:
    rows: list[CaseRecord] = []
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("=") or stripped.startswith("Rank ") or stripped.startswith("-----"):
            continue
        if "|" not in stripped:
            continue
        parts = [part.strip() for part in stripped.split("|")]
        parts = [part for part in parts if part]
        if not parts or not parts[0].isdigit():
            continue

        # translation-aligned rows: rank|rmse|best_shift|run_label|output_dir
        # baseline rows: rank|rmse|run_label|output_dir
        if len(parts) >= 5:
            run_label = parts[3]
            output_dir = parts[4]
        elif len(parts) >= 4:
            run_label = parts[2]
            output_dir = parts[3]
        else:
            continue

        if run_label and output_dir:
            rows.append(CaseRecord(sweep_name=sweep_name, run_label=run_label, output_dir=output_dir))
    return rows


def collect_cases(sweep_root: Path, ranking_name: str) -> list[CaseRecord]:
    seen: set[tuple[str, str]] = set()
    out: list[CaseRecord] = []

    for sweep_dir in sorted(p for p in sweep_root.iterdir() if p.is_dir()):
        if sweep_dir.name == "combined_rankings":
            continue
        ranking_path = sweep_dir / "comparison_plots" / ranking_name
        if not ranking_path.exists():
            continue
        for row in _parse_per_sweep_ranking(ranking_path, sweep_dir.name):
            key = (row.run_label, row.output_dir)
            if key in seen:
                continue
            seen.add(key)
            out.append(row)

    return sorted(out, key=lambda r: (r.sweep_name, r.run_label, r.output_dir))


def recompute_one(case: CaseRecord, args: argparse.Namespace) -> tuple[float, float]:
    sim_name = detect_sim_name(case.output_dir, case.run_label)
    cmd = [
        "python3",
        str(DEFAULT_SCRIPT),
        "--baseline-dir",
        case.output_dir,
        "--baseline-sim-name",
        sim_name,
        "--exp-base-dir",
        str(args.exp_base_dir),
        "--exp-restart-dir",
        args.exp_restart_dir,
        "--nb-nodes",
        str(args.nb_nodes),
        "--domain-length",
        str(args.domain_length),
        "--shift-min",
        str(args.shift_min),
        "--shift-max",
        str(args.shift_max),
        "--shift-step",
        str(args.shift_step),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    output = (proc.stdout or "") + "\n" + (proc.stderr or "")
    if proc.returncode != 0:
        raise RuntimeError(f"non-zero exit {proc.returncode}")

    m_mean = MEAN_RE.search(output)
    m_shift = SHIFT_RE.search(output)
    if not m_mean or not m_shift:
        raise RuntimeError("could not parse RMSE/shift")

    rmse_percent = float(m_mean.group(1))
    shift_m = float(m_shift.group(1))
    return rmse_percent, shift_m


def write_outputs(combined_dir: Path, rows: list[RecomputedRecord]) -> None:
    txt_path = combined_dir / "combined_cross_correlated_rankings.txt"
    tsv_path = combined_dir / "combined_cross_correlated_rankings.tsv"

    with tsv_path.open("w", encoding="utf-8") as f:
        f.write("# Recomputed translation-aligned combined rankings across all available outputs\n")
        f.write("overall_rank\trmse_percent\tbest_shift_m\tsweep_name\trun_label\toutput_dir\n")
        for r in rows:
            f.write(
                f"{r.overall_rank}\t{r.rmse_percent:.5f}\t{r.best_shift_m:+.6f}\t"
                f"{r.sweep_name}\t{r.run_label}\t{r.output_dir}\n"
            )

    headers = ["#", "RMSE(%)", "Shift(m)", "Sweep", "Run"]
    body = [
        [str(r.overall_rank), f"{r.rmse_percent:.5f}", f"{r.best_shift_m:+.6f}", r.sweep_name, r.run_label]
        for r in rows
    ]
    widths = [len(h) for h in headers]
    for row in body:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))

    divider = "-+-".join("-" * w for w in widths)
    with txt_path.open("w", encoding="utf-8") as f:
        f.write("Combined cross-correlated rankings (recomputed for all available outputs)\n")
        f.write(f"Rows: {len(rows)}\n")
        f.write(f"Machine-readable TSV: {tsv_path.name}\n\n")
        f.write(" | ".join(headers[i].ljust(widths[i]) for i in range(len(headers))) + "\n")
        f.write(divider + "\n")
        for row in body:
            f.write(" | ".join(row[i].ljust(widths[i]) for i in range(len(row))) + "\n")


def main() -> None:
    args = parse_args()
    combined_dir = args.combined_dir
    combined_dir.mkdir(parents=True, exist_ok=True)

    cases = collect_cases(args.sweep_root, args.ranking_name)
    if not cases:
        raise SystemExit(f"No per-sweep ranking rows found under {args.sweep_root}")

    print(f"Collected {len(cases)} unique output cases")
    recomputed: list[tuple[float, float, CaseRecord]] = []
    failed: list[tuple[CaseRecord, str]] = []

    total = len(cases)
    for idx, case in enumerate(cases, start=1):
        if idx == 1 or idx % 25 == 0 or idx == total:
            print(f"[{idx}/{total}] recomputing: {case.sweep_name} :: {case.run_label}")
        try:
            rmse_percent, shift_m = recompute_one(case, args)
            recomputed.append((rmse_percent, shift_m, case))
        except Exception as exc:
            failed.append((case, str(exc)))

    recomputed.sort(key=lambda x: (x[0], x[2].sweep_name, x[2].run_label))
    ranked = [
        RecomputedRecord(
            overall_rank=i,
            rmse_percent=rmse,
            best_shift_m=shift,
            sweep_name=case.sweep_name,
            run_label=case.run_label,
            output_dir=case.output_dir,
        )
        for i, (rmse, shift, case) in enumerate(recomputed, start=1)
    ]

    write_outputs(combined_dir, ranked)

    print(f"Wrote {combined_dir / 'combined_cross_correlated_rankings.txt'} ({len(ranked)} rows)")
    print(f"Wrote {combined_dir / 'combined_cross_correlated_rankings.tsv'} ({len(ranked)} rows)")
    if failed:
        fail_path = combined_dir / "combined_cross_correlated_recompute_failures.txt"
        with fail_path.open("w", encoding="utf-8") as f:
            f.write("sweep_name\trun_label\toutput_dir\terror\n")
            for case, err in failed:
                f.write(f"{case.sweep_name}\t{case.run_label}\t{case.output_dir}\t{err}\n")
        print(f"Failures: {len(failed)} (details: {fail_path})")
    else:
        print("Failures: 0")


if __name__ == "__main__":
    main()
