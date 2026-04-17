#!/usr/bin/env python3
"""Generate log-colorbar spacetime plots for top-N cross-correlated ranking entries."""

from __future__ import annotations

import argparse
import csv
import subprocess
from pathlib import Path


DEFAULT_RANKING_TSV = Path(
    "/Users/joshmcneely/uguca/build/simulations/sweep_outputs/combined_rankings/combined_cross_correlated_rankings.tsv"
)
DEFAULT_PLOTTER = Path(
    "/Users/joshmcneely/uguca/build/simulations/automation/plotting/Plot_BaselineExp_Spacetime_LogColorbar.py"
)
DEFAULT_OUTPUT_DIR = Path(
    "/Users/joshmcneely/uguca/build/simulations/sweep_outputs/combined_rankings/top50_cross_correlated_log_spacetime"
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Plot top cross-correlated cases using the log-colorbar spacetime plotter",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--ranking-tsv", type=Path, default=DEFAULT_RANKING_TSV)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--plotter", type=Path, default=DEFAULT_PLOTTER)
    p.add_argument("--top-n", type=int, default=50)
    p.add_argument("--exp-base-dir", type=Path, default=Path("/Users/joshmcneely/introsims/simulation_outputs"))
    p.add_argument("--exp-restart-dir", type=Path, default=Path("/Users/joshmcneely/introsims/simulation_outputs/interface-restart"))
    return p.parse_args()


def read_top_rows(ranking_tsv: Path, top_n: int) -> list[dict[str, str]]:
    if not ranking_tsv.exists():
        raise FileNotFoundError(f"Ranking file not found: {ranking_tsv}")

    rows: list[dict[str, str]] = []
    with ranking_tsv.open("r", encoding="utf-8", errors="ignore") as f:
        header = None
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            header = s.split("\t")
            break
        if header is None:
            return rows

        reader = csv.reader(f, delimiter="\t")
        for parts in reader:
            if not parts:
                continue
            if len(parts) < len(header):
                parts += [""] * (len(header) - len(parts))
            rows.append({header[i]: parts[i].strip() for i in range(len(header))})
            if len(rows) >= top_n:
                break

    return rows


def detect_sim_name(case_dir: Path) -> str:
    time_files = sorted(case_dir.glob("*-interface.time"))
    if time_files:
        return time_files[0].name.replace("-interface.time", "")

    info_files = sorted(case_dir.glob("*-interface.info"))
    if info_files:
        return info_files[0].name.replace("-interface.info", "")

    return "local_baseline_run"


def run_plotter(row: dict[str, str], args: argparse.Namespace, idx: int) -> None:
    case_dir = Path(row["output_dir"])
    sim_name = detect_sim_name(case_dir)
    case_label = f"rank{idx:03d}_{row['sweep_name']}::{row['run_label']}"
    output_tag = f"rank{idx:03d}_{row['sweep_name']}__{row['run_label']}"

    cmd = [
        "python3",
        str(args.plotter),
        "--baseline-dir",
        str(case_dir),
        "--baseline-sim-name",
        sim_name,
        "--case-label",
        case_label,
        "--output-tag",
        output_tag,
        "--plot-dir",
        str(args.output_dir),
        "--exp-base-dir",
        str(args.exp_base_dir),
        "--exp-restart-dir",
        str(args.exp_restart_dir),
    ]

    subprocess.run(cmd, check=True)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows = read_top_rows(args.ranking_tsv, args.top_n)
    if not rows:
        raise SystemExit("No rows found in ranking TSV")

    summary_path = args.output_dir / "README.txt"
    with summary_path.open("w", encoding="utf-8") as f:
        f.write("Top cross-correlated log-colorbar spacetime plots\n")
        f.write(f"Ranking source: {args.ranking_tsv}\n")
        f.write(f"Top N: {len(rows)}\n")
        f.write("Translated plots enabled: False (disabled)\n\n")
        f.write("overall_rank\trmse_percent\tbest_shift_m\tsweep_name\trun_label\toutput_dir\n")
        for row in rows:
            f.write(
                f"{row.get('overall_rank','')}\t{row.get('rmse_percent','')}\t{row.get('best_shift_m','')}\t"
                f"{row.get('sweep_name','')}\t{row.get('run_label','')}\t{row.get('output_dir','')}\n"
            )

    total = len(rows)
    for i, row in enumerate(rows, start=1):
        print(f"[{i}/{total}] plotting {row.get('sweep_name','')} :: {row.get('run_label','')}")
        run_plotter(row, args, i)

    print(f"[DONE] Generated log-colorbar spacetime plots in: {args.output_dir}")


if __name__ == "__main__":
    main()
