#!/usr/bin/env python3
"""Combine per-sweep ranking files into category-level summaries.

This utility scans sweep output folders, reads each per-sweep ranking file,
and writes two combined reports:

- one for translation-aligned / cross-correlated sweeps
- one for the remaining sweeps

The combined reports are tab-separated text files so they are easy to inspect
manually and easy to post-process.
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


DEFAULT_SWEEP_ROOT = Path("/Users/joshmcneely/uguca/build/simulations/sweep_outputs")
DEFAULT_OUTPUT_DIR = DEFAULT_SWEEP_ROOT / "combined_rankings"
DEFAULT_RANKING_NAME = "sweep_rmse_ranking_all.txt"


@dataclass(frozen=True)
class RankingRecord:
    sweep_name: str
    category: str
    rank: int
    rmse_percent: float
    best_shift_m: str
    run_label: str
    output_dir: str
    source_file: str


@dataclass(frozen=True)
class RankingFile:
    path: Path
    sweep_name: str
    translation_aligned: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Combine all per-sweep ranking files into category summaries.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--sweep-root",
        type=Path,
        default=DEFAULT_SWEEP_ROOT,
        help="Root directory containing the sweep output folders",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where the combined ranking files will be written",
    )
    parser.add_argument(
        "--ranking-name",
        type=str,
        default=DEFAULT_RANKING_NAME,
        help="Ranking filename to collect from each sweep",
    )
    return parser.parse_args()


def _has_translation_alignment_marker(path: Path) -> bool:
    try:
        lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    except OSError:
        return False

    for line in lines[:12]:
        upper = line.upper()
        if "TRANSLATION-ALIGNED" in upper or "BEST_SHIFT_M" in upper:
            return True
    return False


def discover_ranking_files(sweep_root: Path, ranking_name: str) -> list[RankingFile]:
    ranking_files: list[RankingFile] = []
    if not sweep_root.exists():
        return ranking_files

    for sweep_dir in sorted(p for p in sweep_root.iterdir() if p.is_dir()):
        ranking_path = sweep_dir / "comparison_plots" / ranking_name
        if not ranking_path.exists():
            continue
        ranking_files.append(
            RankingFile(
                path=ranking_path,
                sweep_name=sweep_dir.name,
                translation_aligned=_has_translation_alignment_marker(ranking_path),
            )
        )
    return ranking_files


def _parse_row(parts: list[str], translation_aligned: bool) -> tuple[int, float, str, str, str]:
    if translation_aligned:
        if len(parts) < 5:
            raise ValueError("translation-aligned ranking row is missing columns")
        rank = int(parts[0])
        rmse_percent = float(parts[1])
        best_shift_m = parts[2]
        run_label = parts[3]
        output_dir = parts[4]
        return rank, rmse_percent, best_shift_m, run_label, output_dir

    if len(parts) < 4:
        raise ValueError("baseline ranking row is missing columns")
    rank = int(parts[0])
    rmse_percent = float(parts[1])
    best_shift_m = ""
    run_label = parts[2]
    output_dir = parts[3]
    return rank, rmse_percent, best_shift_m, run_label, output_dir


def parse_ranking_file(ranking_file: RankingFile) -> list[RankingRecord]:
    records: list[RankingRecord] = []
    lines = ranking_file.path.read_text(encoding="utf-8", errors="ignore").splitlines()
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

        rank, rmse_percent, best_shift_m, run_label, output_dir = _parse_row(parts, ranking_file.translation_aligned)
        category = "translation_aligned" if ranking_file.translation_aligned else "baseline"
        records.append(
            RankingRecord(
                sweep_name=ranking_file.sweep_name,
                category=category,
                rank=rank,
                rmse_percent=rmse_percent,
                best_shift_m=best_shift_m,
                run_label=run_label,
                output_dir=output_dir,
                source_file=str(ranking_file.path),
            )
        )
    return records


def _sort_key(record: RankingRecord) -> tuple[str, int, float, str]:
    return record.sweep_name, record.rank, record.rmse_percent, record.run_label


def write_combined_file(output_path: Path, records: Iterable[RankingRecord], category: str) -> None:
    records = sorted((record for record in records if record.category == category), key=_sort_key)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as stream:
        stream.write("# Combined sweep rankings\n")
        stream.write(f"# category: {category}\n")
        stream.write("# columns: sweep_name\trank\trmse_percent\tbest_shift_m\trun_label\toutput_dir\tsource_file\n" if category == "translation_aligned" else "# columns: sweep_name\trank\trmse_percent\trun_label\toutput_dir\tsource_file\n")
        if category == "translation_aligned":
            stream.write("sweep_name\trank\trmse_percent\tbest_shift_m\trun_label\toutput_dir\tsource_file\n")
            for record in records:
                stream.write(
                    f"{record.sweep_name}\t{record.rank}\t{record.rmse_percent:.5f}\t"
                    f"{record.best_shift_m}\t{record.run_label}\t{record.output_dir}\t{record.source_file}\n"
                )
        else:
            stream.write("sweep_name\trank\trmse_percent\trun_label\toutput_dir\tsource_file\n")
            for record in records:
                stream.write(
                    f"{record.sweep_name}\t{record.rank}\t{record.rmse_percent:.5f}\t"
                    f"{record.run_label}\t{record.output_dir}\t{record.source_file}\n"
                )


def main() -> None:
    args = parse_args()
    ranking_files = discover_ranking_files(args.sweep_root, args.ranking_name)
    if not ranking_files:
        raise SystemExit(f"No ranking files found under {args.sweep_root}")

    all_records: list[RankingRecord] = []
    for ranking_file in ranking_files:
        try:
            all_records.extend(parse_ranking_file(ranking_file))
        except Exception as exc:
            print(f"[WARNING] Skipping {ranking_file.path}: {exc}")

    if not all_records:
        raise SystemExit("No ranking rows could be parsed.")

    translation_output = args.output_dir / "combined_cross_correlated_rankings.txt"
    baseline_output = args.output_dir / "combined_other_rankings.txt"

    write_combined_file(translation_output, all_records, "translation_aligned")
    write_combined_file(baseline_output, all_records, "baseline")

    translation_count = sum(1 for record in all_records if record.category == "translation_aligned")
    baseline_count = sum(1 for record in all_records if record.category == "baseline")

    print(f"Wrote {translation_output} ({translation_count} rows)")
    print(f"Wrote {baseline_output} ({baseline_count} rows)")


if __name__ == "__main__":
    main()
