#!/usr/bin/env python3
"""
User-configurable parameter sweep runner for uguca baseline input files.

Design goals:
- User friendly: interactive wizard can build config files for you.
- Idiot proof: strong validation, clear error messages, dry-run preview.
- Flexible: choose any template parameters to vary with linear/log/list ranges.
- Safe: total-run guardrail, confirmation prompt, per-case input snapshots.
- Quiet terminal: no timestep spam; one progress line per run.

Typical usage:
1) Create a config (interactive wizard):
   python3 automation/sweeps/sweep_configurable.py --create-config automation/sweeps/my_sweep.json

2) Validate without running:
   python3 automation/sweeps/sweep_configurable.py --config automation/sweeps/my_sweep.json --dry-run

3) Run:
   python3 automation/sweeps/sweep_configurable.py --config automation/sweeps/my_sweep.json --run
"""

from __future__ import annotations

import argparse
import json
import math
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from itertools import product
from pathlib import Path
from typing import Any


DEFAULT_WORK_DIR = Path("/Users/joshmcneely/uguca/build/simulations")
DEFAULT_TEMPLATE = Path("/Users/joshmcneely/uguca/simulations/input_files/local_baseline_compare_nu019.in")
DEFAULT_EXP_RESTART = Path("/Users/joshmcneely/introsims/simulation_outputs/interface-restart")

PARAM_LINE_RE = re.compile(r"^(?P<indent>\s*)(?P<key>[A-Za-z_][A-Za-z0-9_]*)\s*=\s*(?P<value>[^#\n\r]*?)(?P<trailing>\s*(?:#.*)?)$")


@dataclass
class SweepParam:
    name: str
    spacing: str
    mode: str
    values: list[float]


class ConfigError(Exception):
    pass


def _now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _fmt_float(x: float) -> str:
    return f"{x:.12g}"


def _slug_text(s: str) -> str:
    out = s.lower().strip()
    out = re.sub(r"[^a-z0-9_]+", "_", out)
    out = re.sub(r"_+", "_", out).strip("_")
    return out if out else "value"


def _slug_num(s: str) -> str:
    out = s.lower().strip()
    out = out.replace("-", "m").replace("+", "p").replace(".", "p")
    out = out.replace("e-", "expm").replace("e+", "expp").replace("e", "exp")
    out = re.sub(r"[^a-z0-9_]+", "_", out)
    out = re.sub(r"_+", "_", out).strip("_")
    return out if out else "value"


def _run_with_log(cmd: list[str], cwd: Path, log_path: Path, quiet: bool) -> int:
    if quiet:
        with log_path.open("w", encoding="utf-8") as log:
            return subprocess.run(
                cmd,
                cwd=str(cwd),
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
            ).returncode

    with log_path.open("w", encoding="utf-8") as log:
        proc = subprocess.Popen(
            cmd,
            cwd=str(cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="")
            log.write(line)
        return proc.wait()


def _read_template_lines(template_path: Path) -> list[str]:
    if not template_path.exists():
        raise ConfigError(f"Template file not found: {template_path}")
    return template_path.read_text(encoding="utf-8").splitlines(keepends=True)


def _parse_template_params(lines: list[str]) -> dict[str, str]:
    params: dict[str, str] = {}
    for line in lines:
        m = PARAM_LINE_RE.match(line)
        if not m:
            continue
        key = m.group("key").strip()
        value = m.group("value").strip()
        params[key] = value
    return params


def _parse_float_or_none(raw: str) -> float | None:
    try:
        return float(raw)
    except Exception:
        return None


def _replace_params_in_lines(lines: list[str], replacements: dict[str, str]) -> list[str]:
    remaining = set(replacements.keys())
    out_lines: list[str] = []

    for line in lines:
        m = PARAM_LINE_RE.match(line)
        if not m:
            out_lines.append(line)
            continue

        key = m.group("key")
        if key not in replacements:
            out_lines.append(line)
            continue

        new_val = replacements[key]
        indent = m.group("indent")
        trailing = m.group("trailing") or ""
        out_lines.append(f"{indent}{key} = {new_val}{trailing}\n")
        remaining.discard(key)

    if remaining:
        missing = ", ".join(sorted(remaining))
        raise ConfigError(f"Template does not contain parameter(s): {missing}")

    return out_lines


def _build_values(spacing: str, min_v: float, max_v: float, count: int) -> list[float]:
    if count < 1:
        raise ConfigError("count must be >= 1")
    if count == 1:
        return [min_v]

    if spacing == "linear":
        step = (max_v - min_v) / (count - 1)
        return [min_v + i * step for i in range(count)]

    if spacing == "log":
        if min_v <= 0.0 or max_v <= 0.0:
            raise ConfigError("log spacing requires min and max > 0")
        la = math.log(min_v)
        lb = math.log(max_v)
        step = (lb - la) / (count - 1)
        return [math.exp(la + i * step) for i in range(count)]

    raise ConfigError(f"Unsupported spacing: {spacing}")


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _prompt(msg: str, default: str | None = None) -> str:
    if default is None:
        suffix = ": "
    else:
        suffix = f" [{default}]: "
    value = input(msg + suffix).strip()
    return value if value else (default or "")


def _prompt_yes_no(msg: str, default: bool = True) -> bool:
    d = "y" if default else "n"
    while True:
        ans = _prompt(msg, d).lower()
        if ans in {"y", "yes"}:
            return True
        if ans in {"n", "no"}:
            return False
        print("Please enter y or n.")


def _prompt_int(msg: str, default: int, min_v: int = 1) -> int:
    while True:
        raw = _prompt(msg, str(default))
        try:
            v = int(raw)
        except ValueError:
            print("Please enter a valid integer.")
            continue
        if v < min_v:
            print(f"Value must be >= {min_v}.")
            continue
        return v


def _prompt_float(msg: str, default: float | None = None, positive: bool = False) -> float:
    while True:
        raw = _prompt(msg, _fmt_float(default) if default is not None else None)
        try:
            v = float(raw)
        except ValueError:
            print("Please enter a valid number.")
            continue
        if positive and v <= 0:
            print("Value must be > 0.")
            continue
        return v


def _interactive_create_config(config_path: Path) -> None:
    print("\n=== Configurable Sweep Wizard ===")
    print("This wizard creates a JSON config. You can edit it later if needed.\n")

    work_dir = Path(_prompt("Simulation working directory", str(DEFAULT_WORK_DIR))).expanduser().resolve()
    template = Path(_prompt("Template input file", str(DEFAULT_TEMPLATE))).expanduser().resolve()
    lines = _read_template_lines(template)
    params = _parse_template_params(lines)

    if not params:
        raise ConfigError("No key=value parameters found in template input file.")

    numeric_params = {k: float(v) for k, v in params.items() if _parse_float_or_none(v) is not None}
    print("\nDetected numeric parameters:")
    print(", ".join(sorted(numeric_params.keys())))

    sweep_name = _slug_text(_prompt("Sweep name", "configurable_sweep"))
    mode = _prompt("Mode", "all").lower()
    if mode not in {"all", "smoke"}:
        mode = "all"

    max_runs = _prompt_int("Maximum allowed runs before confirmation fails", 200, min_v=1)
    quiet_solver = _prompt_yes_no("Suppress solver timestep output in terminal?", default=True)
    confirm_before_run = _prompt_yes_no("Require explicit confirmation before run?", default=True)

    print("\nChoose parameters to vary.")
    print("Enter names separated by commas, e.g. nuc_dtau,nuc_size")
    while True:
        raw_names = _prompt("Parameters to vary")
        vary_names = [n.strip() for n in raw_names.split(",") if n.strip()]
        if not vary_names:
            print("Please provide at least one parameter.")
            continue
        missing = [n for n in vary_names if n not in numeric_params]
        if missing:
            print(f"Unknown or non-numeric parameter(s): {', '.join(missing)}")
            continue
        break

    vary_specs: list[dict[str, Any]] = []
    for name in vary_names:
        base = numeric_params[name]
        print(f"\nParameter: {name} (base={_fmt_float(base)})")
        spacing = _prompt("Spacing type (linear/log/list)", "linear").lower()
        if spacing not in {"linear", "log", "list"}:
            spacing = "linear"

        mode_choice = _prompt("Range mode (scale/absolute)", "scale").lower()
        if mode_choice not in {"scale", "absolute"}:
            mode_choice = "scale"

        if spacing in {"linear", "log"}:
            min_default = 0.8 if mode_choice == "scale" else base * 0.8
            max_default = 1.2 if mode_choice == "scale" else base * 1.2
            if spacing == "log" and min_default <= 0:
                min_default = 0.1
                max_default = 10.0
            min_v = _prompt_float("Range min", min_default, positive=(spacing == "log"))
            max_v = _prompt_float("Range max", max_default, positive=(spacing == "log"))
            while max_v < min_v:
                print("max must be >= min")
                max_v = _prompt_float("Range max", max_default, positive=(spacing == "log"))
            count = _prompt_int("Number of points", 5, min_v=1)
            vary_specs.append(
                {
                    "name": name,
                    "spacing": spacing,
                    "mode": mode_choice,
                    "min": min_v,
                    "max": max_v,
                    "count": count,
                }
            )
        else:
            hint = "0.8,1.0,1.2" if mode_choice == "scale" else f"{_fmt_float(base*0.8)},{_fmt_float(base)},{_fmt_float(base*1.2)}"
            while True:
                raw = _prompt("List values (comma-separated)", hint)
                parts = [x.strip() for x in raw.split(",") if x.strip()]
                try:
                    values = [float(x) for x in parts]
                except ValueError:
                    print("Please provide only numeric values.")
                    continue
                if not values:
                    print("Please provide at least one value.")
                    continue
                vary_specs.append(
                    {
                        "name": name,
                        "spacing": "list",
                        "mode": mode_choice,
                        "values": values,
                    }
                )
                break

    print("\nOptional fixed overrides (set parameters that should be forced to specific values).")
    print("Enter key=value pairs separated by commas, or leave blank.")
    fixed: dict[str, Any] = {}
    raw_fixed = _prompt("Fixed overrides", "")
    if raw_fixed:
        for pair in raw_fixed.split(","):
            pair = pair.strip()
            if not pair:
                continue
            if "=" not in pair:
                raise ConfigError(f"Invalid fixed override (missing '='): {pair}")
            k, v = pair.split("=", 1)
            key = k.strip()
            val = v.strip()
            if key not in params:
                raise ConfigError(f"Unknown parameter in fixed overrides: {key}")
            parsed = _parse_float_or_none(val)
            fixed[key] = parsed if parsed is not None else val

    enable_plot = _prompt_yes_no("Enable spacetime plotting for each run?", default=True)
    enable_rmse = _prompt_yes_no("Enable RMSE computation for each run?", default=True)
    enable_translated_plot = _prompt_yes_no("Also write translated spacetime plots when ranking_mode is translation_aligned?", default=False)
    ranking_mode = _prompt("Ranking mode (baseline_vs_exp/translation_aligned)", "baseline_vs_exp").lower()
    if ranking_mode not in {"baseline_vs_exp", "translation_aligned"}:
        ranking_mode = "baseline_vs_exp"

    config = {
        "meta": {
            "sweep_name": sweep_name,
            "mode": mode,
            "max_total_runs": max_runs,
            "confirm_before_run": confirm_before_run,
            "quiet_solver_output": quiet_solver,
            "ranking_mode": ranking_mode,
        },
        "paths": {
            "work_dir": str(work_dir),
            "solver_exec": "./dd_earthquake",
            "template_input": str(template),
            "output_root": f"./sweep_outputs/{sweep_name}",
            "plot_script": "./automation/plotting/Plot_BaselineExp_Spacetime.py",
            "rmse_script": "./automation/ranking/compute_baseline_vs_exp_rmse.py",
            "rmse_script_translation_aligned": "./automation/ranking/compute_baseline_vs_exp_rmse_translation_aligned.py",
        },
        "analysis": {
            "enable_plot": enable_plot,
            "enable_rmse": enable_rmse,
            "enable_translated_plot": enable_translated_plot,
            "exp_restart_dir": str(DEFAULT_EXP_RESTART),
            "roi_min": 0.0,
            "roi_max": 3.05,
        },
        "vary": vary_specs,
        "fixed": fixed,
    }

    _ensure_parent(config_path)
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(f"\nWrote config: {config_path}")


def _load_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise ConfigError(f"Config file not found: {path}")
    try:
        cfg = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ConfigError(f"Invalid JSON config: {exc}") from exc
    if not isinstance(cfg, dict):
        raise ConfigError("Top-level config must be a JSON object")
    return cfg


def _validated_path(base: Path, value: str) -> Path:
    p = Path(value).expanduser()
    return p if p.is_absolute() else (base / p)


def _upgrade_legacy_script_path(work_dir: Path, current: Path, legacy_rel: str, canonical_rel: str) -> Path:
    legacy = _validated_path(work_dir, legacy_rel)
    canonical = _validated_path(work_dir, canonical_rel)
    if current == legacy and (not legacy.exists()) and canonical.exists():
        return canonical
    return current


def _validate_and_expand(cfg: dict[str, Any]) -> dict[str, Any]:
    meta = cfg.get("meta", {})
    paths = cfg.get("paths", {})
    analysis = cfg.get("analysis", {})
    vary_raw = cfg.get("vary", [])
    fixed_raw = cfg.get("fixed", {})

    if not isinstance(vary_raw, list) or not vary_raw:
        raise ConfigError("Config field 'vary' must be a non-empty list")
    if not isinstance(fixed_raw, dict):
        raise ConfigError("Config field 'fixed' must be an object")

    work_dir = Path(paths.get("work_dir", str(DEFAULT_WORK_DIR))).expanduser().resolve()
    template = Path(paths.get("template_input", str(DEFAULT_TEMPLATE))).expanduser().resolve()
    solver_exec = _validated_path(work_dir, paths.get("solver_exec", "./dd_earthquake"))
    output_root = _validated_path(work_dir, paths.get("output_root", "./sweep_outputs/configurable_sweep"))
    plot_script = _validated_path(work_dir, paths.get("plot_script", "./automation/plotting/Plot_BaselineExp_Spacetime.py"))
    rmse_script = _validated_path(work_dir, paths.get("rmse_script", "./automation/ranking/compute_baseline_vs_exp_rmse.py"))
    rmse_script_translation_aligned = _validated_path(
        work_dir,
        paths.get(
            "rmse_script_translation_aligned",
            "./automation/ranking/compute_baseline_vs_exp_rmse_translation_aligned.py",
        ),
    )

    # Backward compatibility: older configs may still point at removed root wrappers.
    plot_script = _upgrade_legacy_script_path(
        work_dir,
        plot_script,
        "./Plot_BaselineExp_Spacetime.py",
        "./automation/plotting/Plot_BaselineExp_Spacetime.py",
    )
    rmse_script = _upgrade_legacy_script_path(
        work_dir,
        rmse_script,
        "./compute_baseline_vs_exp_rmse.py",
        "./automation/ranking/compute_baseline_vs_exp_rmse.py",
    )
    rmse_script_translation_aligned = _upgrade_legacy_script_path(
        work_dir,
        rmse_script_translation_aligned,
        "./compute_baseline_vs_exp_rmse_translation_aligned.py",
        "./automation/ranking/compute_baseline_vs_exp_rmse_translation_aligned.py",
    )

    if not work_dir.exists():
        raise ConfigError(f"work_dir does not exist: {work_dir}")
    if not template.exists():
        raise ConfigError(f"template_input does not exist: {template}")
    if not solver_exec.exists():
        raise ConfigError(f"solver_exec does not exist: {solver_exec}")

    lines = _read_template_lines(template)
    params = _parse_template_params(lines)
    if not params:
        raise ConfigError("No key=value parameters found in template input")

    sim_name = params.get("simulation_name", "local_baseline_run")
    nb_nodes = int(float(params.get("nb_elements", "512")))
    domain_length = float(params.get("length", "6.0"))

    sweep_name = _slug_text(str(meta.get("sweep_name", "configurable_sweep")))
    mode = str(meta.get("mode", "all")).lower()
    if mode not in {"all", "smoke"}:
        raise ConfigError("meta.mode must be 'all' or 'smoke'")
    ranking_mode = str(meta.get("ranking_mode", "baseline_vs_exp")).lower()
    if ranking_mode not in {"baseline_vs_exp", "translation_aligned"}:
        raise ConfigError("meta.ranking_mode must be baseline_vs_exp or translation_aligned")

    max_total_runs = int(meta.get("max_total_runs", 200))
    if max_total_runs < 1:
        raise ConfigError("meta.max_total_runs must be >= 1")

    quiet_solver_output = bool(meta.get("quiet_solver_output", True))
    confirm_before_run = bool(meta.get("confirm_before_run", True))

    exp_restart_dir = Path(analysis.get("exp_restart_dir", str(DEFAULT_EXP_RESTART))).expanduser().resolve()
    exp_base_dir = exp_restart_dir.parent
    exp_restart_name = exp_restart_dir.name
    enable_plot = bool(analysis.get("enable_plot", True))
    enable_rmse = bool(analysis.get("enable_rmse", True))
    enable_translated_plot = bool(analysis.get("enable_translated_plot", False))
    roi_min = float(analysis.get("roi_min", 0.0))
    roi_max = float(analysis.get("roi_max", 3.05))
    translation_shift_min_m = float(analysis.get("translation_shift_min_m", -0.5))
    translation_shift_max_m = float(analysis.get("translation_shift_max_m", 0.5))
    translation_shift_step_m = float(analysis.get("translation_shift_step_m", 0.01))

    sweep_params: list[SweepParam] = []
    for i, spec in enumerate(vary_raw, start=1):
        if not isinstance(spec, dict):
            raise ConfigError(f"vary[{i}] must be an object")
        name = str(spec.get("name", "")).strip()
        if not name:
            raise ConfigError(f"vary[{i}] missing 'name'")
        if name not in params:
            raise ConfigError(f"vary[{i}] unknown parameter: {name}")

        base_val = _parse_float_or_none(params[name])
        if base_val is None:
            raise ConfigError(f"vary[{i}] parameter is not numeric in template: {name}")

        spacing = str(spec.get("spacing", "linear")).lower()
        mode_choice = str(spec.get("mode", "scale")).lower()
        if spacing not in {"linear", "log", "list"}:
            raise ConfigError(f"vary[{i}] spacing must be linear/log/list")
        if mode_choice not in {"scale", "absolute"}:
            raise ConfigError(f"vary[{i}] mode must be scale/absolute")

        if spacing == "list":
            raw_values = spec.get("values", [])
            if not isinstance(raw_values, list) or not raw_values:
                raise ConfigError(f"vary[{i}] list spacing requires non-empty values[]")
            try:
                values = [float(v) for v in raw_values]
            except Exception as exc:
                raise ConfigError(f"vary[{i}] values must all be numeric") from exc
            if mode_choice == "scale":
                values = [base_val * v for v in values]
            if spacing == "log" and any(v <= 0 for v in values):
                raise ConfigError(f"vary[{i}] log values must be > 0")
        else:
            min_v = float(spec.get("min"))
            max_v = float(spec.get("max"))
            count = int(spec.get("count"))
            if max_v < min_v:
                raise ConfigError(f"vary[{i}] max must be >= min")
            values = _build_values(spacing, min_v, max_v, count)
            if mode_choice == "scale":
                values = [base_val * v for v in values]

        if spacing == "log" and any(v <= 0 for v in values):
            raise ConfigError(f"vary[{i}] produced non-positive values with log spacing")

        sweep_params.append(SweepParam(name=name, spacing=spacing, mode=mode_choice, values=values))

    fixed_values: dict[str, str] = {}
    for k, v in fixed_raw.items():
        key = str(k).strip()
        if key not in params:
            raise ConfigError(f"fixed contains unknown parameter: {key}")
        if isinstance(v, (int, float)):
            fixed_values[key] = _fmt_float(float(v))
        elif isinstance(v, str):
            fixed_values[key] = v
        else:
            raise ConfigError(f"fixed.{key} must be number or string")

    # Ensure no duplicate varying parameter names.
    names = [sp.name for sp in sweep_params]
    if len(set(names)) != len(names):
        raise ConfigError("vary contains duplicate parameter names")

    total_runs = 1
    for sp in sweep_params:
        total_runs *= len(sp.values)

    if mode == "smoke":
        total_runs = 1

    if total_runs > max_total_runs:
        raise ConfigError(
            f"Planned runs ({total_runs}) exceed meta.max_total_runs ({max_total_runs}). "
            f"Increase max_total_runs if this is intentional."
        )

    if enable_plot and not plot_script.exists():
        raise ConfigError(f"Plot script not found: {plot_script}")
    selected_rmse_script = rmse_script_translation_aligned if ranking_mode == "translation_aligned" else rmse_script
    if enable_rmse and not selected_rmse_script.exists():
        raise ConfigError(f"RMSE script not found: {selected_rmse_script}")
    if (enable_plot or enable_rmse) and not exp_restart_dir.exists():
        raise ConfigError(f"Experimental restart directory not found: {exp_restart_dir}")

    return {
        "meta": {
            "sweep_name": sweep_name,
            "mode": mode,
            "max_total_runs": max_total_runs,
            "quiet_solver_output": quiet_solver_output,
            "confirm_before_run": confirm_before_run,
            "ranking_mode": ranking_mode,
        },
        "paths": {
            "work_dir": work_dir,
            "template": template,
            "solver_exec": solver_exec,
            "output_root": output_root,
            "plot_script": plot_script,
            "rmse_script": rmse_script,
            "rmse_script_translation_aligned": rmse_script_translation_aligned,
        },
        "analysis": {
            "enable_plot": enable_plot,
            "enable_rmse": enable_rmse,
            "enable_translated_plot": enable_translated_plot,
            "exp_restart_dir": exp_restart_dir,
            "exp_base_dir": exp_base_dir,
            "exp_restart_name": exp_restart_name,
            "roi_min": roi_min,
            "roi_max": roi_max,
            "translation_shift_min_m": translation_shift_min_m,
            "translation_shift_max_m": translation_shift_max_m,
            "translation_shift_step_m": translation_shift_step_m,
        },
        "template": {
            "lines": lines,
            "params": params,
            "sim_name": sim_name,
            "nb_nodes": nb_nodes,
            "domain_length": domain_length,
        },
        "vary": sweep_params,
        "fixed": fixed_values,
        "total_runs": total_runs,
    }


def _preview_plan(state: dict[str, Any]) -> None:
    print("\n=== Sweep Plan ===")
    print(f"Name: {state['meta']['sweep_name']}")
    print(f"Mode: {state['meta']['mode']}")
    print(f"Working directory: {state['paths']['work_dir']}")
    print(f"Template: {state['paths']['template']}")
    print(f"Output root: {state['paths']['output_root']}")
    print(f"Ranking mode: {state['meta'].get('ranking_mode', 'baseline_vs_exp')}")
    print(f"Total runs: {state['total_runs']}")
    print("Varying parameters:")
    for sp in state["vary"]:
        sample = ", ".join(_fmt_float(v) for v in sp.values[:4])
        if len(sp.values) > 4:
            sample += ", ..."
        print(
            f"  - {sp.name}: {len(sp.values)} values "
            f"({sp.spacing}/{sp.mode}) sample=[{sample}]"
        )
    if state["fixed"]:
        print("Fixed overrides:")
        for k, v in sorted(state["fixed"].items()):
            print(f"  - {k} = {v}")


def _build_combinations(state: dict[str, Any]) -> list[dict[str, float]]:
    vary: list[SweepParam] = state["vary"]
    if state["meta"]["mode"] == "smoke":
        return [{sp.name: sp.values[0] for sp in vary}]

    keys = [sp.name for sp in vary]
    grids = [sp.values for sp in vary]

    combos: list[dict[str, float]] = []
    for values in product(*grids):
        combos.append(dict(zip(keys, values)))
    return combos


def _run_sweep(state: dict[str, Any], dry_run: bool) -> None:
    paths = state["paths"]
    analysis = state["analysis"]
    meta = state["meta"]
    tpl = state["template"]
    ranking_mode = meta.get("ranking_mode", "baseline_vs_exp")
    rmse_script = paths["rmse_script_translation_aligned"] if ranking_mode == "translation_aligned" else paths["rmse_script"]

    sweep_root: Path = paths["output_root"]
    cases_dir = sweep_root / "cases"
    plot_dir = sweep_root / "comparison_plots"
    sweep_root.mkdir(parents=True, exist_ok=True)
    cases_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)

    run_id = f"{meta['mode']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    tmp_input = paths["work_dir"] / f"tmp_configurable_sweep_{run_id}.in"
    summary_log = plot_dir / f"sweep_summary_{meta['mode']}.log"
    ranking_file = plot_dir / f"sweep_rmse_ranking_{meta['mode']}.txt"
    ranking_entries: list[tuple[float, str, str, str]] = []

    combos = _build_combinations(state)

    header = (
        "RUN_LABEL\tRUN_STATUS\tPLOT_STATUS\tRMSE_PERCENT\t"
        + "\t".join(sp.name for sp in state["vary"])
        + "\tOUTPUT_DIR\n"
    )
    if ranking_mode == "translation_aligned":
        header = (
            "RUN_LABEL\tRUN_STATUS\tPLOT_STATUS\tRMSE_PERCENT\tBEST_SHIFT_M\t"
            + "\t".join(sp.name for sp in state["vary"])
            + "\tOUTPUT_DIR\n"
        )
    summary_log.write_text(header, encoding="utf-8")

    print("\n=== Execution ===")
    print(f"Runs to execute: {len(combos)}")
    print(f"Summary log: {summary_log}")
    if dry_run:
        print("Dry-run mode: no solver commands will be executed.")

    try:
        for idx, combo in enumerate(combos, start=1):
            param_suffix = "_".join(f"{k}{_slug_num(_fmt_float(v))}" for k, v in combo.items())
            run_label = f"run{idx:03d}_{param_suffix}"
            case_dir = cases_dir / f"results_baseline_{run_label}"

            print(f"[{_now()}] RUN {idx}/{len(combos)}: {run_label}")

            if case_dir.exists():
                shutil.rmtree(case_dir)
            case_dir.mkdir(parents=True, exist_ok=True)

            replacements: dict[str, str] = {}
            # Varying values
            for k, v in combo.items():
                replacements[k] = _fmt_float(v)
            # Fixed overrides
            replacements.update(state["fixed"])
            # Always direct output into this case directory
            replacements["dump_folder"] = f"{case_dir}/"

            rendered_lines = _replace_params_in_lines(tpl["lines"], replacements)
            tmp_input.write_text("".join(rendered_lines), encoding="utf-8")
            (case_dir / "input_used.in").write_text("".join(rendered_lines), encoding="utf-8")

            run_status = "OK"
            plot_status = "SKIPPED"
            rmse_percent = "NA"
            best_shift_m = "NA"

            if not dry_run:
                solver_log = case_dir / "solver.log"
                rc = _run_with_log(
                    [str(paths["solver_exec"]), str(tmp_input)],
                    cwd=paths["work_dir"],
                    log_path=solver_log,
                    quiet=meta["quiet_solver_output"],
                )
                if rc != 0:
                    run_status = f"FAILED({rc})"
                else:
                    expected_time = case_dir / f"{tpl['sim_name']}-interface.time"
                    if not expected_time.exists() or expected_time.stat().st_size == 0:
                        run_status = "FAILED(no_output)"

                if run_status == "OK" and analysis["enable_plot"]:
                    plot_log = case_dir / "plot.log"
                    plot_args = [
                        "python3",
                        str(paths["plot_script"]),
                        "--baseline-dir",
                        str(case_dir),
                        "--case-label",
                        run_label,
                        "--output-tag",
                        run_label,
                        "--plot-dir",
                        str(plot_dir),
                        "--exp-restart-dir",
                        str(analysis["exp_restart_dir"]),
                        "--baseline-sim-name",
                        str(tpl["sim_name"]),
                        "--nb-nodes",
                        str(tpl["nb_nodes"]),
                        "--domain-length",
                        str(tpl["domain_length"]),
                        "--roi-min",
                        str(analysis["roi_min"]),
                        "--roi-max",
                        str(analysis["roi_max"]),
                    ]
                    if ranking_mode == "translation_aligned" and analysis.get("enable_translated_plot", False) and best_shift_m != "NA":
                        plot_args.extend(["--best-shift-m", str(best_shift_m), "--write-translated-plot"])
                    with plot_log.open("w", encoding="utf-8") as plog:
                        prc = subprocess.run(
                            plot_args,
                            cwd=str(paths["work_dir"]),
                            stdout=plog,
                            stderr=subprocess.STDOUT,
                            check=False,
                        ).returncode
                    plot_status = "OK" if prc == 0 else f"FAILED({prc})"

                if run_status == "OK" and analysis["enable_rmse"]:
                    rmse_log = case_dir / "rmse.log"
                    with rmse_log.open("w", encoding="utf-8") as rlog:
                        proc = subprocess.run(
                            [
                                "python3",
                                str(rmse_script),
                                "--baseline-dir",
                                str(case_dir),
                                "--baseline-sim-name",
                                str(tpl["sim_name"]),
                                "--exp-base-dir",
                                str(analysis["exp_base_dir"]),
                                "--exp-restart-dir",
                                str(analysis["exp_restart_name"]),
                                "--nb-nodes",
                                str(tpl["nb_nodes"]),
                                "--domain-length",
                                str(tpl["domain_length"]),
                            ] + (
                                [
                                    "--shift-min",
                                    str(analysis["translation_shift_min_m"]),
                                    "--shift-max",
                                    str(analysis["translation_shift_max_m"]),
                                    "--shift-step",
                                    str(analysis["translation_shift_step_m"]),
                                ]
                                if ranking_mode == "translation_aligned"
                                else []
                            ),
                            cwd=str(paths["work_dir"]),
                            stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT,
                            text=True,
                            check=False,
                        )
                        rlog.write(proc.stdout)
                        if proc.returncode == 0:
                            m = re.search(r"Average relative error .*\(([0-9.eE+\-]+)%\)", proc.stdout)
                            shift_m = re.search(r"Best shift found \(m\)\s*:\s*([0-9.eE+\-]+)", proc.stdout)
                            if m:
                                rmse_percent = m.group(1)
                                if shift_m:
                                    best_shift_m = shift_m.group(1)
                                try:
                                    ranking_entries.append((float(rmse_percent), run_label, str(case_dir), best_shift_m))
                                except ValueError:
                                    pass

            if ranking_mode == "translation_aligned":
                row = (
                    f"{run_label}\t{run_status}\t{plot_status}\t{rmse_percent}\t{best_shift_m}\t"
                    + "\t".join(_fmt_float(combo[sp.name]) for sp in state["vary"])
                    + f"\t{case_dir}\n"
                )
            else:
                row = (
                    f"{run_label}\t{run_status}\t{plot_status}\t{rmse_percent}\t"
                    + "\t".join(_fmt_float(combo[sp.name]) for sp in state["vary"])
                    + f"\t{case_dir}\n"
                )
            with summary_log.open("a", encoding="utf-8") as slog:
                slog.write(row)

        if not dry_run and ranking_entries:
            entries = sorted(ranking_entries, key=lambda x: x[0])
            with ranking_file.open("w", encoding="utf-8") as rf:
                rf.write("================================================================\n")
                title = "CONFIGURABLE SWEEP RMSE RANKING (Best -> Worst)"
                if ranking_mode == "translation_aligned":
                    title = "CONFIGURABLE SWEEP TRANSLATION-ALIGNED RMSE RANKING (Best -> Worst)"
                rf.write(f"{title}\n")
                rf.write("================================================================\n")
                if ranking_mode == "translation_aligned":
                    rf.write("Rank | RMSE (%)      | BEST_SHIFT_M | RUN_LABEL                  | OUTPUT_DIR\n")
                    rf.write("-----+---------------+--------------+----------------------------+--------------------------------\n")
                    for i, (rmse, label, out_dir, best_shift_m) in enumerate(entries, start=1):
                        rf.write(f"{i:4d} | {rmse:13.5f} | {best_shift_m:>12} | {label:<26} | {out_dir}\n")
                else:
                    rf.write("Rank | RMSE (%)      | RUN_LABEL                  | OUTPUT_DIR\n")
                    rf.write("-----+---------------+----------------------------+--------------------------------\n")
                    for i, (rmse, label, out_dir, _best_shift_m) in enumerate(entries, start=1):
                        rf.write(f"{i:4d} | {rmse:13.5f} | {label:<26} | {out_dir}\n")
                rf.write("================================================================\n")
            print(f"Saved RMSE ranking: {ranking_file}")

        print("\nSweep finished.")
        print(f"Summary: {summary_log}")

    finally:
        if tmp_input.exists():
            tmp_input.unlink(missing_ok=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Configurable parameter sweep runner for uguca",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--create-config", type=Path, default=None,
                   help="Create a JSON config interactively at this path")
    p.add_argument("--config", type=Path, default=None,
                   help="Path to JSON config file")
    p.add_argument("--run", action="store_true", help="Run the sweep")
    p.add_argument("--dry-run", action="store_true", help="Validate and preview only")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.create_config is not None:
        _interactive_create_config(args.create_config.expanduser().resolve())
        print("\nNext steps:")
        print(f"  python3 {Path(__file__).name} --config {args.create_config} --dry-run")
        print(f"  python3 {Path(__file__).name} --config {args.create_config} --run")
        return

    if args.config is None:
        print("No --config provided. Starting interactive config creation.")
        default_cfg = Path.cwd() / "automation" / "sweeps" / "configurable_sweep.json"
        _interactive_create_config(default_cfg)
        print("\nNow run:")
        print(f"  python3 {Path(__file__).name} --config {default_cfg} --dry-run")
        print(f"  python3 {Path(__file__).name} --config {default_cfg} --run")
        return

    cfg = _load_config(args.config.expanduser().resolve())
    state = _validate_and_expand(cfg)
    _preview_plan(state)

    if args.dry_run and args.run:
        raise ConfigError("Use either --dry-run or --run, not both")

    if not args.dry_run and not args.run:
        print("\nNeither --run nor --dry-run supplied; defaulting to --dry-run preview.")
        args.dry_run = True

    if args.run and state["meta"]["confirm_before_run"]:
        msg = f"Proceed with {state['total_runs']} run(s)?"
        if not _prompt_yes_no(msg, default=False):
            print("Cancelled by user.")
            return

    _run_sweep(state, dry_run=args.dry_run)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
        sys.exit(130)
    except ConfigError as exc:
        print(f"[CONFIG ERROR] {exc}")
        sys.exit(2)
