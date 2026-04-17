# UGUCA Data-Driven Simulations
**Author:** Joshua McNeely
**Date:** 17 April 2026

## 1. Scope

I have put this README together to document the scope and operational details of our simulation machinery. The framework is primarily centred on two executables and their associated automation pipelines:

* `dd_earthquake`
* `time_int_opt`
* Automation workflows (specifically handling `automation/sweeps`, `automation/ranking`, `automation/plotting`, and `automation/analysis`).

---

## 2. Directory Layout

Here is a breakdown of the top-level simulation source and the runtime directory layout.

**Top-Level Files and Directories:**
* `CMakeLists.txt`: Registers our simulation targets and handles copying the runtime machinery into the build tree.
* `dd_earthquake.cc`: The data-driven/baseline simulation script for the laboratory-generated earthquakes problem.
* `time_int_opt.cc`: The data-driven/ground-truth simulation script for the 2D fracture problem.
* `input_files/`: Contains the simulation input templates (`.in`) consumed by our executables and parameter sweeps.
* `analysis_and_processing_tools/`: Scripts for post-processing helpers.
* `plotting_tools/`: Scripts to handle plotting tasks.
* `automation/`: High-level orchestration for sweeps, plotting, and ranking.

**Automation Subtrees:**
* `automation/sweeps/`: Contains the sweep runner (`sweep_configurable.py`), the JSON sweep configs (`*.json`), and the launch wrappers (`run_*.sh`).
* `automation/ranking/`: Tools for calculating RMSE and generating rankings.
* `automation/plotting/`: Scripts for generating spacetime and diagnostic plots.
* `automation/analysis/`: Helper scripts for focused, bespoke analysis.

---

## 3. Build and Runtime Model

The `simulations/CMakeLists.txt` file is designed to accomplish two core tasks:

1.  **Register executable targets:**
    * `add_simulation(time_int_opt time_int_opt.cc)`
    * `add_simulation(rs_sw_regular_dumping rs_sw_regular_dumping.cc)`
    * `add_simulation(dd_earthquake dd_earthquake.cc)`
2.  **Copy runtime machinery:** It duplicates `automation/`, `input_files/`, and our plotting/analysis scripts into `build/simulations/`. This ensures jobs execute smoothly from a self-contained runtime layout within the build tree.

### How to Build

From the root of the UGUCA repository, execute the following steps:

1.  **Configure:** `mkdir -p build && cd build && cmake ..`
2.  **Build:** `make -j <num_cores>`
3.  **Note:** Ensure that the simulations folder is toggled to `ON` in the CMake configuration (you can verify this via `ccmake ..`).

Post-compilation, your runtime working directory will be `build/simulations/`. This directory will contain:
* Executable binaries (`dd_earthquake`, `time_int_opt`).
* Copied scripts and configuration files.
* Output roots (`simulation_outputs/`, `sweep_outputs/`).

---

## 4. Executable: `dd_earthquake`

### Purpose
I utilise `dd_earthquake` to run interface earthquake simulations. It is configured to handle specific material and loading parameters, nucleation forcing, friction/interface laws, optional asperity perturbations, and optional data-driven displacement updates.

### Usage
To run the executable, use the following syntax:
`./dd_earthquake <input_file>`

*Example:*
`./dd_earthquake input_files/local_baseline_compare_nu019.in`

### Input Parameters
These are the input parameters pulled from the `.in` files:

* **Geometry and Discretisation:** `length`, `nb_elements`, `free_surface`
* **Time Integration:** `duration`, `time_step_factor`, `nb_predictor_corrector`, `dump_interval`
* **Material Properties:** `E`, `nu`, `rho`
* **Loading:** `shear_load`, `normal_load`
* **Nucleation:** `nuc_center`, `nuc_dtau`, `nuc_size`, `nuc_time`
* **Asperity Modulation:** `asperity`, `asp_center`, `asp_fct`, `asp_size`, `box_car_shape`
* **Friction/Interface Law Block:** Defined by `Law` (e.g., `RateAndState`), alongside law-specific parameters (`a`, `b`, `Dc`, `V0`, `f0`, `V_init`, `evolution_law`).
* **Outputs:** `simulation_name`, `dump_folder`, `dump_fields`

### Outputs
The script writes interface outputs to the designated `dump_folder` using the `simulation_name` as a prefix. This includes the files consumed downstream by our ranking and plotting scripts (e.g., `*-interface.time`, `*-interface-DataFiles/top_disp.out`).

---

## 5. Executable: `time_int_opt`

### Purpose
The `time_int_opt` script is our data-driven tool for 2D fracture problems. It operates in two primary modes:
* **Ground-truth generation mode.**
* **Data-driven mode:** This consumes previously generated restart data and incorporates optional per-node weighting.

### Usage
`./time_int_opt <input_file>`

*Input resolution behaviour:* If your `<input_file>` path does not begin with `input_files/`, the code will automatically prepend it for you.

*Examples:*
* `./time_int_opt ground_truth.in`
* `./time_int_opt data_driven.in`

### Behaviour
* Generates organised outputs stored in `simulation_outputs/<sim_name>_<sim_type>`.
* Facilitates periodic data-driven updates at configured intervals.
* Can dump displacement fields in binary restart format for further downstream analysis.

---

## 6. Automation Workflows (for `dd_earthquake`)

### Sweep Runner (`automation/sweeps/sweep_configurable.py`)

This is the primary configurable sweep engine. I have designed it to support:
* `--create-config`: An interactive setup wizard.
* `--dry-run`: Essential for validation-only execution planning.
* `--run`: Executes the complete solver, plotting, and ranking workflow.

**Execution Flow:**
For each parameter combination, the engine strictly follows this sequence:
1.  Creates a case directory under the sweep output root.
2.  Writes the rendered input out as `input_used.in`.
3.  Runs the solver (`dd_earthquake`) and logs the solver output.
4.  Executes the plotting script (if enabled).
5.  Runs the RMSE script (selecting between standard or translation-aligned, depending on your `ranking_mode`).
6.  Appends the resulting row to the sweep summary log.

Upon completion, it produces a per-sweep ranking file located at:
`comparison_plots/sweep_rmse_ranking_<mode>.txt`

**Ranking Modes:**
* `baseline_vs_exp`
* `translation_aligned`: This mode selects `compute_baseline_vs_exp_rmse_translation_aligned.py` and logs `BEST_SHIFT_M` alongside the RMSE.

### Sweep Configs and Launchers
You can find existing configurations and campaign files under `automation/sweeps/`:
* **Reusable Examples:** `configurable_sweep.example.json`
* **Historical Focused Sweeps:** `single_param_*.json`, `exploratory_*.json`, `focused_*.json`
* **Recent Configs:**
    * `phase1_confirmational_cc_45.json`
    * `phase2_central_nucleation_cc_72.json`
    * `phase3_nuc_time_cc_18.json`
* **Launch Wrappers:**
    * `run_phase1_confirmational_cc_45.sh`
    * `run_phase2_central_nucleation_cc_72.sh`
    * `run_phase3_nuc_time_cc_18.sh`

The launcher contract is strictly standardised:
* `bash run_*.sh dry-run`
* `bash run_*.sh run`

---

## 7. Ranking and Aggregation (`automation/ranking`)

### Metrics
* `compute_baseline_vs_exp_rmse.py`: Handles the standard baseline RMSE workflow.
* `compute_baseline_vs_exp_rmse_translation_aligned.py`: Handles translation-aligned RMSE via global shift searches (`shift-min/max/step`).

### Ranking Generation
The `combine_sweep_rankings.py` script scans all sweep output folders and merges the per-sweep ranking files into two primary outputs:
* `sweep_outputs/combined_rankings/combined_cross_correlated_rankings.txt` (This corresponds to the translation-aligned category).
* `sweep_outputs/combined_rankings/combined_other_rankings.txt` (This corresponds to the non-translation-aligned category).

---

## 8. Plotting (`automation/plotting`)

For visualisation, the primary scripts include:
* `Plot_BaselineExp_Spacetime.py`: The core spacetime comparison tool used during sweeps.
* `Plot_DcA_RMSE.py`, `Plot_DcA_Spacetime.py`, `Plot_DcA_Slip_Isochrones.py`
* Aggregate/overlay helpers and shell wrappers (e.g., `run_all_*`, `plot_*`, `replot_all.sh`).

Note that during configurable sweeps, plotting is strictly governed by the `analysis.enable_plot` flag and the path settings within your sweep JSON file.

---

## 9. Standard Workflow Summary

When executing a sweep, adhere to this protocol:

1.  Build your simulation targets in the `build/` directory.
2.  Execute your sweep in `build/simulations/automation/sweeps` via the appropriate launcher.
    * *Always execute a dry-run first.*
    * Follow up with the full run.
3.  Inspect the resulting sweep outputs:
    * `sweep_outputs/<sweep_name>/comparison_plots/sweep_summary_all.log`
    * `sweep_outputs/<sweep_name>/comparison_plots/sweep_rmse_ranking_all.txt`
4.  Aggregate the overall rankings by running `automation/ranking/combine_sweep_rankings.py`.
5.  Export any shortlist views (such as top-k lists) as necessary for your analysis.

---

## 10. Output Locations

**Primary Output Roots:**
* `build/simulations/sweep_outputs/`
* `build/simulations/simulation_outputs/`

**For Each Sweep:**
* Cases: `sweep_outputs/<sweep_name>/cases/results_baseline_<run_label>/`
* Summary: `.../comparison_plots/sweep_summary_all.log`
* Ranking: `.../comparison_plots/sweep_rmse_ranking_all.txt`

**Combined Rankings:**
* `sweep_outputs/combined_rankings/combined_cross_correlated_rankings.txt`
* `sweep_outputs/combined_rankings/combined_other_rankings.txt`

---

## 11. Final Operational Notes

* **Dry-Runs:** Always utilise a dry-run before initiating any large-scale sweep to validate run counts, paths, and parameter substitutions.
* **Pathing:** Prefer absolute paths in JSON configs to maintain robustness during long computational jobs.
* **Permissions:** Sweep launcher scripts may occasionally lack the executable bit in certain environments; invoking them explicitly with `bash run_*.sh <mode>` is the most robust approach.
* **Rankings:** Remember that a `translation_aligned` ranking differs fundamentally from a baseline ranking. Ensure you keep these categories isolated when comparing performance metrics.