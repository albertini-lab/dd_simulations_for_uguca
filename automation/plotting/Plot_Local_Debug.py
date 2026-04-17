#!/usr/bin/env python3
import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
import os, glob, re
from pathlib import Path

# --- PATH CONFIGURATION ---
# 1. Dynamic Simulation Results (The W-Factor Sweep)
#    These are folders like ./results_w1e-4 created by the bash script
SWEEP_RESULTS_DIR = Path(".") 

# 2. Fixed Baseline Results (Pure Physics)
#    Hardcoded to the location you specified
BASELINE_RESULTS_DIR = Path("/Users/joshmcneely/uguca/build/simulations/results_baseline")

# 3. Experimental Data
EXP_BASE_DIR = Path("/Users/joshmcneely/introsims/simulation_outputs")

# 4. Output Plots
PLOT_DIR = Path("./comparison_plots")

# --- GEOMETRY & SETTINGS ---
DOMAIN_LENGTH = 6.0   
NB_NODES = 2048       
SENSOR_POSITIONS = np.arange(0.05, 3.05 + 0.01, 0.2)
SIM_DENSITY_MULTIPLIER = 20  

# --- SIMULATION NAMES ---
# The sweep script moves ./results to ./results_w{W}
# So we look for the folder name passed implicitly or default to ./results
DD_SIM_NAME = "local_debug_run"
BASE_SIM_NAME = "local_baseline_run"

def load_sim_robust(base_path, sim_name, nb_nodes):
    print(f"\n========== [DEBUG] LOADING SIMULATION: {sim_name} ==========")
    print(f"   -> Looking in: {base_path}")
    
    # Construct paths
    data_dir = base_path / f"{sim_name}-interface-DataFiles"
    path = data_dir / "top_disp.out"
    t_path = base_path / f"{sim_name}-interface.time"
    
    if not path.exists() or not t_path.exists(): 
        print(f"[CRITICAL ERROR] Files not found!")
        print(f"   -> Missing: {path}")
        return None, None
    
    with open(t_path, 'r') as f:
        times = np.array([float(line.split()[1]) for line in f.readlines()])
    
    try:
        raw_data = np.loadtxt(path)
        if raw_data.ndim == 1: raw_data = raw_data.reshape(1, -1)
        
        # --- FIXED SEQUENTIAL EXTRACTOR (Component-Major) ---
        if raw_data.shape[1] >= nb_nodes:
            print(f"[DEBUG] Extracting first {nb_nodes} columns (X-shear)...")
            data = raw_data[:, :nb_nodes]
        else:
            print(f"[WARNING] Unrecognised geometry. Extracting available.")
            data = raw_data
            
    except Exception as e:
        print(f"[CRITICAL ERROR] I/O Failure: {e}")
        return None, None

    final_count = min(len(times), data.shape[0])
    return times[:final_count], data[:final_count, :]

def load_exp(restart_dir, sensor_nodes, nb_nodes):
    print(f"\n========== [DEBUG] LOADING EXPERIMENTAL TARGETS ==========")
    info_file = restart_dir.parent / "mcklaskey_debug.info"
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
    dx = DOMAIN_LENGTH / NB_NODES
    sensor_nodes_raw = np.round(SENSOR_POSITIONS / dx).astype(int)
    valid_sensors = sensor_nodes_raw < NB_NODES
    sensor_nodes = sensor_nodes_raw[valid_sensors]
    actual_positions = sensor_nodes * dx
    
    os.makedirs(PLOT_DIR, exist_ok=True)
    restart_dir = EXP_BASE_DIR / "mcklaskey_debug-restart"

    # 1. Load Experimental Data
    exp_times, exp_data = load_exp(restart_dir, sensor_nodes, NB_NODES)
    
    # 2. Load Baseline (Fixed Path)
    base_t, base_d = load_sim_robust(BASELINE_RESULTS_DIR, BASE_SIM_NAME, NB_NODES)
    
    # 3. Load Data-Driven (Dynamic Path)
    #    The script assumes the data-driven run is currently in ./results 
    #    (before the bash script renames it).
    dd_t, dd_d = load_sim_robust(Path("./results"), DD_SIM_NAME, NB_NODES)
    
    if any(v is None for v in [dd_d, base_d, exp_data]): return

    print(f"\n========== [DEBUG] COMPUTING COMPARATIVE METRICS ==========")
    
    # Extract Exact Temporal Matches for RMSE
    idx_dd = np.searchsorted(dd_t, exp_times).clip(0, len(dd_t)-1)
    idx_base = np.searchsorted(base_t, exp_times).clip(0, len(base_t)-1)
    
    dd_sensors_exact = dd_d[idx_dd][:, sensor_nodes]
    base_sensors_exact = base_d[idx_base][:, sensor_nodes]
    
    # Evaluate Global L2 Error Norms
    dd_rmse = np.sqrt(np.mean((dd_sensors_exact - exp_data)**2, axis=1))
    base_rmse = np.sqrt(np.mean((base_sensors_exact - exp_data)**2, axis=1))

    # Decimate for Plotting
    target_plot_points = len(exp_times) * SIM_DENSITY_MULTIPLIER
    stride_dd = max(1, len(dd_t) // target_plot_points)
    stride_base = max(1, len(base_t) // target_plot_points)
    
    dd_t_plot, dd_d_plot = dd_t[::stride_dd], dd_d[::stride_dd]
    base_t_plot, base_d_plot = base_t[::stride_base], base_d[::stride_base]

    num_sensors = len(sensor_nodes)
    ncols = 4
    nrows = int(np.ceil(num_sensors / ncols)) + 1
    fig = plt.figure(figsize=(20, 4 * nrows))
    gs = fig.add_gridspec(nrows, ncols, hspace=0.4, wspace=0.3)

    # Global Y-Axis Scaling (Fixed to Experiment Range)
    y_min = np.min(exp_data)
    y_max = np.max(exp_data)
    y_margin = (y_max - y_min) * 0.10
    if y_margin == 0: y_margin = 1e-8
    global_y_lim = (y_min - y_margin, y_max + y_margin)

    # Plot Sensors
    for i in range(num_sensors):
        ax = fig.add_subplot(gs[i // ncols, i % ncols])
        
        ax.plot(exp_times*1e3, exp_data[:, i], 'ko', markersize=4, alpha=0.5, label='Experiment')
        ax.plot(base_t_plot*1e3, base_d_plot[:, sensor_nodes[i]], 'k:', linewidth=1.5, alpha=0.7, label='Baseline')
        ax.plot(dd_t_plot*1e3, dd_d_plot[:, sensor_nodes[i]], 'r-', linewidth=2, label='Data-Driven')
        
        ax.set_ylim(global_y_lim)
        ax.set_title(f"Sensor at x = {actual_positions[i]:.2f}m")
        ax.set_xlabel("Time (ms)")
        ax.set_ylabel("Displacement (m)")
        ax.grid(True, alpha=0.3)
        if i == 0: ax.legend()

    # Plot RMSE
    ax_rmse = fig.add_subplot(gs[-1, :])
    ax_rmse.plot(exp_times*1e3, base_rmse, 'k:o', linewidth=1.5, alpha=0.7, label='Baseline RMSE')
    ax_rmse.plot(exp_times*1e3, dd_rmse, 'r-o', linewidth=2, label='Data-Driven RMSE')
    ax_rmse.set_xlabel("Time (ms)")
    ax_rmse.set_ylabel("RMSE (m)")
    ax_rmse.set_title("Global Root Mean Square Error")
    ax_rmse.legend()
    ax_rmse.grid(True, alpha=0.3)

    plot_path = PLOT_DIR / "diagnostic_comparative.png"
    plt.savefig(plot_path, dpi=200, bbox_inches='tight')
    print(f"[SUCCESS] Comparative diagnostic plot generated at: {plot_path}\n")

if __name__ == "__main__":
    main()
