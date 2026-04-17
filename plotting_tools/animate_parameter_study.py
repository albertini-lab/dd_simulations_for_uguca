#!/usr/bin/env python3
"""
Animate parameter study results showing displacement evolution over time.
This creates an animation with the same format as plot_parameter_study.py but 
showing how displacements evolve at each time step.
"""
from __future__ import print_function
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.collections import LineCollection
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches
from matplotlib.legend_handler import HandlerBase
import sys
import os
import glob
import re
import struct

# Add MacTeX to PATH for LaTeX rendering
os.environ['PATH'] = '/usr/local/texlive/2023/bin/universal-darwin:' + os.environ.get('PATH', '')

# --- Matplotlib parameters ---
params = {
    'text.usetex': True,
    'text.latex.preamble': r'\usepackage{lmodern}\usepackage[T1]{fontenc}\usepackage{sansmath}\sansmath',
    'font.family': 'sans-serif',
    'font.sans-serif': ['Latin Modern Sans'],
    'font.size': 11, 'figure.titlesize': 12, 'axes.labelsize': 11,
    'axes.titlesize': 12, 'legend.fontsize': 11, 'legend.frameon': False,
    'lines.linewidth': 1.2, 'axes.linewidth': 0.6,
    'xtick.labelsize': 11, 'ytick.labelsize': 11, 'xtick.major.size': 3,
    'ytick.major.size': 3,
}
mpl.rcParams.update(params)

def load_coords(coords_file):
    """Load coordinate data from file."""
    try:
        return np.loadtxt(coords_file, usecols=0)
    except (IOError, ValueError) as e:
        print(f"Error loading coordinates from {coords_file}: {e}")
        return None

def load_time_data(sim_name, base_path):
    """Load time data from .time file."""
    time_file = os.path.join(base_path, f"{sim_name}.time")
    try:
        time_data = np.loadtxt(time_file)
        if time_data.ndim == 1:
            time_data = time_data.reshape(-1, 2)
        return time_data
    except (IOError, ValueError) as e:
        print(f"Warning: Could not load time data from {time_file}: {e}")
        return None

def find_displacement_data_dir(sim_name):
    """Find the displacement data directory for a simulation."""
    # Try different possible directory structures
    base_path = os.path.join("simulation_outputs", sim_name)
    
    # For ground truth simulations
    if not os.path.isdir(base_path):
        base_path_gt = os.path.join("simulation_outputs", f"{sim_name}_ground_truth")
        if os.path.isdir(base_path_gt):
            base_path = base_path_gt
    
    # For data-driven simulations
    if not os.path.isdir(base_path):
        base_path_dd = os.path.join("simulation_outputs", f"{sim_name}_data_driven")
        if os.path.isdir(base_path_dd):
            base_path = base_path_dd
    
    if not os.path.isdir(base_path):
        return None, None, None
    
    coords_file = os.path.join(base_path, f"{sim_name}.coords")
    
    # Look for displacement dump directory - try multiple naming patterns
    disp_dirs = [
        os.path.join(base_path, f"{sim_name}_displacement_dump-restart"),
        os.path.join(base_path, "calculated_displacements-restart"),
        os.path.join(base_path, f"{sim_name}-restart")
    ]
    
    disp_dir = None
    for d in disp_dirs:
        if os.path.isdir(d):
            disp_dir = d
            break
    
    if not os.path.exists(coords_file) or not disp_dir:
        print(f"Warning: Could not find coords ({os.path.exists(coords_file)}) or disp_dir ({disp_dir is not None}) for {sim_name}")
        print(f"  base_path: {base_path}")
        print(f"  coords_file: {coords_file}")
        if disp_dir:
            print(f"  disp_dir: {disp_dir}")
        return None, None, None
    
    return coords_file, disp_dir, base_path

def detect_file_format(disp_dir):
    """Detect whether files are ASCII or binary format."""
    # Look for a sample file
    sample_files = glob.glob(os.path.join(disp_dir, "*.0.out"))
    if not sample_files:
        return 'binary'
    
    sample_file = sample_files[0]
    try:
        with open(sample_file, 'r') as f:
            f.read(100)
        return 'ascii'
    except UnicodeDecodeError:
        return 'binary'

def load_displacement_step_binary(disp_dir, step, num_nodes, field_prefix):
    """Load displacement data for a specific step from binary files."""
    # Single file contains both x and y components as float32
    file_path = os.path.join(disp_dir, f"{field_prefix}.proc0.s{step}.out")
    
    try:
        if os.path.exists(file_path):
            with open(file_path, 'rb') as f:
                data = f.read()
                # Data is float32, 2 components (x, y) per node
                num_floats = len(data) // 4
                expected_floats = num_nodes * 2
                
                if num_floats != expected_floats:
                    print(f"Warning: File {file_path} has {num_floats} values, expected {expected_floats}")
                    if num_floats < expected_floats:
                        return None, None
                
                # Unpack as float32 and convert to float64
                values = struct.unpack(f'{expected_floats}f', data[:expected_floats * 4])
                values = np.array(values, dtype=np.float64)
                
                # Reshape to (2, num_nodes) - first row is x, second is y
                displacement_data = values.reshape((2, num_nodes))
                return displacement_data[0, :], displacement_data[1, :]
        else:
            return None, None
    except Exception as e:
        print(f"Error loading binary step {step} from {file_path}: {e}")
        return None, None

def get_available_steps(disp_dir, field_prefix='interface_top_disp'):
    """Get list of available time steps in the displacement directory."""
    # Look for files matching the pattern
    pattern = os.path.join(disp_dir, f"{field_prefix}.proc0.s*.out")
    files = glob.glob(pattern)
    
    steps = []
    for f in files:
        match = re.search(r's(\d+)\.out$', f)
        if match:
            steps.append(int(match.group(1)))
    
    return sorted(steps)

def load_all_steps(sim_name, max_steps=None, step_skip=1):
    """Load displacement data for all time steps of a simulation."""
    coords_file, disp_dir, base_path = find_displacement_data_dir(sim_name)
    
    if not coords_file or not disp_dir:
        print(f"Could not find data for {sim_name}")
        return None, None, None, None, None
    
    X = load_coords(coords_file)
    if X is None:
        return None, None, None, None, None
    
    num_nodes = len(X)
    
    # Determine field prefix based on directory type
    if 'displacement_dump' in disp_dir:
        # Ground truth format
        field_prefix_top = 'interface_top_disp'
        field_prefix_bot = 'interface_bot_disp'
    else:
        # Data-driven format
        field_prefix_top = 'calculated_top_disp'
        field_prefix_bot = 'calculated_bot_disp'
    
    # Get available steps
    steps = get_available_steps(disp_dir, field_prefix_top)
    if not steps:
        print(f"No displacement steps found for {sim_name}")
        return None, None, None, None, None
    
    # Apply step skipping and max steps
    steps = steps[::step_skip]
    if max_steps:
        steps = steps[:max_steps]
    
    print(f"Loading {len(steps)} steps for {sim_name}...")
    
    # Load time data
    time_data = load_time_data(sim_name, base_path)
    times = []
    
    # Pre-allocate arrays
    top_disp_x = np.zeros((len(steps), num_nodes))
    top_disp_y = np.zeros((len(steps), num_nodes))
    bot_disp_x = np.zeros((len(steps), num_nodes))
    bot_disp_y = np.zeros((len(steps), num_nodes))
    
    for i, step in enumerate(steps):
        top_x, top_y = load_displacement_step_binary(disp_dir, step, num_nodes, field_prefix_top)
        bot_x, bot_y = load_displacement_step_binary(disp_dir, step, num_nodes, field_prefix_bot)
        
        if top_x is not None and bot_x is not None:
            top_disp_x[i, :] = top_x
            top_disp_y[i, :] = top_y
            bot_disp_x[i, :] = bot_x
            bot_disp_y[i, :] = bot_y
            
            # Get time for this step
            if time_data is not None:
                time_idx = np.where(time_data[:, 0] == step)[0]
                if len(time_idx) > 0:
                    times.append(time_data[time_idx[0], 1])
                else:
                    times.append(step * 1.5e-7)  # Approximate
            else:
                times.append(step * 1.5e-7)  # Approximate
    
    times = np.array(times)
    
    return X, top_disp_x, bot_disp_x, steps, times

if __name__ == "__main__":
    # --- Configuration ---
    step_skip = 1  # Use every time step (1 = all steps)
    max_steps = None  # Set to limit number of frames, None = all steps
    animation_fps = 35  # Playback speed; 35 fps is ~1.75x faster than 20 fps
    
    # Select a subset of intervals to animate (for manageable file size and clarity)
    # Custom selection with more detail on the low end (1-200) and sparse on high end
    selected_intervals = 'custom_weighted'  # Will be computed from available intervals
    
    print("="*70)
    print("PARAMETER STUDY DISPLACEMENT ANIMATION")
    print("="*70)
    print(f"Step skip: {step_skip} (using every {step_skip}th time step)")
    print(f"Selected intervals: Custom with more detail on low end (1-200)")
    print()
    
    # --- Find Simulation Results ---
    sim_output_dir = "simulation_outputs"
    if not os.path.isdir(sim_output_dir):
        sys.exit(f"Error: simulation_outputs directory not found.")
    
    # --- Load Ground Truth ---
    print("Loading ground truth data...")
    gt_sim_name = "time_int_opt_gt"
    X_gt, gt_top_x, gt_bot_x, gt_steps, gt_times = load_all_steps(gt_sim_name, max_steps, step_skip)
    
    # --- Load Baseline ---
    print("Loading baseline data...")
    baseline_sim_name = "time_int_opt_dd_baseline_no_dd"
    X_baseline, baseline_top_x, baseline_bot_x, baseline_steps, baseline_times = load_all_steps(
        baseline_sim_name, max_steps, step_skip)
    
    # --- Load Parameter Study Data ---
    results = []
    cmap = plt.cm.viridis
    all_intervals = []
    
    # First, find all available intervals
    sim_dirs = glob.glob(os.path.join(sim_output_dir, "time_int_opt_dd_interval_*"))
    for s_dir in sim_dirs:
        match = re.search(r"interval_(\d+)", s_dir)
        if match:
            all_intervals.append(int(match.group(1)))
    all_intervals.sort()
    
    # Select intervals with more detail on low end (1-200)
    if selected_intervals == 'custom_weighted':
        # Select more intervals in the 1-200 range (about 10 intervals)
        # and fewer in the 200-500 range (about 5 intervals)
        low_intervals = [i for i in all_intervals if i <= 200]
        high_intervals = [i for i in all_intervals if i > 200]
        
        # Get 10 equally spaced from low range
        if len(low_intervals) <= 10:
            selected_low = low_intervals
        else:
            indices_low = np.linspace(0, len(low_intervals) - 1, 10, dtype=int)
            selected_low = [low_intervals[i] for i in indices_low]
        
        # Get 5 equally spaced from high range
        if len(high_intervals) <= 5:
            selected_high = high_intervals
        else:
            indices_high = np.linspace(0, len(high_intervals) - 1, 5, dtype=int)
            selected_high = [high_intervals[i] for i in indices_high]
        
        intervals_to_load = sorted(selected_low + selected_high)
    elif isinstance(selected_intervals, list):
        # Use specified list
        intervals_to_load = [i for i in selected_intervals if i in all_intervals]
    else:
        # Use all intervals
        intervals_to_load = all_intervals
    
    print(f"\nLoading {len(intervals_to_load)} simulations: {intervals_to_load}")
    print(f"From {len(all_intervals)} total available intervals")
    print()
    
    norm = mpl.colors.Normalize(vmin=min(all_intervals), vmax=max(all_intervals))
    
    for interval in intervals_to_load:
        sim_name = f"time_int_opt_dd_interval_{interval}"
        print(f"Loading interval {interval}...")
        
        X, top_x, bot_x, steps, times = load_all_steps(sim_name, max_steps, step_skip)
        
        if X is not None:
            color = cmap(norm(interval))
            results.append({
                'interval': interval,
                'X': X,
                'top_disp_x': top_x,
                'bot_disp_x': bot_x,
                'steps': steps,
                'times': times,
                'color': color
            })
    
    if not results:
        sys.exit("Error: No parameter study data loaded successfully.")
    
    # Determine common time steps (use ground truth as reference)
    if gt_times is None:
        sys.exit("Error: Ground truth data required for animation.")
    
    num_frames = len(gt_times)
    print(f"\nCreating animation with {num_frames} frames...")
    
    # --- Setup Animation ---
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    # Storage for line objects
    lines_dd = []
    lines_gt = []
    lines_baseline = []
    
    # Initialize lines for data-driven results
    for result in results:
        line_top, = ax.plot([], [], color=result['color'], alpha=0.65, linewidth=1.0, zorder=1)
        line_bot, = ax.plot([], [], color=result['color'], alpha=0.65, linewidth=1.0, zorder=1)
        lines_dd.append((line_top, line_bot))
    
    # Initialize ground truth lines
    if X_gt is not None:
        line_gt_top, = ax.plot([], [], 'r-', linewidth=1, label=r'$u_{GT}$', alpha=1.0, zorder=10)
        line_gt_bot, = ax.plot([], [], 'r-', linewidth=1, alpha=1.0, zorder=10)
        lines_gt = [line_gt_top, line_gt_bot]
    
    # Initialize baseline lines
    if X_baseline is not None:
        line_base_top, = ax.plot([], [], color='orange', linewidth=1, label=r'$u_{Base}$', alpha=1.0, zorder=9)
        line_base_bot, = ax.plot([], [], color='orange', linewidth=1, alpha=1.0, zorder=9)
        lines_baseline = [line_base_top, line_base_bot]
    
    # Text for time display
    time_text = ax.text(0.02, 0.98, '', transform=ax.transAxes, 
                       verticalalignment='top', fontsize=12,
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Setup axes
    ax.set_xlabel(r'Position, $x/L$')
    ax.set_ylabel(r'Displacement $u$')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1)
    
    # Determine y-limits from final frame
    all_final_values = []
    if gt_top_x is not None:
        all_final_values.extend([gt_top_x[-1].min(), gt_top_x[-1].max(), 
                                gt_bot_x[-1].min(), gt_bot_x[-1].max()])
    for result in results:
        all_final_values.extend([result['top_disp_x'][-1].min(), result['top_disp_x'][-1].max(),
                                result['bot_disp_x'][-1].min(), result['bot_disp_x'][-1].max()])
    
    y_margin = (max(all_final_values) - min(all_final_values)) * 0.1
    ax.set_ylim(min(all_final_values) - y_margin, max(all_final_values) + y_margin)
    
    # Create legend with gradient representation
    class GradientHandler(HandlerBase):
        def create_artists(self, legend, orig_handle, xdescent, ydescent, width, height, fontsize, trans):
            n_segments = 20
            segment_width = width / n_segments
            artists = []
            for i in range(n_segments):
                x_pos = xdescent + i * segment_width
                color_val = i / (n_segments - 1)
                color = cmap(color_val)
                mini_rect = Rectangle((x_pos, ydescent), segment_width, height, 
                                    facecolor=color, edgecolor='none', transform=trans)
                artists.append(mini_rect)
            return artists
    
    legend_elements = []
    if X_gt is not None:
        legend_elements.append(plt.Line2D([0], [0], color='red', linewidth=1, label=r'$u_{GT}$'))
    if X_baseline is not None:
        legend_elements.append(plt.Line2D([0], [0], color='orange', linewidth=1, label=r'$u_{Base}$'))
    
    gradient_patch = mpatches.Patch(label=r'$u_{DD}$')
    legend_elements.append(gradient_patch)
    
    handler_map = {gradient_patch: GradientHandler()}
    ax.legend(handles=legend_elements, loc='lower left', handler_map=handler_map)
    
    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, orientation='vertical', fraction=0.025, pad=0.05, shrink=0.8)
    cbar.set_label(r'Data-Driven Update Interval (Time Steps)')
    
    def init():
        """Initialize animation."""
        for line_top, line_bot in lines_dd:
            line_top.set_data([], [])
            line_bot.set_data([], [])
        for line in lines_gt:
            line.set_data([], [])
        for line in lines_baseline:
            line.set_data([], [])
        time_text.set_text('')
        return [time_text] + [l for pair in lines_dd for l in pair] + lines_gt + lines_baseline
    
    def animate(frame):
        """Update animation frame."""
        # Update data-driven lines
        for i, result in enumerate(results):
            if frame < len(result['top_disp_x']):
                lines_dd[i][0].set_data(result['X'], result['top_disp_x'][frame])
                lines_dd[i][1].set_data(result['X'], result['bot_disp_x'][frame])
        
        # Update ground truth
        if X_gt is not None and frame < len(gt_top_x):
            lines_gt[0].set_data(X_gt, gt_top_x[frame])
            lines_gt[1].set_data(X_gt, gt_bot_x[frame])
        
        # Update baseline
        if X_baseline is not None and frame < len(baseline_top_x):
            lines_baseline[0].set_data(X_baseline, baseline_top_x[frame])
            lines_baseline[1].set_data(X_baseline, baseline_bot_x[frame])
        
        # Update time text
        current_time = gt_times[frame] if frame < len(gt_times) else 0
        time_text.set_text(f'Time: {current_time:.2e} s\nStep: {gt_steps[frame] if frame < len(gt_steps) else 0}')
        
        return [time_text] + [l for pair in lines_dd for l in pair] + lines_gt + lines_baseline
    
    # Create animation
    frame_interval_ms = 1000.0 / animation_fps
    anim = animation.FuncAnimation(
        fig,
        animate,
        init_func=init,
        frames=num_frames,
        interval=frame_interval_ms,
        blit=True,
        repeat=True,
    )
    
    # Save animation
    figures_dir = "figures_and_plots"
    if not os.path.exists(figures_dir):
        os.makedirs(figures_dir)
    
    output_file = os.path.join(figures_dir, 'parameter_study_animation.mp4')
    print(f"\nSaving animation to: {output_file}")
    print("This may take several minutes...")
    
    # Save as MP4 using ffmpeg
    Writer = animation.writers['ffmpeg']
    # Use QuickTime/PowerPoint-friendly encoding (H.264 + yuv420p + web-playback flags)
    writer = Writer(
        fps=animation_fps,
        bitrate=3000,
        codec='libx264',
        extra_args=['-pix_fmt', 'yuv420p', '-profile:v', 'baseline', '-level', '3.1', '-movflags', 'faststart']
    )
    
    try:
        anim.save(output_file, writer=writer, dpi=150)
        print(f"\n✓ Animation saved successfully: {output_file}")
        print(f"  Frames: {num_frames}")
        print(f"  Duration: ~{num_frames/animation_fps:.1f} seconds at {animation_fps} fps")
    except Exception as e:
        print(f"\n✗ Error saving animation: {e}")
        print("\nTrying GIF format as fallback...")
        output_file_gif = os.path.join(figures_dir, 'parameter_study_animation.gif')
        anim.save(output_file_gif, writer='pillow', fps=10, dpi=100)
        print(f"✓ GIF saved: {output_file_gif}")
    
    plt.close()
