#!/usr/bin/env python
from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import glob

# Add MacTeX to PATH for LaTeX rendering
os.environ['PATH'] = '/usr/local/texlive/2023/bin/universal-darwin:' + os.environ.get('PATH', '')

# Matplotlib parameters for consistent styling
plt.rcParams.update({
    'text.usetex': True,
    'text.latex.preamble': r'\usepackage{lmodern}\usepackage[T1]{fontenc}\usepackage{sansmath}\sansmath',
    'font.family': 'sans-serif',
    'font.sans-serif': ['Latin Modern Sans'],
    'font.size': 11,
    'axes.labelsize': 11,
    'legend.fontsize': 11,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11
})

def find_simulation_files(sim_name):
    """
    Search for simulation files in the organized directory structure.
    Returns a dictionary with the paths to the found files.
    """
    # Search patterns for different locations
    search_patterns = [
        # New organized structure
        f"simulation_outputs/{sim_name}_*/{sim_name}.*",
        f"simulation_outputs/{sim_name}_*/{sim_name}-DataFiles/",
        # Legacy structure (current directory)
        f"{sim_name}.*",
        f"{sim_name}-DataFiles/",
        # Build directory
        f"../{sim_name}.*",
        f"../{sim_name}-DataFiles/",
    ]
    
    found_files = {
        'time_file': None,
        'coords_file': None,
        'data_dir': None,
        'base_path': None
    }
    
    # Search for time and coords files
    for pattern in [f"simulation_outputs/{sim_name}_*", ".", ".."]:
        if os.path.exists(pattern):
            for file in glob.glob(f"{pattern}/{sim_name}.time"):
                if found_files['time_file'] is None:
                    found_files['time_file'] = file
                    found_files['base_path'] = os.path.dirname(file)
                    break
            
            for file in glob.glob(f"{pattern}/{sim_name}.coords"):
                if found_files['coords_file'] is None:
                    found_files['coords_file'] = file
                    if found_files['base_path'] is None:
                        found_files['base_path'] = os.path.dirname(file)
                    break
                    
            for data_dir in glob.glob(f"{pattern}/{sim_name}-DataFiles"):
                if found_files['data_dir'] is None and os.path.isdir(data_dir):
                    found_files['data_dir'] = data_dir
                    break
    
    return found_files

def plot(sim_name, fldname, comp, ax):
    # Find simulation files in organized structure
    files = find_simulation_files(sim_name)
    
    if files['time_file'] is None:
        print(f"Error: Could not find time file for simulation: {sim_name}")
        print("Searched in: simulation_outputs/, current directory, and parent directory")
        return None
        
    if files['coords_file'] is None:
        print(f"Error: Could not find coordinates file for simulation: {sim_name}")
        print("Searched in: simulation_outputs/, current directory, and parent directory")
        return None
        
    if files['data_dir'] is None:
        print(f"Error: Could not find data directory for simulation: {sim_name}")
        print("Searched in: simulation_outputs/, current directory, and parent directory")
        return None

    # Construct file paths using found locations
    time_file = files['time_file']
    coords_file = files['coords_file']
    data_file = os.path.join(files['data_dir'], fldname + ".out")
    num_nodes_expected = 1024

    print(f"Using files:")
    print(f"  Time: {time_file}")
    print(f"  Coords: {coords_file}")
    print(f"  Data: {data_file}")

    # get time data
    Tdata = []
    try:
        with open(time_file, 'r') as fl:
            lines = fl.readlines()
        for line in lines:
            if line.strip():
                Tdata.append(float(line.strip().split()[1]))
    except IOError:
        print(f"Error: Could not read time file: {time_file}")
        return None
    except (ValueError, IndexError) as e:
        print(f"Error parsing time data in {time_file}: {e}")
        return None

    # get space data
    Xdata = []
    try:
        with open(coords_file, 'r') as fl:
            lines = fl.readlines()
        for line in lines:
            if line.strip():
                Xdata.append(float(line.strip().split()[0]))
    except IOError:
        print(f"Error: Could not read coordinates file: {coords_file}")
        return None
    except (ValueError, IndexError) as e:
        print(f"Error parsing coordinate data in {coords_file}: {e}")
        return None

    # get field data
    Vdata = []
    try:
        with open(data_file, 'r') as fl:
            lines = fl.readlines()
        if not lines:
            print(f"Error: Data file {data_file} is empty.")
            return None
        for line in lines:
            if line.strip():
                Vdata.append(list(map(float, line.strip().split())))
        Vdata = np.array(Vdata)
        if Vdata.ndim == 1:
            Vdata = Vdata.reshape(1, -1)
    except IOError:
        print(f"Error: Could not read data file: {data_file}")
        return None
    except ValueError as e:
        print(f"Error parsing data in {data_file}: {e}")
        return None

    if Vdata.shape[1] == 0:
        print(f"Error: No data columns found in {data_file}")
        return None

    # --- Component Extraction Logic ---
    known_scalar_fields = ['tau_max']
    is_scalar = fldname in known_scalar_fields

    if is_scalar:
        if Vdata.shape[1] != num_nodes_expected:
            print(f"Warning: Scalar field '{fldname}' in {data_file} has {Vdata.shape[1]} columns (expected {num_nodes_expected}). Plotting first {num_nodes_expected} if possible.")
            if Vdata.shape[1] < num_nodes_expected:
                print(f"Error: Not enough columns for scalar field '{fldname}'.")
                return None
            Vdata_comp = Vdata[:, :num_nodes_expected]
        else:
            Vdata_comp = Vdata
        if comp != 0:
            print(f"Warning: Component {comp} requested for scalar field '{fldname}'. Using component 0.")
    else:
        total_cols = Vdata.shape[1]

        if total_cols % num_nodes_expected != 0:
            print(f"Error: Total columns ({total_cols}) for field '{fldname}' is not a multiple of expected nodes per component ({num_nodes_expected}).")
            return None
        
        num_components_in_file = total_cols // num_nodes_expected

        if not (0 <= comp < num_components_in_file):
            print(f"Error: Requested component {comp} for field '{fldname}' is out of range. File has {num_components_in_file} component(s).")
            return None
        
        start_col = comp * num_nodes_expected
        end_col = (comp + 1) * num_nodes_expected
        Vdata_comp = Vdata[:, start_col:end_col]
    # --- End Component Extraction ---

    # plot - Only plot if multiple time steps exist
    if len(Tdata) > 1 and Vdata_comp.shape[0] > 1:
        if Vdata_comp.shape[1] != len(Xdata):
            print(f"Error: Mismatch between number of nodes in Vdata_comp ({Vdata_comp.shape[1]}) and Xdata ({len(Xdata)}) for field {fldname}.")
            return None

        XV, TV = np.meshgrid(Xdata, Tdata)
        if Vdata_comp.shape[0] != len(Tdata):
            print(f"Warning: Mismatch in time steps between field data ({Vdata_comp.shape[0]}) and time data ({len(Tdata)}). Truncating to shorter length.")
            min_len = min(Vdata_comp.shape[0], len(Tdata))
            Vdata_comp = Vdata_comp[:min_len, :]
            TV = TV[:min_len, :]

        pc = ax.pcolor(XV, TV, Vdata_comp, shading='nearest')
        return pc
    elif len(Tdata) == 1 or Vdata_comp.shape[0] == 1:
        print(f"Info: Only one time step found for {fldname}. Skipping pcolor plot. To plot a line, use a different script.")
        return None
    else:
         print(f"Info: No time steps found to plot for {fldname}.")
         return None


# -----------------------------------------------------------------------------
if __name__ == "__main__":
    if len(sys.argv) not in [4]:
        sys.exit('Usage: ./time_int_opt_plot.py <sim_name> <fieldname> <comp>\n'
                 + '  sim_name: Base name of the simulation (e.g., time_int_opt_gt)\n'
                 + '  fieldname: cohesion, top_disp, bot_disp, tau_max\n'
                 + '  comp: 0 (for x/scalar) or 1 (for y)')

    sim_name = str(sys.argv[1])
    fldname = str(sys.argv[2])
    comp = int(sys.argv[3])

    print(f"\nPlotting '{fldname}' (Component {comp}) for simulation '{sim_name}'...")

    fig = plt.figure()
    ax = fig.add_subplot(111)

    pc = plot(sim_name, fldname, comp, ax)

    if pc:
        cbar = fig.colorbar(pc)
        cbar.set_label(f"{fldname} (comp {comp})")
    ax.set_xlabel('x')
    ax.set_ylabel('t')
    plt.title(f'{fldname} (Component {comp}) for {sim_name}')
    plt.show()

    print("Plot window closed.")
