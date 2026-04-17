#!/usr/bin/env python
"""
postprocess_named.py

A modified version of the standard postprocess.py script designed to handle 
simulation names that are arbitrary strings (e.g. data-driven experiments) 
rather than the strict "Basename_IntegerID" format.

Usage:
    python3 postprocess_named.py <simulation_name_string> [new_name]
"""

import sys
import time
import glob
import os
import shutil as shtl
import filecmp

# Import ifasha data manager
# Assumes 'ifasha' is in PYTHONPATH, similar to the original script
try:
    import ifasha.datamanager as idm
except ImportError:
    print("\nCRITICAL ERROR: Could not import 'ifasha'")
    print("Please ensure your PYTHONPATH includes the 'ifasha' library.")
    print("Example: export PYTHONPATH=$PYTHONPATH:/path/to/libraries\n")
    sys.exit(1)


def postprocess(sname, mode='interactive', new_sname=None):
    modes = ['interactive', 'save', 'forced']
    if mode not in modes:
        raise ValueError(f'Choose a possible mode: {modes}')

    # -------------------------------------------------------------------------
    # 1. Locate the Progress File
    # -------------------------------------------------------------------------
    
    # Define search directories
    sdirs = []
    if os.path.exists('source_directories.txt'):
        sdirs = [line.strip() for line in open('source_directories.txt')]
    
    # Add sensible defaults for this workspace
    default_paths = ['.', 'simulation_outputs', 'results', '../simulation_outputs']
    for p in default_paths:
        if p not in sdirs:
            sdirs.append(p)

    print(f"Searching for simulation '{sname}'...")
    print(f"Search directories: {sdirs}")

    # Look for {sname}.progress
    ofiles = []
    found_sdir = None
    
    for s in sdirs:
        # Construct pattern: path/sname.progress
        pat = os.path.join(s, f"{sname}.progress")
        found = glob.glob(pat)
        ofiles.extend(found)
        if found:
            found_sdir = s
    
    if not ofiles:
        print(f"ERROR: No progress file found for: {sname}")
        print(f"       Expected file: {sname}.progress")
        return False, None
    
    # If duplicates found, warn but use first
    if len(ofiles) > 1:
        print(f"WARNING: Multiple progress files found. Using: {ofiles[0]}")
    
    ofile = ofiles[0]
    print(f"Found: {ofile}")

    # -------------------------------------------------------------------------
    # 2. Parse Progress File for Metadata
    # -------------------------------------------------------------------------
    
    dumper_sname = ''
    bname_sep = ''
    outputfolder = ''
    groups = set()
    code = 'unknown'

    try:
        with open(ofile, 'r') as fl:
            for line in fl:
                parts = line.strip().split()
                if not parts: continue
                
                key = parts[0]
                if key == 'dumper_bname':
                    dumper_sname = parts[2]
                elif key == 'bname_sep':
                    bname_sep = parts[2]
                elif key == 'dumper_group':
                    groups.add(parts[2])
                elif key == 'output_folder':
                    outputfolder = parts[2]
                elif key == 'simulation_code':
                    code = parts[2]
                    
    except Exception as e:
        print(f"Error reading progress file: {e}")
        return False, None

    print(f"  Internal Name: {dumper_sname}")
    print(f"  Code: {code}")
    print(f"  Groups: {groups}")

    # -------------------------------------------------------------------------
    # 3. Determine Dump Path (Where binary files are)
    # -------------------------------------------------------------------------
    
    dumppath = ofile 
    
    # Check if header points to a valid path
    if outputfolder and os.path.exists(outputfolder):
        dumppath = outputfolder
    elif outputfolder and os.path.exists(os.path.join(found_sdir, outputfolder)):
        dumppath = os.path.join(found_sdir, outputfolder)
    else:
        # Fallback: Files are usually in the same dir as .progress
        dumppath = os.path.dirname(ofile)

    if not dumppath.endswith(os.path.sep):
        dumppath += os.path.sep
        
    print(f"  Dump Path: {dumppath}")

    # -------------------------------------------------------------------------
    # 4. Locate Input File (.in)
    # -------------------------------------------------------------------------
    
    input_file = os.path.join(os.path.dirname(ofile), f"{sname}.in")
    if not os.path.exists(input_file):
        input_file = os.path.join(dumppath, f"{sname}.in")
        
    if not os.path.exists(input_file):
        print(f"WARNING: Input file not found ({sname}.in)")
        print(f"         Checked: {os.path.dirname(ofile)} and {dumppath}")
        input_file = None
        if mode == 'save': return False, None
    else:
        print(f"  Input File: {input_file}")

    # -------------------------------------------------------------------------
    # 5. Initialize DataManager
    # -------------------------------------------------------------------------
    
    target_name = new_sname if new_sname else sname
    wdir = './data/'
    
    if not os.path.exists(wdir):
        os.makedirs(wdir)
        
    # Check if DM exists
    exists = False
    try:
        dm = idm.DataManager(target_name, wdir, False)
        exists = True
    except IOError:
        # Does not exist, create new
        dm = idm.DataManager(target_name, wdir, True)

    if exists:
        if mode == 'interactive':
            print(f"DataManager '{target_name}' already exists in {wdir}")
            resp = input("Overwrite? [y/N]: ").strip().lower()
            if resp == 'y':
                dm.destroy()
                dm = idm.DataManager(target_name, wdir, True)
            else:
                print("Skipping.")
                return False, None
        elif mode == 'forced':
            dm.destroy()
            dm = idm.DataManager(target_name, wdir, True)
        elif mode == 'save':
            return False, None

    # -------------------------------------------------------------------------
    # 6. Import Data
    # -------------------------------------------------------------------------

    # Add supplementary files
    if input_file:
        dm.add_supplementary(f"{target_name}.in", input_file, True)
    
    dm.add_supplementary(f"{target_name}.out", ofile, True)

    # Copy any other associated files (.sub, .run, etc)
    prefix = input_file.strip('.in') if input_file else os.path.join(os.path.dirname(ofile), sname)
    for ext in ['.sub', '.run']:
        sup = prefix + ext
        if os.path.exists(sup):
            dm.add_supplementary(f"{target_name}{ext}", sup, True)

    # Process Dump Groups
    for group in groups:
        if group in dm:
            dm.remove_field_collection(group)
            
        fc = dm.get_new_field_collection(group)
        fc.sim_info = f"sim-name={sname}" # Storing name instead of ID
        
        # Select Reader
        if code == 'weak-interface':
            fcioh = idm.FieldCollectionWeakInterface(fc)
        else:
            fcioh = idm.FieldCollectionIOHelper(fc)
            
        # Info file format: {InternalName}{Sep}{Group}.info
        info_file = f"{dumper_sname}{bname_sep}{group}.info"
        
        print(f"  Reading Group '{group}' via info file: {info_file}")
        try:
            fcioh.read_simulation_output(info_file, dumppath)
        except Exception as e:
            print(f"    ERROR reading group {group}: {e}")
            
    print(f"\nSUCCESS: Postprocessed '{sname}' -> '{target_name}'")
    return True, target_name


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 postprocess_named.py <simulation_name> [new_name]")
        sys.exit(1)
        
    sname = sys.argv[1]
    new_name = sys.argv[2] if len(sys.argv) > 2 else None
    
    postprocess(sname, 'interactive', new_name)
