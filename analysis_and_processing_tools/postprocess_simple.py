#!/usr/bin/env python
from __future__ import print_function, division, absolute_import

import sys
import os

# Ensure ifasha can be imported if it's in a known location relative to this script
# or assume it's in PYTHONPATH

try:
    import ifasha.datamanager as idm
except ImportError:
    print("Error: 'ifasha' module not found. Please ensure it is in your PYTHONPATH.")
    sys.exit(1)

# input file name
def get_input_fname(sname):
    return '{}.in'.format(sname)
# output file name
def get_output_fname(sname):
    return '{}.out'.format(sname)

def add_io_files(dmioh,sdir,new_bname):
    # Try to locate input file in sdir or input_files/
    sdir_in = os.path.join(sdir, new_bname+'.in')
    local_in = os.path.join('input_files', new_bname+'.in')
    
    if os.path.exists(sdir_in):
        input_file = sdir_in
    elif os.path.exists(local_in):
        input_file = local_in
    else:
        # Fallback to current dir
        input_file = new_bname+'.in'
        
    print(f"Adding input file: {input_file}")
    dmioh.add_supplementary(get_input_fname(new_bname),input_file,True)
    
    # Try to locate progress file
    progress_file = os.path.join(sdir,new_bname+'.progress')
    if not os.path.exists(progress_file):
         # Try local
         progress_file = new_bname+'.progress'
         
    if os.path.exists(progress_file):
        print(f"Adding progress file: {progress_file}")
        dmioh.add_supplementary(get_output_fname(new_bname),progress_file,True)
    else:
        print(f"Warning: Progress file not found for {new_bname}")

def main():
    usage="./postprocess.py <raw_data_dir/bname>"
    if len(sys.argv) != 2:
        sys.exit(usage)

    # Parse the path to get simulation directory and basename
    input_path = sys.argv[1]
    
    # Strip trailing slash if present to avoid basename issues
    input_path = input_path.rstrip('/')
    
    if '/' in input_path:
        # Split path into directory and basename
        sdir = os.path.dirname(input_path)
        bname = os.path.basename(input_path)
    else:
        bname = input_path
        sdir = '.'
        
        # Check source_directories.txt if it exists
        if os.path.exists('source_directories.txt'):
             with open('source_directories.txt', 'r') as f:
                 sdirs = [line.strip() for line in f if line.strip()]
             
             print(f"Searching for {bname} in: {sdirs}")
             found = False
             for path in sdirs:
                 if os.path.exists(os.path.join(path, bname + '.progress')):
                     sdir = path
                     print(f"Found in {sdir}")
                     found = True
                     break
             
             if not found and sdir == '.':
                 print(f"Warning: {bname}.progress not found in source_directories.txt paths. Checking local directory.")

    wdir = 'data/'
    if not os.path.exists(wdir):
        os.makedirs(wdir)

    new_bname = bname
    groups = ['interface'] # dd_earthquake default group

    # verify it does not yet exist
    try:
        dmioh = idm.DataManager(new_bname,wdir,False)
    except IOError: # it does not yet exist
        dmioh = idm.DataManager(new_bname,wdir,True)
        # add io files to datamanager
        add_io_files(dmioh,sdir,new_bname)

    else:
        # In automated scripts, we usually want to replace or modify.
        # For simplicity, we'll auto-replace here or ask if interactive.
        if sys.stdin.isatty():
            answer = input('DataManager named "{}" exists already.\n'.format(dmioh.name) 
                            + 'Want to: [m] modify it, [r] replace it, [s] stop here? ')
            answer = answer.strip().lower()
            if answer == 's':
                sys.exit()
            elif answer == 'r':
                dmioh.destroy()
                dmioh = idm.DataManager(new_bname,wdir,True)
                add_io_files(dmioh,sdir,new_bname)
            elif answer == 'm':
                add_io_files(dmioh,sdir,new_bname)
                pass
            else:
                print('Incorrect answer!')
                sys.exit()
        else:
            print(f"DataManager {new_bname} exists. Overwriting (non-interactive mode).")
            dmioh.destroy()
            dmioh = idm.DataManager(new_bname,wdir,True)
            add_io_files(dmioh,sdir,new_bname)


    for group in groups:
        # FieldCollection check
        if group in dmioh:
            if sys.stdin.isatty():
                answer = input('FieldCollection named "{}" exists already.\n'
                               .format(group) + 'Want to: [m] modify it, [r] replace it, [s] stop here? ')
                answer = answer.strip().lower()

                if answer == 's':
                    sys.exit()
                elif answer == 'r':
                    dmioh.remove_field_collection(group)
                    fc = dmioh.get_new_field_collection(group)
                elif answer == 'm':
                    fc = dmioh.get_field_collection(group)
                else:
                    print('Incorrect answer!')
                    sys.exit()
            else:
                print(f"Replacing FieldCollection '{group}' (non-interactive).")
                dmioh.remove_field_collection(group)
                fc = dmioh.get_new_field_collection(group)
        else:
            fc = dmioh.get_new_field_collection(group)

        # "cast" FieldCollection to IOHelperReader
        # Try WeakInterface first if the user used that previously, but standard IOHelper is more generic.
        # The user's sample `postprocess_wi_simple.py` used `FieldCollectionWeakInterface`.
        # We will stick to that to match their provided example.
        fcwi = idm.FieldCollectionWeakInterface(fc)
        
        info_file_1 = os.path.join(sdir, bname + ".info")
        info_file_2 = os.path.join(sdir, bname + "-interface.info")
        
        if os.path.exists(info_file_1):
            print(f"Reading {info_file_1}")
            fcwi.read_simulation_output(bname + '.info', sdir)
        elif os.path.exists(info_file_2):
            print(f"Reading {info_file_2}")
            fcwi.read_simulation_output(bname + '-interface.info', sdir)
        else:
            print(f"Error: No info file found for {bname} in {sdir}")
            print(f"Checked: {info_file_1} and {info_file_2}")

if __name__ == "__main__":
    main()
