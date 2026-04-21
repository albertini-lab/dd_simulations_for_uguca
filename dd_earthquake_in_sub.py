#!/usr/bin/env python3
"""
Input-file generator for dd_earthquake simulations.

Modelled after rs_sw_comparison_in_sub.py but adapted for:
  - dd_earthquake.cc (new uguca, no type-prefix syntax)
  - plain key = value input format (no 'string'/'double'/etc.)
  - local output folders instead of cluster paths

Usage
-----
Edit the parameter dicts below and run:

    python3 dd_earthquake_in_sub.py

Each call to generate() writes a <sname>.in file.
"""
import os
import numpy as np

# ---------------------------------------------------------------------------
# Input file templates
# ---------------------------------------------------------------------------

_base_template = """\
simulation_name = {SNAME}
dump_folder     = {DDIR}

# geometry
length      = {L}
nb_elements = {N}
free_surface = {free_surface}

# time
duration = {T}
time_step_factor = {dtfct}
nb_predictor_corrector = {nb_pc}

# material
E   = {E}
nu  = {nu}
rho = {rho}

# load
shear_load  = {load_T}
normal_load = {load_N}

# nucleation
nuc_center = {nloc}
nuc_size   = {nsize}
nuc_time   = {ntime}
nuc_dtau   = {ndtau}

# asperity
asperity = {asperity}
"""

_asperity_template = """\
asp_center    = {asp_center}
asp_fct       = {asp_fct}
asp_size      = {asp_size}
box_car_shape = {box_car_shape}
"""

_output_template = """\

# output
dump_interval = {dtout}
dump_fields   = {fldout}
"""

_rs_template = """\

# interface law
Law          = RateAndState
a            = {a}
b            = {b}
Dc           = {Dc}
V0           = {V0}
f0           = {f0}
V_init       = {V_init}
evolution_law = {evolution_law}
"""

_dd_template = """\

# data-driven mode
enable_data_driven_mode = True
ground_truth_data_path  = {gt_path}
ground_truth_data_name  = {gt_name}
w_factor_constant       = {w_factor}
noise_stddev_sim        = {noise_stddev}
data_driven_update_interval = {dd_update_interval}
"""


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------

def generate(info, out_dir=None):
    """Write <sname>.in (and optionally create the dump folder).

    Parameters
    ----------
    info : dict
        Simulation parameters (see ref_info below for all keys).
    out_dir : str, optional
        Directory where the .in file is written.  Defaults to the
        current working directory.
    """
    info = dict(info)  # don't mutate the caller's dict

    # resolve dump folder
    if "DDIR" not in info:
        info["DDIR"] = "./simulation_outputs/{SNAME}/".format(**info)

    if out_dir is None:
        out_dir = "."
    os.makedirs(out_dir, exist_ok=True)

    fname = os.path.join(out_dir, "{SNAME}.in".format(**info))

    with open(fname, "w") as f:
        f.write(_base_template.format(**info))
        if info.get("asperity", False):
            f.write(_asperity_template.format(**info))
        f.write(_output_template.format(**info))

        law = info.get("Law", "RateAndState")
        if law == "RateAndState":
            f.write(_rs_template.format(**info))
        else:
            raise ValueError(f"Unsupported Law: {law}. dd_earthquake only supports RateAndState.")

        if info.get("enable_data_driven_mode", False):
            f.write(_dd_template.format(**info))

    print(f"wrote {fname}")
    return fname


# ---------------------------------------------------------------------------
# Default / reference parameter set  (Cornell 3m Granite, Rate-and-State)
# ---------------------------------------------------------------------------

normal_load = -10e6
a = 0.011
b = 0.014
Dc = 1e-6
mu = 46e9 / (2 * (1 + 0.19))  # shear modulus

Lnuc = mu * Dc / (b - a) / abs(normal_load)
print(f"critical nucleation length h* = {Lnuc:.4f} m")

ref_info = {
    # identification
    "SNAME": "baseline",
    # "DDIR": "./simulation_outputs/baseline/",  # inferred from SNAME if absent

    # geometry
    "L": 6.0,
    "N": 512,
    "free_surface": False,

    # time
    "T": 5e-3,
    "dtfct": 0.1,
    "nb_pc": 2,

    # material  (Cornell 3m Granite)
    "E":   46e9,
    "nu":  0.19,
    "rho": 2670.0,

    # loading
    "load_T": 5.622e6,
    "load_N": normal_load,

    # nucleation
    "nloc":  0.865,
    "nsize": 0.1,
    "ntime": 2.5e-3,
    "ndtau": 2e6,

    # asperity (only used when asperity=True)
    "asperity":     False,
    "asp_center":   1.0,
    "asp_fct":      3.0,
    "asp_size":     0.2,
    "box_car_shape": False,

    # output
    "dtout":  1e-6,
    "fldout": "top_disp,top_velo,cohesion",

    # interface law
    "Law":          "RateAndState",
    "a":            a,
    "b":            b,
    "Dc":           Dc,
    "V0":           1e-6,
    "f0":           0.6,
    "V_init":       1e-12,
    "evolution_law": "Slip",

    # data-driven mode (only used when enable_data_driven_mode=True)
    "enable_data_driven_mode": False,
    "gt_path":            "",
    "gt_name":            "",
    "w_factor":           1.0,
    "noise_stddev":       0.0,
    "dd_update_interval": 1,
}


# ---------------------------------------------------------------------------
# Example sweeps  (uncomment / adapt as needed)
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    import copy

    build_sim_dir = os.path.join(os.path.dirname(__file__), "..", "build", "simulations")
    build_sim_dir = os.path.normpath(build_sim_dir)

    # ------------------------------------------------------------------
    # Single baseline run (matches baseline_run016.in)
    # ------------------------------------------------------------------
    info = copy.deepcopy(ref_info)
    info["SNAME"] = "baseline_run016"
    info["load_T"] = 5622448.97959
    info["nuc_center"] if False else None   # keep nloc = 0.865
    generate(info, out_dir=build_sim_dir)

    # ------------------------------------------------------------------
    # Shear-load sweep
    # ------------------------------------------------------------------
    tau0_norms = [0.450, 0.475, 0.500, 0.525, 0.550]
    for tau0norm in tau0_norms:
        info = copy.deepcopy(ref_info)
        info["load_T"] = -normal_load * tau0norm
        info["SNAME"] = "rs_slip_{:05.3f}".format(tau0norm).replace(".", "p")
        generate(info, out_dir=build_sim_dir)

    # ------------------------------------------------------------------
    # Asperity case
    # ------------------------------------------------------------------
    info = copy.deepcopy(ref_info)
    info["SNAME"]     = "rs_asp_ref"
    info["asperity"]  = True
    info["asp_center"] = 1.0
    info["asp_fct"]   = 3.0
    info["asp_size"]  = 0.2
    info["nloc"]      = 1.5
    generate(info, out_dir=build_sim_dir)
