#!/usr/bin/env python3
import os
import runpy
import sys

sys.argv[0] = os.path.join(os.path.dirname(__file__), "automation", "ranking", "compute_baseline_vs_exp_rmse.py")
runpy.run_path(sys.argv[0], run_name="__main__")
