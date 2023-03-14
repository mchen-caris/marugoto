#!/usr/bin/env python3

__author__ = "Jeff"
__copyright__ = "Copyright 2023, Kather Lab"
__license__ = "MIT"
__version__ = "0.1.0"
__maintainer__ = ["Jeff"]
__email__ = "jiefu.zhu@tu-dresden.de"
from roc import plot_roc_curves_
import os
from pathlib import Path
out_dir = '/opt/hpe/swarm-learning-hpe/workspace/marugoto_mri/user/data-and-scratch/scratch/2023_02_13_190655_40-30-10-20_swarm_learning/'
in_path = [(os.path.join(out_dir, 'patient-preds.csv'))]
out_path = Path(out_dir)
print(in_path)
print(out_path)
plot_roc_curves_(in_path, outpath=out_path, target_label="Malign",
                 true_label='1', subgroup_label=None, clini_table=None, subgroups=None)
