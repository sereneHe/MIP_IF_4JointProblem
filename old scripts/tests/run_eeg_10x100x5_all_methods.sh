#!/bin/sh
set -eu

cd /Users/xiaoyuhe/Joint-Problem
export GRB_LICENSE_FILE=/Users/xiaoyuhe/gurobi.lic
export PATH=/Users/xiaoyuhe/Joint-Problem/.venv.before-flatten-20260416-220804/bin:$PATH

/Users/xiaoyuhe/Joint-Problem/.venv.before-flatten-20260416-220804/bin/python - <<'PY'
import numpy as np
import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, '/Users/xiaoyuhe/Joint-Problem/scripts')
from MIP4_Kcluster import MIP4KCluster

base = Path('/Users/xiaoyuhe/Joint-Problem/data/EEG')
X = np.load(base / 'eeg_seizure_3D_normalized.npy', allow_pickle=True)
y = pd.read_csv(base / 'seizure_label.csv').iloc[:, 0].to_numpy()

normal_idx = np.where(y == 0)[0][:5]
seizure_idx = np.where(y == 1)[0][:5]
idx = np.r_[normal_idx, seizure_idx]

X_sub = X[idx, :100, :5]
y_sub = y[idx]

print('subset shapes:', X_sub.shape, y_sub.shape)
print('labels:', y_sub.tolist())

runner = MIP4KCluster()
runner.Test_MIP4KCluster(
    data_in=X_sub,
    label_in=y_sub,
    met=['IF-Gurobi', 'IF', 'EM', 'DTW', 'FFT'],
    name='EEG_10x100x5_all_methods',
    regularization=1,
    seed_times=5,
    seed_start=30,
    thresh=0.25,
)
PY
