#!/bin/sh
set -eu

PROJECT_DIR="/Users/xiaoyuhe/Joint-Problem"
cd "${PROJECT_DIR}"

export PYTHONPATH="${PYTHONPATH:-}:${PROJECT_DIR}"
export KMP_DUPLICATE_LIB_OK=TRUE
export GRB_LICENSE_FILE="${GRB_LICENSE_FILE:-/Users/xiaoyuhe/gurobi.lic}"

mkdir -p reports/figures

CMD="uv run python -m mixture_lds.experiments --multirun --config-name=config-cluster experiment='MIP4Cluster'"

# EEG-10-100-5
${CMD} \
  problem="EEG" \
  problem.eeg_mode="test" \
  problem.n_normal="5" \
  problem.n_seizure="5" \
  problem.time_points="100" \
  problem.channels="5" \
  result_dir="Result_EEG_10x100x5_solve_methods" \
  solver="if_gurobi,if,em,dtw,fft" \
  solver.regularization="270" \
  solver.thresh="0.25" \
  solver.time_limit="360" \
  solver.gap="0.01" \
  solver.hidden_dim="3" \
  seed="range(30,35)"

# Plot EEG
${CMD} \
  plot="true" \
  problem="EEG" \
  result_path="./Result_EEG_10x100x5_solve_methods/" \
  solver="if_gurobi,if,em,dtw,fft" \
  plot.cutdown="0"
