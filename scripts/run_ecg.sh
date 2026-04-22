#!/bin/sh
set -eu

PROJECT_DIR="/Users/xiaoyuhe/Joint-Problem"
cd "${PROJECT_DIR}"

export PYTHONPATH="${PYTHONPATH:-}:${PROJECT_DIR}"
export KMP_DUPLICATE_LIB_OK=TRUE
export GRB_LICENSE_FILE="${GRB_LICENSE_FILE:-/Users/xiaoyuhe/gurobi.lic}"

mkdir -p reports/figures

CMD="uv run python -m mixture_lds.experiments --multirun --config-name=config-cluster experiment='MIP4Cluster'"

# ECG
${CMD} \
  problem="ECG" \
  result_dir="Result_ECG" \
  solver="if_gurobi" \
  solver.regularization="270" \
  solver.thresh="0.25" \
  seed="30"

# Plot ECG
${CMD} \
  plot="true" \
  problem="ECG" \
  result_path="./Result_ECG/" \
  solver="if_gurobi" \
  plot.cutdown="0"
