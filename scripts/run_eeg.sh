#!/bin/sh
set -eu

PROJECT_DIR="/Users/xiaoyuhe/Joint-Problem"
cd "${PROJECT_DIR}"

export PYTHONPATH="${PYTHONPATH:-}:${PROJECT_DIR}"
export KMP_DUPLICATE_LIB_OK=TRUE
export GRB_LICENSE_FILE="${GRB_LICENSE_FILE:-/Users/xiaoyuhe/gurobi.lic}"

mkdir -p reports/figures

RAW_ROOT="${PROJECT_DIR}/data/raw/EEG/archive.physionet.org/pn6/chbmit"
EEG_27S="${PROJECT_DIR}/data/processed/EEG/eeg-27s.npy"

if [ -d "${RAW_ROOT}" ]; then
  uv run python -m mixture_lds.data.build_chbmit_eeg \
    --skip-download \
    --raw-root "${RAW_ROOT}" \
    --output-dir "${PROJECT_DIR}/data/processed/EEG" \
    --patients chb01 chb03 chb05
elif [ -f "${EEG_27S}" ]; then
  echo "Raw CHB-MIT directory not found: ${RAW_ROOT}"
  echo "Using existing packed EEG file: ${EEG_27S}"
else
  echo "Missing raw CHB-MIT directory and packed EEG file." >&2
  echo "Expected raw: ${RAW_ROOT}" >&2
  echo "Expected packed: ${EEG_27S}" >&2
  exit 1
fi

CMD="uv run python -m mixture_lds.experiments --multirun --config-name=config-cluster experiment='MIP4Cluster'"

# EEG-CHBMIT 27s packed data
${CMD} \
  problem="EEG" \
  problem.eeg_mode="eeg_27s" \
  result_dir="Result_EEG_27s_solve_methods" \
  solver="if,em,dtw,fft,if_gurobi" \
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
  result_path="./Result_EEG_27s_solve_methods/" \
  solver="if,em,dtw,fft,if_gurobi" \
  plot.cutdown="0"
