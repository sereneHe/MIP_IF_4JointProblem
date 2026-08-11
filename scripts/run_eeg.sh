#!/bin/sh
set -eu

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(CDPATH= cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_DIR}"

if [ -n "${VIRTUAL_ENV:-}" ] && [ "${VIRTUAL_ENV}" != "${PROJECT_DIR}/.venv" ]; then
  unset VIRTUAL_ENV
fi

export PYTHONPATH="${PYTHONPATH:-}:${PROJECT_DIR}/src"
export KMP_DUPLICATE_LIB_OK=TRUE
export GRB_LICENSE_FILE="${GRB_LICENSE_FILE:-/Users/xiaoyuhe/gurobi.lic}"

PYTHON_BIN="${PROJECT_DIR}/.venv/bin/python"
CMD="${PYTHON_BIN} -m mixture_lds.experiments --multirun --config-name=config-cluster experiment='MIP4Cluster'"

${CMD} \
  problem="EEG" \
  solver="if_gurobi,if_gurobi_k,if,em,fft,dtw,mosek"
