#!/bin/bash

# Usage: bash train.sh <experiment_id> <device>

# bash train.sh 1103_twist2 cuda:0


REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

cd legged_gym/legged_gym/scripts

robot_name="g1"
exptid=$1
device=$2

task_name="${robot_name}_stu_future"
proj_name="${robot_name}_stu_future"

# ---------------------------------------------------------------------
# IsaacGym dynamic loader fix:
# gym_38.so depends on libpython3.8.so.1.0, which is inside the conda env's
# lib/ directory. When running via plain `bash train.sh ...`, LD search paths
# may not include it (or `python` may point to base env), causing:
#   ImportError: libpython3.8.so.1.0: cannot open shared object file
# ---------------------------------------------------------------------
TWIST2_ENV_PREFIX="${TWIST2_ENV_PREFIX:-/home/heng/miniconda3/envs/twist2}"

# Allow override: TWIST2_PYTHON=/path/to/python bash train.sh ...
PYTHON_BIN="${TWIST2_PYTHON:-}"
if [[ -z "${PYTHON_BIN}" ]]; then
    # Prefer the dedicated twist2 env (Python 3.8 for IsaacGym). Users often have
    # CONDA_PREFIX pointing to base, whose Python may be 3.1x and incompatible.
    if [[ -n "${CONDA_PREFIX:-}" && "${CONDA_PREFIX##*/}" == "twist2" && -x "${CONDA_PREFIX}/bin/python" ]]; then
        PYTHON_BIN="${CONDA_PREFIX}/bin/python"
    elif [[ -x "${TWIST2_ENV_PREFIX}/bin/python" ]]; then
        PYTHON_BIN="${TWIST2_ENV_PREFIX}/bin/python"
    else
        PYTHON_BIN="$(command -v python)"
    fi
fi

# Ensure the matching conda lib/ is visible to the dynamic linker (libpython3.8.so.1.0).
PY_PREFIX="$(cd "$(dirname "${PYTHON_BIN}")/.." && pwd)"
if [[ -d "${PY_PREFIX}/lib" ]]; then
    export LD_LIBRARY_PATH="${PY_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
fi
# Ensure build tools installed in the env (e.g. ninja) are discoverable.
if [[ -d "${PY_PREFIX}/bin" ]]; then
    export PATH="${PY_PREFIX}/bin:${PATH}"
fi

# Ensure repo-local IsaacGym python package is importable even if not pip-installed.
# (README suggests: `cd isaacgym/python && pip install -e .`, but PYTHONPATH works too.)
if [[ -d "${REPO_ROOT}/isaacgym/python" ]]; then
    export PYTHONPATH="${REPO_ROOT}/isaacgym/python:${PYTHONPATH:-}"
fi

# Disable Weights & Biases by default to avoid interactive `wandb login` in
# headless/scripted runs. Set USE_WANDB=1 to enable.
WANDB_FLAG="--no_wandb"
if [[ "${USE_WANDB:-0}" == "1" ]]; then
    WANDB_FLAG=""
fi

# Run the training script
"${PYTHON_BIN}" train.py --task "${task_name}" \
                --proj_name "${proj_name}" \
                --exptid "${exptid}" \
                --device "${device}" \
                --teacher_exptid "None" \
                ${WANDB_FLAG} \
                # --resume \
                # --debug \