#!/bin/bash
# ============================================================
# Shared helper for all SuperCLIP-Recon slurm scripts.
#
# Usage (from a slurm script):
#   source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
#   cd "$PROJECT_ROOT"
#
# What this script does:
#   1. Resolves PROJECT_ROOT from the location of the slurm script
#      (NOT from $SLURM_SUBMIT_DIR — that's the directory where sbatch
#      was run, which may be wrong on a cluster).
#   2. Exports controlled cache / temp env vars under $HOME so nothing
#      writes to random system paths and blows out quotas.
#   3. Creates all the directories the project expects.
#   4. Defines helper functions: activate_env, ensure_data.
# ============================================================

# --- 1. Resolve PROJECT_ROOT from the script's own location ---
# BASH_SOURCE[0] is the sourced script; we want its parent's parent
# (slurm/common.sh → slurm → repo_root).
_COMMON_SH_PATH="${BASH_SOURCE[0]}"
_COMMON_SH_DIR="$(cd "$(dirname "$_COMMON_SH_PATH")" && pwd)"
export PROJECT_ROOT="$(cd "$_COMMON_SH_DIR/.." && pwd)"

echo "[common.sh] PROJECT_ROOT=$PROJECT_ROOT"

# --- 2. Cache / temp env vars under $HOME (HPC-safe) ---
# All library caches go under $HOME/.cache/superclip so we can see
# and clean them up. /tmp on compute nodes is often tiny.
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$HOME/.cache}"
export TORCH_HOME="${TORCH_HOME:-$HOME/.cache/torch}"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
export WANDB_DIR="${WANDB_DIR:-$HOME/.cache/wandb}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$HOME/.cache/matplotlib}"
export TMPDIR="${TMPDIR:-$HOME/.cache/tmp}"

# Disable pip cache to avoid growth; if user wants it, they can override.
export PIP_NO_CACHE_DIR="${PIP_NO_CACHE_DIR:-1}"

# Make sure the directories exist
mkdir -p \
    "$XDG_CACHE_HOME" \
    "$TORCH_HOME" \
    "$HF_HOME" \
    "$TRANSFORMERS_CACHE" \
    "$HF_DATASETS_CACHE" \
    "$WANDB_DIR" \
    "$MPLCONFIGDIR" \
    "$TMPDIR"

# --- 3. Project directories ---
mkdir -p \
    "$PROJECT_ROOT/out" \
    "$PROJECT_ROOT/err" \
    "$PROJECT_ROOT/logs" \
    "$PROJECT_ROOT/checkpoints" \
    "$PROJECT_ROOT/results" \
    "$PROJECT_ROOT/results/preflight" \
    "$PROJECT_ROOT/results/smoke" \
    "$PROJECT_ROOT/results/pilot" \
    "$PROJECT_ROOT/results/ablations" \
    "$PROJECT_ROOT/results/figures"

# --- 4. Helper functions ---

# Activate conda env 'superclip' (or whatever ENV_NAME is set to).
activate_env() {
    local env_name="${ENV_NAME:-superclip}"
    if ! command -v module >/dev/null 2>&1; then
        echo "[common.sh] 'module' command not found; assuming conda already on PATH"
    else
        module load miniconda3 2>/dev/null || \
          echo "[common.sh] WARN: 'module load miniconda3' failed (module may have different name)"
    fi

    if command -v conda >/dev/null 2>&1; then
        eval "$(conda shell.bash hook)"
    else
        echo "[common.sh] ERROR: conda not available after module load"
        return 1
    fi

    conda activate "$env_name" || {
        echo "[common.sh] ERROR: could not activate conda env '$env_name'"
        return 1
    }

    echo "[common.sh] Activated env: $env_name ($(which python))"
}

# Abort with a helpful message if required data is missing.
# Usage: ensure_data_file ./vocab.json
ensure_data_file() {
    local f="$1"
    if [ ! -e "$f" ]; then
        echo "[common.sh] ERROR: required file/dir not found: $f"
        echo "  Did you run: bash slurm/setup_env_ready.sh"
        return 1
    fi
}

# Print a header with environment info for debuggability.
print_job_header() {
    echo "========================================"
    echo "Job:        ${SLURM_JOB_NAME:-unknown}"
    echo "Job ID:     ${SLURM_JOB_ID:-unknown}"
    echo "Node:       $(hostname)"
    echo "Partition:  ${SLURM_JOB_PARTITION:-unknown}"
    echo "Started:    $(date)"
    echo "CWD:        $(pwd)"
    echo "PROJECT:    $PROJECT_ROOT"
    echo "Python:     $(which python 2>/dev/null || echo 'not yet activated')"
    echo "========================================"
}
