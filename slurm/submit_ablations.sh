#!/bin/bash
# ============================================================
# Submit the ablation grid as independent jobs.
#
# This avoids the old "one 24h chained job" topology.  Each ablation
# runs in its own job so queue time and storage are both smaller.
#
# Grid (dedup-aware; reuse of main experiments handled locally via
# run_ablations.py, not here):
#   lambda sweep:       lambda in {0.1, 1.0}         (var=A, mr=0.15)
#     (0.0 == baseline, 0.5 == variant_a — already submitted)
#   maskrate sweep:     mr     in {0.10, 0.25}       (var=A, lambda=0.5)
#     (0.15 == variant_a — already submitted)
#   variant comparison: not needed here — submit_main_experiments.sh runs
#                       variant_a and variant_b separately.
#
# By default we submit ONLY the unique ablation configs that are NOT
# already covered by the main runs.  Pass --all to also resubmit those.
#
# Usage:
#   bash slurm/submit_ablations.sh
#   bash slurm/submit_ablations.sh --all
# ============================================================

set -euo pipefail

source "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/common.sh"
cd "$PROJECT_ROOT"

INCLUDE_REDUNDANT=0
if [ "${1:-}" = "--all" ]; then
    INCLUDE_REDUNDANT=1
fi

# Default settings for ablation runs — shorter to fit cluster time budget.
# Override by exporting before calling this script.
export EPOCHS="${EPOCHS:-5}"
export BATCH_SIZE="${BATCH_SIZE:-128}"
export LR="${LR:-1e-5}"
export EVAL_MAX_IMAGES="${EVAL_MAX_IMAGES:-2500}"
export SAVE_STRATEGY=last_and_best
export KEEP_LAST_K=1

submit_exp() {
    local name="$1" variant="$2" lam="$3" mr="$4"
    export RUN_NAME="$name"
    export VARIANT="$variant"
    export LAMBDA_RECON="$lam"
    export MASK_RATIO="$mr"
    export SAVE_DIR="./checkpoints/ablations/$name"
    export RESULTS_FILE="./results/ablations/$name.json"
    mkdir -p "$(dirname "$RESULTS_FILE")" "$SAVE_DIR"
    echo ">>> Submit ablation $name  variant=$variant  lambda=$lam  mr=$mr"
    sbatch --job-name="superclip-abl-$name" slurm/run_one_experiment.sh
}

# --- Lambda sweep (Variant A, mr=0.15) ---
submit_exp "lambda_0.1" A 0.1  0.15
submit_exp "lambda_1.0" A 1.0  0.15
if [ "$INCLUDE_REDUNDANT" = "1" ]; then
    submit_exp "lambda_0.0" A 0.0 0.15
    submit_exp "lambda_0.5" A 0.5 0.15
fi

# --- Masking-rate sweep (Variant A, lambda=0.5) ---
submit_exp "maskrate_0.10" A 0.5 0.10
submit_exp "maskrate_0.25" A 0.5 0.25
if [ "$INCLUDE_REDUNDANT" = "1" ]; then
    submit_exp "maskrate_0.15" A 0.5 0.15
fi

echo ""
echo "Ablation jobs submitted. Monitor with: squeue -u \$USER"
