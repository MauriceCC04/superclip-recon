#!/bin/bash
# ============================================================
# Submit the main-study experiments as THREE INDEPENDENT jobs.
#
# This replaces the old chained run_main_experiments.sh so that:
#   - each experiment can be restarted independently
#   - a single failure doesn't kill the other two
#   - we stay inside the 24h time budget per experiment
#
# Usage:
#   bash slurm/submit_main_experiments.sh
#   bash slurm/submit_main_experiments.sh baseline variant_a    # subset
# ============================================================

set -euo pipefail

# Resolve project root + cache env vars consistently
source "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/common.sh"
cd "$PROJECT_ROOT"

ALL_TARGETS=("baseline" "variant_a" "variant_b")
TARGETS=("${@:-${ALL_TARGETS[@]}}")

echo "Submitting main experiments: ${TARGETS[*]}"

for name in "${TARGETS[@]}"; do
    case "$name" in
        baseline)
            export RUN_NAME=baseline
            export VARIANT=A
            export LAMBDA_RECON=0.0
            export MASK_RATIO=0.15
            ;;
        variant_a)
            export RUN_NAME=variant_a
            export VARIANT=A
            export LAMBDA_RECON=0.5
            export MASK_RATIO=0.15
            ;;
        variant_b)
            export RUN_NAME=variant_b
            export VARIANT=B
            export LAMBDA_RECON=0.5
            export MASK_RATIO=0.15
            ;;
        *)
            echo "Unknown target: $name (expected one of: ${ALL_TARGETS[*]})"
            exit 1
            ;;
    esac

    export SAVE_DIR="./checkpoints/$RUN_NAME"
    export RESULTS_FILE="./results/$RUN_NAME.json"
    export SAVE_STRATEGY=last_and_best
    export KEEP_LAST_K=1

    echo ">>> Submitting $RUN_NAME (variant=$VARIANT, lambda=$LAMBDA_RECON)"
    sbatch --job-name="superclip-$RUN_NAME" slurm/run_one_experiment.sh
done

echo ""
echo "Done. Check status with: squeue -u \$USER"
