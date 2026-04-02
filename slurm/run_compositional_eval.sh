#!/bin/bash
#SBATCH --job-name=superclip-comp
#SBATCH --account=3202029
#SBATCH --partition=stud
#SBATCH --qos=stud
#SBATCH --output=out/%x_%j.out
#SBATCH --error=err/%x_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=3202029@studbocconi.it
#SBATCH --time=08:00:00
#SBATCH --gres=gpu:4g.40gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G

# Optional compositional evaluation.
# Requires HF_TOKEN if you want Winoground.

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"

module load miniconda3
if command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook)"
else
    echo "ERROR: conda not available after module load"
    exit 1
fi
conda activate superclip

mkdir -p results

for dir in checkpoints/baseline checkpoints/variant_a checkpoints/variant_b; do
    NAME=$(basename "$dir")
    LAST_CKPT=$(ls -t "$dir"/epoch_*.pt 2>/dev/null | head -1)
    if [ -n "$LAST_CKPT" ]; then
        echo "Compositional eval: $NAME"
        python eval_compositional.py \
            --checkpoint "$LAST_CKPT" \
            --benchmark all \
            --output "./results/compositional_${NAME}.json" \
            ${HF_TOKEN:+--hf_token "$HF_TOKEN"}
    fi
done