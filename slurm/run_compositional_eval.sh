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

# Optional compositional evaluation (Winoground requires HF_TOKEN).

set -euo pipefail

source "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/common.sh"
cd "$PROJECT_ROOT"

print_job_header
activate_env

mkdir -p results

for dir in checkpoints/baseline checkpoints/variant_a checkpoints/variant_b; do
    NAME=$(basename "$dir")
    # Find best/last checkpoint
    LAST_CKPT=$(ls -t "$dir"/epoch_*.pt 2>/dev/null | head -1 || true)
    if [ -n "$LAST_CKPT" ]; then
        echo "Compositional eval: $NAME -> $LAST_CKPT"
        python eval_compositional.py \
            --checkpoint "$LAST_CKPT" \
            --benchmark all \
            --output "./results/compositional_${NAME}.json" \
            ${HF_TOKEN:+--hf_token "$HF_TOKEN"}
    else
        echo "No checkpoint found for $NAME — skipping"
    fi
done

echo "=== Compositional eval complete: $(date) ==="
