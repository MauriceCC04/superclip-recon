#!/bin/bash
#SBATCH --job-name=superclip-recon
#SBATCH --account=3202029
#SBATCH --partition=stud
#SBATCH --qos=stud
#SBATCH --output=out/%x_%j.out
#SBATCH --error=err/%x_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=3202029@studbocconi.it
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:4g.40gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

# ============================================================
# SuperCLIP-Recon: Full training + evaluation pipeline
# Bocconi HPC (Jupiter I) — A100 MIG 40GB
#
# Prerequisites: run slurm/setup_env.sh ONCE before this.
#
# Usage:
#   sbatch slurm/run_pipeline.sh
#
# Monitor:
#   squeue -u 3202029
#   tail -f out/superclip-recon_<jobid>.out
#
# Cancel:
#   scancel <jobid>
# ============================================================

set -e

echo "=== Job started: $(date) ==="
echo "Node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"

# --- Activate environment ---
module load miniconda3
conda activate superclip

echo "Python: $(which python)"
echo "GPU:"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "N/A"

# --- Verify data exists ---
if [ ! -d "data/coco/train2017" ]; then
    echo "ERROR: COCO data not found. Run slurm/setup_env.sh first."
    exit 1
fi
if [ ! -f "vocab.json" ]; then
    echo "ERROR: vocab.json not found. Run slurm/setup_env.sh first."
    exit 1
fi

mkdir -p checkpoints results

# --- Step 1: Sanity check ---
echo ""
echo "=== Step 1: Sanity check ==="
python sanity_check.py --coco_root ./data/coco --vocab_path ./vocab.json

# --- Step 2: Train baseline (lambda=0) ---
echo ""
echo "=== Step 2: Train baseline ==="
python train.py \
    --lambda_recon 0.0 \
    --epochs 10 \
    --batch_size 128 \
    --lr 1e-5 \
    --save_dir ./checkpoints/baseline \
    --run_name baseline \
    --results_file ./results/baseline.json

# --- Step 3: Train Variant A (masked token prediction) ---
echo ""
echo "=== Step 3: Train Variant A ==="
python train.py \
    --lambda_recon 0.5 \
    --variant A \
    --mask_ratio 0.15 \
    --epochs 10 \
    --batch_size 128 \
    --lr 1e-5 \
    --save_dir ./checkpoints/variant_a \
    --run_name variant_a \
    --results_file ./results/variant_a.json

# --- Step 4: Train Variant B (phrase reconstruction) ---
echo ""
echo "=== Step 4: Train Variant B ==="
python train.py \
    --lambda_recon 0.5 \
    --variant B \
    --mask_ratio 0.15 \
    --epochs 10 \
    --batch_size 128 \
    --lr 1e-5 \
    --phrase_path ./phrases.json \
    --save_dir ./checkpoints/variant_b \
    --run_name variant_b \
    --results_file ./results/variant_b.json

# --- Step 5: Ablation sweep ---
echo ""
echo "=== Step 5: Ablation sweep ==="
python run_ablations.py \
    --coco_root ./data/coco \
    --vocab_path ./vocab.json \
    --phrase_path ./phrases.json \
    --results_dir ./results/ablations

# --- Step 6: Final retrieval evaluation ---
echo ""
echo "=== Step 6: Final evaluation ==="
for dir in checkpoints/baseline checkpoints/variant_a checkpoints/variant_b; do
    if [ -d "$dir" ]; then
        LAST_CKPT=$(ls -t "$dir"/epoch_*.pt 2>/dev/null | head -1)
        if [ -n "$LAST_CKPT" ]; then
            echo "Evaluating $LAST_CKPT"
            python evaluate.py --checkpoint "$LAST_CKPT" --coco_root ./data/coco
        fi
    fi
done

# --- Step 7: Compositional evaluation ---
echo ""
echo "=== Step 7: Compositional evaluation ==="
for dir in checkpoints/baseline checkpoints/variant_a checkpoints/variant_b; do
    NAME=$(basename "$dir")
    LAST_CKPT=$(ls -t "$dir"/epoch_*.pt 2>/dev/null | head -1)
    if [ -n "$LAST_CKPT" ]; then
        echo "Compositional eval: $NAME"
        python eval_compositional.py \
            --checkpoint "$LAST_CKPT" \
            --benchmark all \
            --output ./results/compositional_${NAME}.json \
            ${HF_TOKEN:+--hf_token "$HF_TOKEN"}
    fi
done

# --- Step 8: Generate analysis plots ---
echo ""
echo "=== Step 8: Results analysis ==="
python analyze_results.py --results_dir ./results --output_dir ./results/figures

echo ""
echo "=== All done: $(date) ==="
echo "Results:  ./results/"
echo "Figures:  ./results/figures/"
echo "Checkpoints: ./checkpoints/"
echo ""
echo "To view results, copy them to your local machine:"
echo "  scp -r 3202029@slogin.hpc.unibocconi.it:~/superclip-recon/results ./"
