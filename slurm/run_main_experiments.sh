#!/bin/bash
#SBATCH --job-name=superclip-main
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

# Main study only:
#   - baseline
#   - Variant A
#   - Variant B
# No ablation sweep here.

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"

echo "=== Main experiments job started: $(date) ==="
echo "Node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"

module load miniconda3
if command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook)"
else
    echo "ERROR: conda not available after module load"
    exit 1
fi
conda activate superclip

if [ ! -d "data/coco/train2017" ]; then
    echo "ERROR: COCO data not found. Run bash slurm/setup_env_ready.sh first."
    exit 1
fi
if [ ! -f "vocab.json" ]; then
    echo "ERROR: vocab.json not found. Run bash slurm/setup_env_ready.sh first."
    exit 1
fi
if [ ! -f "phrases.json" ]; then
    echo "ERROR: phrases.json not found. Run bash slurm/setup_env_ready.sh first."
    exit 1
fi

mkdir -p checkpoints results

echo "=== Sanity check ==="
python sanity_check.py --coco_root ./data/coco --vocab_path ./vocab.json

echo "=== Baseline ==="
python train.py \
    --lambda_recon 0.0 \
    --epochs 10 \
    --batch_size 128 \
    --lr 1e-5 \
    --save_dir ./checkpoints/baseline \
    --run_name baseline \
    --results_file ./results/baseline.json

echo "=== Variant A ==="
echo "=== Main experiments complete: $(date) ==="