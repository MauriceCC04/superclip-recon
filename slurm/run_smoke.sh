#!/bin/bash
#SBATCH --job-name=superclip-smoke
#SBATCH --account=3202029
#SBATCH --partition=stud
#SBATCH --qos=stud
#SBATCH --output=out/%x_%j.out
#SBATCH --error=err/%x_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=3202029@studbocconi.it
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:4g.40gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"

echo "=== Smoke job started: $(date) ==="
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

mkdir -p checkpoints/smoke results/smoke out err

echo "=== Sanity check ==="
python sanity_check.py --coco_root ./data/coco --vocab_path ./vocab.json

echo "=== Short smoke training + quick retrieval ==="
python tests/smoke_test.py \
    --coco_root ./data/coco \
    --vocab_path ./vocab.json \
    --steps 3 \
    --batch_size 16 \
    --eval_images 64 \
    --save_path ./checkpoints/smoke/smoke_step.pt \
    --results_file ./results/smoke/smoke_results.json

echo "=== Smoke job complete: $(date) ==="