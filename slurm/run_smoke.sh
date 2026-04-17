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

# Resolve PROJECT_ROOT + cache env from script location
source "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/common.sh"
cd "$PROJECT_ROOT"

print_job_header

activate_env

ensure_data_file "data/coco/train2017"
ensure_data_file "data/coco/val2017"
ensure_data_file "data/coco/annotations/captions_val2017.json"
ensure_data_file "vocab.json"

mkdir -p checkpoints/smoke results/smoke

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
