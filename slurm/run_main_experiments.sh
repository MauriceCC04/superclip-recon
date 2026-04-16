#!/bin/bash
#SBATCH --job-name=superclip-main-chained
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
# LEGACY CHAINED MAIN EXPERIMENTS
#
# This runs baseline + Variant A + Variant B in one job.
# **Preferred** workflow is to use slurm/submit_main_experiments.sh
# which submits them as three independent jobs.
#
# Kept here only for emergencies (e.g. quick total-of-3 run under
# a single job budget).
# ============================================================

set -euo pipefail

source "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/common.sh"
cd "$PROJECT_ROOT"

print_job_header
activate_env

ensure_data_file "data/coco/train2017"
ensure_data_file "vocab.json"

mkdir -p checkpoints results

echo "=== Sanity check ==="
python sanity_check.py --coco_root ./data/coco --vocab_path ./vocab.json

echo "=== Baseline ==="
python train.py \
    --lambda_recon 0.0 --variant A --mask_ratio 0.15 \
    --epochs 10 --batch_size 128 --lr 1e-5 \
    --save_strategy last_and_best --keep_last_k 1 \
    --save_dir ./checkpoints/baseline \
    --run_name baseline \
    --results_file ./results/baseline.json

echo "=== Variant A ==="
python train.py \
    --lambda_recon 0.5 --variant A --mask_ratio 0.15 \
    --epochs 10 --batch_size 128 --lr 1e-5 \
    --save_strategy last_and_best --keep_last_k 1 \
    --save_dir ./checkpoints/variant_a \
    --run_name variant_a \
    --results_file ./results/variant_a.json

echo "=== Variant B ==="
python train.py \
    --lambda_recon 0.5 --variant B --mask_ratio 0.15 \
    --epochs 10 --batch_size 128 --lr 1e-5 \
    --phrase_path ./phrases.json \
    --save_strategy last_and_best --keep_last_k 1 \
    --save_dir ./checkpoints/variant_b \
    --run_name variant_b \
    --results_file ./results/variant_b.json

echo "=== All main experiments complete: $(date) ==="
