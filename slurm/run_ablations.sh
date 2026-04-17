#!/bin/bash
#SBATCH --job-name=superclip-ablate-chained
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
# LEGACY CHAINED ABLATION RUNNER
#
# **Preferred** workflow on this cluster is to use
# slurm/submit_ablations.sh which submits each ablation as an
# independent job.
#
# Keeping this only for local-style use of run_ablations.py with
# its main-result reuse logic.
# ============================================================

set -euo pipefail

if [ "${ALLOW_LEGACY:-0}" != "1" ]; then
    echo "=============================================================="
    echo "LEGACY SCRIPT — chained ablation runner."
    echo "=============================================================="
    echo "Preferred workflow (docs/HPC_RUNBOOK.md Gate 5):"
    echo "    bash slurm/submit_ablations.sh"
    echo ""
    echo "It submits each ablation point as an independent job and"
    echo "automatically skips configs already covered by main runs."
    echo ""
    echo "If you really want the in-process runner (run_ablations.py"
    echo "with its main-result reuse logic), use:"
    echo "    ALLOW_LEGACY=1 sbatch slurm/run_ablations.sh"
    echo "=============================================================="
    exit 1
fi

source "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/common.sh"
cd "$PROJECT_ROOT"

print_job_header
activate_env

ensure_data_file "data/coco/train2017"
ensure_data_file "data/coco/val2017"
ensure_data_file "data/coco/annotations/captions_val2017.json"
ensure_data_file "vocab.json"

mkdir -p results/ablations

python run_ablations.py \
    --coco_root ./data/coco \
    --vocab_path ./vocab.json \
    --phrase_path ./phrases.json \
    --results_dir ./results/ablations \
    --main_results_dir ./results \
    --epochs 5

echo "=== Ablations complete: $(date) ==="
