#!/bin/bash
#SBATCH --job-name=superclip-gpusmoke
#SBATCH --account=3202029
#SBATCH --partition=stud
#SBATCH --qos=stud
#SBATCH --output=out/%x_%j.out
#SBATCH --error=err/%x_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=3202029@studbocconi.it
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:4g.40gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G

# Gate 2 — Cluster GPU smoke.
# Runs sanity_check.py + smoke_test.py and writes a structured JSON
# under results/smoke/ with wall time, peak GPU memory, checkpoint size,
# and retrieval metrics.

set -euo pipefail

if [ "${ALLOW_LEGACY:-0}" != "1" ]; then
    echo "=============================================================="
    echo "LEGACY SCRIPT — older smoke harness."
    echo "=============================================================="
    echo "Preferred workflow (docs/HPC_RUNBOOK.md Gate 2):"
    echo "    sbatch slurm/run_gpu_smoke.sh"
    echo ""
    echo "It writes results/smoke/gpu_smoke_results.json with peak"
    echo "memory, checkpoint size, and retrieval metrics — which is"
    echo "what the runbook inspects."
    echo ""
    echo "Override: ALLOW_LEGACY=1 sbatch slurm/run_smoke.sh"
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

mkdir -p checkpoints/smoke results/smoke

echo "=== Sanity check ==="
python sanity_check.py --coco_root ./data/coco --vocab_path ./vocab.json

echo "=== GPU smoke training + quick retrieval ==="
# smoke_test.py writes results_file including peak mem, ckpt size, eval time
python tests/smoke_test.py \
    --coco_root ./data/coco \
    --vocab_path ./vocab.json \
    --steps 5 \
    --batch_size 32 \
    --eval_images 128 \
    --save_path ./checkpoints/smoke/smoke_step.pt \
    --results_file ./results/smoke/gpu_smoke_results.json

echo "=== GPU smoke complete: $(date) ==="
cat ./results/smoke/gpu_smoke_results.json | head -60
