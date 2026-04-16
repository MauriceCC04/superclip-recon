#!/bin/bash
#SBATCH --job-name=superclip-preflight
#SBATCH --account=3202029
#SBATCH --partition=stud
#SBATCH --qos=stud
#SBATCH --output=out/%x_%j.out
#SBATCH --error=err/%x_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=3202029@studbocconi.it
#SBATCH --time=00:30:00
#SBATCH --gres=gpu:4g.40gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

set -euo pipefail

source "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/common.sh"
cd "$PROJECT_ROOT"

print_job_header
activate_env

mkdir -p results/preflight

python tools/hpc_preflight.py \
    --coco_root ./data/coco \
    --vocab_path ./vocab.json \
    --phrase_path ./phrases.json \
    --home_quota_gb 100 \
    --output ./results/preflight/preflight_report.json \
    --project_root "$PROJECT_ROOT"

echo "=== Preflight complete: $(date) ==="
cat ./results/preflight/preflight_report.json | head -80
