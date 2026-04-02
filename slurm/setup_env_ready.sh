#!/bin/bash
# ============================================================
# One-time environment setup for HPC.
#
# Recommended usage:
#   bash slurm/setup_env_ready.sh
#
# This script:
#   1. creates/activates the conda env
#   2. installs Python dependencies
#   3. pre-caches OpenCLIP weights on home storage
#   4. downloads COCO
#   5. builds vocab + phrase files
# ============================================================

set -euo pipefail

cd "$(dirname "$0")/.."

echo "=== SuperCLIP-Recon HPC setup ==="
echo "Started: $(date)"

module load miniconda3
if command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook)"
else
    echo "ERROR: conda not available after module load"
    exit 1
fi

if conda info --envs | grep -q "superclip"; then
    echo "Conda env 'superclip' already exists"
else
    echo "Creating conda env 'superclip'"
    conda create -n superclip python=3.12 -y
fi
conda activate superclip

echo "Python: $(which python)"
python --version

echo "=== Installing requirements ==="
pip install -r requirements.txt
python -m spacy download en_core_web_sm

echo "=== Pre-caching OpenCLIP ==="
python slurm/cache_clip.py

echo "=== Downloading COCO ==="
mkdir -p data/coco
bash download_coco.sh

echo "=== Building token vocabulary ==="
python build_vocab.py --coco_root ./data/coco --top_k 1000 --output vocab.json

echo "=== Extracting phrases for Variant B ==="
python extract_phrases.py --coco_root ./data/coco --output phrases.json

mkdir -p out err logs checkpoints results

echo ""
echo "=== Setup complete: $(date) ==="
echo ""
echo "Next steps:"
echo "  sbatch slurm/run_smoke.sh"
echo "  sbatch slurm/run_main_experiments.sh"
echo "  sbatch slurm/run_ablations.sh"