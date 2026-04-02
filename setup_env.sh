#!/bin/bash
# ============================================================
# ONE-TIME SETUP — Run this on the login node BEFORE submitting jobs.
#
# What it does:
#   1. Creates a conda environment with all dependencies
#   2. Downloads COCO 2017 (~20GB — takes ~15-30 min)
#   3. Builds the token vocabulary
#   4. Extracts phrases for Variant B
#
# Usage:
#   ssh 3202029@slogin.hpc.unibocconi.it
#   cd ~/superclip-recon
#   bash slurm/setup_env.sh
#
# You only need to run this ONCE. After that, just sbatch the pipeline.
# ============================================================

set -e

echo "=== SuperCLIP-Recon: One-time setup ==="
echo "Started: $(date)"

# --- 1. Load miniconda and create environment ---
echo ""
echo "=== Step 1: Create conda environment ==="
module load miniconda3

if conda info --envs | grep -q "superclip"; then
    echo "Conda env 'superclip' already exists, activating..."
    conda activate superclip
else
    echo "Creating conda env 'superclip' with Python 3.12..."
    conda create -n superclip python=3.12 -y
    conda activate superclip
fi

echo "Python: $(which python)"
echo "Version: $(python --version)"

# --- 2. Install dependencies ---
echo ""
echo "=== Step 2: Install Python packages ==="
pip install -r requirements.txt
pip install datasets  # for compositional evaluation

# spaCy model for phrase extraction
python -m spacy download en_core_web_sm

echo ""
echo "=== Step 3: Download COCO 2017 (~20GB) ==="
echo "This will take 15-30 minutes depending on network speed."
mkdir -p data/coco
bash download_coco.sh

# --- 4. Build vocabulary (fast, CPU-only) ---
echo ""
echo "=== Step 4: Build vocabulary ==="
python build_vocab.py --coco_root ./data/coco --top_k 1000 --output vocab.json

# --- 5. Extract phrases (fast, CPU-only) ---
echo ""
echo "=== Step 5: Extract phrases ==="
python extract_phrases.py --coco_root ./data/coco --output phrases.json

# --- 6. Create output directories ---
mkdir -p out err logs checkpoints results

echo ""
echo "=== Setup complete: $(date) ==="
echo ""
echo "Disk usage:"
du -sh data/coco/ vocab.json phrases.json 2>/dev/null
echo ""
echo "Total home usage:"
du -sh ~ 2>/dev/null || echo "N/A"
echo ""
echo "Next step: submit the training pipeline:"
echo "  sbatch slurm/run_pipeline.sh"
