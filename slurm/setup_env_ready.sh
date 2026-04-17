#!/bin/bash
# ============================================================
# Modular, resumable setup for SuperCLIP-Recon on HPC.
#
# Usage:
#   bash slurm/setup_env_ready.sh                # run all steps in order
#   bash slurm/setup_env_ready.sh --step env     # only conda env + pip
#   bash slurm/setup_env_ready.sh --step cache   # only pre-cache OpenCLIP
#   bash slurm/setup_env_ready.sh --step data    # only download COCO
#   bash slurm/setup_env_ready.sh --step vocab   # only build vocab.json
#   bash slurm/setup_env_ready.sh --step phrases # (OPTIONAL) extract phrases.json —
#                                                  NOT used by training; inspection only
#   bash slurm/setup_env_ready.sh --spacy        # use spaCy for phrases
#                                                  (default: regex — HPC-friendly,
#                                                  no extra install, recommended).
#                                                  --spacy additionally installs
#                                                  the en_core_web_sm model and
#                                                  needs internet access during
#                                                  `--step env`.
#
# Each step is idempotent and can be re-run independently.
# ============================================================

set -euo pipefail

source "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/common.sh"
cd "$PROJECT_ROOT"

STEP="all"
USE_SPACY=0

while [[ $# -gt 0 ]]; do
    case $1 in
        --step)   STEP="$2"; shift 2 ;;
        --spacy)  USE_SPACY=1; shift ;;
        -h|--help)
            grep '^#' "$0" | head -25
            exit 0 ;;
        *)
            echo "Unknown arg: $1"; exit 1 ;;
    esac
done

echo "=== SuperCLIP-Recon setup ==="
echo "Step:     $STEP"
echo "Use spaCy: $USE_SPACY"
echo "Started:  $(date)"

# ---------- Step: env ----------
do_env() {
    echo "[env] Setting up conda env 'superclip'..."
    if ! command -v module >/dev/null 2>&1; then
        echo "[env] WARN: module system not found; assuming conda is on PATH"
    else
        module load miniconda3 || echo "[env] WARN: module load miniconda3 failed"
    fi

    if command -v conda >/dev/null 2>&1; then
        eval "$(conda shell.bash hook)"
    else
        echo "[env] ERROR: conda not on PATH"; exit 1
    fi

    if conda info --envs | grep -q "^superclip"; then
        echo "[env] Env 'superclip' already exists"
    else
        conda create -n superclip python=3.12 -y
    fi
    conda activate superclip
    echo "[env] Active python: $(which python)"
    python --version

    pip install -r requirements.txt
    if [ "$USE_SPACY" = "1" ]; then
        # spaCy is not in requirements.txt (kept optional for HPC);
        # install it only when the user explicitly opts in.
        pip install "spacy>=3.7.0,<3.9.0"
        python -m spacy download en_core_web_sm
    fi
    echo "[env] Done."
}

# ---------- Step: cache ----------
do_cache() {
    echo "[cache] Pre-caching OpenCLIP weights..."
    activate_env
    python slurm/cache_clip.py
    echo "[cache] Done."
}

# ---------- Step: data ----------
do_data() {
    echo "[data] Downloading COCO..."
    if [ -d "./data/coco/train2017" ] && [ -d "./data/coco/val2017" ] \
       && [ -f "./data/coco/annotations/captions_train2017.json" ] \
       && [ -f "./data/coco/annotations/captions_val2017.json" ]; then
        echo "[data] COCO already present, skipping."
        return
    fi
    mkdir -p data/coco
    bash download_coco.sh
    echo "[data] Done."
}

# ---------- Step: vocab ----------
do_vocab() {
    echo "[vocab] Building token vocabulary..."
    if [ -f "./vocab.json" ]; then
        echo "[vocab] vocab.json already exists — re-running (will overwrite)"
    fi
    activate_env
    python build_vocab.py --coco_root ./data/coco --top_k 1000 --output vocab.json
    echo "[vocab] Done."
}

# ---------- Step: phrases ----------
do_phrases() {
    echo "[phrases] Extracting phrases (OPTIONAL — not used by training)..."
    echo "[phrases] Variant B training extracts phrases inline per caption."
    echo "[phrases] phrases.json is only produced for inspection/debugging."
    if [ -f "./phrases.json" ]; then
        echo "[phrases] phrases.json already exists — re-running (will overwrite)"
    fi
    activate_env
    if [ "$USE_SPACY" = "1" ]; then
        echo "[phrases] Using spaCy extractor"
        python extract_phrases.py --coco_root ./data/coco --output phrases.json
    else
        echo "[phrases] Using regex extractor (HPC default — faster, no spaCy model)"
        python extract_phrases.py --coco_root ./data/coco --use_regex \
            --output phrases.json
    fi
    echo "[phrases] Done."
}

# ---------- Dispatch ----------
case "$STEP" in
    env)      do_env ;;
    cache)    do_cache ;;
    data)     do_data ;;
    vocab)    do_vocab ;;
    phrases)  do_phrases ;;
    all)
        do_env
        do_cache
        do_data
        do_vocab
        # do_phrases is OPTIONAL and not used by training. Run it explicitly via
        # `bash slurm/setup_env_ready.sh --step phrases` if you want phrases.json
        # for debugging or inspection.
        ;;
    *)
        echo "Unknown step: $STEP"; exit 1 ;;
esac

echo ""
echo "=== Setup step '$STEP' complete: $(date) ==="
echo ""
if [ "$STEP" = "all" ]; then
    echo "Next steps:"
    echo "  sbatch slurm/run_preflight.sh         # gate 1 (static+runtime preflight)"
    echo "  sbatch slurm/run_gpu_smoke.sh         # gate 2 (cluster GPU smoke)"
    echo "  sbatch slurm/run_pilot_baseline.sh    # gate 3 (1-epoch pilot)"
    echo "  bash slurm/submit_main_experiments.sh # gate 4 (3 independent jobs)"
    echo "  bash slurm/submit_ablations.sh        # gate 5 (ablation grid)"
fi
