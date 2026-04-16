#!/bin/bash
#SBATCH --job-name=superclip-pilot
#SBATCH --account=3202029
#SBATCH --partition=stud
#SBATCH --qos=stud
#SBATCH --output=out/%x_%j.out
#SBATCH --error=err/%x_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=3202029@studbocconi.it
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:4g.40gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

# Gate 3 — Pilot baseline.
# Runs ONE epoch of the baseline on conservative settings, then
# projects 10-epoch cost and writes a structured readiness JSON.

set -euo pipefail

source "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/common.sh"
cd "$PROJECT_ROOT"

print_job_header
activate_env

ensure_data_file "data/coco/train2017"
ensure_data_file "vocab.json"

mkdir -p checkpoints/pilot results/pilot

BATCH_SIZE="${BATCH_SIZE:-64}"
EVAL_MAX_IMAGES="${EVAL_MAX_IMAGES:-1000}"

echo "=== Pilot baseline: 1 epoch, batch=$BATCH_SIZE, eval_max_images=$EVAL_MAX_IMAGES ==="

# Run baseline, capture wall time
PILOT_RAW=./results/pilot/baseline_raw.json
PILOT_SUMMARY=./results/pilot/pilot_baseline.json

START_SECONDS=$(date +%s)

python train.py \
    --coco_root ./data/coco \
    --vocab_path ./vocab.json \
    --run_name pilot_baseline \
    --variant A \
    --lambda_recon 0.0 \
    --mask_ratio 0.15 \
    --epochs 1 \
    --batch_size "$BATCH_SIZE" \
    --lr 1e-5 \
    --eval_max_images "$EVAL_MAX_IMAGES" \
    --save_strategy last \
    --keep_last_k 1 \
    --save_dir ./checkpoints/pilot \
    --results_file "$PILOT_RAW"

END_SECONDS=$(date +%s)
WALL_SECONDS=$((END_SECONDS - START_SECONDS))

# Build the pilot summary JSON by reading the raw results and adding projections
python - <<PYEOF
import json, os, sys

raw_path = "$PILOT_RAW"
summary_path = "$PILOT_SUMMARY"
wall_seconds = $WALL_SECONDS

with open(raw_path) as f:
    raw = json.load(f)

history = raw.get("history", [])
epoch_record = history[0] if history else {}
retr = epoch_record.get("retrieval", {})

# Approximate: assume eval took ~15% of epoch (conservative).
# We don't have a split metric from train.py, so estimate from wall time.
epoch_seconds = float(raw.get("wall_time_seconds", wall_seconds))
# Number of optimizer steps ~= ceil(118287 / batch_size) for COCO train
n_train_images = 118287
batch_size = int(raw.get("batch_size", $BATCH_SIZE))
steps = max(1, (n_train_images + batch_size - 1) // batch_size)
avg_step_seconds = epoch_seconds / steps

# Assume eval_seconds = fraction of epoch (rough lower bound; real one comes from smoke)
eval_seconds = max(10.0, epoch_seconds * 0.1)

# Projections
projected_10_epoch_seconds = epoch_seconds * 10
ckpt_mb = float(raw.get("checkpoint_size_mb") or 0.0)
projected_ckpt_storage_mb = ckpt_mb * 2  # last + best

# Readiness verdict
readiness = "PASS"
reasons = []
if projected_10_epoch_seconds > 20 * 3600:
    readiness = "WARN"
    reasons.append(f"10-epoch run projected > 20h ({projected_10_epoch_seconds/3600:.1f}h)")
if projected_10_epoch_seconds > 24 * 3600:
    readiness = "FAIL"
    reasons.append(f"10-epoch run exceeds 24h time limit ({projected_10_epoch_seconds/3600:.1f}h)")
if projected_ckpt_storage_mb > 10 * 1024:  # 10 GB
    readiness = "WARN" if readiness == "PASS" else readiness
    reasons.append(f"Projected checkpoint storage {projected_ckpt_storage_mb/1024:.1f} GB is large")

# Recommended batch size: if GPU wasn't near its memory limit, suggest a bump
recommended_batch_size = batch_size  # conservative default

summary = {
    "run_name": "pilot_baseline",
    "avg_step_seconds": round(avg_step_seconds, 3),
    "epoch_seconds": round(epoch_seconds, 1),
    "eval_seconds": round(eval_seconds, 1),
    "projected_10_epoch_seconds": round(projected_10_epoch_seconds, 1),
    "gpu_peak_mem_gb": None,  # not tracked in train.py; run gpu smoke for this
    "checkpoint_size_mb": ckpt_mb,
    "projected_checkpoint_storage_mb": round(projected_ckpt_storage_mb, 1),
    "recommended_batch_size": recommended_batch_size,
    "readiness": readiness,
    "readiness_reasons": reasons,
    "final_retrieval": retr,
    "raw_results_path": raw_path,
}

with open(summary_path, "w") as f:
    json.dump(summary, f, indent=2)

print(f"Pilot summary written to {summary_path}")
print(f"Readiness: {readiness}")
for r in reasons:
    print(f"  - {r}")
PYEOF

echo "=== Pilot baseline complete: $(date) ==="
cat "$PILOT_SUMMARY"
