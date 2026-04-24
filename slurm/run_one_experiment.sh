#!/bin/bash
#SBATCH --job-name=superclip-exp
#SBATCH --account=3202029
#SBATCH --partition=stud
#SBATCH --qos=stud
#SBATCH --output=out/%x_%j.out
#SBATCH --error=err/%x_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=3202029@studbocconi.it
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:4g.40gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

set -euo pipefail

source "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/common.sh"
cd "$PROJECT_ROOT"

print_job_header
activate_env

ensure_data_file "data/coco/train2017"
ensure_data_file "data/coco/val2017"
ensure_data_file "data/coco/annotations/captions_val2017.json"
ensure_data_file "vocab.json"

: "${RUN_NAME:?RUN_NAME must be set}"
: "${VARIANT:?VARIANT must be set (A or B)}"
: "${LAMBDA_RECON:?LAMBDA_RECON must be set}"
: "${MASK_RATIO:?MASK_RATIO must be set}"

EPOCHS="${EPOCHS:-10}"
BATCH_SIZE="${BATCH_SIZE:-128}"
LR="${LR:-1e-5}"
TRAIN_MODE="${TRAIN_MODE:-auto}"
LAMBDA_CLIP="${LAMBDA_CLIP:-1.0}"
LAMBDA_TOKEN_CLS="${LAMBDA_TOKEN_CLS:-1.0}"
EVAL_MAX_IMAGES="${EVAL_MAX_IMAGES:-5000}"
SAVE_STRATEGY="${SAVE_STRATEGY:-last_and_best}"
KEEP_LAST_K="${KEEP_LAST_K:-1}"
PHRASE_PATH="${PHRASE_PATH:-./phrases.json}"
SAVE_DIR="${SAVE_DIR:-./checkpoints/$RUN_NAME}"
RESULTS_FILE="${RESULTS_FILE:-./results/$RUN_NAME.json}"
SEED="${SEED:-42}"
NUM_WORKERS="${NUM_WORKERS:-4}"
DETERMINISTIC="${DETERMINISTIC:-0}"

mkdir -p "$(dirname "$RESULTS_FILE")"
mkdir -p "$SAVE_DIR"

echo "Experiment config:"
echo "  RUN_NAME=$RUN_NAME"
echo "  TRAIN_MODE=$TRAIN_MODE"
echo "  VARIANT=$VARIANT"
echo "  LAMBDA_CLIP=$LAMBDA_CLIP"
echo "  LAMBDA_TOKEN_CLS=$LAMBDA_TOKEN_CLS"
echo "  LAMBDA_RECON=$LAMBDA_RECON"
echo "  MASK_RATIO=$MASK_RATIO"
echo "  EPOCHS=$EPOCHS  BATCH=$BATCH_SIZE  LR=$LR"
echo "  SEED=$SEED  NUM_WORKERS=$NUM_WORKERS  DETERMINISTIC=$DETERMINISTIC"
echo "  SAVE=$SAVE_STRATEGY keep=$KEEP_LAST_K"

EXTRA_ARGS=()
if [ "$VARIANT" = "B" ] && [ -f "$PHRASE_PATH" ]; then
    EXTRA_ARGS+=(--phrase_path "$PHRASE_PATH")
fi
if [ "${FREEZE_TEXT_TOWER:-0}" = "1" ]; then
    EXTRA_ARGS+=(--freeze_text_tower)
fi
if [ "${FREEZE_VISION_TOWER:-0}" = "1" ]; then
    EXTRA_ARGS+=(--freeze_vision_tower)
fi
if [ "${FREEZE_LOGIT_SCALE:-0}" = "1" ]; then
    EXTRA_ARGS+=(--freeze_logit_scale)
fi
if [ "${NO_AMP:-0}" = "1" ]; then
    EXTRA_ARGS+=(--no_amp)
fi
if [ "$DETERMINISTIC" = "1" ]; then
    EXTRA_ARGS+=(--deterministic)
fi

python train.py \
    --coco_root ./data/coco \
    --vocab_path ./vocab.json \
    --run_name "$RUN_NAME" \
    --train_mode "$TRAIN_MODE" \
    --variant "$VARIANT" \
    --lambda_clip "$LAMBDA_CLIP" \
    --lambda_token_cls "$LAMBDA_TOKEN_CLS" \
    --lambda_recon "$LAMBDA_RECON" \
    --mask_ratio "$MASK_RATIO" \
    --epochs "$EPOCHS" \
    --batch_size "$BATCH_SIZE" \
    --lr "$LR" \
    --eval_max_images "$EVAL_MAX_IMAGES" \
    --save_strategy "$SAVE_STRATEGY" \
    --keep_last_k "$KEEP_LAST_K" \
    --save_dir "$SAVE_DIR" \
    --results_file "$RESULTS_FILE" \
    --seed "$SEED" \
    --num_workers "$NUM_WORKERS" \
    "${EXTRA_ARGS[@]}"

echo "=== $RUN_NAME complete: $(date) ==="