#!/bin/bash
#SBATCH --job-name=superclip-lambda-sweep
#SBATCH --account=<USER_ID>
#SBATCH --partition=stud
#SBATCH --qos=stud
#SBATCH --exclude=gnode04
#SBATCH --output=out/%x_%j.out
#SBATCH --error=err/%x_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=<USER_ID>@studbocconi.it
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:4g.40gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

set -uo pipefail

cd /mnt/beegfsstudents/home/<USER_ID>/superclip-recon
source slurm/common.sh
cd "$PROJECT_ROOT"

print_job_header
activate_env

ensure_data_file data/coco/train2017
ensure_data_file data/coco/val2017
ensure_data_file data/coco/annotations/captions_val2017.json
ensure_data_file vocab.json

mkdir -p results/ablations checkpoints/ablations logs

LAMBDAS=("0.0" "0.1" "0.5" "1.0")
SEED="${SEED:-43}"
BATCH_SIZE="${BATCH_SIZE:-128}"
MASK_RATIO="${MASK_RATIO:-0.15}"
LR="${LR:-1e-5}"
EPOCHS="${EPOCHS:-10}"
VARIANT="${VARIANT:-A}"
EVAL_MAX_IMAGES="${EVAL_MAX_IMAGES:-5000}"

SUMMARY="results/ablations/lambda_sweep_summary_s${SEED}_b${BATCH_SIZE}.jsonl"
touch "$SUMMARY"

for LAMBDA in "${LAMBDAS[@]}"; do
  TAG=$(echo "$LAMBDA" | tr '.' 'p')
  RUN_NAME="lambda_${LAMBDA}_s${SEED}_b${BATCH_SIZE}"
  SAVE_DIR="./checkpoints/ablations/${RUN_NAME}"
  RESULTS_FILE="./results/ablations/${RUN_NAME}.json"
  LOG_FILE="./logs/${RUN_NAME}.log"

  echo "=================================================="
  echo "Starting ${RUN_NAME}"
  echo "=================================================="

  if [ -f "$RESULTS_FILE" ]; then
    echo "Skipping ${RUN_NAME} (results already exist)"
    echo "{\"run_name\":\"${RUN_NAME}\",\"lambda\":${LAMBDA},\"status\":\"skipped_existing\"}" >> "$SUMMARY"
    continue
  fi

  mkdir -p "$SAVE_DIR"

  START_TS=$(date +%s)

  python train.py \
    --coco_root ./data/coco \
    --vocab_path ./vocab.json \
    --run_name "$RUN_NAME" \
    --variant "$VARIANT" \
    --lambda_recon "$LAMBDA" \
    --mask_ratio "$MASK_RATIO" \
    --epochs "$EPOCHS" \
    --batch_size "$BATCH_SIZE" \
    --lr "$LR" \
    --eval_max_images "$EVAL_MAX_IMAGES" \
    --save_strategy last_and_best \
    --keep_last_k 1 \
    --seed "$SEED" \
    --save_dir "$SAVE_DIR" \
    --results_file "$RESULTS_FILE" \
    > "$LOG_FILE" 2>&1

  STATUS=$?
  END_TS=$(date +%s)
  ELAPSED=$((END_TS - START_TS))

  if [ "$STATUS" -eq 0 ] && [ -f "$RESULTS_FILE" ]; then
    echo "{\"run_name\":\"${RUN_NAME}\",\"lambda\":${LAMBDA},\"status\":\"ok\",\"elapsed_sec\":${ELAPSED}}" >> "$SUMMARY"
  else
    echo "{\"run_name\":\"${RUN_NAME}\",\"lambda\":${LAMBDA},\"status\":\"failed\",\"exit_code\":${STATUS},\"elapsed_sec\":${ELAPSED}}" >> "$SUMMARY"
    echo "Run ${RUN_NAME} failed; continuing to next lambda"
  fi
done

echo "Sweep complete"