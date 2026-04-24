#!/bin/bash
#SBATCH --job-name=superclip-lambda-sweep
#SBATCH --account=3202029
#SBATCH --partition=stud
#SBATCH --qos=stud
#SBATCH --exclude=gnode04
#SBATCH --output=out/%x_%j.out
#SBATCH --error=err/%x_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=3202029@studbocconi.it
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:4g.40gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

set -uo pipefail

cd /mnt/beegfsstudents/home/3202029/superclip-recon
source slurm/common.sh
cd "$PROJECT_ROOT"

print_job_header
activate_env

ensure_data_file data/coco/train2017
ensure_data_file data/coco/val2017
ensure_data_file data/coco/annotations/captions_val2017.json
ensure_data_file vocab.json

mkdir -p results/ablations checkpoints/ablations logs

# Allow: bash slurm/run_lambda_sweep.sh A
# or:    VARIANT=B sbatch slurm/run_lambda_sweep.sh
VARIANT="${1:-${VARIANT:-A}}"
if [[ "$VARIANT" != "A" && "$VARIANT" != "B" ]]; then
  echo "Error: VARIANT must be A or B, got '$VARIANT'"
  exit 1
fi

LAMBDAS=("0" "0.1" "0.5" "0.75" "1" "1.5" "2" "5")

SEED="${SEED:-43}"
BATCH_SIZE="${BATCH_SIZE:-128}"
MASK_RATIO="${MASK_RATIO:-0.15}"
LR="${LR:-1e-5}"
EPOCHS="${EPOCHS:-10}"
EVAL_MAX_IMAGES="${EVAL_MAX_IMAGES:-5000}"
TRAIN_MODE="${TRAIN_MODE:-auto}"
LAMBDA_CLIP="${LAMBDA_CLIP:-1.0}"
LAMBDA_TOKEN_CLS="${LAMBDA_TOKEN_CLS:-1.0}"
SAVE_STRATEGY="${SAVE_STRATEGY:-last_and_best}"
KEEP_LAST_K="${KEEP_LAST_K:-1}"
PHRASE_PATH="${PHRASE_PATH:-./phrases.json}"
NUM_WORKERS="${NUM_WORKERS:-4}"
DETERMINISTIC="${DETERMINISTIC:-0}"

SUMMARY="results/ablations/lambda_sweep_variant_${VARIANT}_summary_s${SEED}_b${BATCH_SIZE}.jsonl"
touch "$SUMMARY"

echo "Lambda sweep config:"
echo "  VARIANT=$VARIANT"
echo "  SEED=$SEED"
echo "  BATCH_SIZE=$BATCH_SIZE"
echo "  NUM_WORKERS=$NUM_WORKERS"
echo "  DETERMINISTIC=$DETERMINISTIC"
echo "  MASK_RATIO=$MASK_RATIO"
echo "  LAMBDAS=${LAMBDAS[*]}"
echo "  SUMMARY=$SUMMARY"

for LAMBDA in "${LAMBDAS[@]}"; do
  TAG=$(echo "$LAMBDA" | tr '.' 'p')
  RUN_NAME="lambda_${TAG}_var${VARIANT}_s${SEED}_b${BATCH_SIZE}"
  SAVE_DIR="./checkpoints/ablations/${RUN_NAME}"
  RESULTS_FILE="./results/ablations/${RUN_NAME}.json"
  LOG_FILE="./logs/${RUN_NAME}.log"

  echo "=================================================="
  echo "Starting ${RUN_NAME}"
  echo "  variant=$VARIANT lambda=$LAMBDA"
  echo "=================================================="

  if [ -f "$RESULTS_FILE" ]; then
    echo "Skipping ${RUN_NAME} (results already exist)"
    echo "{\"run_name\":\"${RUN_NAME}\",\"variant\":\"${VARIANT}\",\"lambda\":${LAMBDA},\"seed\":${SEED},\"num_workers\":${NUM_WORKERS},\"deterministic\":${DETERMINISTIC},\"status\":\"skipped_existing\"}" >> "$SUMMARY"
    continue
  fi

  mkdir -p "$SAVE_DIR"

  START_TS=$(date +%s)

  export RUN_NAME
  export VARIANT
  export LAMBDA_RECON="$LAMBDA"
  export MASK_RATIO
  export EPOCHS
  export BATCH_SIZE
  export LR
  export EVAL_MAX_IMAGES
  export TRAIN_MODE
  export LAMBDA_CLIP
  export LAMBDA_TOKEN_CLS
  export SAVE_STRATEGY
  export KEEP_LAST_K
  export PHRASE_PATH
  export SAVE_DIR
  export RESULTS_FILE
  export SEED
  export NUM_WORKERS
  export DETERMINISTIC

  bash slurm/run_one_experiment.sh > "$LOG_FILE" 2>&1
  STATUS=$?

  END_TS=$(date +%s)
  ELAPSED=$((END_TS - START_TS))

  if [ "$STATUS" -eq 0 ] && [ -f "$RESULTS_FILE" ]; then
    echo "{\"run_name\":\"${RUN_NAME}\",\"variant\":\"${VARIANT}\",\"lambda\":${LAMBDA},\"seed\":${SEED},\"num_workers\":${NUM_WORKERS},\"deterministic\":${DETERMINISTIC},\"status\":\"ok\",\"elapsed_sec\":${ELAPSED}}" >> "$SUMMARY"
  else
    echo "{\"run_name\":\"${RUN_NAME}\",\"variant\":\"${VARIANT}\",\"lambda\":${LAMBDA},\"seed\":${SEED},\"num_workers\":${NUM_WORKERS},\"deterministic\":${DETERMINISTIC},\"status\":\"failed\",\"exit_code\":${STATUS},\"elapsed_sec\":${ELAPSED}}" >> "$SUMMARY"
    echo "Run ${RUN_NAME} failed; continuing to next lambda"
  fi
done

echo "Sweep complete"