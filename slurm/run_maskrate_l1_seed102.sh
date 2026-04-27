#!/bin/bash
#SBATCH --job-name=maskrate-l1-s102
#SBATCH --account=3202029
#SBATCH --partition=stud
#SBATCH --qos=stud
#SBATCH --exclude=gnode04
#SBATCH --output=out/%x_%j.out
#SBATCH --error=err/%x_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=3202029@studbocconi.it
#SBATCH --time=06:00:00
#SBATCH --gres=gpu:4g.40gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

set -euo pipefail

cd /mnt/beegfsstudents/home/3202029/superclip-recon
source slurm/common.sh
cd "$PROJECT_ROOT"

print_job_header
activate_env

mkdir -p results/final_checks checkpoints/final_checks logs out err

run_one () {
  local name="$1"
  local mask_ratio="$2"

  echo "=================================================="
  echo "Starting ${name}"
  echo "mask_ratio=${mask_ratio}"
  echo "=================================================="

  export RUN_NAME="${name}"
  export VARIANT="A"
  export TRAIN_MODE="superclip_recon"
  export LAMBDA_CLIP="1.0"
  export LAMBDA_TOKEN_CLS="1.0"
  export LAMBDA_RECON="1.0"
  export MASK_RATIO="${mask_ratio}"
  export LR="1e-5"
  export BATCH_SIZE="128"
  export EPOCHS="10"
  export SEED="102"
  export EVAL_MAX_IMAGES="5000"
  export SAVE_STRATEGY="last_and_best"
  export KEEP_LAST_K="1"
  export SAVE_DIR="./checkpoints/final_checks/${name}"
  export RESULTS_FILE="./results/final_checks/${name}.json"

  bash slurm/run_one_experiment.sh | tee "./logs/${name}.log"
}

run_one maskrateA_l1_m30_s102_b128_e10 0.30
run_one maskrateA_l1_m50_s102_b128_e10 0.50

echo "=== mask-rate ablation complete: $(date) ==="
