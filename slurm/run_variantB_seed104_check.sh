#!/bin/bash
#SBATCH --job-name=variantB-s104
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
  local lambda="$2"

  echo "=================================================="
  echo "Starting ${name}"
  echo "lambda=${lambda}"
  echo "=================================================="

  export RUN_NAME="${name}"
  export VARIANT="B"
  export TRAIN_MODE="superclip_recon"
  export LAMBDA_CLIP="1.0"
  export LAMBDA_TOKEN_CLS="1.0"
  export LAMBDA_RECON="${lambda}"
  export MASK_RATIO="0.15"
  export LR="1e-5"
  export BATCH_SIZE="128"
  export EPOCHS="10"
  export SEED="104"
  export EVAL_MAX_IMAGES="5000"
  export SAVE_STRATEGY="last_and_best"
  export KEEP_LAST_K="1"
  export SAVE_DIR="./checkpoints/final_checks/${name}"
  export RESULTS_FILE="./results/final_checks/${name}.json"

  bash slurm/run_one_experiment.sh | tee "./logs/${name}.log"
}

run_one confirm_reconB_l0p1_s104_b128_e10 0.1
run_one confirm_reconB_l0p5_s104_b128_e10 0.5

echo "=== variant B extra-seed check complete: $(date) ==="
