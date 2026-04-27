#!/bin/bash
#SBATCH --job-name=confirm6-seq
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

set -euo pipefail

cd /mnt/beegfsstudents/home/3202029/superclip-recon
source slurm/common.sh
cd "$PROJECT_ROOT"

print_job_header
activate_env

mkdir -p results/confirm6 checkpoints/confirm6 logs out err

run_one () {
  local name="$1"
  local train_mode="$2"
  local lambda="$3"
  local seed="$4"

  echo "=================================================="
  echo "Starting ${name}"
  echo "train_mode=${train_mode} lambda=${lambda} seed=${seed}"
  echo "=================================================="

  export RUN_NAME="${name}"
  export VARIANT="A"
  export TRAIN_MODE="${train_mode}"
  export LAMBDA_RECON="${lambda}"
  export MASK_RATIO="0.15"
  export SEED="${seed}"
  export BATCH_SIZE="128"
  export EPOCHS="10"
  export EVAL_MAX_IMAGES="5000"
  export SAVE_STRATEGY="last_and_best"
  export KEEP_LAST_K="1"
  export SAVE_DIR="./checkpoints/confirm6/${name}"
  export RESULTS_FILE="./results/confirm6/${name}.json"

  bash slurm/run_one_experiment.sh | tee "./logs/${name}.log"
}

run_one confirm_baseline_varA_l0_s101_b128_e10 superclip_baseline 0.0 101
run_one confirm_reconA_l0p5_s101_b128_e10 superclip_recon 0.5 101
run_one confirm_baseline_varA_l0_s102_b128_e10 superclip_baseline 0.0 102
run_one confirm_reconA_l0p5_s102_b128_e10 superclip_recon 0.5 102
run_one confirm_reconA_l1_s101_b128_e10 superclip_recon 1.0 101
run_one confirm_reconA_l1_s102_b128_e10 superclip_recon 1.0 102

echo "Confirm6 sequential batch complete"
