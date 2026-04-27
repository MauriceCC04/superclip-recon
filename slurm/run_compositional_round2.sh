#!/bin/bash
#SBATCH --job-name=compositional-r2
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

mkdir -p results/compositional_round2 checkpoints/compositional_round2 logs out err

pick_best_ckpt() {
  local dir="$1"
  mapfile -t files < <(find "$dir" -maxdepth 1 -type f -name 'epoch_*.pt' | sort -V)
  if [ "${#files[@]}" -eq 0 ]; then
    echo "ERROR: no checkpoints found in $dir" >&2
    return 1
  fi
  if [ "${#files[@]}" -eq 1 ]; then
    echo "${files[0]}"
    return 0
  fi
  echo "${files[${#files[@]}-2]}"
}

run_aro() {
  local name="$1"
  local ckpt_dir="$2"
  local ckpt
  ckpt="$(pick_best_ckpt "$ckpt_dir")"

  echo "=================================================="
  echo "ARO eval: ${name}"
  echo "Checkpoint dir: ${ckpt_dir}"
  echo "Using checkpoint: ${ckpt}"
  echo "=================================================="

  python eval_compositional.py \
    --checkpoint "${ckpt}" \
    --benchmark aro \
    --output "./results/compositional_round2/${name}_aro.json" \
    | tee "./logs/${name}_aro.log"
}

run_train() {
  local name="$1"
  local train_mode="$2"
  local lambda="$3"
  local seed="$4"

  echo "=================================================="
  echo "Training: ${name}"
  echo "train_mode=${train_mode} lambda=${lambda} seed=${seed}"
  echo "=================================================="

  export RUN_NAME="${name}"
  export VARIANT="A"
  export TRAIN_MODE="${train_mode}"
  export LAMBDA_CLIP="1.0"
  export LAMBDA_TOKEN_CLS="1.0"
  export LAMBDA_RECON="${lambda}"
  export MASK_RATIO="0.15"
  export LR="1e-5"
  export BATCH_SIZE="128"
  export EPOCHS="10"
  export SEED="${seed}"
  export EVAL_MAX_IMAGES="5000"
  export SAVE_STRATEGY="last_and_best"
  export KEEP_LAST_K="1"
  export SAVE_DIR="./checkpoints/compositional_round2/${name}"
  export RESULTS_FILE="./results/compositional_round2/${name}.json"

  bash slurm/run_one_experiment.sh | tee "./logs/${name}.log"
}

# Existing confirm6 seed 102 pair
run_aro \
  baseline_confirm_s102_l0_varA \
  ./checkpoints/confirm6/confirm_baseline_varA_l0_s102_b128_e10

run_aro \
  reconA_confirm_s102_l1_varA \
  ./checkpoints/confirm6/confirm_reconA_l1_s102_b128_e10

# New strict matched training pair, seed 103
run_train \
  compositional_baseline_varA_l0_s103_b128_e10 \
  superclip_baseline \
  0.0 \
  103

run_train \
  compositional_reconA_l1_s103_b128_e10 \
  superclip_recon \
  1.0 \
  103

# ARO on the new seed 103 pair
run_aro \
  baseline_newpair_s103_l0_varA \
  ./checkpoints/compositional_round2/compositional_baseline_varA_l0_s103_b128_e10

run_aro \
  reconA_newpair_s103_l1_varA \
  ./checkpoints/compositional_round2/compositional_reconA_l1_s103_b128_e10

echo "=== compositional round2 complete: $(date) ==="
