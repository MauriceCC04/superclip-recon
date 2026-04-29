#!/bin/bash
#SBATCH --job-name=confirm-s106
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

if [ -z "${HF_TOKEN:-}" ]; then
  echo "ERROR: HF_TOKEN is not set"
  exit 1
fi

mkdir -p results/final_confirm checkpoints/final_confirm logs out err

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

run_train() {
  local name="$1"
  local train_mode="$2"
  local lambda="$3"

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
  export SEED="106"
  export EVAL_MAX_IMAGES="5000"
  export SAVE_STRATEGY="last_and_best"
  export KEEP_LAST_K="1"
  export SAVE_DIR="./checkpoints/final_confirm/${name}"
  export RESULTS_FILE="./results/final_confirm/${name}.json"

  bash slurm/run_one_experiment.sh | tee "./logs/${name}.log"
}

run_aro() {
  local name="$1"
  local ckpt_dir="$2"
  local ckpt
  ckpt="$(pick_best_ckpt "$ckpt_dir")"

  python eval_compositional.py \
    --checkpoint "${ckpt}" \
    --benchmark aro \
    --output "./results/final_confirm/${name}_aro.json" \
    | tee "./logs/${name}_aro.log"
}

run_winoground() {
  local name="$1"
  local ckpt_dir="$2"
  local ckpt
  ckpt="$(pick_best_ckpt "$ckpt_dir")"

  python eval_compositional.py \
    --checkpoint "${ckpt}" \
    --benchmark winoground \
    --hf_token "${HF_TOKEN}" \
    --output "./results/final_confirm/${name}_winoground.json" \
    | tee "./logs/${name}_winoground.log"
}

run_train confirm_baseline_varA_l0_s106_b128_e10 superclip_baseline 0.0
run_train confirm_reconA_l1_s106_b128_e10 superclip_recon 1.0

run_aro baseline_s106_l0_varA ./checkpoints/final_confirm/confirm_baseline_varA_l0_s106_b128_e10
run_aro reconA_s106_l1_varA ./checkpoints/final_confirm/confirm_reconA_l1_s106_b128_e10

run_winoground baseline_s106_l0_varA ./checkpoints/final_confirm/confirm_baseline_varA_l0_s106_b128_e10
run_winoground reconA_s106_l1_varA ./checkpoints/final_confirm/confirm_reconA_l1_s106_b128_e10

echo "=== confirm seed106 complete: $(date) ==="
