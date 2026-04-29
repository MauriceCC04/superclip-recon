#!/bin/bash
#SBATCH --job-name=comp-s104
#SBATCH --account=3202029
#SBATCH --partition=stud
#SBATCH --qos=stud
#SBATCH --exclude=gnode04
#SBATCH --output=out/%x_%j.out
#SBATCH --error=err/%x_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=3202029@studbocconi.it
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:4g.40gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

set -euo pipefail

cd /mnt/beegfsstudents/home/3202029/superclip-recon
source slurm/common.sh
cd "$PROJECT_ROOT"

print_job_header
activate_env

mkdir -p results/final_confirm logs out err

if [ -z "${HF_TOKEN:-}" ]; then
  echo "ERROR: HF_TOKEN is not set"
  exit 1
fi

pick_best_ckpt() {
  local dir="$1"
  mapfile -t files < <(find "$dir" -maxdepth 1 -type f -name 'epoch_*.pt' | sort -V)
  if [ "${#files[@]}" -lt 2 ]; then
    echo "ERROR: expected at least epoch_9 and epoch_10 in $dir" >&2
    return 1
  fi
  echo "${files[${#files[@]}-2]}"
}

run_aro() {
  local name="$1"
  local dir="$2"
  local ckpt
  ckpt="$(pick_best_ckpt "$dir")"

  python eval_compositional.py \
    --checkpoint "${ckpt}" \
    --benchmark aro \
    --output "./results/final_confirm/${name}_aro.json" \
    | tee "./logs/${name}_aro.log"
}

run_wino() {
  local name="$1"
  local dir="$2"
  local ckpt
  ckpt="$(pick_best_ckpt "$dir")"

  python eval_compositional.py \
    --checkpoint "${ckpt}" \
    --benchmark winoground \
    --hf_token "${HF_TOKEN}" \
    --output "./results/final_confirm/${name}_winoground.json" \
    | tee "./logs/${name}_winoground.log"
}

run_aro  baseline_s104_l0_varA ./checkpoints/final_confirm/confirm_baseline_varA_l0_s104_b128_e10
run_aro  reconA_s104_l1_varA   ./checkpoints/final_confirm/confirm_reconA_l1_s104_b128_e10
run_wino baseline_s104_l0_varA ./checkpoints/final_confirm/confirm_baseline_varA_l0_s104_b128_e10
run_wino reconA_s104_l1_varA   ./checkpoints/final_confirm/confirm_reconA_l1_s104_b128_e10

echo "=== comp-s104 complete: $(date) ==="
