#!/bin/bash
#SBATCH --job-name=compositional-more
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
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

set -euo pipefail

cd /mnt/beegfsstudents/home/3202029/superclip-recon
source slurm/common.sh
cd "$PROJECT_ROOT"

print_job_header
activate_env

mkdir -p results/compositional_round3 results/winoground logs out err

if [ -z "${HF_TOKEN:-}" ]; then
  echo "ERROR: HF_TOKEN is not set"
  exit 1
fi

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
  echo "Using checkpoint: ${ckpt}"
  echo "=================================================="

  python eval_compositional.py \
    --checkpoint "${ckpt}" \
    --benchmark aro \
    --output "./results/compositional_round3/${name}_aro.json" \
    | tee "./logs/${name}_aro.log"
}

run_wino() {
  local name="$1"
  local ckpt_dir="$2"
  local ckpt
  ckpt="$(pick_best_ckpt "$ckpt_dir")"

  echo "=================================================="
  echo "Winoground eval: ${name}"
  echo "Using checkpoint: ${ckpt}"
  echo "=================================================="

  python eval_compositional.py \
    --checkpoint "${ckpt}" \
    --benchmark winoground \
    --hf_token "${HF_TOKEN}" \
    --output "./results/winoground/${name}_winoground.json" \
    | tee "./logs/${name}_winoground.log"
}

# Seed 104 pair
run_aro  baseline_confirm_s104_l0_varA ./checkpoints/confirm8/confirm_baseline_varA_l0_s104_b128_e10
run_aro  reconA_confirm_s104_l1_varA   ./checkpoints/confirm8/confirm_reconA_l1_s104_b128_e10
run_wino baseline_s104                 ./checkpoints/confirm8/confirm_baseline_varA_l0_s104_b128_e10
run_wino reconA_l1_s104                ./checkpoints/confirm8/confirm_reconA_l1_s104_b128_e10

# Seed 105 pair
run_aro  baseline_confirm_s105_l0_varA ./checkpoints/confirm8/confirm_baseline_varA_l0_s105_b128_e10
run_aro  reconA_confirm_s105_l1_varA   ./checkpoints/confirm8/confirm_reconA_l1_s105_b128_e10
run_wino baseline_s105                 ./checkpoints/confirm8/confirm_baseline_varA_l0_s105_b128_e10
run_wino reconA_l1_s105                ./checkpoints/confirm8/confirm_reconA_l1_s105_b128_e10

echo "=== compositional-more complete: $(date) ==="
