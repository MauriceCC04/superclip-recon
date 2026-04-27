#!/bin/bash
#SBATCH --job-name=winoground-bestpair
#SBATCH --account=3202029
#SBATCH --partition=stud
#SBATCH --qos=stud
#SBATCH --exclude=gnode04
#SBATCH --output=out/%x_%j.out
#SBATCH --error=err/%x_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=3202029@studbocconi.it
#SBATCH --time=03:00:00
#SBATCH --gres=gpu:4g.40gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

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

mkdir -p results/winoground logs out err

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

BASE_CKPT="$(pick_best_ckpt ./checkpoints/compositional_round2/compositional_baseline_varA_l0_s103_b128_e10)"
RECON_CKPT="$(pick_best_ckpt ./checkpoints/compositional_round2/compositional_reconA_l1_s103_b128_e10)"

echo "=== Winoground: baseline s103 ==="
python eval_compositional.py \
  --checkpoint "${BASE_CKPT}" \
  --benchmark winoground \
  --hf_token "${HF_TOKEN}" \
  --output ./results/winoground/baseline_s103_winoground.json \
  | tee ./logs/winoground_baseline_s103.log

echo "=== Winoground: reconA l1 s103 ==="
python eval_compositional.py \
  --checkpoint "${RECON_CKPT}" \
  --benchmark winoground \
  --hf_token "${HF_TOKEN}" \
  --output ./results/winoground/reconA_l1_s103_winoground.json \
  | tee ./logs/winoground_reconA_l1_s103.log

echo "=== Winoground pair complete: $(date) ==="
