#!/bin/bash
#SBATCH --job-name=compositional-pair
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

mkdir -p results/compositional logs out err

echo "=== ARO probe: baseline s101 ==="
python eval_compositional.py \
  --checkpoint ./checkpoints/confirm6/confirm_baseline_varA_l0_s101_b128_e10/epoch_10.pt \
  --benchmark aro \
  --output ./results/compositional/confirm_baseline_varA_l0_s101_b128_e10_aro.json \
  | tee ./logs/compositional_confirm_baseline_varA_l0_s101_b128_e10_aro.log

echo "=== ARO probe: reconA l1 s101 ==="
python eval_compositional.py \
  --checkpoint ./checkpoints/confirm6/confirm_reconA_l1_s101_b128_e10/epoch_10.pt \
  --benchmark aro \
  --output ./results/compositional/confirm_reconA_l1_s101_b128_e10_aro.json \
  | tee ./logs/compositional_confirm_reconA_l1_s101_b128_e10_aro.log

echo "=== Compositional pair eval complete: $(date) ==="
