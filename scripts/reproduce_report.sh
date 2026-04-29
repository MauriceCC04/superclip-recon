#!/bin/bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

usage() {
  cat <<'EOF'
Usage:
  bash scripts/reproduce_report.sh plan
  bash scripts/reproduce_report.sh main
  bash scripts/reproduce_report.sh confirm6
  bash scripts/reproduce_report.sh confirm_more
  bash scripts/reproduce_report.sh ablations
  bash scripts/reproduce_report.sh compositional_core
  bash scripts/reproduce_report.sh compositional_plus
  bash scripts/reproduce_report.sh final_confirm
  bash scripts/reproduce_report.sh analyze

Notes:
- Run one stage at a time on Bocconi HPC.
- This avoids QoS / queue issues and matches docs/HPC_RUNBOOK.md.
- Winoground-related stages require HF_TOKEN.
EOF
}

plan() {
  cat <<'EOF'
Recommended stage order:

  1. main
     bash scripts/reproduce_report.sh main

  2. confirm6
     bash scripts/reproduce_report.sh confirm6

  3. confirm_more
     bash scripts/reproduce_report.sh confirm_more

  4. ablations
     bash scripts/reproduce_report.sh ablations

  5. compositional_core
     bash scripts/reproduce_report.sh compositional_core

  6. compositional_plus    (optional; requires HF_TOKEN)
     bash scripts/reproduce_report.sh compositional_plus

  7. final_confirm         (optional extra confirmation family)
     bash scripts/reproduce_report.sh final_confirm

  8. analyze
     bash scripts/reproduce_report.sh analyze
EOF
}

require_hf_token() {
  if [ -z "${HF_TOKEN:-}" ]; then
    echo "ERROR: HF_TOKEN is required for this stage." >&2
    exit 1
  fi
}

stage="${1:-plan}"

case "$stage" in
  plan)
    plan
    ;;

  main)
    bash slurm/submit_main_experiments.sh baseline variant_a variant_b
    ;;

  confirm6)
    sbatch slurm/run_confirm6_sequential.sh
    ;;

  confirm_more)
    sbatch slurm/run_confirm_more_sequential.sh
    ;;

  ablations)
    sbatch slurm/run_maskrate_l1_seed102.sh
    sbatch slurm/run_variantB_seed104_check.sh
    ;;

  compositional_core)
    sbatch slurm/run_compositional_confirm_pair.sh
    sbatch slurm/run_compositional_round2.sh
    ;;

  compositional_plus)
    require_hf_token
    sbatch slurm/run_compositional_seed104.sh
    sbatch slurm/run_winoground_bestpair.sh
    sbatch slurm/run_compositional_more.sh
    ;;

  final_confirm)
    require_hf_token
    sbatch slurm/run_confirm_seed104_varA.sh
    sbatch slurm/run_confirm_seed105.sh
    sbatch slurm/run_confirm_seed106.sh
    ;;

  analyze)
    python scripts/build_results_bundle.py --results_dir ./results
    ;;

  -h|--help|help)
    usage
    ;;

  *)
    echo "Unknown stage: $stage" >&2
    usage
    exit 1
    ;;
esac
