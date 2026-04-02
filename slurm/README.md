# Recommended HPC workflow

Use these scripts in this order.

1. One-time setup

```bash
bash slurm/setup_env_ready.sh
```
Smoke test first
```bash
sbatch slurm/run_smoke.sh
```
Main experiments only
```bash
sbatch slurm/run_main_experiments.sh
```
Ablations separately
```bash
sbatch slurm/run_ablations.sh
```