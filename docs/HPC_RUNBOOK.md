# HPC Runbook — SuperCLIP-Recon (Bocconi)

**Cluster:** Bocconi Jupiter I, `slogin.hpc.unibocconi.it`  
**Account / user:** `<USER_ID>`  
**Partition / QoS:** `stud` / `stud`  
**GPU request:** `gpu:4g.40gb:1`  
**Home quota:** 50 GB total  
**Repo root on cluster:** `/mnt/beegfsstudents/home/<USER_ID>/superclip-recon`

This document is the single source of truth for syncing, validating, and running SuperCLIP-Recon on Bocconi HPC.

It is also the source of truth for the project’s reproducibility rules:

- seeds must be forwarded into `train.py`
- matched comparisons should use deterministic caption selection
- the clean baseline is `lambda_recon=0` with reconstruction disabled
- Variant A is the official baseline anchor; Variant B is a phrase-reconstruction extension

---

## Core rules

1. Run `rsync` from your Mac, not from inside the HPC shell.
2. Always `cd` into the repo before running repo-relative commands.
3. Submit SLURM jobs with wrapped `sbatch` from the repo root.
4. Treat `--exclude=gnode04` as default for long jobs.
5. Do not start the next gate until the previous one has finished and been checked.
6. Monitor long jobs from `err/`, not `out/`.
7. Keep storage below ~40 GB used before long sweeps or ablations.
8. For matched baseline vs improvement comparisons, prefer `NUM_WORKERS=0 DETERMINISTIC=1 NO_AMP=1`.

---

## Reproducibility policy

### What is now logged per run

Every real run should record at least:

- `seed`
- `num_workers`
- `deterministic`
- `amp_enabled`
- `requested_train_mode`
- `train_mode`
- `variant`
- `effective_variant`
- `recon_enabled`
- module-wise parameter counts
- first-step gradient norms
- hostname
- git commit if available
- argv

If a result file does not contain these fields, treat it as lower-confidence evidence.

### Clean comparison rules

For the project report, use this logic:

- **Baseline anchor:** Variant A, `lambda_recon=0`
- **Improvement:** same setup plus reconstruction with nonzero `lambda_recon`
- **Variant B:** phrase-reconstruction extension, not the baseline reproduction

For the cleanest matched reruns:

- set `SEED=<n>`
- set `NUM_WORKERS=0`
- set `DETERMINISTIC=1`
- set `NO_AMP=1`

This is stricter than your normal throughput setting, but it is the right mode for reviewer-proof comparisons.

---

## Why the commands look the way they do

### Wrapped `sbatch`

Always submit jobs like:

```bash
sbatch --wrap='cd /mnt/beegfsstudents/home/<USER_ID>/superclip-recon && bash slurm/<script>.sh'