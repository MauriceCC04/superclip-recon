````md
# HPC Runbook — SuperCLIP-Recon

**Target cluster**: partition `stud`, account `3202029`, QoS `stud`, one MIG-sliced A100 40GB (`gpu:4g.40gb:1`), 100 GB home quota, no scratch.

This runbook defines a **gated** execution order. Each gate emits a structured JSON artefact and a PASS / WARN / FAIL verdict. **Do not proceed to the next gate until the current gate is PASS or a justified WARN.**

**Bocconi-specific note**: on this cluster, the safest submission pattern is to use `sbatch --wrap='cd /mnt/beegfsstudents/home/3202029/superclip-recon && bash ...'`. Direct `sbatch slurm/<script>.sh` can fail with a `/var/spool/.../common.sh: No such file or directory` error if the job starts from SLURM’s spool directory instead of the repo root.

---

## Gate 0 — Local synthetic tests (no COCO, no GPU)

**Purpose**: catch shape/logic/pipeline bugs before touching the cluster.

```bash
python tests/run_tests.py
````

| Verdict  | Criterion                                                                                                                                              | Action                                                                                                              |
| -------- | ------------------------------------------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------- |
| **PASS** | All tests pass.                                                                                                                                        | Proceed to Gate 1.                                                                                                  |
| **WARN** | Only environment-dependent tests fail (no internet for CLIP weights, no `datasets` package). Core logic, masking, retention, schema, slurm tests pass. | Proceed to Gate 1, but note that model-building tests were not exercised locally; they will be exercised in Gate 1. |
| **FAIL** | Any test in the fixes suite fails (numbers 06b, 06c, 12, 13, 14, 15, 16, 17).                                                                          | Stop. Fix the underlying code before queueing anything.                                                             |

---

## Gate 1 — Static + runtime preflight (cluster)

**Purpose**: verify the env is set up, data is present, imports work, and one tiny forward+backward step runs under real cluster conditions.

```bash
cd /mnt/beegfsstudents/home/3202029/superclip-recon

# one-time on Bocconi: accept conda ToS
module load miniconda3
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

# one-time setup
bash slurm/setup_env_ready.sh

# Bocconi-safe submission: wrap the job so it starts from the repo root
sbatch \
  --job-name=superclip-preflight \
  --account=3202029 \
  --partition=stud \
  --qos=stud \
  --output=out/%x_%j.out \
  --error=err/%x_%j.err \
  --mail-type=END,FAIL \
  --mail-user=3202029@studbocconi.it \
  --time=00:30:00 \
  --gres=gpu:4g.40gb:1 \
  --cpus-per-task=4 \
  --mem=16G \
  --wrap='cd /mnt/beegfsstudents/home/3202029/superclip-recon && bash slurm/run_preflight.sh'
```

Monitor with:

```bash
squeue -u 3202029
tail -f out/superclip-preflight_<JOBID>.out
```

**Inspect**: `results/preflight/preflight_report.json`

```bash
cat results/preflight/preflight_report.json | python -m json.tool | head -80
```

```jsonc
{
  "overall_status": "PASS|WARN|FAIL",
  "checks":   { "repo", "imports", "gpu", "data", "cache", "storage", "runtime" },
  "metrics":  { "gpu_peak_mem_gb", "checkpoint_size_mb",
                "one_step_seconds", "tiny_eval_seconds" },
  "recommendations": [ ... ]
}
```

| Verdict  | Criterion                                                                                                                                                                                | Action                                                                                               |
| -------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------- |
| **PASS** | `overall_status == "PASS"`. GPU detected, data present, one_step_seconds < 5 s.                                                                                                          | Proceed to Gate 2.                                                                                   |
| **WARN** | `overall_status == "WARN"` and the warnings are about **optional** pieces (`phrases.json` missing → only affects Variant B; cache vars need tuning).                                     | Fix the warnings or accept and proceed. Do **not** proceed if the warning is about home-quota usage. |
| **FAIL** | Any of: `repo.status=="FAIL"` (missing code files), `imports.status=="FAIL"` (broken env), `gpu.status=="FAIL"`, `runtime.status=="FAIL"`, or `storage.status=="FAIL"` (home >85% full). | Stop. Follow the `recommendations` list in the report.                                               |

---

## Gate 2 — Cluster GPU smoke

**Purpose**: run a few real training steps + a small retrieval eval on the real GPU, measure peak memory and checkpoint size.

```bash
cd /mnt/beegfsstudents/home/3202029/superclip-recon

sbatch \
  --job-name=superclip-gpusmoke \
  --account=3202029 \
  --partition=stud \
  --qos=stud \
  --output=out/%x_%j.out \
  --error=err/%x_%j.err \
  --mail-type=END,FAIL \
  --mail-user=3202029@studbocconi.it \
  --time=00:30:00 \
  --gres=gpu:4g.40gb:1 \
  --cpus-per-task=4 \
  --mem=16G \
  --wrap='cd /mnt/beegfsstudents/home/3202029/superclip-recon && bash slurm/run_gpu_smoke.sh'
```

Monitor with:

```bash
squeue -u 3202029
tail -f out/superclip-gpusmoke_<JOBID>.out
```

**Inspect**: `results/smoke/gpu_smoke_results.json`

```bash
cat results/smoke/gpu_smoke_results.json | python -m json.tool | head -80
```

Look for:

* `gpu_peak_mem_gb` — should be well under the 40 GB slice limit
* `checkpoint_size_mb` — checkpoint footprint
* `retrieval.i2t_r1` and `retrieval.t2i_r1` — only need to be > 0 for the smoke run

| Verdict  | Criterion                                                            | Action                                                                                                  |
| -------- | -------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------- |
| **PASS** | JSON produced, `gpu_peak_mem_gb < 35`, no NaN in losses, ckpt saved. | Proceed to Gate 3.                                                                                      |
| **WARN** | `gpu_peak_mem_gb` between 32 and 38 GB, i.e. headroom is tight.      | Reduce batch size for Gate 3 and main runs, or disable `save_optimizer_state` (already off by default). |
| **FAIL** | OOM, NaN, import error on compute node, or job killed.               | Stop. Read the SLURM error log in `err/`; likely need to fix env or reduce batch size.                  |

---

## Gate 3 — Pilot baseline

**Purpose**: run **one** real training epoch of the baseline, measure wall time, and project the 10-epoch cost before committing to the main runs.

```bash
cd /mnt/beegfsstudents/home/3202029/superclip-recon

sbatch \
  --job-name=superclip-pilot \
  --account=3202029 \
  --partition=stud \
  --qos=stud \
  --output=out/%x_%j.out \
  --error=err/%x_%j.err \
  --mail-type=END,FAIL \
  --mail-user=3202029@studbocconi.it \
  --time=04:00:00 \
  --gres=gpu:4g.40gb:1 \
  --cpus-per-task=8 \
  --mem=32G \
  --wrap='cd /mnt/beegfsstudents/home/3202029/superclip-recon && bash slurm/run_pilot_baseline.sh'
```

Monitor with:

```bash
squeue -u 3202029
# On Bocconi, training progress is usually visible in stderr because tqdm writes there
tail -f err/superclip-pilot_<JOBID>.err
```

**Inspect**: `results/pilot/pilot_baseline.json`

```bash
cat results/pilot/pilot_baseline.json | python -m json.tool | head -80
```

Look for:

* `epoch_seconds` — one epoch wall time
* `projected_10_epoch_seconds` — epoch × 10
* `projected_checkpoint_storage_mb` — checkpoint size × retention
* `readiness`: `PASS|WARN|FAIL` with reasons

| Verdict  | Criterion                                                                                       | Action                                                                                     |
| -------- | ----------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------ |
| **PASS** | `projected_10_epoch_seconds < 20 * 3600` (20 h), `projected_checkpoint_storage_mb < 10 * 1024`. | Proceed to Gate 4.                                                                         |
| **WARN** | Projected run > 20 h but < 24 h.                                                                | Reduce epochs to 7–8, or raise batch size if Gate 2 peak memory allows.                    |
| **FAIL** | Projected run > 24 h, or projected storage would break the home quota.                          | Stop. Shrink: smaller batch, fewer epochs, smaller eval subset, stronger retention policy. |

---

## Gate 4 — Main experiments

**Purpose**: run baseline, Variant A, Variant B as separate SLURM jobs.

**Important on Bocconi**:

* Prefer **one job at a time**, or at most two total, to avoid `QOSMaxSubmitJobPerUserLimit` / `QOSMaxJobsPerUserLimit`.
* Monitor **stderr** for progress.
* If a run fails, fix and resubmit **only that one**.

### 4A — Optional phrase extraction for Variant B

```bash
cd /mnt/beegfsstudents/home/3202029/superclip-recon

module load miniconda3
eval "$(conda shell.bash hook)"
conda activate superclip

python extract_phrases.py --use_regex --coco_root ./data/coco --output phrases.json
```

### 4B — Baseline

```bash
cd /mnt/beegfsstudents/home/3202029/superclip-recon

sbatch \
  --job-name=superclip-baseline \
  --account=3202029 \
  --partition=stud \
  --qos=stud \
  --output=out/%x_%j.out \
  --error=err/%x_%j.err \
  --mail-type=END,FAIL \
  --mail-user=3202029@studbocconi.it \
  --time=12:00:00 \
  --gres=gpu:4g.40gb:1 \
  --cpus-per-task=8 \
  --mem=32G \
  --wrap='cd /mnt/beegfsstudents/home/3202029/superclip-recon && export RUN_NAME=baseline VARIANT=A LAMBDA_RECON=0.0 MASK_RATIO=0.15 SAVE_DIR=./checkpoints/baseline RESULTS_FILE=./results/baseline.json SAVE_STRATEGY=last_and_best KEEP_LAST_K=1 BATCH_SIZE=64 EVAL_MAX_IMAGES=5000 && bash slurm/run_one_experiment.sh'
```

Monitor:

```bash
squeue -u 3202029
tail -f err/superclip-baseline_<JOBID>.err
```

Inspect:

```bash
cat results/baseline.json | python -m json.tool | head -40
```

### 4C — Variant A

```bash
cd /mnt/beegfsstudents/home/3202029/superclip-recon

sbatch \
  --job-name=superclip-variant_a \
  --account=3202029 \
  --partition=stud \
  --qos=stud \
  --output=out/%x_%j.out \
  --error=err/%x_%j.err \
  --mail-type=END,FAIL \
  --mail-user=3202029@studbocconi.it \
  --time=12:00:00 \
  --gres=gpu:4g.40gb:1 \
  --cpus-per-task=8 \
  --mem=32G \
  --wrap='cd /mnt/beegfsstudents/home/3202029/superclip-recon && export RUN_NAME=variant_a VARIANT=A LAMBDA_RECON=0.5 MASK_RATIO=0.15 SAVE_DIR=./checkpoints/variant_a RESULTS_FILE=./results/variant_a.json SAVE_STRATEGY=last_and_best KEEP_LAST_K=1 BATCH_SIZE=64 EVAL_MAX_IMAGES=5000 && bash slurm/run_one_experiment.sh'
```

Monitor:

```bash
squeue -u 3202029
tail -f err/superclip-variant_a_<JOBID>.err
```

Inspect:

```bash
cat results/variant_a.json | python -m json.tool | head -40
```

### 4D — Variant B

```bash
cd /mnt/beegfsstudents/home/3202029/superclip-recon

sbatch \
  --job-name=superclip-variant_b \
  --account=3202029 \
  --partition=stud \
  --qos=stud \
  --output=out/%x_%j.out \
  --error=err/%x_%j.err \
  --mail-type=END,FAIL \
  --mail-user=3202029@studbocconi.it \
  --time=12:00:00 \
  --gres=gpu:4g.40gb:1 \
  --cpus-per-task=8 \
  --mem=32G \
  --wrap='cd /mnt/beegfsstudents/home/3202029/superclip-recon && export RUN_NAME=variant_b VARIANT=B LAMBDA_RECON=0.5 MASK_RATIO=0.15 PHRASE_PATH=./phrases.json SAVE_DIR=./checkpoints/variant_b RESULTS_FILE=./results/variant_b.json SAVE_STRATEGY=last_and_best KEEP_LAST_K=1 BATCH_SIZE=64 EVAL_MAX_IMAGES=5000 && bash slurm/run_one_experiment.sh'
```

Monitor:

```bash
squeue -u 3202029
tail -f err/superclip-variant_b_<JOBID>.err
```

Inspect:

```bash
cat results/variant_b.json | python -m json.tool | head -40
```

| Verdict  | Criterion                                                                                | Action                                                                                                      |
| -------- | ---------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------- |
| **PASS** | All three JSON files present, all have `final_retrieval.i2t_r1 > 0`, all losses finite.  | Proceed to Gate 5.                                                                                          |
| **WARN** | One run finished; others still queued or still running.                                  | Wait. On Bocconi, this is normal if you submit one at a time or the second is pending on the QoS job limit. |
| **FAIL** | Any run produced NaN, OOM, a SLURM submission failure, or did not produce a JSON at all. | Inspect `err/superclip-<run>_<jobid>.err`, fix, resubmit only that one.                                     |

---

## Gate 5 — Ablations

**Purpose**: compact grid (lambda sweep, masking-rate sweep, Variant A vs B). Do **not** start until Gate 4 is green and storage headroom is verified.

```bash
cd /mnt/beegfsstudents/home/3202029/superclip-recon

# verify storage first
du -sh checkpoints results
df -h $HOME
```

If your `slurm/submit_ablations.sh` helper has already been patched for Bocconi, run:

```bash
bash slurm/submit_ablations.sh
```

If it still fails with the SLURM spool-path `common.sh` error, patch it first or submit ablations with the same wrapped-`sbatch` pattern used in Gate 4.

After submission:

```bash
squeue -u 3202029
```

When complete:

```bash
python analyze_results.py --results_dir ./results --output_dir ./results/figures
```

| Verdict  | Criterion                                                                                                           | Action                                                                                                                   |
| -------- | ------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------ |
| **PASS** | All unique ablation JSONs present under `results/ablations/`, and `analyze_results.py` completes.                   | Done — write up.                                                                                                         |
| **WARN** | Some ablation runs failed but the *informative* points (at least two lambda values and two mask rates) are present. | Accept; document which runs failed and why.                                                                              |
| **FAIL** | Home quota exceeded mid-grid, helper script fails, or `analyze_results.py` complains of missing files.              | Stop the remaining jobs (`scancel`), clean up, revisit retention policy, then rerun only the missing informative points. |

---

## Optional — compositional evaluation

Only run this after Gate 4.

```bash
cd /mnt/beegfsstudents/home/3202029/superclip-recon

sbatch \
  --job-name=superclip-compositional \
  --account=3202029 \
  --partition=stud \
  --qos=stud \
  --output=out/%x_%j.out \
  --error=err/%x_%j.err \
  --mail-type=END,FAIL \
  --mail-user=3202029@studbocconi.it \
  --time=02:00:00 \
  --gres=gpu:4g.40gb:1 \
  --cpus-per-task=8 \
  --mem=32G \
  --wrap='cd /mnt/beegfsstudents/home/3202029/superclip-recon && bash slurm/run_compositional_eval.sh'
```

---

## Quick reference — commands

```bash
# Local (Gate 0)
python tests/run_tests.py

# One-time setup on Bocconi
module load miniconda3
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
bash slurm/setup_env_ready.sh

# Modular setup steps
bash slurm/setup_env_ready.sh --step vocab
bash slurm/setup_env_ready.sh --step data
bash slurm/setup_env_ready.sh --step cache
bash slurm/setup_env_ready.sh --step phrases --spacy   # optional spaCy
```

```bash
# Activate env manually on the login node when needed
module load miniconda3
eval "$(conda shell.bash hook)"
conda activate superclip
```

```bash
# Gate 1
sbatch \
  --job-name=superclip-preflight \
  --account=3202029 \
  --partition=stud \
  --qos=stud \
  --output=out/%x_%j.out \
  --error=err/%x_%j.err \
  --time=00:30:00 \
  --gres=gpu:4g.40gb:1 \
  --cpus-per-task=4 \
  --mem=16G \
  --wrap='cd /mnt/beegfsstudents/home/3202029/superclip-recon && bash slurm/run_preflight.sh'

# Gate 2
sbatch \
  --job-name=superclip-gpusmoke \
  --account=3202029 \
  --partition=stud \
  --qos=stud \
  --output=out/%x_%j.out \
  --error=err/%x_%j.err \
  --time=00:30:00 \
  --gres=gpu:4g.40gb:1 \
  --cpus-per-task=4 \
  --mem=16G \
  --wrap='cd /mnt/beegfsstudents/home/3202029/superclip-recon && bash slurm/run_gpu_smoke.sh'

# Gate 3
sbatch \
  --job-name=superclip-pilot \
  --account=3202029 \
  --partition=stud \
  --qos=stud \
  --output=out/%x_%j.out \
  --error=err/%x_%j.err \
  --time=04:00:00 \
  --gres=gpu:4g.40gb:1 \
  --cpus-per-task=8 \
  --mem=32G \
  --wrap='cd /mnt/beegfsstudents/home/3202029/superclip-recon && bash slurm/run_pilot_baseline.sh'
```

```bash
# Monitoring helpers
squeue -u 3202029
sacct -j <JOBID> --format=JobID,State,Elapsed,ExitCode
scontrol show job <JOBID>
tail -f out/<JOBNAME>_<JOBID>.out
tail -f err/<JOBNAME>_<JOBID>.err
```

```bash
# Analysis
python analyze_results.py --results_dir ./results --output_dir ./results/figures
```

---

## Safety defaults (current recommended values)

| Setting                                         | Default                                            | Why                                                           |
| ----------------------------------------------- | -------------------------------------------------- | ------------------------------------------------------------- |
| `--save_strategy`                               | `last_and_best`                                    | Keep only the most useful checkpoints.                        |
| `--keep_last_k`                                 | `1`                                                | Minimize checkpoint growth under home-quota limits.           |
| `--save_optimizer_state`                        | `False`                                            | Reduces checkpoint size substantially.                        |
| `--eval_max_images`                             | `5000` for main; `1000` for pilot; `128` for smoke | Keep eval cheap where it is only a sanity signal.             |
| `BATCH_SIZE`                                    | `64` on Bocconi for pilot/main                     | This was the safe working batch size in the successful pilot. |
| `PIP_NO_CACHE_DIR`                              | `1` (via `common.sh`)                              | Prevent pip cache from growing on home.                       |
| `XDG_CACHE_HOME`, `HF_HOME`, `TORCH_HOME`, etc. | Under `$HOME/.cache` (via `common.sh`)             | Centralize cache growth under one visible path.               |
| Submission style                                | `sbatch --wrap='cd ... && bash ...'`               | Avoid SLURM spool-path failures on Bocconi.                   |
| Training monitoring                             | Check `err/` first                                 | tqdm and many warnings go to stderr.                          |

---

## When something goes wrong

**`CondaToSNonInteractiveError` during setup**
Accept the two Anaconda channels once on the login node:

```bash
module load miniconda3
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
```

**`/var/spool/.../common.sh: No such file or directory`**
The job started from SLURM’s spool directory instead of the repo root. Submit with:

```bash
sbatch --wrap='cd /mnt/beegfsstudents/home/3202029/superclip-recon && bash slurm/<script>.sh'
```

or use the full wrapped command blocks from this runbook.

**`QOSMaxSubmitJobPerUserLimit` or `QOSMaxJobsPerUserLimit`**
You have hit the active-job limit. Wait for the running job to finish, then resubmit the next one.

**No visible progress in `.out` for a training run**
Check `.err` instead:

```bash
tail -f err/<JOBNAME>_<JOBID>.err
```

**Job killed at OOM**
Lower `BATCH_SIZE`, keep `save_optimizer_state=False`, and reduce `EVAL_MAX_IMAGES` if needed.

**Job killed at time limit**
Check Gate 3 projections. Reduce `EPOCHS`, reduce eval size, or split the plan into smaller runs.

**`COCO data not found`**
Run:

```bash
bash slurm/setup_env_ready.sh --step data
```

**`vocab.json not found`**
Run:

```bash
bash slurm/setup_env_ready.sh --step vocab
```

**`phrases.json` missing for Variant B**
Generate it once:

```bash
module load miniconda3
eval "$(conda shell.bash hook)"
conda activate superclip
python extract_phrases.py --use_regex --coco_root ./data/coco --output phrases.json
```

**CLIP weights will not download on compute node**
Compute nodes may not have internet. Pre-cache them on the login node:

```bash
bash slurm/setup_env_ready.sh --step cache
```

**Home quota alarm**
Check growth here first:

```bash
du -sh $HOME/.cache/*
du -sh checkpoints results out err
df -h $HOME
```

Then delete stale logs/checkpoints or reduce retention.

**Ablation grid taking too long**
Run a shorter grid first, then expand only if needed. For a course project, a smaller but interpretable ablation grid is better than an incomplete large one.
