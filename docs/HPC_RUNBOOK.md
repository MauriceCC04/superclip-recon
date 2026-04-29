# HPC Runbook — SuperCLIP-Recon (Bocconi)

**Cluster:** Bocconi Jupiter I, `slogin.hpc.unibocconi.it`  
**Account / user:** `<USER_ID>`  
**Partition / QoS:** `stud` / `stud`  
**GPU request:** `gpu:4g.40gb:1`  
**Home quota:** 50 GB total  
**Repo root on cluster:** `/mnt/beegfsstudents/home/<USER_ID>/superclip-recon`

This document is the single source of truth for syncing, validating, running, monitoring, and recovering SuperCLIP-Recon on Bocconi HPC.

It is also the source of truth for the project’s reproducibility rules:

- seeds must be forwarded into `train.py`
- matched comparisons should use the same seed within each baseline vs improvement pair
- the clean baseline is Variant A with `lambda_recon=0`
- Variant A is the official baseline anchor; Variant B is a phrase-reconstruction extension
- compositional probing is supporting evidence, not the main finish line

---

## Core rules

1. Run `rsync` from your Mac, not from inside the HPC shell.
2. Always `cd` into the repo before running repo-relative commands.
3. Prefer **one sequential SLURM job** over mass-submitting many jobs.
4. Treat `--exclude=gnode04` as default for long jobs.
5. Do not start the next gate until the previous one has finished and been checked.
6. Monitor long training runs from `err/`, but check `out/` for sweep-level or sequential-batch summaries.
7. Keep storage below about **40 GB used** before long sweeps or multi-run batches.
8. For matched baseline vs improvement comparisons, prefer `NUM_WORKERS=0 DETERMINISTIC=1 NO_AMP=1` if you want the cleanest reproducibility.
9. When using `slurm/run_one_experiment.sh`, always set `RUN_NAME`, `VARIANT`, `LAMBDA_RECON`, and `MASK_RATIO`.
10. If you rerun a single configuration outside a sweep, explicitly set `SAVE_DIR` and `RESULTS_FILE` so outputs land in the intended folder.
11. The first ARO or Winoground run on a fresh cache may spend noticeable time downloading datasets or generating splits before real evaluation progress appears.

---

## Reproducibility policy

### What is logged per real run

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

### Git commit provenance on the HPC copy

`train.py` records `git_commit` on a best-effort basis. This is useful, but the normal HPC sync intentionally excludes `.git` to keep transfers light.

That means the HPC copy may legitimately produce:

- `git_commit: null` in a result JSON
- a terminal log line like `fatal: not a git repository ...`

Treat that as missing provenance metadata, not as a model failure, if the run otherwise completed and wrote its outputs.

If exact revision tracking matters for a reporting run, record `git rev-parse HEAD` on the Mac before syncing, or store the SHA explicitly in a lightweight tracked note.

### Clean comparison rules

For the report, use this logic:

- **Baseline anchor:** Variant A, `lambda_recon=0`
- **Improvement:** same setup plus reconstruction with nonzero `lambda_recon`
- **Variant B:** phrase-reconstruction extension, not the baseline reproduction

For the cleanest matched reruns:

- set `SEED=<n>`
- set `NUM_WORKERS=0`
- set `DETERMINISTIC=1`
- set `NO_AMP=1`

This is slower than your normal throughput setting, but it is the right mode for reviewer-proof paired comparisons.

---

## Why the commands look the way they do

### Wrapped `sbatch`

Always submit jobs like:

```bash
sbatch --wrap='cd /mnt/beegfsstudents/home/<USER_ID>/superclip-recon && bash slurm/<script>.sh'
```

or submit a script that already `cd`s into the repo and sources `slurm/common.sh` from the real repo path.

Reason: on Bocconi, directly submitting `sbatch slurm/<script>.sh` can cause path resolution to happen under `/var/spool/slurmd/jobNNNN/`, which breaks relative `common.sh` sourcing.

### Sequential jobs instead of many queued jobs

The `stud` QoS punishes mass submission. If you need 4 to 8 runs, prefer one sequential job that calls `slurm/run_one_experiment.sh` multiple times in a row.

This avoids:

- `QOSMaxSubmitJobPerUserLimit`
- queue clutter
- partial submission success where only the first one or two jobs enter the queue

### Repo-relative paths

Nearly every useful directory in this project is repo-relative:

- `results/`
- `checkpoints/`
- `logs/`
- `out/`
- `err/`

So if a command says a file does not exist, first ask whether you are in `~` instead of the repo.

---

## 0. First-time setup

### 0.1 Log in

```bash
ssh <USER_ID>@slogin.hpc.unibocconi.it
cd /mnt/beegfsstudents/home/<USER_ID>/superclip-recon
```

### 0.2 Accept conda terms once

```bash
module load miniconda3
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
```

### 0.3 Activate the env

```bash
module load miniconda3
eval "$(conda shell.bash hook)"
conda activate superclip
```

### 0.4 Verify the repo is runnable

```bash
cd /mnt/beegfsstudents/home/<USER_ID>/superclip-recon

pwd
ls slurm | sort
ls -l slurm/common.sh slurm/run_preflight.sh slurm/run_gpu_smoke.sh slurm/run_one_experiment.sh slurm/run_lambda_sweep.sh
ls -lh vocab.json
test -d data/coco/train2017 && echo "train2017 OK"
test -d data/coco/val2017 && echo "val2017 OK"
test -f data/coco/annotations/captions_train2017.json && echo "captions_train2017 OK"
test -f data/coco/annotations/captions_val2017.json && echo "captions_val2017 OK"
```

---

## 1. Syncing between Mac and HPC

### 1.1 Sync code from Mac to HPC

Run this on your Mac, not on the cluster:

```bash
rsync -av --delete \
  --exclude '.venv' \
  --exclude 'venv' \
  --exclude '__pycache__' \
  --exclude '.pytest_cache' \
  --exclude '.mypy_cache' \
  --exclude '.DS_Store' \
  --exclude '.git' \
  --exclude 'data/coco' \
  --exclude 'checkpoints' \
  --exclude 'results' \
  --exclude 'out' \
  --exclude 'err' \
  --exclude 'logs' \
  --exclude 'vocab.json' \
  ~/PycharmProjects/superclip-recon/ \
  <USER_ID>@slogin.hpc.unibocconi.it:/mnt/beegfsstudents/home/<USER_ID>/superclip-recon/
```

This preserves the large data and vocab already stored on HPC.

It also means the HPC copy is usually **not** a full Git checkout. So a successful run may still end with a best-effort metadata message like `fatal: not a git repository ...` when `train.py` tries to record the commit SHA. That is usually benign.

### 1.2 Sync results from HPC to Mac

For a folder-style sync:

```bash
rsync -av \
  <USER_ID>@slogin.hpc.unibocconi.it:/mnt/beegfsstudents/home/<USER_ID>/superclip-recon/results/<SUBDIR>/ \
  ~/superclip_hpc_backup/results/<SUBDIR>/
```

Examples used during the project:

- `results/ablations/`
- `results/core8/`
- `results/confirm6/`
- `results/compositional/`
- `results/compositional_round2/`

### 1.3 Sync checkpoints only when you need them

Result JSONs are the important artifacts for reporting. Checkpoints are large. Sync them only if you need:

- a backup of trained weights
- later compositional evaluation
- later local inspection of best/last checkpoints

---

## 2. Submission strategy

### 2.1 Gate order

Always use this order when the code changed recently:

1. preflight
2. GPU smoke
3. short pilot
4. main sweep or main batch
5. compositional eval

Do not skip earlier gates unless you already know the repo is stable and unchanged.

### 2.2 Preflight

```bash
sbatch \
  --job-name=superclip-preflight \
  --account=<USER_ID> \
  --partition=stud \
  --qos=stud \
  --exclude=gnode04 \
  --output=out/%x_%j.out \
  --error=err/%x_%j.err \
  --mail-type=END,FAIL \
  --mail-user=<USER_ID>@studbocconi.it \
  --time=00:20:00 \
  --cpus-per-task=4 \
  --mem=8G \
  --wrap='cd /mnt/beegfsstudents/home/<USER_ID>/superclip-recon && bash slurm/run_preflight.sh'
```

### 2.3 GPU smoke

```bash
sbatch \
  --job-name=superclip-gpu-smoke \
  --account=<USER_ID> \
  --partition=stud \
  --qos=stud \
  --exclude=gnode04 \
  --output=out/%x_%j.out \
  --error=err/%x_%j.err \
  --mail-type=END,FAIL \
  --mail-user=<USER_ID>@studbocconi.it \
  --time=00:30:00 \
  --gres=gpu:4g.40gb:1 \
  --cpus-per-task=4 \
  --mem=16G \
  --wrap='cd /mnt/beegfsstudents/home/<USER_ID>/superclip-recon && bash slurm/run_gpu_smoke.sh'
```

### 2.4 Short pilot via `run_one_experiment.sh`

Minimum required env vars:

```bash
RUN_NAME=pilot_varA \
VARIANT=A \
LAMBDA_RECON=0 \
MASK_RATIO=0.15 \
...
```

A missing `MASK_RATIO` causes the launcher to stop immediately.

Example:

```bash
sbatch \
  --job-name=superclip-pilot-a \
  --account=<USER_ID> \
  --partition=stud \
  --qos=stud \
  --exclude=gnode04 \
  --output=out/%x_%j.out \
  --error=err/%x_%j.err \
  --mail-type=END,FAIL \
  --mail-user=<USER_ID>@studbocconi.it \
  --time=02:00:00 \
  --gres=gpu:4g.40gb:1 \
  --cpus-per-task=8 \
  --mem=32G \
  --wrap='cd /mnt/beegfsstudents/home/<USER_ID>/superclip-recon && export RUN_NAME=pilot_varA VARIANT=A LAMBDA_RECON=0 MASK_RATIO=0.15 SEED=43 BATCH_SIZE=64 EPOCHS=1 EVAL_MAX_IMAGES=128 TRAIN_MODE=auto SAVE_STRATEGY=last_and_best KEEP_LAST_K=1 && bash slurm/run_one_experiment.sh'
```

---

## 3. Lambda sweeps

### 3.1 Variant-parameterized sweep

`slurm/run_lambda_sweep.sh` now supports both variants with the same script.

Default grid:

```text
0, 0.1, 0.5, 0.75, 1, 1.5, 2, 5
```

Expected naming pattern:

- `results/ablations/lambda_<tag>_varA_s<seed>_b<batch>.json`
- `results/ablations/lambda_<tag>_varB_s<seed>_b<batch>.json`
- `results/ablations/lambda_sweep_variant_A_summary_s<seed>_b<batch>.jsonl`
- `results/ablations/lambda_sweep_variant_B_summary_s<seed>_b<batch>.jsonl`

This variant tag in filenames is important so A and B sweeps do not overwrite each other.

### 3.2 Variant A sweep

```bash
sbatch \
  --job-name=superclip-lambda-sweep-a \
  --account=<USER_ID> \
  --partition=stud \
  --qos=stud \
  --exclude=gnode04 \
  --output=out/%x_%j.out \
  --error=err/%x_%j.err \
  --mail-type=END,FAIL \
  --mail-user=<USER_ID>@studbocconi.it \
  --time=12:00:00 \
  --gres=gpu:4g.40gb:1 \
  --cpus-per-task=8 \
  --mem=32G \
  --wrap='cd /mnt/beegfsstudents/home/<USER_ID>/superclip-recon && export VARIANT=A SEED=43 BATCH_SIZE=128 MASK_RATIO=0.15 && bash slurm/run_lambda_sweep.sh'
```

### 3.3 Variant B sweep

```bash
sbatch \
  --job-name=superclip-lambda-sweep-b \
  --account=<USER_ID> \
  --partition=stud \
  --qos=stud \
  --exclude=gnode04 \
  --output=out/%x_%j.out \
  --error=err/%x_%j.err \
  --mail-type=END,FAIL \
  --mail-user=<USER_ID>@studbocconi.it \
  --time=12:00:00 \
  --gres=gpu:4g.40gb:1 \
  --cpus-per-task=8 \
  --mem=32G \
  --wrap='cd /mnt/beegfsstudents/home/<USER_ID>/superclip-recon && export VARIANT=B SEED=43 BATCH_SIZE=128 MASK_RATIO=0.15 && bash slurm/run_lambda_sweep.sh'
```

### 3.4 What to do when a sweep partially succeeds

If a sweep fails partway through:

1. keep all existing result JSONs
2. sync them to your laptop if needed
3. free storage by deleting checkpoint directories, not result JSONs
4. rerun only the missing lambdas as one-off jobs

Do **not** rerun the whole sweep unless outputs were corrupted or overwritten.

### 3.5 One-off reruns after a partial sweep

When rerunning a missing lambda manually, explicitly set:

- `SAVE_DIR`
- `RESULTS_FILE`

Otherwise the outputs may land in top-level `results/` and `checkpoints/` instead of the ablation folders.

Example:

```bash
sbatch \
  --job-name=varb-lambda-1p5 \
  --account=<USER_ID> \
  --partition=stud \
  --qos=stud \
  --exclude=gnode04 \
  --output=out/%x_%j.out \
  --error=err/%x_%j.err \
  --mail-type=END,FAIL \
  --mail-user=<USER_ID>@studbocconi.it \
  --time=12:00:00 \
  --gres=gpu:4g.40gb:1 \
  --cpus-per-task=8 \
  --mem=32G \
  --wrap='cd /mnt/beegfsstudents/home/<USER_ID>/superclip-recon && export RUN_NAME=lambda_1p5_varB_s43_b128 VARIANT=B LAMBDA_RECON=1.5 MASK_RATIO=0.15 SEED=43 BATCH_SIZE=128 EPOCHS=10 EVAL_MAX_IMAGES=5000 TRAIN_MODE=auto SAVE_STRATEGY=last_and_best KEEP_LAST_K=1 SAVE_DIR=./checkpoints/ablations/lambda_1p5_varB_s43_b128 RESULTS_FILE=./results/ablations/lambda_1p5_varB_s43_b128.json && bash slurm/run_one_experiment.sh'
```

---

## 4. Sequential confirmation batches

For 6 to 8 related runs, prefer a custom sequential script that:

- sources `slurm/common.sh`
- activates the env once
- creates a dedicated results/checkpoints folder
- calls `slurm/run_one_experiment.sh` repeatedly with `tee` into per-run logs

This pattern was used successfully for:

- `core8`
- `confirm6`
- paired baseline vs recon confirmation jobs

Recommended dedicated output folders:

- `results/core8`, `checkpoints/core8`
- `results/confirm6`, `checkpoints/confirm6`
- `results/compositional_round2`, `checkpoints/compositional_round2`

This keeps later sync and cleanup manageable.

---

## 5. Compositional evaluation

### 5.1 Available evaluator

The repo includes `eval_compositional.py` and `slurm/run_compositional_eval.sh`.

Supported benchmark flag:

```bash
--benchmark winoground
--benchmark aro
--benchmark all
```

Do **not** use `--benchmarks`; that flag is invalid.

### 5.2 Recommended benchmark choice

Use **ARO** by default for the course project.

Why:

- it is already implemented
- it does not require Hugging Face auth in the same fragile way as Winoground
- it gives a direct compositional probe with less operational friction

Winoground is fine if you already have a valid `HF_TOKEN` and want an extra supporting check.

### 5.3 First-run cache warm-up for ARO and Winoground

The first evaluation-only run on a fresh account or freshly cleaned cache may spend real time on:

- Hugging Face dataset downloads
- split generation
- cache population under `HF_HOME` / `HF_DATASETS_CACHE`

This is normal. The log may show long stretches of `Downloading data`, `Generating test split`, or similar output before the evaluator reaches steady-state example progress.

The first Winoground fetch is especially large compared with ARO. If a job appears stuck at `0%` during its first download, check quota and cache paths before concluding that the evaluator itself is broken.

### 5.4 Which checkpoints to evaluate

For a fair comparison, evaluate matched baseline vs recon pairs with the same seed.

Good examples used in this project:

- `confirm_baseline_varA_l0_s101_b128_e10` vs `confirm_reconA_l1_s101_b128_e10`
- `confirm_baseline_varA_l0_s102_b128_e10` vs `confirm_reconA_l1_s102_b128_e10`
- fresh seed-103 matched pairs created only for compositional follow-up

### 5.5 Best checkpoint selection

With `last_and_best`, two checkpoints often remain, for example `epoch_9.pt` and `epoch_10.pt`.

Do **not** blindly assume `epoch_10.pt` is the best checkpoint. If the distinction matters:

- inspect the run log for which checkpoint was designated best
- or use an explicit helper function in your evaluation script to choose the intended checkpoint consistently

### 5.6 Stock compositional script caveat

`slurm/run_compositional_eval.sh` is fine for simple baseline / variant_a / variant_b folder layouts, but it is not enough for arbitrary named checkpoint families like:

- ablation runs
- `confirm6`
- fresh compositional-round training runs

For those, use a small custom sequential script that points directly at the desired checkpoint directories.

### 5.7 Safe compositional workflow

1. verify the checkpoint dirs you need actually exist on HPC
2. if they do not, copy them back from your laptop first
3. run ARO on existing matched pairs
4. only then train any fresh compositional follow-up pairs
5. run ARO on the new pair

---

## 6. Storage management

### 6.1 What to keep

Must keep:

- result JSON files
- sweep summary JSONLs
- per-run logs if you want later auditability

Optional:

- checkpoints
- cached datasets or model caches beyond what is needed for current jobs

### 6.2 What to delete first when storage is tight

Delete in this order:

1. old smoke / pilot checkpoints
2. old sweep checkpoint directories already synced to your laptop
3. incomplete one-off rerun checkpoint directories
4. large caches under `~/.cache`
5. conda package caches if still necessary

### 6.3 Quota checks

```bash
lquota
du -sh checkpoints results logs out err ~/.cache ~/.conda 2>/dev/null
```

If usage is near 50 GB, do not submit long jobs until you free space.

### 6.4 Sync then delete

A safe pattern that worked repeatedly during the project:

1. sync result JSONs, logs, and any needed checkpoints to the laptop
2. verify the files exist locally
3. delete the checkpoint directories on HPC
4. recheck `lquota`
5. submit the next jobs

---

## 7. Monitoring and post-mortem

### 7.1 During training or eval-only runs

Training progress usually lives in `err/` because `tqdm` and warnings often write there.

The same is often true for pure ARO or Winoground evaluation jobs, so `err/` is still the first place to look for example-level progress.

```bash
tail -f err/<job>.err
```

### 7.2 During sequential batches or sweep wrappers

Sweep-level progress and multi-run sequential scripts often summarize in `out/`.

```bash
tail -f out/<job>.out
```

### 7.3 Post-mortem tools

Use:

```bash
sacct -j <JOBID> -o JobID,JobName,State,Elapsed,ExitCode,NodeList
```

Preferred detailed form:

```bash
sacct -j <JOBID> -P -o JobID,State,ExitCode,Reason,DerivedExitCode,MaxRSS,MaxVMSize,ReqMem
```

### 7.4 Reading the real log tail

Because `tqdm` rewrites lines, a plain `tail` can be misleading. If the end looks strange:

```bash
tail -c 2000 err/<job>.err | od -c | tail -40
```

### 7.5 When a sequential batch ends with `SIGNAL Terminated`

Treat this as a partial-batch recovery problem first, not as proof that every subrun failed.

Use this sequence:

1. run `sacct` for the wrapper job
2. inspect `out/`, `err/`, and any per-run logs under `logs/`
3. identify the last subrun or evaluator that clearly finished
4. keep any result JSONs or completed evaluation outputs already written
5. rerun only the unfinished tail

This matters because a long sequential wrapper may complete several valid subruns before SLURM terminates the job.

---

## 8. Troubleshooting

### `common.sh` not found under `/var/spool/slurmd/job...`

Submit with wrapped `sbatch`, or use a script that `cd`s into the repo and sources the real `slurm/common.sh`.

### `QOSMaxSubmitJobPerUserLimit`

You submitted too many jobs. Cancel extras if needed and switch to a sequential batch.

### `PD (QOSMaxJobsPerUserLimit)` or `PD (Priority)` or `PD (Resources)`

These are queue states, not code failures.

### `MASK_RATIO must be set`

You called `slurm/run_one_experiment.sh` without exporting `MASK_RATIO`. Add it explicitly.

### GPU smoke fails with `KeyError: 'train_mode'` from `sanity_check.py`

This means the repo is using an older broken loss-call pattern in `sanity_check.py`. Update the file so it calls `total_loss(...)` with the full keyword API, then rerun the smoke test.

### `tail: cannot open 'err/...` or `find: 'results': No such file or directory`

You are probably in `~` instead of the repo. `cd /mnt/beegfsstudents/home/<USER_ID>/superclip-recon` first.

### `eval_compositional.py: error: unrecognized arguments: --benchmarks aro`

The flag is `--benchmark aro`, singular.

### A compositional batch fails immediately with a tiny `.out` and no real work

The script may have been corrupted while pasting. Recreate it with a here-doc, wait for the prompt to return, inspect it with `sed -n`, then submit.

### `find ... lambda_1_varB ...` shows files in top-level `results/` or `checkpoints/`

A one-off rerun probably omitted `SAVE_DIR` and `RESULTS_FILE`. Sync those files from their real location, then update the rerun command next time.

### Sweep job failed but sweep `.err` is empty

Check:

- sweep `.out`
- summary JSONL
- per-run logs under `logs/`
- quota via `lquota`

An empty `.err` with a long runtime often means a storage or write-stage failure rather than a Python traceback.

### Exit `120:0` with no traceback

Check `NodeList` in `sacct` first. If the job ran on `gnode04`, resubmit with `--exclude=gnode04`.

### `fatal: not a git repository (or any parent up to mount point /mnt)` at the end of a run

This is usually a provenance warning, not a training failure. The default HPC sync excludes `.git`, so the best-effort `git rev-parse HEAD` call may fail even when the model run itself succeeded.

Treat it as benign if:

1. `sacct` says the job completed normally
2. the expected result JSON exists
3. the log shows the usual completion markers such as `Results saved to ...`

### `*** JOB ... DUE to SIGNAL Terminated ***`

Treat this as an external wrapper termination first. Check how much of the batch completed before the kill, keep the finished artifacts, and rerun only the unfinished tail.

### Benign warning clusters: `QuickGELU`, `TRANSFORMERS_CACHE`, or `torch.cuda.amp` deprecations

These warnings are noisy but are not, by themselves, evidence that the run failed.

- `QuickGELU` is a model-config warning from `open_clip`
- `TRANSFORMERS_CACHE` is a deprecation warning; the cache still works, but newer tooling prefers `HF_HOME`
- `torch.cuda.amp` warnings indicate future API migration work, not an immediate runtime break

### Job finishes with `COMPLETED 0:0` but you are not sure it succeeded

Check three things:

1. `sacct` says `COMPLETED 0:0`
2. the expected result JSON exists
3. the log contains `Results saved to ...`

### `CondaToSNonInteractiveError`

Accept the terms once with the two `conda tos accept` commands in §0.2.

### `ModuleNotFoundError` or `matplotlib is required`

Activate the `superclip` env first.

---

## 9. Clean rerun workflow

When you want to restart from a clean state after code changes:

### Step 1 — Stop current jobs

```bash
squeue -u <USER_ID>
scancel -u <USER_ID>
squeue -u <USER_ID>
```

### Step 2 — Clean outputs only

```bash
cd /mnt/beegfsstudents/home/<USER_ID>/superclip-recon

rm -rf checkpoints results out err logs
find . -type d \( -name '__pycache__' -o -name '.pytest_cache' -o -name '.mypy_cache' \) -prune -exec rm -rf {} +
mkdir -p checkpoints results out err logs
```

Do not delete `data/coco` or `vocab.json`.

### Step 3 — Resync code from your Mac

Use the rsync command in §1.1.

### Step 4 — Verify repo and data

Use the checks in §0.4.

### Step 5 — Run preflight, smoke, pilot

Only then move on to sweeps or larger confirmation batches.

---
