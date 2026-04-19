# HPC Runbook — SuperCLIP-Recon (Bocconi)

**Cluster:** Bocconi Jupiter I, `slogin.hpc.unibocconi.it`
**User:** `<USER_ID>`, account `<USER_ID>`, partition `stud`, QoS `stud`
**GPU:** one MIG-sliced A100, request as `gpu:4g.40gb:1` (40 GB slice)
**Storage:** home-only, quota **50 GB**, check with `lquota`
**Repo root on cluster:** `/mnt/beegfsstudents/home/<USER_ID>/superclip-recon`

This runbook is the **single source of truth** for running SuperCLIP-Recon on Bocconi HPC. It is written to be followed top-to-bottom by a collaborator or a future-you, with every `sbatch` command stated in full and every failure mode documented. Everything in it has been learned the hard way from real runs — see `INCIDENTS.md` in the same directory for the decision log behind these conventions.

---

## Why the commands look the way they do

You will notice three patterns that repeat in every submission in this runbook. All three are direct responses to specific failures, not stylistic choices. If you change them, you will reintroduce the failures.

1. **All jobs are submitted as `sbatch --wrap='cd <REPO> && ... && bash slurm/<script>.sh'`** rather than `sbatch slurm/<script>.sh`. Bocconi copies the submitted script into `/var/spool/slurmd/jobNNNN/` and runs it from there, so `slurm/common.sh`'s `BASH_SOURCE[0]`-based path resolution fails with `/var/spool/.../common.sh: No such file or directory`. The wrap pattern runs the script from the repo, so `BASH_SOURCE[0]` resolves correctly.
2. **Live progress is tailed from `err/`, not `out/`.** `tqdm` writes progress bars to stderr, and most Python warnings do too. Epoch progress shows up in `err/<JOBNAME>_<JOBID>.err`; `out/` is mostly empty until the job finishes.
3. **Jobs are submitted one at a time, two at most.** The `stud` QoS enforces `QOSMaxSubmitJobPerUserLimit` (queue cap) and `QOSMaxJobsPerUserLimit` (running cap). Batch-submitting many ablations will reject everything past the 1st or 2nd; submitting two at a time is the safe maximum.

---

## 0. Access and first-time setup

I have taken most of my set up information and information about the cluster from Bocconi's official HPC documentation and support. If you have not yet set up your access, please follow their instructions first: https://bocconi.sharepoint.com/sites/BocconiStudentsHPC/SitePages/Home.aspx

### 0.1. SSH from your Mac

Off campus, connect to the Bocconi VPN first. Then:

If the alias is not set up, use the full form:

```bash
ssh <USER_ID>@slogin.hpc.unibocconi.it
cd /mnt/beegfsstudents/home/<USER_ID>/superclip-recon
```

Do **not** combine ssh and cd in one line like `ssh <USER_ID>@... cd ~/superclip-recon`. That resolves to the wrong home/path on Bocconi and is painful to debug.

### 0.2. Sync code from your Mac

Prefer `rsync` with explicit excludes over a blind copy:

```bash
rsync -av \
  --exclude '.venv' --exclude 'venv' \
  --exclude '__pycache__' --exclude '.pytest_cache' --exclude '.mypy_cache' \
  --exclude '.DS_Store' --exclude '.git' \
  --exclude 'data/coco' --exclude 'checkpoints' --exclude 'results' \
  ~/PycharmProjects/superclip-recon/ \
  bocconi-hpc:/mnt/beegfsstudents/home/<USER_ID>/superclip-recon/
```

Never `rsync` `data/coco` or `checkpoints` — you have 50 GB total, COCO alone is 20+ GB (already on the cluster), and one run's checkpoints are ~1.4 GB.

### 0.3. One-time conda environment setup

On first login to the cluster, run the repo's setup script from the login node:

```bash
cd /mnt/beegfsstudents/home/<USER_ID>/superclip-recon

module load miniconda3
# Accept conda ToS for the default channels — without this, env creation errors out
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

bash slurm/setup_env_ready.sh
```

This creates the `superclip` conda env (Python 3.12), installs `requirements.txt`, pre-caches OpenCLIP weights, downloads COCO, and builds `vocab.json`. It does **not** build `phrases.json` — Variant B extracts phrases inline per caption and no longer reads that file.

If you need to reactivate the env in an interactive session later:

```bash
module load miniconda3
eval "$(conda shell.bash hook)"
conda activate superclip
```

### 0.4. Sanity: verify data and artifacts are in place

```bash
cd /mnt/beegfsstudents/home/<USER_ID>/superclip-recon
ls data/coco/train2017 | head -3
ls data/coco/val2017 | head -3
ls data/coco/annotations
ls vocab.json
```

`phrases.json` is optional — missing is not a blocker.

---

## 1. Gates — run once, in order, before any real training

The repo is organized around progressively more expensive sanity checks. Each "gate" is cheap and rules out a class of failures. Skipping a gate is almost always a false economy.

### Gate 0 — Local synthetic tests (no cluster needed)

Run on your Mac, before pushing to the cluster:

```bash
python tests/run_tests.py
```

**Verdicts:**

| Verdict | Criterion | Action |
|---|---|---|
| PASS | All tests pass | Proceed to Gate 1 |
| WARN | Only environment-dependent tests fail (no internet for CLIP weights, `datasets` missing locally) | Proceed to Gate 1 |
| FAIL | Any core masking/retention/schema/pipeline test fails | Stop and fix locally — do not burn cluster time on a known-broken pipeline |

### Gate 1 — Cluster preflight

Runs `tools/hpc_preflight.py` on a compute node: checks imports, GPU visibility, data presence, cache vars, storage, and does one tiny forward+backward+eval+checkpoint step.

```bash
sbatch \
  --job-name=superclip-preflight \
  --account=<USER_ID> \
  --partition=stud \
  --qos=stud \
  --output=out/%x_%j.out \
  --error=err/%x_%j.err \
  --mail-type=END,FAIL \
  --mail-user=<USER_ID>@studbocconi.it \
  --time=00:30:00 \
  --gres=gpu:4g.40gb:1 \
  --cpus-per-task=4 \
  --mem=16G \
  --wrap='cd /mnt/beegfsstudents/home/<USER_ID>/superclip-recon && bash slurm/run_preflight.sh'
```

Monitor:

```bash
squeue -u <USER_ID>
tail -f out/superclip-preflight_<JOBID>.out      # preflight writes to stdout
cat results/preflight/preflight_report.json | python -m json.tool | head -80
```

**Verdicts:**

| Verdict | Criterion | Action |
|---|---|---|
| PASS | Overall PASS | Proceed to Gate 2 |
| WARN | Only optional failures (missing `phrases.json`, home quota > 70%) | Proceed, note them |
| FAIL | Import / GPU / runtime / storage failure | Stop and fix |

### Gate 2 — GPU smoke

Runs a few real training steps and a small retrieval eval on the compute node. Writes peak memory, checkpoint size, and retrieval metrics to a JSON.

```bash
sbatch \
  --job-name=superclip-gpusmoke \
  --account=<USER_ID> \
  --partition=stud \
  --qos=stud \
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

Look for: finite losses, checkpoint saved, retrieval JSON produced, `gpu_peak_mem_gb` well below 40. Historical peak has been ~3 GB — anything much larger means something is wrong.

### Gate 3 — Pilot baseline

One real epoch of the baseline on conservative settings, then a projection of the full-run cost. This is the last chance to catch surprises cheaply.

```bash
sbatch \
  --job-name=superclip-pilot \
  --account=<USER_ID> \
  --partition=stud \
  --qos=stud \
  --output=out/%x_%j.out \
  --error=err/%x_%j.err \
  --mail-type=END,FAIL \
  --mail-user=<USER_ID>@studbocconi.it \
  --time=04:00:00 \
  --gres=gpu:4g.40gb:1 \
  --cpus-per-task=8 \
  --mem=32G \
  --wrap='cd /mnt/beegfsstudents/home/<USER_ID>/superclip-recon && bash slurm/run_pilot_baseline.sh'
```

Long-running progress is in `err/`:

```bash
tail -f err/superclip-pilot_<JOBID>.err
cat results/pilot/pilot_baseline.json | python -m json.tool | head -80
```

Confirm: `readiness == PASS`, projected 10-epoch runtime fits the 12 h budget, projected checkpoint storage is comfortable.

---

## 2. Main experiments

**Three independent jobs**: baseline, Variant A, Variant B. Each is ~76 minutes at batch 128.

### 2.1. Submission policy

- Submit **one at a time**, at most two total in-flight.
- If a submission is rejected with `QOSMaxSubmitJobPerUserLimit`, wait for a running job to finish, then submit again.
- If a submission lands as `PD (QOSMaxJobsPerUserLimit)`, that is not an error — it will start as soon as the running job exits.

### 2.2. Main runs

#### Baseline (λ=0)

```bash
sbatch \
  --job-name=superclip-baseline \
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
  --wrap='cd /mnt/beegfsstudents/home/<USER_ID>/superclip-recon && export RUN_NAME=baseline VARIANT=A LAMBDA_RECON=0.0 MASK_RATIO=0.15 SAVE_DIR=./checkpoints/baseline RESULTS_FILE=./results/baseline.json SAVE_STRATEGY=last_and_best KEEP_LAST_K=1 EVAL_MAX_IMAGES=5000 && bash slurm/run_one_experiment.sh'
```

#### Variant A (λ=0.5, masked token reconstruction)

```bash
sbatch \
  --job-name=superclip-variant_a \
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
  --wrap='cd /mnt/beegfsstudents/home/<USER_ID>/superclip-recon && export RUN_NAME=variant_a VARIANT=A LAMBDA_RECON=0.5 MASK_RATIO=0.15 SAVE_DIR=./checkpoints/variant_a RESULTS_FILE=./results/variant_a.json SAVE_STRATEGY=last_and_best KEEP_LAST_K=1 EVAL_MAX_IMAGES=5000 && bash slurm/run_one_experiment.sh'
```

#### Variant B (λ=0.5, phrase reconstruction, per-caption inline)

```bash
sbatch \
  --job-name=superclip-variant_b \
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
  --wrap='cd /mnt/beegfsstudents/home/<USER_ID>/superclip-recon && export RUN_NAME=variant_b VARIANT=B LAMBDA_RECON=0.5 MASK_RATIO=0.15 PHRASE_PATH=./phrases.json SAVE_DIR=./checkpoints/variant_b RESULTS_FILE=./results/variant_b.json SAVE_STRATEGY=last_and_best KEEP_LAST_K=1 EVAL_MAX_IMAGES=5000 && bash slurm/run_one_experiment.sh'
```

> **Why `--exclude=gnode04`?** gnode04 has produced recurring exit-code `120:0` failures at eval-start for our jobs, with no Python traceback, while the same config succeeds on gnode02. See `INCIDENTS.md` §"Node failures on gnode04". Keep the exclusion until HPC support confirms the issue is resolved.

### 2.3. Monitoring

```bash
squeue -u <USER_ID>
tail -f err/superclip-baseline_<JOBID>.err
```

### 2.4. Inspecting results

```bash
cd /mnt/beegfsstudents/home/<USER_ID>/superclip-recon
ls -lh results/baseline.json results/variant_a.json results/variant_b.json
python -m json.tool results/baseline.json | head -40
```

**Note:** the successful main runs were all done at `batch_size=128`. If you rerun or compare against these, keep the batch size the same — cross-batch-size comparisons confound the metric.

---

## 3. Matched seed reruns (Gate 4b)

When you want clean comparisons between baseline and a specific ablation under the **same seed and batch size**, do not rely on `run_one_experiment.sh` — it doesn't expose `--seed`. Submit `python train.py ... --seed <n>` directly through `--wrap`.

### Baseline, seed 43, batch 128

```bash
sbatch \
  --job-name=superclip-baseline-s43-b128 \
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
  --wrap='cd /mnt/beegfsstudents/home/<USER_ID>/superclip-recon && source slurm/common.sh && cd "$PROJECT_ROOT" && print_job_header && activate_env && ensure_data_file data/coco/train2017 && ensure_data_file data/coco/val2017 && ensure_data_file data/coco/annotations/captions_val2017.json && ensure_data_file vocab.json && python train.py --coco_root ./data/coco --vocab_path ./vocab.json --run_name baseline_s43_b128 --variant A --lambda_recon 0.0 --mask_ratio 0.15 --epochs 10 --batch_size 128 --lr 1e-5 --eval_max_images 5000 --save_strategy last_and_best --keep_last_k 1 --seed 43 --save_dir ./checkpoints/baseline_s43_b128 --results_file ./results/baseline_s43_b128.json'
```

### Lambda sweep at fixed seed (generic template)

Replace `LAMBDA` with the value you want (e.g. `1.0`, `1.5`, `2.0`). Keep the `_s43_b128` suffix so the results slot into the same comparison table:

```bash
LAMBDA=2.0
LAMBDA_TAG=$(echo "$LAMBDA" | tr '.' 'p')   # e.g. 2.0 -> 2p0

sbatch \
  --job-name=superclip-lambda${LAMBDA_TAG}-s43-b128 \
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
  --wrap="cd /mnt/beegfsstudents/home/<USER_ID>/superclip-recon && source slurm/common.sh && cd \"\$PROJECT_ROOT\" && print_job_header && activate_env && ensure_data_file data/coco/train2017 && ensure_data_file data/coco/val2017 && ensure_data_file data/coco/annotations/captions_val2017.json && ensure_data_file vocab.json && python train.py --coco_root ./data/coco --vocab_path ./vocab.json --run_name lambda_${LAMBDA}_s43_b128 --variant A --lambda_recon ${LAMBDA} --mask_ratio 0.15 --epochs 10 --batch_size 128 --lr 1e-5 --eval_max_images 5000 --save_strategy last_and_best --keep_last_k 1 --seed 43 --save_dir ./checkpoints/ablations/lambda_${LAMBDA}_s43_b128 --results_file ./results/ablations/lambda_${LAMBDA}_s43_b128.json"
```

This is the preferred pattern whenever you need clean, matched comparisons. It is what was used successfully for the project's λ sweep.

---

## 4. Ablations

Only start after main runs are green and storage has been checked.

### 4.1. Storage check

```bash
lquota
du -sh checkpoints results out err
du -sh ~/.cache
```

Target: **< 40 GB used** before starting ablations. A single run peaks around ~2 GB during checkpoint rotation; 40 GB gives headroom for two runs plus the existing main-experiment checkpoints.

Common cleanup to get there:

```bash
# Drop intermediate pilot/smoke artifacts
rm -rf checkpoints/pilot checkpoints/smoke
rm -f results/preflight/_tiny_ckpt.pt

# Drop stale early-epoch checkpoints from completed runs
# (last_and_best keeps `best` + `current`; older epochs are safe to delete)
rm -f checkpoints/baseline/epoch_2.pt checkpoints/baseline_s43_b128/epoch_2.pt
# ... adjust per actual contents
```

### 4.2. Submit ablation grid

Recommended: use matched-seed wrapped submissions (§3) one at a time, not `slurm/submit_ablations.sh`, because submit_ablations reuses defaults that do not match the main runs and can mass-submit beyond the QoS limit.

If you do want to use the batch submitter:

```bash
bash slurm/submit_ablations.sh
squeue -u <USER_ID>
```

Expect `QOSMaxSubmitJobPerUserLimit` rejections on submissions past the first two. That is not a code bug — it is the cluster's policy — and the only fix is to wait and resubmit the remainder later.

### 4.3. Analysis

After runs complete, always activate the env first:

```bash
module load miniconda3
eval "$(conda shell.bash hook)"
conda activate superclip
python analyze_results.py --results_dir ./results --output_dir ./results/figures
ls -lh results/figures
```

If you see `matplotlib is required`, you are almost certainly outside the `superclip` conda env. Activate it and retry.

---

## 5. Compositional evaluation (optional)

Standard `slurm/run_compositional_eval.sh` only targets `checkpoints/baseline`, `checkpoints/variant_a`, `checkpoints/variant_b`. If your best checkpoint lives under `checkpoints/ablations/` (e.g. `lambda_1.0_s43_b128`), do **not** modify the stock script with a long inline `--wrap` — a real script file is safer. Example:

```bash
cat > slurm/run_comp_eval_matched.sh <<'SCRIPT_EOF'
#!/usr/bin/env bash
#SBATCH --job-name=superclip-comp-s43
#SBATCH --account=<USER_ID>
#SBATCH --partition=stud
#SBATCH --qos=stud
#SBATCH --exclude=gnode04
#SBATCH --output=out/%x_%j.out
#SBATCH --error=err/%x_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=<USER_ID>@studbocconi.it
#SBATCH --time=08:00:00
#SBATCH --gres=gpu:4g.40gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G

set -euo pipefail

cd /mnt/beegfsstudents/home/<USER_ID>/superclip-recon
source slurm/common.sh
cd "$PROJECT_ROOT"

print_job_header
activate_env
mkdir -p results

for dir in checkpoints/baseline_s43_b128 checkpoints/ablations/lambda_1.0_s43_b128; do
  NAME=$(basename "$dir")
  LAST_CKPT=$(ls -t "$dir"/epoch_*.pt 2>/dev/null | head -1 || true)

  if [ -n "$LAST_CKPT" ]; then
    echo "Compositional eval: $NAME -> $LAST_CKPT"
    python eval_compositional.py \
      --checkpoint "$LAST_CKPT" \
      --benchmark all \
      --output "./results/compositional_${NAME}.json" \
      ${HF_TOKEN:+--hf_token "$HF_TOKEN"}
  else
    echo "No checkpoint found for $NAME -- skipping"
  fi
done
SCRIPT_EOF

chmod +x slurm/run_comp_eval_matched.sh
sbatch slurm/run_comp_eval_matched.sh
```

If `HF_TOKEN` is unset, Winoground will be skipped; ARO still runs. That's acceptable — Winoground requires HF auth by design.

---

## 6. Monitoring cheatsheet

### Running jobs

```bash
squeue -u <USER_ID>
tail -f err/<JOBNAME>_<JOBID>.err        # live progress (tqdm writes here)
tail -f out/<JOBNAME>_<JOBID>.out        # setup banner + final summary
```

### Completed or failed jobs

```bash
sacct -j <JOBID> -o JobID,JobName,State,Elapsed,ExitCode,NodeList
sacct -j <JOBID> -P -o JobID,State,ExitCode,Reason,DerivedExitCode,MaxRSS,MaxVMSize,ReqMem
```

`sacct` is the right tool for post-mortem; `scontrol show job` only works while the job is still in SLURM's memory.

### After a failure, read the error file ends-first

Because `tqdm` fills err files with carriage-return progress updates, the useful bytes are often at the end. Get them reliably with:

```bash
tail -c 2000 err/<JOBNAME>_<JOBID>.err | od -c | tail -40
grep -iE "error|fatal|kill|terminat|signal|abort|segfault|oom|bus|traceback" \
  err/<JOBNAME>_<JOBID>.err
```

---

## 7. Safety defaults

These are the values that have been validated against real runs on this cluster. Deviating from them should be a deliberate choice, not a default.

| Setting | Recommended value | Why |
|---|---:|---|
| `--save_strategy` | `last_and_best` | Retains only current + best checkpoints, ~1.4 GB per run |
| `--keep_last_k` | `1` | Minimizes checkpoint growth under the 50 GB quota |
| `--save_optimizer_state` | off (default) | Optimizer state would double checkpoint size |
| `--eval_max_images` | `5000` main, `1000` pilot, `128` smoke | Keeps sanity stages cheap |
| `--batch_size` (main) | `128` | Matches the successful completed runs |
| `--batch_size` (pilot) | `64` | Conservative, passed readiness checks |
| GPU request | `gpu:4g.40gb:1` | The MIG slice is a hard constraint of `stud` QoS |
| CPUs / RAM | `8` / `32G` | Used by all successful main runs |
| Wall time | `12:00:00` main, `04:00:00` pilot | 10-epoch runs take ~76 min, fits comfortably |
| `--exclude` | `gnode04` | Until HPC confirms the node is healthy — see §8 |
| Submission style | wrapped `sbatch` from repo root | Avoids SLURM spool-path `common.sh` failure |
| Live monitoring | `err/` first, then `out/` | `tqdm` and warnings go to stderr |

---

## 8. When something goes wrong

These are the failure modes this project has actually hit, in roughly decreasing order of frequency. The full narrative for each is in `INCIDENTS.md`.

### `Disk quota exceeded`

Free space in `checkpoints/`, `results/`, `out/`, `err/`, and `~/.cache` before trying anything else. See §4.1 for the cleanup commands. Do not resubmit until `lquota` shows < 40 GB used.

### `sbatch: error: QOSMaxSubmitJobPerUserLimit`

Cluster policy, not a code bug. You hit the per-user submit-queue cap. Wait for a running job to finish, then resubmit. Plan on submitting one at a time going forward.

### `PD (QOSMaxJobsPerUserLimit)` in `squeue`

Not an error. Your job is queued and will start automatically when a running job exits. Leave it alone.

### `/var/spool/slurmd/jobNNNN/common.sh: No such file or directory`

You submitted `sbatch slurm/<script>.sh` directly. Use the wrapped form:

```bash
sbatch --wrap='cd /mnt/beegfsstudents/home/<USER_ID>/superclip-recon && bash slurm/<script>.sh'
```

### `CondaToSNonInteractiveError`

The cluster refuses to create envs until Terms of Service are accepted for the default channels. One-time fix:

```bash
module load miniconda3
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
```

### `ModuleNotFoundError: No module named 'torch'` / `matplotlib is required ...`

You are outside the `superclip` conda env. Activate it:

```bash
module load miniconda3
eval "$(conda shell.bash hook)"
conda activate superclip
```

### Job fails with exit `120:0` and no Python traceback in `err/`

Check the node. If `NodeList` in `sacct` is `gnode04` — known-bad for this project. Resubmit with `--exclude=gnode04`. If it's a different node, check `sacct -P -o MaxRSS,ReqMem` for host-RAM OOM, and check `lquota` for disk-full.

### Job fails with exit `0:53` and `Elapsed=00:00:00`

Usually a secondary failure — the job could not even start its prolog, often because the filesystem is quota-full or SLURM cancelled it due to a dependency/wrapper issue. Clean storage first, then resubmit.

### No progress in `out/`

Check `err/` instead — `tqdm` writes progress bars to stderr. `out/` is mostly empty during long runs and only fills in at the end.

### Missing COCO / vocab

```bash
bash slurm/setup_env_ready.sh --step data    # download COCO
bash slurm/setup_env_ready.sh --step vocab   # build vocab.json
```

### CLIP weights cannot download on compute nodes

Pre-cache them from the login node:

```bash
bash slurm/setup_env_ready.sh --step cache
```

### Retrieval eval crashes with a shape/alignment error

Rare now, but the `evaluate.py` retrieval pipeline was previously broken by an indentation bug that nested the entire pipeline inside the caption-collection loop. If you see `n_captions_per_image` asserts fire or the pipeline loops forever on the first image, pull the latest `evaluate.py` — test 18 in `tests/run_tests.py` guards against regressions.

---

## 9. End-of-run checklist

When all target runs are finished:

```bash
# From your Mac — pull results back
rsync -av bocconi-hpc:/mnt/beegfsstudents/home/<USER_ID>/superclip-recon/results/ \
          ~/PycharmProjects/superclip-recon/results/

# Optional: pull checkpoints back (much larger; only if needed)
rsync -av bocconi-hpc:/mnt/beegfsstudents/home/<USER_ID>/superclip-recon/checkpoints/ \
          ~/PycharmProjects/superclip-recon/checkpoints/
```

You can safely `exit` your SSH session while jobs are still running — SLURM jobs do not depend on the submitting shell.

Finally, update `INCIDENTS.md` with any new failure modes you encountered. The next collaborator (or the next version of you, in a month) will thank you.