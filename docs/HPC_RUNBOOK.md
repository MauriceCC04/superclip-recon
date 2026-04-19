# HPC Runbook — SuperCLIP-Recon (Bocconi, updated)

**Target cluster**: partition `stud`, account `<USER_ID>`, QoS `stud`, one MIG-sliced A100 40GB (`gpu:4g.40gb:1`), home-quota-limited storage, no separate scratch assumed.

This runbook keeps the original **gated** flow, but updates it with what actually worked on Bocconi:

- use **wrapped `sbatch` submissions** from the repo root
- expect long training progress in **`err/`**, not `out/`
- submit **one job at a time**, or at most two total, because of Bocconi QoS job limits
- keep an eye on **home quota** before ablations, reruns, or custom job scripts
- for **seed-controlled reruns**, submit `python train.py ... --seed ...` directly rather than relying on `run_one_experiment.sh`
- if your winning checkpoint is an **ablation**, use a **custom compositional-eval job** that targets that checkpoint explicitly

**Important Bocconi-specific note**: the safest submission pattern is:

```bash
sbatch --wrap='cd /mnt/beegfsstudents/home/<USER_ID>/superclip-recon && bash slurm/<script>.sh'
```

Direct `sbatch slurm/<script>.sh` can fail with a `/var/spool/.../common.sh: No such file or directory` error if the job starts from SLURM’s spool directory instead of the repo root.

---

## 0. Access and sync

### From your Mac

Off campus: connect to the Bocconi VPN first.

Prefer `rsync` over copying the whole project tree blindly:

```bash
rsync -av \
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
  ~/PycharmProjects/superclip-recon/ \
  bocconi-hpc:/mnt/beegfsstudents/home/<USER_ID>/superclip-recon/
```

Then connect:

```bash
ssh bocconi-hpc
cd /mnt/beegfsstudents/home/<USER_ID>/superclip-recon
```

If the alias is not configured, use:

```bash
ssh <username>@slogin.hpc.unibocconi.it
cd /mnt/beegfsstudents/home/<USER_ID>/superclip-recon
```

Do **not** use a command like this from your Mac:

```bash
ssh <username>@slogin.hpc.unibocconi.it cd ~/superclip-recon
```

That can resolve to the wrong home/path context and is harder to debug.

---

## Gate 0 — Local synthetic tests (no COCO, no GPU)

**Purpose**: catch shape/logic/pipeline bugs before touching the cluster.

```bash
python tests/run_tests.py
```

| Verdict | Criterion | Action |
|---|---|---|
| **PASS** | All tests pass. | Proceed to Gate 1. |
| **WARN** | Only environment-dependent tests fail (no internet for CLIP weights, no `datasets` package locally). | Proceed to Gate 1. |
| **FAIL** | Any core masking / retention / schema / pipeline test fails. | Stop and fix locally. |

---

## Gate 1 — Cluster preflight

**Purpose**: verify environment, imports, GPU visibility, data presence, cache vars, storage, and one tiny runtime step.

### One-time setup on Bocconi

```bash
cd /mnt/beegfsstudents/home/<USER_ID>/superclip-recon

module load miniconda3
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

bash slurm/setup_env_ready.sh
```

### Manual env activation when needed

```bash
module load miniconda3
eval "$(conda shell.bash hook)"
conda activate superclip
```

### Verify basics

```bash
python -c "import torch; print(torch.__version__)"
ls data/coco/train2017 | head -5
ls data/coco/val2017 | head -5
ls data/coco/annotations | head
ls vocab.json
```

`phrases.json` is **recommended for Variant B but optional**. Missing `phrases.json` should be treated as a warning, not a blocker.

### Submit preflight

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

Monitor and inspect:

```bash
squeue -u <USER_ID>
tail -f out/superclip-preflight_<JOBID>.out
cat results/preflight/preflight_report.json | python -m json.tool | head -80
```

### Acceptable outcomes

| Verdict | Criterion | Action |
|---|---|---|
| **PASS** | Overall PASS. | Proceed to Gate 2. |
| **WARN** | Only optional issues, e.g. missing `phrases.json`. | Proceed, but note them. |
| **FAIL** | Import/GPU/runtime/storage failure. | Stop and fix. |

---

## Gate 2 — GPU smoke

**Purpose**: confirm end-to-end GPU training + quick retrieval evaluation on the real compute node.

Use the preferred smoke script, not legacy `run_smoke.sh`:

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

Monitor and inspect:

```bash
squeue -u <USER_ID>
tail -f out/superclip-gpusmoke_<JOBID>.out
cat results/smoke/gpu_smoke_results.json | python -m json.tool | head -80
```

What you want:

- finite losses
- checkpoint saved
- retrieval JSON produced
- `gpu_peak_mem_gb` comfortably below 40 GB

---

## Gate 3 — Pilot baseline

**Purpose**: run one real epoch, project full-run time and storage, and get a readiness verdict.

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

On Bocconi, long-running training progress is usually visible in `err/<job>_<jobid>.err`, because `tqdm` progress bars and warnings go to stderr.

```bash
squeue -u <USER_ID>
tail -f err/superclip-pilot_<JOBID>.err
cat results/pilot/pilot_baseline.json | python -m json.tool | head -80
```

What matters:

- `readiness == PASS`
- projected 10-epoch runtime fits comfortably inside your wall-time budget
- projected checkpoint storage stays below quota risk

---

## Gate 4 — Main experiments

**Purpose**: run baseline, Variant A, and Variant B.

### Bocconi policy for this gate

- submit **one job at a time**, or at most two total
- if the second job is pending with `QOSMaxJobsPerUserLimit`, that is expected
- watch **`err/` first** for live training progress
- always `cd` into the repo before `tail`, `cat`, or `python -m json.tool`

### Optional phrase extraction for Variant B

```bash
module load miniconda3
eval "$(conda shell.bash hook)"
conda activate superclip
python extract_phrases.py --use_regex --coco_root ./data/coco --output phrases.json
```

### Main runs

You can still use `bash slurm/submit_main_experiments.sh`, but the safest documented pattern is a wrapped job from the repo root.

#### Baseline

```bash
sbatch \
  --job-name=superclip-baseline \
  --account=<USER_ID> \
  --partition=stud \
  --qos=stud \
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

#### Variant A

```bash
sbatch \
  --job-name=superclip-variant_a \
  --account=<USER_ID> \
  --partition=stud \
  --qos=stud \
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

#### Variant B

```bash
sbatch \
  --job-name=superclip-variant_b \
  --account=<USER_ID> \
  --partition=stud \
  --qos=stud \
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

### Monitoring

```bash
squeue -u <USER_ID>
tail -f err/superclip-baseline_<JOBID>.err
tail -f err/superclip-variant_a_<JOBID>.err
tail -f err/superclip-variant_b_<JOBID>.err
```

### Inspect results

```bash
cd /mnt/beegfsstudents/home/<USER_ID>/superclip-recon
ls -lh results/baseline.json results/variant_a.json results/variant_b.json
python -m json.tool results/baseline.json | head -40
python -m json.tool results/variant_a.json | head -40
python -m json.tool results/variant_b.json | head -40
```

**Practical note**: the successful main runs in this project ended up recorded with `batch_size: 128` in the JSON results. Keep that in mind when you compare or rerun experiments; match the batch size of the runs you are comparing.

---

## Gate 4b — Controlled seed reruns

**Purpose**: compare baseline vs a chosen setting under the **same** seed and **same** batch size.

Do **not** rely on `run_one_experiment.sh` for this if you need a custom seed. Submit `python train.py ... --seed <n>` directly.

### Example: clean matched reruns

#### Baseline, seed 43, batch 128

```bash
sbatch \
  --job-name=superclip-baseline-s43-b128 \
  --account=<USER_ID> \
  --partition=stud \
  --qos=stud \
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

#### Lambda 1.0, seed 43, batch 128

```bash
sbatch \
  --job-name=superclip-lambda1p0-s43-b128 \
  --account=<USER_ID> \
  --partition=stud \
  --qos=stud \
  --output=out/%x_%j.out \
  --error=err/%x_%j.err \
  --mail-type=END,FAIL \
  --mail-user=<USER_ID>@studbocconi.it \
  --time=12:00:00 \
  --gres=gpu:4g.40gb:1 \
  --cpus-per-task=8 \
  --mem=32G \
  --wrap='cd /mnt/beegfsstudents/home/<USER_ID>/superclip-recon && source slurm/common.sh && cd "$PROJECT_ROOT" && print_job_header && activate_env && ensure_data_file data/coco/train2017 && ensure_data_file data/coco/val2017 && ensure_data_file data/coco/annotations/captions_val2017.json && ensure_data_file vocab.json && python train.py --coco_root ./data/coco --vocab_path ./vocab.json --run_name lambda_1.0_s43_b128 --variant A --lambda_recon 1.0 --mask_ratio 0.15 --epochs 10 --batch_size 128 --lr 1e-5 --eval_max_images 5000 --save_strategy last_and_best --keep_last_k 1 --seed 43 --save_dir ./checkpoints/ablations/lambda_1.0_s43_b128 --results_file ./results/ablations/lambda_1.0_s43_b128.json'
```

This is the preferred pattern whenever you need **clean seed comparisons**.

---

## Gate 5 — Ablations

Do not start ablations until:

- main runs are green
- storage is checked
- you have quota headroom for more checkpoints and logs

### Storage check first

```bash
lquota
du -sh checkpoints results out err
df -h $HOME
```

### Submit ablations

```bash
bash slurm/submit_ablations.sh
squeue -u <USER_ID>
```

If some submissions fail with `QOSMaxSubmitJobPerUserLimit`, that is a cluster policy issue, not necessarily a code issue. Let one running job finish, then resubmit the missing informative points.

### Analyze after ablations

```bash
module load miniconda3
eval "$(conda shell.bash hook)"
conda activate superclip
python analyze_results.py --results_dir ./results --output_dir ./results/figures
ls -lh results/figures
```

---

## Optional — compositional evaluation

Only run this after the main experiments exist.

### Important limitation

The stock `slurm/run_compositional_eval.sh` targets the standard checkpoint folders (`baseline`, `variant_a`, `variant_b`). If your best model is an **ablation checkpoint** such as `lambda_1.0_s43_b128`, use a **custom job script** that targets those exact checkpoint directories.

### Safer custom compositional-eval pattern

Create a temporary script file rather than pasting a huge inline `--wrap` command. Long wrapped one-liners were easy to mangle in practice.

```bash
cat > slurm/run_comp_eval_matched.sh <<'SCRIPT_EOF'
#!/usr/bin/env bash
#SBATCH --job-name=superclip-comp-s43
#SBATCH --account=<USER_ID>
#SBATCH --partition=stud
#SBATCH --qos=stud
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

If `HF_TOKEN` is not set, Winoground may be unavailable; treat that as optional and still keep ARO if it runs.

---

## Storage and cleanup

This project is checkpoint-heavy. A single checkpoint is around **677 MB** in the observed runs. Failed or obsolete runs can block new jobs and even prevent creation of tiny helper scripts if the home quota is full.

### Check quota often

```bash
lquota
du -sh checkpoints results out err
```

### Clean obvious failures before new runs

```bash
rm -rf checkpoints/ablations/<failed_run_dir>
rm -f out/<failed_job_log>.out
rm -f err/<failed_job_log>.err
```

### Prune old runs you no longer need

Keep only the runs you actually need for the report and seed comparison.

### Important symptom

If you see:

```bash
Disk quota exceeded
```

fix storage **before** trying to submit more jobs or create more scripts.

---

## Monitoring and result inspection

### Running jobs

```bash
squeue -u <USER_ID>
tail -f err/<JOBNAME>_<JOBID>.err
tail -f out/<JOBNAME>_<JOBID>.out
```

### Completed jobs

For finished jobs, prefer `sacct`:

```bash
sacct -j <JOBID> --format=JobID,State,Elapsed,ExitCode
```

`scontrol show job <JOBID>` is most useful while the job is still active or recently finished.

### Always `cd` into the repo first

```bash
cd /mnt/beegfsstudents/home/<USER_ID>/superclip-recon
```

Otherwise `tail`, `cat`, or `python -m json.tool` may look in the wrong place.

---

## Analysis

Run analysis from the activated `superclip` environment:

```bash
module load miniconda3
eval "$(conda shell.bash hook)"
conda activate superclip
python -m pip install "matplotlib>=3.8,<3.10"
python analyze_results.py --results_dir ./results --output_dir ./results/figures
ls -lh results/figures
```

If you get a `matplotlib is required` error, you are almost certainly outside the correct environment.

---

## Safety defaults (recommended)

| Setting | Recommended value | Why |
|---|---:|---|
| `--save_strategy` | `last_and_best` | Keeps only the most useful checkpoints. |
| `--keep_last_k` | `1` | Minimizes checkpoint growth under quota limits. |
| `--save_optimizer_state` | `False` | Reduces checkpoint size. |
| `--eval_max_images` | `5000` for main, `1000` for pilot, `128` for smoke | Keeps sanity stages cheap. |
| Pilot batch size | `64` | Conservative pilot default that passed readiness. |
| Main / matched rerun batch size | `128` | Matches the successful main runs and clean seed comparisons. |
| Submission style | wrapped `sbatch` from repo root | Avoids SLURM spool-path failures. |
| Training monitoring | `err/` first | `tqdm` and warnings usually go to stderr. |

---

## When something goes wrong

### `CondaToSNonInteractiveError`

```bash
module load miniconda3
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
```

### `/var/spool/.../common.sh: No such file or directory`

Submit with:

```bash
sbatch --wrap='cd /mnt/beegfsstudents/home/<USER_ID>/superclip-recon && bash slurm/<script>.sh'
```

### `QOSMaxSubmitJobPerUserLimit` or `QOSMaxJobsPerUserLimit`

Wait for a running job to finish, then resubmit the next one.

### No progress in `.out`

Check `.err` first:

```bash
tail -f err/<JOBNAME>_<JOBID>.err
```

### `matplotlib is required`

Activate the conda env and install with `python -m pip`, not bare `pip`.

### `Disk quota exceeded`

Free space in `checkpoints/`, `results/`, `out/`, and `err/` before trying again.

### Missing COCO / vocab / phrases

```bash
bash slurm/setup_env_ready.sh --step data
bash slurm/setup_env_ready.sh --step vocab
python extract_phrases.py --use_regex --coco_root ./data/coco --output phrases.json
```

### CLIP weights cannot download on compute nodes

Pre-cache them on the login node:

```bash
bash slurm/setup_env_ready.sh --step cache
```

---

## End-of-run checklist

When jobs are finished:

```bash
exit
rsync -av bocconi-hpc:/mnt/beegfsstudents/home/<USER_ID>/superclip-recon/results/ ~/PycharmProjects/superclip-recon/results/
rsync -av bocconi-hpc:/mnt/beegfsstudents/home/<USER_ID>/superclip-recon/checkpoints/ ~/PycharmProjects/superclip-recon/checkpoints/
```

You can close the terminal after submission; SLURM jobs continue running after your SSH session ends.
