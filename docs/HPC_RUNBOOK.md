# HPC Runbook — SuperCLIP-Recon (Bocconi)

**Cluster:** Bocconi Jupiter I, `slogin.hpc.unibocconi.it`  
**Account / user:** `<USER_ID>`  
**Partition / QoS:** `stud` / `stud`  
**GPU request:** `gpu:4g.40gb:1` (one 40 GB MIG slice of an A100 80GB)  
**Home quota:** 50 GB total  
**Repo root on cluster:** `/mnt/beegfsstudents/home/<USER_ID>/superclip-recon`

This document is the **single source of truth** for syncing, validating, and running SuperCLIP-Recon on Bocconi HPC.

It is written to prevent the failures that have already happened in this project:

- submitting too many jobs and hitting QoS limits
- running repo-relative commands from `~` instead of the repo root
- using `sbatch slurm/<script>.sh` directly and breaking `common.sh`
- filling the 50 GB home quota and getting strange late-job failures
- landing on `gnode04` and getting silent job termination
- running `rsync` from the wrong machine

For the decision history behind these rules, see `docs/INCIDENTS.md`.

---

## Core rules

Before doing anything else, keep these rules in mind.

1. **Run `rsync` from your Mac, not from inside the HPC shell.**
2. **Always `cd` into the repo before running repo-relative commands.**
3. **Submit SLURM jobs with wrapped `sbatch` from the repo root.**
4. **Treat `--exclude=gnode04` as default for long jobs.**
5. **Do not start the next gate until the previous one has finished and been checked.**
6. **Monitor long jobs from `err/`, not `out/`.**
7. **Keep storage below ~40 GB used before long sweeps or ablations.**

---

## Why the commands look the way they do

The repeated patterns in this runbook are deliberate.

### Wrapped `sbatch`

Always submit jobs like:

```bash
sbatch --wrap='cd /mnt/beegfsstudents/home/<USER_ID>/superclip-recon && bash slurm/<script>.sh'
```

Do **not** submit jobs as:

```bash
sbatch slurm/<script>.sh
```

Bocconi copies submitted scripts into `/var/spool/slurmd/jobNNNN/` and runs the copy from there. If the script uses `BASH_SOURCE[0]` to find `slurm/common.sh`, direct submission can break with a path like:

```text
/var/spool/slurmd/jobNNNN/common.sh: No such file or directory
```

The wrapped form avoids that.

### `err/` over `out/`

Training progress bars (`tqdm`) and many warnings are written to **stderr**, not stdout. During long runs, `out/` can look empty while the job is actually training normally.

### One job, maybe two

The `stud` QoS enforces limits on both queued and running jobs. Mass-submitting ablations will hit `QOSMaxSubmitJobPerUserLimit` or sit in `PD (QOSMaxJobsPerUserLimit)`.

That is why this runbook prefers:

- one-off submissions
- sequential sweep jobs
- explicit gate progression

---

## 0. Access and first-time setup

### 0.1. SSH from your Mac

If you are off campus, connect to the Bocconi VPN first.

Use either your alias or the full hostname:

```bash
ssh <USER_ID>@slogin.hpc.unibocconi.it
```

Then enter the repo:

```bash
cd /mnt/beegfsstudents/home/<USER_ID>/superclip-recon
```

Do **not** combine SSH and `cd` into one command like:

```bash
ssh <USER_ID>@slogin.hpc.unibocconi.it cd ~/superclip-recon
```

If the alias or hostname fails to resolve:

- check whether you are on the Bocconi VPN if required
- try the full hostname from your **Mac terminal**
- do not debug this from inside an already-open HPC shell

### 0.2. Sync code from your Mac

**Run the following command from your Mac, not after SSH-ing into the cluster.**

Correct mental model:

- local terminal on Mac -> `rsync` -> HPC repo directory

Wrong mental model:

- SSH into HPC first -> run the same `rsync` from the login node

The sync command:

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

Why the exclusions matter:

- `data/coco` is already on the cluster and large
- `checkpoints`, `results`, `out`, `err`, `logs` are cluster outputs
- `vocab.json` is treated as an already-built cluster artifact unless you deliberately rebuild it
- `--delete` keeps the HPC repo aligned to your local repo, but it also means you should verify critical files after sync

## 0.2b. After sync: verify the repo is still runnable

After a sync, SSH into the cluster and check the repo.

```bash
cd /mnt/beegfsstudents/home/<USER_ID>/superclip-recon

pwd
ls slurm | sort
ls -l slurm/common.sh slurm/run_preflight.sh slurm/run_gpu_smoke.sh slurm/run_one_experiment.sh slurm/run_lambda_sweep.sh
ls data/coco/annotations | head
ls -lh vocab.json
```

This catches accidental deletion of required scripts before you waste a compute allocation.
### 0.3. One-time environment setup

On first login, set up the conda environment from the login node.

```bash
cd /mnt/beegfsstudents/home/<USER_ID>/superclip-recon

module load miniconda3
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

bash slurm/setup_env_ready.sh
```

This creates the `superclip` env, installs dependencies, pre-caches CLIP weights, downloads COCO if needed, and builds `vocab.json`.

If you later log in again and just want to reactivate the env:

```bash
module load miniconda3
eval "$(conda shell.bash hook)"
conda activate superclip
```

### 0.4. Always `cd` before repo-relative commands

Many commands in this runbook assume you are already in the repo.

Before running things like:

```bash
bash slurm/setup_env_ready.sh ...
python train.py ...
ls slurm
test -f vocab.json
```

first do:

```bash
cd /mnt/beegfsstudents/home/<USER_ID>/superclip-recon
```

If you stay in `~`, commands like `bash slurm/setup_env_ready.sh` will fail even if the file exists in the repo.

### 0.5. Basic sanity check

```bash
cd /mnt/beegfsstudents/home/<USER_ID>/superclip-recon

ls data/coco/train2017 | head -3
ls data/coco/val2017 | head -3
ls data/coco/annotations | head
ls -lh vocab.json
```

`phrases.json` is optional and not required for normal training.

---

## 1. Gates — run in order

The project should be run in a strict gate sequence.

Do **not** submit the next stage until the current one has finished and you have checked the outputs.

### Gate 0 — Local tests

Run on your Mac before syncing major code changes.

```bash
python tests/run_tests.py
```

Interpretation:

- **PASS**: all good
- **WARN**: only environment-dependent issues locally
- **FAIL**: do not use cluster time yet

### Gate 1 — Cluster preflight

This verifies:

- imports
- GPU visibility
- data presence
- cache environment
- storage situation
- one tiny forward/backward step
- one tiny retrieval evaluation
- one tiny checkpoint write

Submission:

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

Check it with:

```bash
squeue -u <USER_ID>
tail -50 out/superclip-preflight_<JOBID>.out
cat results/preflight/preflight_report.json | python -m json.tool | head -80
sacct -j <JOBID> -o JobID,JobName,State,Elapsed,ExitCode,NodeList
```

You may proceed if:

- `preflight_report.json` exists
- `overall_status` is `PASS` or acceptable `WARN`
- `sacct` shows `COMPLETED`

A `WARN` caused only by missing `phrases.json` is acceptable.

### Gate 2 — GPU smoke

Runs a few real steps and a small retrieval eval.

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

Look for:

- finite losses
- checkpoint written
- retrieval JSON written
- peak memory comfortably below 40 GB

### Gate 3 — Pilot baseline

One short real baseline run used to catch surprises cheaply.

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

Only proceed to full training if pilot outputs look healthy.

---

## 2. Main experiments

Standard main experiments are:

- baseline (`lambda_recon=0.0`)
- Variant A
- Variant B

Use `--exclude=gnode04` for all long-running jobs unless HPC support confirms that node is healthy again.

### 2.1. Submission policy

Default policy:

- submit **one** long job at a time
- two in-flight jobs is the upper safe limit
- queued `PD` states are normal

If you see:

- `QOSMaxSubmitJobPerUserLimit`
- `QOSMaxJobsPerUserLimit`

that is a scheduler policy issue, not a code bug.

### 2.2. Baseline

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

### 2.3. Variant A

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

### 2.4. Variant B

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

---

## 3. Matched-seed reruns

For clean comparisons, prefer direct `python train.py` submissions when you need control over seed and naming.

Example baseline rerun at seed 43, batch 128:

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

---
## 3b. Sequential lambda sweep in one SLURM job

When you want to evaluate several lambdas without manually resubmitting every ~75 minutes, prefer a **single sequential sweep job**.

This is better than a job array on this cluster because job arrays still create many jobs and can still hit QoS limits.

The sweep script is now **variant-parameterized**, so the same script can run either:

* **Variant A** lambda sweep
* **Variant B** lambda sweep

The launcher delegates each run to `slurm/run_one_experiment.sh`, so Variant B automatically uses the same shared experiment path as normal one-off submissions.

Recommended uses:

* matched seed and batch size
* `--exclude=gnode04`
* one variant sweep at a time
* preflight already confirmed healthy

Default lambda grid:

```text
{0, 0.1, 0.5, 0.75, 1, 1.5, 2, 5}
```

### Variant A sweep

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
  --wrap='cd /mnt/beegfsstudents/home/<USER_ID>/superclip-recon && export VARIANT=A && bash slurm/run_lambda_sweep.sh'
```

### Variant B sweep

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
  --wrap='cd /mnt/beegfsstudents/home/<USER_ID>/superclip-recon && export VARIANT=B && bash slurm/run_lambda_sweep.sh'
```

### Optional overrides

You can also override the seed and batch size at submission time.

Example:

```bash
sbatch \
  --job-name=superclip-lambda-sweep-a-s43 \
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
  --wrap='cd /mnt/beegfsstudents/home/<USER_ID>/superclip-recon && export VARIANT=A SEED=43 BATCH_SIZE=128 && bash slurm/run_lambda_sweep.sh'
```

Operational expectations:

* the sweep should skip lambdas whose results already exist
* one failing lambda should not kill the whole sweep if the script is written defensively
* progress appears in `err/`
* do not submit the sweep until preflight is confirmed healthy
* **run names and summary files should include the variant** so A and B sweeps do not overwrite each other

Expected output pattern:

* per-run results:

  * `results/ablations/lambda_<tag>_varA_s<seed>_b<batch>.json`
  * `results/ablations/lambda_<tag>_varB_s<seed>_b<batch>.json`
* sweep summary:

  * `results/ablations/lambda_sweep_variant_A_summary_s<seed>_b<batch>.jsonl`
  * `results/ablations/lambda_sweep_variant_B_summary_s<seed>_b<batch>.jsonl`

### Important interpretation rule

For this project, keep the baseline story clean:

* **baseline anchor** = Variant A with `lambda_recon=0`
* **extension sweep** = nonzero lambda values
* **Variant B sweep** = phrase-reconstruction extension, not the baseline reproduction


---

## 4. Storage management

Before long runs or ablations, check storage.

```bash
lquota
du -sh checkpoints results out err ~/.cache
```

Target: **stay below ~40 GB used** before starting long sweeps or many runs.

Why:

- the home quota is 50 GB
- quota exhaustion has already caused jobs to fail at write time or fail to start at all

Common cleanup:

```bash
cd /mnt/beegfsstudents/home/<USER_ID>/superclip-recon

rm -rf checkpoints results out err logs
find . -type d \( -name '__pycache__' -o -name '.pytest_cache' -o -name '.mypy_cache' \) -prune -exec rm -rf {} +
mkdir -p checkpoints results out err logs
```

Additional cleanup if needed:

```bash
rm -rf checkpoints/pilot checkpoints/smoke
rm -f results/preflight/_tiny_ckpt.pt
find checkpoints -name 'epoch_2.pt' -delete
```

Retention defaults:

- `--save_strategy last_and_best`
- `--keep_last_k 1`
- optimizer-state saving off unless explicitly needed

---

## 5. Compositional evaluation

Use the stock compositional eval for standard baseline/variant directories.

If you need to evaluate a matched rerun or ablation checkpoint under a different path, create a small dedicated script rather than trying to force a giant inline `--wrap` command.

Keep `--exclude=gnode04` for this as well.

---

## 6. Monitoring

### Running jobs

```bash
squeue -u <USER_ID>
tail -f err/<JOBNAME>_<JOBID>.err
tail -f out/<JOBNAME>_<JOBID>.out
```

Use `err/` first.

### Completed jobs

```bash
sacct -j <JOBID> -o JobID,JobName,State,Elapsed,ExitCode,NodeList
sacct -j <JOBID> -P -o JobID,State,ExitCode,Reason,DerivedExitCode,MaxRSS,MaxVMSize,ReqMem
```

### Read the end of the error log when `tqdm` has overwritten it

```bash
tail -c 2000 err/<JOBNAME>_<JOBID>.err | od -c | tail -40
grep -iE "error|fatal|kill|terminat|signal|abort|segfault|oom|bus|traceback" err/<JOBNAME>_<JOBID>.err
```


---

## 7. Safety defaults

These defaults have been the most reliable for this project.

| Setting                   |                      Recommended value | Why                                                                    |
| ------------------------- | -------------------------------------: | ---------------------------------------------------------------------- |
| save strategy             |                        `last_and_best` | keeps best + current without exploding storage                         |
| keep last k               |                                    `1` | minimizes quota pressure                                               |
| optimizer state           |                                    off | avoids doubling checkpoint size                                        |
| main batch size           |                                  `128` | matches the stable main runs                                           |
| pilot batch size          |                                   `64` | conservative test setting                                              |
| eval max images           | `5000` main, `1000` pilot, `128` smoke | keeps small gates cheap                                                |
| default lambda sweep grid |      `0, 0.1, 0.5, 0.75, 1, 1.5, 2, 5` | covers baseline, in-scope settings, and a few higher-lambda extensions |
| sweep variant selection   |             `VARIANT=A` or `VARIANT=B` | same script can run either reconstruction variant                      |
| GPU                       |                        `gpu:4g.40gb:1` | required by the `stud` QoS                                             |
| CPUs / RAM                |                            `8` / `32G` | stable for long runs                                                   |
| wall time                 |      `12:00:00` main, `04:00:00` pilot | enough for typical runs                                                |
| node exclusion            |                    `--exclude=gnode04` | protects against known-bad node behavior                               |
| submission style          |                       wrapped `sbatch` | avoids spool-path failures                                             |
| monitoring                |                           `err/` first | where progress actually appears                                        |

---


## 8. Troubleshooting

### `slurm/setup_env_ready.sh: No such file or directory`

You are probably in `~`, not in the repo.

Fix:

```bash
cd /mnt/beegfsstudents/home/<USER_ID>/superclip-recon
bash slurm/setup_env_ready.sh --step data
```

### `QOSMaxSubmitJobPerUserLimit`

You submitted too many jobs.

Fix: wait for a running job to finish, then resubmit.

### `PD (QOSMaxJobsPerUserLimit)` or `PD (Priority)` or `PD (Resources)`

These are normal queue states, not code errors.

### `/var/spool/slurmd/jobNNNN/common.sh: No such file or directory`

You submitted a script directly with `sbatch slurm/<script>.sh`.

Use wrapped submission instead.

### `CondaToSNonInteractiveError`

Accept the Anaconda terms once on the login node:

```bash
module load miniconda3
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
```

### `ModuleNotFoundError` / `matplotlib is required`

You are not inside the `superclip` environment.

```bash
module load miniconda3
eval "$(conda shell.bash hook)"
conda activate superclip
```

### Sweep produced Variant A-style filenames during a Variant B run

Your sweep naming is probably missing the variant tag.

Fix: make sure the sweep writes run names and summary files that include `varA` / `varB` or `variant_A` / `variant_B`.

### Variant B sweep fails because `phrases.json` is missing

This should not block normal use.

Variant B runs through the same shared launcher as standard experiments. If `phrases.json` exists, it may be passed through; if it does not, the run should still be allowed to proceed under the current project setup.

### Job fails with exit `120:0` and no traceback

First check the node:

```bash
sacct -j <JOBID> -o JobID,JobName,State,Elapsed,ExitCode,NodeList
```

If the job ran on `gnode04`, cancel/resubmit with `--exclude=gnode04`.

If it ran elsewhere, also inspect memory and quota.

### Job fails with exit `0:53` and `Elapsed=00:00:00`

Usually a startup/prolog failure, often caused by quota or wrapper problems.

Check storage first.

### No progress in `out/`

Look in `err/` instead.

### Missing COCO or vocab

From the repo root:

```bash
bash slurm/setup_env_ready.sh --step data
bash slurm/setup_env_ready.sh --step vocab
bash slurm/setup_env_ready.sh --step cache
```

### `phrases.json` missing in preflight

This is acceptable. Variant B uses inline phrase extraction; `phrases.json` is optional.

---

## 9. Recommended clean rerun workflow

When you want to restart from a clean state after code changes:

### Step 1 — On HPC: clean outputs only

```bash
ssh <USER_ID>@slogin.hpc.unibocconi.it
cd /mnt/beegfsstudents/home/<USER_ID>/superclip-recon

rm -rf checkpoints results out err logs
find . -type d \( -name '__pycache__' -o -name '.pytest_cache' -o -name '.mypy_cache' \) -prune -exec rm -rf {} +
mkdir -p checkpoints results out err logs

lquota
du -sh checkpoints results out err ~/.cache
```

### Step 2 — On your Mac: resync code

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

### Step 3 — On HPC: verify repo/data/artifacts

```bash
ssh <USER_ID>@slogin.hpc.unibocconi.it

module load miniconda3
eval "$(conda shell.bash hook)"
conda activate superclip

cd /mnt/beegfsstudents/home/<USER_ID>/superclip-recon

pwd
ls slurm | sort
ls -l slurm/common.sh slurm/run_preflight.sh slurm/run_gpu_smoke.sh slurm/run_one_experiment.sh slurm/run_lambda_sweep.sh
test -d data/coco/train2017 && echo "train2017 OK"
test -d data/coco/val2017 && echo "val2017 OK"
test -f data/coco/annotations/captions_train2017.json && echo "captions_train2017 OK"
test -f data/coco/annotations/captions_val2017.json && echo "captions_val2017 OK"
test -f vocab.json && echo "vocab.json OK"
```

### Step 4 — Run Gate 0 / Gate 1 / Gate 2

Only then move on to baseline or sweep.