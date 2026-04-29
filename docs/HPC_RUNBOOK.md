# Bocconi HPC Runbook

Operational guide for running SuperCLIP-Recon on the Bocconi HPC cluster.

This runbook is based on the actual execution issues encountered during the project:

- queue and QoS limits
- quota exhaustion
- stuck runs during evaluation
- partial result folders
- empty Winoground outputs
- accidental confusion between Mac and HPC paths
- stage success being confused with SLURM success alone

---

## 1. Ground rules

- Do not run training or evaluation on the login node
- Use SLURM for any meaningful compute
- Prefer one **sequential** launcher at a time
- Exclude known-problem nodes if your scripts already do so
- Validate outputs before launching the next stage

If in doubt, choose the simpler launcher and the smaller run first.

---

## 2. Before every long run

Run all of these first:

```bash
squeue -u <USER_ID>
lquota
du -sh checkpoints results logs ~/.cache out err 2>/dev/null
```

Check:

- queue is not already full of your jobs
- home quota is not near the limit
- `checkpoints/` or `~/.cache` are not already huge
- you know which output directory the job will write to
- you are in the repo root before using `tail`, `find`, or cleanup commands

If quota is tight, sync important JSONs locally before deleting anything.

---

## 3. Repo-root rule

Many commands in this project use repo-relative paths such as `err/`, `out/`, `results/`, and `checkpoints/`.

Always do this first on HPC:

```bash
cd /mnt/beegfsstudents/home/<USER_ID>/superclip-recon
pwd
```

If you are in `~` instead of the repo root, `tail err/...` and `find results ...` can silently point to the wrong place or fail entirely.

---

## 4. Job submission habits

### Prefer one stage at a time

Do **not** submit many independent batches at once just because multiple scripts exist.

The cluster can reject submissions with:

- `QOSMaxSubmitJobPerUserLimit`
- related job-count or QoS-limit errors

### Prefer sequential launchers

A single sequential SLURM script is safer than many separate small jobs when:

- queue limits are tight
- you need matched baseline and reconstruction pairs
- you care more about runability than throughput

### Inspect the script before submitting

Always print the file you are about to submit:

```bash
sed -n '1,240p' slurm/some_launcher.sh
```

This matters because pasted heredocs can become corrupted.

### Use wrapped submission if you are not the original project account

Some committed SLURM headers and example paths still reflect the original project account.

If you are not using that same account, prefer:

```bash
sbatch --wrap='cd /mnt/beegfsstudents/home/<USER_ID>/superclip-recon && bash slurm/some_script.sh'
```

or adapt account, email, and path values accordingly.

---

## 5. Standard verification commands

### Queue status

```bash
squeue -u <USER_ID>
```

### Accounting status

```bash
sacct -u <USER_ID> -P --format=JobID,JobName,State,Elapsed,ExitCode -S today
```

### Job details

```bash
scontrol show job <JOBID>
```

### Current output files

```bash
ls -lt out | head -20
ls -lt err | head -20
```

### Inspect logs

```bash
tail -120 out/<jobname>_<jobid>.out
tail -120 err/<jobname>_<jobid>.err
```

Important: if you use a wildcard like `tail -f err/somejob_*.err`, you may end up reading an old failed log instead of the current run. Prefer the exact `<jobid>` once SLURM assigns it.

---

## 6. What counts as success

A job is usable only if all of the following are true:

1. `sacct` shows `COMPLETED` and `0:0`
2. the expected JSON output file exists
3. the JSON contains real metric keys
4. the log ends with a save message for the expected output
5. there is no `Traceback`, `No space left`, `403`, `Killed`, or `[SKIP]`

Important:

- `COMPLETED` alone is **not enough**
- a job may finish and still write `{}` or incomplete outputs
- a sweep is not validated until both the per-run JSONs and the summary JSONL exist

---

## 7. Exact gate commands

These are the safest generic commands because they do not rely on editing committed SLURM headers.

Replace `<USER_ID>` with your account.

### 7.1 Preflight

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
  --time=00:30:00 \
  --gres=gpu:4g.40gb:1 \
  --cpus-per-task=4 \
  --mem=16G \
  --wrap='cd /mnt/beegfsstudents/home/<USER_ID>/superclip-recon && bash slurm/run_preflight.sh'
```

Expected artifact:

- `results/preflight/preflight_report.json`

### 7.2 GPU smoke

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

Expected artifact:

- `results/smoke/gpu_smoke_results.json`

### 7.3 Pilot baseline

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

Expected artifacts:

- `results/pilot_varA.json`
- `checkpoints/pilot_varA/epoch_1.pt`

### 7.4 Pilot reconstruction

```bash
sbatch \
  --job-name=superclip-pilot-b \
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
  --wrap='cd /mnt/beegfsstudents/home/<USER_ID>/superclip-recon && export RUN_NAME=pilot_varB VARIANT=B LAMBDA_RECON=0.1 MASK_RATIO=0.15 SEED=43 BATCH_SIZE=64 EPOCHS=1 EVAL_MAX_IMAGES=128 TRAIN_MODE=auto SAVE_STRATEGY=last_and_best KEEP_LAST_K=1 && bash slurm/run_one_experiment.sh'
```

Expected artifacts:

- `results/pilot_varB.json`
- `checkpoints/pilot_varB/epoch_1.pt`

Do not start larger sweeps until all four gates above are validated.

---

## 8. Sweep commands

### 8.1 Variant A lambda sweep

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

### 8.2 Variant B lambda sweep

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

Expected sweep outputs:

- per-run JSONs in `results/ablations/`
- checkpoints in `checkpoints/ablations/`
- summary JSONLs such as `results/ablations/lambda_sweep_variant_A_summary_s43_b128.jsonl`

---

## 9. Known failure signatures

These are the most useful real failure patterns to recognize quickly.

### `KeyError: 'train_mode'` during `sanity_check.py`

Treat this as a code-path mismatch in the sanity check. Do not launch larger jobs until GPU smoke passes.

### `MASK_RATIO must be set`

This means `run_one_experiment.sh` was called without all required exported variables. Re-submit with `MASK_RATIO` explicitly exported.

### `No such file or directory` for `err/...` or `out/...`

You are probably not in the repo root. `cd` into the repo and retry.

### `COMPLETED` but no usable JSON

Do not count the run. Check whether the expected JSON exists and whether it contains real metrics.

### `No space left` or quota unexpectedly full

Treat quota as the first suspect, especially if earlier runs worked and a later sweep failed.

### Winoground output is `{}`

Treat it as invalid until Hugging Face access is confirmed and the JSON contains the expected Winoground keys.

---

## 10. Quota management

This project can generate large checkpoints quickly.

Main space consumers:

- `checkpoints/`
- `data/coco/`
- `~/.cache/`
- sometimes `out/` and `err/`

### Check usage

```bash
lquota
du -sh checkpoints results logs ~/.cache out err 2>/dev/null
```

### Practical warning

A later sweep can fail because quota is exhausted even when the code is correct. If a run family suddenly fails after earlier gates passed, check `lquota` before debugging model code.

### Safe cleanup order

Only after syncing important results:

```bash
rm -rf checkpoints
rm -rf ~/.cache/*
rm -rf out/*
rm -rf err/*
```

If you are completely done and have synced everything:

- remove the repo directory if desired
- remove the Conda env if you no longer need it

---

## 11. Syncing results off the cluster

Before large cleanup, sync important outputs to your Mac.

Recommended repo sync without checkpoints or data:

```bash
rsync -av --progress \
  --exclude 'checkpoints/' \
  --exclude '**/checkpoints/' \
  --exclude 'data/' \
  --exclude '**/data/' \
  --exclude '.git/' \
  --exclude '__pycache__/' \
  --exclude '*.pyc' \
  bocconi-hpc:/mnt/beegfsstudents/home/<USER_ID>/superclip-recon/ \
  ~/superclip_hpc_backup/superclip-recon/
```

This keeps:

- scripts
- results
- logs
- docs
- code

and skips:

- checkpoints
- COCO data

---

## 12. Result folder discipline

Do not assume all important outputs are in one folder.

Different experiment families may write to different locations under `results/`, for example:

- `results/ablations/`
- `results/confirm6/`
- `results/compositional_round2/`
- `results/winoground/`
- `results/final_checks/`
- `results/final_confirm/`

Before packaging results, always run:

```bash
find results -type f | sort
```

This is the easiest way to catch “the runs were done, but the zip omitted the folder”.

---

## 13. Winoground on HPC

Winoground requires:

- `HF_TOKEN`
- approved dataset access

Even if the SLURM job finishes, the result may still be invalid if the dataset access check fails.

A valid Winoground JSON should contain:

- `winoground_text_score`
- `winoground_image_score`
- `winoground_group_score`
- `winoground_n`

If the file is `{}`, treat it as failed.

---

## 14. Mac vs HPC path warning

HPC paths such as:

```text
/mnt/beegfsstudents/home/<USER_ID>/superclip-recon
```

do not exist on your Mac.

If `cd` into that path fails on your Mac, every subsequent command will run in your current local directory. This can cause you to clear local caches while thinking you are cleaning the cluster.

Always confirm where you are before destructive commands:

```bash
pwd
```

---

## 15. Recovery pattern when something looks wrong

If a job seems wrong:

1. check the exact job name and command
2. inspect the matching `out/` and `err/`
3. check whether the expected JSON already exists
4. do **not** rerun everything blindly
5. rerun only the missing stage if possible

This is especially important for:

- ARO
- Winoground
- seed-specific confirmation runs
- sweeps that may have partially succeeded

---

## 16. Minimal after-run checklist

After every important job:

```bash
sacct -j <JOBID> -o JobID,JobName,State,Elapsed,ExitCode,NodeList
find results -type f | sort
grep -R "Traceback\|No space left\|403\|\[SKIP\]" -n out err
```

Then inspect the expected JSON file itself and sync the JSONs you care about.

---

## 17. Bottom line

Use the cluster conservatively:

- one stage at a time
- validate before advancing
- sync before deleting
- package all report-relevant result folders, not just one
- treat a run as reproducible only when you can point to the exact output JSON and show that it is non-empty and valid
