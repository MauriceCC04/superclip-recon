# Bocconi HPC Runbook

Operational guide for running SuperCLIP-Recon on the Bocconi HPC cluster.

This runbook is based on the actual execution issues encountered during the project:
- queue/QoS limits
- quota exhaustion
- stuck runs during evaluation
- partial result folders
- empty Winoground outputs
- accidental confusion between Mac and HPC paths

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
squeue -u <USER>
lquota
du -sh checkpoints results logs ~/.cache out err 2>/dev/null
```

Check:

- queue is not already full of your jobs
- home quota is not near the limit
- `checkpoints/` or `~/.cache` are not already huge
- you know which output directory the job will write to

If quota is tight, sync important JSONs locally before deleting anything.

---

## 3. Job submission habits

### Prefer one stage at a time

Do **not** submit many independent batches at once just because multiple scripts exist.

The cluster can reject submissions with:
- `QOSMaxSubmitJobPerUserLimit`
- related job-count or QoS-limit errors

### Prefer sequential launchers

A single sequential SLURM script is safer than many separate small jobs when:
- queue limits are tight
- you need matched baseline/recon pairs
- you care more about runability than throughput

### Inspect the script before submitting

Always print the file you are about to submit:

```bash
sed -n '1,240p' slurm/some_launcher.sh
```

This matters because pasted heredocs can become corrupted.

---

## 4. Standard verification commands

### Queue status

```bash
squeue -u <USER>
```

### Accounting status

```bash
sacct -u <USER> -P --format=JobID,JobName,State,Elapsed,ExitCode -S today
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

---

## 5. What counts as success

A job is usable only if all of the following are true:

1. `sacct` shows `COMPLETED` and `0:0`
2. the expected JSON output file exists
3. the JSON contains real metric keys
4. the log ends with `Saved to ...`
5. there is no `Traceback`, `No space left`, `403`, `Killed`, or `[SKIP]`

Important:
- `COMPLETED` alone is **not enough**
- a job may finish and still write `{}` or incomplete outputs

---

## 6. Quota management

This project can generate large checkpoints quickly.

Main space consumers:
- `checkpoints/`
- `data/coco/`
- `~/.cache/`
- sometimes `out/` / `err/`

### Check usage

```bash
lquota
du -sh checkpoints results logs ~/.cache out err 2>/dev/null
```

### Safe cleanup order

Only after syncing important results:

```bash
rm -rf checkpoints
rm -rf ~/.cache/*
rm -rf out/*
rm -rf err/*
```

If you are completely done and have synced everything:
- remove the repo directory
- remove the Conda env if you no longer need it

---

## 7. Syncing results off the cluster

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
  bocconi-hpc:/mnt/beegfsstudents/home/3202029/superclip-recon/ \
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

## 8. Result folder discipline

Do not assume all important outputs are in one folder.

Different experiment families may write to different locations under `results/`, for example:
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

## 9. Winoground on HPC

Winoground requires:
- `HF_TOKEN`
- approved dataset access

Even if the SLURM job finishes, the result may be invalid if the dataset access check fails.

A valid Winoground JSON should contain:
- `winoground_text_score`
- `winoground_image_score`
- `winoground_group_score`
- `winoground_n`

If the file is `{}`, treat it as failed.

---

## 10. Mac vs HPC path warning

HPC paths such as:

```text
/mnt/beegfsstudents/home/<USER>/superclip-recon
```

do not exist on your Mac.

If `cd` into that path fails on your Mac, every subsequent command will run in your current local directory. This can cause you to clear `~/.cache` locally while thinking you are cleaning the cluster.

Always confirm where you are before destructive commands:

```bash
pwd
```

---

## 11. Recovery pattern when something looks wrong

If a job seems wrong:

1. check the exact job name and command
2. inspect the matching `out/` and `err/`
3. check whether expected JSONs already exist
4. do **not** rerun everything blindly
5. rerun only the missing stage if possible

This is especially important for:
- ARO
- Winoground
- seed-specific confirmation runs

---

## 12. Minimal after-run checklist

After every important job:

```bash
sacct -j <JOBID> -o JobID,JobName,State,Elapsed,ExitCode,NodeList
find results -type f | sort
grep -R "Traceback\|No space left\|403\|\[SKIP\]" -n out err
```

Then sync the JSONs you care about.

---

## 13. Bottom line

Use the cluster conservatively:

- one stage at a time
- validate before advancing
- sync before deleting
- package all report-relevant result folders, not just one
