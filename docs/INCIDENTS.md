# Incidents and Recovery Guide

This file records the failure patterns that actually occurred during project execution and tells the user what to do next.

The goal is not just to describe incidents, but to make the next recovery step obvious.

---

## 1. Job is `COMPLETED`, but the result is unusable

### Symptom
- `sacct` shows `COMPLETED`
- but the JSON is missing, empty, or incomplete

### Likely causes
- evaluation was skipped
- Winoground access failed
- the script ended before writing the final JSON
- the job completed a wrapper script but not the intended inner output path

### What to run next

```bash
find results -type f | sort
tail -120 out/<job>.out
tail -120 err/<job>.err
```

Then open the JSON and confirm it contains real metric keys.

### Minimal fix
Rerun only the missing evaluation stage, not the entire training job.

---

## 2. `QOSMaxSubmitJobPerUserLimit` or related queue-limit rejection

### Symptom
- SLURM submission fails with:
  - `QOSMaxSubmitJobPerUserLimit`
  - similar job-count / QoS-limit errors

### Likely causes
- too many jobs already queued or running
- multiple independent launches submitted at once

### What to run next

```bash
squeue -u <USER>
sacct -u <USER> -P --format=JobID,JobName,State,Elapsed,ExitCode -S today
```

### Minimal fix
- cancel accidental duplicate pending jobs
- use one sequential launcher
- wait for queue headroom before submitting the next stage

---

## 3. Quota exhaustion / `No space left`

### Symptom
- run fails during checkpointing or evaluation
- quota is at or above the limit
- `checkpoints/` is huge

### Likely causes
- too many retained checkpoints
- cached datasets or Hugging Face downloads
- stale large folders not yet cleared

### What to run next

```bash
lquota
du -sh checkpoints results logs ~/.cache out err 2>/dev/null
```

### Minimal fix
After syncing important JSONs:
```bash
rm -rf checkpoints
rm -rf ~/.cache/*
rm -rf out/*
rm -rf err/*
```

Do not delete result JSONs you have not synced yet.

---

## 4. Winoground JSON is `{}`

### Symptom
- file exists, but contains only `{}`

### Likely causes
- `HF_TOKEN` missing
- HF account lacks Winoground access
- evaluation skipped after a 403 or gated-access failure

### What to run next

```bash
tail -120 out/<winoground_job>.out
tail -120 err/<winoground_job>.err
```

Check for:
- `403`
- `[SKIP]`
- gated dataset access messages

### Minimal fix
- confirm dataset access from the active environment
- set `HF_TOKEN`
- rerun **Winoground only**

Do not retrain the model just because Winoground failed.

---

## 5. A long job is still running far beyond expectation

### Symptom
- run time is much longer than a comparable prior run
- logs stop progressing or stall during retrieval/evaluation

### Likely causes
- stuck during evaluation
- storage pressure
- partial job corruption
- wrapper launched the wrong script

### What to run next

```bash
scontrol show job <JOBID>
tail -120 out/<job>.out
tail -120 err/<job>.err
find results -type f | grep '<seed_or_name>' | sort
```

### Minimal fix
If the logs are stale and no new result JSON is being produced:
- cancel the job
- keep any valid finished outputs
- rerun only the missing stage

---

## 6. Pasted launcher script became corrupted

### Symptom
- heredoc or pasted script contains broken fragments
- unexpected characters appear in the script
- wrong commands are embedded
- SLURM submission uses the wrong file or malformed content

### Likely causes
- terminal paste corruption
- interrupted heredoc
- command pasted into the wrong shell state

### What to run next

```bash
sed -n '1,240p' slurm/<script>.sh
```

### Minimal fix
- rewrite the file cleanly
- inspect it before submission
- do not submit until the printed file looks correct

---

## 7. Missing result files in the final zip

### Symptom
- runs were completed on HPC
- but the review zip is missing seed families or later folders

### Likely causes
- only one `results/*` folder was zipped
- later runs were stored under a different result subfolder
- local sync omitted some directories

### What to run next

```bash
find results -type f | sort
```

### Minimal fix
Include all report-relevant folders, especially:
- `results/confirm6/`
- `results/compositional_round2/`
- `results/winoground/`
- `results/final_checks/`
- `results/final_confirm/`

Do not assume the newest runs are in the same folder as earlier ones.

---

## 8. Mac cleanup command accidentally used an HPC path

### Symptom
- `cd /mnt/beegfsstudents/...` fails on Mac
- destructive commands then run in `~`

### Likely causes
- using an HPC path outside the cluster

### What to run next

```bash
pwd
ls -lah ~
```

### Minimal fix
Confirm whether the local project copy still exists.
In most cases, only local cache files were deleted, not the synced repo.

---

## 9. ARO/Winoground chained job failed after the first partial step

### Symptom
- one combined compositional job fails
- some outputs exist, others do not

### Likely causes
- first evaluation completed, later one failed
- one large chain lost all later steps because of a single failure

### What to run next

```bash
find results -type f | grep '<seed>' | sort
tail -120 out/<job>.out
tail -120 err/<job>.err
```

### Minimal fix
Rerun each missing evaluation as a separate tiny job:
- baseline ARO
- reconA ARO
- baseline Winoground
- reconA Winoground

This is safer than chaining everything in one script.

---

## 10. `git_commit` is `null` in result JSONs

### Symptom
- result JSON includes `git_commit: null`

### Likely causes
- `.git` was not present in the HPC copy
- the repo was synced without Git metadata

### What to run next
Confirm that the code state is still the intended one and note this in the run record.

### Minimal fix
Do not treat `git_commit: null` as a runtime failure by itself, but do not overclaim exact provenance either.

---

## 11. Safe decision rule

When something fails, ask:

1. Did the training JSON already get written?
2. Did the checkpoint already get written?
3. Did the compositional JSON already get written?
4. Is the missing piece just evaluation?

If yes, rerun only the missing evaluation.

If no, inspect logs before rerunning the larger stage.

---

## 12. Bottom line

The most common mistake is rerunning too much.

Preferred recovery strategy:

- inspect
- keep valid partial outputs
- rerun only the missing stage
- recheck the JSON contents
- sync locally before cleanup
