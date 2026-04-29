# Reproducibility and Runability Guide

This document explains how to run the project in a way that is consistent with the report and with the actual Bocconi HPC constraints encountered during development.

This is a **runability-first** guide. Its purpose is to help a user:

- launch the stages in the correct order
- avoid common HPC mistakes
- verify whether a run really succeeded
- package the correct raw results for review

---

## 1. Scope of the report workflow

The report is built around three layers:

### Baseline anchor
- SuperCLIP-style token classification
- COCO retrieval as the primary metric

### Main extension
- Variant A reconstruction loss
- matched baseline vs reconA comparisons with the same seed

### Follow-up checks
- ARO
- Winoground
- mask-rate ablation
- limited Variant B confirmation

The intended interpretation is:

- **baseline first**
- then **reconA comparison**
- then **supporting compositional/follow-up checks**

Do not treat Variant B as the main baseline anchor.

---

## 2. External prerequisites

This repository is not fully self-contained.

You must provide:

- COCO under `./data/coco`
- `vocab.json`
- a working environment with the packages in `requirements.txt`
- access to the Bocconi HPC resources if using the SLURM launchers
- `HF_TOKEN` for Winoground only

If you are on Bocconi HPC, also read:

- `docs/HPC_RUNBOOK.md`
- `INCIDENTS.md`

---

## 3. Start small: required execution order

Do not begin with a large staged run.

Recommended order:

1. **preflight**
2. **GPU smoke**
3. **one clean baseline/recon sanity pair**
4. **main retrieval runs**
5. **ablations**
6. **ARO / compositional follow-up**
7. **Winoground**
8. **final packaging**

If you use the staged helper, inspect it first and then run one stage at a time.

Example:

```bash
bash scripts/reproduce_report.sh --help
```

Typical report-oriented stages include:

- `preflight`
- `smoke`
- `main`
- `ablations`
- `compositional_core`
- `compositional_plus`
- `final_confirm`

Important: a “stage” is a logical unit, not necessarily one SLURM job. Some stages may still submit multiple jobs internally. On Bocconi HPC, do not advance to the next stage until the current one has been validated.

---

## 4. Required verification after every stage

A stage is **not** considered successful until all of the following are true.

### 4.1 Check SLURM completion

```bash
sacct -u <USER> -P --format=JobID,JobName,State,Elapsed,ExitCode -S today
```

You want:
- `COMPLETED`
- `0:0`

### 4.2 Check expected result files exist

```bash
find results -type f | sort
```

### 4.3 Open the JSONs and confirm they are non-empty

A job can finish and still produce `{}` or partial outputs.

Use:

```bash
python - <<'PY'
import json, sys
for f in sys.argv[1:]:
    with open(f) as fh:
        d = json.load(fh)
    print(f, "EMPTY" if not d else sorted(d.keys()))
PY results/some_file.json
```

### 4.4 Check `out/` and `err/`

Search for real failures:

```bash
grep -R "Traceback\|No space left\|403\|Killed\|\[SKIP\]" -n out err
```

### 4.5 Sync important result JSONs locally

Do this **before** deleting checkpoints or clearing quota-heavy directories.

---

## 5. Expected metric keys

Use these as your first-pass validation standard.

### Retrieval result JSONs

Expected patterns include fields such as:
- `best_retrieval`
- `best_retrieval_score`
- `final_retrieval`
- `history`

and retrieval metrics such as:
- `i2t_r1`, `i2t_r5`, `i2t_r10`
- `t2i_r1`, `t2i_r5`, `t2i_r10`

### ARO JSONs

Expected keys:
- `aro_vg_attribution_accuracy`
- `aro_vg_attribution_n`
- `aro_vg_relation_accuracy`
- `aro_vg_relation_n`

### Winoground JSONs

Expected keys:
- `winoground_text_score`
- `winoground_image_score`
- `winoground_group_score`
- `winoground_n`

If a Winoground JSON is `{}`, do **not** count it as a valid run.

---

## 6. Winoground caveat

Winoground is a gated dataset.

You need:
- `HF_TOKEN`
- approved access for that Hugging Face account

A Winoground job may finish even when access is missing, but produce empty or skipped outputs.

Before counting a Winoground result as valid, confirm that the JSON includes:

- `winoground_text_score`
- `winoground_image_score`
- `winoground_group_score`
- `winoground_n`

---

## 7. Report evidence map

The report evidence is spread across multiple result folders. Do **not** package only one subfolder unless you are certain it contains all report-required outputs.

Typical report-relevant folders from this project history are:

- `results/confirm6/`
  - matched seed retrieval confirmations
- `results/compositional_round2/`
  - additional retrieval + ARO follow-up
- `results/winoground/`
  - Winoground outputs
- `results/final_checks/`
  - mask-rate and limited Variant B follow-up checks
- `results/final_confirm/`
  - later confirmation family, including extra seed checks

Before creating a zip, always run:

```bash
find results -type f | sort
```

and make sure the archive contains every report-relevant subfolder you intend to claim.

---

## 8. Common HPC-safe workflow

### Before launching a long job

```bash
squeue -u <USER>
lquota
du -sh checkpoints results logs ~/.cache out err 2>/dev/null
```

Confirm:
- you are under job-count limits
- you have quota headroom
- you know exactly where JSONs will be written

### After a long run

```bash
sacct -u <USER> -P --format=JobID,JobName,State,Elapsed,ExitCode -S today
find results -type f | sort
```

Then inspect the specific JSONs and logs.

---

## 9. Syncing results locally

If you want to sync the repo back to your Mac but skip checkpoints and data:

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

## 10. Packaging checklist

Before you create the final zip:

1. Run:
   ```bash
   find results -type f | sort
   ```
2. Confirm all report-relevant folders are present
3. Confirm the important JSONs are non-empty
4. Confirm Winoground files are real, not `{}`

A very common failure mode is:
- the runs were completed on HPC
- but later result folders were omitted from the zip

Do not assume the archive is correct just because the HPC run succeeded.

---

## 11. Practical rules

- Run one stage at a time
- Validate outputs before advancing
- Sync JSONs before deleting checkpoints
- Never rely on SLURM status alone
- Never rely on folder names alone
- Always inspect the actual JSON contents

---

## 12. Bottom line

A stage is only “done” when:

- the SLURM job finished cleanly
- the expected JSON files exist
- those JSON files contain real metrics
- the results have been safely copied into your local evidence bundle

If something behaves strangely, go next to:

- `docs/HPC_RUNBOOK.md`
- `INCIDENTS.md`
