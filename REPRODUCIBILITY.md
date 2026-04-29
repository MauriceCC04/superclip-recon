# Reproducibility and Runability Guide

This document explains how to run the project in a way that is consistent with the report and with the actual Bocconi HPC constraints encountered during development.

This is a **runability-first** guide. Its purpose is to help a user:

- launch the stages in the correct order
- avoid common HPC mistakes
- verify whether a run really succeeded
- package the correct raw results for review

---

## 1. Scope of the report workflow

The report is built around three layers.

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
- then **supporting compositional and follow-up checks**

Do not treat Variant B as the main baseline anchor.

---

## 2. External prerequisites

This repository is not fully self-contained.

You must provide:

- COCO under `./data/coco`
- `vocab.json`
- a working environment with the packages in `requirements.txt`
- access to Bocconi HPC resources if using the SLURM launchers
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
3. **one clean baseline and reconstruction sanity pair**
4. **main retrieval runs**
5. **ablations**
6. **ARO and compositional follow-up**
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

Important: a “stage” is a logical unit, not necessarily one SLURM job. Some stages may still submit multiple jobs internally. On Bocconi HPC, do not advance to the next stage until the current stage has been validated.

---

## 4. Exact gate order for a clean rerun

If your goal is to prove the repo is runnable before doing larger experiments, use this minimum path.

### 4.1 Preflight

Expected artifact:

- `results/preflight/preflight_report.json`

### 4.2 GPU smoke

Expected artifacts:

- `results/smoke/gpu_smoke_results.json`
- `checkpoints/smoke/`

### 4.3 Pilot baseline

Expected artifacts:

- `results/pilot_varA.json`
- `checkpoints/pilot_varA/`

### 4.4 Pilot reconstruction

Expected artifacts:

- `results/pilot_varB.json`
- `checkpoints/pilot_varB/`

Only after these four gates pass should you move to main report-oriented runs.

---

## 5. Launcher-to-output map

Different launchers write to different places. This matters for both validation and packaging.

| Launcher | Typical use | Output location |
| --- | --- | --- |
| `slurm/run_preflight.sh` | preflight gate | `results/preflight/` |
| `slurm/run_gpu_smoke.sh` | short end-to-end GPU gate | `results/smoke/`, `checkpoints/smoke/` |
| `slurm/run_one_experiment.sh` | one training run | by default `results/<RUN_NAME>.json` and `checkpoints/<RUN_NAME>/` unless `RESULTS_FILE` and `SAVE_DIR` are exported |
| `slurm/run_lambda_sweep.sh` | sequential lambda sweep | `results/ablations/`, `checkpoints/ablations/`, `results/ablations/lambda_sweep_variant_*.jsonl` |
| custom report scripts in `slurm/` | confirm / compositional / final runs | subfolders such as `results/confirm6/`, `results/compositional_round2/`, `results/final_checks/`, `results/final_confirm/`, `results/winoground/` |

Two practical rules follow from this:

1. never assume all outputs live in one folder
2. never assume a successful training run will be picked up automatically by analysis code unless it was written to the location that analysis code actually reads

---

## 6. Required verification after every stage

A stage is **not** considered successful until all of the following are true.

### 6.1 Check SLURM completion

```bash
sacct -u <USER_ID> -P --format=JobID,JobName,State,Elapsed,ExitCode -S today
```

You want:

- `COMPLETED`
- `0:0`

### 6.2 Check the expected result files exist

```bash
find results -type f | sort
```

### 6.3 Open the JSONs and confirm they are non-empty

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

### 6.4 Check `out/` and `err/`

Search for real failures:

```bash
grep -R "Traceback\|No space left\|403\|Killed\|\[SKIP\]" -n out err
```

### 6.5 Sync important result JSONs locally

Do this **before** deleting checkpoints or clearing quota-heavy directories.

---

## 7. Stage-specific success signatures

These are the fastest practical checks.

### 7.1 Preflight passes if

- `sacct` shows `COMPLETED` and `0:0`
- `results/preflight/preflight_report.json` exists
- the JSON is non-empty

### 7.2 GPU smoke passes if

- `sacct` shows `COMPLETED` and `0:0`
- `results/smoke/gpu_smoke_results.json` exists
- the JSON contains retrieval keys such as `i2t_r1`, `t2i_r1`, `i2t_r5`, `t2i_r5`
- the log contains a line indicating that smoke results were saved

### 7.3 Pilot baseline passes if

- `results/pilot_varA.json` exists
- `checkpoints/pilot_varA/epoch_1.pt` exists
- the log ends with a result-save message for `pilot_varA`

### 7.4 Pilot reconstruction passes if

- `results/pilot_varB.json` exists
- `checkpoints/pilot_varB/epoch_1.pt` exists
- the log shows the reconstruction path ran
- for Variant B, the log may note that phrases are extracted inline per caption

---

## 8. Known-good pilot and sweep expectations

These are useful first-pass expectations for the repo as currently written.

- `run_one_experiment.sh` requires exported variables such as `RUN_NAME`, `VARIANT`, `LAMBDA_RECON`, and `MASK_RATIO`
- `run_lambda_sweep.sh` writes to `results/ablations/` and `checkpoints/ablations/`
- `run_lambda_sweep.sh` writes a summary JSONL named like `lambda_sweep_variant_A_summary_s<seed>_b<batch>.jsonl`
- a Variant A sweep should include the baseline anchor at `lambda=0`
- a Variant B sweep should be interpreted as extension analysis, not the baseline anchor

---

## 9. Expected metric keys

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

## 10. Winoground caveat

Winoground is a gated dataset.

You need:

- `HF_TOKEN`
- approved access for that Hugging Face account

A Winoground job may finish even when access is missing, but still produce empty or skipped outputs.

Before counting a Winoground result as valid, confirm that the JSON includes:

- `winoground_text_score`
- `winoground_image_score`
- `winoground_group_score`
- `winoground_n`

---

## 11. Report evidence map

The report evidence is spread across multiple result folders. Do **not** package only one subfolder unless you are certain it contains all report-required outputs.

Typical report-relevant folders from this project history are:

- `results/confirm6/`
  - matched-seed retrieval confirmations
- `results/compositional_round2/`
  - additional retrieval plus ARO follow-up
- `results/winoground/`
  - Winoground outputs
- `results/final_checks/`
  - mask-rate and limited Variant B follow-up checks
- `results/final_confirm/`
  - later confirmation family, including extra-seed checks

Before creating a zip, always run:

```bash
find results -type f | sort
```

and make sure the archive contains every report-relevant subfolder you intend to claim.

---

## 12. Important note about report-bundle scripts

The report-bundle and analysis scripts are useful, but they are **not a substitute for checking the raw result layout yourself**.

In the current repo:

- main training results are read non-recursively from the top level of `results/`
- ablations are read from `results/ablations/`
- compositional results are read from top-level `compositional_*.json`

Therefore:

- do not assume that every historical result subfolder under `results/` will automatically appear in a generated summary or figure
- before building report figures or tables, verify that the raw JSONs you care about are present in the locations the current analysis code actually reads
- always keep the raw JSON evidence bundle, even if you also generate summary figures or tables

---

## 13. Common HPC-safe workflow

### Before launching a long job

```bash
squeue -u <USER_ID>
lquota
du -sh checkpoints results logs ~/.cache out err 2>/dev/null
```

Confirm:

- you are under job-count limits
- you have quota headroom
- you know exactly where JSONs will be written
- you are in the repo root before using `tail`, `find`, or `rm`

### After a long run

```bash
sacct -u <USER_ID> -P --format=JobID,JobName,State,Elapsed,ExitCode -S today
find results -type f | sort
```

Then inspect the specific JSONs and logs.

---

## 14. Syncing results locally

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

## 15. Packaging checklist

Before you create the final zip:

1. Run:
   ```bash
   find results -type f | sort
   ```
2. Confirm all report-relevant folders are present
3. Confirm the important JSONs are non-empty
4. Confirm Winoground files are real, not `{}`
5. Confirm the raw JSON evidence you cite in the report actually made it into the archive

A very common failure mode is:

- the runs completed on HPC
- but later result folders were omitted from the zip

Do not assume the archive is correct just because the HPC run succeeded.

---

## 16. Practical rules

- run one stage at a time
- validate outputs before advancing
- sync JSONs before deleting checkpoints
- never rely on SLURM status alone
- never rely on folder names alone
- always inspect the actual JSON contents
- always know which launcher wrote which file

---

## 17. Bottom line

A stage is only “done” when:

- the SLURM job finished cleanly
- the expected JSON files exist
- those JSON files contain real metrics
- the results have been safely copied into your local evidence bundle

If something behaves strangely, go next to:

- `docs/HPC_RUNBOOK.md`
- `INCIDENTS.md`
