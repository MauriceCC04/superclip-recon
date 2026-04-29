# SuperCLIP-Recon

Course project repository for reproducing a **SuperCLIP-style token-classification baseline** and evaluating a lightweight **caption reconstruction auxiliary loss** on top of it.

## Project scope

This repository is organized around three distinct layers:

1. **Baseline reproduction**  
   Reproduce the SuperCLIP-style token-classification setup on COCO.

2. **Reconstruction-loss extension**  
   Add a lightweight auxiliary reconstruction loss on image-derived features:
   - **Variant A**: masked token prediction
   - **Variant B**: short phrase reconstruction

3. **Report-oriented evaluation**  
   Run the retrieval, compositional, and follow-up checks used in the report.

The intended course-project framing is:

- **Dataset:** MS-COCO Captions
- **Backbone:** ViT-B/32 initialized from CLIP
- **Primary metric:** COCO image-text retrieval
- **Supporting probes:** ARO and Winoground
- **Main comparison:** baseline vs Variant A with matched seeds
- **Extension check:** Variant B at limited scope

## What “runnable” means here

This repository includes the code, launchers, and documentation needed to run the project, but it is **not fully self-contained**.

To run it successfully, you still need:

- COCO available at `./data/coco`
- a valid `vocab.json`
- a working Python/Conda environment for this repo
- an HPC environment compatible with the documented SLURM launchers
- `HF_TOKEN` only if you want to run **Winoground**

For the Bocconi HPC workflow used in this project, read these files in order:

1. `REPRODUCIBILITY.md`
2. `docs/HPC_RUNBOOK.md`
3. `INCIDENTS.md`

## What this repository is for

Use this repo if you want to:

- train the baseline and reconstruction variants
- reproduce the report-oriented result folders
- validate whether a run actually succeeded
- package the correct raw result files for review

## Repository layout

Typical important paths:

- `train.py` — main training entry point
- `eval_compositional.py` — ARO / Winoground evaluation
- `scripts/reproduce_report.sh` — staged helper for report reproduction
- `slurm/` — HPC launchers
- `results/` — raw result JSONs
- `logs/` — per-run logs
- `out/`, `err/` — SLURM stdout/stderr
- `docs/HPC_RUNBOOK.md` — Bocconi HPC operating guide
- `REPRODUCIBILITY.md` — stage order and validation guidance
- `INCIDENTS.md` — failure patterns and recovery steps

## Environment setup

Install the dependencies in your preferred environment.

Example:

```bash
pip install -r requirements.txt
```

If you use Conda on HPC, activate the environment described in the runbook before submitting jobs.

## Data expectations

The repository expects COCO under:

```text
./data/coco
```

You should have the standard images and caption annotations needed by the training and evaluation scripts.

This repository does **not** redistribute COCO.

## Recommended first steps

Do **not** start with a long training job.

Run in this order:

1. preflight checks
2. GPU smoke test
3. one clean baseline run
4. one clean reconstruction run
5. only then the larger staged experiments

See `REPRODUCIBILITY.md` for the recommended order.

## Very important HPC note

On Bocconi HPC, do **not** submit many independent jobs at once just because multiple launchers exist.

Use one stage at a time, verify outputs, and then move to the next stage.

A stage is not considered successful until:

- the SLURM job finished with `COMPLETED` and `0:0`
- the expected JSON files exist
- the JSON files contain real metrics, not empty objects
- the result files have been synced locally if they matter for the report

## Winoground note

Winoground is gated.

A valid `HF_TOKEN` may still be insufficient if the Hugging Face account does not have access to the dataset. In that case, a job may finish but produce empty or skipped outputs.

Always inspect the resulting JSON files.

## Report-oriented result folders

The report evidence is spread across multiple subfolders under `results/`.

Do **not** assume that one folder contains everything.

The most important report-related folders are documented in `REPRODUCIBILITY.md`.

## Minimal runability checklist

Before a long run:

- confirm queue status
- confirm quota headroom
- confirm the launcher script path and output directory
- confirm where the JSON will be written

After a long run:

- check `sacct`
- inspect the expected result JSON
- inspect `out/` and `err/`
- sync result files before deleting checkpoints

## Common pitfalls

- running an HPC path command on your Mac
- assuming `COMPLETED` means the JSON is usable
- forgetting that results may be spread across multiple `results/*` folders
- packaging only one subfolder into a zip
- deleting checkpoints before syncing the result JSONs
- running Winoground without confirmed HF access

## Documentation

- `REPRODUCIBILITY.md` — exact stage order, validation, result-folder map, packaging guidance
- `docs/HPC_RUNBOOK.md` — Bocconi HPC operational rules and cleanup guidance
- `INCIDENTS.md` — known failure modes and the next command to run when they happen

## Summary

This repository is designed to be **runnable and reviewable** for the course project, provided that the user follows the documented stage order and validation checks.

For actual execution, start with:

- `REPRODUCIBILITY.md`
- then `docs/HPC_RUNBOOK.md`
