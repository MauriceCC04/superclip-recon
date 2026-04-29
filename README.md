# SuperCLIP-Recon

Course project repository for reproducing a **SuperCLIP-style token-classification baseline** and evaluating a lightweight **caption reconstruction auxiliary loss** on top of it.

## Project scope

This repository is organized around three distinct layers:

1. **Baseline reproduction**  
   Reproduce a SuperCLIP-style token-classification setup on COCO.

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

You still need:

- COCO available at `./data/coco`
- a valid `vocab.json`
- a working Python or Conda environment for this repo
- an HPC environment compatible with the documented SLURM launchers if you use the HPC path
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
- `out/`, `err/` — SLURM stdout and stderr
- `docs/HPC_RUNBOOK.md` — Bocconi HPC operating guide
- `REPRODUCIBILITY.md` — stage order, validation, output map, packaging guidance
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

See `REPRODUCIBILITY.md` for the exact stage order and validation rules.

## Launcher-to-output map

This is the single most important runability detail in the repo.

Different launchers write to different output locations.

| Launcher | Typical use | Expected outputs |
| --- | --- | --- |
| `slurm/run_preflight.sh` | preflight gate | `results/preflight/preflight_report.json` |
| `slurm/run_gpu_smoke.sh` | short end-to-end GPU gate | `results/smoke/gpu_smoke_results.json`, `checkpoints/smoke/` |
| `slurm/run_one_experiment.sh` | one training run | by default `results/<RUN_NAME>.json` and `checkpoints/<RUN_NAME>/` unless overridden by exported `RESULTS_FILE` and `SAVE_DIR` |
| `slurm/run_lambda_sweep.sh` | sequential lambda sweep | `results/ablations/`, `checkpoints/ablations/`, and a summary JSONL under `results/ablations/` |
| staged / custom SLURM scripts in `slurm/` | report follow-up families | custom subfolders such as `results/confirm6/`, `results/compositional_round2/`, `results/final_checks/`, `results/final_confirm/`, `results/winoground/` |

If you cannot name the exact JSON file a job is supposed to produce, do **not** treat that job as reproducible yet.

## What counts as a successful stage

A stage is not considered successful until:

- the SLURM job finished with `COMPLETED` and `0:0`
- the expected JSON file exists in the expected folder
- the JSON contains real metrics, not `{}` or a partial schema
- `out/` and `err/` do not show a real failure such as `Traceback`, `No space left`, `403`, or `[SKIP]`
- the result files have been synced locally if they matter for the report

A run is not reproducible unless you can point to the exact JSON file it produced.

## Known-good gate artifacts

These are the minimum files you should expect before moving to larger experiments.

| Stage | Minimum artifact to verify |
| --- | --- |
| preflight | `results/preflight/preflight_report.json` |
| GPU smoke | `results/smoke/gpu_smoke_results.json` |
| pilot baseline | `results/pilot_varA.json` |
| pilot reconstruction | `results/pilot_varB.json` |

## Very important HPC note

On Bocconi HPC, do **not** submit many independent jobs at once just because multiple launchers exist.

Use one stage at a time, verify outputs, and then move to the next stage.

Also note:

- some committed SLURM headers contain the original project account details such as `3202029`
- if you are not running on that same account, prefer the wrapped `sbatch --wrap='cd ... && bash ...'` commands documented in `docs/HPC_RUNBOOK.md`
- always `cd` into the repo root before using `tail`, `find`, `grep`, or destructive cleanup commands

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
- confirm you are in the repo root, not on your Mac in the wrong folder

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
- tailing the wrong log because you are not in the repo root
- using `run_one_experiment.sh` without exporting all required variables

## Documentation

- `REPRODUCIBILITY.md` — exact stage order, validation rules, launcher-to-output map, packaging guidance
- `docs/HPC_RUNBOOK.md` — Bocconi HPC operating guide, exact gate commands, cleanup guidance, common failure signatures
- `INCIDENTS.md` — known failure modes and the next command to run when they happen

## Summary

This repository is designed to be **runnable and reviewable** for the course project, provided that the user follows the documented stage order and validation checks.

For actual execution, start with:

- `REPRODUCIBILITY.md`
- then `docs/HPC_RUNBOOK.md`
