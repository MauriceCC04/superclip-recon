# Reproducibility Guide

This document explains how to reproduce the experiments described in the report from this repository.

The repository already contains the exact SLURM launchers used during the project. The goal of this guide is to make the execution order explicit and to point to the expected output locations.

---

## 1. Prerequisites

Before launching experiments, make sure the following are available:

- Python dependencies installed from `requirements.txt`
- COCO 2017 images and captions under `data/coco/`
- `vocab.json` present in the repo root
- on Bocconi HPC: the `superclip` environment and the setup in `docs/HPC_RUNBOOK.md`

If `vocab.json` is missing:

```bash
python build_vocab.py --coco_root ./data/coco --top_k 1000 --output vocab.json
```

---

## 2. Recommended execution order

The safest way to reproduce the report is to run one stage at a time.

A helper script is provided for this:

```bash
bash scripts/reproduce_report.sh plan
```

That prints the recommended order below.

### Stage A — main study

Runs the three main report configurations:

- baseline (`lambda_recon=0`, Variant A)
- Variant A reconstruction
- Variant B reconstruction

Command:

```bash
bash scripts/reproduce_report.sh main
```

Primary scripts used:

- `slurm/submit_main_experiments.sh`
- `slurm/run_one_experiment.sh`

Expected raw outputs:

- `results/baseline.json`
- `results/variant_a.json`
- `results/variant_b.json`
- `checkpoints/baseline/`
- `checkpoints/variant_a/`
- `checkpoints/variant_b/`

### Stage B — matched confirmation runs

Reproduces the strict same-seed confirmation runs used to strengthen the main comparison.

Commands:

```bash
bash scripts/reproduce_report.sh confirm6
bash scripts/reproduce_report.sh confirm_more
```

Primary scripts used:

- `slurm/run_confirm6_sequential.sh`
- `slurm/run_confirm_more_sequential.sh`

Expected raw outputs:

- `results/confirm6/`
- `results/confirm8/`
- matching checkpoint folders under `checkpoints/`

### Stage C — targeted ablations / extra checks

Reproduces the report-side follow-up checks:

- mask-rate ablation
- extra Variant B seed check

Command:

```bash
bash scripts/reproduce_report.sh ablations
```

Primary scripts used:

- `slurm/run_maskrate_l1_seed102.sh`
- `slurm/run_variantB_seed104_check.sh`

Expected raw outputs:

- `results/final_checks/`
- matching checkpoint folders under `checkpoints/final_checks/`

### Stage D — compositional evaluation

Reproduces the ARO-based compositional probes used as supporting evidence.

Command:

```bash
bash scripts/reproduce_report.sh compositional_core
```

Primary scripts used:

- `slurm/run_compositional_confirm_pair.sh`
- `slurm/run_compositional_round2.sh`

Expected raw outputs:

- `results/compositional/`
- `results/compositional_round2/`

### Stage E — optional Winoground / extra compositional runs

These require `HF_TOKEN` and are supporting, not mandatory, report reproduction stages.

Command:

```bash
bash scripts/reproduce_report.sh compositional_plus
```

Primary scripts used:

- `slurm/run_compositional_seed104.sh`
- `slurm/run_winoground_bestpair.sh`
- `slurm/run_compositional_more.sh`

Expected raw outputs:

- `results/final_confirm/`
- `results/winoground/`
- `results/compositional_round3/`

### Stage F — report artifacts

Once raw JSON results are present, generate the report-ready tables and plots:

```bash
bash scripts/reproduce_report.sh analyze
```

This runs:

```bash
python scripts/build_results_bundle.py --results_dir ./results
```

Expected derived outputs:

- `results/figures/`
- `results/tables/`
- `results/qualitative/`

---

## 3. What is already separated in the repository

The repository keeps the main project concerns in separate modules:

- **Data processing**
  - `build_vocab.py`
  - `dataset.py`
- **Model definition**
  - `model.py`
- **Training**
  - `train.py`
  - `losses.py`
  - `slurm/run_one_experiment.sh`
- **Evaluation**
  - `evaluate.py`
  - `eval_compositional.py`
  - `analyze_results.py`

This means the code path for data preparation, model construction, training, and evaluation can be inspected independently.

---

## 4. Results convention

Raw experiment outputs are stored close to the corresponding experiment family, for example:

- `results/confirm6/`
- `results/final_checks/`
- `results/compositional_round2/`

Report-ready artifacts are stored in the dedicated presentation layer:

- `results/figures/`
- `results/tables/`
- `results/qualitative/`

This keeps the reproducibility artifacts separate from the raw run logs and per-run JSON files.

---

## 5. Notes

- On Bocconi HPC, prefer reproducing one stage at a time instead of submitting everything at once.
- Some optional Winoground steps require `HF_TOKEN`.
- For cluster-specific recovery steps, cache handling, and troubleshooting, always defer to `docs/HPC_RUNBOOK.md`.
