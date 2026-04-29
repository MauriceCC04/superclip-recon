# SuperCLIP-Recon

SuperCLIP-Recon is a CLIP-based image-text retrieval project built around the COCO Captions dataset.

The repository contains:

- a **baseline path** built around CLIP retrieval plus token classification
- a **reconstruction path** that adds caption reconstruction
- **Variant A** as the baseline anchor used throughout the project
- **Variant B** as a phrase-reconstruction extension

The project was run primarily on Bocconi HPC, and the repository reflects that workflow.

For anything about actually running, monitoring, recovering, or reproducing jobs on the cluster, use these two documents as the source of truth:

- `docs/HPC_RUNBOOK.md`
- `docs/INCIDENTS.md`

Those two docs were updated from the real HPC stderr logs and document the actual operational behavior seen during the project, including cache warm-up, SLURM issues, checkpoint handling, quota failures, partial sequential jobs, and benign warnings.

---

## What is in the repository

The code is already separated by role:

- **Data processing**
  - `build_vocab.py`
  - `dataset.py`
- **Model definition**
  - `model.py`
  - `config.py`
- **Training**
  - `train.py`
  - `losses.py`
  - `slurm/run_one_experiment.sh`
- **Evaluation**
  - `evaluate.py`
  - `eval_compositional.py`
  - `analyze_results.py`
- **HPC orchestration**
  - `slurm/`
- **Operational documentation**
  - `docs/HPC_RUNBOOK.md`
  - `docs/INCIDENTS.md`

---

## Setup

### Python environment

For package installation, use the dependency versions in:

```bash
pip install -r requirements.txt
```

The repo expects at least the packages listed in `requirements.txt`, including `torch`, `open_clip_torch`, `transformers`, `pycocotools`, `datasets`, and `matplotlib`.

### Data layout

The code expects COCO under:

```text
data/coco/
├── annotations/
│   ├── captions_train2017.json
│   └── captions_val2017.json
├── train2017/
└── val2017/
```

If `vocab.json` is missing, build it with the script already present in the repo:

```bash
python build_vocab.py --coco_root ./data/coco --top_k 1000 --output vocab.json
```

### HPC environment

Do not rely on this README for the cluster setup details.

Use:

- `docs/HPC_RUNBOOK.md` for environment setup, conda activation, cache locations, job submission, monitoring, recovery, and reproducibility rules
- `docs/INCIDENTS.md` for the real failure modes encountered during the project and the decisions taken to handle them

---

## Main entry points

### Train one experiment

The main training entry point is:

```bash
python train.py \
  --coco_root ./data/coco \
  --vocab_path ./vocab.json \
  --run_name baseline \
  --train_mode superclip_baseline \
  --variant A \
  --lambda_recon 0.0 \
  --mask_ratio 0.15 \
  --epochs 10 \
  --batch_size 128 \
  --lr 1e-5 \
  --save_dir ./checkpoints/baseline \
  --results_file ./results/baseline.json
```

For real HPC runs, the project normally goes through:

```bash
bash slurm/submit_main_experiments.sh
```

or directly through:

```bash
sbatch slurm/run_one_experiment.sh
```

with the required environment variables described in `docs/HPC_RUNBOOK.md`.

### Evaluate COCO retrieval

```bash
python evaluate.py --checkpoint ./checkpoints/baseline/epoch_10.pt --coco_root ./data/coco
```

### Run compositional evaluation

```bash
python eval_compositional.py \
  --checkpoint ./checkpoints/baseline/epoch_10.pt \
  --benchmark aro \
  --output ./results/compositional_baseline.json
```

The evaluator supports:

- `--benchmark aro`
- `--benchmark winoground`
- `--benchmark all`

The real logs showed that `--benchmarks` is wrong; the correct flag is singular: `--benchmark`.

### Build summary plots and tables

```bash
python scripts/build_results_bundle.py --results_dir ./results
```

This script uses the existing `analyze_results.py` utilities and writes derived artifacts into:

- `results/figures/`
- `results/tables/`

It also creates `results/qualitative/` as the reserved place for qualitative report artifacts.

---

## Results layout

The repository uses `results/` for raw outputs and report artifacts.

Raw outputs from training / evaluation jobs appear in locations such as:

- `results/confirm6/`
- `results/final_checks/`
- `results/compositional/`
- `results/compositional_round2/`
- `results/winoground/`

Report-ready artifacts are kept separately in:

- `results/figures/`
- `results/tables/`
- `results/qualitative/`

For the detailed folder convention, see `results/README.md`.

---

## HPC scripts that are actually present in the repository

The following project scripts are present and correspond to runs reflected in the stderr logs:

### Main / training

- `slurm/submit_main_experiments.sh`
- `slurm/run_one_experiment.sh`
- `slurm/run_confirm6_sequential.sh`
- `slurm/run_maskrate_l1_seed102.sh`
- `slurm/run_variantB_seed104_check.sh`

### Compositional / evaluation

- `slurm/run_compositional_confirm_pair.sh`
- `slurm/run_compositional_round2.sh`
- `slurm/run_winoground_bestpair.sh`

### Analysis / reporting

- `scripts/build_results_bundle.py`
- `analyze_results.py`

This README intentionally does **not** claim more than that.

---

## What the stderr logs showed in practice

The attached HPC error logs and the runbook/incidents docs reinforce a few practical rules:

- monitor training progress in `err/` first, because `tqdm` and warnings often write there
- for sequential wrappers, also check `out/` and `logs/`
- the first ARO or Winoground evaluation may spend real time downloading datasets or building cache before steady-state example progress appears
- `fatal: not a git repository` at the end of a completed run is usually benign on the HPC copy, because `.git` is intentionally excluded from normal syncs
- `SIGNAL Terminated` on a long sequential batch does **not** necessarily mean the whole batch failed; some subruns may already be valid and worth keeping
- if `slurm/run_one_experiment.sh` is used directly, `RUN_NAME`, `VARIANT`, `LAMBDA_RECON`, and `MASK_RATIO` must be set

All of that is documented in more detail in:

- `docs/HPC_RUNBOOK.md`
- `docs/INCIDENTS.md`

---

## Reproducibility note

This repository contains the code and several real HPC launchers used during the project.

However, the authoritative instructions for reproducing the project workflow are in the HPC runbook, because that is where the real cluster constraints, job patterns, cache behavior, output paths, and recovery rules are documented.

If you want to reproduce the project from the repository:

1. set up the data and `vocab.json`
2. follow `docs/HPC_RUNBOOK.md` for environment and submission
3. use the real scripts listed above
4. use `scripts/build_results_bundle.py` to generate tables and plots from the resulting JSON outputs

That is the current, grounded path supported by the repository as it exists now.
