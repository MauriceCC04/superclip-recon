# SuperCLIP-Recon

SuperCLIP-Recon is a retrieval-first extension of CLIP built around the COCO Captions dataset.

The project keeps the standard CLIP image-text contrastive objective as the main signal, then adds:

- a **SuperCLIP-style token classification head** as the baseline extension
- an optional **caption reconstruction head** as the main experimental improvement

The repository is organized so that the main project components are already separated by responsibility:

- **Data processing**: `build_vocab.py`, `dataset.py`
- **Model definition**: `model.py`
- **Training**: `train.py`, `losses.py`, `slurm/run_one_experiment.sh`
- **Evaluation**: `evaluate.py`, `eval_compositional.py`, `analyze_results.py`
- **HPC / orchestration**: `slurm/`, `docs/HPC_RUNBOOK.md`, `docs/INCIDENTS.md`

For the Bocconi HPC workflow used during the project, see `docs/HPC_RUNBOOK.md`.

---

## Repository layout

```text
.
├── README.md
├── REPRODUCIBILITY.md
├── build_vocab.py              # build token-class vocab from COCO captions
├── dataset.py                  # COCO caption dataset loader
├── model.py                    # SuperCLIP-Recon model definition
├── losses.py                   # training objectives and masking helpers
├── train.py                    # main training entry point
├── evaluate.py                 # COCO retrieval evaluation
├── eval_compositional.py       # ARO / Winoground evaluation
├── analyze_results.py          # plotting / summary helper
├── scripts/
│   ├── build_results_bundle.py # export tables + figures into results/
│   └── reproduce_report.sh     # staged reproduction helper
├── slurm/                      # HPC launchers used in the project report
├── docs/
│   ├── HPC_RUNBOOK.md
│   └── INCIDENTS.md
└── results/
    ├── README.md
    ├── figures/
    ├── tables/
    └── qualitative/
```

---

## Setup

### Local Python setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### Data layout

The code expects COCO 2017 captions and images in the following layout:

```text
data/coco/
├── annotations/
│   ├── captions_train2017.json
│   └── captions_val2017.json
├── train2017/
└── val2017/
```

If `vocab.json` is missing, build it with:

```bash
python build_vocab.py --coco_root ./data/coco --top_k 1000 --output vocab.json
```

### HPC setup

For the exact Bocconi HPC environment, conda activation, cache setup, and SLURM usage, follow:

- `docs/HPC_RUNBOOK.md`

---

## Quick start

### Train one run

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

### Evaluate a checkpoint on COCO retrieval

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

### Export figures and tables into `results/`

```bash
python scripts/build_results_bundle.py --results_dir ./results
```

This writes derived artifacts to:

- `results/figures/`
- `results/tables/`
- `results/qualitative/`

---

## Reproducing the report experiments

The repository includes a staged helper for the experiments used in the report:

```bash
bash scripts/reproduce_report.sh plan
```

This prints the recommended execution order.

To run a specific stage on HPC:

```bash
bash scripts/reproduce_report.sh main
bash scripts/reproduce_report.sh confirm6
bash scripts/reproduce_report.sh ablations
bash scripts/reproduce_report.sh compositional_core
```

For the full stage-by-stage description, expected outputs, and notes on optional Winoground runs, see:

- `REPRODUCIBILITY.md`

---

## Results folder convention

Raw experiment outputs are stored as JSON / JSONL files under `results/` and its experiment subfolders.

Derived report artifacts are stored in a dedicated structure:

- `results/figures/` — plots used in the report
- `results/tables/` — summary CSV tables
- `results/qualitative/` — selected qualitative examples / notes

This keeps raw experiment outputs separate from report-ready artifacts.
