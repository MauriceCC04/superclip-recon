# Results folder

This directory is the dedicated home for both:

1. **raw experiment outputs** produced by training / evaluation jobs
2. **report-ready artifacts** such as figures, tables, and qualitative notes

## Raw outputs

The SLURM launchers in `slurm/` write raw JSON or JSONL outputs into subfolders such as:

- `results/ablations/`
- `results/confirm6/`
- `results/confirm8/`
- `results/final_checks/`
- `results/final_confirm/`
- `results/compositional/`
- `results/compositional_round2/`
- `results/compositional_round3/`
- `results/winoground/`

These files are the reproducibility layer and should be kept separate from the report presentation layer.

## Report-ready artifacts

This repository reserves the following subfolders for presentation artifacts:

- `results/figures/` — plots for the report
- `results/tables/` — summary CSV tables
- `results/qualitative/` — qualitative examples, notes, and failure cases

To build the figures and tables from existing raw JSON results, run:

```bash
python scripts/build_results_bundle.py --results_dir ./results
```
