# Report figure scripts

These scripts assume the repo has the same `results/` layout as the attached archive.

## Expected location inside the repo

Copy this whole folder into the repo, for example as:

```bash
scripts/report_figures/
```

Then run from the repo root.

## Generate every figure

```bash
python scripts/report_figures/make_all_report_figures.py --repo-root .
```

## Generated files

All figures are written to:

```bash
results/figures/generated/
```

The scripts generate:

- `paired_retrieval_main.png`
- `metric_deltas_main.png`
- `epoch_curves_main.png`
- `aro_summary.png`
- `winoground_summary.png`
- `lambda_ablation_varA.png`
- `maskrate_ablation_varA_seed102.png`
- `runtime_overhead_main.png`

## Individual commands

```bash
python scripts/report_figures/plot_paired_retrieval.py --repo-root .
python scripts/report_figures/plot_metric_deltas.py --repo-root .
python scripts/report_figures/plot_epoch_curves.py --repo-root .
python scripts/report_figures/plot_compositional_summary.py --repo-root .
python scripts/report_figures/plot_ablation_summary.py --repo-root .
python scripts/report_figures/plot_runtime_overhead.py --repo-root .
```

## What each figure supports

- `paired_retrieval_main.png`
  - Small paired-seed comparison for baseline vs ReconA lambda=1.
- `metric_deltas_main.png`
  - Shows which retrieval metrics improve and which do not.
- `epoch_curves_main.png`
  - Shows both methods train correctly and converge similarly.
- `aro_summary.png` / `winoground_summary.png`
  - Supports the compositional-semantics motivation.
- `lambda_ablation_varA.png`
  - Limited evidence that lambda=1 was better than lambda=0.5 in the available runs.
- `maskrate_ablation_varA_seed102.png`
  - Limited evidence that higher mask ratios did not help in the tested seed.
- `runtime_overhead_main.png`
  - Supports the claim that the auxiliary loss is lightweight in runtime.

## Assumptions built into the scripts

For the main comparison figures, the scripts look for paired runs with:

- Variant A
- batch size 128
- 10 epochs
- baseline (`effective_variant == "baseline"`, `lambda_recon == 0.0`, `mask_ratio == 0.15`)
- ReconA main run (`effective_variant == "reconA"`, `lambda_recon == 1.0`, `mask_ratio == 0.15`)

For ablations, the scripts use the files present in the attached result structure:

- Variant A lambda values 0.0, 0.5, 1.0
- Variant A mask-rates 0.15, 0.30, 0.50 for seed 102

If your naming changes but the JSON fields stay the same, the retrieval scripts should still work.
