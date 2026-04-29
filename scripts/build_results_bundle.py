#!/usr/bin/env python3
"""Build report-ready figures and tables under results/.

This is a thin wrapper around the existing plotting / summary utilities in
analyze_results.py. It keeps the presentation artifacts in a dedicated layout:

- results/figures/
- results/tables/
- results/qualitative/
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analyze_results import (  # noqa: E402
    load_ablation_results,
    load_all_results,
    load_compositional_results,
    plot_compositional,
    plot_lambda_sweep,
    plot_loss_curves,
    plot_maskrate_sweep,
    plot_retrieval_comparison,
    print_summary_table,
    setup_style,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, default="./results")
    args = parser.parse_args()

    results_dir = os.path.abspath(args.results_dir)
    figures_dir = os.path.join(results_dir, "figures")
    tables_dir = os.path.join(results_dir, "tables")
    qualitative_dir = os.path.join(results_dir, "qualitative")

    os.makedirs(figures_dir, exist_ok=True)
    os.makedirs(tables_dir, exist_ok=True)
    os.makedirs(qualitative_dir, exist_ok=True)

    setup_style()

    print(f"Loading raw results from: {results_dir}")
    main_results = load_all_results(results_dir)
    ablation_results = load_ablation_results(results_dir)
    compositional_results = load_compositional_results(results_dir)

    print(f"  Main runs: {list(main_results.keys())}")
    print(f"  Ablations: {list(ablation_results.keys())}")
    print(f"  Compositional: {list(compositional_results.keys())}")

    print_summary_table(main_results, compositional_results, tables_dir)

    print("\nGenerating figures...")
    plot_retrieval_comparison(main_results, figures_dir)
    plot_loss_curves(main_results, figures_dir)
    plot_lambda_sweep(ablation_results, figures_dir)
    plot_maskrate_sweep(ablation_results, figures_dir)
    plot_compositional(compositional_results, figures_dir)

    print("\nBundle complete.")
    print(f"  Figures:     {figures_dir}")
    print(f"  Tables:      {tables_dir}")
    print(f"  Qualitative: {qualitative_dir}")


if __name__ == "__main__":
    main()
