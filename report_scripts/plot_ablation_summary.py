from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import (
    MAIN_BATCH_SIZE,
    MAIN_EPOCHS,
    find_repo_root,
    get_eval_retrieval,
    main_output_dir,
    retrieval_score_from_metrics,
    scan_retrieval_runs,
)


def plot_lambda_ablation(repo_root: Path, out_path: Path) -> None:
    runs = scan_retrieval_runs(repo_root)

    by_key = defaultdict(dict)
    for run in runs:
        if run.get("variant") != "A":
            continue
        if run.get("batch_size") != MAIN_BATCH_SIZE or run.get("epochs") != MAIN_EPOCHS:
            continue
        seed = run.get("seed")
        lam = run.get("lambda_recon")
        mask_ratio = run.get("mask_ratio")
        if seed is None or mask_ratio != 0.15:
            continue
        key = None
        if (not bool(run.get("recon_enabled"))) and run.get("effective_variant") == "baseline" and lam == 0.0:
            key = 0.0
        elif bool(run.get("recon_enabled")) and lam in (0.5, 1.0):
            key = float(lam)
        if key is not None:
            by_key[key][int(seed)] = retrieval_score_from_metrics(get_eval_retrieval(run))

    common_seeds = set(by_key[0.0]) & set(by_key[0.5]) & set(by_key[1.0])
    if not common_seeds:
        raise RuntimeError("No common seeds were found for the Variant A lambda ablation.")

    lambdas = [0.0, 0.5, 1.0]
    means = []
    for lam in lambdas:
        values = [by_key[lam][seed] for seed in sorted(common_seeds)]
        means.append(sum(values) / len(values))

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(lambdas, means, marker="o")
    ax.set_xlabel("Lambda recon")
    ax.set_ylabel("Mean retrieval score")
    ax.set_title(f"Variant A lambda ablation on shared seeds {sorted(common_seeds)}")
    ax.grid(True, alpha=0.3)
    for x, y in zip(lambdas, means):
        ax.text(x, y + 0.15, f"{y:.2f}", ha="center", va="bottom", fontsize=9)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"Saved {out_path}")


def plot_maskrate_ablation(repo_root: Path, out_path: Path) -> None:
    runs = scan_retrieval_runs(repo_root)
    seed = 102
    mask_to_score = {}
    for run in runs:
        if run.get("variant") != "A":
            continue
        if run.get("batch_size") != MAIN_BATCH_SIZE or run.get("epochs") != MAIN_EPOCHS:
            continue
        if run.get("seed") != seed:
            continue
        if (not bool(run.get("recon_enabled"))) or run.get("lambda_recon") != 1.0:
            continue
        mask = float(run.get("mask_ratio"))
        mask_to_score[mask] = retrieval_score_from_metrics(get_eval_retrieval(run))

    expected_masks = [0.15, 0.30, 0.50]
    missing = [m for m in expected_masks if m not in mask_to_score]
    if missing:
        raise RuntimeError(f"Missing mask-rate runs for seed {seed}: {missing}")

    x = expected_masks
    y = [mask_to_score[m] for m in x]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(x, y, marker="o")
    ax.set_xlabel("Mask ratio")
    ax.set_ylabel("Retrieval score")
    ax.set_title(f"Variant A mask-rate check at lambda=1, seed={seed}")
    ax.grid(True, alpha=0.3)
    for xi, yi in zip(x, y):
        ax.text(xi, yi + 0.15, f"{yi:.2f}", ha="center", va="bottom", fontsize=9)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"Saved {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot limited ablation summaries.")
    parser.add_argument("--repo-root", type=str, default=".")
    parser.add_argument("--out-dir", type=str, default=None)
    args = parser.parse_args()

    repo_root = find_repo_root(args.repo_root)
    out_dir = Path(args.out_dir) if args.out_dir else main_output_dir(repo_root)
    out_dir.mkdir(parents=True, exist_ok=True)

    plot_lambda_ablation(repo_root, out_dir / "lambda_ablation_varA.png")
    plot_maskrate_ablation(repo_root, out_dir / "maskrate_ablation_varA_seed102.png")


if __name__ == "__main__":
    main()
