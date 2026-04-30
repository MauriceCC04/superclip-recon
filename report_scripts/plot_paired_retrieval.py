from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import find_repo_root, get_main_pairs, main_output_dir, retrieval_score_from_metrics, get_eval_retrieval


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot paired baseline vs ReconA retrieval scores.")
    parser.add_argument("--repo-root", type=str, default=".", help="Path inside the repo.")
    parser.add_argument("--out", type=str, default=None, help="Optional output PNG path.")
    args = parser.parse_args()

    repo_root = find_repo_root(args.repo_root)
    out_path = Path(args.out) if args.out else main_output_dir(repo_root) / "paired_retrieval_main.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    pairs = get_main_pairs(repo_root)
    if not pairs:
        raise RuntimeError("No paired main baseline vs ReconA runs were found.")

    baseline_scores = []
    recon_scores = []
    seeds = []
    for baseline, recon in pairs:
        seeds.append(int(baseline["seed"]))
        baseline_scores.append(retrieval_score_from_metrics(get_eval_retrieval(baseline)))
        recon_scores.append(retrieval_score_from_metrics(get_eval_retrieval(recon)))

    fig, ax = plt.subplots(figsize=(8, 5))
    x = [0, 1]
    for seed, b, r in zip(seeds, baseline_scores, recon_scores):
        ax.plot(x, [b, r], marker="o", linewidth=1.25)
        ax.text(-0.05, b, f"s{seed}", va="center", ha="right", fontsize=8)

    mean_b = sum(baseline_scores) / len(baseline_scores)
    mean_r = sum(recon_scores) / len(recon_scores)
    ax.scatter([0, 1], [mean_b, mean_r], s=110, marker="D", zorder=4)
    ax.hlines(mean_b, -0.18, 0.18, linewidth=2.2)
    ax.hlines(mean_r, 0.82, 1.18, linewidth=2.2)

    ax.set_xticks(x)
    ax.set_xticklabels(["Baseline", "ReconA (lambda=1)"])
    ax.set_ylabel("Aggregate retrieval score")
    ax.set_title("Paired COCO retrieval comparison across shared seeds")
    ax.grid(True, axis="y", alpha=0.3)

    delta = mean_r - mean_b
    ax.text(
        0.5,
        max(max(baseline_scores), max(recon_scores)) + 1.0,
        f"Mean delta = {delta:+.3f}",
        ha="center",
        va="bottom",
    )

    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
