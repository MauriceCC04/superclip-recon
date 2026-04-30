from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import find_repo_root, main_output_dir, mean, scan_aro_results, scan_winoground_results


def plot_aro(repo_root: Path, out_path: Path) -> None:
    aro = scan_aro_results(repo_root)
    seeds = sorted(set(aro["baseline"]) & set(aro["recon"]))
    if not seeds:
        raise RuntimeError("No paired ARO results were found.")

    labels = ["Attribution", "Relation"]
    baseline_values = [
        mean([float(aro["baseline"][s]["aro_vg_attribution_accuracy"]) for s in seeds]),
        mean([float(aro["baseline"][s]["aro_vg_relation_accuracy"]) for s in seeds]),
    ]
    recon_values = [
        mean([float(aro["recon"][s]["aro_vg_attribution_accuracy"]) for s in seeds]),
        mean([float(aro["recon"][s]["aro_vg_relation_accuracy"]) for s in seeds]),
    ]

    x = range(len(labels))
    width = 0.36
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar([i - width / 2 for i in x], baseline_values, width=width, label="Baseline")
    ax.bar([i + width / 2 for i in x], recon_values, width=width, label="ReconA (lambda=1)")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.set_ylabel("Accuracy")
    ax.set_title("ARO summary across paired seeds")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()

    for i, (b, r) in enumerate(zip(baseline_values, recon_values)):
        ax.text(i, max(b, r) + 0.12, f"Delta {r - b:+.2f}", ha="center", va="bottom", fontsize=9)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"Saved {out_path}")


def plot_winoground(repo_root: Path, out_path: Path) -> None:
    wino = scan_winoground_results(repo_root)
    seeds = sorted(set(wino["baseline"]) & set(wino["recon"]))
    if not seeds:
        raise RuntimeError("No paired Winoground results were found.")

    labels = ["Text", "Image", "Group"]
    baseline_values = [
        mean([float(wino["baseline"][s]["winoground_text_score"]) for s in seeds]),
        mean([float(wino["baseline"][s]["winoground_image_score"]) for s in seeds]),
        mean([float(wino["baseline"][s]["winoground_group_score"]) for s in seeds]),
    ]
    recon_values = [
        mean([float(wino["recon"][s]["winoground_text_score"]) for s in seeds]),
        mean([float(wino["recon"][s]["winoground_image_score"]) for s in seeds]),
        mean([float(wino["recon"][s]["winoground_group_score"]) for s in seeds]),
    ]

    x = range(len(labels))
    width = 0.36
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar([i - width / 2 for i in x], baseline_values, width=width, label="Baseline")
    ax.bar([i + width / 2 for i in x], recon_values, width=width, label="ReconA (lambda=1)")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.set_ylabel("Score")
    ax.set_title("Winoground summary across paired seeds")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()

    for i, (b, r) in enumerate(zip(baseline_values, recon_values)):
        ax.text(i, max(b, r) + 0.10, f"Delta {r - b:+.2f}", ha="center", va="bottom", fontsize=9)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"Saved {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot ARO and Winoground summaries.")
    parser.add_argument("--repo-root", type=str, default=".")
    parser.add_argument("--out-dir", type=str, default=None)
    args = parser.parse_args()

    repo_root = find_repo_root(args.repo_root)
    out_dir = Path(args.out_dir) if args.out_dir else main_output_dir(repo_root)
    out_dir.mkdir(parents=True, exist_ok=True)

    plot_aro(repo_root, out_dir / "aro_summary.png")
    plot_winoground(repo_root, out_dir / "winoground_summary.png")


if __name__ == "__main__":
    main()
