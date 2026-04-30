from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import epoch_mean_curve, find_repo_root, get_main_pairs, main_output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot mean retrieval score vs epoch for baseline and ReconA.")
    parser.add_argument("--repo-root", type=str, default=".")
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()

    repo_root = find_repo_root(args.repo_root)
    out_path = Path(args.out) if args.out else main_output_dir(repo_root) / "epoch_curves_main.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    pairs = get_main_pairs(repo_root)
    if not pairs:
        raise RuntimeError("No paired main baseline vs ReconA runs were found.")

    baseline_runs = [b for b, _ in pairs]
    recon_runs = [r for _, r in pairs]

    epochs_b, curve_b = epoch_mean_curve(baseline_runs)
    epochs_r, curve_r = epoch_mean_curve(recon_runs)
    if epochs_b != epochs_r:
        raise RuntimeError("Baseline and ReconA epoch grids do not match.")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs_b, curve_b, marker="o", label="Baseline")
    ax.plot(epochs_r, curve_r, marker="o", label="ReconA (lambda=1)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Mean retrieval score")
    ax.set_title("Training curves: mean retrieval score across paired seeds")
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
