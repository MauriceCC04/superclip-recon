from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import find_repo_root, get_main_pairs, main_output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot average runtime overhead for ReconA vs baseline.")
    parser.add_argument("--repo-root", type=str, default=".")
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()

    repo_root = find_repo_root(args.repo_root)
    out_path = Path(args.out) if args.out else main_output_dir(repo_root) / "runtime_overhead_main.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    pairs = get_main_pairs(repo_root)
    if not pairs:
        raise RuntimeError("No paired main baseline vs ReconA runs were found.")

    baseline_times = [float(b["wall_time_seconds"]) for b, _ in pairs]
    recon_times = [float(r["wall_time_seconds"]) for _, r in pairs]
    mean_b = sum(baseline_times) / len(baseline_times)
    mean_r = sum(recon_times) / len(recon_times)

    fig, ax = plt.subplots(figsize=(7, 5))
    x = [0, 1]
    ax.bar(x, [mean_b, mean_r], width=0.55)
    for idx, (b, r) in enumerate(zip(baseline_times, recon_times)):
        ax.plot([0, 1], [b, r], marker="o", linewidth=0.9, alpha=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels(["Baseline", "ReconA (lambda=1)"])
    ax.set_ylabel("Wall time (seconds)")
    ax.set_title("Runtime overhead of the reconstruction loss")
    ax.grid(True, axis="y", alpha=0.3)
    ax.text(0.5, max(mean_b, mean_r) + 10.0, f"Mean overhead = {mean_r - mean_b:+.1f}s", ha="center")

    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
