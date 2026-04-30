from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import find_repo_root, get_eval_retrieval, get_main_pairs, main_output_dir


METRICS = [
    ("i2t_r1", "i2t R@1"),
    ("t2i_r1", "t2i R@1"),
    ("i2t_r5", "i2t R@5"),
    ("t2i_r5", "t2i R@5"),
    ("i2t_r10", "i2t R@10"),
    ("t2i_r10", "t2i R@10"),
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot average metric deltas (ReconA - Baseline).")
    parser.add_argument("--repo-root", type=str, default=".")
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()

    repo_root = find_repo_root(args.repo_root)
    out_path = Path(args.out) if args.out else main_output_dir(repo_root) / "metric_deltas_main.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    pairs = get_main_pairs(repo_root)
    if not pairs:
        raise RuntimeError("No paired main baseline vs ReconA runs were found.")

    labels = []
    deltas = []
    for key, label in METRICS:
        metric_diffs = []
        for baseline, recon in pairs:
            b = float(get_eval_retrieval(baseline)[key])
            r = float(get_eval_retrieval(recon)[key])
            metric_diffs.append(r - b)
        labels.append(label)
        deltas.append(sum(metric_diffs) / len(metric_diffs))

    fig, ax = plt.subplots(figsize=(8, 5))
    y = list(range(len(labels)))
    ax.barh(y, deltas)
    ax.axvline(0.0, linewidth=1.0)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Average delta (ReconA - Baseline)")
    ax.set_title("Metric-level retrieval changes from the reconstruction loss")
    ax.grid(True, axis="x", alpha=0.3)

    for yi, delta in zip(y, deltas):
        offset = 0.01 if delta >= 0 else -0.01
        ha = "left" if delta >= 0 else "right"
        ax.text(delta + offset, yi, f"{delta:+.3f}", va="center", ha=ha, fontsize=9)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
