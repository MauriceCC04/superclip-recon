from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

SCRIPTS = [
    "plot_paired_retrieval.py",
    "plot_metric_deltas.py",
    "plot_epoch_curves.py",
    "plot_compositional_summary.py",
    "plot_ablation_summary.py",
    "plot_runtime_overhead.py",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate all report figures.")
    parser.add_argument("--repo-root", type=str, default=".")
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    for script_name in SCRIPTS:
        script_path = script_dir / script_name
        cmd = [sys.executable, str(script_path), "--repo-root", args.repo_root]
        print("Running:", " ".join(cmd))
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
