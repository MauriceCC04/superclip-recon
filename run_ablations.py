"""
Ablation sweep runner for SuperCLIP-Recon.

Runs a compact grid of experiments and collects results in one directory.
Designed to run inside a single SLURM job — experiments run sequentially.

Grid:
    1. Lambda sweep:        lambda in {0.0, 0.1, 0.5, 1.0}  (Variant A, mask_ratio=0.15)
    2. Masking rate sweep:  mask_ratio in {0.10, 0.15, 0.25} (Variant A, lambda=0.5)
    3. Variant comparison:  A vs B                            (lambda=0.5, mask_ratio=0.15)

Total: ~8 training runs (some overlap, deduplicated below).

Usage:
    python run_ablations.py --coco_root ./data/coco --vocab_path ./vocab.json \
                            --phrase_path ./phrases.json --results_dir ./results/ablations

    # Shorter runs for debugging:
    python run_ablations.py --epochs 2 --batch_size 64
"""

import os
import json
import subprocess
import argparse
from itertools import product


def build_experiment_grid():
    """
    Build the ablation grid. Returns list of dicts, one per experiment.
    Duplicates are removed by run_name.
    """
    experiments = {}

    # --- 1. Lambda sweep (Variant A, mask_ratio=0.15) ---
    for lam in [0.0, 0.1, 0.5, 1.0]:
        name = f"lambda_{lam:.1f}"
        experiments[name] = {
            "run_name": name,
            "variant": "A",
            "lambda_recon": lam,
            "mask_ratio": 0.15,
        }

    # --- 2. Masking rate sweep (Variant A, lambda=0.5) ---
    for mr in [0.10, 0.15, 0.25]:
        name = f"maskrate_{mr:.2f}"
        if name not in experiments:  # avoid duplicate with lambda sweep
            experiments[name] = {
                "run_name": name,
                "variant": "A",
                "lambda_recon": 0.5,
                "mask_ratio": mr,
            }

    # --- 3. Variant A vs B (lambda=0.5, mask_ratio=0.15) ---
    for var in ["A", "B"]:
        name = f"variant_{var}"
        if name not in experiments:
            experiments[name] = {
                "run_name": name,
                "variant": var,
                "lambda_recon": 0.5,
                "mask_ratio": 0.15,
            }

    return list(experiments.values())


def run_single_experiment(exp, base_args):
    """Run one training experiment as a subprocess."""
    cmd = [
        "python", "train.py",
        "--coco_root", base_args.coco_root,
        "--vocab_path", base_args.vocab_path,
        "--variant", exp["variant"],
        "--lambda_recon", str(exp["lambda_recon"]),
        "--mask_ratio", str(exp["mask_ratio"]),
        "--epochs", str(base_args.epochs),
        "--batch_size", str(base_args.batch_size),
        "--lr", str(base_args.lr),
        "--seed", str(base_args.seed),
        "--run_name", exp["run_name"],
        "--save_dir", os.path.join(base_args.results_dir, "checkpoints", exp["run_name"]),
        "--results_file", os.path.join(base_args.results_dir, f"{exp['run_name']}.json"),
    ]

    if exp["variant"] == "B" and base_args.phrase_path:
        cmd.extend(["--phrase_path", base_args.phrase_path])

    print(f"\n{'='*60}")
    print(f"RUNNING: {exp['run_name']}")
    print(f"  variant={exp['variant']}  lambda={exp['lambda_recon']}  mask_ratio={exp['mask_ratio']}")
    print(f"  cmd: {' '.join(cmd)}")
    print(f"{'='*60}\n")

    result = subprocess.run(cmd, capture_output=False)
    return result.returncode


def collect_results(results_dir):
    """Collect all individual result JSONs into one summary table."""
    summary = []
    for fname in sorted(os.listdir(results_dir)):
        if fname.endswith(".json") and fname != "summary.json":
            path = os.path.join(results_dir, fname)
            with open(path) as f:
                data = json.load(f)
            row = {
                "run_name": data.get("run_name", fname),
                "variant": data.get("variant"),
                "lambda_recon": data.get("lambda_recon"),
                "mask_ratio": data.get("mask_ratio"),
                "wall_time_min": round(data.get("wall_time_seconds", 0) / 60, 1),
            }
            # Add final retrieval metrics
            final = data.get("final_retrieval", {})
            for k, v in final.items():
                row[k] = v
            summary.append(row)

    # Save summary
    summary_path = os.path.join(results_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    # Print table
    print(f"\n{'='*80}")
    print("ABLATION SUMMARY")
    print(f"{'='*80}")
    header = f"{'Run':<20} {'Var':>3} {'λ':>5} {'MR':>5} | {'I2T@1':>6} {'I2T@5':>6} {'T2I@1':>6} {'T2I@5':>6}"
    print(header)
    print("-" * len(header))
    for row in summary:
        print(f"{row['run_name']:<20} {row.get('variant','?'):>3} "
              f"{row.get('lambda_recon',0):>5.1f} {row.get('mask_ratio',0):>5.2f} | "
              f"{row.get('i2t_r1',0):>6.2f} {row.get('i2t_r5',0):>6.2f} "
              f"{row.get('t2i_r1',0):>6.2f} {row.get('t2i_r5',0):>6.2f}")

    print(f"\nFull results saved to {summary_path}")
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--coco_root", type=str, default="./data/coco")
    parser.add_argument("--vocab_path", type=str, default="./vocab.json")
    parser.add_argument("--phrase_path", type=str, default="./phrases.json")
    parser.add_argument("--results_dir", type=str, default="./results/ablations")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry_run", action="store_true",
                        help="Print experiments without running")
    args = parser.parse_args()

    os.makedirs(args.results_dir, exist_ok=True)
    os.makedirs(os.path.join(args.results_dir, "checkpoints"), exist_ok=True)

    grid = build_experiment_grid()
    print(f"Ablation grid: {len(grid)} experiments")
    for exp in grid:
        print(f"  {exp['run_name']}: variant={exp['variant']} "
              f"lambda={exp['lambda_recon']} mask_ratio={exp['mask_ratio']}")

    if args.dry_run:
        print("\n[DRY RUN] No experiments executed.")
        return

    # Run experiments
    failed = []
    for i, exp in enumerate(grid):
        print(f"\n>>> Experiment {i+1}/{len(grid)}")
        # Skip if results already exist (resume support)
        results_path = os.path.join(args.results_dir, f"{exp['run_name']}.json")
        if os.path.exists(results_path):
            print(f"  Results already exist at {results_path}, skipping.")
            continue

        ret = run_single_experiment(exp, args)
        if ret != 0:
            print(f"  WARNING: {exp['run_name']} exited with code {ret}")
            failed.append(exp['run_name'])

    # Collect and print summary
    summary = collect_results(args.results_dir)

    if failed:
        print(f"\nFailed experiments: {failed}")
    else:
        print(f"\nAll {len(grid)} experiments completed successfully.")


if __name__ == "__main__":
    main()