"""
Ablation sweep runner for SuperCLIP-Recon.

Runs a compact grid of experiments and collects results in one directory.
Designed to run inside a single SLURM job — experiments run sequentially.

Grid:
    1. Lambda sweep:        lambda in {0.0, 0.1, 0.5, 1.0}  (Variant A, mask_ratio=0.15)
    2. Masking rate sweep:  mask_ratio in {0.10, 0.15, 0.25} (Variant A, lambda=0.5)
    3. Variant comparison:  A vs B                            (lambda=0.5, mask_ratio=0.15)

Usage:
    python run_ablations.py --coco_root ./data/coco --vocab_path ./vocab.json \
                            --phrase_path ./phrases.json --results_dir ./results/ablations
"""

import os
import json
import shutil
import subprocess
import argparse


def _config_key(variant, lam, mr):
    return (variant, round(lam, 4), round(mr, 4))


def build_experiment_grid():
    seen_configs = {}

    def _add(name, variant, lam, mr):
        key = _config_key(variant, lam, mr)
        if key not in seen_configs:
            seen_configs[key] = {
                "run_name": name,
                "variant": variant,
                "lambda_recon": lam,
                "mask_ratio": mr,
            }

    for lam in [0.0, 0.1, 0.5, 1.0]:
        _add(f"lambda_{lam:.1f}", "A", lam, 0.15)

    for mr in [0.10, 0.15, 0.25]:
        _add(f"maskrate_{mr:.2f}", "A", 0.5, mr)

    for var in ["A", "B"]:
        _add(f"variant_{var}", var, 0.5, 0.15)

    return list(seen_configs.values())


_MAIN_RUN_MAP = {
    ("A", 0.0,  0.15): "baseline.json",
    ("A", 0.5,  0.15): "variant_a.json",
    ("B", 0.5,  0.15): "variant_b.json",
}


def try_reuse_main_result(exp, main_results_dir, ablation_results_dir):
    key = _config_key(exp["variant"], exp["lambda_recon"], exp["mask_ratio"])
    main_fname = _MAIN_RUN_MAP.get(key)
    if main_fname is None:
        return False

    main_path = os.path.join(main_results_dir, main_fname)
    if not os.path.isfile(main_path):
        return False

    dest_path = os.path.join(ablation_results_dir, f"{exp['run_name']}.json")
    shutil.copy2(main_path, dest_path)
    print(f"  Reused main result {main_fname} → {dest_path}")
    return True


def run_single_experiment(exp, base_args):
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
        "--save_strategy", "last_and_best",
        "--keep_last_k", "1",
    ]

    if exp["variant"] == "B" and base_args.phrase_path:
        cmd.extend(["--phrase_path", base_args.phrase_path])

    print(f"\n{'='*60}")
    print(f"RUNNING: {exp['run_name']}")
    print(f"  variant={exp['variant']}  lambda={exp['lambda_recon']}  mask_ratio={exp['mask_ratio']}")
    print(f"{'='*60}\n")

    result = subprocess.run(cmd, capture_output=False)
    return result.returncode


def collect_results(results_dir):
    summary = []
    for fname in sorted(os.listdir(results_dir)):
        if fname.endswith(".json") and fname != "summary.json":
            path = os.path.join(results_dir, fname)
            if not os.path.isfile(path):
                continue
            with open(path) as f:
                data = json.load(f)
            row = {
                "run_name": data.get("run_name", fname),
                "variant": data.get("variant"),
                "lambda_recon": data.get("lambda_recon"),
                "mask_ratio": data.get("mask_ratio"),
                "wall_time_min": round(data.get("wall_time_seconds", 0) / 60, 1),
            }
            final = data.get("final_retrieval", {})
            for k, v in final.items():
                row[k] = v
            summary.append(row)

    summary_path = os.path.join(results_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

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
    parser.add_argument("--main_results_dir", type=str, default="./results")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.results_dir, exist_ok=True)
    os.makedirs(os.path.join(args.results_dir, "checkpoints"), exist_ok=True)

    grid = build_experiment_grid()
    print(f"Ablation grid: {len(grid)} unique experiments")
    for exp in grid:
        print(f"  {exp['run_name']}: variant={exp['variant']} "
              f"lambda={exp['lambda_recon']} mask_ratio={exp['mask_ratio']}")

    if args.dry_run:
        print("\n[DRY RUN] No experiments executed.")
        return

    failed = []
    skipped = 0
    reused = 0
    for i, exp in enumerate(grid):
        print(f"\n>>> Experiment {i+1}/{len(grid)}: {exp['run_name']}")

        results_path = os.path.join(args.results_dir, f"{exp['run_name']}.json")
        if os.path.exists(results_path):
            print(f"  Results already exist at {results_path}, skipping.")
            skipped += 1
            continue

        if try_reuse_main_result(exp, args.main_results_dir, args.results_dir):
            reused += 1
            continue

        ret = run_single_experiment(exp, args)
        if ret != 0:
            print(f"  WARNING: {exp['run_name']} exited with code {ret}")
            failed.append(exp['run_name'])

    summary = collect_results(args.results_dir)

    trained = len(grid) - skipped - reused - len(failed)
    print(f"\n  Grid size:  {len(grid)}")
    print(f"  Trained:    {trained}")
    print(f"  Reused:     {reused} (from main experiments)")
    print(f"  Skipped:    {skipped} (already had results)")
    print(f"  Failed:     {len(failed)}")
    if failed:
        print(f"  Failed runs: {failed}")
    else:
        print(f"\nAll {len(grid)} experiments accounted for.")


if __name__ == "__main__":
    main()
