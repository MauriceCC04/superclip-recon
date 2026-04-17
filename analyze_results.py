"""
Analyze results from SuperCLIP-Recon experiments.

Reads result JSONs from training runs and compositional evaluations,
then produces:
    1. Summary table (printed + saved as CSV)
    2. Retrieval comparison bar chart (baseline vs Variant A vs Variant B)
    3. Lambda sweep plot
    4. Masking rate sweep plot
    5. Training loss curves
    6. Compositional evaluation comparison (if available)

Usage:
    python analyze_results.py --results_dir ./results --output_dir ./results/figures
"""

import os
import json
import argparse
import csv
from collections import defaultdict

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError as e:
    raise SystemExit(
        "matplotlib is required for analyze_results.py. "
        "Install it with: pip install 'matplotlib>=3.8.0,<3.10.0' "
        f"(original error: {e})"
    )
import numpy as np

# ─── Styling ─────────────────────────────────────────────────────────────────

COLORS = {
    "baseline": "#888888",
    "variant_a": "#2196F3",
    "variant_b": "#FF9800",
}

def setup_style():
    plt.rcParams.update({
        "figure.figsize": (8, 5),
        "figure.dpi": 150,
        "font.size": 11,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


# ─── Data Loading ────────────────────────────────────────────────────────────

# Files that are NOT main training results and should be excluded
_NON_MAIN_PREFIXES = ("compositional_", "summary", "preflight", "smoke", "pilot")
_NON_MAIN_SUBDIRS = ("ablations", "figures", "preflight", "smoke")


def _is_main_result(fname: str, data: dict) -> bool:
    """
    Return True only if this JSON looks like a main training result.
    Filters out compositional evals, smoke results, preflight reports, etc.
    """
    base = fname.replace(".json", "")

    # Reject known non-main prefixes
    for prefix in _NON_MAIN_PREFIXES:
        if base.startswith(prefix):
            return False

    # Schema check: a main training result must have these keys
    required_keys = {"run_name", "variant", "lambda_recon"}
    if not required_keys.issubset(data.keys()):
        return False

    # Must have final_retrieval or history (actual training output)
    if "final_retrieval" not in data and "history" not in data:
        return False

    return True


def load_all_results(results_dir):
    """Load all main training result JSONs from directory (non-recursive)."""
    results = {}
    for fname in sorted(os.listdir(results_dir)):
        if not fname.endswith(".json"):
            continue
        # Skip subdirectory names that happen to match
        path = os.path.join(results_dir, fname)
        if not os.path.isfile(path):
            continue
        try:
            with open(path) as f:
                data = json.load(f)
        except (json.JSONDecodeError, IOError):
            continue

        if _is_main_result(fname, data):
            key = fname.replace(".json", "")
            results[key] = data
    return results


def load_ablation_results(results_dir):
    """Load ablation results from subdirectory."""
    ablation_dir = os.path.join(results_dir, "ablations")
    if not os.path.isdir(ablation_dir):
        return {}
    # Ablation results use the same schema as main results
    raw = {}
    for fname in sorted(os.listdir(ablation_dir)):
        if not fname.endswith(".json") or fname == "summary.json":
            continue
        path = os.path.join(ablation_dir, fname)
        if not os.path.isfile(path):
            continue
        try:
            with open(path) as f:
                data = json.load(f)
        except (json.JSONDecodeError, IOError):
            continue
        key = fname.replace(".json", "")
        raw[key] = data
    return raw


def load_compositional_results(results_dir):
    """Load compositional eval results."""
    comp = {}
    for fname in sorted(os.listdir(results_dir)):
        if fname.startswith("compositional_") and fname.endswith(".json"):
            path = os.path.join(results_dir, fname)
            if not os.path.isfile(path):
                continue
            try:
                with open(path) as f:
                    data = json.load(f)
            except (json.JSONDecodeError, IOError):
                continue
            key = fname.replace("compositional_", "").replace(".json", "")
            comp[key] = data
    return comp


# ─── Summary Table ───────────────────────────────────────────────────────────

def print_summary_table(results, compositional, output_dir):
    """Print and save a summary table of all main runs."""
    rows = []
    header = [
        "Run", "Variant", "λ", "MR",
        "I→T R@1", "I→T R@5", "I→T R@10",
        "T→I R@1", "T→I R@5", "T→I R@10",
        "Wino(G)", "ARO(Attr)", "ARO(Rel)",
        "Time(min)"
    ]

    for key, data in results.items():
        retr = data.get("final_retrieval", {})
        comp = compositional.get(key, {})
        row = [
            key,
            data.get("variant", "—"),
            data.get("lambda_recon", "—"),
            data.get("mask_ratio", "—"),
            f"{retr.get('i2t_r1', 0):.1f}",
            f"{retr.get('i2t_r5', 0):.1f}",
            f"{retr.get('i2t_r10', 0):.1f}",
            f"{retr.get('t2i_r1', 0):.1f}",
            f"{retr.get('t2i_r5', 0):.1f}",
            f"{retr.get('t2i_r10', 0):.1f}",
            f"{comp.get('winoground_group_score', '—')}",
            f"{comp.get('aro_vg_attribution_accuracy', '—')}",
            f"{comp.get('aro_vg_relation_accuracy', '—')}",
            f"{data.get('wall_time_seconds', 0) / 60:.0f}",
        ]
        rows.append(row)

    print("\n" + "=" * 120)
    print("RESULTS SUMMARY")
    print("=" * 120)
    fmt = "{:<16}" + "{:>8}" * (len(header) - 1)
    print(fmt.format(*header))
    print("-" * 120)
    for row in rows:
        print(fmt.format(*row))

    csv_path = os.path.join(output_dir, "summary.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)
    print(f"\nSaved: {csv_path}")


# ─── Plot 1: Retrieval Comparison ───────────────────────────────────────────

def plot_retrieval_comparison(results, output_dir):
    main_runs = {}
    for key in ["baseline", "variant_a", "variant_b"]:
        if key in results:
            main_runs[key] = results[key]

    if not main_runs:
        print("  [SKIP] No main run results found for retrieval comparison")
        return

    metrics = ["i2t_r1", "i2t_r5", "i2t_r10", "t2i_r1", "t2i_r5", "t2i_r10"]
    labels = ["I→T\nR@1", "I→T\nR@5", "I→T\nR@10", "T→I\nR@1", "T→I\nR@5", "T→I\nR@10"]

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(metrics))
    width = 0.25

    for i, (name, data) in enumerate(main_runs.items()):
        retr = data.get("final_retrieval", {})
        values = [retr.get(m, 0) for m in metrics]
        color = COLORS.get(name, f"C{i}")
        label = {"baseline": "Baseline (λ=0)", "variant_a": "Variant A",
                 "variant_b": "Variant B"}.get(name, name)
        ax.bar(x + i * width, values, width, label=label, color=color, alpha=0.85)

    ax.set_ylabel("Recall (%)")
    ax.set_title("COCO Retrieval: Baseline vs Reconstruction Variants")
    ax.set_xticks(x + width)
    ax.set_xticklabels(labels)
    ax.legend()
    ax.set_ylim(bottom=0)

    path = os.path.join(output_dir, "retrieval_comparison.png")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")


# ─── Plot 2: Lambda Sweep ───────────────────────────────────────────────────

def plot_lambda_sweep(ablations, output_dir):
    lambda_runs = {}
    for key, data in ablations.items():
        if key.startswith("lambda_"):
            lam = data.get("lambda_recon", 0)
            lambda_runs[lam] = data

    if len(lambda_runs) < 2:
        print("  [SKIP] Not enough lambda sweep results")
        return

    lambdas = sorted(lambda_runs.keys())
    i2t_r1 = [lambda_runs[l].get("final_retrieval", {}).get("i2t_r1", 0) for l in lambdas]
    t2i_r1 = [lambda_runs[l].get("final_retrieval", {}).get("t2i_r1", 0) for l in lambdas]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(lambdas, i2t_r1, "o-", color="#2196F3", label="I→T R@1", linewidth=2)
    ax.plot(lambdas, t2i_r1, "s--", color="#FF9800", label="T→I R@1", linewidth=2)
    ax.set_xlabel("λ (reconstruction loss weight)")
    ax.set_ylabel("Recall@1 (%)")
    ax.set_title("Lambda Sweep: Effect of Reconstruction Loss Weight")
    ax.legend()
    ax.set_xticks(lambdas)

    path = os.path.join(output_dir, "lambda_sweep.png")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")


# ─── Plot 3: Masking Rate Sweep ─────────────────────────────────────────────

def plot_maskrate_sweep(ablations, output_dir):
    mr_runs = {}
    for key, data in ablations.items():
        if key.startswith("maskrate_"):
            mr = data.get("mask_ratio", 0)
            mr_runs[mr] = data

    if len(mr_runs) < 2:
        print("  [SKIP] Not enough mask rate sweep results")
        return

    rates = sorted(mr_runs.keys())
    i2t_r1 = [mr_runs[r].get("final_retrieval", {}).get("i2t_r1", 0) for r in rates]
    t2i_r1 = [mr_runs[r].get("final_retrieval", {}).get("t2i_r1", 0) for r in rates]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(rates, i2t_r1, "o-", color="#2196F3", label="I→T R@1", linewidth=2)
    ax.plot(rates, t2i_r1, "s--", color="#FF9800", label="T→I R@1", linewidth=2)
    ax.set_xlabel("Masking Rate")
    ax.set_ylabel("Recall@1 (%)")
    ax.set_title("Masking Rate Sweep")
    ax.legend()
    ax.set_xticks(rates)

    path = os.path.join(output_dir, "maskrate_sweep.png")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")


# ─── Plot 4: Training Loss Curves ───────────────────────────────────────────

def plot_loss_curves(results, output_dir):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    loss_keys = [("l_total", "Total Loss"), ("l_token_cls", "Token Cls Loss"),
                 ("l_recon", "Recon Loss")]

    for ax, (lk, title) in zip(axes, loss_keys):
        for name in ["baseline", "variant_a", "variant_b"]:
            if name not in results:
                continue
            history = results[name].get("history", [])
            epochs = [h["epoch"] for h in history]
            values = [h.get("losses", {}).get(lk, 0) for h in history]
            if any(v > 0 for v in values):
                color = COLORS.get(name, "gray")
                label = {"baseline": "Baseline", "variant_a": "Var A",
                         "variant_b": "Var B"}.get(name, name)
                ax.plot(epochs, values, "o-", color=color, label=label,
                        linewidth=1.5, markersize=4)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title(title)
        ax.legend(fontsize=9)

    path = os.path.join(output_dir, "loss_curves.png")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")


# ─── Plot 5: Compositional Evaluation ───────────────────────────────────────

def plot_compositional(compositional, output_dir):
    if not compositional:
        print("  [SKIP] No compositional results found")
        return

    metrics_to_plot = []
    metric_labels = []

    sample = list(compositional.values())[0]
    if "winoground_group_score" in sample:
        metrics_to_plot.append("winoground_group_score")
        metric_labels.append("Wino\nGroup")
    if "winoground_text_score" in sample:
        metrics_to_plot.append("winoground_text_score")
        metric_labels.append("Wino\nText")
    if "winoground_image_score" in sample:
        metrics_to_plot.append("winoground_image_score")
        metric_labels.append("Wino\nImage")
    if "aro_vg_attribution_accuracy" in sample:
        metrics_to_plot.append("aro_vg_attribution_accuracy")
        metric_labels.append("ARO\nAttr")
    if "aro_vg_relation_accuracy" in sample:
        metrics_to_plot.append("aro_vg_relation_accuracy")
        metric_labels.append("ARO\nRel")

    if not metrics_to_plot:
        print("  [SKIP] No recognized compositional metrics")
        return

    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(metrics_to_plot))
    width = 0.25
    names = list(compositional.keys())

    for i, name in enumerate(names):
        data = compositional[name]
        values = [data.get(m, 0) for m in metrics_to_plot]
        color = COLORS.get(name, f"C{i}")
        label = {"baseline": "Baseline", "variant_a": "Variant A",
                 "variant_b": "Variant B"}.get(name, name)
        ax.bar(x + i * width, values, width, label=label, color=color, alpha=0.85)

    ax.set_ylabel("Score / Accuracy (%)")
    ax.set_title("Compositional Evaluation")
    ax.set_xticks(x + width * (len(names) - 1) / 2)
    ax.set_xticklabels(metric_labels)
    ax.legend()
    ax.set_ylim(bottom=0)

    path = os.path.join(output_dir, "compositional_comparison.png")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, default="./results")
    parser.add_argument("--output_dir", type=str, default="./results/figures")
    args = parser.parse_args()

    setup_style()
    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading results...")
    main_results = load_all_results(args.results_dir)
    ablation_results = load_ablation_results(args.results_dir)
    compositional_results = load_compositional_results(args.results_dir)

    print(f"  Main runs: {list(main_results.keys())}")
    print(f"  Ablations: {list(ablation_results.keys())}")
    print(f"  Compositional: {list(compositional_results.keys())}")

    print_summary_table(main_results, compositional_results, args.output_dir)

    print("\nGenerating plots...")
    plot_retrieval_comparison(main_results, args.output_dir)
    plot_loss_curves(main_results, args.output_dir)
    plot_lambda_sweep(ablation_results, args.output_dir)
    plot_maskrate_sweep(ablation_results, args.output_dir)
    plot_compositional(compositional_results, args.output_dir)

    print(f"\nAll outputs in {args.output_dir}/")


if __name__ == "__main__":
    main()
