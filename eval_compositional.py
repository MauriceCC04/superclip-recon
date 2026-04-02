"""
Compositional evaluation for SuperCLIP-Recon.

Benchmarks:
    1. Winoground  — 400 examples testing compositional understanding
    2. ARO (subset) — tests attribute binding and relation understanding

Both measure whether the model can distinguish captions that contain the
same words but in different compositional structures.

Winoground:
    Each example has (image_0, image_1, caption_0, caption_1) where the
    captions use the same words in different structures.
    Metrics: text_score, image_score, group_score (all 0-100).

ARO (VG-Attribution / VG-Relation):
    Each example has one image and two captions (correct vs swapped-attribute
    or swapped-relation). Model must score the correct caption higher.
    Metric: accuracy (0-100).

Usage:
    # Winoground (requires HuggingFace auth token)
    python eval_compositional.py --checkpoint ./checkpoints/epoch_10.pt \
                                  --benchmark winoground \
                                  --hf_token YOUR_TOKEN

    # ARO
    python eval_compositional.py --checkpoint ./checkpoints/epoch_10.pt \
                                  --benchmark aro

    # Both
    python eval_compositional.py --checkpoint ./checkpoints/epoch_10.pt \
                                  --benchmark all

Setup:
    pip install datasets  # for HuggingFace dataset loading

    For Winoground: you need a HuggingFace account and must accept the
    dataset license at https://huggingface.co/datasets/facebook/winoground
    Then pass --hf_token or set HF_TOKEN env var.

    For ARO: auto-downloads from HuggingFace (no special access needed).
"""

import os
import json
import argparse
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from tqdm import tqdm


# ─── Winoground ──────────────────────────────────────────────────────────────

def evaluate_winoground(model, device, hf_token=None):
    """
    Evaluate on Winoground benchmark.

    Each example: (image_0, image_1, caption_0, caption_1)
    Text score:  sim(img0, cap0) > sim(img0, cap1) AND sim(img1, cap1) > sim(img1, cap0)
    Image score: sim(img0, cap0) > sim(img1, cap0) AND sim(img1, cap1) > sim(img0, cap1)
    Group score: both text AND image correct

    Returns:
        dict with text_score, image_score, group_score (0-100)
    """
    try:
        from datasets import load_dataset
    except ImportError:
        print("  [SKIP] Install 'datasets' package: pip install datasets")
        return None

    token = hf_token or os.environ.get("HF_TOKEN")
    if not token:
        print("  [SKIP] Winoground requires HF auth. Pass --hf_token or set HF_TOKEN.")
        return None

    try:
        dataset = load_dataset("facebook/winoground", split="test", token=token)
    except Exception as e:
        print(f"  [SKIP] Could not load Winoground: {e}")
        print("  Make sure you accepted the license at:")
        print("  https://huggingface.co/datasets/facebook/winoground")
        return None

    print(f"  Loaded Winoground: {len(dataset)} examples")

    model.eval()
    text_correct = 0
    image_correct = 0
    group_correct = 0
    total = 0

    for example in tqdm(dataset, desc="  Winoground"):
        # Load and preprocess images
        img0 = example["image_0"].convert("RGB")
        img1 = example["image_1"].convert("RGB")
        img0_t = model.preprocess(img0).unsqueeze(0).to(device)
        img1_t = model.preprocess(img1).unsqueeze(0).to(device)

        # Tokenize captions
        cap0 = example["caption_0"]
        cap1 = example["caption_1"]
        tok0 = model.tokenizer(cap0).to(device)
        tok1 = model.tokenizer(cap1).to(device)

        with torch.no_grad():
            # Encode
            img0_emb = model.encode_image(img0_t)  # [1, D]
            img1_emb = model.encode_image(img1_t)
            txt0_emb = model.encode_text(tok0)      # [1, D]
            txt1_emb = model.encode_text(tok1)

            # Compute similarities
            s00 = (img0_emb @ txt0_emb.T).item()  # img0 <-> cap0
            s01 = (img0_emb @ txt1_emb.T).item()  # img0 <-> cap1
            s10 = (img1_emb @ txt0_emb.T).item()  # img1 <-> cap0
            s11 = (img1_emb @ txt1_emb.T).item()  # img1 <-> cap1

        # Text score: each image prefers its own caption
        text_ok = (s00 > s01) and (s11 > s10)
        # Image score: each caption prefers its own image
        image_ok = (s00 > s10) and (s11 > s01)
        # Group score: both correct
        group_ok = text_ok and image_ok

        text_correct += int(text_ok)
        image_correct += int(image_ok)
        group_correct += int(group_ok)
        total += 1

    metrics = {
        "winoground_text_score": round(text_correct / total * 100, 2),
        "winoground_image_score": round(image_correct / total * 100, 2),
        "winoground_group_score": round(group_correct / total * 100, 2),
        "winoground_n": total,
    }
    return metrics


# ─── ARO (Attribution, Relation, Order) ──────────────────────────────────────

def evaluate_aro(model, device, subset="VG-Attribution", max_examples=2000):
    """
    Evaluate on ARO benchmark subset.

    Each example: one image + correct caption + hard-negative caption
    (attributes or relations swapped).
    Metric: accuracy = fraction where sim(img, correct) > sim(img, negative).

    Subsets: VG-Attribution, VG-Relation, COCO-Order, Flickr30k-Order

    Returns:
        dict with aro_{subset}_accuracy (0-100)
    """
    try:
        from datasets import load_dataset
    except ImportError:
        print("  [SKIP] Install 'datasets' package: pip install datasets")
        return None

    # ARO dataset identifiers on HuggingFace
    # Note: the original cambridgeltl/aro uses a loading script that newer
    # versions of `datasets` no longer support. These are Parquet re-uploads
    # with the same data and field names (image, true_caption, false_caption).
    aro_configs = {
        "VG-Attribution": ("gowitheflow/ARO-Visual-Attribution", None),
        "VG-Relation": ("gowitheflow/ARO-Visual-Relation", None),
    }

    if subset not in aro_configs:
        print(f"  [SKIP] Unknown ARO subset: {subset}. Available: {list(aro_configs.keys())}")
        return None

    dataset_name, config = aro_configs[subset]

    try:
        if config:
            dataset = load_dataset(dataset_name, config, split="test")
        else:
            dataset = load_dataset(dataset_name, split="test")
    except Exception as e:
        print(f"  [SKIP] Could not load ARO {subset}: {e}")
        return None

    # Cap examples for speed
    if len(dataset) > max_examples:
        dataset = dataset.select(range(max_examples))

    print(f"  Loaded ARO {subset}: {len(dataset)} examples")

    model.eval()
    correct = 0
    total = 0

    for example in tqdm(dataset, desc=f"  ARO {subset}"):
        try:
            # Load image
            img = example["image"].convert("RGB")
            img_t = model.preprocess(img).unsqueeze(0).to(device)

            # Get captions
            true_cap = example["true_caption"]
            false_cap = example["false_caption"]

            tok_true = model.tokenizer(true_cap).to(device)
            tok_false = model.tokenizer(false_cap).to(device)

            with torch.no_grad():
                img_emb = model.encode_image(img_t)
                txt_true_emb = model.encode_text(tok_true)
                txt_false_emb = model.encode_text(tok_false)

                s_true = (img_emb @ txt_true_emb.T).item()
                s_false = (img_emb @ txt_false_emb.T).item()

            if s_true > s_false:
                correct += 1
            total += 1

        except Exception as e:
            # Skip malformed examples
            continue

    if total == 0:
        print(f"  [WARN] No valid examples processed for ARO {subset}")
        return None

    key = f"aro_{subset.lower().replace('-', '_')}_accuracy"
    metrics = {
        key: round(correct / total * 100, 2),
        f"aro_{subset.lower().replace('-', '_')}_n": total,
    }
    return metrics


# ─── Runner ──────────────────────────────────────────────────────────────────

def run_compositional_eval(model, device, benchmarks="all", hf_token=None):
    """
    Run selected compositional benchmarks.

    Args:
        benchmarks: "winoground", "aro", or "all"
    Returns:
        dict of all metrics
    """
    all_metrics = {}

    if benchmarks in ("winoground", "all"):
        print("\n--- Winoground ---")
        wino = evaluate_winoground(model, device, hf_token=hf_token)
        if wino:
            all_metrics.update(wino)
            print(f"  Text:  {wino['winoground_text_score']:.1f}")
            print(f"  Image: {wino['winoground_image_score']:.1f}")
            print(f"  Group: {wino['winoground_group_score']:.1f}")

    if benchmarks in ("aro", "all"):
        for subset in ["VG-Attribution", "VG-Relation"]:
            print(f"\n--- ARO {subset} ---")
            aro = evaluate_aro(model, device, subset=subset)
            if aro:
                all_metrics.update(aro)
                for k, v in aro.items():
                    if "accuracy" in k:
                        print(f"  Accuracy: {v:.1f}")

    return all_metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--benchmark", type=str, default="all",
                        choices=["winoground", "aro", "all"])
    parser.add_argument("--hf_token", type=str, default=None)
    parser.add_argument("--output", type=str, default=None,
                        help="Save results to JSON file")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load model
    from config import Config
    from model import SuperCLIPRecon

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    model = SuperCLIPRecon(cfg).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"Loaded checkpoint: {args.checkpoint}")

    # Run evaluation
    metrics = run_compositional_eval(model, device,
                                     benchmarks=args.benchmark,
                                     hf_token=args.hf_token)

    # Print summary
    print("\n=== Compositional Evaluation Summary ===")
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    # Save
    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
