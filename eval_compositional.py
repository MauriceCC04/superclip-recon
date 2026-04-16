"""
Compositional evaluation for SuperCLIP-Recon.

Benchmarks:
    1. Winoground  — 400 examples testing compositional understanding
    2. ARO (subset) — tests attribute binding and relation understanding

Usage:
    python eval_compositional.py --checkpoint ./checkpoints/epoch_10.pt \
                                  --benchmark all
"""

import os
import json
import argparse
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from tqdm import tqdm


def evaluate_winoground(model, device, hf_token=None):
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
        return None

    print(f"  Loaded Winoground: {len(dataset)} examples")

    model.eval()
    text_correct = 0
    image_correct = 0
    group_correct = 0
    total = 0

    for example in tqdm(dataset, desc="  Winoground"):
        img0 = example["image_0"].convert("RGB")
        img1 = example["image_1"].convert("RGB")
        img0_t = model.preprocess(img0).unsqueeze(0).to(device)
        img1_t = model.preprocess(img1).unsqueeze(0).to(device)

        cap0 = example["caption_0"]
        cap1 = example["caption_1"]
        tok0 = model.tokenizer(cap0).to(device)
        tok1 = model.tokenizer(cap1).to(device)

        with torch.no_grad():
            img0_emb = model.encode_image(img0_t)
            img1_emb = model.encode_image(img1_t)
            txt0_emb = model.encode_text(tok0)
            txt1_emb = model.encode_text(tok1)

            s00 = (img0_emb @ txt0_emb.T).item()
            s01 = (img0_emb @ txt1_emb.T).item()
            s10 = (img1_emb @ txt0_emb.T).item()
            s11 = (img1_emb @ txt1_emb.T).item()

        text_ok = (s00 > s01) and (s11 > s10)
        image_ok = (s00 > s10) and (s11 > s01)
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


def evaluate_aro(model, device, subset="VG-Attribution", max_examples=2000):
    try:
        from datasets import load_dataset
    except ImportError:
        print("  [SKIP] Install 'datasets' package: pip install datasets")
        return None

    aro_configs = {
        "VG-Attribution": ("gowitheflow/ARO-Visual-Attribution", None),
        "VG-Relation": ("gowitheflow/ARO-Visual-Relation", None),
    }

    if subset not in aro_configs:
        print(f"  [SKIP] Unknown ARO subset: {subset}.")
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

    if len(dataset) > max_examples:
        dataset = dataset.select(range(max_examples))

    print(f"  Loaded ARO {subset}: {len(dataset)} examples")

    model.eval()
    correct = 0
    total = 0

    for example in tqdm(dataset, desc=f"  ARO {subset}"):
        try:
            img = example["image"].convert("RGB")
            img_t = model.preprocess(img).unsqueeze(0).to(device)

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

        except Exception:
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


def run_compositional_eval(model, device, benchmarks="all", hf_token=None):
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
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    from config import Config
    from model import SuperCLIPRecon

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    model = SuperCLIPRecon(cfg).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"Loaded checkpoint: {args.checkpoint}")

    metrics = run_compositional_eval(model, device,
                                     benchmarks=args.benchmark,
                                     hf_token=args.hf_token)

    print("\n=== Compositional Evaluation Summary ===")
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
