"""
Training script for SuperCLIP-Recon.

Usage:
    # Baseline only (lambda=0):
    python train.py --lambda_recon 0.0

    # With reconstruction loss (Variant A):
    python train.py --lambda_recon 0.5 --variant A

    # Variant B (phrase reconstruction):
    python train.py --lambda_recon 0.5 --variant B --phrase_path ./phrases.json

    # Save results to file (for ablation analysis):
    python train.py --lambda_recon 0.5 --results_file ./results/run1.json
"""

import os
import json
import time
import argparse
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import Config
from dataset import COCOCaptionsDataset
from model import SuperCLIPRecon
from losses import build_token_labels, create_mask, create_phrase_mask, total_loss
from build_vocab import load_vocab
from evaluate import run_retrieval_eval


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--coco_root", type=str, default="./data/coco")
    parser.add_argument("--vocab_path", type=str, default="./vocab.json")
    parser.add_argument("--lambda_recon", type=float, default=0.5)
    parser.add_argument("--variant", type=str, default="A", choices=["A", "B"])
    parser.add_argument("--mask_ratio", type=float, default=0.15)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--save_dir", type=str, default="./checkpoints")
    parser.add_argument("--run_name", type=str, default="superclip-recon")
    parser.add_argument("--results_file", type=str, default=None,
                        help="Path to save final results JSON (for ablation collection)")
    parser.add_argument("--phrase_path", type=str, default=None,
                        help="Path to phrases.json (required for Variant B)")
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def train():
    args = parse_args()
    wall_start = time.time()

    # --- Config ---
    cfg = Config()
    cfg.data.coco_root = args.coco_root
    cfg.model.mask_ratio = args.mask_ratio
    cfg.model.variant = args.variant
    cfg.train.lambda_recon = args.lambda_recon
    cfg.train.batch_size = args.batch_size
    cfg.train.epochs = args.epochs
    cfg.train.lr = args.lr
    cfg.train.save_dir = args.save_dir
    cfg.train.run_name = args.run_name
    cfg.train.use_wandb = args.use_wandb
    cfg.train.seed = args.seed

    set_seed(cfg.train.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Run: {args.run_name} | variant={args.variant} | "
          f"lambda={args.lambda_recon} | mask_ratio={args.mask_ratio}")

    # --- Model ---
    model = SuperCLIPRecon(cfg).to(device)
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model created. Trainable params: {n_trainable:,}")

    # --- Data ---
    train_dataset = COCOCaptionsDataset(
        root=cfg.data.coco_root,
        ann_file=cfg.data.train_ann,
        image_dir=cfg.data.train_images,
        transform=model.preprocess,
        tokenizer=model.tokenizer,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        num_workers=cfg.data.num_workers,
        pin_memory=cfg.data.pin_memory,
        drop_last=True,
    )

    # --- Vocab map for token classification ---
    vocab_map = load_vocab(args.vocab_path)
    print(f"Loaded vocab with {len(vocab_map)} token classes")

    # --- Phrase data for Variant B ---
    phrase_data = None
    if args.variant == "B":
        if args.phrase_path is None:
            raise ValueError("Variant B requires --phrase_path. Run extract_phrases.py first.")
        with open(args.phrase_path) as f:
            phrase_data = json.load(f)
        print(f"Loaded phrase data: {sum(len(v) for v in phrase_data.values())} phrases")

    # --- Optimizer ---
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_params, lr=cfg.train.lr, weight_decay=cfg.train.weight_decay
    )

    # --- Optional: wandb ---
    if cfg.train.use_wandb:
        import wandb
        wandb.init(project="superclip-recon", name=cfg.train.run_name, config={
            "variant": args.variant, "lambda_recon": args.lambda_recon,
            "mask_ratio": args.mask_ratio, "lr": args.lr, "batch_size": args.batch_size,
        })

    # --- Max masks for reconstruction ---
    max_masks = int(cfg.data.max_caption_length * cfg.model.mask_ratio) + 1

    # --- Training loop ---
    os.makedirs(cfg.train.save_dir, exist_ok=True)
    history = []  # per-epoch records for results file

    for epoch in range(cfg.train.epochs):
        model.train()
        epoch_losses = {"l_token_cls": 0, "l_recon": 0, "l_total": 0}
        n_batches = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg.train.epochs}")
        for batch_idx, (images, token_ids, captions_raw, img_ids) in enumerate(pbar):
            images = images.to(device)
            token_ids = token_ids.to(device)

            # --- Create masks for reconstruction ---
            if args.variant == "B" and phrase_data is not None:
                # img_ids comes directly from the dataset (shuffle-safe)
                batch_img_ids = img_ids.tolist()
                masked_token_ids, mask_targets, mask_positions = create_phrase_mask(
                    token_ids, phrase_data, batch_img_ids, max_masks
                )
            else:
                masked_token_ids, mask_targets, mask_positions = create_mask(
                    token_ids, cfg.model.mask_ratio, max_masks
                )

            # --- Forward ---
            outputs = model(images, token_ids)

            # --- Build token classification labels ---
            token_cls_labels = build_token_labels(
                token_ids, vocab_map, cfg.model.num_token_classes
            )

            # --- Loss ---
            loss, losses_dict = total_loss(
                token_cls_logits=outputs["token_cls_logits"],
                token_cls_labels=token_cls_labels,
                recon_logits=outputs["recon_logits"],
                mask_targets=mask_targets,
                lambda_recon=cfg.train.lambda_recon,
            )

            # --- Backward ---
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
            optimizer.step()

            # --- Logging ---
            for k, v in losses_dict.items():
                epoch_losses[k] += v
            n_batches += 1

            if batch_idx % cfg.train.log_every == 0:
                pbar.set_postfix({
                    "tc": f"{losses_dict['l_token_cls']:.4f}",
                    "rc": f"{losses_dict['l_recon']:.4f}",
                    "tot": f"{losses_dict['l_total']:.4f}",
                })

                if cfg.train.use_wandb:
                    import wandb
                    wandb.log(losses_dict, step=epoch * len(train_loader) + batch_idx)

        # --- Epoch summary ---
        avg = {k: v / max(n_batches, 1) for k, v in epoch_losses.items()}
        print(f"Epoch {epoch+1} avg losses: "
              f"tc={avg['l_token_cls']:.4f}  rc={avg['l_recon']:.4f}  total={avg['l_total']:.4f}")

        # --- Save checkpoint ---
        ckpt_path = os.path.join(cfg.train.save_dir, f"epoch_{epoch+1}.pt")
        torch.save({
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": cfg,
            "avg_losses": avg,
        }, ckpt_path)
        print(f"Saved checkpoint: {ckpt_path}")

        # --- Evaluation ---
        epoch_record = {"epoch": epoch + 1, "losses": avg}
        if (epoch + 1) % cfg.train.eval_every_epoch == 0:
            print("Running retrieval evaluation...")
            metrics = run_retrieval_eval(model, cfg, device)
            print(f"  I->T R@1={metrics['i2t_r1']:.2f}  "
                  f"R@5={metrics['i2t_r5']:.2f}  R@10={metrics['i2t_r10']:.2f}")
            print(f"  T->I R@1={metrics['t2i_r1']:.2f}  "
                  f"R@5={metrics['t2i_r5']:.2f}  R@10={metrics['t2i_r10']:.2f}")
            epoch_record["retrieval"] = metrics

            if cfg.train.use_wandb:
                import wandb
                wandb.log(metrics, step=(epoch + 1) * len(train_loader))

        history.append(epoch_record)

    # --- Save results file ---
    wall_time = time.time() - wall_start
    results = {
        "run_name": args.run_name,
        "variant": args.variant,
        "lambda_recon": args.lambda_recon,
        "mask_ratio": args.mask_ratio,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "seed": args.seed,
        "wall_time_seconds": round(wall_time, 1),
        "trainable_params": n_trainable,
        "history": history,
        "final_retrieval": history[-1].get("retrieval", {}),
    }

    if args.results_file:
        os.makedirs(os.path.dirname(args.results_file) or ".", exist_ok=True)
        with open(args.results_file, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {args.results_file}")
    else:
        fallback_path = os.path.join(cfg.train.save_dir, "results.json")
        with open(fallback_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {fallback_path}")

    print(f"Training complete. Wall time: {wall_time/60:.1f} min")


if __name__ == "__main__":
    train()
