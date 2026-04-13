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
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm

from config import Config
from dataset import COCOCaptionsDataset
from model import SuperCLIPRecon
from losses import (
    build_token_labels, create_mask,
    create_phrase_mask_from_captions, total_loss,
)
from build_vocab import load_vocab
from evaluate import run_retrieval_eval


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_warmup_scheduler(optimizer, warmup_steps: int, total_steps: int):
    """
    Linear warmup for `warmup_steps`, then linear decay to 0 over the rest.
    If warmup_steps <= 0 or total_steps <= warmup_steps, returns constant LR.
    """
    if warmup_steps <= 0:
        return LambdaLR(optimizer, lr_lambda=lambda _: 1.0)

    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        # Linear decay after warmup
        remaining = total_steps - warmup_steps
        if remaining <= 0:
            return 1.0
        return max(0.0, float(total_steps - current_step) / float(remaining))

    return LambdaLR(optimizer, lr_lambda=lr_lambda)


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
                        help="Path to phrases.json (optional; Variant B extracts phrases inline)")
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

    # --- Vocab (load early so we can set head size before model creation) ---
    vocab_map = load_vocab(args.vocab_path)
    cfg.model.num_token_classes = len(vocab_map)        # ← Fix 4
    print(f"Loaded vocab with {len(vocab_map)} token classes "
          f"(num_token_classes set to {cfg.model.num_token_classes})")

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

    # --- Phrase data for Variant B ---
    # phrase_data is still loaded for fallback / reference, but the primary
    # masking path now extracts phrases directly from the current caption
    # (see create_phrase_mask_from_captions in losses.py — Fix 3).
    phrase_data = None
    if args.variant == "B":
        if args.phrase_path is not None and os.path.isfile(args.phrase_path):
            with open(args.phrase_path) as f:
                phrase_data = json.load(f)
            print(f"Loaded phrase data: {sum(len(v) for v in phrase_data.values())} phrases "
                  "(used as fallback; primary extraction is per-caption)")
        else:
            print("Variant B: no phrase_path provided or file missing. "
                  "Using per-caption inline extraction only.")

    # --- Optimizer ---
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_params, lr=cfg.train.lr, weight_decay=cfg.train.weight_decay
    )

    # --- Warmup scheduler (Fix 2) ---
    steps_per_epoch = len(train_loader)
    total_steps = steps_per_epoch * cfg.train.epochs
    scheduler = build_warmup_scheduler(optimizer, cfg.train.warmup_steps, total_steps)
    print(f"Scheduler: linear warmup for {cfg.train.warmup_steps} steps, "
          f"then linear decay over {total_steps} total steps")

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
    global_step = 0

    for epoch in range(cfg.train.epochs):
        model.train()
        epoch_losses = {"l_token_cls": 0, "l_recon": 0, "l_total": 0}
        n_batches = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg.train.epochs}")
        for batch_idx, (images, token_ids, captions_raw, img_ids) in enumerate(pbar):

            # --- Create masks on CPU BEFORE moving to GPU ---
            # Masking uses Python loops with .item() calls; doing this on CPU
            # avoids thousands of GPU→CPU scalar transfers per batch.
            if args.variant == "B":
                # Fix 3: extract phrases from the CURRENT caption, not from
                # the image-level pool.  This ensures the phrase is always
                # present in the token sequence we are trying to mask.
                _, mask_targets, mask_positions = create_phrase_mask_from_captions(
                    token_ids, captions_raw, model.tokenizer, max_masks
                )
            else:
                _, mask_targets, mask_positions = create_mask(
                    token_ids, cfg.model.mask_ratio, max_masks
                )

            # --- Build token classification labels on CPU ---
            token_cls_labels = build_token_labels(
                token_ids, vocab_map, cfg.model.num_token_classes
            )

            # --- Move everything to GPU ---
            images = images.to(device)
            token_ids = token_ids.to(device)
            mask_targets = mask_targets.to(device)
            token_cls_labels = token_cls_labels.to(device)

            # --- Forward ---
            # Note: the text encoder receives the original unmasked token_ids.
            # Only mask_targets (ground-truth masked token IDs) are needed for
            # the reconstruction loss.  text_features are not used in the loss
            # but are computed by the frozen text encoder at negligible cost.
            outputs = model(images, token_ids)

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
            scheduler.step()          # ← Fix 2: advance LR schedule each step
            global_step += 1

            # --- Logging ---
            for k, v in losses_dict.items():
                epoch_losses[k] += v
            n_batches += 1

            if batch_idx % cfg.train.log_every == 0:
                current_lr = scheduler.get_last_lr()[0]
                pbar.set_postfix({
                    "tc": f"{losses_dict['l_token_cls']:.4f}",
                    "rc": f"{losses_dict['l_recon']:.4f}",
                    "tot": f"{losses_dict['l_total']:.4f}",
                    "lr": f"{current_lr:.2e}",
                })

                if cfg.train.use_wandb:
                    import wandb
                    wandb.log({**losses_dict, "lr": current_lr},
                              step=epoch * len(train_loader) + batch_idx)

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
            "scheduler_state_dict": scheduler.state_dict(),
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