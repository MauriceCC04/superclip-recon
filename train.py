"""
Training script for SuperCLIP-Recon.

Usage:
    # Baseline only (lambda=0):
    python train.py --lambda_recon 0.0

    # With reconstruction loss (Variant A):
    python train.py --lambda_recon 0.5 --variant A

    # Variant B (phrase reconstruction, per-caption inline):
    python train.py --lambda_recon 0.5 --variant B

    # Save results to file (for ablation analysis):
    python train.py --lambda_recon 0.5 --results_file ./results/run1.json

    # HPC-safe settings (default: optimizer state NOT saved; pass
        # --save_optimizer_state to include it):
        python train.py --save_strategy last_and_best --keep_last_k 2 \
                        --eval_max_images 1000"""

import os
import json
import time
import argparse
import random
import glob
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
    """
    if warmup_steps <= 0:
        return LambdaLR(optimizer, lr_lambda=lambda _: 1.0)

    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        remaining = total_steps - warmup_steps
        if remaining <= 0:
            return 1.0
        return max(0.0, float(total_steps - current_step) / float(remaining))

    return LambdaLR(optimizer, lr_lambda=lr_lambda)


# ─── Checkpoint retention ────────────────────────────────────────────────────

def manage_checkpoints(save_dir, save_strategy, keep_last_k, current_epoch,
                       best_metric_epoch):
    """
    Delete old checkpoints according to retention policy.

    save_strategy:
        all            — keep every epoch checkpoint
        last           — keep only the latest checkpoint
        best           — keep only the best checkpoint
        last_and_best  — keep latest + best (default, safest for HPC)

    keep_last_k: when > 0, also keep the most recent K checkpoints
                 regardless of strategy (extra safety net).
    """
    if save_strategy == "all":
        return  # keep everything

    ckpt_files = sorted(glob.glob(os.path.join(save_dir, "epoch_*.pt")))
    if not ckpt_files:
        return

    keep = set()

    # Always keep current epoch
    current_path = os.path.join(save_dir, f"epoch_{current_epoch}.pt")
    keep.add(current_path)

    # Keep best if strategy includes it
    if save_strategy in ("best", "last_and_best") and best_metric_epoch is not None:
        best_path = os.path.join(save_dir, f"epoch_{best_metric_epoch}.pt")
        keep.add(best_path)

    # Keep last K
    if keep_last_k > 0:
        for p in ckpt_files[-keep_last_k:]:
            keep.add(p)

    # Delete the rest
    for p in ckpt_files:
        if p not in keep and os.path.isfile(p):
            os.remove(p)


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
                        help="[DEPRECATED] Ignored. Variant B extracts phrases inline "
                             "per caption; training does not read phrases.json.")
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    # --- Checkpoint retention (HPC safety) ---
    parser.add_argument("--save_strategy", type=str, default="last_and_best",
                        choices=["all", "last", "best", "last_and_best"],
                        help="Checkpoint retention strategy (default: last_and_best)")
    parser.add_argument("--keep_last_k", type=int, default=2,
                        help="Also keep the K most recent checkpoints (0 to disable)")
    parser.add_argument("--save_optimizer_state", action="store_true", default=False,
                        help="Include optimizer state in checkpoints (doubles size)")
    # --- Eval cost controls ---
    parser.add_argument("--eval_every_epoch", type=int, default=1,
                        help="Run retrieval eval every N epochs (default: 1)")
    parser.add_argument("--eval_max_images", type=int, default=5000,
                        help="Max images for retrieval eval (default: 5000 = full COCO val)")
    parser.add_argument("--skip_eval", action="store_true",
                        help="Skip retrieval evaluation entirely")
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
    cfg.train.eval_every_epoch = args.eval_every_epoch

    set_seed(cfg.train.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Run: {args.run_name} | variant={args.variant} | "
          f"lambda={args.lambda_recon} | mask_ratio={args.mask_ratio}")
    print(f"Checkpoint strategy: {args.save_strategy} | keep_last_k={args.keep_last_k} | "
          f"save_optimizer={args.save_optimizer_state}")

    # --- Vocab (load early so we can set head size before model creation) ---
    vocab_map = load_vocab(args.vocab_path)
    cfg.model.num_token_classes = len(vocab_map)
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

    # --- Variant B: phrases are extracted inline per caption (no external file read) ---
    if args.variant == "B":
        print("Variant B: phrases are extracted inline per caption by "
              "create_phrase_mask_from_captions — no external phrases.json is used.")
        if args.phrase_path is not None:
            print(f"[DEPRECATED] --phrase_path={args.phrase_path} is ignored by training.")

    # --- Optimizer ---
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_params, lr=cfg.train.lr, weight_decay=cfg.train.weight_decay
    )

    # --- Warmup scheduler ---
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
    history = []
    global_step = 0
    best_metric_epoch = None
    best_metric_val = -1.0
    ckpt_size_mb = None  # set after first checkpoint is saved

    for epoch in range(cfg.train.epochs):
        model.train()
        epoch_losses = {"l_token_cls": 0, "l_recon": 0, "l_total": 0}
        n_batches = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg.train.epochs}")
        for batch_idx, (images, token_ids, captions_raw, img_ids) in enumerate(pbar):

            # --- Create masks on CPU BEFORE moving to GPU ---
            if args.variant == "B":
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

            # --- Forward (skip text encoding during training — not needed for loss) ---
            outputs = model(images, token_ids, encode_text=False)

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
            scheduler.step()
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
        ckpt_data = {
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "config": cfg,
            "avg_losses": avg,
        }
        if args.save_optimizer_state:
            ckpt_data["optimizer_state_dict"] = optimizer.state_dict()
            ckpt_data["scheduler_state_dict"] = scheduler.state_dict()

        ckpt_path = os.path.join(cfg.train.save_dir, f"epoch_{epoch+1}.pt")
        torch.save(ckpt_data, ckpt_path)
        ckpt_size_mb = os.path.getsize(ckpt_path) / (1024 * 1024)
        print(f"Saved checkpoint: {ckpt_path} ({ckpt_size_mb:.1f} MB)")

        # --- Evaluation ---
        epoch_record = {"epoch": epoch + 1, "losses": avg}
        if not args.skip_eval and (epoch + 1) % cfg.train.eval_every_epoch == 0:
            print(f"Running retrieval evaluation (max_images={args.eval_max_images})...")
            metrics = run_retrieval_eval(model, cfg, device, max_images=args.eval_max_images)
            print(f"  I->T R@1={metrics['i2t_r1']:.2f}  "
                  f"R@5={metrics['i2t_r5']:.2f}  R@10={metrics['i2t_r10']:.2f}")
            print(f"  T->I R@1={metrics['t2i_r1']:.2f}  "
                  f"R@5={metrics['t2i_r5']:.2f}  R@10={metrics['t2i_r10']:.2f}")
            epoch_record["retrieval"] = metrics

            # Track best for checkpoint retention
            rsum = metrics["i2t_r1"] + metrics["t2i_r1"]
            if rsum > best_metric_val:
                best_metric_val = rsum
                best_metric_epoch = epoch + 1

            if cfg.train.use_wandb:
                import wandb
                wandb.log(metrics, step=(epoch + 1) * len(train_loader))

        history.append(epoch_record)

        # --- Checkpoint retention ---
        manage_checkpoints(
            save_dir=cfg.train.save_dir,
            save_strategy=args.save_strategy,
            keep_last_k=args.keep_last_k,
            current_epoch=epoch + 1,
            best_metric_epoch=best_metric_epoch,
        )

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
        "save_strategy": args.save_strategy,
        "checkpoint_size_mb": round(ckpt_size_mb, 1) if ckpt_size_mb is not None else None,
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
