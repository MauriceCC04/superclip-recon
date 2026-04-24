"""
Training script for SuperCLIP-Recon.

Modes:
    - auto (default): lambda_recon == 0 -> superclip_baseline,
                      lambda_recon  > 0 -> superclip_recon
    - clip_only: train only the CLIP contrastive objective
    - superclip_baseline: CLIP contrastive + token classification
    - superclip_recon: CLIP contrastive + token classification + reconstruction

The key repair in this version is that lambda=0 is now a real SuperCLIP-style
baseline rather than image-only token classification.
"""

import os
import sys
import json
import time
import math
import socket
import argparse
import random
import glob
import subprocess
from contextlib import nullcontext

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm

from config import Config
from dataset import COCOCaptionsDataset
from model import SuperCLIPRecon
from losses import (
    build_token_labels,
    create_mask,
    create_phrase_mask_from_captions,
    total_loss,
    resolve_train_mode,
)
from build_vocab import load_vocab
from evaluate import run_retrieval_eval


RETRIEVAL_KEYS = ["i2t_r1", "i2t_r5", "i2t_r10", "t2i_r1", "t2i_r5", "t2i_r10"]


def set_seed(seed: int, deterministic: bool = False):
    os.environ["PYTHONHASHSEED"] = str(seed)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception:
            pass
    else:
        torch.backends.cudnn.benchmark = True


def build_warmup_scheduler(optimizer, warmup_steps: int, total_steps: int):
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


def manage_checkpoints(save_dir, save_strategy, keep_last_k, current_epoch, best_metric_epoch):
    if save_strategy == "all":
        return

    ckpt_files = sorted(glob.glob(os.path.join(save_dir, "epoch_*.pt")))
    if not ckpt_files:
        return

    keep = set()
    current_path = os.path.join(save_dir, f"epoch_{current_epoch}.pt")
    keep.add(current_path)

    if save_strategy in ("best", "last_and_best") and best_metric_epoch is not None:
        best_path = os.path.join(save_dir, f"epoch_{best_metric_epoch}.pt")
        keep.add(best_path)

    if keep_last_k > 0:
        for path in ckpt_files[-keep_last_k:]:
            keep.add(path)

    for path in ckpt_files:
        if path not in keep and os.path.isfile(path):
            os.remove(path)


def retrieval_score(metrics: dict) -> float:
    return float(sum(metrics.get(key, 0.0) for key in RETRIEVAL_KEYS))


def safe_git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return None


def count_params(module: torch.nn.Module | None, trainable_only: bool = False) -> int:
    if module is None:
        return 0
    return sum(
        p.numel()
        for p in module.parameters()
        if (p.requires_grad or not trainable_only)
    )


def grad_norm(module: torch.nn.Module | None) -> float:
    if module is None:
        return 0.0
    vals = []
    for p in module.parameters():
        if p.grad is not None:
            vals.append(float(p.grad.detach().norm().item()))
    return float(sum(vals)) if vals else 0.0


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--coco_root", type=str, default="./data/coco")
    parser.add_argument("--vocab_path", type=str, default="./vocab.json")
    parser.add_argument("--lambda_recon", type=float, default=0.5)
    parser.add_argument("--lambda_clip", type=float, default=1.0)
    parser.add_argument("--lambda_token_cls", type=float, default=1.0)
    parser.add_argument(
        "--train_mode",
        type=str,
        default="auto",
        choices=["auto", "clip_only", "superclip_baseline", "superclip_recon"],
        help="Training objective. auto resolves to baseline when lambda_recon=0, otherwise recon.",
    )
    parser.add_argument("--variant", type=str, default="A", choices=["A", "B"])
    parser.add_argument("--mask_ratio", type=float, default=0.15)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--save_dir", type=str, default="./checkpoints")
    parser.add_argument("--run_name", type=str, default="superclip-recon")
    parser.add_argument("--results_file", type=str, default=None)
    parser.add_argument("--phrase_path", type=str, default=None, help="Deprecated. Variant B extracts phrases inline.")
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--freeze_text_tower", action="store_true")
    parser.add_argument("--freeze_vision_tower", action="store_true")
    parser.add_argument("--freeze_logit_scale", action="store_true")
    parser.add_argument("--no_amp", action="store_true", help="Disable CUDA AMP.")
    parser.add_argument("--save_strategy", type=str, default="last_and_best", choices=["all", "last", "best", "last_and_best"])
    parser.add_argument("--keep_last_k", type=int, default=1)
    parser.add_argument("--save_optimizer_state", action="store_true", default=False)
    parser.add_argument("--eval_every_epoch", type=int, default=1)
    parser.add_argument("--eval_max_images", type=int, default=5000)
    parser.add_argument("--skip_eval", action="store_true")
    return parser.parse_args()


def train():
    args = parse_args()
    wall_start = time.time()

    cfg = Config()
    cfg.data.coco_root = args.coco_root
    cfg.data.num_workers = args.num_workers
    cfg.model.mask_ratio = args.mask_ratio
    cfg.model.variant = args.variant
    cfg.train.train_mode = resolve_train_mode(args.train_mode, args.lambda_recon)
    cfg.train.lambda_clip = args.lambda_clip
    cfg.train.lambda_token_cls = args.lambda_token_cls
    cfg.train.lambda_recon = args.lambda_recon
    cfg.train.batch_size = args.batch_size
    cfg.train.epochs = args.epochs
    cfg.train.lr = args.lr
    cfg.train.save_dir = args.save_dir
    cfg.train.run_name = args.run_name
    cfg.train.use_wandb = args.use_wandb
    cfg.train.seed = args.seed
    cfg.train.eval_every_epoch = args.eval_every_epoch
    cfg.train.freeze_text_tower = args.freeze_text_tower
    cfg.train.freeze_vision_tower = args.freeze_vision_tower
    cfg.train.freeze_logit_scale = args.freeze_logit_scale
    cfg.train.use_amp = not args.no_amp

    set_seed(cfg.train.seed, deterministic=args.deterministic)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_enabled = device.type == "cuda" and cfg.train.use_amp
    scaler = GradScaler(enabled=amp_enabled)

    recon_enabled = cfg.train.train_mode == "superclip_recon" and cfg.train.lambda_recon > 0.0
    effective_variant = args.variant if recon_enabled else "baseline"

    print(f"Device: {device}")
    print(
        f"Run: {args.run_name} | requested_mode={args.train_mode} | effective_mode={cfg.train.train_mode} | "
        f"requested_variant={args.variant} | effective_variant={effective_variant} | "
        f"lambda_clip={cfg.train.lambda_clip} | lambda_tc={cfg.train.lambda_token_cls} | "
        f"lambda_recon={cfg.train.lambda_recon} | mask_ratio={args.mask_ratio}"
    )
    print(
        f"Seed / loader: seed={cfg.train.seed} | num_workers={cfg.data.num_workers} | "
        f"deterministic={args.deterministic} | amp={amp_enabled}"
    )
    print(
        f"Freeze flags: text={cfg.train.freeze_text_tower} | vision={cfg.train.freeze_vision_tower} | "
        f"logit_scale={cfg.train.freeze_logit_scale}"
    )
    print(
        f"Checkpoint strategy: {args.save_strategy} | keep_last_k={args.keep_last_k} | "
        f"save_optimizer={args.save_optimizer_state}"
    )
    print(f"Reconstruction enabled: {recon_enabled}")

    vocab_map = load_vocab(args.vocab_path)
    cfg.model.num_token_classes = len(vocab_map)
    print(f"Loaded vocab with {len(vocab_map)} token classes")

    model = SuperCLIPRecon(cfg).to(device)

    for p in model.recon_head.parameters():
        p.requires_grad = recon_enabled

    param_summary = {
        "clip_total": count_params(model.clip_model, trainable_only=False),
        "clip_trainable": count_params(model.clip_model, trainable_only=True),
        "token_cls_total": count_params(model.token_cls_head, trainable_only=False),
        "token_cls_trainable": count_params(model.token_cls_head, trainable_only=True),
        "recon_total": count_params(model.recon_head, trainable_only=False),
        "recon_trainable": count_params(model.recon_head, trainable_only=True),
        "model_total": count_params(model, trainable_only=False),
        "model_trainable": count_params(model, trainable_only=True),
    }
    print(f"Param summary: {param_summary}")

    train_dataset = COCOCaptionsDataset(
        root=cfg.data.coco_root,
        ann_file=cfg.data.train_ann,
        image_dir=cfg.data.train_images,
        transform=model.preprocess,
        tokenizer=model.tokenizer,
        base_seed=cfg.train.seed,
        deterministic_caption=args.deterministic,
    )

    def seed_worker(worker_id: int):
        worker_seed = cfg.train.seed + worker_id
        random.seed(worker_seed)
        np.random.seed(worker_seed)
        torch.manual_seed(worker_seed)

    loader_gen = torch.Generator()
    loader_gen.manual_seed(cfg.train.seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        num_workers=cfg.data.num_workers,
        pin_memory=cfg.data.pin_memory,
        drop_last=True,
        worker_init_fn=seed_worker if cfg.data.num_workers > 0 else None,
        generator=loader_gen,
        persistent_workers=(cfg.data.num_workers > 0),
    )

    if args.variant == "B" and recon_enabled:
        print("Variant B: phrases are extracted inline per caption.")
        if args.phrase_path is not None:
            print(f"[DEPRECATED] --phrase_path={args.phrase_path} is ignored by training.")
    elif args.variant == "B" and not recon_enabled:
        print("Variant B requested with lambda=0; reconstruction is disabled, so baseline path ignores variant-specific masking.")

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)

    steps_per_epoch = len(train_loader)
    total_steps = steps_per_epoch * cfg.train.epochs
    scheduler = build_warmup_scheduler(optimizer, cfg.train.warmup_steps, total_steps)
    print(f"Scheduler: linear warmup for {cfg.train.warmup_steps} steps, then linear decay over {total_steps} total steps")

    if cfg.train.use_wandb:
        import wandb
        wandb.init(
            project="superclip-recon",
            name=cfg.train.run_name,
            config={
                "requested_train_mode": args.train_mode,
                "effective_train_mode": cfg.train.train_mode,
                "requested_variant": args.variant,
                "effective_variant": effective_variant,
                "lambda_clip": cfg.train.lambda_clip,
                "lambda_token_cls": cfg.train.lambda_token_cls,
                "lambda_recon": cfg.train.lambda_recon,
                "mask_ratio": args.mask_ratio,
                "lr": args.lr,
                "batch_size": args.batch_size,
                "seed": args.seed,
                "num_workers": args.num_workers,
                "deterministic": args.deterministic,
                "amp_enabled": amp_enabled,
                "recon_enabled": recon_enabled,
            },
        )

    max_masks = int(cfg.data.max_caption_length * cfg.model.mask_ratio) + 1

    os.makedirs(cfg.train.save_dir, exist_ok=True)
    history = []
    global_step = 0
    best_metric_epoch = None
    best_metric_val = -1.0
    best_retrieval = {}
    ckpt_size_mb = None
    first_step_grad_summary = None

    for epoch in range(cfg.train.epochs):
        if hasattr(train_dataset, "set_epoch"):
            train_dataset.set_epoch(epoch + 1)

        model.train()
        epoch_losses = {"l_clip": 0.0, "l_token_cls": 0.0, "l_recon": 0.0, "l_total": 0.0}
        n_batches = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{cfg.train.epochs}")
        for batch_idx, (images, token_ids, captions_raw, img_ids) in enumerate(pbar):
            mask_targets = None
            if recon_enabled:
                if args.variant == "B":
                    _, mask_targets, _ = create_phrase_mask_from_captions(
                        token_ids,
                        captions_raw,
                        model.tokenizer,
                        max_masks,
                    )
                else:
                    _, mask_targets, _ = create_mask(token_ids, cfg.model.mask_ratio, max_masks)

            token_cls_labels = build_token_labels(token_ids, vocab_map, cfg.model.num_token_classes)

            images = images.to(device)
            token_ids = token_ids.to(device)
            token_cls_labels = token_cls_labels.to(device)
            if mask_targets is not None:
                mask_targets = mask_targets.to(device)

            optimizer.zero_grad(set_to_none=True)

            with autocast(enabled=amp_enabled) if amp_enabled else nullcontext():
                outputs = model(
                    images,
                    token_ids,
                    encode_text=True,
                    compute_recon=recon_enabled,
                )
                loss, losses_dict = total_loss(
                    train_mode=cfg.train.train_mode,
                    image_features=outputs["image_features"],
                    text_features=outputs["text_features"],
                    logit_scale=outputs["logit_scale"],
                    token_cls_logits=outputs["token_cls_logits"],
                    token_cls_labels=token_cls_labels,
                    recon_logits=outputs["recon_logits"],
                    mask_targets=mask_targets,
                    lambda_clip=cfg.train.lambda_clip,
                    lambda_token_cls=cfg.train.lambda_token_cls,
                    lambda_recon=cfg.train.lambda_recon,
                    token_cls_freq=outputs["token_cls_freq"],
                    token_cls_num_updates=outputs["token_cls_num_updates"],
                    token_cls_use_reweighting=cfg.model.token_cls_use_reweighting,
                )

            if amp_enabled:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                if first_step_grad_summary is None:
                    first_step_grad_summary = {
                        "clip_grad_norm": grad_norm(model.clip_model),
                        "token_cls_grad_norm": grad_norm(model.token_cls_head),
                        "recon_grad_norm": grad_norm(model.recon_head),
                    }
                    print(f"First-step grad summary: {first_step_grad_summary}")
                torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=cfg.train.grad_clip_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if first_step_grad_summary is None:
                    first_step_grad_summary = {
                        "clip_grad_norm": grad_norm(model.clip_model),
                        "token_cls_grad_norm": grad_norm(model.token_cls_head),
                        "recon_grad_norm": grad_norm(model.recon_head),
                    }
                    print(f"First-step grad summary: {first_step_grad_summary}")
                torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=cfg.train.grad_clip_norm)
                optimizer.step()

            scheduler.step()
            global_step += 1

            with torch.no_grad():
                if hasattr(model.clip_model, "logit_scale"):
                    model.clip_model.logit_scale.clamp_(0, math.log(100))

            for key in epoch_losses:
                epoch_losses[key] += losses_dict[key]
            n_batches += 1

            if batch_idx % cfg.train.log_every == 0:
                current_lr = scheduler.get_last_lr()[0]
                pbar.set_postfix(
                    {
                        "clip": f"{losses_dict['l_clip']:.4f}",
                        "tc": f"{losses_dict['l_token_cls']:.4f}",
                        "rc": f"{losses_dict['l_recon']:.4f}",
                        "tot": f"{losses_dict['l_total']:.4f}",
                        "lr": f"{current_lr:.2e}",
                    }
                )
                if cfg.train.use_wandb:
                    import wandb
                    wandb.log(
                        {
                            **losses_dict,
                            "lr": current_lr,
                            "recon_enabled": float(recon_enabled),
                        },
                        step=global_step,
                    )

        avg = {key: value / max(n_batches, 1) for key, value in epoch_losses.items()}
        print(
            f"Epoch {epoch + 1} avg losses: clip={avg['l_clip']:.4f}  "
            f"tc={avg['l_token_cls']:.4f}  rc={avg['l_recon']:.4f}  total={avg['l_total']:.4f}"
        )

        ckpt_data = {
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "config": cfg,
            "avg_losses": avg,
            "requested_train_mode": args.train_mode,
            "train_mode": cfg.train.train_mode,
            "requested_variant": args.variant,
            "effective_variant": effective_variant,
            "recon_enabled": recon_enabled,
            "param_summary": param_summary,
            "first_step_grad_summary": first_step_grad_summary,
            "best_metric_epoch": best_metric_epoch,
            "best_metric_val": best_metric_val,
        }
        if args.save_optimizer_state:
            ckpt_data["optimizer_state_dict"] = optimizer.state_dict()
            ckpt_data["scheduler_state_dict"] = scheduler.state_dict()
            if amp_enabled:
                ckpt_data["scaler_state_dict"] = scaler.state_dict()

        ckpt_path = os.path.join(cfg.train.save_dir, f"epoch_{epoch + 1}.pt")
        torch.save(ckpt_data, ckpt_path)
        ckpt_size_mb = os.path.getsize(ckpt_path) / (1024 * 1024)
        print(f"Saved checkpoint: {ckpt_path} ({ckpt_size_mb:.1f} MB)")

        epoch_record = {"epoch": epoch + 1, "losses": avg}
        if not args.skip_eval and (epoch + 1) % cfg.train.eval_every_epoch == 0:
            print(f"Running retrieval evaluation (max_images={args.eval_max_images})...")
            metrics = run_retrieval_eval(model, cfg, device, max_images=args.eval_max_images)
            score = retrieval_score(metrics)
            print(
                f"  I->T R@1={metrics['i2t_r1']:.2f}  R@5={metrics['i2t_r5']:.2f}  R@10={metrics['i2t_r10']:.2f}"
            )
            print(
                f"  T->I R@1={metrics['t2i_r1']:.2f}  R@5={metrics['t2i_r5']:.2f}  R@10={metrics['t2i_r10']:.2f}"
            )
            print(f"  Retrieval score (R-sum@1/5/10 both directions): {score:.2f}")
            epoch_record["retrieval"] = metrics
            epoch_record["retrieval_score"] = score

            if score > best_metric_val:
                best_metric_val = score
                best_metric_epoch = epoch + 1
                best_retrieval = metrics

            if cfg.train.use_wandb:
                import wandb
                wandb.log({**metrics, "retrieval_score": score}, step=(epoch + 1) * len(train_loader))

        history.append(epoch_record)

        manage_checkpoints(
            save_dir=cfg.train.save_dir,
            save_strategy=args.save_strategy,
            keep_last_k=args.keep_last_k,
            current_epoch=epoch + 1,
            best_metric_epoch=best_metric_epoch,
        )

    wall_time = time.time() - wall_start
    results = {
        "run_name": args.run_name,
        "requested_train_mode": args.train_mode,
        "train_mode": cfg.train.train_mode,
        "variant": args.variant,
        "effective_variant": effective_variant,
        "recon_enabled": recon_enabled,
        "lambda_clip": cfg.train.lambda_clip,
        "lambda_token_cls": cfg.train.lambda_token_cls,
        "lambda_recon": cfg.train.lambda_recon,
        "mask_ratio": args.mask_ratio,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "seed": args.seed,
        "num_workers": args.num_workers,
        "deterministic": args.deterministic,
        "amp_enabled": amp_enabled,
        "wall_time_seconds": round(wall_time, 1),
        "trainable_params": param_summary["model_trainable"],
        "param_summary": param_summary,
        "first_step_grad_summary": first_step_grad_summary,
        "save_strategy": args.save_strategy,
        "checkpoint_size_mb": round(ckpt_size_mb, 1) if ckpt_size_mb is not None else None,
        "best_epoch": best_metric_epoch,
        "best_retrieval_score": round(best_metric_val, 3) if best_metric_epoch is not None else None,
        "best_retrieval": best_retrieval,
        "hostname": socket.gethostname(),
        "git_commit": safe_git_commit(),
        "argv": sys.argv,
        "history": history,
        "final_retrieval": history[-1].get("retrieval", {}) if history else {},
    }

    if args.results_file:
        os.makedirs(os.path.dirname(args.results_file) or ".", exist_ok=True)
        out_path = args.results_file
    else:
        out_path = os.path.join(cfg.train.save_dir, "results.json")

    with open(out_path, "w") as handle:
        json.dump(results, handle, indent=2)
    print(f"Results saved to {out_path}")
    print(f"Training complete. Wall time: {wall_time / 60:.1f} min")


if __name__ == "__main__":
    train()