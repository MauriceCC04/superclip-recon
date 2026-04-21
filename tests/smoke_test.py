"""
Short smoke test for SuperCLIP-Recon on real COCO paths.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import time
import argparse
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from contextlib import nullcontext

from config import Config
from dataset import COCOCaptionsDataset
from model import SuperCLIPRecon
from losses import build_token_labels, create_mask, create_phrase_mask_from_captions, total_loss, resolve_train_mode
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
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--mask_ratio", type=float, default=0.15)
    parser.add_argument("--lambda_recon", type=float, default=0.5)
    parser.add_argument("--lambda_clip", type=float, default=1.0)
    parser.add_argument("--lambda_token_cls", type=float, default=1.0)
    parser.add_argument("--train_mode", type=str, default="auto", choices=["auto", "clip_only", "superclip_baseline", "superclip_recon"])
    parser.add_argument("--variant", type=str, default="A", choices=["A", "B"])
    parser.add_argument("--eval_images", type=int, default=64)
    parser.add_argument("--save_path", type=str, default="./checkpoints/smoke/smoke_step.pt")
    parser.add_argument("--results_file", type=str, default="./results/smoke/smoke_results.json")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no_amp", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    wall_start = time.time()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_enabled = device.type == "cuda" and not args.no_amp
    scaler = GradScaler(enabled=amp_enabled)
    print(f"Device: {device}")

    cfg = Config()
    cfg.data.coco_root = args.coco_root
    cfg.data.num_workers = 0
    cfg.model.mask_ratio = args.mask_ratio
    cfg.model.variant = args.variant
    cfg.train.lambda_clip = args.lambda_clip
    cfg.train.lambda_token_cls = args.lambda_token_cls
    cfg.train.lambda_recon = args.lambda_recon
    cfg.train.train_mode = resolve_train_mode(args.train_mode, args.lambda_recon)
    cfg.train.batch_size = args.batch_size
    cfg.train.lr = args.lr
    cfg.train.seed = args.seed
    cfg.train.run_name = "smoke_test"
    cfg.train.save_dir = os.path.dirname(args.save_path) or "."
    cfg.train.use_amp = amp_enabled

    vocab_map = load_vocab(args.vocab_path)
    cfg.model.num_token_classes = len(vocab_map)
    print(f"Loaded vocab with {len(vocab_map)} token classes")

    model = SuperCLIPRecon(cfg).to(device)
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable params: {n_trainable:,}")

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
        drop_last=False,
    )

    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
    max_masks = int(cfg.data.max_caption_length * cfg.model.mask_ratio) + 1

    os.makedirs(os.path.dirname(args.save_path) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(args.results_file) or ".", exist_ok=True)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    model.train()
    step_logs = []
    steps_run = 0
    last_loss_dict = None

    for images, token_ids, captions_raw, img_ids in train_loader:
        if steps_run >= args.steps:
            break

        if args.variant == "B":
            _, mask_targets, _ = create_phrase_mask_from_captions(token_ids, captions_raw, model.tokenizer, max_masks)
        else:
            _, mask_targets, _ = create_mask(token_ids, cfg.model.mask_ratio, max_masks)
        token_cls_labels = build_token_labels(token_ids, vocab_map, cfg.model.num_token_classes)

        images = images.to(device)
        token_ids = token_ids.to(device)
        mask_targets = mask_targets.to(device)
        token_cls_labels = token_cls_labels.to(device)

        optimizer.zero_grad(set_to_none=True)
        amp_ctx = autocast if amp_enabled else nullcontext
        with amp_ctx(enabled=amp_enabled) if amp_enabled else amp_ctx():
            outputs = model(images, token_ids, encode_text=True)
            loss, loss_dict = total_loss(
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
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        steps_run += 1
        last_loss_dict = loss_dict
        step_record = {
            "step": steps_run,
            "l_clip": loss_dict["l_clip"],
            "l_token_cls": loss_dict["l_token_cls"],
            "l_recon": loss_dict["l_recon"],
            "l_total": loss_dict["l_total"],
        }
        step_logs.append(step_record)
        print(
            f"[step {steps_run}/{args.steps}] clip={loss_dict['l_clip']:.4f} "
            f"tc={loss_dict['l_token_cls']:.4f} rc={loss_dict['l_recon']:.4f} "
            f"tot={loss_dict['l_total']:.4f}"
        )

    if steps_run == 0:
        raise RuntimeError("Smoke test ran zero optimizer steps.")

    torch.save({"smoke_steps": steps_run, "model_state_dict": model.state_dict(), "config": cfg, "last_losses": last_loss_dict}, args.save_path)
    ckpt_size_mb = os.path.getsize(args.save_path) / (1024 * 1024)
    print(f"Saved smoke checkpoint to {args.save_path} ({ckpt_size_mb:.1f} MB)")

    eval_start = time.time()
    print(f"Running quick retrieval evaluation (max_images={args.eval_images})...")
    metrics = run_retrieval_eval(model, cfg, device, max_images=args.eval_images)
    eval_seconds = time.time() - eval_start
    print(f"I->T: R@1={metrics['i2t_r1']:.2f} R@5={metrics['i2t_r5']:.2f} R@10={metrics['i2t_r10']:.2f}")
    print(f"T->I: R@1={metrics['t2i_r1']:.2f} R@5={metrics['t2i_r5']:.2f} R@10={metrics['t2i_r10']:.2f}")

    gpu_peak_gb = None
    if device.type == "cuda":
        gpu_peak_gb = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
        print(f"GPU peak memory: {gpu_peak_gb:.2f} GB")

    wall_time = time.time() - wall_start
    results = {
        "run_name": "smoke_test",
        "train_mode": cfg.train.train_mode,
        "variant": args.variant,
        "steps": steps_run,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "mask_ratio": args.mask_ratio,
        "lambda_clip": args.lambda_clip,
        "lambda_token_cls": args.lambda_token_cls,
        "lambda_recon": args.lambda_recon,
        "seed": args.seed,
        "trainable_params": n_trainable,
        "wall_time_seconds": round(wall_time, 1),
        "eval_seconds": round(eval_seconds, 1),
        "gpu_peak_mem_gb": round(gpu_peak_gb, 3) if gpu_peak_gb is not None else None,
        "checkpoint_size_mb": round(ckpt_size_mb, 1),
        "step_logs": step_logs,
        "retrieval": metrics,
        "checkpoint": args.save_path,
    }

    with open(args.results_file, "w") as handle:
        json.dump(results, handle, indent=2)
    print(f"Saved smoke results to {args.results_file}")
    print(f"Smoke test complete. Wall time: {wall_time / 60:.1f} min")


if __name__ == "__main__":
    main()
