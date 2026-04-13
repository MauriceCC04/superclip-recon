"""
Short smoke test for SuperCLIP-Recon on real COCO paths.

Purpose:
    - verify the conda environment works on a compute node
    - verify OpenCLIP weights are already cached and load correctly
    - verify a few real training steps run without error
    - save a checkpoint and quick retrieval metrics

This is intentionally much smaller than train.py so it can be used as the
first SLURM job before launching the full study.
"""

# ── Ensure repo root is importable when launched as `python tests/smoke_test.py` ──
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

from config import Config
from dataset import COCOCaptionsDataset
from model import SuperCLIPRecon
from losses import build_token_labels, create_mask, total_loss
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
    parser.add_argument("--steps", type=int, default=3,
                        help="Number of optimizer steps to run")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--mask_ratio", type=float, default=0.15)
    parser.add_argument("--lambda_recon", type=float, default=0.5)
    parser.add_argument("--eval_images", type=int, default=64,
                        help="Small retrieval eval subset for speed")
    parser.add_argument("--save_path", type=str,
                        default="./checkpoints/smoke/smoke_step.pt")
    parser.add_argument("--results_file", type=str,
                        default="./results/smoke/smoke_results.json")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    wall_start = time.time()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    cfg = Config()
    cfg.data.coco_root = args.coco_root
    cfg.data.num_workers = 0
    cfg.model.mask_ratio = args.mask_ratio
    cfg.train.lambda_recon = args.lambda_recon
    cfg.train.batch_size = args.batch_size
    cfg.train.lr = args.lr
    cfg.train.seed = args.seed
    cfg.train.run_name = "smoke_test"
    cfg.train.save_dir = os.path.dirname(args.save_path) or "."

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

    vocab_map = load_vocab(args.vocab_path)
    # Sync head size to actual vocab
    cfg.model.num_token_classes = len(vocab_map)
    print(f"Loaded vocab with {len(vocab_map)} token classes")

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=cfg.train.lr,
        weight_decay=cfg.train.weight_decay,
    )

    max_masks = int(cfg.data.max_caption_length * cfg.model.mask_ratio) + 1

    os.makedirs(os.path.dirname(args.save_path) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(args.results_file) or ".", exist_ok=True)

    model.train()
    step_logs = []
    steps_run = 0
    last_loss_dict = None

    for batch_idx, (images, token_ids, captions_raw, img_ids) in enumerate(train_loader):
        if steps_run >= args.steps:
            break

        images = images.to(device)
        token_ids = token_ids.to(device)

        _, mask_targets, _ = create_mask(token_ids, cfg.model.mask_ratio, max_masks)
        outputs = model(images, token_ids)

        token_cls_labels = build_token_labels(
            token_ids, vocab_map, cfg.model.num_token_classes
        )

        loss, loss_dict = total_loss(
            token_cls_logits=outputs["token_cls_logits"],
            token_cls_labels=token_cls_labels,
            recon_logits=outputs["recon_logits"],
            mask_targets=mask_targets,
            lambda_recon=cfg.train.lambda_recon,
        )

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
        optimizer.step()

        steps_run += 1
        last_loss_dict = loss_dict
        step_record = {
            "step": steps_run,
            "l_token_cls": loss_dict["l_token_cls"],
            "l_recon": loss_dict["l_recon"],
            "l_total": loss_dict["l_total"],
        }
        step_logs.append(step_record)
        print(
            f"[step {steps_run}/{args.steps}] "
            f"tc={loss_dict['l_token_cls']:.4f} "
            f"rc={loss_dict['l_recon']:.4f} "
            f"tot={loss_dict['l_total']:.4f}"
        )

    if steps_run == 0:
        raise RuntimeError("Smoke test ran zero optimizer steps.")

    torch.save(
        {
            "smoke_steps": steps_run,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": cfg,
            "last_losses": last_loss_dict,
        },
        args.save_path,
    )
    print(f"Saved smoke checkpoint to {args.save_path}")

    print("Running quick retrieval evaluation...")
    metrics = run_retrieval_eval(model, cfg, device, max_images=args.eval_images)
    print(
        f"I->T: R@1={metrics['i2t_r1']:.2f} "
        f"R@5={metrics['i2t_r5']:.2f} "
        f"R@10={metrics['i2t_r10']:.2f}"
    )
    print(
        f"T->I: R@1={metrics['t2i_r1']:.2f} "
        f"R@5={metrics['t2i_r5']:.2f} "
        f"R@10={metrics['t2i_r10']:.2f}"
    )

    wall_time = time.time() - wall_start
    results = {
        "run_name": "smoke_test",
        "steps": steps_run,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "mask_ratio": args.mask_ratio,
        "lambda_recon": args.lambda_recon,
        "seed": args.seed,
        "trainable_params": n_trainable,
        "wall_time_seconds": round(wall_time, 1),
        "step_logs": step_logs,
        "retrieval": metrics,
        "checkpoint": args.save_path,
    }

    with open(args.results_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved smoke results to {args.results_file}")
    print(f"Smoke test complete. Wall time: {wall_time/60:.1f} min")


if __name__ == "__main__":
    main()