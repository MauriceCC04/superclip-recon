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

import os
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

    main()