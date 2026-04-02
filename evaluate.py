"""
Evaluation for SuperCLIP-Recon.

Primary metric: COCO image-text retrieval (R@1, R@5, R@10).
Uses the COCO val split (5K images, 5 captions each).
"""

import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import COCOCaptionsDataset


def compute_retrieval_metrics(image_embs, text_embs, ks=(1, 5, 10)):
    """
    Compute image→text and text→image retrieval recall.

    For COCO: each image has 5 captions. We evaluate on the standard
    5K val set where text_embs has 5x the rows of image_embs.

    Args:
        image_embs: [N, D] numpy array
        text_embs:  [N*5, D] numpy array (5 captions per image)
        ks: tuple of recall@K values

    Returns:
        dict with i2t_r1, i2t_r5, i2t_r10, t2i_r1, t2i_r5, t2i_r10
    """
    N = image_embs.shape[0]
    n_captions_per_image = text_embs.shape[0] // N  # typically 5

    # Similarity matrix: [N, N*5]
    sims = image_embs @ text_embs.T

    # --- Image → Text retrieval ---
    i2t_ranks = []
    for i in range(N):
        # Ground truth caption indices for image i
        gt_indices = list(range(i * n_captions_per_image, (i + 1) * n_captions_per_image))
        # Rank all texts by similarity
        sorted_indices = np.argsort(-sims[i])
        # Find best rank among ground truth captions
        rank = min(np.where(np.isin(sorted_indices, gt_indices))[0])
        i2t_ranks.append(rank)

    # --- Text → Image retrieval ---
    t2i_ranks = []
    sims_t2i = text_embs @ image_embs.T  # [N*5, N]
    for j in range(text_embs.shape[0]):
        gt_image = j // n_captions_per_image
        sorted_indices = np.argsort(-sims_t2i[j])
        rank = np.where(sorted_indices == gt_image)[0][0]
        t2i_ranks.append(rank)

    i2t_ranks = np.array(i2t_ranks)
    t2i_ranks = np.array(t2i_ranks)

    metrics = {}
    for k in ks:
        metrics[f"i2t_r{k}"] = float((i2t_ranks < k).mean() * 100)
        metrics[f"t2i_r{k}"] = float((t2i_ranks < k).mean() * 100)

    return metrics


@torch.no_grad()
def run_retrieval_eval(model, cfg, device, max_images=5000):
    """
    Run COCO retrieval evaluation on val set.

    Args:
        model: SuperCLIPRecon model
        cfg: Config object
        device: torch device
        max_images: cap on number of images (COCO val has 5K)

    Returns:
        dict of retrieval metrics
    """
    model.eval()

    val_dataset = COCOCaptionsDataset(
        root=cfg.data.coco_root,
        ann_file=cfg.data.val_ann,
        image_dir=cfg.data.val_images,
        transform=model.preprocess,
        tokenizer=model.tokenizer,
    )

    # For retrieval we need all 5 captions per image.
    # We'll iterate manually to collect them.
    import json, os
    ann_path = os.path.join(cfg.data.coco_root, cfg.data.val_ann)
    with open(ann_path) as f:
        val_data = json.load(f)

    img_id_to_caps = {}
    for ann in val_data["annotations"]:
        iid = ann["image_id"]
        if iid not in img_id_to_caps:
            img_id_to_caps[iid] = []
        img_id_to_caps[iid].append(ann["caption"])

    # Use the same image ordering as the dataset
    image_ids = val_dataset.image_ids[:max_images]

    # Collect image embeddings
    all_image_embs = []
    all_text_embs = []

    loader = DataLoader(val_dataset, batch_size=64, shuffle=False,
                        num_workers=cfg.data.num_workers)

    n_collected = 0
    for images, token_ids, _, _ in tqdm(loader, desc="Eval (images)"):
        if n_collected >= max_images:
            break
        images = images.to(device)
        img_emb = model.encode_image(images)
        all_image_embs.append(img_emb.cpu().numpy())
        n_collected += images.size(0)

    all_image_embs = np.concatenate(all_image_embs, axis=0)[:max_images]

    # Collect text embeddings (all 5 captions per image)
    for img_id in tqdm(image_ids, desc="Eval (texts)"):
        captions = img_id_to_caps.get(img_id, [])[:5]
        for cap in captions:
            tok = model.tokenizer(cap).to(device)
            txt_emb = model.encode_text(tok)
            all_text_embs.append(txt_emb.cpu().numpy())

    all_text_embs = np.concatenate(all_text_embs, axis=0)

    # Compute metrics
    metrics = compute_retrieval_metrics(all_image_embs, all_text_embs)
    model.train()
    return metrics


if __name__ == "__main__":
    """Quick test: evaluate a checkpoint."""
    import argparse
    from config import Config
    from model import SuperCLIPRecon

    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--coco_root", type=str, default="./data/coco")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(args.checkpoint, map_location=device)
    cfg = ckpt["config"]
    cfg.data.coco_root = args.coco_root

    model = SuperCLIPRecon(cfg).to(device)
    model.load_state_dict(ckpt["model_state_dict"])

    metrics = run_retrieval_eval(model, cfg, device)
    print("Retrieval results:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.2f}")
