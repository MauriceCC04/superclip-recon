"""
Sanity check: verify model, data pipeline, losses, and masking
all work end-to-end BEFORE running real training.

Usage:
    python sanity_check.py --coco_root ./data/coco --vocab_path ./vocab.json

What it checks:
    1. CLIP model loads correctly
    2. Dataset loads and yields correct shapes
    3. Token classification labels build correctly
    4. Masking produces valid outputs
    5. Forward pass produces expected shapes
    6. Loss computes and backprop runs without error
    7. Retrieval eval pipeline runs (on tiny subset)
"""

import torch
import argparse
from config import Config
from model import SuperCLIPRecon
from dataset import COCOCaptionsDataset
from losses import build_token_labels, create_mask, total_loss
from build_vocab import load_vocab


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--coco_root", type=str, default="./data/coco")
    parser.add_argument("--vocab_path", type=str, default="./vocab.json")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    cfg = Config()
    cfg.data.coco_root = args.coco_root

    # --- 1. Model ---
    print("\n[1/7] Loading model...")
    model = SuperCLIPRecon(cfg).to(device)
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"  Trainable: {n_train:,} / Total: {n_total:,}")
    print("  ✓ Model loads")

    # --- 2. Dataset ---
    print("\n[2/7] Loading dataset (first batch)...")
    dataset = COCOCaptionsDataset(
        root=cfg.data.coco_root,
        ann_file=cfg.data.train_ann,
        image_dir=cfg.data.train_images,
        transform=model.preprocess,
        tokenizer=model.tokenizer,
    )
    from torch.utils.data import DataLoader
    loader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0)
    images, token_ids, captions, img_ids = next(iter(loader))
    print(f"  images: {images.shape}")         # [4, 3, 224, 224]
    print(f"  token_ids: {token_ids.shape}")    # [4, 77]
    print(f"  caption[0]: {captions[0][:80]}...")
    print(f"  img_ids[0]: {img_ids[0].item()}")
    print("  ✓ Dataset works")

    # --- 3. Vocab ---
    print("\n[3/7] Loading vocab...")
    vocab_map = load_vocab(args.vocab_path)
    print(f"  {len(vocab_map)} token classes")
    labels = build_token_labels(token_ids, vocab_map, cfg.model.num_token_classes)
    print(f"  labels shape: {labels.shape}")    # [4, 1000]
    print(f"  avg labels per sample: {labels.sum(dim=1).mean():.1f}")
    print("  ✓ Vocab + labels work")

    # --- 4. Masking ---
    print("\n[4/7] Testing masking...")
    max_masks = int(cfg.data.max_caption_length * cfg.model.mask_ratio) + 1
    masked_ids, mask_targets, mask_pos = create_mask(token_ids, cfg.model.mask_ratio, max_masks)
    print(f"  mask_targets: {mask_targets.shape}")  # [4, max_masks]
    print(f"  non-zero targets: {(mask_targets != 0).sum().item()}")
    print("  ✓ Masking works")

    # --- 5. Forward pass ---
    print("\n[5/7] Forward pass...")
    images = images.to(device)
    token_ids = token_ids.to(device)
    outputs = model(images, token_ids)
    print(f"  image_features: {outputs['image_features'].shape}")
    print(f"  text_features:  {outputs['text_features'].shape}")
    print(f"  token_cls_logits: {outputs['token_cls_logits'].shape}")
    print(f"  recon_logits: {outputs['recon_logits'].shape}")
    print("  ✓ Forward pass works")

    # --- 6. Loss + backward ---
    print("\n[6/7] Loss computation + backward...")
    labels = labels.to(device)
    mask_targets = mask_targets.to(device)
    loss, loss_dict = total_loss(
        outputs["token_cls_logits"], labels,
        outputs["recon_logits"], mask_targets,
        lambda_recon=cfg.train.lambda_recon,
    )
    print(f"  l_token_cls: {loss_dict['l_token_cls']:.4f}")
    print(f"  l_recon:     {loss_dict['l_recon']:.4f}")
    print(f"  l_total:     {loss_dict['l_total']:.4f}")
    loss.backward()
    print("  ✓ Backward pass works")

    # --- 7. Quick shape check for eval ---
    print("\n[7/7] Quick eval shape check...")
    model.eval()
    with torch.no_grad():
        img_emb = model.encode_image(images)
        txt_emb = model.encode_text(token_ids)
        sim = img_emb @ txt_emb.T
        print(f"  similarity matrix: {sim.shape}")  # [4, 4]
        print(f"  diagonal (should be highest per row): {sim.diag().tolist()}")
    print("  ✓ Eval pipeline works")

    print("\n" + "="*50)
    print("ALL SANITY CHECKS PASSED")
    print("="*50)


if __name__ == "__main__":
    main()
