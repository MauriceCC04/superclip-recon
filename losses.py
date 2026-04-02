"""
Loss functions for SuperCLIP-Recon.

    L_total = L_token_cls + lambda * L_recon

- L_token_cls: multi-label binary cross-entropy (SuperCLIP baseline)
- L_recon:     cross-entropy on masked caption tokens (our contribution)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─── Token Classification Loss (SuperCLIP baseline) ─────────────────────────

def build_token_labels(token_ids: torch.Tensor, vocab_map: dict, num_classes: int) -> torch.Tensor:
    """
    Convert tokenized captions into multi-hot label vectors.

    Args:
        token_ids:   [B, seq_len] CLIP token IDs
        vocab_map:   dict mapping token_id -> class_index (for top-K vocabulary)
        num_classes: number of token classes

    Returns:
        labels: [B, num_classes] multi-hot float tensor
    """
    B = token_ids.size(0)
    labels = torch.zeros(B, num_classes, device=token_ids.device)
    for i in range(B):
        for tid in token_ids[i].tolist():
            if tid in vocab_map:
                labels[i, vocab_map[tid]] = 1.0
    return labels


def token_classification_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """
    Multi-label BCE loss for token classification.

    Args:
        logits: [B, num_classes] raw logits
        labels: [B, num_classes] multi-hot targets
    Returns:
        scalar loss
    """
    return F.binary_cross_entropy_with_logits(logits, labels)


# ─── Masking Utilities ───────────────────────────────────────────────────────

# CLIP special tokens (do not mask these)
SOT_TOKEN = 49406  # <start_of_text>
EOT_TOKEN = 49407  # <end_of_text>
PAD_TOKEN = 0


def create_mask(token_ids: torch.Tensor, mask_ratio: float, max_masks: int):
    """
    Randomly mask content tokens in a caption.

    Args:
        token_ids:  [B, seq_len] CLIP token IDs
        mask_ratio: fraction of content tokens to mask
        max_masks:  maximum number of masked positions

    Returns:
        masked_token_ids: [B, seq_len] with some tokens replaced by 0
        mask_targets:     [B, max_masks] ground-truth token IDs at masked positions
        mask_positions:   [B, max_masks] positions in seq that were masked (-1 = pad)
    """
    B, seq_len = token_ids.shape
    device = token_ids.device

    masked_token_ids = token_ids.clone()
    mask_targets = torch.zeros(B, max_masks, dtype=torch.long, device=device)
    mask_positions = torch.full((B, max_masks), -1, dtype=torch.long, device=device)

    for i in range(B):
        # Find content token positions (exclude SOT, EOT, PAD)
        content_pos = []
        for j in range(seq_len):
            tid = token_ids[i, j].item()
            if tid not in (SOT_TOKEN, EOT_TOKEN, PAD_TOKEN):
                content_pos.append(j)

        if len(content_pos) == 0:
            continue

        # Determine how many to mask
        n_mask = max(1, min(int(len(content_pos) * mask_ratio), max_masks))

        # Random selection
        perm = torch.randperm(len(content_pos))[:n_mask]
        selected = [content_pos[p] for p in perm]

        for k, pos in enumerate(selected):
            mask_targets[i, k] = token_ids[i, pos]
            mask_positions[i, k] = pos
            masked_token_ids[i, pos] = PAD_TOKEN  # mask by zeroing

    return masked_token_ids, mask_targets, mask_positions


# ─── Variant B: Phrase Masking ────────────────────────────────────────────────

def create_phrase_mask(token_ids: torch.Tensor, phrase_data: dict, image_ids: list,
                       max_masks: int):
    """
    Mask an entire noun phrase span in each caption (Variant B).

    Instead of masking random tokens, we mask a contiguous phrase that was
    pre-extracted from the caption. This forces the reconstruction head to
    recover compositional structure (e.g., "a red car"), not just isolated tokens.

    Args:
        token_ids:   [B, seq_len] CLIP token IDs
        phrase_data: dict {str(image_id): [{"phrase": ..., "token_ids": [...]}, ...]}
        image_ids:   list of image IDs for this batch (length B)
        max_masks:   maximum number of masked positions

    Returns:
        masked_token_ids: [B, seq_len]
        mask_targets:     [B, max_masks]
        mask_positions:   [B, max_masks]
    """
    import random as _random

    B, seq_len = token_ids.shape
    device = token_ids.device

    masked_token_ids = token_ids.clone()
    mask_targets = torch.zeros(B, max_masks, dtype=torch.long, device=device)
    mask_positions = torch.full((B, max_masks), -1, dtype=torch.long, device=device)

    for i in range(B):
        img_id_str = str(image_ids[i])
        phrases = phrase_data.get(img_id_str, [])

        if not phrases:
            # Fallback to random masking if no phrases for this image
            content_pos = [j for j in range(seq_len)
                           if token_ids[i, j].item() not in (SOT_TOKEN, EOT_TOKEN, PAD_TOKEN)]
            if not content_pos:
                continue
            n_mask = max(1, min(len(content_pos) // 4, max_masks))
            perm = torch.randperm(len(content_pos))[:n_mask]
            for k, p_idx in enumerate(perm):
                pos = content_pos[p_idx]
                mask_targets[i, k] = token_ids[i, pos]
                mask_positions[i, k] = pos
                masked_token_ids[i, pos] = PAD_TOKEN
            continue

        # Pick a random phrase for this sample
        phrase_entry = _random.choice(phrases)
        phrase_toks = phrase_entry["token_ids"]

        # Find the phrase span in the tokenized caption
        caption_toks = token_ids[i].tolist()
        span_start = _find_sublist(caption_toks, phrase_toks)

        if span_start == -1:
            # Phrase not found in this specific tokenization — fallback to first token
            content_pos = [j for j in range(seq_len)
                           if token_ids[i, j].item() not in (SOT_TOKEN, EOT_TOKEN, PAD_TOKEN)]
            if content_pos:
                pos = content_pos[0]
                mask_targets[i, 0] = token_ids[i, pos]
                mask_positions[i, 0] = pos
                masked_token_ids[i, pos] = PAD_TOKEN
            continue

        # Mask the entire phrase span
        n_phrase_toks = min(len(phrase_toks), max_masks)
        for k in range(n_phrase_toks):
            pos = span_start + k
            mask_targets[i, k] = token_ids[i, pos]
            mask_positions[i, k] = pos
            masked_token_ids[i, pos] = PAD_TOKEN

    return masked_token_ids, mask_targets, mask_positions


def _find_sublist(haystack: list, needle: list) -> int:
    """Find starting index of needle in haystack, or -1."""
    for i in range(len(haystack) - len(needle) + 1):
        if haystack[i:i + len(needle)] == needle:
            return i
    return -1


# ─── Reconstruction Loss (our contribution) ─────────────────────────────────

def reconstruction_loss(recon_logits: torch.Tensor, mask_targets: torch.Tensor) -> torch.Tensor:
    """
    Cross-entropy loss on masked token predictions.

    Args:
        recon_logits: [B, max_masks, vocab_size]
        mask_targets: [B, max_masks] ground-truth token IDs (0 = ignore)

    Returns:
        scalar loss (averaged over non-padding positions)
    """
    B, M, V = recon_logits.shape

    # Flatten for cross-entropy
    logits_flat = recon_logits.view(B * M, V)
    targets_flat = mask_targets.view(B * M)

    # Only compute loss on non-padding positions (target != 0)
    valid = targets_flat != 0
    if valid.sum() == 0:
        return torch.tensor(0.0, device=recon_logits.device, requires_grad=True)

    loss = F.cross_entropy(logits_flat[valid], targets_flat[valid])
    return loss


# ─── Combined Loss ───────────────────────────────────────────────────────────

def total_loss(token_cls_logits, token_cls_labels, recon_logits, mask_targets, lambda_recon):
    """
    L_total = L_token_cls + lambda * L_recon

    Returns:
        total: scalar
        losses_dict: {l_token_cls, l_recon, l_total} for logging
    """
    l_tc = token_classification_loss(token_cls_logits, token_cls_labels)
    l_recon = reconstruction_loss(recon_logits, mask_targets)
    l_total = l_tc + lambda_recon * l_recon

    return l_total, {
        "l_token_cls": l_tc.item(),
        "l_recon": l_recon.item(),
        "l_total": l_total.item(),
    }
