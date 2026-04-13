"""
Loss functions for SuperCLIP-Recon.

    L_total = L_token_cls + lambda * L_recon

- L_token_cls: multi-label binary cross-entropy (SuperCLIP baseline)
- L_recon:     cross-entropy on masked caption tokens (our contribution)
"""

import re
import random as _random

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─── Token Classification Loss (SuperCLIP baseline) ─────────────────────────

def build_token_labels(token_ids: torch.Tensor, vocab_map: dict, num_classes: int) -> torch.Tensor:
    """
    Convert tokenized captions into multi-hot label vectors.

    Uses a vectorized lookup table instead of Python loops for speed.
    On batch_size=128, seq_len=77 this is ~50x faster than the loop version.

    Args:
        token_ids:   [B, seq_len] CLIP token IDs
        vocab_map:   dict mapping token_id -> class_index (for top-K vocabulary)
        num_classes: number of token classes

    Returns:
        labels: [B, num_classes] multi-hot float tensor
    """
    B, seq_len = token_ids.shape
    device = token_ids.device

    labels = torch.zeros(B, num_classes, device=device)

    if not vocab_map:
        return labels

    # Build lookup table: token_id → class_index (or -1 if not in vocab)
    # This is O(|vocab|) ≈ 1000 iterations — negligible
    max_tid = max(vocab_map.keys()) + 1
    lookup = torch.full((max_tid,), -1, dtype=torch.long, device=device)
    for tid, cidx in vocab_map.items():
        lookup[tid] = cidx

    # Clamp token IDs to valid range for indexing (out-of-range → index 0, filtered below)
    safe_ids = token_ids.clamp(0, max_tid - 1)

    # Map every token to its class index (or -1)
    cls_idx = lookup[safe_ids]                               # [B, seq_len]

    # Valid = token was in range AND mapped to a real class
    valid = (token_ids < max_tid) & (token_ids >= 0) & (cls_idx >= 0)  # [B, seq_len]

    # Scatter into labels: labels[b, cls_idx[b,j]] = 1.0 for all valid (b,j)
    batch_idx = torch.arange(B, device=device).unsqueeze(1).expand_as(cls_idx)
    b_flat = batch_idx[valid]
    c_flat = cls_idx[valid]
    labels[b_flat, c_flat] = 1.0

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


# ─── Variant B improved: per-caption phrase extraction (Fix 3) ───────────────

# Lightweight regex pattern for noun phrases: (det? adj* noun+), 2-5 words.
# This is intentionally simple — spaCy is too heavy to run per-batch.
_DET = r"(?:a|an|the|this|that|some|two|three|four|five|six|many|several|few|no|my|his|her|their|its)"
_NP_PATTERN = re.compile(
    rf"\b({_DET}\s+\w+(?:\s+\w+){{0,3}})\b",
    re.IGNORECASE,
)


def _extract_phrases_from_caption(caption: str):
    """
    Quick regex extraction of 2-5 word noun-phrase candidates from a single
    caption string.  Returns a list of lowercase phrase strings.
    """
    phrases = []
    for m in _NP_PATTERN.finditer(caption.lower()):
        phrase = " ".join(m.group(1).split())  # normalize whitespace
        n_words = len(phrase.split())
        if 2 <= n_words <= 5:
            phrases.append(phrase)
    return phrases


def create_phrase_mask_from_captions(
    token_ids: torch.Tensor,
    captions_raw: list,
    tokenizer,
    max_masks: int,
):
    """
    Variant B masking that extracts phrases directly from the current batch
    captions, guaranteeing that each phrase actually appears in the
    corresponding token sequence.

    This fixes the alignment problem where pre-extracted image-level phrases
    may come from a different caption than the one sampled at training time.

    Args:
        token_ids:    [B, seq_len] CLIP token IDs (on CPU)
        captions_raw: list[str] of length B — the raw captions for this batch
        tokenizer:    open_clip tokenizer callable
        max_masks:    maximum number of masked positions

    Returns:
        masked_token_ids: [B, seq_len]
        mask_targets:     [B, max_masks]
        mask_positions:   [B, max_masks]
    """
    B, seq_len = token_ids.shape
    device = token_ids.device

    masked_token_ids = token_ids.clone()
    mask_targets = torch.zeros(B, max_masks, dtype=torch.long, device=device)
    mask_positions = torch.full((B, max_masks), -1, dtype=torch.long, device=device)

    for i in range(B):
        caption = captions_raw[i] if isinstance(captions_raw[i], str) else captions_raw[i]
        phrases = _extract_phrases_from_caption(caption)
        caption_toks = token_ids[i].tolist()

        masked_something = False

        if phrases:
            # Shuffle so we don't always pick the first phrase
            _random.shuffle(phrases)

            for phrase in phrases:
                # Tokenize the phrase and strip special tokens
                phrase_tok_ids = tokenizer(phrase).squeeze(0).tolist()
                content_toks = [t for t in phrase_tok_ids
                                if t not in (SOT_TOKEN, EOT_TOKEN, PAD_TOKEN)]

                if not content_toks or len(content_toks) > max_masks:
                    continue

                span_start = _find_sublist(caption_toks, content_toks)
                if span_start == -1:
                    continue

                # Mask the whole phrase span
                n_phrase_toks = min(len(content_toks), max_masks)
                for k in range(n_phrase_toks):
                    pos = span_start + k
                    mask_targets[i, k] = token_ids[i, pos]
                    mask_positions[i, k] = pos
                    masked_token_ids[i, pos] = PAD_TOKEN
                masked_something = True
                break  # one phrase per sample is enough

        if not masked_something:
            # Fallback: mask ~25% of content tokens (same as original fallback)
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