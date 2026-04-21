"""
Loss functions for SuperCLIP-Recon.

Baseline modes:
    - clip_only:          L = L_clip
    - superclip_baseline: L = L_clip + alpha * L_token_cls
    - superclip_recon:    L = L_clip + alpha * L_token_cls + lambda * L_recon

This restores the missing CLIP alignment loss so lambda=0 is a valid
SuperCLIP-style baseline rather than an image-only bag-of-words objective.
"""

import re
import random as _random

import torch
import torch.nn.functional as F


SOT_TOKEN = 49406
EOT_TOKEN = 49407
PAD_TOKEN = 0


def resolve_train_mode(train_mode: str, lambda_recon: float) -> str:
    if train_mode == "auto":
        return "superclip_recon" if lambda_recon > 0 else "superclip_baseline"
    return train_mode


def build_token_labels(token_ids: torch.Tensor, vocab_map: dict, num_classes: int) -> torch.Tensor:
    """Convert tokenized captions into multi-hot label vectors."""
    batch_size, _ = token_ids.shape
    device = token_ids.device
    labels = torch.zeros(batch_size, num_classes, device=device)

    if not vocab_map:
        return labels

    max_tid = max(vocab_map.keys()) + 1
    lookup = torch.full((max_tid,), -1, dtype=torch.long, device=device)
    for tid, cidx in vocab_map.items():
        lookup[tid] = cidx

    safe_ids = token_ids.clamp(0, max_tid - 1)
    cls_idx = lookup[safe_ids]
    valid = (token_ids < max_tid) & (token_ids >= 0) & (cls_idx >= 0)
    batch_idx = torch.arange(batch_size, device=device).unsqueeze(1).expand_as(cls_idx)
    labels[batch_idx[valid], cls_idx[valid]] = 1.0
    return labels


def contrastive_clip_loss(image_features: torch.Tensor, text_features: torch.Tensor, logit_scale: torch.Tensor) -> torch.Tensor:
    if image_features is None or text_features is None or logit_scale is None:
        raise ValueError("contrastive_clip_loss requires image_features, text_features, and logit_scale")
    logits_per_image = logit_scale * image_features @ text_features.t()
    logits_per_text = logits_per_image.t()
    labels = torch.arange(image_features.size(0), device=image_features.device)
    return (F.cross_entropy(logits_per_image, labels) + F.cross_entropy(logits_per_text, labels)) / 2


def token_classification_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    token_cls_freq: torch.Tensor | None = None,
    token_cls_num_updates: torch.Tensor | None = None,
    use_reweighting: bool = True,
) -> torch.Tensor:
    """SuperCLIP-style token classification loss with optional frequency reweighting."""
    targets = labels.float()

    if use_reweighting and token_cls_freq is not None and token_cls_num_updates is not None:
        with torch.no_grad():
            batch_avg = targets.sum(dim=0, keepdim=True).to(dtype=token_cls_freq.dtype)
            batch_avg /= max(targets.shape[0], 1)
            token_cls_freq.add_(batch_avg)
            token_cls_num_updates.add_(1)
            all_batch_size = max(targets.shape[0], 1)
            weights = torch.log(
                (token_cls_num_updates + 1.0 / all_batch_size)
                / (token_cls_freq + 1.0 / all_batch_size)
            )
        targets = targets * weights.to(dtype=targets.dtype)

    row_sums = targets.sum(dim=1, keepdim=True).clamp(min=1e-6)
    norm_targets = targets / row_sums
    return -(F.log_softmax(logits, dim=1) * norm_targets).sum(dim=1).mean()


def create_mask(token_ids: torch.Tensor, mask_ratio: float, max_masks: int):
    batch_size, seq_len = token_ids.shape
    device = token_ids.device
    masked_token_ids = token_ids.clone()
    mask_targets = torch.zeros(batch_size, max_masks, dtype=torch.long, device=device)
    mask_positions = torch.full((batch_size, max_masks), -1, dtype=torch.long, device=device)

    for i in range(batch_size):
        content_pos = []
        for j in range(seq_len):
            tid = token_ids[i, j].item()
            if tid not in (SOT_TOKEN, EOT_TOKEN, PAD_TOKEN):
                content_pos.append(j)

        if not content_pos:
            continue

        n_mask = max(1, min(int(len(content_pos) * mask_ratio), max_masks))
        perm = torch.randperm(len(content_pos))[:n_mask]
        selected = sorted([content_pos[p] for p in perm])

        for k, pos in enumerate(selected):
            mask_targets[i, k] = token_ids[i, pos]
            mask_positions[i, k] = pos
            masked_token_ids[i, pos] = PAD_TOKEN

    return masked_token_ids, mask_targets, mask_positions


def create_phrase_mask(token_ids: torch.Tensor, phrase_data: dict, image_ids: list, max_masks: int):
    batch_size, seq_len = token_ids.shape
    device = token_ids.device
    masked_token_ids = token_ids.clone()
    mask_targets = torch.zeros(batch_size, max_masks, dtype=torch.long, device=device)
    mask_positions = torch.full((batch_size, max_masks), -1, dtype=torch.long, device=device)

    for i in range(batch_size):
        img_id_str = str(image_ids[i])
        phrases = phrase_data.get(img_id_str, [])

        if not phrases:
            content_pos = [j for j in range(seq_len) if token_ids[i, j].item() not in (SOT_TOKEN, EOT_TOKEN, PAD_TOKEN)]
            if not content_pos:
                continue
            n_mask = max(1, min(len(content_pos) // 4, max_masks))
            perm = torch.randperm(len(content_pos))[:n_mask]
            selected = sorted([content_pos[p_idx] for p_idx in perm])
            for k, pos in enumerate(selected):
                mask_targets[i, k] = token_ids[i, pos]
                mask_positions[i, k] = pos
                masked_token_ids[i, pos] = PAD_TOKEN
            continue

        phrase_entry = _random.choice(phrases)
        phrase_toks = phrase_entry["token_ids"]
        caption_toks = token_ids[i].tolist()
        span_start = _find_sublist(caption_toks, phrase_toks)

        if span_start == -1:
            content_pos = [j for j in range(seq_len) if token_ids[i, j].item() not in (SOT_TOKEN, EOT_TOKEN, PAD_TOKEN)]
            if content_pos:
                pos = content_pos[0]
                mask_targets[i, 0] = token_ids[i, pos]
                mask_positions[i, 0] = pos
                masked_token_ids[i, pos] = PAD_TOKEN
            continue

        n_phrase_toks = min(len(phrase_toks), max_masks)
        for k in range(n_phrase_toks):
            pos = span_start + k
            mask_targets[i, k] = token_ids[i, pos]
            mask_positions[i, k] = pos
            masked_token_ids[i, pos] = PAD_TOKEN

    return masked_token_ids, mask_targets, mask_positions


_DET = r"(?:a|an|the|this|that|some|two|three|four|five|six|many|several|few|no|my|his|her|their|its)"
_NP_PATTERN = re.compile(rf"\b({_DET}\s+\w+(?:\s+\w+){{0,3}})\b", re.IGNORECASE)


def _extract_phrases_from_caption(caption: str):
    phrases = []
    for match in _NP_PATTERN.finditer(caption.lower()):
        phrase = " ".join(match.group(1).split())
        n_words = len(phrase.split())
        if 2 <= n_words <= 5:
            phrases.append(phrase)
    return phrases


def create_phrase_mask_from_captions(token_ids: torch.Tensor, captions_raw: list, tokenizer, max_masks: int):
    batch_size, seq_len = token_ids.shape
    device = token_ids.device
    masked_token_ids = token_ids.clone()
    mask_targets = torch.zeros(batch_size, max_masks, dtype=torch.long, device=device)
    mask_positions = torch.full((batch_size, max_masks), -1, dtype=torch.long, device=device)

    for i in range(batch_size):
        caption = captions_raw[i] if isinstance(captions_raw[i], str) else captions_raw[i]
        phrases = _extract_phrases_from_caption(caption)
        caption_toks = token_ids[i].tolist()
        masked_something = False

        if phrases:
            _random.shuffle(phrases)
            for phrase in phrases:
                phrase_tok_ids = tokenizer(phrase).squeeze(0).tolist()
                content_toks = [t for t in phrase_tok_ids if t not in (SOT_TOKEN, EOT_TOKEN, PAD_TOKEN)]
                if not content_toks or len(content_toks) > max_masks:
                    continue
                span_start = _find_sublist(caption_toks, content_toks)
                if span_start == -1:
                    continue
                n_phrase_toks = min(len(content_toks), max_masks)
                for k in range(n_phrase_toks):
                    pos = span_start + k
                    mask_targets[i, k] = token_ids[i, pos]
                    mask_positions[i, k] = pos
                    masked_token_ids[i, pos] = PAD_TOKEN
                masked_something = True
                break

        if not masked_something:
            content_pos = [j for j in range(seq_len) if token_ids[i, j].item() not in (SOT_TOKEN, EOT_TOKEN, PAD_TOKEN)]
            if not content_pos:
                continue
            n_mask = max(1, min(len(content_pos) // 4, max_masks))
            perm = torch.randperm(len(content_pos))[:n_mask]
            selected = sorted([content_pos[p_idx] for p_idx in perm])
            for k, pos in enumerate(selected):
                mask_targets[i, k] = token_ids[i, pos]
                mask_positions[i, k] = pos
                masked_token_ids[i, pos] = PAD_TOKEN

    return masked_token_ids, mask_targets, mask_positions


def _find_sublist(haystack: list, needle: list) -> int:
    for i in range(len(haystack) - len(needle) + 1):
        if haystack[i:i + len(needle)] == needle:
            return i
    return -1


def reconstruction_loss(recon_logits: torch.Tensor, mask_targets: torch.Tensor) -> torch.Tensor:
    batch_size, max_masks, vocab_size = recon_logits.shape
    logits_flat = recon_logits.view(batch_size * max_masks, vocab_size)
    targets_flat = mask_targets.view(batch_size * max_masks)
    valid = targets_flat != 0
    if valid.sum() == 0:
        return torch.tensor(0.0, device=recon_logits.device, requires_grad=True)
    return F.cross_entropy(logits_flat[valid], targets_flat[valid])


def total_loss(*args, **kwargs):
    """Compute the configured training loss.

    Preferred API: keyword-only arguments for the retrieval-first training path.
    Backward compatibility: the legacy positional signature
        total_loss(token_cls_logits, token_cls_labels, recon_logits, mask_targets, lambda_recon)
    is still accepted for older local tests, but it omits the CLIP term and
    should not be used for real experiments.
    """
    if args and not kwargs:
        if len(args) != 5:
            raise TypeError("Legacy total_loss expects 5 positional arguments")
        token_cls_logits, token_cls_labels, recon_logits, mask_targets, lambda_recon = args
        l_clip = token_cls_logits.new_zeros(())
        l_token_cls = token_classification_loss(token_cls_logits, token_cls_labels, use_reweighting=False)
        l_recon = reconstruction_loss(recon_logits, mask_targets)
        l_total = l_token_cls + lambda_recon * l_recon
        return l_total, {
            "train_mode": "legacy_image_only",
            "l_clip": float(l_clip.detach().item()),
            "l_token_cls": float(l_token_cls.detach().item()),
            "l_recon": float(l_recon.detach().item()),
            "l_total": float(l_total.detach().item()),
        }

    train_mode = kwargs["train_mode"]
    image_features = kwargs["image_features"]
    text_features = kwargs.get("text_features")
    logit_scale = kwargs.get("logit_scale")
    token_cls_logits = kwargs.get("token_cls_logits")
    token_cls_labels = kwargs.get("token_cls_labels")
    recon_logits = kwargs.get("recon_logits")
    mask_targets = kwargs.get("mask_targets")
    lambda_clip = kwargs["lambda_clip"]
    lambda_token_cls = kwargs["lambda_token_cls"]
    lambda_recon = kwargs["lambda_recon"]
    token_cls_freq = kwargs.get("token_cls_freq")
    token_cls_num_updates = kwargs.get("token_cls_num_updates")
    token_cls_use_reweighting = kwargs.get("token_cls_use_reweighting", True)

    effective_mode = resolve_train_mode(train_mode, lambda_recon)
    zero = image_features.new_zeros(())

    l_clip = zero
    l_token_cls = zero
    l_recon = zero

    if effective_mode in {"clip_only", "superclip_baseline", "superclip_recon"}:
        l_clip = contrastive_clip_loss(image_features, text_features, logit_scale)

    if effective_mode in {"superclip_baseline", "superclip_recon"}:
        if token_cls_logits is None or token_cls_labels is None:
            raise ValueError("Token classification tensors are required for SuperCLIP modes")
        l_token_cls = token_classification_loss(
            token_cls_logits,
            token_cls_labels,
            token_cls_freq=token_cls_freq,
            token_cls_num_updates=token_cls_num_updates,
            use_reweighting=token_cls_use_reweighting,
        )

    if effective_mode == "superclip_recon":
        if recon_logits is None or mask_targets is None:
            raise ValueError("Reconstruction tensors are required for superclip_recon mode")
        l_recon = reconstruction_loss(recon_logits, mask_targets)

    l_total = lambda_clip * l_clip + lambda_token_cls * l_token_cls + lambda_recon * l_recon
    return l_total, {
        "train_mode": effective_mode,
        "l_clip": float(l_clip.detach().item()),
        "l_token_cls": float(l_token_cls.detach().item()),
        "l_recon": float(l_recon.detach().item()),
        "l_total": float(l_total.detach().item()),
    }
