"""
Model definitions for SuperCLIP-Recon.

Architecture overview:
    ┌─────────────┐
    │  CLIP ViT   │  (frozen or fine-tuned backbone)
    │  ViT-B/32   │
    └──────┬──────┘
           │ image_features [B, D]
           ├──────────────────────────────┐
           │                              │
    ┌──────▼──────┐               ┌───────▼────────┐
    │  Token-Cls  │ (baseline)    │  Recon Head    │ (our contribution)
    │  Linear     │               │  2-layer MLP   │
    └──────┬──────┘               └───────┬────────┘
           │                              │
    L_token_cls                     L_recon
           │                              │
           └──────────┬───────────────────┘
                      │
              L_total = L_tc + λ * L_recon
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import open_clip


class TokenClassificationHead(nn.Module):
    """
    SuperCLIP-style token classification head.
    Maps image features -> per-token presence logits.
    This is a simple linear layer: [B, D] -> [B, num_classes].
    """

    def __init__(self, embed_dim: int, num_classes: int):
        super().__init__()
        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, image_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            image_features: [B, D] normalized CLIP image embeddings
        Returns:
            logits: [B, num_classes] token presence logits
        """
        return self.classifier(image_features)


class ReconstructionHead(nn.Module):
    """
    Lightweight 2-layer MLP that predicts masked caption tokens
    from image features.

    Variant A (masked token prediction):
        Input:  image_features [B, D]
        Output: logits [B, max_masks, vocab_size]

    Variant B (phrase reconstruction):
        Same architecture, but trained on short noun-phrase token sequences.
    """

    def __init__(self, embed_dim: int, hidden_dim: int, vocab_size: int, max_masks: int = 10):
        super().__init__()
        self.max_masks = max_masks
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, max_masks * vocab_size),
        )
        self.vocab_size = vocab_size

    def forward(self, image_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            image_features: [B, D]
        Returns:
            logits: [B, max_masks, vocab_size]
        """
        out = self.mlp(image_features)                      # [B, max_masks * vocab_size]
        return out.view(-1, self.max_masks, self.vocab_size) # [B, max_masks, vocab_size]


class SuperCLIPRecon(nn.Module):
    """
    Full model: CLIP backbone + token-cls head + reconstruction head.
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        # --- Load CLIP backbone ---
        self.clip_model, _, self.preprocess = open_clip.create_model_and_transforms(
            cfg.model.clip_model,
            pretrained=cfg.model.clip_pretrained,
        )
        self.tokenizer = open_clip.get_tokenizer(cfg.model.clip_model)

        # Freeze CLIP text encoder (we only fine-tune vision side + heads)
        for param in self.clip_model.parameters():
            param.requires_grad = False

        # Unfreeze visual encoder for fine-tuning
        for param in self.clip_model.visual.parameters():
            param.requires_grad = True

        # --- Token classification head (baseline) ---
        self.token_cls_head = TokenClassificationHead(
            embed_dim=cfg.model.embed_dim,
            num_classes=cfg.model.num_token_classes,
        )

        # --- Reconstruction head (our contribution) ---
        self.recon_head = ReconstructionHead(
            embed_dim=cfg.model.embed_dim,
            hidden_dim=cfg.model.recon_hidden_dim,
            vocab_size=cfg.model.recon_vocab_size,
            max_masks=int(cfg.data.max_caption_length * cfg.model.mask_ratio) + 1,
        )

    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        """Encode images through CLIP visual encoder. Returns L2-normalized features."""
        features = self.clip_model.encode_image(images)
        features = F.normalize(features, dim=-1)
        return features

    def encode_text(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Encode text through CLIP text encoder. Returns L2-normalized features."""
        features = self.clip_model.encode_text(token_ids)
        features = F.normalize(features, dim=-1)
        return features

    def forward(self, images: torch.Tensor, token_ids: torch.Tensor):
        """
        Forward pass for training.

        Args:
            images:    [B, 3, 224, 224]
            token_ids: [B, 77] tokenized captions

        Returns:
            dict with keys:
                image_features: [B, D]
                text_features:  [B, D]
                token_cls_logits: [B, num_classes]
                recon_logits: [B, max_masks, vocab_size]
        """
        image_features = self.encode_image(images)
        text_features = self.encode_text(token_ids)
        token_cls_logits = self.token_cls_head(image_features)
        recon_logits = self.recon_head(image_features)

        return {
            "image_features": image_features,
            "text_features": text_features,
            "token_cls_logits": token_cls_logits,
            "recon_logits": recon_logits,
        }
