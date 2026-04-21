"""
Model definitions for SuperCLIP-Recon.

This implementation is retrieval-first:
    - baseline: SuperCLIP-style CLIP contrastive loss + token classification
    - extension: add lightweight caption reconstruction on top

The key repair relative to the earlier implementation is that training always
keeps the CLIP image-text alignment objective active, so lambda=0 corresponds
to a real SuperCLIP-style baseline rather than image-only token supervision.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import open_clip


class TokenClassificationHead(nn.Module):
    """Maps image embeddings to token-presence logits."""

    def __init__(self, embed_dim: int, num_classes: int):
        super().__init__()
        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, image_features: torch.Tensor) -> torch.Tensor:
        return self.classifier(image_features)


class ReconstructionHead(nn.Module):
    """Predict masked caption tokens from image embeddings."""

    SLOT_DIM = 64

    def __init__(self, embed_dim: int, hidden_dim: int, vocab_size: int, max_masks: int = 10):
        super().__init__()
        self.max_masks = max_masks
        self.vocab_size = vocab_size
        self.slot_embeddings = nn.Embedding(max_masks, self.SLOT_DIM)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim + self.SLOT_DIM, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, vocab_size),
        )

    def forward(self, image_features: torch.Tensor) -> torch.Tensor:
        batch_size = image_features.size(0)
        device = image_features.device
        slot_ids = torch.arange(self.max_masks, device=device)
        slot_embs = self.slot_embeddings(slot_ids)
        img_exp = image_features.unsqueeze(1).expand(-1, self.max_masks, -1)
        slot_exp = slot_embs.unsqueeze(0).expand(batch_size, -1, -1)
        combined = torch.cat([img_exp, slot_exp], dim=-1)
        return self.mlp(combined)


class SuperCLIPRecon(nn.Module):
    """CLIP backbone + SuperCLIP token head + reconstruction head."""

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        self.clip_model, _, self.preprocess = open_clip.create_model_and_transforms(
            cfg.model.clip_model,
            pretrained=cfg.model.clip_pretrained,
        )
        self.tokenizer = open_clip.get_tokenizer(cfg.model.clip_model)

        self._configure_trainable_parameters()

        self.token_cls_head = TokenClassificationHead(
            embed_dim=cfg.model.embed_dim,
            num_classes=cfg.model.num_token_classes,
        )
        self.recon_head = ReconstructionHead(
            embed_dim=cfg.model.embed_dim,
            hidden_dim=cfg.model.recon_hidden_dim,
            vocab_size=cfg.model.recon_vocab_size,
            max_masks=int(cfg.data.max_caption_length * cfg.model.mask_ratio) + 1,
        )

        self.register_buffer(
            "token_cls_freq",
            torch.zeros(1, cfg.model.num_token_classes, dtype=torch.float64),
        )
        self.register_buffer(
            "token_cls_num_updates",
            torch.zeros(1, 1, dtype=torch.float64),
        )

    def _configure_trainable_parameters(self):
        for name, param in self.clip_model.named_parameters():
            if name.startswith("visual."):
                param.requires_grad = not self.cfg.train.freeze_vision_tower
            elif name in {"logit_scale", "logit_bias"}:
                param.requires_grad = not self.cfg.train.freeze_logit_scale
            else:
                param.requires_grad = not self.cfg.train.freeze_text_tower

    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        features = self.clip_model.encode_image(images)
        return F.normalize(features, dim=-1)

    def encode_text(self, token_ids: torch.Tensor) -> torch.Tensor:
        features = self.clip_model.encode_text(token_ids)
        return F.normalize(features, dim=-1)

    def get_logit_scale(self) -> torch.Tensor:
        return self.clip_model.logit_scale.exp()

    def forward(self, images: torch.Tensor, token_ids: torch.Tensor, encode_text: bool = True):
        image_features = self.encode_image(images)
        text_features = self.encode_text(token_ids) if encode_text else None
        token_cls_logits = self.token_cls_head(image_features)
        recon_logits = self.recon_head(image_features)
        return {
            "image_features": image_features,
            "text_features": text_features,
            "logit_scale": self.get_logit_scale(),
            "token_cls_logits": token_cls_logits,
            "recon_logits": recon_logits,
            "token_cls_freq": self.token_cls_freq,
            "token_cls_num_updates": self.token_cls_num_updates,
        }
