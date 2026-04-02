"""
Configuration for SuperCLIP-Recon project.
All hyperparameters and paths in one place.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class DataConfig:
    coco_root: str = "./data/coco"                # path to COCO dataset root
    train_ann: str = "annotations/captions_train2017.json"
    val_ann: str = "annotations/captions_val2017.json"
    train_images: str = "train2017"
    val_images: str = "val2017"
    max_caption_length: int = 77                   # CLIP default context length
    num_workers: int = 4
    pin_memory: bool = True


@dataclass
class ModelConfig:
    clip_model: str = "ViT-B-32"                   # OpenCLIP model name
    clip_pretrained: str = "openai"                 # pretrained weights source
    embed_dim: int = 512                            # CLIP embedding dimension
    # --- Token classification head (SuperCLIP baseline) ---
    num_token_classes: int = 1000                   # top-K COCO vocabulary tokens
    # --- Reconstruction head (our contribution) ---
    recon_hidden_dim: int = 512                     # hidden dim of 2-layer MLP
    recon_vocab_size: int = 49408                   # CLIP tokenizer vocab size
    mask_ratio: float = 0.15                        # fraction of caption tokens to mask
    variant: str = "A"                              # "A" = masked token, "B" = phrase recon


@dataclass
class TrainConfig:
    batch_size: int = 128
    epochs: int = 10
    lr: float = 1e-5                                # fine-tuning LR (small)
    weight_decay: float = 0.01
    warmup_steps: int = 500
    # --- Loss weights ---
    lambda_recon: float = 0.5                       # weight for L_recon
    # --- Logging ---
    log_every: int = 50
    eval_every_epoch: int = 1
    save_dir: str = "./checkpoints"
    run_name: str = "superclip-recon"
    use_wandb: bool = False
    seed: int = 42


@dataclass
class EvalConfig:
    retrieval_k: list = field(default_factory=lambda: [1, 5, 10])
    # compositional probe dataset (optional, Phase 4)
    winoground_path: Optional[str] = None
    aro_path: Optional[str] = None


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
