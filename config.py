"""
Configuration for SuperCLIP-Recon project.
All hyperparameters and paths in one place.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class DataConfig:
    coco_root: str = "./data/coco"
    train_ann: str = "annotations/captions_train2017.json"
    val_ann: str = "annotations/captions_val2017.json"
    train_images: str = "train2017"
    val_images: str = "val2017"
    max_caption_length: int = 77
    num_workers: int = 4
    pin_memory: bool = True


@dataclass
class ModelConfig:
    clip_model: str = "ViT-B-32"
    clip_pretrained: str = "openai"
    embed_dim: int = 512
    num_token_classes: int = 1000
    token_cls_use_reweighting: bool = True
    recon_hidden_dim: int = 512
    recon_vocab_size: int = 49408
    mask_ratio: float = 0.15
    variant: str = "A"


@dataclass
class TrainConfig:
    batch_size: int = 128
    epochs: int = 10
    lr: float = 1e-5
    weight_decay: float = 0.01
    warmup_steps: int = 500
    train_mode: str = "auto"
    lambda_clip: float = 1.0
    lambda_token_cls: float = 1.0
    lambda_recon: float = 0.5
    use_amp: bool = True
    grad_clip_norm: float = 1.0
    freeze_text_tower: bool = False
    freeze_vision_tower: bool = False
    freeze_logit_scale: bool = False
    log_every: int = 50
    eval_every_epoch: int = 1
    save_dir: str = "./checkpoints"
    run_name: str = "superclip-recon"
    use_wandb: bool = False
    seed: int = 42


@dataclass
class EvalConfig:
    retrieval_k: list = field(default_factory=lambda: [1, 5, 10])
    winoground_path: Optional[str] = None
    aro_path: Optional[str] = None


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
