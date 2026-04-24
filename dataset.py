"""
COCO Captions dataset for SuperCLIP-Recon.

Each sample yields:
    image:       [3, 224, 224] tensor (CLIP-preprocessed)
    token_ids:   [max_len] long tensor (CLIP-tokenized caption)
    caption_raw: str (original caption text, for debugging)
"""

import os
import json
import random
import hashlib
from PIL import Image
from torch.utils.data import Dataset


class COCOCaptionsDataset(Dataset):
    """
    Loads COCO images + captions. For each image, picks ONE caption.

    Modes:
        - stochastic mode (default): caption is sampled with random.choice(...)
        - deterministic mode: caption is chosen deterministically from
          (base_seed, epoch, image_id), which is reproducible across runs

    Deterministic mode is recommended for matched baseline vs improvement runs.
    """

    def __init__(
        self,
        root,
        ann_file,
        image_dir,
        transform=None,
        tokenizer=None,
        base_seed: int = 42,
        deterministic_caption: bool = False,
    ):
        self.image_root = os.path.join(root, image_dir)
        ann_path = os.path.join(root, ann_file)

        with open(ann_path, "r") as f:
            data = json.load(f)

        self.img_id_to_captions = {}
        for ann in data["annotations"]:
            img_id = ann["image_id"]
            if img_id not in self.img_id_to_captions:
                self.img_id_to_captions[img_id] = []
            self.img_id_to_captions[img_id].append(ann["caption"])

        self.img_id_to_file = {}
        for img_info in data["images"]:
            self.img_id_to_file[img_info["id"]] = img_info["file_name"]

        self.image_ids = sorted(
            [iid for iid in self.img_id_to_captions if iid in self.img_id_to_file]
        )

        self.transform = transform
        self.tokenizer = tokenizer
        self.base_seed = int(base_seed)
        self.deterministic_caption = bool(deterministic_caption)
        self.epoch = 0

        mode = "deterministic" if self.deterministic_caption else "stochastic"
        print(f"[Dataset] Loaded {len(self.image_ids)} images from {ann_path} ({mode} caption selection)")

    def __len__(self):
        return len(self.image_ids)

    def set_epoch(self, epoch: int):
        self.epoch = int(epoch)

    def _deterministic_index(self, img_id: int, n_captions: int) -> int:
        key = f"{self.base_seed}:{self.epoch}:{int(img_id)}".encode("utf-8")
        digest = hashlib.sha1(key).hexdigest()
        value = int(digest[:16], 16)
        return value % n_captions

    def _choose_caption(self, captions, img_id: int):
        if not captions:
            return ""

        if self.deterministic_caption:
            idx = self._deterministic_index(img_id, len(captions))
            return captions[idx]

        return random.choice(captions)

    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        filename = self.img_id_to_file[img_id]
        img_path = os.path.join(self.image_root, filename)

        image = Image.open(img_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)

        captions = self.img_id_to_captions[img_id]
        caption_raw = self._choose_caption(captions, img_id)

        token_ids = self.tokenizer(caption_raw).squeeze(0)

        return image, token_ids, caption_raw, img_id