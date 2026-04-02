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
from PIL import Image
from torch.utils.data import Dataset


class COCOCaptionsDataset(Dataset):
    """
    Loads COCO images + captions. For each image, randomly picks ONE caption
    per epoch (standard practice for contrastive training).
    """

    def __init__(self, root, ann_file, image_dir, transform=None, tokenizer=None):
        """
        Args:
            root:      path to coco root, e.g. "./data/coco"
            ann_file:  relative path to annotation json
            image_dir: relative path to image folder
            transform: CLIP image preprocessing transform
            tokenizer: open_clip tokenizer callable
        """
        self.image_root = os.path.join(root, image_dir)
        ann_path = os.path.join(root, ann_file)

        with open(ann_path, "r") as f:
            data = json.load(f)

        # Build image_id -> list of captions
        self.img_id_to_captions = {}
        for ann in data["annotations"]:
            img_id = ann["image_id"]
            if img_id not in self.img_id_to_captions:
                self.img_id_to_captions[img_id] = []
            self.img_id_to_captions[img_id].append(ann["caption"])

        # Build image_id -> filename
        self.img_id_to_file = {}
        for img_info in data["images"]:
            self.img_id_to_file[img_info["id"]] = img_info["file_name"]

        # Only keep images that have captions
        self.image_ids = sorted(
            [iid for iid in self.img_id_to_captions if iid in self.img_id_to_file]
        )

        self.transform = transform
        self.tokenizer = tokenizer

        print(f"[Dataset] Loaded {len(self.image_ids)} images from {ann_path}")

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        filename = self.img_id_to_file[img_id]
        img_path = os.path.join(self.image_root, filename)

        # Load and preprocess image
        image = Image.open(img_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)

        # Pick a random caption for this image
        captions = self.img_id_to_captions[img_id]
        caption_raw = random.choice(captions)

        # Tokenize (returns [1, 77] tensor, squeeze to [77])
        token_ids = self.tokenizer(caption_raw).squeeze(0)

        return image, token_ids, caption_raw, img_id


class COCORetrievalDataset(Dataset):
    """
    For retrieval evaluation: returns image + ALL 5 captions.
    """

    def __init__(self, root, ann_file, image_dir, transform=None, tokenizer=None):
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

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        filename = self.img_id_to_file[img_id]
        img_path = os.path.join(self.image_root, filename)

        image = Image.open(img_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)

        captions = self.img_id_to_captions[img_id]
        # Tokenize all captions: list of [77] tensors
        all_token_ids = [self.tokenizer(c).squeeze(0) for c in captions]

        return image, all_token_ids, captions
