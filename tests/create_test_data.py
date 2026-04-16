"""
Generate a tiny fake COCO-format dataset for local testing.

Creates:
    test_data/coco/train2017/          (16 random JPEG images)
    test_data/coco/val2017/            (8 random JPEG images)
    test_data/coco/annotations/        (captions JSON files)

These are structurally identical to real COCO but tiny, so every
code path can be exercised on CPU without downloading 20GB.

Usage:
    python tests/create_test_data.py
"""

import os
import json
import random
from PIL import Image


FAKE_CAPTIONS = [
    "a red car parked on the street",
    "two black dogs playing in the park",
    "a woman riding a brown horse",
    "the tall building near the river",
    "a small child holding a yellow balloon",
    "three white birds sitting on a fence",
    "a man wearing a blue jacket",
    "the old wooden bridge over the lake",
    "a large pizza on a white plate",
    "a cat sleeping on a red couch",
    "two people walking on the beach",
    "a green bicycle leaning against a wall",
    "the young girl eating an ice cream",
    "a big truck driving down the highway",
    "a white cup of coffee on the table",
    "the brown dog chasing a tennis ball",
]


def create_test_data(root="test_data/coco", n_train=16, n_val=8, captions_per_image=5):
    """Create fake COCO-format data."""
    os.makedirs(f"{root}/train2017", exist_ok=True)
    os.makedirs(f"{root}/val2017", exist_ok=True)
    os.makedirs(f"{root}/annotations", exist_ok=True)

    for split, n_images, image_dir in [
        ("train", n_train, "train2017"),
        ("val", n_val, "val2017"),
    ]:
        images_info = []
        annotations = []
        ann_id = 0

        for i in range(n_images):
            img_id = i + 1 + (0 if split == "train" else 1000)
            filename = f"{img_id:012d}.jpg"

            img = Image.new("RGB", (64, 64),
                            color=(random.randint(0, 255),
                                   random.randint(0, 255),
                                   random.randint(0, 255)))
            img.save(f"{root}/{image_dir}/{filename}")

            images_info.append({
                "id": img_id,
                "file_name": filename,
                "height": 64,
                "width": 64,
            })

            for j in range(captions_per_image):
                cap = random.choice(FAKE_CAPTIONS)
                annotations.append({
                    "id": ann_id,
                    "image_id": img_id,
                    "caption": cap,
                })
                ann_id += 1

        ann_file = f"{root}/annotations/captions_{image_dir}.json"
        with open(ann_file, "w") as f:
            json.dump({"images": images_info, "annotations": annotations}, f)

        print(f"Created {split}: {n_images} images, {len(annotations)} captions -> {ann_file}")

    print(f"Test data ready at {root}/")


if __name__ == "__main__":
    create_test_data()
