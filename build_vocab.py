"""
Build the top-K token vocabulary from COCO captions.

This maps the most frequent CLIP tokens in the training captions
to class indices for the token-classification head.

Usage:
    python build_vocab.py --coco_root ./data/coco --top_k 1000 --output vocab.json
"""

import json
import argparse
from collections import Counter
import open_clip


SOT_TOKEN = 49406
EOT_TOKEN = 49407
PAD_TOKEN = 0


def build_vocab(coco_root: str, top_k: int, output_path: str):
    tokenizer = open_clip.get_tokenizer("ViT-B-32")

    ann_path = f"{coco_root}/annotations/captions_train2017.json"
    with open(ann_path, "r") as f:
        data = json.load(f)

    print(f"Counting tokens across {len(data['annotations'])} captions...")
    counter = Counter()
    for ann in data["annotations"]:
        token_ids = tokenizer(ann["caption"]).squeeze(0).tolist()
        for tid in token_ids:
            if tid not in (SOT_TOKEN, EOT_TOKEN, PAD_TOKEN):
                counter[tid] += 1

    most_common = counter.most_common(top_k)
    vocab_map = {}
    for class_idx, (token_id, count) in enumerate(most_common):
        vocab_map[token_id] = class_idx

    save_data = {
        "top_k": top_k,
        "total_unique_tokens": len(counter),
        "vocab_map": {str(k): v for k, v in vocab_map.items()},
        "token_counts": {str(tid): cnt for tid, cnt in most_common},
    }

    with open(output_path, "w") as f:
        json.dump(save_data, f, indent=2)

    print(f"Saved top-{top_k} vocab to {output_path}")
    print(f"Coverage: top-{top_k} tokens cover "
          f"{sum(c for _, c in most_common) / sum(counter.values()) * 100:.1f}% of all token occurrences")

    return vocab_map


def load_vocab(path: str) -> dict:
    """Load vocab_map from JSON. Returns {int_token_id: int_class_index}."""
    with open(path, "r") as f:
        data = json.load(f)
    return {int(k): v for k, v in data["vocab_map"].items()}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--coco_root", type=str, default="./data/coco")
    parser.add_argument("--top_k", type=int, default=1000)
    parser.add_argument("--output", type=str, default="vocab.json")
    args = parser.parse_args()
    build_vocab(args.coco_root, args.top_k, args.output)
