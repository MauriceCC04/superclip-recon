"""
Extract short noun/compositional phrases from COCO captions for Variant B.

Strategy: use spaCy noun chunks to find phrases like "a red car",
"the tall building", "two black dogs". These are exactly the compositional
structures that bag-of-words supervision misses.

Fallback: if spaCy is unavailable, uses a simple regex-based extractor
that catches (det? adj* noun+) patterns.

Usage:
    python extract_phrases.py --coco_root ./data/coco --output phrases.json
"""

import json
import argparse
import re
from collections import defaultdict

import open_clip


def extract_with_spacy(captions_by_id):
    """Extract noun phrases using spaCy."""
    import spacy
    nlp = spacy.load("en_core_web_sm")

    phrases_by_id = defaultdict(list)
    for img_id, captions in captions_by_id.items():
        seen = set()
        for cap in captions:
            doc = nlp(cap.lower())
            for chunk in doc.noun_chunks:
                phrase = chunk.text.strip()
                n_words = len(phrase.split())
                if 2 <= n_words <= 5 and phrase not in seen:
                    seen.add(phrase)
                    phrases_by_id[img_id].append(phrase)
    return phrases_by_id


def extract_with_regex(captions_by_id):
    """Fallback: regex-based noun phrase extraction."""
    det = r"(?:a|an|the|this|that|some|two|three|four|five|many|several|few)?"
    adj = r"(?:\s+(?:big|small|large|tall|short|red|blue|green|white|black|brown|yellow|old|new|young|little|long|dark|bright|wooden|metal|glass|stone|plastic)\s*)*"
    noun = r"(?:\s+\w+){1,2}"
    pattern = re.compile(rf"\b({det}{adj}{noun})\b", re.IGNORECASE)

    phrases_by_id = defaultdict(list)
    for img_id, captions in captions_by_id.items():
        seen = set()
        for cap in captions:
            matches = pattern.findall(cap.lower())
            for m in matches:
                phrase = " ".join(m.split())
                n_words = len(phrase.split())
                if 2 <= n_words <= 5 and phrase not in seen:
                    seen.add(phrase)
                    phrases_by_id[img_id].append(phrase)
    return phrases_by_id


def tokenize_phrases(phrases_by_id, tokenizer):
    """Tokenize each phrase and store token IDs."""
    SOT, EOT, PAD = 49406, 49407, 0
    result = {}
    for img_id, phrases in phrases_by_id.items():
        entries = []
        for phrase in phrases:
            toks = tokenizer(phrase).squeeze(0).tolist()
            content_toks = [t for t in toks if t not in (SOT, EOT, PAD)]
            if 1 <= len(content_toks) <= 8:
                entries.append({
                    "phrase": phrase,
                    "token_ids": content_toks,
                })
        result[str(img_id)] = entries
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--coco_root", type=str, default="./data/coco")
    parser.add_argument("--output", type=str, default="phrases.json")
    parser.add_argument("--use_regex", action="store_true",
                        help="Use regex fallback instead of spaCy")
    args = parser.parse_args()

    ann_path = f"{args.coco_root}/annotations/captions_train2017.json"
    with open(ann_path) as f:
        data = json.load(f)

    captions_by_id = defaultdict(list)
    for ann in data["annotations"]:
        captions_by_id[ann["image_id"]].append(ann["caption"])

    print(f"Processing {len(captions_by_id)} images...")

    if args.use_regex:
        print("Using regex-based extraction (fallback)")
        phrases_by_id = extract_with_regex(captions_by_id)
    else:
        try:
            print("Using spaCy noun chunk extraction")
            phrases_by_id = extract_with_spacy(captions_by_id)
        except (ImportError, OSError) as e:
            print(f"spaCy not available ({e}), falling back to regex")
            phrases_by_id = extract_with_regex(captions_by_id)

    tokenizer = open_clip.get_tokenizer("ViT-B-32")
    result = tokenize_phrases(phrases_by_id, tokenizer)

    total_phrases = sum(len(v) for v in result.values())
    images_with_phrases = sum(1 for v in result.values() if len(v) > 0)
    avg_per_image = total_phrases / max(images_with_phrases, 1)

    print(f"Extracted {total_phrases} phrases from {images_with_phrases} images")
    print(f"Average {avg_per_image:.1f} phrases per image")

    with open(args.output, "w") as f:
        json.dump(result, f)
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
