"""
Local test suite for SuperCLIP-Recon.

Runs ALL code paths on tiny synthetic data, on CPU, with no real COCO download.
Catches shape mismatches, import errors, logic bugs, and pipeline breaks
BEFORE you waste HPC queue time.

Usage:
    cd superclip-recon
    python tests/run_tests.py

Expected output: 10/10 tests passed (takes ~60s on CPU).
"""

import os
import sys
import json
import traceback
import shutil

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

# ---------- Config ----------
TEST_DATA_ROOT = "test_data/coco"
TEST_VOCAB_PATH = "test_data/vocab.json"
TEST_PHRASE_PATH = "test_data/phrases.json"
DEVICE = torch.device("cpu")

results = []


def test(name):
    """Decorator to register and run a test."""
    def decorator(fn):
        def wrapper():
            print(f"\n{'─'*60}")
            print(f"TEST: {name}")
            print(f"{'─'*60}")
            try:
                fn()
                print(f"  ✓ PASSED")
                results.append((name, True, None))
            except Exception as e:
                print(f"  ✗ FAILED: {e}")
                traceback.print_exc()
                results.append((name, False, str(e)))
        wrapper.test_name = name
        return wrapper
    return decorator


# ============================================================
# Tests
# ============================================================

@test("1. Create synthetic test data")
def test_create_data():
    from tests.create_test_data import create_test_data
    create_test_data(root=TEST_DATA_ROOT, n_train=16, n_val=8)

    assert os.path.isdir(f"{TEST_DATA_ROOT}/train2017")
    assert os.path.isdir(f"{TEST_DATA_ROOT}/val2017")
    assert os.path.isfile(f"{TEST_DATA_ROOT}/annotations/captions_train2017.json")
    assert os.path.isfile(f"{TEST_DATA_ROOT}/annotations/captions_val2017.json")
    assert len(os.listdir(f"{TEST_DATA_ROOT}/train2017")) == 16
    assert len(os.listdir(f"{TEST_DATA_ROOT}/val2017")) == 8


@test("2. Build vocabulary")
def test_build_vocab():
    from build_vocab import build_vocab, load_vocab

    vocab_map = build_vocab(TEST_DATA_ROOT, top_k=50, output_path=TEST_VOCAB_PATH)
    assert len(vocab_map) > 0, "Vocab is empty"
    assert len(vocab_map) <= 50, f"Vocab too large: {len(vocab_map)}"

    # Reload and verify
    loaded = load_vocab(TEST_VOCAB_PATH)
    assert len(loaded) == len(vocab_map)
    # Keys should be ints, values should be ints
    for k, v in loaded.items():
        assert isinstance(k, int), f"Key {k} is not int"
        assert isinstance(v, int), f"Value {v} is not int"


@test("3. Dataset loading")
def test_dataset():
    from config import Config
    from model import SuperCLIPRecon
    from dataset import COCOCaptionsDataset
    from torch.utils.data import DataLoader

    cfg = Config()
    cfg.data.coco_root = TEST_DATA_ROOT
    cfg.model.num_token_classes = 50

    model = SuperCLIPRecon(cfg)

    ds = COCOCaptionsDataset(
        root=TEST_DATA_ROOT,
        ann_file=cfg.data.train_ann,
        image_dir=cfg.data.train_images,
        transform=model.preprocess,
        tokenizer=model.tokenizer,
    )
    assert len(ds) == 16, f"Expected 16 images, got {len(ds)}"

    loader = DataLoader(ds, batch_size=4, shuffle=True, num_workers=0)
    images, token_ids, captions, img_ids = next(iter(loader))

    assert images.shape == (4, 3, 224, 224), f"Bad image shape: {images.shape}"
    assert token_ids.shape == (4, 77), f"Bad token shape: {token_ids.shape}"
    assert len(captions) == 4
    assert img_ids.shape == (4,), f"Bad img_ids shape: {img_ids.shape}"


@test("4. Model forward pass")
def test_model_forward():
    from config import Config
    from model import SuperCLIPRecon

    cfg = Config()
    cfg.data.coco_root = TEST_DATA_ROOT
    cfg.model.num_token_classes = 50

    model = SuperCLIPRecon(cfg).to(DEVICE)

    B = 4
    images = torch.randn(B, 3, 224, 224)
    token_ids = torch.randint(1, 49405, (B, 77))

    outputs = model(images, token_ids)

    assert outputs["image_features"].shape == (B, 512), \
        f"Bad image_features: {outputs['image_features'].shape}"
    assert outputs["text_features"].shape == (B, 512), \
        f"Bad text_features: {outputs['text_features'].shape}"
    assert outputs["token_cls_logits"].shape == (B, 50), \
        f"Bad token_cls: {outputs['token_cls_logits'].shape}"

    max_masks = int(77 * cfg.model.mask_ratio) + 1
    assert outputs["recon_logits"].shape[0] == B
    assert outputs["recon_logits"].shape[1] == max_masks
    assert outputs["recon_logits"].shape[2] == cfg.model.recon_vocab_size


@test("5. Token classification labels")
def test_token_labels():
    from losses import build_token_labels
    from build_vocab import load_vocab

    vocab_map = load_vocab(TEST_VOCAB_PATH)
    token_ids = torch.randint(1, 49405, (4, 77))
    labels = build_token_labels(token_ids, vocab_map, num_classes=50)

    assert labels.shape == (4, 50), f"Bad labels shape: {labels.shape}"
    assert labels.dtype == torch.float32
    assert labels.min() >= 0 and labels.max() <= 1


@test("6. Masking — Variant A (random tokens)")
def test_mask_variant_a():
    from losses import create_mask

    token_ids = torch.randint(1, 49405, (4, 77))
    # Set SOT/EOT
    token_ids[:, 0] = 49406
    for i in range(4):
        token_ids[i, 10:] = 49407  # EOT at pos 10, rest padding

    max_masks = 12
    masked, targets, positions = create_mask(token_ids, mask_ratio=0.15, max_masks=max_masks)

    assert masked.shape == token_ids.shape
    assert targets.shape == (4, max_masks)
    assert positions.shape == (4, max_masks)

    # Verify at least some tokens were masked
    n_masked = (targets != 0).sum().item()
    assert n_masked > 0, "No tokens were masked"

    # Verify masked positions were zeroed
    for i in range(4):
        for k in range(max_masks):
            pos = positions[i, k].item()
            if pos >= 0:
                assert masked[i, pos].item() == 0, "Masked position not zeroed"


@test("7. Masking — Variant B (phrase masking)")
def test_mask_variant_b():
    from losses import create_phrase_mask
    import open_clip

    tokenizer = open_clip.get_tokenizer("ViT-B-32")

    # Create a known caption and its tokenization
    caption = "a red car parked on the street"
    token_ids = tokenizer(caption)  # [1, 77]
    token_ids = token_ids.repeat(4, 1)  # [4, 77]

    # Create fake phrase data
    phrase_toks = tokenizer("red car").squeeze(0).tolist()
    content_toks = [t for t in phrase_toks if t not in (49406, 49407, 0)]

    phrase_data = {
        "1": [{"phrase": "red car", "token_ids": content_toks}],
        "2": [{"phrase": "red car", "token_ids": content_toks}],
        "3": [{"phrase": "red car", "token_ids": content_toks}],
        "4": [{"phrase": "red car", "token_ids": content_toks}],
    }
    image_ids = [1, 2, 3, 4]

    masked, targets, positions = create_phrase_mask(
        token_ids, phrase_data, image_ids, max_masks=12
    )

    assert masked.shape == token_ids.shape
    assert targets.shape == (4, 12)
    n_masked = (targets != 0).sum().item()
    assert n_masked > 0, "No phrase tokens were masked"


@test("8. Loss computation + backward")
def test_loss_backward():
    from config import Config
    from model import SuperCLIPRecon
    from losses import build_token_labels, create_mask, total_loss
    from build_vocab import load_vocab

    cfg = Config()
    cfg.data.coco_root = TEST_DATA_ROOT
    cfg.model.num_token_classes = 50

    model = SuperCLIPRecon(cfg).to(DEVICE)
    vocab_map = load_vocab(TEST_VOCAB_PATH)

    B = 4
    images = torch.randn(B, 3, 224, 224)
    token_ids = torch.randint(1, 49405, (B, 77))
    token_ids[:, 0] = 49406
    for i in range(B):
        token_ids[i, 12:] = 0

    max_masks = int(77 * cfg.model.mask_ratio) + 1
    _, mask_targets, _ = create_mask(token_ids, cfg.model.mask_ratio, max_masks)

    outputs = model(images, token_ids)
    labels = build_token_labels(token_ids, vocab_map, 50)

    loss, loss_dict = total_loss(
        outputs["token_cls_logits"], labels,
        outputs["recon_logits"], mask_targets,
        lambda_recon=0.5,
    )

    assert loss.requires_grad, "Loss doesn't require grad"
    assert not torch.isnan(loss), "Loss is NaN"
    assert not torch.isinf(loss), "Loss is Inf"

    loss.backward()

    # Check gradients exist
    has_grad = any(p.grad is not None and p.grad.abs().sum() > 0
                   for p in model.parameters() if p.requires_grad)
    assert has_grad, "No gradients computed"

    for k in ["l_token_cls", "l_recon", "l_total"]:
        assert k in loss_dict, f"Missing key {k} in loss_dict"


@test("9. Phrase extraction")
def test_phrase_extraction():
    from extract_phrases import extract_with_regex, tokenize_phrases
    from collections import defaultdict
    import open_clip

    captions_by_id = defaultdict(list)
    captions_by_id[1] = ["a red car parked on the street"]
    captions_by_id[2] = ["two black dogs playing in the park"]

    phrases = extract_with_regex(captions_by_id)
    assert len(phrases) > 0, "No phrases extracted"

    tokenizer = open_clip.get_tokenizer("ViT-B-32")
    tokenized = tokenize_phrases(phrases, tokenizer)

    total = sum(len(v) for v in tokenized.values())
    assert total > 0, "No tokenized phrases"

    # Save for other tests
    with open(TEST_PHRASE_PATH, "w") as f:
        json.dump(tokenized, f)


@test("10. End-to-end mini training + eval")
def test_e2e_train_eval():
    """Run 1 epoch of training + retrieval eval on synthetic data."""
    from config import Config
    from model import SuperCLIPRecon
    from dataset import COCOCaptionsDataset
    from losses import build_token_labels, create_mask, total_loss
    from build_vocab import load_vocab
    from evaluate import compute_retrieval_metrics
    from torch.utils.data import DataLoader
    import numpy as np

    cfg = Config()
    cfg.data.coco_root = TEST_DATA_ROOT
    cfg.model.num_token_classes = 50
    cfg.train.batch_size = 4

    model = SuperCLIPRecon(cfg).to(DEVICE)
    vocab_map = load_vocab(TEST_VOCAB_PATH)
    max_masks = int(77 * cfg.model.mask_ratio) + 1

    ds = COCOCaptionsDataset(
        root=TEST_DATA_ROOT,
        ann_file=cfg.data.train_ann,
        image_dir=cfg.data.train_images,
        transform=model.preprocess,
        tokenizer=model.tokenizer,
    )
    loader = DataLoader(ds, batch_size=4, shuffle=True, num_workers=0, drop_last=True)

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=1e-5)

    # Train 1 batch
    model.train()
    images, token_ids, _, _ = next(iter(loader))
    _, mask_targets, _ = create_mask(token_ids, cfg.model.mask_ratio, max_masks)
    outputs = model(images, token_ids)
    labels = build_token_labels(token_ids, vocab_map, 50)
    loss, _ = total_loss(
        outputs["token_cls_logits"], labels,
        outputs["recon_logits"], mask_targets, 0.5,
    )
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Eval: compute retrieval on random embeddings (just test the function works)
    N = 8
    img_embs = np.random.randn(N, 512).astype(np.float32)
    img_embs = img_embs / np.linalg.norm(img_embs, axis=1, keepdims=True)
    txt_embs = np.random.randn(N * 5, 512).astype(np.float32)
    txt_embs = txt_embs / np.linalg.norm(txt_embs, axis=1, keepdims=True)

    metrics = compute_retrieval_metrics(img_embs, txt_embs)
    for k in ["i2t_r1", "i2t_r5", "i2t_r10", "t2i_r1", "t2i_r5", "t2i_r10"]:
        assert k in metrics, f"Missing metric: {k}"
        assert 0 <= metrics[k] <= 100, f"Bad metric value: {k}={metrics[k]}"


@test("11. Compositional eval imports and logic")
def test_compositional_eval():
    """Test that eval_compositional.py loads and its functions are callable.
    Does NOT download real datasets — only checks imports and function signatures."""
    from eval_compositional import (
        evaluate_winoground, evaluate_aro, run_compositional_eval
    )
    from config import Config
    from model import SuperCLIPRecon

    cfg = Config()
    cfg.data.coco_root = TEST_DATA_ROOT
    cfg.model.num_token_classes = 50
    model = SuperCLIPRecon(cfg).to(DEVICE)

    # Only test Winoground path (skips immediately with no HF token — no download)
    metrics = run_compositional_eval(model, DEVICE, benchmarks="winoground", hf_token=None)
    assert isinstance(metrics, dict), "run_compositional_eval should return a dict"

    # Verify ARO function exists and is callable (don't actually call it —
    # it would try to download ~1GB of images)
    assert callable(evaluate_aro), "evaluate_aro should be callable"


@test("12. Analysis script on synthetic results")
def test_analysis_script():
    """Test that analyze_results.py runs on fake result files."""
    from analyze_results import (
        plot_retrieval_comparison, plot_loss_curves,
        plot_lambda_sweep, plot_maskrate_sweep,
        plot_compositional, setup_style
    )

    setup_style()

    fake_results_dir = "test_data/results"
    fake_figures_dir = "test_data/results/figures"
    os.makedirs(fake_figures_dir, exist_ok=True)

    # Create fake main results
    for name in ["baseline", "variant_a", "variant_b"]:
        lam = 0.0 if name == "baseline" else 0.5
        fake = {
            "run_name": name,
            "variant": "A" if name != "variant_b" else "B",
            "lambda_recon": lam,
            "mask_ratio": 0.15,
            "wall_time_seconds": 600,
            "history": [
                {"epoch": 1, "losses": {"l_total": 2.0, "l_token_cls": 1.5, "l_recon": 1.0},
                 "retrieval": {"i2t_r1": 20, "i2t_r5": 45, "i2t_r10": 60,
                               "t2i_r1": 18, "t2i_r5": 42, "t2i_r10": 57}},
                {"epoch": 2, "losses": {"l_total": 1.5, "l_token_cls": 1.2, "l_recon": 0.6},
                 "retrieval": {"i2t_r1": 25, "i2t_r5": 50, "i2t_r10": 65,
                               "t2i_r1": 22, "t2i_r5": 47, "t2i_r10": 62}},
            ],
            "final_retrieval": {"i2t_r1": 25, "i2t_r5": 50, "i2t_r10": 65,
                                "t2i_r1": 22, "t2i_r5": 47, "t2i_r10": 62},
        }
        with open(os.path.join(fake_results_dir, f"{name}.json"), "w") as f:
            json.dump(fake, f)

    # Create fake ablation results
    abl_dir = os.path.join(fake_results_dir, "ablations")
    os.makedirs(abl_dir, exist_ok=True)
    for lam in [0.0, 0.1, 0.5, 1.0]:
        fake = {
            "run_name": f"lambda_{lam}",
            "lambda_recon": lam,
            "mask_ratio": 0.15,
            "final_retrieval": {"i2t_r1": 20 + lam * 5, "t2i_r1": 18 + lam * 4},
        }
        with open(os.path.join(abl_dir, f"lambda_{lam:.1f}.json"), "w") as f:
            json.dump(fake, f)

    for mr in [0.10, 0.15, 0.25]:
        fake = {
            "run_name": f"maskrate_{mr}",
            "mask_ratio": mr,
            "lambda_recon": 0.5,
            "final_retrieval": {"i2t_r1": 22 + mr * 10, "t2i_r1": 20 + mr * 8},
        }
        with open(os.path.join(abl_dir, f"maskrate_{mr:.2f}.json"), "w") as f:
            json.dump(fake, f)

    # Test all plot functions (they should not crash)
    from analyze_results import load_all_results, load_ablation_results
    main_results = load_all_results(fake_results_dir)
    ablation_results = load_ablation_results(fake_results_dir)

    plot_retrieval_comparison(main_results, fake_figures_dir)
    plot_loss_curves(main_results, fake_figures_dir)
    plot_lambda_sweep(ablation_results, fake_figures_dir)
    plot_maskrate_sweep(ablation_results, fake_figures_dir)
    plot_compositional({}, fake_figures_dir)  # empty is fine, should skip

    # Verify files were created
    expected_files = ["retrieval_comparison.png", "loss_curves.png",
                      "lambda_sweep.png", "maskrate_sweep.png"]
    for fname in expected_files:
        path = os.path.join(fake_figures_dir, fname)
        assert os.path.isfile(path), f"Missing plot: {fname}"


# ============================================================
# Runner
# ============================================================

def main():
    all_tests = [v for v in globals().values()
                 if callable(v) and hasattr(v, "test_name")]

    print("="*60)
    print("SuperCLIP-Recon Local Test Suite")
    print(f"Running {len(all_tests)} tests on CPU with synthetic data")
    print("="*60)

    for t in all_tests:
        t()

    # Summary
    print(f"\n{'='*60}")
    print("RESULTS")
    print(f"{'='*60}")
    passed = sum(1 for _, ok, _ in results if ok)
    total = len(results)
    for name, ok, err in results:
        status = "✓" if ok else "✗"
        print(f"  {status} {name}" + (f" — {err}" if err else ""))

    print(f"\n{passed}/{total} tests passed")

    # Cleanup
    if os.path.isdir("test_data"):
        shutil.rmtree("test_data")
        print("Cleaned up test_data/")

    if passed < total:
        sys.exit(1)
    else:
        print("\nAll clear — safe to submit to HPC.")


if __name__ == "__main__":
    main()
