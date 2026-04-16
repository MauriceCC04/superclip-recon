"""
Local test suite for SuperCLIP-Recon.

Runs ALL code paths on tiny synthetic data, on CPU, with no real COCO download.
Catches shape mismatches, import errors, logic bugs, and pipeline breaks
BEFORE you waste HPC queue time.

Usage:
    cd superclip-recon
    python tests/run_tests.py
"""

import os
import sys
import json
import traceback
import shutil
import subprocess

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

TEST_DATA_ROOT = "test_data/coco"
TEST_VOCAB_PATH = "test_data/vocab.json"
TEST_PHRASE_PATH = "test_data/phrases.json"
DEVICE = torch.device("cpu")

results = []


def test(name):
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


@test("01. Create synthetic test data")
def test_create_data():
    from tests.create_test_data import create_test_data
    create_test_data(root=TEST_DATA_ROOT, n_train=16, n_val=8)
    assert os.path.isdir(f"{TEST_DATA_ROOT}/train2017")
    assert os.path.isdir(f"{TEST_DATA_ROOT}/val2017")
    assert os.path.isfile(f"{TEST_DATA_ROOT}/annotations/captions_train2017.json")
    assert len(os.listdir(f"{TEST_DATA_ROOT}/train2017")) == 16


@test("02. Build vocabulary")
def test_build_vocab():
    from build_vocab import build_vocab, load_vocab
    vocab_map = build_vocab(TEST_DATA_ROOT, top_k=50, output_path=TEST_VOCAB_PATH)
    assert 0 < len(vocab_map) <= 50
    loaded = load_vocab(TEST_VOCAB_PATH)
    assert len(loaded) == len(vocab_map)
    for k, v in loaded.items():
        assert isinstance(k, int) and isinstance(v, int)


@test("03. Dataset loading")
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
        root=TEST_DATA_ROOT, ann_file=cfg.data.train_ann,
        image_dir=cfg.data.train_images,
        transform=model.preprocess, tokenizer=model.tokenizer,
    )
    assert len(ds) == 16

    loader = DataLoader(ds, batch_size=4, shuffle=True, num_workers=0)
    images, token_ids, captions, img_ids = next(iter(loader))
    assert images.shape == (4, 3, 224, 224)
    assert token_ids.shape == (4, 77)
    assert len(captions) == 4


@test("04. Model forward pass (with text encoding)")
def test_model_forward():
    from config import Config
    from model import SuperCLIPRecon

    cfg = Config()
    cfg.model.num_token_classes = 50
    model = SuperCLIPRecon(cfg).to(DEVICE)

    B = 4
    images = torch.randn(B, 3, 224, 224)
    token_ids = torch.randint(1, 49405, (B, 77))
    outputs = model(images, token_ids)

    assert outputs["image_features"].shape == (B, 512)
    assert outputs["text_features"].shape == (B, 512)
    assert outputs["token_cls_logits"].shape == (B, 50)
    max_masks = int(77 * cfg.model.mask_ratio) + 1
    assert outputs["recon_logits"].shape == (B, max_masks, cfg.model.recon_vocab_size)


@test("04b. Model forward skipping text encoding (training fast path)")
def test_model_forward_no_text():
    """model.forward must support encode_text=False so training can skip
    the text encoder (text_features are not used in the loss)."""
    from config import Config
    from model import SuperCLIPRecon

    cfg = Config()
    cfg.model.num_token_classes = 50
    model = SuperCLIPRecon(cfg).to(DEVICE)

    B = 4
    images = torch.randn(B, 3, 224, 224)
    token_ids = torch.randint(1, 49405, (B, 77))
    outputs = model(images, token_ids, encode_text=False)

    assert outputs["image_features"].shape == (B, 512)
    assert outputs["text_features"] is None, \
        "text_features MUST be None when encode_text=False"
    assert outputs["token_cls_logits"].shape == (B, 50)


@test("05. Token classification labels")
def test_token_labels():
    from losses import build_token_labels
    from build_vocab import load_vocab

    vocab_map = load_vocab(TEST_VOCAB_PATH)
    token_ids = torch.randint(1, 49405, (4, 77))
    labels = build_token_labels(token_ids, vocab_map, num_classes=50)
    assert labels.shape == (4, 50)
    assert labels.dtype == torch.float32


@test("06. Masking — Variant A (random tokens)")
def test_mask_variant_a():
    from losses import create_mask

    token_ids = torch.randint(1, 49405, (4, 77))
    token_ids[:, 0] = 49406
    for i in range(4):
        token_ids[i, 10:] = 49407

    max_masks = 12
    masked, targets, positions = create_mask(token_ids, mask_ratio=0.15, max_masks=max_masks)
    assert masked.shape == token_ids.shape
    assert targets.shape == (4, max_masks)
    assert (targets != 0).sum().item() > 0


@test("06b. Mask positions are sorted (create_mask)")
def test_mask_positions_sorted():
    """Slot k must correspond to the k-th masked token in caption order."""
    from losses import create_mask

    torch.manual_seed(0)
    token_ids = torch.randint(100, 49000, (8, 77))
    token_ids[:, 0] = 49406
    for i in range(8):
        token_ids[i, 40:] = 49407

    max_masks = 12
    _, targets, positions = create_mask(token_ids, mask_ratio=0.3, max_masks=max_masks)

    for i in range(8):
        valid = [positions[i, k].item() for k in range(max_masks)
                 if positions[i, k].item() >= 0]
        assert valid == sorted(valid), \
            f"Sample {i}: positions not sorted: {valid}"


@test("06c. Mask positions are sorted (inline phrase extraction fallback)")
def test_phrase_mask_positions_sorted():
    from losses import create_phrase_mask_from_captions
    import open_clip

    tokenizer = open_clip.get_tokenizer("ViT-B-32")
    captions = ["runs jumps", "walks flies", "sings dances", "goes leaves"]
    token_ids = torch.cat([tokenizer(c) for c in captions], dim=0)

    _, targets, positions = create_phrase_mask_from_captions(
        token_ids, captions, tokenizer, max_masks=12
    )

    for i in range(4):
        valid = [positions[i, k].item() for k in range(12)
                 if positions[i, k].item() >= 0]
        assert valid == sorted(valid), \
            f"Sample {i}: positions not sorted: {valid}"


@test("07. Masking — Variant B (image-level phrase)")
def test_mask_variant_b():
    from losses import create_phrase_mask
    import open_clip

    tokenizer = open_clip.get_tokenizer("ViT-B-32")
    caption = "a red car parked on the street"
    token_ids = tokenizer(caption).repeat(4, 1)

    phrase_toks = tokenizer("red car").squeeze(0).tolist()
    content_toks = [t for t in phrase_toks if t not in (49406, 49407, 0)]
    phrase_data = {
        str(i): [{"phrase": "red car", "token_ids": content_toks}]
        for i in range(1, 5)
    }

    masked, targets, positions = create_phrase_mask(
        token_ids, phrase_data, [1, 2, 3, 4], max_masks=12
    )
    assert (targets != 0).sum().item() > 0


@test("07b. Masking — Variant B (per-caption inline)")
def test_mask_variant_b_inline():
    from losses import create_phrase_mask_from_captions
    import open_clip

    tokenizer = open_clip.get_tokenizer("ViT-B-32")
    captions = [
        "a red car parked on the street",
        "two black dogs playing in the park",
        "a woman riding a brown horse",
        "the tall building near the river",
    ]
    token_ids = torch.cat([tokenizer(c) for c in captions], dim=0)

    masked, targets, positions = create_phrase_mask_from_captions(
        token_ids, captions, tokenizer, max_masks=12
    )

    n_masked = (targets != 0).sum().item()
    assert n_masked > 0

    # Targets must match token_ids at each masked position
    for i in range(4):
        for k in range(12):
            pos = positions[i, k].item()
            if pos >= 0:
                assert targets[i, k].item() == token_ids[i, pos].item()
                assert masked[i, pos].item() == 0


@test("08. Loss + backward")
def test_loss_backward():
    from config import Config
    from model import SuperCLIPRecon
    from losses import build_token_labels, create_mask, total_loss
    from build_vocab import load_vocab

    cfg = Config()
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

    outputs = model(images, token_ids, encode_text=False)
    labels = build_token_labels(token_ids, vocab_map, 50)
    loss, _ = total_loss(
        outputs["token_cls_logits"], labels,
        outputs["recon_logits"], mask_targets, 0.5,
    )
    assert loss.requires_grad
    loss.backward()


@test("09. Phrase extraction")
def test_phrase_extraction():
    from extract_phrases import extract_with_regex, tokenize_phrases
    from collections import defaultdict
    import open_clip

    captions_by_id = defaultdict(list)
    captions_by_id[1] = ["a red car parked on the street"]
    captions_by_id[2] = ["two black dogs playing in the park"]

    phrases = extract_with_regex(captions_by_id)
    assert len(phrases) > 0

    tokenizer = open_clip.get_tokenizer("ViT-B-32")
    tokenized = tokenize_phrases(phrases, tokenizer)
    assert sum(len(v) for v in tokenized.values()) > 0

    with open(TEST_PHRASE_PATH, "w") as f:
        json.dump(tokenized, f)


@test("10. E2E mini training + retrieval metrics")
def test_e2e():
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

    model = SuperCLIPRecon(cfg).to(DEVICE)
    vocab_map = load_vocab(TEST_VOCAB_PATH)
    max_masks = int(77 * cfg.model.mask_ratio) + 1

    ds = COCOCaptionsDataset(
        root=TEST_DATA_ROOT, ann_file=cfg.data.train_ann,
        image_dir=cfg.data.train_images,
        transform=model.preprocess, tokenizer=model.tokenizer,
    )
    loader = DataLoader(ds, batch_size=4, shuffle=True, num_workers=0, drop_last=True)

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=1e-5)

    model.train()
    images, token_ids, _, _ = next(iter(loader))
    _, mask_targets, _ = create_mask(token_ids, cfg.model.mask_ratio, max_masks)
    outputs = model(images, token_ids, encode_text=False)
    labels = build_token_labels(token_ids, vocab_map, 50)
    loss, _ = total_loss(
        outputs["token_cls_logits"], labels,
        outputs["recon_logits"], mask_targets, 0.5,
    )
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    N = 8
    img_embs = np.random.randn(N, 512).astype(np.float32)
    img_embs = img_embs / np.linalg.norm(img_embs, axis=1, keepdims=True)
    txt_embs = np.random.randn(N * 5, 512).astype(np.float32)
    txt_embs = txt_embs / np.linalg.norm(txt_embs, axis=1, keepdims=True)
    metrics = compute_retrieval_metrics(img_embs, txt_embs)
    for k in ["i2t_r1", "i2t_r5", "i2t_r10", "t2i_r1", "t2i_r5", "t2i_r10"]:
        assert k in metrics


@test("11. Compositional eval imports")
def test_compositional_eval():
    from eval_compositional import evaluate_aro, run_compositional_eval
    from config import Config
    from model import SuperCLIPRecon

    cfg = Config()
    cfg.model.num_token_classes = 50
    model = SuperCLIPRecon(cfg).to(DEVICE)

    metrics = run_compositional_eval(model, DEVICE, benchmarks="winoground", hf_token=None)
    assert isinstance(metrics, dict)
    assert callable(evaluate_aro)


@test("12. analyze_results filters non-main JSON files")
def test_analyze_results_filter():
    """load_all_results must ignore compositional_*.json, summary.json,
    preflight_*.json, smoke_*.json, and files missing required schema."""
    from analyze_results import load_all_results, _is_main_result

    fake_dir = "test_data/results_filter"
    os.makedirs(fake_dir, exist_ok=True)

    with open(os.path.join(fake_dir, "baseline.json"), "w") as f:
        json.dump({
            "run_name": "baseline", "variant": "A", "lambda_recon": 0.0,
            "mask_ratio": 0.15, "final_retrieval": {"i2t_r1": 25.0},
        }, f)
    with open(os.path.join(fake_dir, "compositional_baseline.json"), "w") as f:
        json.dump({"winoground_group_score": 10.0}, f)
    with open(os.path.join(fake_dir, "smoke_results.json"), "w") as f:
        json.dump({"run_name": "smoke", "retrieval": {}}, f)
    with open(os.path.join(fake_dir, "preflight_report.json"), "w") as f:
        json.dump({"overall_status": "PASS"}, f)
    with open(os.path.join(fake_dir, "summary.json"), "w") as f:
        json.dump([{"x": 1}], f)
    with open(os.path.join(fake_dir, "random_thing.json"), "w") as f:
        json.dump({"unrelated": True}, f)

    loaded = load_all_results(fake_dir)

    assert "baseline" in loaded
    assert "compositional_baseline" not in loaded, \
        "compositional files must NOT be loaded as main results"
    assert "smoke_results" not in loaded
    assert "preflight_report" not in loaded
    assert "summary" not in loaded
    assert "random_thing" not in loaded, \
        "files lacking run_name/variant/lambda_recon must NOT be loaded as main"

    # Predicate tests
    assert _is_main_result("baseline.json",
                           {"run_name": "x", "variant": "A", "lambda_recon": 0.0,
                            "final_retrieval": {}}) is True
    assert _is_main_result("compositional_x.json", {"anything": 1}) is False
    assert _is_main_result("smoke_x.json", {"run_name": "x"}) is False


@test("13. Checkpoint retention policy")
def test_checkpoint_retention():
    """manage_checkpoints must delete old checkpoints while always keeping
    current epoch + best + last_k, per save_strategy."""
    from train import manage_checkpoints

    save_dir = "test_data/ckpt_retention"
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)
    os.makedirs(save_dir)

    for epoch in range(1, 6):
        path = os.path.join(save_dir, f"epoch_{epoch}.pt")
        torch.save({"epoch": epoch, "dummy": torch.zeros(10)}, path)

    # last_and_best, current=5, best=3, keep_last_k=1 → keep {3, 5}
    manage_checkpoints(save_dir, "last_and_best", keep_last_k=1,
                       current_epoch=5, best_metric_epoch=3)
    remaining = sorted(os.listdir(save_dir))
    assert "epoch_3.pt" in remaining
    assert "epoch_5.pt" in remaining
    assert "epoch_1.pt" not in remaining
    assert "epoch_2.pt" not in remaining
    assert "epoch_4.pt" not in remaining

    # all: keep everything
    shutil.rmtree(save_dir); os.makedirs(save_dir)
    for epoch in range(1, 4):
        torch.save({"epoch": epoch}, os.path.join(save_dir, f"epoch_{epoch}.pt"))
    manage_checkpoints(save_dir, "all", keep_last_k=0,
                       current_epoch=3, best_metric_epoch=1)
    assert len(os.listdir(save_dir)) == 3

    # last: only keep current
    shutil.rmtree(save_dir); os.makedirs(save_dir)
    for epoch in range(1, 5):
        torch.save({"epoch": epoch}, os.path.join(save_dir, f"epoch_{epoch}.pt"))
    manage_checkpoints(save_dir, "last", keep_last_k=0,
                       current_epoch=4, best_metric_epoch=2)
    assert os.listdir(save_dir) == ["epoch_4.pt"]


@test("14. Repo root resolution via slurm/common.sh")
def test_repo_root_resolution():
    """slurm scripts must resolve PROJECT_ROOT from common.sh location,
    not from CWD. Sourcing common.sh from /tmp must still set PROJECT_ROOT
    to the repo root."""
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    common_sh = os.path.join(project_root, "slurm", "common.sh")
    assert os.path.isfile(common_sh), f"slurm/common.sh missing: {common_sh}"

    result = subprocess.run(
        ["bash", "-c", f'source "{common_sh}" >/dev/null 2>&1 && echo "$PROJECT_ROOT"'],
        capture_output=True, text=True, cwd="/tmp",
    )
    assert result.returncode == 0, f"common.sh failed: {result.stderr}"
    resolved = result.stdout.strip().splitlines()[-1]
    assert os.path.realpath(resolved) == os.path.realpath(project_root), \
        f"PROJECT_ROOT resolved to {resolved}, expected {project_root}"


@test("15. Preflight JSON schema")
def test_preflight_schema():
    """hpc_preflight must define a report skeleton with all required keys."""
    from tools.hpc_preflight import build_report_skeleton, validate_report_schema

    report = build_report_skeleton()
    validate_report_schema(report)  # must not raise

    assert report["overall_status"] in ("PASS", "WARN", "FAIL")
    for key in ("repo", "imports", "gpu", "data", "cache", "storage", "runtime"):
        assert key in report["checks"]
    for key in ("gpu_peak_mem_gb", "checkpoint_size_mb",
                "one_step_seconds", "tiny_eval_seconds"):
        assert key in report["metrics"]


@test("16. Preflight storage estimator")
def test_preflight_storage_estimator():
    """The estimator must not crash on missing dirs and must classify
    usage against a quota."""
    from tools.hpc_preflight import estimate_storage, classify_storage

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    est = estimate_storage(project_root)
    assert est["total_bytes"] >= 0
    assert isinstance(est["breakdown"], dict)

    est = estimate_storage("/definitely/does/not/exist/xyz123")
    assert est["total_bytes"] == 0

    assert classify_storage(used_gb=10, quota_gb=100) == "PASS"
    assert classify_storage(used_gb=92, quota_gb=100) == "WARN"
    assert classify_storage(used_gb=101, quota_gb=100) == "FAIL"


@test("17. Slurm scripts parse and source common.sh")
def test_slurm_scripts_parseable():
    """All slurm/*.sh must have valid bash syntax and must source common.sh
    (except common.sh itself)."""
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    slurm_dir = os.path.join(project_root, "slurm")

    shs = [f for f in os.listdir(slurm_dir) if f.endswith(".sh")]
    assert "common.sh" in shs, "slurm/common.sh missing"

    for fname in shs:
        path = os.path.join(slurm_dir, fname)
        # Syntax check
        result = subprocess.run(["bash", "-n", path], capture_output=True, text=True)
        assert result.returncode == 0, \
            f"{fname} has bash syntax errors:\n{result.stderr}"

        if fname == "common.sh":
            continue
        with open(path) as f:
            content = f.read()
        assert "common.sh" in content, \
            f"{fname} does not source common.sh — will break if sbatch'd from wrong CWD"


def main():
    all_tests = [v for v in globals().values()
                 if callable(v) and hasattr(v, "test_name")]
    all_tests.sort(key=lambda f: f.test_name)

    print("=" * 60)
    print("SuperCLIP-Recon Local Test Suite")
    print(f"Running {len(all_tests)} tests on CPU with synthetic data")
    print("=" * 60)

    for t in all_tests:
        t()

    print(f"\n{'='*60}")
    print("RESULTS")
    print(f"{'='*60}")
    passed = sum(1 for _, ok, _ in results if ok)
    total = len(results)
    for name, ok, err in results:
        status = "✓" if ok else "✗"
        print(f"  {status} {name}" + (f" — {err}" if err else ""))

    print(f"\n{passed}/{total} tests passed")

    if os.path.isdir("test_data"):
        shutil.rmtree("test_data")
        print("Cleaned up test_data/")

    if passed < total:
        sys.exit(1)
    print("\nAll clear — safe to submit to HPC.")


if __name__ == "__main__":
    main()
