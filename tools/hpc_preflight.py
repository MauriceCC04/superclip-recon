"""
HPC readiness preflight for SuperCLIP-Recon.

What it does:
    - static repo checks (required files exist)
    - import checks (torch, open_clip, etc.)
    - GPU checks (CUDA, device name, memory)
    - data / path / cache checks
    - storage footprint estimate vs $HOME quota
    - ONE tiny runtime step:
        * build model with synced vocab size
        * load one batch
        * forward + backward
        * tiny retrieval eval
        * tiny checkpoint write

Emits a machine-readable JSON report and exits nonzero on FAIL.

Usage:
    python tools/hpc_preflight.py \
        --coco_root ./data/coco \
        --vocab_path ./vocab.json \
        --phrase_path ./phrases.json \
        --home_quota_gb 100 \
        --output ./results/preflight/preflight_report.json
"""

import os
import sys
import json
import time
import shutil
import argparse
import traceback
from pathlib import Path

# Make the project root importable no matter where this is invoked from
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ─── Schema ──────────────────────────────────────────────────────────────────

REQUIRED_TOP_KEYS = {"overall_status", "checks", "metrics", "recommendations"}
REQUIRED_CHECK_KEYS = {"repo", "imports", "gpu", "data", "cache", "storage", "runtime"}
REQUIRED_METRIC_KEYS = {"gpu_peak_mem_gb", "checkpoint_size_mb",
                        "one_step_seconds", "tiny_eval_seconds"}
VALID_STATUSES = {"PASS", "WARN", "FAIL"}


def build_report_skeleton() -> dict:
    """
    Return a minimal but valid preflight report skeleton.
    Used by tests and as the starting point in main().
    """
    return {
        "overall_status": "PASS",
        "checks": {k: {"status": "PASS"} for k in REQUIRED_CHECK_KEYS},
        "metrics": {k: None for k in REQUIRED_METRIC_KEYS},
        "recommendations": [],
    }


def validate_report_schema(report: dict):
    """Validate report structure. Raises AssertionError on invalid report."""
    assert isinstance(report, dict), "Report is not a dict"
    assert REQUIRED_TOP_KEYS.issubset(report.keys()), \
        f"Missing top-level keys: {REQUIRED_TOP_KEYS - report.keys()}"
    assert report["overall_status"] in VALID_STATUSES, \
        f"Invalid overall_status: {report['overall_status']}"
    assert REQUIRED_CHECK_KEYS.issubset(report["checks"].keys()), \
        f"Missing check keys: {REQUIRED_CHECK_KEYS - report['checks'].keys()}"
    assert REQUIRED_METRIC_KEYS.issubset(report["metrics"].keys()), \
        f"Missing metric keys: {REQUIRED_METRIC_KEYS - report['metrics'].keys()}"
    assert isinstance(report["recommendations"], list), \
        "recommendations must be a list"
    return True


def estimate_storage(path: str) -> dict:
    """
    Test-friendly storage estimator.
    Returns {"total_bytes": int, "breakdown": {subdir_name: int_bytes}}.
    Handles missing directories gracefully.
    """
    if not os.path.isdir(path):
        return {"total_bytes": 0, "breakdown": {}}

    breakdown = {}
    total = 0
    try:
        for entry in os.scandir(path):
            size = 0
            try:
                if entry.is_file(follow_symlinks=False):
                    size = entry.stat().st_size
                elif entry.is_dir(follow_symlinks=False):
                    size = _dir_size_bytes(entry.path)
            except OSError:
                continue
            breakdown[entry.name] = size
            total += size
    except OSError:
        pass
    return {"total_bytes": total, "breakdown": breakdown}


def classify_storage(used_gb: float, quota_gb: float) -> str:
    """Return PASS/WARN/FAIL based on usage relative to quota."""
    if used_gb >= quota_gb:
        return "FAIL"
    if used_gb >= quota_gb * 0.9:
        return "WARN"
    return "PASS"


# ─── Storage estimator ───────────────────────────────────────────────────────

def _dir_size_bytes(path: str) -> int:
    """Return total size in bytes of all files under path."""
    total = 0
    for dirpath, _, filenames in os.walk(path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            try:
                total += os.path.getsize(fp)
            except OSError:
                pass
    return total


def estimate_storage_footprint(root: str) -> dict:
    """
    Estimate storage used under `root`, broken down by common subdirs.
    Returns sizes in MB.
    """
    result = {"path": root}
    subdirs = ["data", "checkpoints", "results", "logs",
               ".cache", "out", "err"]
    total = 0
    for sd in subdirs:
        p = os.path.join(root, sd)
        if os.path.isdir(p):
            s = _dir_size_bytes(p) / (1024 * 1024)
            result[f"{sd}_mb"] = round(s, 2)
            total += s
        else:
            result[f"{sd}_mb"] = 0.0

    # Total for the whole directory (catches anything we didn't enumerate)
    if os.path.isdir(root):
        total_all = _dir_size_bytes(root) / (1024 * 1024)
        result["total_mb"] = round(total_all, 2)
    else:
        result["total_mb"] = round(total, 2)
    return result


# ─── Individual checks ───────────────────────────────────────────────────────

def check_repo(project_root: str) -> dict:
    """Verify required repo files are present."""
    required = [
        "config.py", "model.py", "losses.py", "dataset.py",
        "train.py", "evaluate.py", "build_vocab.py",
        "sanity_check.py", "tests/smoke_test.py",
    ]
    missing = []
    present = []
    for f in required:
        p = os.path.join(project_root, f)
        if os.path.isfile(p):
            present.append(f)
        else:
            missing.append(f)
    status = "PASS" if not missing else "FAIL"
    return {
        "status": status,
        "present": present,
        "missing": missing,
    }


def check_imports() -> dict:
    """Verify the key Python packages can be imported."""
    pkgs = ["torch", "numpy", "PIL", "open_clip", "tqdm"]
    optional = ["datasets", "wandb", "matplotlib", "spacy"]
    ok = []
    fail = []
    for name in pkgs:
        try:
            __import__(name)
            ok.append(name)
        except Exception as e:
            fail.append({"pkg": name, "error": str(e)})
    optional_status = {}
    for name in optional:
        try:
            __import__(name)
            optional_status[name] = "ok"
        except Exception as e:
            optional_status[name] = f"missing ({e})"
    status = "PASS" if not fail else "FAIL"
    return {"status": status, "ok": ok, "fail": fail, "optional": optional_status}


def check_gpu() -> dict:
    """Inspect GPU availability."""
    try:
        import torch
    except ImportError:
        return {"status": "FAIL", "reason": "torch not importable"}

    info = {"cuda_available": torch.cuda.is_available()}
    if not torch.cuda.is_available():
        info["status"] = "WARN"
        info["reason"] = "No CUDA device visible — preflight will run on CPU"
        return info

    dev = torch.cuda.current_device()
    info["device_index"] = dev
    info["device_name"] = torch.cuda.get_device_name(dev)
    props = torch.cuda.get_device_properties(dev)
    info["total_memory_gb"] = round(props.total_memory / (1024 ** 3), 2)
    info["compute_capability"] = f"{props.major}.{props.minor}"
    info["status"] = "PASS"
    return info


def check_data(coco_root: str, vocab_path: str, phrase_path: str) -> dict:
    """Check for COCO data and project artefacts."""
    checks = {}
    train_dir = os.path.join(coco_root, "train2017")
    val_dir = os.path.join(coco_root, "val2017")
    ann_train = os.path.join(coco_root, "annotations", "captions_train2017.json")
    ann_val = os.path.join(coco_root, "annotations", "captions_val2017.json")

    checks["coco_train_dir"] = os.path.isdir(train_dir)
    checks["coco_val_dir"] = os.path.isdir(val_dir)
    checks["coco_ann_train"] = os.path.isfile(ann_train)
    checks["coco_ann_val"] = os.path.isfile(ann_val)
    checks["vocab_json"] = os.path.isfile(vocab_path)
    checks["phrase_json"] = os.path.isfile(phrase_path)

    missing_required = [k for k in ("coco_ann_train", "coco_ann_val", "vocab_json")
                        if not checks[k]]
    if missing_required:
        status = "FAIL"
    elif not checks["coco_train_dir"] or not checks["coco_val_dir"]:
        # Annotations present but images missing — warn
        status = "WARN"
    elif not checks["phrase_json"]:
        # Only needed for Variant B
        status = "WARN"
    else:
        status = "PASS"

    return {"status": status, "checks": checks}


def check_cache() -> dict:
    """Verify cache env vars are set somewhere writable."""
    vars_to_check = ["XDG_CACHE_HOME", "TORCH_HOME", "HF_HOME",
                     "TRANSFORMERS_CACHE", "HF_DATASETS_CACHE",
                     "WANDB_DIR", "MPLCONFIGDIR", "TMPDIR"]
    info = {}
    warnings = []
    for v in vars_to_check:
        val = os.environ.get(v)
        info[v] = val
        if val is None:
            warnings.append(f"{v} not set (will use system default)")
        elif not val.startswith(os.path.expanduser("~")) and not val.startswith("/tmp"):
            warnings.append(f"{v}={val} is not under $HOME or /tmp (may violate quota policy)")

    status = "PASS" if not warnings else "WARN"
    return {"status": status, "vars": info, "warnings": warnings}


def check_storage(project_root: str, home_quota_gb: float) -> dict:
    """Estimate footprint and compare to quota."""
    home = os.path.expanduser("~")
    home_use_mb = None
    try:
        home_use_mb = _dir_size_bytes(home) / (1024 * 1024)
    except Exception:
        pass

    footprint = estimate_storage_footprint(project_root)
    info = {
        "home_quota_gb": home_quota_gb,
        "project_footprint_mb": footprint["total_mb"],
        "project_breakdown_mb": {k: v for k, v in footprint.items()
                                 if k.endswith("_mb") and k != "total_mb"},
    }
    if home_use_mb is not None:
        info["home_usage_gb"] = round(home_use_mb / 1024, 2)
        info["home_quota_used_pct"] = round((home_use_mb / 1024) / home_quota_gb * 100, 1)
        if info["home_quota_used_pct"] > 85:
            info["status"] = "FAIL"
            info["reason"] = "Home directory is >85% full"
        elif info["home_quota_used_pct"] > 70:
            info["status"] = "WARN"
            info["reason"] = "Home directory is >70% full"
        else:
            info["status"] = "PASS"
    else:
        info["status"] = "WARN"
        info["reason"] = "Could not measure home directory usage"

    return info


def check_runtime(project_root: str, coco_root: str, vocab_path: str) -> dict:
    """
    Run one tiny forward+backward step and a tiny retrieval eval.
    Returns metrics: one_step_seconds, tiny_eval_seconds, gpu_peak_mem_gb,
    checkpoint_size_mb.
    """
    metrics = {
        "one_step_seconds": None,
        "tiny_eval_seconds": None,
        "gpu_peak_mem_gb": None,
        "checkpoint_size_mb": None,
    }
    info = {"metrics": metrics}

    # Need COCO to run this step; if missing, degrade to WARN.
    ann_train = os.path.join(coco_root, "annotations", "captions_train2017.json")
    if not os.path.isfile(ann_train) or not os.path.isfile(vocab_path):
        info["status"] = "WARN"
        info["reason"] = "Skipping runtime step — COCO or vocab not present"
        return info

    try:
        import torch
        from config import Config
        from model import SuperCLIPRecon
        from dataset import COCOCaptionsDataset
        from build_vocab import load_vocab
        from losses import build_token_labels, create_mask, total_loss
        from evaluate import run_retrieval_eval
        from torch.utils.data import DataLoader

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)

        cfg = Config()
        cfg.data.coco_root = coco_root
        cfg.data.num_workers = 0

        # Sync vocab size BEFORE building the model
        vocab_map = load_vocab(vocab_path)
        cfg.model.num_token_classes = len(vocab_map)

        model = SuperCLIPRecon(cfg).to(device)
        model.train()

        ds = COCOCaptionsDataset(
            root=cfg.data.coco_root,
            ann_file=cfg.data.train_ann,
            image_dir=cfg.data.train_images,
            transform=model.preprocess,
            tokenizer=model.tokenizer,
        )
        loader = DataLoader(ds, batch_size=4, shuffle=True, num_workers=0)
        images, token_ids, captions_raw, _ = next(iter(loader))

        max_masks = int(cfg.data.max_caption_length * cfg.model.mask_ratio) + 1
        _, mask_targets, _ = create_mask(token_ids, cfg.model.mask_ratio, max_masks)
        labels = build_token_labels(token_ids, vocab_map, cfg.model.num_token_classes)

        images = images.to(device)
        token_ids = token_ids.to(device)
        mask_targets = mask_targets.to(device)
        labels = labels.to(device)

        trainable = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(trainable, lr=1e-5)

        # --- One step ---
        t0 = time.time()
        outputs = model(images, token_ids, encode_text=False)
        loss, _ = total_loss(
            outputs["token_cls_logits"], labels,
            outputs["recon_logits"], mask_targets,
            lambda_recon=0.5,
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if device.type == "cuda":
            torch.cuda.synchronize()
        metrics["one_step_seconds"] = round(time.time() - t0, 3)

        # --- Tiny eval ---
        t0 = time.time()
        _ = run_retrieval_eval(model, cfg, device, max_images=16)
        if device.type == "cuda":
            torch.cuda.synchronize()
        metrics["tiny_eval_seconds"] = round(time.time() - t0, 3)

        # --- Tiny checkpoint ---
        tmp_ckpt = os.path.join(project_root, "results", "preflight",
                                "_tiny_ckpt.pt")
        os.makedirs(os.path.dirname(tmp_ckpt), exist_ok=True)
        torch.save({"model_state_dict": model.state_dict(), "config": cfg},
                   tmp_ckpt)
        metrics["checkpoint_size_mb"] = round(os.path.getsize(tmp_ckpt) / (1024 * 1024), 2)
        # Keep it so the user can see the real size; no deletion.

        if device.type == "cuda":
            metrics["gpu_peak_mem_gb"] = round(
                torch.cuda.max_memory_allocated(device) / (1024 ** 3), 3)
        else:
            metrics["gpu_peak_mem_gb"] = 0.0

        info["status"] = "PASS"

    except Exception as e:
        info["status"] = "FAIL"
        info["reason"] = str(e)
        info["traceback"] = traceback.format_exc()

    return info


# ─── Recommendation builder ──────────────────────────────────────────────────

def build_recommendations(report: dict) -> list:
    recs = []
    checks = report["checks"]
    metrics = report["metrics"]

    if checks["gpu"].get("status") == "WARN":
        recs.append("No GPU detected — preflight ran on CPU. "
                    "Re-run on the compute node via slurm/run_preflight.sh.")

    data = checks["data"].get("checks", {})
    if not data.get("coco_train_dir") or not data.get("coco_val_dir"):
        recs.append("COCO image folders missing. Run: bash download_coco.sh "
                    "(or slurm/setup_env_ready.sh --step data)")
    if not data.get("vocab_json"):
        recs.append("vocab.json missing. Run: python build_vocab.py "
                    "--coco_root ./data/coco --top_k 1000 --output vocab.json")
    if not data.get("phrase_json"):
        recs.append("phrases.json missing. Variant B will use inline extraction only. "
                    "To pre-extract: python extract_phrases.py --use_regex "
                    "--coco_root ./data/coco --output phrases.json")

    if checks["cache"].get("warnings"):
        recs.append("Cache env vars not fully configured — "
                    "source slurm/common.sh before running anything.")

    if checks["storage"].get("home_quota_used_pct", 0) > 70:
        recs.append("Home directory >70% full. "
                    "Enable --save_strategy last_and_best --keep_last_k 1.")

    ckpt_mb = metrics.get("checkpoint_size_mb") or 0
    if ckpt_mb > 0:
        projected = ckpt_mb * 10  # 10 epochs, naive upper bound
        recs.append(f"One checkpoint ≈ {ckpt_mb:.1f} MB. "
                    f"10-epoch baseline ≤ {projected:.0f} MB with retention=last_and_best; "
                    f"naively keeping all epochs would be ~{projected:.0f} MB per run.")

    step_s = metrics.get("one_step_seconds") or 0
    if step_s > 5:
        recs.append(f"One-step time is {step_s:.2f}s — unusually slow. "
                    "Check num_workers and that the GPU is being used.")

    return recs


# ─── Main ────────────────────────────────────────────────────────────────────

def compute_overall_status(report: dict) -> str:
    """FAIL if any check is FAIL; WARN if any is WARN; else PASS."""
    worst = "PASS"
    for _, c in report["checks"].items():
        s = c.get("status", "WARN")
        if s == "FAIL":
            return "FAIL"
        if s == "WARN":
            worst = "WARN"
    return worst


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--coco_root", type=str, default="./data/coco")
    parser.add_argument("--vocab_path", type=str, default="./vocab.json")
    parser.add_argument("--phrase_path", type=str, default="./phrases.json")
    parser.add_argument("--home_quota_gb", type=float, default=100.0)
    parser.add_argument("--output", type=str,
                        default="./results/preflight/preflight_report.json")
    parser.add_argument("--project_root", type=str, default=".")
    args = parser.parse_args()

    project_root = os.path.abspath(args.project_root)
    print(f"=== HPC Preflight ===")
    print(f"Project root: {project_root}")
    print(f"Output:       {args.output}")
    print()

    report = {"checks": {}, "metrics": {}, "recommendations": []}

    print("[1/7] Checking repo files...")
    report["checks"]["repo"] = check_repo(project_root)
    print(f"  -> {report['checks']['repo']['status']}")

    print("[2/7] Checking imports...")
    report["checks"]["imports"] = check_imports()
    print(f"  -> {report['checks']['imports']['status']}")

    print("[3/7] Checking GPU...")
    report["checks"]["gpu"] = check_gpu()
    print(f"  -> {report['checks']['gpu']['status']}")

    print("[4/7] Checking data...")
    report["checks"]["data"] = check_data(args.coco_root, args.vocab_path,
                                           args.phrase_path)
    print(f"  -> {report['checks']['data']['status']}")

    print("[5/7] Checking cache env vars...")
    report["checks"]["cache"] = check_cache()
    print(f"  -> {report['checks']['cache']['status']}")

    print("[6/7] Estimating storage...")
    report["checks"]["storage"] = check_storage(project_root, args.home_quota_gb)
    print(f"  -> {report['checks']['storage']['status']}")

    print("[7/7] Running tiny runtime step...")
    runtime = check_runtime(project_root, args.coco_root, args.vocab_path)
    report["checks"]["runtime"] = {k: v for k, v in runtime.items()
                                    if k != "metrics"}
    # Lift the metrics out
    rm = runtime.get("metrics", {})
    report["metrics"] = {
        "gpu_peak_mem_gb": rm.get("gpu_peak_mem_gb"),
        "checkpoint_size_mb": rm.get("checkpoint_size_mb"),
        "one_step_seconds": rm.get("one_step_seconds"),
        "tiny_eval_seconds": rm.get("tiny_eval_seconds"),
    }
    print(f"  -> {report['checks']['runtime']['status']}")

    report["overall_status"] = compute_overall_status(report)
    report["recommendations"] = build_recommendations(report)

    # --- Schema validation ---
    try:
        validate_report_schema(report)
    except AssertionError as e:
        print(f"[WARN] Report schema validation failed: {e}")

    # --- Write output ---
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(report, f, indent=2, default=str)

    print()
    print(f"=== Overall: {report['overall_status']} ===")
    for rec in report["recommendations"]:
        print(f"  * {rec}")
    print(f"\nReport: {args.output}")

    if report["overall_status"] == "FAIL":
        sys.exit(2)
    sys.exit(0)


if __name__ == "__main__":
    main()
