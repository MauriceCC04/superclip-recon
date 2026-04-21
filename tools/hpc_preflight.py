"""
HPC readiness preflight for SuperCLIP-Recon.
"""

import os
import sys
import json
import time
import argparse
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

REQUIRED_TOP_KEYS = {"overall_status", "checks", "metrics", "recommendations"}
REQUIRED_CHECK_KEYS = {"repo", "imports", "gpu", "data", "cache", "storage", "runtime"}
REQUIRED_METRIC_KEYS = {"gpu_peak_mem_gb", "checkpoint_size_mb", "one_step_seconds", "tiny_eval_seconds"}
VALID_STATUSES = {"PASS", "WARN", "FAIL"}


def build_report_skeleton() -> dict:
    return {
        "overall_status": "PASS",
        "checks": {key: {"status": "PASS"} for key in REQUIRED_CHECK_KEYS},
        "metrics": {key: None for key in REQUIRED_METRIC_KEYS},
        "recommendations": [],
    }


def validate_report_schema(report: dict):
    assert isinstance(report, dict), "Report is not a dict"
    assert REQUIRED_TOP_KEYS.issubset(report.keys())
    assert report["overall_status"] in VALID_STATUSES
    assert REQUIRED_CHECK_KEYS.issubset(report["checks"].keys())
    assert REQUIRED_METRIC_KEYS.issubset(report["metrics"].keys())
    assert isinstance(report["recommendations"], list)
    return True


def estimate_storage(path: str) -> dict:
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
    if used_gb >= quota_gb:
        return "FAIL"
    if used_gb >= quota_gb * 0.9:
        return "WARN"
    return "PASS"


def _dir_size_bytes(path: str) -> int:
    total = 0
    for dirpath, _, filenames in os.walk(path):
        for filename in filenames:
            file_path = os.path.join(dirpath, filename)
            try:
                total += os.path.getsize(file_path)
            except OSError:
                pass
    return total


def estimate_storage_footprint(root: str) -> dict:
    result = {"path": root}
    subdirs = ["data", "checkpoints", "results", "logs", ".cache", "out", "err"]
    total = 0
    for subdir in subdirs:
        path = os.path.join(root, subdir)
        if os.path.isdir(path):
            size_mb = _dir_size_bytes(path) / (1024 * 1024)
            result[f"{subdir}_mb"] = round(size_mb, 2)
            total += size_mb
        else:
            result[f"{subdir}_mb"] = 0.0
    result["total_mb"] = round((_dir_size_bytes(root) / (1024 * 1024)) if os.path.isdir(root) else total, 2)
    return result


def check_repo(project_root: str) -> dict:
    required = [
        "config.py", "model.py", "losses.py", "dataset.py",
        "train.py", "evaluate.py", "build_vocab.py", "sanity_check.py", "tests/smoke_test.py",
    ]
    present = []
    missing = []
    for rel_path in required:
        path = os.path.join(project_root, rel_path)
        if os.path.isfile(path):
            present.append(rel_path)
        else:
            missing.append(rel_path)
    return {"status": "PASS" if not missing else "FAIL", "present": present, "missing": missing}


def check_imports() -> dict:
    pkgs = ["torch", "numpy", "PIL", "open_clip", "tqdm"]
    optional = ["datasets", "wandb", "matplotlib", "spacy"]
    ok = []
    fail = []
    for name in pkgs:
        try:
            __import__(name)
            ok.append(name)
        except Exception as exc:
            fail.append({"pkg": name, "error": str(exc)})
    optional_status = {}
    for name in optional:
        try:
            __import__(name)
            optional_status[name] = "ok"
        except Exception as exc:
            optional_status[name] = f"missing ({exc})"
    return {"status": "PASS" if not fail else "FAIL", "ok": ok, "fail": fail, "optional": optional_status}


def check_gpu() -> dict:
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
    props = torch.cuda.get_device_properties(dev)
    info.update(
        {
            "status": "PASS",
            "device_index": dev,
            "device_name": torch.cuda.get_device_name(dev),
            "total_memory_gb": round(props.total_memory / (1024 ** 3), 2),
            "compute_capability": f"{props.major}.{props.minor}",
        }
    )
    return info


def check_data(coco_root: str, vocab_path: str, phrase_path: str) -> dict:
    checks = {
        "coco_train_dir": os.path.isdir(os.path.join(coco_root, "train2017")),
        "coco_val_dir": os.path.isdir(os.path.join(coco_root, "val2017")),
        "coco_ann_train": os.path.isfile(os.path.join(coco_root, "annotations", "captions_train2017.json")),
        "coco_ann_val": os.path.isfile(os.path.join(coco_root, "annotations", "captions_val2017.json")),
        "vocab_json": os.path.isfile(vocab_path),
        "phrase_json": os.path.isfile(phrase_path),
    }
    missing_required = [key for key in ("coco_ann_train", "coco_ann_val", "vocab_json") if not checks[key]]
    if missing_required:
        status = "FAIL"
    elif not checks["coco_train_dir"] or not checks["coco_val_dir"] or not checks["phrase_json"]:
        status = "WARN"
    else:
        status = "PASS"
    return {"status": status, "checks": checks}


def check_cache() -> dict:
    vars_to_check = ["XDG_CACHE_HOME", "TORCH_HOME", "HF_HOME", "TRANSFORMERS_CACHE", "HF_DATASETS_CACHE", "WANDB_DIR", "MPLCONFIGDIR", "TMPDIR"]
    info = {}
    warnings = []
    for name in vars_to_check:
        value = os.environ.get(name)
        info[name] = value
        if value is None:
            warnings.append(f"{name} not set (will use system default)")
        elif not value.startswith(os.path.expanduser("~")) and not value.startswith("/tmp"):
            warnings.append(f"{name}={value} is not under $HOME or /tmp")
    return {"status": "PASS" if not warnings else "WARN", "vars": info, "warnings": warnings}


def check_storage(project_root: str, home_quota_gb: float) -> dict:
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
        "project_breakdown_mb": {k: v for k, v in footprint.items() if k.endswith("_mb") and k != "total_mb"},
    }
    if home_use_mb is None:
        info["status"] = "WARN"
        info["reason"] = "Could not measure home directory usage"
        return info
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
    return info


def check_runtime(project_root: str, coco_root: str, vocab_path: str) -> dict:
    metrics = {
        "one_step_seconds": None,
        "tiny_eval_seconds": None,
        "gpu_peak_mem_gb": None,
        "checkpoint_size_mb": None,
    }
    info = {"metrics": metrics}

    if not os.path.isfile(os.path.join(coco_root, "annotations", "captions_train2017.json")) or not os.path.isfile(vocab_path):
        info["status"] = "WARN"
        info["reason"] = "Skipping runtime step — COCO or vocab not present"
        return info

    try:
        import torch
        from torch.utils.data import DataLoader
        from torch.cuda.amp import autocast, GradScaler
        from contextlib import nullcontext
        from config import Config
        from model import SuperCLIPRecon
        from dataset import COCOCaptionsDataset
        from build_vocab import load_vocab
        from losses import build_token_labels, create_mask, total_loss
        from evaluate import run_retrieval_eval

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        amp_enabled = device.type == "cuda"
        scaler = GradScaler(enabled=amp_enabled)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)

        cfg = Config()
        cfg.data.coco_root = coco_root
        cfg.data.num_workers = 0
        cfg.train.train_mode = "superclip_baseline"
        cfg.train.lambda_recon = 0.0
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

        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-5)

        t0 = time.time()
        amp_ctx = autocast if amp_enabled else nullcontext
        with amp_ctx(enabled=amp_enabled) if amp_enabled else amp_ctx():
            outputs = model(images, token_ids, encode_text=True)
            loss, _ = total_loss(
                train_mode=cfg.train.train_mode,
                image_features=outputs["image_features"],
                text_features=outputs["text_features"],
                logit_scale=outputs["logit_scale"],
                token_cls_logits=outputs["token_cls_logits"],
                token_cls_labels=labels,
                recon_logits=outputs["recon_logits"],
                mask_targets=mask_targets,
                lambda_clip=cfg.train.lambda_clip,
                lambda_token_cls=cfg.train.lambda_token_cls,
                lambda_recon=cfg.train.lambda_recon,
                token_cls_freq=outputs["token_cls_freq"],
                token_cls_num_updates=outputs["token_cls_num_updates"],
                token_cls_use_reweighting=cfg.model.token_cls_use_reweighting,
            )
        optimizer.zero_grad(set_to_none=True)
        if amp_enabled:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        if device.type == "cuda":
            torch.cuda.synchronize()
        metrics["one_step_seconds"] = round(time.time() - t0, 3)

        t0 = time.time()
        _ = run_retrieval_eval(model, cfg, device, max_images=16)
        if device.type == "cuda":
            torch.cuda.synchronize()
        metrics["tiny_eval_seconds"] = round(time.time() - t0, 3)

        tmp_ckpt = os.path.join(project_root, "results", "preflight", "_tiny_ckpt.pt")
        os.makedirs(os.path.dirname(tmp_ckpt), exist_ok=True)
        torch.save({"model_state_dict": model.state_dict(), "config": cfg}, tmp_ckpt)
        metrics["checkpoint_size_mb"] = round(os.path.getsize(tmp_ckpt) / (1024 * 1024), 2)
        metrics["gpu_peak_mem_gb"] = round(torch.cuda.max_memory_allocated(device) / (1024 ** 3), 3) if device.type == "cuda" else 0.0
        info["status"] = "PASS"
    except Exception as exc:
        info["status"] = "FAIL"
        info["reason"] = str(exc)
        info["traceback"] = traceback.format_exc()

    return info


def build_recommendations(report: dict) -> list:
    recs = []
    checks = report["checks"]
    metrics = report["metrics"]
    if checks["gpu"].get("status") == "WARN":
        recs.append("No GPU detected — re-run preflight on a compute node.")
    data = checks["data"].get("checks", {})
    if not data.get("coco_train_dir") or not data.get("coco_val_dir"):
        recs.append("COCO image folders missing. Run slurm/setup_env_ready.sh --step data")
    if not data.get("vocab_json"):
        recs.append("vocab.json missing. Run python build_vocab.py --coco_root ./data/coco --top_k 1000 --output vocab.json")
    if not data.get("phrase_json"):
        recs.append("phrases.json missing. Variant B will still work via inline phrase extraction.")
    if checks["cache"].get("warnings"):
        recs.append("Cache env vars not fully configured — source slurm/common.sh before cluster runs.")
    if checks["storage"].get("home_quota_used_pct", 0) > 70:
        recs.append("Home directory >70% full. Keep save_strategy=last_and_best and keep_last_k=1.")
    ckpt_mb = metrics.get("checkpoint_size_mb") or 0
    if ckpt_mb > 0:
        projected = ckpt_mb * 10
        recs.append(
            f"One checkpoint ≈ {ckpt_mb:.1f} MB. 10 epochs would be ≈ {projected:.0f} MB if you kept every checkpoint."
        )
    return recs


def compute_overall_status(report: dict) -> str:
    worst = "PASS"
    for value in report["checks"].values():
        status = value.get("status", "WARN")
        if status == "FAIL":
            return "FAIL"
        if status == "WARN":
            worst = "WARN"
    return worst


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--coco_root", type=str, default="./data/coco")
    parser.add_argument("--vocab_path", type=str, default="./vocab.json")
    parser.add_argument("--phrase_path", type=str, default="./phrases.json")
    parser.add_argument("--home_quota_gb", type=float, default=100.0)
    parser.add_argument("--output", type=str, default="./results/preflight/preflight_report.json")
    parser.add_argument("--project_root", type=str, default=".")
    args = parser.parse_args()

    project_root = os.path.abspath(args.project_root)
    print("=== HPC Preflight ===")
    print(f"Project root: {project_root}")
    print(f"Output:       {args.output}")
    print()

    report = build_report_skeleton()
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
    report["checks"]["data"] = check_data(args.coco_root, args.vocab_path, args.phrase_path)
    print(f"  -> {report['checks']['data']['status']}")
    print("[5/7] Checking cache env vars...")
    report["checks"]["cache"] = check_cache()
    print(f"  -> {report['checks']['cache']['status']}")
    print("[6/7] Estimating storage...")
    report["checks"]["storage"] = check_storage(project_root, args.home_quota_gb)
    print(f"  -> {report['checks']['storage']['status']}")
    print("[7/7] Running tiny runtime step...")
    runtime = check_runtime(project_root, args.coco_root, args.vocab_path)
    report["checks"]["runtime"] = {k: v for k, v in runtime.items() if k != "metrics"}
    report["metrics"] = runtime.get("metrics", {})
    print(f"  -> {report['checks']['runtime']['status']}")
    report["overall_status"] = compute_overall_status(report)
    report["recommendations"] = build_recommendations(report)

    try:
        validate_report_schema(report)
    except AssertionError as exc:
        print(f"[WARN] Report schema validation failed: {exc}")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as handle:
        json.dump(report, handle, indent=2, default=str)

    print()
    print(f"=== Overall: {report['overall_status']} ===")
    for rec in report["recommendations"]:
        print(f"  * {rec}")
    print(f"\nReport: {args.output}")
    sys.exit(2 if report["overall_status"] == "FAIL" else 0)


if __name__ == "__main__":
    main()
