from __future__ import annotations

import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


MAIN_BATCH_SIZE = 128
MAIN_EPOCHS = 10
MAIN_VARIANT = "A"
MAIN_LAMBDA = 1.0


def find_repo_root(start: str | Path | None = None) -> Path:
    """Find the repo root by looking for a top-level results/ directory."""
    start_path = Path(start or ".").resolve()
    for candidate in [start_path, *start_path.parents]:
        if (candidate / "results").is_dir():
            return candidate
    raise FileNotFoundError(
        f"Could not find a repo root above {start_path} containing a results/ directory."
    )


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_json(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _preferred_rank(path: Path) -> Tuple[int, str]:
    text = str(path).replace("\\", "/")
    for idx, token in enumerate([
        "/final_confirm/",
        "/compositional_round2/",
        "/confirm6/",
        "/final_checks/",
        "/winoground/",
        "/compositional/",
    ]):
        if token in text:
            return (idx, text)
    return (999, text)


def _pick_preferred(entries: Iterable[dict]) -> dict:
    entries = list(entries)
    if not entries:
        raise ValueError("No entries to choose from.")
    entries.sort(key=lambda x: _preferred_rank(Path(x["path"])))
    return entries[0]


def _close(a: float | None, b: float, tol: float = 1e-9) -> bool:
    return a is not None and abs(a - b) <= tol


def scan_retrieval_runs(repo_root: str | Path) -> List[dict]:
    repo_root = Path(repo_root)
    results_dir = repo_root / "results"
    runs: List[dict] = []
    for path in results_dir.rglob("*.json"):
        try:
            data = load_json(path)
        except Exception:
            continue
        if not isinstance(data, dict):
            continue
        if "history" not in data or "final_retrieval" not in data:
            continue
        runs.append(
            {
                "path": str(path),
                "run_name": data.get("run_name", path.stem),
                "seed": data.get("seed"),
                "variant": data.get("variant"),
                "effective_variant": data.get("effective_variant"),
                "requested_train_mode": data.get("requested_train_mode"),
                "train_mode": data.get("train_mode"),
                "recon_enabled": data.get("recon_enabled"),
                "lambda_recon": data.get("lambda_recon"),
                "mask_ratio": data.get("mask_ratio"),
                "batch_size": data.get("batch_size"),
                "epochs": data.get("epochs"),
                "best_epoch": data.get("best_epoch"),
                "best_retrieval_score": data.get("best_retrieval_score"),
                "best_retrieval": data.get("best_retrieval"),
                "final_retrieval": data.get("final_retrieval"),
                "history": data.get("history", []),
                "wall_time_seconds": data.get("wall_time_seconds"),
            }
        )
    return runs


def get_eval_retrieval(run: dict) -> dict:
    return run.get("best_retrieval") or run.get("final_retrieval") or {}


def retrieval_score_from_metrics(metrics: dict) -> float:
    keys = ["i2t_r1", "t2i_r1", "i2t_r5", "t2i_r5", "i2t_r10", "t2i_r10"]
    return float(sum(float(metrics[k]) for k in keys))


def get_main_pairs(repo_root: str | Path) -> List[Tuple[dict, dict]]:
    runs = scan_retrieval_runs(repo_root)

    baseline_candidates: Dict[int, List[dict]] = defaultdict(list)
    recon_candidates: Dict[int, List[dict]] = defaultdict(list)

    for run in runs:
        seed = run.get("seed")
        if seed is None:
            continue
        if run.get("variant") != MAIN_VARIANT:
            continue
        if run.get("batch_size") != MAIN_BATCH_SIZE or run.get("epochs") != MAIN_EPOCHS:
            continue

        effective_variant = run.get("effective_variant")
        recon_enabled = bool(run.get("recon_enabled"))
        lambda_recon = run.get("lambda_recon")
        mask_ratio = run.get("mask_ratio")

        if (not recon_enabled) and effective_variant == "baseline" and _close(lambda_recon, 0.0) and _close(mask_ratio, 0.15):
            baseline_candidates[int(seed)].append(run)
        elif recon_enabled and run.get("variant") == "A" and _close(lambda_recon, MAIN_LAMBDA) and _close(mask_ratio, 0.15):
            recon_candidates[int(seed)].append(run)

    seeds = sorted(set(baseline_candidates) & set(recon_candidates))
    pairs: List[Tuple[dict, dict]] = []
    for seed in seeds:
        pairs.append((_pick_preferred(baseline_candidates[seed]), _pick_preferred(recon_candidates[seed])))
    return pairs


def epoch_mean_curve(runs: List[dict]) -> Tuple[List[int], List[float]]:
    by_epoch: Dict[int, List[float]] = defaultdict(list)
    for run in runs:
        for item in run.get("history", []):
            epoch = int(item["epoch"])
            by_epoch[epoch].append(float(item["retrieval_score"]))
    epochs = sorted(by_epoch)
    means = [sum(by_epoch[e]) / len(by_epoch[e]) for e in epochs]
    return epochs, means


def scan_aro_results(repo_root: str | Path) -> Dict[str, Dict[int, dict]]:
    repo_root = Path(repo_root)
    results_dir = repo_root / "results"
    aro: Dict[str, Dict[int, List[dict]]] = {"baseline": defaultdict(list), "recon": defaultdict(list)}
    for path in results_dir.rglob("*.json"):
        name = path.name.lower()
        if not name.endswith("_aro.json"):
            continue
        data = load_json(path)
        if not isinstance(data, dict):
            continue
        if "aro_vg_attribution_accuracy" not in data:
            continue
        m = re.search(r"(?:^|_)s(\d+)(?:_|$)", path.stem)
        if not m:
            continue
        seed = int(m.group(1))
        bucket = "recon" if "recon" in name else "baseline"
        aro[bucket][seed].append({"path": str(path), **data})

    resolved: Dict[str, Dict[int, dict]] = {"baseline": {}, "recon": {}}
    for bucket, per_seed in aro.items():
        for seed, entries in per_seed.items():
            resolved[bucket][seed] = _pick_preferred(entries)
    return resolved


def scan_winoground_results(repo_root: str | Path) -> Dict[str, Dict[int, dict]]:
    repo_root = Path(repo_root)
    results_dir = repo_root / "results"
    wino: Dict[str, Dict[int, List[dict]]] = {"baseline": defaultdict(list), "recon": defaultdict(list)}
    for path in results_dir.rglob("*.json"):
        name = path.name.lower()
        if "winoground" not in name:
            continue
        data = load_json(path)
        if not isinstance(data, dict):
            continue
        if "winoground_group_score" not in data:
            continue
        m = re.search(r"(?:^|_)s(\d+)(?:_|$)", path.stem)
        if not m:
            continue
        seed = int(m.group(1))
        bucket = "recon" if "recon" in name else "baseline"
        wino[bucket][seed].append({"path": str(path), **data})

    resolved: Dict[str, Dict[int, dict]] = {"baseline": {}, "recon": {}}
    for bucket, per_seed in wino.items():
        for seed, entries in per_seed.items():
            resolved[bucket][seed] = _pick_preferred(entries)
    return resolved


def mean(values: List[float]) -> float:
    if not values:
        raise ValueError("Cannot compute mean of empty list.")
    return sum(values) / len(values)


def sem(values: List[float]) -> float:
    if len(values) <= 1:
        return 0.0
    mu = mean(values)
    var = sum((x - mu) ** 2 for x in values) / (len(values) - 1)
    return math.sqrt(var / len(values))


def main_output_dir(repo_root: str | Path) -> Path:
    return ensure_dir(Path(repo_root) / "results" / "figures" / "generated")
