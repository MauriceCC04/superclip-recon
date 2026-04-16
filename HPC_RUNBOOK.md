# HPC Runbook — SuperCLIP-Recon

**Target cluster**: partition `stud`, account `3202029`, QoS `stud`, one MIG-sliced A100 40GB (`gpu:4g.40gb:1`), 100 GB home quota, no scratch.

This runbook defines a **gated** execution order. Each gate emits a structured JSON artefact and a PASS / WARN / FAIL verdict. **Do not proceed to the next gate until the current gate is PASS or a justified WARN.**

---

## Gate 0 — Local synthetic tests (no COCO, no GPU)

**Purpose**: catch shape/logic/pipeline bugs before touching the cluster.

```bash
python tests/run_tests.py
```

| Verdict | Criterion | Action |
|---|---|---|
| **PASS** | All tests pass. | Proceed to Gate 1. |
| **WARN** | Only environment-dependent tests fail (no internet for CLIP weights, no `datasets` package). Core logic, masking, retention, schema, slurm tests pass. | Proceed to Gate 1, but note that model-building tests were not exercised locally; they will be exercised in Gate 1. |
| **FAIL** | Any test in the fixes suite fails (numbers 06b, 06c, 12, 13, 14, 15, 16, 17). | Stop. Fix the underlying code before queueing anything. |

---

## Gate 1 — Static + runtime preflight (cluster)

**Purpose**: verify the env is set up, data is present, imports work, and one tiny forward+backward step runs under real cluster conditions.

```bash
bash slurm/setup_env_ready.sh       # one-time: env + cache + data + vocab + phrases
sbatch slurm/run_preflight.sh
# when complete:
cat results/preflight/preflight_report.json
```

**Inspect**: `results/preflight/preflight_report.json`

```jsonc
{
  "overall_status": "PASS|WARN|FAIL",
  "checks":   { "repo", "imports", "gpu", "data", "cache", "storage", "runtime" },
  "metrics":  { "gpu_peak_mem_gb", "checkpoint_size_mb",
                "one_step_seconds", "tiny_eval_seconds" },
  "recommendations": [ ... ]
}
```

| Verdict | Criterion | Action |
|---|---|---|
| **PASS** | `overall_status == "PASS"`. GPU detected, data present, one_step_seconds < 5 s. | Proceed to Gate 2. |
| **WARN** | `overall_status == "WARN"` and the warnings are about **optional** pieces (`phrases.json` missing → only affects Variant B; cache vars need tuning). | Fix the warnings or accept and proceed. Do **not** proceed if the warning is about home-quota usage. |
| **FAIL** | Any of: `repo.status=="FAIL"` (missing code files), `imports.status=="FAIL"` (broken env), `gpu.status=="FAIL"`, `runtime.status=="FAIL"`, or `storage.status=="FAIL"` (home >85% full). | Stop. Follow the `recommendations` list in the report. |

---

## Gate 2 — Cluster GPU smoke

**Purpose**: run a few real training steps + a small retrieval eval on the real GPU, measure peak memory and checkpoint size.

```bash
sbatch slurm/run_gpu_smoke.sh
cat results/smoke/gpu_smoke_results.json
```

**Inspect**: `results/smoke/gpu_smoke_results.json` — look for:
- `gpu_peak_mem_gb` — should be well under the 40 GB slice limit (expect ~5–8 GB for batch=32).
- `checkpoint_size_mb` — typically ~700–1300 MB for ViT-B/32 + recon head.
- `retrieval.i2t_r1` and `retrieval.t2i_r1` — just need to be > 0 (this is 5-step training, not a real metric).

| Verdict | Criterion | Action |
|---|---|---|
| **PASS** | JSON produced, `gpu_peak_mem_gb < 35`, no NaN in losses, ckpt saved. | Proceed to Gate 3. |
| **WARN** | `gpu_peak_mem_gb` between 32 and 38 GB, i.e. headroom is tight. | Reduce batch_size for Gate 3 and main runs, or disable `save_optimizer_state` (already off by default). |
| **FAIL** | OOM, NaN, import error on compute node, or job killed. | Stop. Read the SLURM error log in `err/`; likely need to fix env or reduce batch size. |

---

## Gate 3 — Pilot baseline

**Purpose**: run **one** real training epoch of the baseline, measure wall time, and project the 10-epoch cost before committing to the main runs.

```bash
sbatch slurm/run_pilot_baseline.sh
cat results/pilot/pilot_baseline.json
```

**Inspect**: `results/pilot/pilot_baseline.json` — the script computes:
- `epoch_seconds` — one epoch wall time
- `projected_10_epoch_seconds` — epoch × 10
- `projected_checkpoint_storage_mb` — checkpoint size × retention
- `readiness`: `PASS|WARN|FAIL` with reasons

| Verdict | Criterion | Action |
|---|---|---|
| **PASS** | `projected_10_epoch_seconds < 20 * 3600` (20 h), `projected_checkpoint_storage_mb < 10 * 1024`. | Proceed to Gate 4. |
| **WARN** | Projected run > 20 h but < 24 h. | Reduce epochs to 7–8, or raise batch_size if Gate-2 peak memory allows. |
| **FAIL** | Projected run > 24 h, or projected storage would break the home quota. | Stop. Shrink: smaller batch, fewer epochs, smaller eval subset, stronger retention policy. |

---

## Gate 4 — Main experiments (three independent jobs)

**Purpose**: run baseline, Variant A, Variant B — each as its own SLURM job. Failure of one does not kill the others; each can be resubmitted independently.

```bash
bash slurm/submit_main_experiments.sh               # submit all three
bash slurm/submit_main_experiments.sh variant_a     # resubmit just one

squeue -u $USER                                     # check status
```

Per-job output: `results/<run_name>.json` and `checkpoints/<run_name>/epoch_*.pt`.

| Verdict | Criterion | Action |
|---|---|---|
| **PASS** | All three JSON files present, all have `final_retrieval.i2t_r1 > 0`, all losses finite. | Proceed to Gate 5. |
| **WARN** | One run finished; others still queued or still running. | Wait; the submit script is idempotent — re-running skips existing outputs. |
| **FAIL** | Any run produced NaN, OOM, or did not produce a JSON at all. | Inspect `err/superclip-<run>_<jobid>.err`, fix, resubmit only that one. |

---

## Gate 5 — Ablations

**Purpose**: compact grid (lambda sweep, masking-rate sweep, Variant A vs B). Do **not** start until Gate 4 is green and storage headroom is verified.

```bash
# Verify storage first:
du -sh checkpoints results
df -h $HOME

bash slurm/submit_ablations.sh
```

The submit script skips configs that already exist from Gate 4 (e.g. `lambda_0.0` ≡ `baseline`, `lambda_0.5` ≡ `variant_a`), so only the truly new points get queued.

| Verdict | Criterion | Action |
|---|---|---|
| **PASS** | All unique ablation JSONs present under `results/ablations/`, then run `python analyze_results.py` and confirm the plots render. | Done — write up. |
| **WARN** | Some ablation runs failed but the *informative* points (at least two lambda values and two mask rates) are present. | Accept; document which runs failed and why. |
| **FAIL** | Home quota exceeded mid-grid, or `analyze_results.py` complains of missing files. | Stop the remaining jobs (`scancel`), clean up, revisit retention policy. |

---

## Quick reference — commands

```bash
# Local (Gate 0)
python tests/run_tests.py

# One-time setup (modular; any step can be re-run)
bash slurm/setup_env_ready.sh
bash slurm/setup_env_ready.sh --step vocab
bash slurm/setup_env_ready.sh --step phrases --spacy   # optional spaCy

# Cluster gates (submit in order)
sbatch slurm/run_preflight.sh          # Gate 1
sbatch slurm/run_gpu_smoke.sh          # Gate 2
sbatch slurm/run_pilot_baseline.sh     # Gate 3
bash   slurm/submit_main_experiments.sh  # Gate 4
bash   slurm/submit_ablations.sh         # Gate 5

# Optional: compositional evals after Gate 4
sbatch slurm/run_compositional_eval.sh

# Analysis (anytime after Gate 4 or 5)
python analyze_results.py --results_dir ./results --output_dir ./results/figures
```

---

## Safety defaults (all already enabled)

| Setting | Default | Why |
|---|---|---|
| `--save_strategy` | `last_and_best` | Keep only the most useful checkpoints. |
| `--keep_last_k` | `2` | Safety net against `last_and_best` deleting a needed epoch. |
| `--save_optimizer_state` | `False` | Halves checkpoint size. Training restart from mid-run is rare. |
| `--eval_max_images` | `5000` for main; `1000` for pilot; `128` for smoke | Keep eval cheap where it's only a sanity signal. |
| `PIP_NO_CACHE_DIR` | `1` (via `common.sh`) | Prevent pip cache from growing on home. |
| `XDG_CACHE_HOME`, `HF_HOME`, `TORCH_HOME`, etc. | Under `$HOME/.cache` (via `common.sh`) | Centralize cache growth under one visible path. |
| Slurm scripts | `cd "$PROJECT_ROOT"` from script location | Independent of `sbatch`'s working directory. |

---

## When something goes wrong

**Job killed at OOM**: lower `--batch_size`, disable `--save_optimizer_state`, reduce `--eval_max_images`.

**Job killed at time limit**: check Gate 3 projections; reduce `--epochs` or raise `--batch_size` (if Gate 2 shows room).

**"COCO data not found"**: run `bash slurm/setup_env_ready.sh --step data`.

**"vocab.json not found"**: run `bash slurm/setup_env_ready.sh --step vocab`.

**CLIP weights won't download on compute node**: compute nodes often can't reach the internet. Pre-cache on the login node via `bash slurm/setup_env_ready.sh --step cache`, which hits `slurm/cache_clip.py`. Weights then live under `$HOME/.cache/huggingface` and `$HOME/.cache/torch` where `common.sh` points the env vars.

**Home quota alarm**: run `du -sh $HOME/.cache/*` and `du -sh checkpoints results` to find the growth, then drop `--keep_last_k` to 1 and delete old `out/` and `err/` logs.

**Ablation grid taking too long**: cut `EPOCHS=3 EVAL_MAX_IMAGES=1000 bash slurm/submit_ablations.sh` — shorter training is usually enough to compare settings relative to each other.
