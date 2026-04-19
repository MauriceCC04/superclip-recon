# INCIDENTS.md — Runtime Errors, Difficulties, and Decisions

This document is the **decision log** for SuperCLIP-Recon on Bocconi HPC. For each real issue encountered during the project, it records:

- **What happened** — the symptom as observed
- **What it meant** — the actual underlying cause
- **What was decided** — the fix or the workaround, and why
- **What we changed in the repo / runbook** — so the same issue does not bite twice

Entries are ordered roughly by when they were first seen during the project, and grouped by category. For operational recovery instructions, see the "When something goes wrong" section of `HPC_RUNBOOK.md`.

---

## A. Environment setup issues

### A1. Conda Terms of Service blocks environment creation

**Symptom.** First attempt to run `bash slurm/setup_env_ready.sh` errored out with:

```
CondaToSNonInteractiveError: Terms of Service have not been accepted for the following channels.
Please accept or remove them before proceeding:
    - https://repo.anaconda.com/pkgs/main
    - https://repo.anaconda.com/pkgs/r
```

Follow-up attempts then failed with `CondaError: Run 'conda init' before 'conda activate'` and `ModuleNotFoundError: No module named 'torch'`, because the env was never actually created.

**Cause.** Anaconda changed default-channel ToS enforcement; on a shared HPC where users can't click "Accept" in a GUI, the CLI must be used to accept ToS once per user.

**Decision.** Accept ToS on the login node before running setup, as a one-time step. This is cheap and reversible, and it is the documented escape hatch from Anaconda.

**Repo / runbook change.**
- Added a dedicated "first-time setup" subsection to `HPC_RUNBOOK.md` §0.3 with the two `conda tos accept` commands called out explicitly.
- Listed the same error and fix in the troubleshooting section (§8) so a future user hitting the error finds it without reading the setup section again.

---

### A2. `matplotlib is required` when running `analyze_results.py`

**Symptom.** After logging in fresh and running the analysis script:

```
matplotlib is required for analyze_results.py. Install it with:
pip install 'matplotlib>=3.8.0,<3.10.0' (original error: No module named 'matplotlib')
```

**Cause.** The user was not inside the `superclip` conda env. matplotlib is in `requirements.txt` and does exist inside the env; the login shell's system Python doesn't have it.

**Decision.** Do not "fix" this by installing matplotlib into the system Python. The env is the source of truth; the right reflex when seeing any "module not found" error is to activate the env first.

**Runbook change.**
- §4.3 now states explicitly: "always activate the env first" and gives the `module load miniconda3 / eval ... / conda activate superclip` incantation before calling `analyze_results.py`.
- §8 lists `matplotlib is required` and `ModuleNotFoundError` together as a single symptom with a single fix.

---

## B. SLURM submission issues

### B1. `common.sh` not found when submitting via `sbatch slurm/<script>.sh`

**Symptom.** First preflight submission died at job start with:

```
/var/spool/slurmd/jobNNNNNN/slurm_script: line 17:
/var/spool/slurmd/jobNNNNNN/common.sh: No such file or directory
```

The job ran for 1 second and failed.

**Cause.** Bocconi copies the submitted script into `/var/spool/slurmd/jobNNNN/` and executes the copy, not the original. `slurm/common.sh` was being sourced via:

```bash
source "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/common.sh"
```

`BASH_SOURCE[0]` under Bocconi's setup resolves to the spool copy, so the `dirname` resolves to `/var/spool/slurmd/jobNNNN/`, where `common.sh` does not exist.

**Decision considered.** Options were:
1. Rewrite `common.sh`-sourcing to use an absolute path.
2. Rewrite every SLURM script to start with `cd /mnt/beegfsstudents/home/3202029/superclip-recon`.
3. Submit every job as `sbatch --wrap='cd <REPO> && bash slurm/<script>.sh'` so the wrapped bash process sees the real repo path.

Option 3 chosen because it (a) does not touch the scripts themselves — keeping them reusable on other clusters, (b) is explicit at submission time, and (c) composes naturally with `--exclude`, `--time`, and other per-submission overrides.

**Repo / runbook change.**
- Every `sbatch` example in `HPC_RUNBOOK.md` uses the wrapped form.
- §"Why the commands look the way they do" calls this out as pattern #1 so future readers understand why the boilerplate is there.
- §8 lists the error text verbatim so a search-match lands on the fix.

---

### B2. `QOSMaxSubmitJobPerUserLimit` when batch-submitting main / ablation runs

**Symptom.** Running `bash slurm/submit_main_experiments.sh` submitted the baseline successfully but rejected Variant A and Variant B with:

```
sbatch: error: QOSMaxSubmitJobPerUserLimit
```

Later, `bash slurm/submit_ablations.sh` hit the same wall.

**Cause.** The `stud` QoS enforces a per-user cap on both the submit-queue (`QOSMaxSubmitJobPerUserLimit`) and the running set (`QOSMaxJobsPerUserLimit`). Mass-submitting 3+ jobs at once always loses.

**Decision.** Do not try to beat the scheduler. Adopt a "submit one, maybe two" policy and make it the default in the runbook. Accept that the ablation grid will finish over wall-clock hours instead of minutes.

**Runbook change.**
- §2.1 documents the policy explicitly.
- §4.2 warns the reader that `submit_ablations.sh` may hit the same rejection and that the right response is patience, not retry.
- Added an explicit note that `PD (QOSMaxJobsPerUserLimit)` is not an error — jobs in that state will start automatically.

---

## C. Storage and quota issues

### C1. Home quota filled to 50 GB / 50 GB, new jobs fail at eval-completion

**Symptom.** Job `479585` (λ=1.5 ablation) ran ~8 minutes, completed epoch-1 training, saved a checkpoint, ran retrieval eval to completion, and then failed with exit `120:0` and no Python traceback. Shortly after, a second submission (job `479586`) failed with exit `0:53` and `Elapsed=00:00:00` — i.e. it could not even start its prolog. `lquota` showed 50.5 GB / 50 GB used.

**Cause.** Disk quota exhaustion, specifically triggered during the post-eval write of epoch-2 checkpoint + results JSON + checkpoint-rotation logic. The first job died at the write attempt; the second job couldn't even create its `.out`/`.err` files.

**Decision considered.** Options were:
1. Reduce checkpoint size by turning off optimizer-state saving (already off by default).
2. Request a quota extension from HPC support.
3. Aggressive cleanup of `~/.cache` (Huggingface / torch hub) and pruning failed-run directories.
4. Switch to `last` retention instead of `last_and_best` to halve per-run footprint.

Chose option 3 first (immediate, reversible) and option 4 as the fallback. Cleanup brought usage down to ~44 GB, which was enough to unblock the next submission. Retention policy was kept at `last_and_best` because the best-checkpoint is important for compositional eval — losing it would cost more than the extra ~700 MB.

**Repo / runbook change.**
- `HPC_RUNBOOK.md` §4.1 now mandates a `lquota` check with a **< 40 GB** target before starting ablations.
- §8 lists `Disk quota exceeded` first in the troubleshooting order, because it is the most common upstream cause of weirder-looking downstream failures.
- §7 documents `last_and_best + keep_last_k=1 + no optimizer state` as the safety defaults, with a rationale for each setting tied to the 50 GB quota.

---

### C2. Stale early-epoch checkpoints surviving retention policy

**Symptom.** In `checkpoints/baseline/` and `checkpoints/baseline_s43_b128/`, three `.pt` files remained: `epoch_2.pt`, `epoch_9.pt`, `epoch_10.pt`. With `--save_strategy last_and_best --keep_last_k 1`, only two should exist (current + best).

**Cause.** The retention logic preserves the most-recent K checkpoints regardless of which epoch was "best", so an early best-checkpoint plus the current checkpoint plus `keep_last_k` occasionally leaves a spurious third file. Minor — it does not grow unbounded — but it costs ~700 MB per affected run, which matters at the quota margin.

**Decision.** Tolerate the artifact (it does not compound across runs) and document the manual cleanup step. Not worth rewriting the retention logic late in the project when the cost is bounded.

**Runbook change.** §4.1 now explicitly calls out the stray `epoch_2.pt` files and includes a line to delete them by hand as part of pre-ablation cleanup.

---

## D. Node-specific hardware issues

### D1. Recurring exit-`120:0` failures at eval start on `gnode04`

**Symptom.** Two consecutive submissions of the same configuration (λ=1.5, seed 42, batch 128) — job IDs `479585` and `479593` — both failed with exit `120:0` after ~8 minutes of runtime. Each got through epoch-1 training, checkpoint save, and into retrieval eval before dying with no Python traceback in the err file. The same configuration at a different seed (job `479572`, seed 43) completed in 76 minutes with no issues.

**Initial (wrong) hypothesis.** Seed 42 is deterministically bad. Ruled out once we compared node assignments.

**Actual cause.** Node assignment, not seed. Both failing jobs landed on **gnode04**; the successful run landed on **gnode02**. Exit code `120:0` with no Python-level traceback indicates external termination (kernel OOM-killer, MIG-slice hardware fault, cgroup kill — all possible, all invisible to userland). `sinfo` shows gnode04 as `mixed` (not drained), so the scheduler considers it healthy, but in practice it was killing our jobs at eval-start.

**Decision considered.** Options were:
1. Debug the exact termination cause (probably host-RAM OOM during eval dataloader spin-up).
2. Exclude `gnode04` from future submissions via `--exclude=gnode04`.
3. Abandon the three affected runs as out-of-scope for the project.

Chose a combination of 2 and 3. The three affected runs (λ=1.5 seed 42, λ=2.0, λ=2.5) were all **extensions beyond the proposed sweep grid** {0, 0.1, 0.5, 1.0}, so losing them does not threaten the defensible result set. `--exclude=gnode04` is free insurance for any future submission, and a single λ=2.0 seed-43 resubmission gives us a 6-point λ curve for the writeup.

We also reported the behavior to `hpc@unibocconi.it` with the failing and passing job IDs as evidence. That costs nothing and helps them triage the node.

**Runbook change.**
- Every `sbatch` example in §2 and §3 now includes `--exclude=gnode04`.
- §8 documents "exit `120:0` with no traceback" as a node-health symptom and tells the reader to check `NodeList` in `sacct` first.
- §"When something goes wrong" lists `--exclude=gnode04` as the default and notes that the exclusion can be removed once HPC support confirms the node is fixed.

**Lesson.** When a job fails deterministically for config A and passes for config B, check node assignment **before** forming a hypothesis about the config difference. Node is a hidden confound.

---

## E. Code / pipeline issues

### E1. Retrieval pipeline nested inside caption-collection loop

**Symptom.** A prior version of `evaluate.py` would, on the first image of retrieval eval, hit an assertion about `n_captions_per_image` uniformity — or, in other versions, loop forever on the first image without making progress.

**Cause.** The entire retrieval pipeline (building image embeddings, text embeddings, computing similarities) had been indented one level too far, so it was inside the per-annotation caption-collection loop. The pipeline was running once per annotation (~600k times) instead of once total. The first iteration couldn't validate uniformity because the `img_id_to_caps` dict only had one entry at that point.

**Decision.** Fix the indentation and pull the pipeline out of the loop. Add a regression test so this cannot come back silently.

**Repo change.**
- `evaluate.py` was restructured so the pipeline runs once, after `img_id_to_caps` is fully populated.
- `tests/run_tests.py` test 18 (`run_retrieval_eval completes end-to-end on synthetic val`) is an end-to-end guard: it builds the real model on synthetic data and runs retrieval, so a re-indentation bug would fail locally before it touched the cluster.

**Lesson.** End-to-end tests on tiny synthetic data catch whole-pipeline indentation bugs that unit tests miss.

---

### E2. `num_token_classes` must be synced to `vocab.json` before the model is built

**Symptom.** (Historical, caught early.) If `cfg.model.num_token_classes` stayed at its dataclass default (1000) while `vocab.json` actually had, say, 999 entries, the token-classification head's output dim would not match the labels, leading to a shape mismatch that only shows up at loss computation.

**Cause.** The model is built from the config, but the effective vocab size comes from the JSON file. These must be synced explicitly.

**Decision.** Load the vocab first, overwrite `cfg.model.num_token_classes = len(vocab_map)`, then build the model. Do this in `train.py`, `sanity_check.py`, and `tests/smoke_test.py` identically.

**Repo change.**
- All training and evaluation entry points now sync the config before constructing `SuperCLIPRecon`.
- Sanity-check logs explicitly print `Synced num_token_classes = N` so a collaborator can see it is actually happening.

**Lesson.** Any config value that is conceptually derived from an external file should be derived, not defaulted. Defaults are for truly independent hyperparameters.

---

### E3. Variant B `phrases.json` dependency was removed but the flag stayed

**Symptom.** Early runbook drafts told users to extract `phrases.json` before Variant B. Users then occasionally panicked when the file was missing or out of date.

**Cause.** Variant B was reworked to extract phrases inline from each batch's captions (see `create_phrase_mask_from_captions` in `losses.py`). The external `phrases.json` is no longer read during training. The `--phrase_path` CLI flag still exists in `train.py` as a deprecated no-op for backward compatibility.

**Decision.** Treat `phrases.json` as optional and never as a blocker. Keep the generator (`extract_phrases.py`) for inspection and debugging, but do not run it in the default setup.

**Repo / runbook change.**
- `slurm/setup_env_ready.sh` does not build `phrases.json` in its default "all" path.
- `HPC_RUNBOOK.md` says `phrases.json` is optional and describes how to build it only if the user wants it for inspection.
- `train.py` prints a deprecation notice when `--phrase_path` is passed for Variant B.

**Lesson.** When a feature's requirement is relaxed, update the runbook at the same commit. The runbook is the user's mental model.

---

## F. Meta-lessons

A few things are not specific to any single incident but were reinforced across several.

### F1. Read `err/`, not `out/`, during training

`tqdm` and most Python warnings go to stderr. `out/` stays near-empty for hours into a multi-epoch run; `err/` is where progress actually lives. Every monitoring example in the runbook now puts `tail -f err/...` first.

### F2. Read logs end-first and with `od -c` when tqdm has rewritten them

Because `tqdm` uses carriage returns to update the progress bar in-place, a plain `tail` of an err file can look like it ends at "Epoch 1/10: 48%". The useful bytes — the traceback, the OOM message, the terminated-by-signal marker — are often past that. Use `tail -c 2000 err/... | od -c | tail -40` to get the raw trailing bytes.

### F3. `sacct` is the right post-mortem tool

`scontrol show job <JOBID>` only works while SLURM still has the job in memory (usually minutes after completion). `sacct -j <JOBID> -o JobID,State,ExitCode,NodeList` works after the fact and includes the node assignment, which has been critical for diagnosing `gnode04` failures.

### F4. Check node assignment before blaming the config

Two otherwise-identical runs with different outcomes → check `NodeList` before forming a hypothesis about seeds, hyperparameters, or code changes. On a shared cluster, the node is a silent confound that looks like randomness until you expose it.

### F5. The proposal defines the finish line

At several points during the project, tempting-looking extensions (λ=2.0, λ=2.5, extra seeds) competed for quota and time with the documented proposal grid {0, 0.1, 0.5, 1.0}. The right call was always to finish the proposal grid first and treat extensions as optional. `--exclude=gnode04` + one λ=2.0 run got us a 6-point curve for the writeup without derailing the in-scope deliverables. Extensions that don't complete are not worth the debugging time at the margin.

---

## G. Open items

Things that remain unresolved at the time of writing, for the next person to pick up:

- **gnode04 health.** Reported to HPC support. `--exclude=gnode04` should be removed from submissions once HPC confirms the node is fixed. Until then, the workaround is the default.
- **`submit_ablations.sh` behavior under QoS limits.** The script does not currently back off when submissions are rejected; it just fails for all ablations past the first two. Not worth fixing now — the documented workaround (submit one at a time with wrapped `sbatch`) is reliable — but a future maintainer could add retry-with-backoff.
- **Checkpoint retention edge case (C2).** `keep_last_k` + `best` interaction occasionally leaves one stray intermediate checkpoint. Bounded cost, works around manually. A small `tests/run_tests.py` addition could pin this down if the project is extended.
