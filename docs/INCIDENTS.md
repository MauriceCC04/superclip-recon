# INCIDENTS.md — Runtime Errors, Difficulties, and Decisions

This document is the decision log for SuperCLIP-Recon on Bocconi HPC. For each real issue encountered during the project, it records:

- **What happened** — the symptom as observed
- **What it meant** — the actual underlying cause
- **What was decided** — the fix or workaround, and why
- **What we changed in the repo / runbook** — so the same issue does not bite twice

Entries are ordered roughly by when they first appeared during the project, and grouped by category. For operational recovery instructions, see `HPC_RUNBOOK.md`.

---

## A. Environment setup issues

### A1. Conda Terms of Service blocks environment creation

**Symptom.** First attempt to run `bash slurm/setup_env_ready.sh` errored out with `CondaToSNonInteractiveError`, then follow-up attempts failed with `CondaError: Run 'conda init' before 'conda activate'` and missing Python packages.

**Cause.** Default-channel ToS must be accepted once per user on the CLI.

**Decision.** Accept the terms on the login node before any setup step.

**Runbook change.** Added a first-time setup section with the exact `conda tos accept` commands and mirrored the same fix in troubleshooting.

---

### A2. `matplotlib is required` or `ModuleNotFoundError` on a fresh login

**Symptom.** Scripts that worked earlier suddenly failed with missing packages.

**Cause.** The user was in the system Python, not the `superclip` conda env.

**Decision.** Never patch the system Python. The env is the source of truth.

**Runbook change.** The runbook now states that every analysis or evaluation command should begin with env activation if the shell session is fresh.

---

## B. SLURM submission and shell issues

### B1. `common.sh` not found when submitting `sbatch slurm/<script>.sh`

**Symptom.** Jobs died immediately at start with a path under `/var/spool/slurmd/jobNNNN/` and `common.sh` missing.

**Cause.** Bocconi runs a spool copy of the submitted script, so `BASH_SOURCE[0]` resolves to the spool path, not the repo path.

**Decision.** Use wrapped `sbatch` or a script that explicitly `cd`s into the repo and sources the real `slurm/common.sh`.

**Runbook change.** All examples now use wrapped submission or scripts that enter the repo explicitly.

---

### B2. `QOSMaxSubmitJobPerUserLimit` and partial batch submission

**Symptom.** A batch submit would accept the first one or two jobs and reject the rest with `QOSMaxSubmitJobPerUserLimit`.

**Cause.** The `stud` QoS enforces per-user submission and running-job caps.

**Decision.** Stop trying to mass-submit many independent jobs. Prefer one sequential batch job that runs multiple experiments in order.

**Runbook change.** The runbook now prefers sequential jobs for 4+ related runs and treats mass submission as the exception, not the default.

---

### B3. Jobs appear to fail because `tail` and `find` cannot locate files

**Symptom.** Commands like `tail err/...`, `find results ...`, or `ls out` said files did not exist.

**Cause.** The user was in `~` instead of `/mnt/beegfsstudents/home/<USER_ID>/superclip-recon`.

**Decision.** Treat `cd` into the repo as the first debugging reflex.

**Runbook change.** Added an explicit troubleshooting entry for repo-relative path errors and reinforced the “always cd into the repo” rule.

---

### B4. Pasted here-doc scripts became corrupted and submitted broken files

**Symptom.** A script created with `cat > file <<'EOF'` contained fragments of unrelated shell input, submitted successfully, then failed in 1–2 seconds with tiny `.out` files and meaningless command errors.

**Cause.** The shell kept reading input into the here-doc while additional commands were typed before the terminating `EOF` was properly completed and the prompt returned.

**Decision.** After writing a script with a here-doc:

1. wait for the prompt to return
2. inspect the file with `sed -n` before submitting
3. only then run `sbatch`

**Runbook change.** Added a troubleshooting entry for corrupted pasted scripts and a recommendation to inspect generated scripts before submission.

---

## C. Storage and quota issues

### C1. Home quota fills to 50 GB and jobs fail late or fail to start

**Symptom.** Jobs failed near checkpoint save or result write, sometimes with no Python traceback, and later jobs could not even start. `lquota` showed usage at or above 50 GB.

**Cause.** Disk quota exhaustion during checkpoint save, results write, or even job prolog file creation.

**Decision.** Keep result JSONs, sync needed checkpoints to the laptop, then delete checkpoint directories and caches aggressively.

**Runbook change.** The runbook now recommends staying below about 40 GB before long sweeps and explicitly prioritizes deleting checkpoints over result JSONs.

---

### C2. Variant B sweep failed partway because storage ran out

**Symptom.** The Variant A sweep completed, but the Variant B sweep failed after several lambdas. The sweep `.err` was empty, the job had run for hours, and `lquota` showed the home quota maxed out.

**Cause.** Storage exhaustion during the later part of the sweep. The early Variant B runs completed and wrote valid result files; later ones could not finish their write path.

**Decision.** Keep the successful Variant B JSONs, sync them to the laptop, delete checkpoint directories, and rerun only the missing Variant B lambdas.

**Runbook change.** Added a dedicated partial-sweep recovery policy: keep successful JSONs, free storage, rerun only missing lambdas, and do not rerun the whole sweep by default.

---

### C3. One-off reruns wrote to top-level `results/` and `checkpoints/`

**Symptom.** Missing-lambda Variant B reruns completed, but the expected files were not found under `results/ablations/`. Instead they appeared in top-level `results/` and `checkpoints/`.

**Cause.** The one-off rerun command set `RUN_NAME` but did not set `SAVE_DIR` and `RESULTS_FILE`, so `slurm/run_one_experiment.sh` fell back to its top-level defaults.

**Decision.** For any manual rerun outside a sweep, explicitly set `SAVE_DIR` and `RESULTS_FILE`.

**Runbook change.** Added a specific warning and example command for missing-lambda reruns with explicit output paths.

---

### C4. Checkpoints, not JSONs, were the real storage problem

**Symptom.** Result JSONs took little space, but checkpoint trees consumed most of the quota.

**Cause.** Each checkpoint was about 677 MB, so keeping many runs and many epochs quickly filled the 50 GB home quota.

**Decision.** Treat result JSONs as the must-keep artifacts. Checkpoints are optional after sync unless needed for more evaluation.

**Runbook change.** Storage management now explicitly prioritizes syncing and keeping JSONs, and only optionally keeping checkpoints.

---

## D. Node-specific hardware issues

### D1. Recurring exit `120:0` failures on `gnode04`

**Symptom.** Some runs failed at eval start with exit `120:0` and no Python traceback, while equivalent runs on a different node succeeded.

**Cause.** Node assignment, not seed or hyperparameter choice. `gnode04` was an unstable confound.

**Decision.** Treat `--exclude=gnode04` as default.

**Runbook change.** Every long-job example includes `--exclude=gnode04`, and troubleshooting now tells the user to inspect `NodeList` first for silent external kills.

---

## E. Code and pipeline issues

### E1. Retrieval pipeline was nested inside the caption-collection loop

**Symptom.** Retrieval evaluation asserted incorrectly or appeared to loop forever on the first image.

**Cause.** The retrieval pipeline in `evaluate.py` had been indented inside the per-annotation caption-collection loop.

**Decision.** Pull the full retrieval pipeline out of the loop and protect it with an end-to-end synthetic test.

**Repo change.** Retrieval now runs once, after caption collection is complete.

---

### E2. `num_token_classes` must be synced to `vocab.json`

**Symptom.** If the config default and `vocab.json` size diverged, the token classification head dim and label dim no longer matched.

**Cause.** Effective token-class count comes from the vocab file, not the dataclass default.

**Decision.** Sync the config to the loaded vocab before model construction everywhere.

**Repo change.** The sync now happens consistently in training and validation entry points.

---

### E3. Variant B `phrases.json` is optional, not a blocker

**Symptom.** Users worried that missing `phrases.json` meant Variant B could not train.

**Cause.** Variant B had already been reworked to extract phrases inline; the old external file was no longer required during training.

**Decision.** Treat `phrases.json` as optional inspection/debugging output only.

**Runbook change.** The runbook now states clearly that missing `phrases.json` is not a blocker.

---

### E4. GPU smoke failed with `KeyError: 'train_mode'` from `sanity_check.py`

**Symptom.** `superclip-gpu-smoke` failed in the loss computation phase with:

```python
KeyError: 'train_mode'
```

**Cause.** `sanity_check.py` called `total_loss(...)` with a mixed positional-plus-keyword pattern after `losses.py` had moved to a keyword-oriented API that expected `train_mode` in `kwargs`.

**Decision.** Patch `sanity_check.py` to call `total_loss(...)` with the full keyword API, matching the working path used elsewhere.

**Repo / workflow change.** GPU smoke is now again a valid gate before pilots and sweeps. The runbook troubleshooting section now names this exact failure and fix.

---

### E5. `slurm/run_one_experiment.sh` hard-requires `MASK_RATIO`

**Symptom.** Pilot jobs died instantly with:

```bash
MASK_RATIO must be set
```

**Cause.** The launcher explicitly validates `RUN_NAME`, `VARIANT`, `LAMBDA_RECON`, and `MASK_RATIO`.

**Decision.** Every pilot, one-off rerun, and custom sequential batch must set `MASK_RATIO` explicitly.

**Runbook change.** The runbook now lists `MASK_RATIO` as a required launcher variable and includes an example pilot command that exports it.

---

### E6. Compositional evaluation failed on `--benchmarks aro`

**Symptom.** The first compositional pair job died immediately with:

```text
eval_compositional.py: error: unrecognized arguments: --benchmarks aro
```

**Cause.** The CLI flag is `--benchmark`, singular.

**Decision.** Standardize all compositional commands on `--benchmark aro`.

**Runbook change.** Added an explicit troubleshooting entry and updated the compositional-eval section to use the singular flag.

---

### E7. The stock compositional script was too narrow for named checkpoint families

**Symptom.** `slurm/run_compositional_eval.sh` was written for `checkpoints/baseline`, `checkpoints/variant_a`, and `checkpoints/variant_b`, but real project checkpoints lived in more specific directories like:

- `checkpoints/ablations/...`
- `checkpoints/confirm6/...`
- `checkpoints/compositional_round2/...`

**Cause.** The stock script assumed a simplified checkpoint layout that no longer matched the actual experiment families used in the project.

**Decision.** Keep the stock script for simple cases, but use custom sequential compositional scripts when evaluating specific named checkpoint families.

**Runbook change.** The compositional section now describes the stock script as limited and recommends custom direct-path scripts for ablation and confirmation checkpoints.

---

### E8. Missing old ablation checkpoints became a practical blocker for later compositional follow-up

**Symptom.** A later compositional round intended to evaluate the old `s43` ablation pair, but those checkpoint folders had already been deleted from HPC after being synced to the laptop.

**Cause.** Storage cleanup had removed the old checkpoint directories to free quota, which was correct operationally, but it meant later follow-up evaluation could not use them unless they were copied back from the laptop.

**Decision.** If old checkpoint families are needed for later evaluation, either restore them from the laptop first or simplify the follow-up plan to use checkpoint families that still exist on HPC.

**Runbook change.** The compositional workflow now begins with a checkpoint-existence check and treats restore-from-laptop as a normal, not exceptional, operation.

---

## F. Monitoring lessons

### F1. `err/` is usually right for training progress, but not always for wrapper scripts

`err/` remains the right first monitor for pure training runs because `tqdm` and warnings often write there.

But sweep wrappers and sequential multi-run jobs often put their high-level progress into `out/` or per-run logs in `logs/`.

**Runbook change.** The monitoring section now distinguishes training runs from sweep/sequential wrappers instead of saying “always read err first” as an absolute rule.

---

### F2. `sacct` is the post-mortem source of truth

This was reinforced multiple times. It gives:

- state
- exit code
- elapsed time
- node assignment

and survives after the job has left active memory, unlike `scontrol show job`.

---

### F3. A tiny `.out` plus a 1–2 second failure usually means wrapper/script failure, not model failure

This pattern showed up in corrupted pasted scripts and invalid flag usage. It is a strong clue that the job died before real training or evaluation started.

---

### F4. `SIGNAL Terminated` can kill the tail of a sequential batch after earlier steps already succeeded

**Symptom.** Longer sequential jobs such as the mask-rate batch and the later compositional-round batch made real progress, then ended with a SLURM trailer like:

```text
*** JOB <JOBID> ON <NODE> CANCELLED ... DUE to SIGNAL Terminated ***
```

**Cause.** External SLURM termination or cancellation after part of the wrapper had already finished. From the stderr logs alone, the exact upstream reason is not always recoverable, but the important point is that the batch may already contain valid completed subruns before the kill.

**Decision.** Treat partial sequential batches as salvageable by default. Inspect which subrun or evaluation finished last, keep the artifacts already written, and rerun only the unfinished tail instead of discarding the whole batch.

**Runbook change.** Added a partial-batch recovery rule and a troubleshooting entry for `SIGNAL Terminated` in sequential jobs.

---

### F5. `fatal: not a git repository` at the end of a successful run is alarming but usually benign on the HPC copy

**Symptom.** Many completed training logs ended with:

```text
fatal: not a git repository (or any parent up to mount point /mnt)
Stopping at filesystem boundary (GIT_DISCOVERY_ACROSS_FILESYSTEM not set).
```

**Cause.** `train.py` records `git_commit` via a best-effort `git rev-parse HEAD`, but the normal HPC `rsync` excludes `.git`, so the HPC copy is often not a real Git checkout.

**Decision.** Do not treat this line as a training failure if the job otherwise completed and wrote results. Treat it as missing provenance metadata. If exact revision tracking matters, record the commit SHA on the Mac before syncing or store it explicitly in a lightweight tracked note.

**Runbook change.** The runbook now explains why this message appears, why it is usually benign, and how to preserve commit provenance when needed.

---

## G. Project-level decisions reinforced during execution

### G1. The proposal still defines the finish line

Throughout the project, quota and cluster behavior repeatedly tried to pull attention toward optional extensions and operational recovery work.

The right prioritization remained:

1. baseline reproduction
2. Variant A reconstruction check
3. controlled confirmation runs
4. one compositional probe
5. only then optional extras

This was the correct way to stay aligned with the project proposal under limited compute and storage.

---

### G2. Use matched pairs whenever possible

The strongest comparisons were always same-seed baseline vs recon pairs:

- same seed
- same epochs
- same batch size
- same evaluation pipeline
- only `lambda_recon` / `train_mode` changed

This became the preferred comparison template for confirmation runs and compositional follow-up.

---

## H. Open items

- **A formal “best checkpoint” marker.** The project still relies on inferring or choosing the intended checkpoint from the saved files and logs. A future maintainer could make this explicit in metadata.
- **A more general compositional-eval launcher.** The stock compositional SLURM script still assumes simplified checkpoint families. A generalized launcher that accepts arbitrary checkpoint dirs would remove the need for ad hoc custom scripts.
- **Automated retry/backoff for QoS rejections.** The documented workaround is reliable, but the helper scripts still do not back off automatically when the QoS rejects additional submissions.
- **A lightweight way to preserve Git provenance on HPC copies.** Because the normal sync excludes `.git`, a small tracked file or env-based SHA handoff would make later auditing cleaner without syncing the full Git history.
