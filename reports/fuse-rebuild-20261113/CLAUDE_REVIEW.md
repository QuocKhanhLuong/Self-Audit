# Candidate FUSE: independent Claude review (FINAL)

- Reviewer: the one independent Claude reviewer (task_26565d361dd9, dispatch ctx_b43c87cc97a7).
- Mode: read-only. This file is the only thing I wrote. Branch `fuse-rebuild`, shared uncommitted checkout. Nothing was committed.
- Scope: the full `docs/CANDIDATE_FUSE_SPEC.md` (316 lines, including the amendments for checkpoint every 20 updates, CUDA warn-only, and independent per-frame L/U masks), all of `self_audit/candidate_fuse/*`, `scripts/fuse_*`, `scripts/smoke_fuse.py`, and the FUSE registrations in `train.py` and `evaluate.py`.
- Evidence class: SOFTWARE_ONLY. I read the code, ran CPU synthetic repros, ran the focused tests, `check_repo` and the CPU smoke. No GPU, patient data, server or service was touched. GPU profile, real training and patient efficacy are all NOT_RUN.

## 1. Final verdict

**PASS (logic and formulas).** The FUSE math and control flow match the spec. All blockers and
High/Medium findings from the interim review are fixed and re-verified on the final files. Five
Low items remain (section 4). None of them blocks the GPU profile or a real T→J→VAL3→export→TEST3
launch.

Re-verified on the final tree (2026-11-13):
- `python -m pytest -q tests/test_fuse_*.py`: **66 passed**.
- `python scripts/check_repo.py`: **pass**.
- `python scripts/smoke_fuse.py --out <scratch>`: **SOFTWARE_ONLY PASS**. This covered T, J, VAL3 and deploy_export. Measured results:
  - `teacher_has_no_grads_after_J=true` and `export_clean=true`.
  - Fresh-process cold-load was bitwise equal on 3 odd-shape volumes for each of teacher, student and fusion.
- I read the coordinator's separate-process CPU campaign record (`/tmp/fuse-campaign-e2e-report.md`). It reports `T→J→VAL3→EXPORT→TEST_ONCE` COMPLETED, with the TEST transaction COMMITTED on attempt 1. That run is software evidence only.

What a real launch still has to prove on GPU, because CPU cannot show it:
- profiled peak VRAM and time;
- the epoch budget lock;
- bf16 finite losses;
- best-effort GPU determinism, with the warnings kept.

## 2. Resolution of earlier findings (with final evidence)

| ID | Finding (interim) | Final status / evidence |
|---|---|---|
| B1 BLOCKER | `fuse_build_manifest.py` called `Path.get` for VAL/TEST geometry, and one original root could not serve ACDC `testing/` | FIXED. Per-role roots (`fuse_build_manifest.py:73-76`); geometry taken from the parsed source manifests (`:110-115`); provenance kept outside the runtime keys (`:115`); split locked to 80/18/50 (`:66`). New `scripts/fuse_prepare_sources.py` builds the missing TEST cine cache and IDs 101–150. I checked its normalization against `self_audit/preparation.normalize_cine` on int16 data: **bitwise equal** (360,000/360,000 voxels). TEST masks are never read by the preparation step. |
| B2 BLOCKER | Strict `use_deterministic_algorithms(True)` on CUDA makes the trilinear and adaptive-pool backward raise | FIXED. `engine.py:470-490` sets `CUBLAS_WORKSPACE_CONFIG` before any CUDA query and uses `warn_only=True` on CUDA (strict on CPU). It records `determinism_status` and `determinism_note` in RESULT. The profile matches (`fuse_profile.py:78-80,119-120`). This follows spec lines 127–131. |
| B3 BLOCKER (ops) | A 1.4 GB `resume.pt` was written and re-hashed every update | FIXED. `_checkpoint_due` saves every 20 updates plus every epoch boundary, and every update only for synthetic runs (`engine.py:216-219,603-611,717-725`). The J payload now holds a SHA reference to the teacher instead of teacher weights and the teacher optimizer (`:719-723,736-740`). After T the teacher optimizer is released (`:673-678`). Remaining I/O: T is ~400 saves of ~1.4 GB; J is ~400 saves of ~160 MB. That is acceptable. |
| H1 | `evaluate.py` defaulted the FUSE role to `test` and ran TEST single-mode with no ledger | FIXED. `evaluation.py:209-215` refuses `role="test"` in `evaluate_export`. `evaluate.py:193-196` requires `--split dev`. TEST is reachable only through the locked three-mode `evaluate_exports` ledger (`evaluation.py:269-364`). |
| H2 | J via the campaign's `--fuse-resume` skipped the teacher meta/SHA/PROFILE checks | FIXED. `_teacher_reference` validates meta identity, SHA and status, and rejects PROFILE in production (`engine.py:409-422`). The T→J transition and J resume both go through `_load_teacher_reference`, which also checks that a supplied `--fuse-teacher-artifact` matches the pinned weights tensor by tensor (`:391-406,520-532`). |
| M1 | The T-b warmup/cosine schedule differed from Q3D | FIXED. `teacher.py:180-207` now calls Q3D's own `_schedule_lr` with `step=update-t_a`, `total=t_b`. T-a = round(0.05·8000) = 400. |
| M2 | A campaign T resume could not recover from a crash before the first checkpoint | FIXED. `fuse_campaign.py:88-120` archives the uncommitted attempt into `recovery_archive/` and only adds `--fuse-resume` when a committed T checkpoint verifies. |
| M3 | `profile_t` read the wrong meta filename | FIXED (code removed). |
| M4 | Stage-T resume at the final boundary raised `NameError` | FIXED. `last_validation` is stored in the boundary payload (`engine.py:586,629,640-641`). |
| M5 | Config locked only the epoch counts | FIXED. Every training scalar is locked to its spec value (`config.py:93-112`). |
| M6 | Builder dropped geometry provenance | FIXED (`fuse_build_manifest.py:115`). |
| M7 | VAL/TEST sizes were not enforced | FIXED (`data.py:136`: train 80, development 18, test 50). |
| Pending (coordinator list) | native L/U shape mismatch, leftover T grads, shared export schema, three-mode TEST transaction, phase probe, metric replay | All verified FIXED. L/U are transformed independently and need no matching shape (`engine.py:118-127`). Freeze clears grads (`teacher.py:121-128`, `engine.py:303-312`). A single export schema `self_audit.candidate_fuse.export.v1` is used in export, engine and campaign. The TEST transaction locks before loading labels, checks that student and fusion share the same student state, and blocks reruns on any existing ledger. The phase probe test passes. Checkpoints record a metrics offset and SHA; resume archives the uncommitted suffix (`engine.py:211-242,295,518`). |

## 3. Spec compliance, verified item by item

**Data (§2).**
- Real TRAIN requires exactly 80 patients, with L IDs = U IDs = `patients` = `split.train`. L and U must reference identical cine bytes (`data.py:94-115`).
- L is exactly the annotated ED/ES frame map; 2×80 gives 160 keys. U is every other frame, image-only. Loading an ED/ES frame as U raises, and so does loading a non-ED/ES frame as L (`data.py:187-230`).
- TRAIN, development and TEST must be pairwise disjoint and match the locked split (`data.py:129-141`).
- The engine only opens train and development (`engine.py:499`). TEST is opened only by `evaluate_exports`.
- Epochs and budgets:
  - T epoch: a permutation of all 160 L keys; 50 epochs = 8000 updates; T-a = 400.
  - J epoch: a permutation of the 80 patients with one random U frame each; L draws from a separate 160-key stream spanning two J epochs.
  - J total: 80·100 = 8000 updates.
- All RNG states and cursors are checkpointed. The T→J lazy L-stream is restored equivalently (`engine.py:425-450`).

**Teacher (§3).**
- Built by `build_q3d_teacher` from the recipe copied by value (`config.py:84`). I built the real ResEnc-L architecture locally (no weights): 101,750,172 parameters.
- Live feature phases are `((1,1,1),(1,2,2),(1,4,4),(2,8,8),(2,16,16))`, all offsets 0, widths `(32,64,128,256,320)`. They are derived from the live main-path convs and checked against the Q3D `phase_offsets` (`teacher.py:35-75`).
- Five features are captured by scoped hooks during one encoder pass (`teacher.py:99-119`).
- Parameter groups are deduplicated, so the decoder alias stays in the encoder group.
- In T-a the encoder has `requires_grad=False` and grads are set to None, so AdamW skips it.
- The loss is `q3d_teacher_loss` with Q3D's coefficients.
- The teacher is selected by VAL FG every epoch with strict `>`, so ties keep the earliest.
- `freeze_for_joint` keeps the teacher in eval even when `.train()` is called later. The engine asserts no grads and eval mode before and after every J step, and after export.

**Student and fusion (§4).**
- Live student phases are `((1,1,1),(1,2,2),(1,4,4),(1,8,8),(1,16,16))`, so native z stride stays 1.
- P/G placements are correct: F0 after the stem conv and before its residual block; each S level after its down-conv and before that level's blocks (`student.py:119-123`).
- Gate init: zero weight, bias −2. Measured G = 0.119203 on all five levels.
- F = F_S + m·G·P(F_T) with a detached teacher.
- Alignment uses `align_teacher_to_student4` for each level's phase, then replicates the internal high-side pad. Repro: S3 z gives [0,2,4,6,8] from teacher [0,4,8], which is exact.
- m=0 is bitwise identical to the pure route (shapes (5,37,29), (8,48,64) and (9,23,41)), and no unused fusion is computed.
- Mimic is 1−cos at S2/S3/S4 over the native valid-cell mean, averaged over the 3 levels, applied only to m=0 frames, and averaged over B. m=1 gives a graph-preserving 0.
- p_drop(t) = 0.5·t/max(T−1,1). Measured end-of-run keep rate: 0.501.

**J objective (§5).**
- `engine.py:134-138` computes `CE_Dice(L) + min(1, t/ceil(0.1·8000))·SoftCE(U, sg softmax T_U) + 0.1·(d_L + d_U)`.
  - One pair gives one step, with no halving.
  - SoftCE is the mean over native voxels at temperature 1. Teacher and student logits are both cropped to native.
- One keep mask is drawn independently per frame. Each frame is transformed once and shared between its teacher and student passes.
- Student optimizer: AdamW 3e-4 / 1e-4, 5% warmup then cosine, clip 1, bf16 autocast on GPU.
- J selection: fusion VAL FG with strict `>`. Deploy rule: `student >= fusion − 0.01`, with the margin locked.
- Evaluation: patient macro over ED/ES, FG = mean of RV/MYO/LV, with absent classes excluded the same way in every mode.

**Export and TEST.**
- Three exports with SHA sidecars. The student export has no `fusions.*` or `mimic_heads.*`. The fusion export has the teacher + P/G + student and no H heads.
- Cold-load is bitwise equal in a fresh interpreter.
- The campaign verifies the three-artifact bundle and calls `evaluate_exports` once. Any existing ledger, or a non-`NOT_RUN` campaign ledger, refuses a rerun.

## 4. Remaining findings (all Low, none blocking)

- **L1: an orphaned selection artifact can survive a crash.** `engine.py:615-631` (T) and `:731-742` (J) write `teacher.pt` / `joint_best.pt` before the boundary checkpoint.
  - Failure path: a crash in that window resumes from the last 20-update checkpoint with an older `best_*`. If the best-effort GPU replay scores at or below that restored best, the on-disk artifact from the archived epoch stays in place. It is then used at the end (`:635-637`, `:746-749`) even though its validation row went to `metrics.uncommitted.*`.
  - Suggested fix: store the selected artifact SHA in each checkpoint payload, and on resume archive any on-disk selection whose SHA differs.
- **L2: dead profile path in the engine.** `engine.py:620,681,689` call `_teacher_reference(...)` without `profile=profile`. A `run_training(profile=True)` would raise as soon as it wrote the PROFILE `teacher.pt`. No caller uses this (`fuse_profile.py` drives the modules directly, and `train.py` never passes `profile`). Suggested fix: pass `profile=profile`, or remove the `profile` parameter.
- **L3: VAL3/EXPORT verification failures raise instead of recording FAILED.** In `fuse_campaign.py` (the VAL3/EXPORT branch), `RUN_STATE` stays `RUNNING` until the watchdog flags staleness. This is cosmetic for a supervised launch.
- **L4: TEST runs inside the campaign process.** The campaign calls `evaluate_exports` in-process rather than through the `evaluate.py` CLI. Spec §6 says "canonical entrypoints". The ledger semantics are correct; document this as intentional.
- **L5: stale docstring.** The `run_training` docstring still says "checkpointing every committed optimizer update" (`engine.py:462`).

## 5. Launch notes for the coordinator (not defects)

- Before launch, check that the real `--original-train-root` and `--original-test-root` resolve to ACDC `training/` and `testing/`. Record `BUILD_RECEIPT.json` (`u_mask_reads: 0`) and the prepare-sources receipt next to the run.
- After the 20T+20J profile, lock the J epoch budget before any VAL score is seen. The profile now uses the same determinism setting as production, but it excludes checkpoint I/O. Add about 400 T saves of ~1.4 GB plus SHA re-reads to the ETA.
- Report GPU determinism as BEST_EFFORT_GPU_WARN_ONLY. Do not claim bitwise GPU reproducibility for training.
