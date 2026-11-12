# Independent Claude review: Q3D ResEnc campaign (read-only)

- Date: 2026-11-12, task `task_2a2ad7dcc354`
- Target worktree: `/Users/alvinluong/orca/workspaces/Self-Audit/q3d-resenc-20261112` (uncommitted state on top of `37851c7`)
- Read: `AGENTS.md`, `ACTIVE_PIPELINE.json`, the full `docs/CANDIDATE_Q3D_SPEC.md` (418 lines), `scripts/q3d_campaign.py`, `self_audit/candidate_q3d/campaign_runtime.py`, `self_audit/candidate_q3d/artifacts.py`, and the relevant parts of `stages.py`, `engine.py`, `config.py`, `pretrained.py`, `teacher_resenc.py`, `data.py`, `train.py`, and `self_audit/fewshot.py`. Tests read: `tests/test_q3d_campaign.py`, `test_q3d_campaign_runtime.py`, `test_q3d_reduced_campaign.py`, `test_q3d_config_campaign.py`, and `test_q3d_teacher_schedule.py`.
- No repo edits. I trained nothing on patients, downloaded nothing, and changed no jobs. The only outputs are synthetic and live in the session scratchpad.
- **Concurrency caveat:** other workers kept editing `scripts/q3d_campaign.py` during this review (mtime 17:04:08; it grew from 1856 to 1915 lines, adding a schema/role lineage audit) and staged files (`A configs/candidate_q3d/teacher_supervised.json`, `teacher_resenc.py`, and several tests). I re-checked the findings below against the 17:04 copy. Line numbers refer to that copy unless noted.

## Evidence I produced (SOFTWARE_ONLY, synthetic, CPU)

1. `pytest -q tests/test_q3d_campaign.py tests/test_q3d_campaign_runtime.py tests/test_q3d_reduced_campaign.py tests/test_q3d_config_campaign.py tests/test_q3d_teacher_schedule.py`: **42 passed** (3.8 s). `git status --porcelain` was unchanged by the run.
2. End-to-end synthetic shared campaign, run in scratch through the real `train.py` subprocesses:
   `q3d_campaign.py plan --synthetic --smoke-updates 1` (default six-run catalogue), then `run --execute --setting all --device cpu`. Exit code was 0.
   - Exactly **3 T, 3 Q, 3 P and 6 J** jobs ran and finished COMPLETED.
   - V1@8, V1u@8 and C1@8 recorded the same T/Q/P manifest SHAs (`T=a040ce04…`, `Q=c527a147…`, `P=6e28327a…`). C1's J command carries the same three `--q3d-stage-artifact[-sha]` pins as V1u, with Q as optional metadata.
   - A real engine T checkpoint does load with `weights_only=True`, and its `teacher_quality` is sealed in `T.artifact.json`: selected DEV FG, per-class scores, the selected update, terminal scores, and `dev_curve_still_improving` (`INSUFFICIENT` when there are fewer than 3 points).
   - `results/RESULTS.md` showed teacher DEV Dice@K for K=5/8/10, and C0 DEV Dice@8 beside the K8 teacher with a delta. C0@5 and C0@10 were correctly `NOT_RUN`. The teacher score was bit-identical across two independent runs, so it is deterministic.
   - These numbers are synthetic fixture values only. They are not efficacy evidence.
3. Real-plan probe (scratch only, fake non-K8 manifest JSONs): planning with the actual checkpoint at the default path `/Users/alvinluong/Self-Audit-runs/q3d-resenc-20261112/checkpoint_final.pth` gave `READY` for T/Q/P/J. `validate_real_teacher_provenance` hashed the file and checked the pinned URL, pins, dataset, overlap note and license. I did not repeat the strict 448-key load; I took that PASS from the brief.

## Findings (own campaign scope)

### F1 — BUG (low-medium): an explicit run catalogue without K8 cannot be planned
`create_campaign_plan` builds `baseline_settings` (`q3d_campaign.py:1067-1071`) as `[teacher_supervised]` only when K8 is in the runs. Otherwise it falls back to `list(BASELINE_ARMS)`, because `include_baselines` defaults to True. Validation (`:1536-1540`) expects `[]` when K8 is absent, so planning fails.
- Repro: `create_campaign_plan(out=..., runs=("v1@5","v1@10"), synthetic=True)` and `runs=("c0@5",)` both raise `ValueError: explicit run baseline selection is stale`.
- The default six-run catalogue includes K8, so the main path is unaffected. A curve-only or partial replan does fail.
- Fix: use `[]` whenever `explicit_runs is not None` and 8 is not in the runs. No test covers this case.

### F2 — Fail-closed gap (medium): the nesting check trusts lineage metadata, not the cohort actually trained on
`_validate_nested_train_lineage` (`:234-291`) compares only `fewshot.selected_ids` across K. It never checks that `{r["id"] for r in labeled}` equals `selected_ids`. It also does not re-derive the cohort with `label_subset(eligible_ids, k, seed)`.
- Probe: a K5 manifest whose `selected_ids` are nested in K10, but whose `labeled` IDs are disjoint (`x1..x5`), was **accepted**.
- K8 is protected by its SHA lock. K5 and K10 are protected only by the lineage claim. The runtime engine checks `len(labeled)==k` (`engine.py:254`) and protocol-A teacher/student equality (`data.py:40-56`), but it never checks nesting.
- Fix: require `sorted(labeled ids) == sorted(selected_ids)`. Optionally also recompute the subset from `eligible_labeled_ids`.

### F3 — Fail-closed gap (low): plan readiness can say READY for a manifest that is the wrong K
In the legacy/`--k` path, or in explicit runs where no K-specific train is given, a global `acdc:train` manifest is used for every K. `label_budget` is only enforced when `enforce_k8_lock` is set (K8 only, `:217-224`).
- Probe: a `label_budget=8` manifest planned for K5 gave `READY` on all 7 jobs.
- Runtime still fails closed: the engine checks `len(labeled)!=k` at `engine.py:254`. The plan's readiness and `lineage_audit` are misleading, though.
- Fix: check `label_budget == k` for every K in `_set_manifest_inputs`.

### F4 — Fail-closed gap (low): `lineage_audit` and the new schema audit are report-only
`execute_shared_campaign` never reads `plan["lineage_audit"]`. The new schema/role audit (`:258-269`, added concurrently) marks a non-`self_audit.data.v2` K manifest `NOT_RUN` and then `continue`s. That skips the nesting comparison for that K, yet the jobs keep `READY`. Nothing propagates `nested_validation=PARTIAL_NOT_RUN` or `schema_loader_status=NOT_RUN` to job readiness.
- `load_config(real_run=True)` or the data loader would probably reject such a manifest at run time.
- Suggestion: make readiness `NOT_RUN` whenever the lineage or schema audit is not `PASS`/`READY` for a K that has jobs.
- This code is concurrent unfinished work, so the owning worker may still be wiring it up.

### F5 — Hardening (low): plan-time provenance accepts any self-consistent digest
`config.validate_real_teacher_provenance` (`config.py:835-892`) checks file SHA == declared SHA and the pinned URL. It does **not** check declared SHA == `NNFOUNDATION_WEIGHTS_SHA256`. `teacher_resenc.load` also just sets `hf_revision=None` for a non-pinned file.
- With `--teacher-weights other.pth --teacher-weights-sha256 <its sha>`, the plan therefore reports READY.
- The engine build path does fail closed: `pretrained.py:91` rejects a non-pinned digest, so this is a readiness/reporting inconsistency, not a training hole.
- Fix: add the pinned-SHA equality to `validate_real_teacher_provenance`.

### F6 — Hardening (low): runtime receipt checks
- `campaign_runtime.py:373` accepts `COMPLETED_SOFTWARE_ONLY` even when `plan.selection.synthetic` is False. It also never checks the receipt's `runtime_synthetic`/`evidence_status` against the plan. A real campaign could record a synthetic receipt as completed if a trainer misbehaved.
- `runner` results without `returncode` default to code 0 (`:368`). In practice `json.loads("")` then fails, so the job is still marked FAILED.

### F7 — Dead code (low): the C0 `metrics.jsonl` fallback can never yield a score
`_selected_c0_score` (`campaign_runtime.py:138-151`) passes `development_selection` records to `_score_from_mapping`. That function looks for `student_dev_fg_dice`, `best_student_score`, `development_patient_macro_fg_dice` or `development.score`, but the engine records `score` and `best_score` (`engine.py:3021-3025`). The fallback therefore always returns None (NOT_RUN), which is safe but dead. The `latest.pt` `best_student_score` path works: it produced the C0 value in the E2E run.

### F8 — Hardening (low): the guard for a published T checkpoint can be bypassed
`artifacts._teacher_quality_from_checkpoint` returns None on *any* `torch.load` exception, before the `is_published_t` missing-fields guard runs. If a future T snapshot stopped loading with `weights_only=True` (for example numpy RNG objects), the manifest would be written without `teacher_quality`, and results would quietly show teacher `NOT_RUN` instead of failing. It loads today (E2E verified). Consider raising when `stage=="T"` and the file is a torch zip but fails to load.

### F9 — Spec drift / NOT_RUN gaps (informational)
- `DEFAULT_REPORT_DIR` is still `reports/q3d-implementation-20261112/campaign`. Spec §12 says the new PLAN goes under `reports/q3d-resenc-plan-20261112/`.
- The DEV18 SHA lock applies only to the K8 development manifest. V1@5 and V1@10 could be planned against a different DEV manifest without error.
- Teacher-supervised: `evaluation_jobs[*]` (T region head, 0 J) are planned but nothing executes them. `execute_shared_campaign` marks the teacher_supervised experiment COMPLETED with `evaluation: NOT_RUN`, and the teacher DEV score comes from the T manifest. This is reported honestly, but TEST metrics for `teacher_supervised@8` stay NOT_RUN.
- `--independent-stages` / the legacy path still exists alongside shared mode (intentional per the docstring). It is not part of the six-run contract.

## Checked and OK (assumptions stated)

- **Exactly six J jobs:** the plan, the test, and the E2E run all give 3T/3Q/3P/6J in order `v1@8 v1u@8 c1@8 c0@8 v1@5 v1@10`. C0 has only J. K5, K8 and K10 each have a distinct T.
- **Shared producer hashes:** V1, V1u and C1 at K8 consume the same producer IDs. Runtime pins the manifest-file SHA at production, revalidates it before every reuse (`_verified_output` with `pinned_sha`), and passes `--q3d-stage-artifact-sha` to `train.py`, which binds the SHA to its manifest. `load_stage_manifest` recursively re-hashes the checkpoint, the source_config seal, the payload, the parents and `teacher_quality`. A tampered T manifest stops the next arm (tested).
- **Manifest locks:** the K8 TRAIN and DEV SHA locks are enforced in `_set_manifest_inputs` and again in the nesting check. A non-locked K8 file is rejected (tested). Nesting has the caveats in F2 and F4.
- **A-3D:** reference only. It is an `external_references` record with `NOT_RUN` / `REFERENCE_ONLY_NOT_RUN`, it creates no experiment or job, and validation rejects any experiment that has it.
- **Real READY provenance:** requires the nnFoundation architecture, the 6-stride config, the DNA/nnssl/nnunet pins, `READY`, the exact pinned HF URL, the dataset, both overlaps false, the overlap note, CC-BY-SA-4.0 and the file SHA, with the F5 caveat. Real execution requires CUDA plus `load_config(real_run=True)`, and the engine re-verifies the pinned SHA.
- **Teacher quality beside C0:** T exports `best_teacher_state` (the DEV-selected one, loaded strict before export). Manifest `teacher_dev_fg_dice` is the selected score, and the terminal score is kept separately. The join key is exactly (dataset, protocol, K, seed), with no cross-K inference, and missing values stay `NOT_RUN`.
- **Not reviewed in depth** (outside campaign scope, or owned by other workers): teacher_resenc forward/phase/ERF math, losses/gradients, tokenizer/physiology numerics, export. Phase-0 ERF calibration is pending its report, per the brief.

## Scope attribution

- Own scope (campaign planner/runtime): F1, F2, F3, F5 (config readiness), F6, F7, F8, F9.
- Concurrent unfinished work by other workers: F4, which comes from the new schema/role lineage audit added during this review. The staged `teacher_resenc.py`, `teacher_schedule` and `reduced_campaign` files were in flux. Any root commit should re-run the campaign tests after those workers finish.
