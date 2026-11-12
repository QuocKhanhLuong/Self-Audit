# Q3D ResEnc cleanup — independent read-only Claude review

- Date: 2026-11-12
- Worktree: `/Users/alvinluong/orca/workspaces/Self-Audit/q3d-resenc-20261112` (base `37851c7`, dirty, uncommitted)
- Mode: read-only. I made no repo edits. Tests and smoke wrote only to pytest tmp and the Claude scratchpad.
- Scope reviewed: `train.py`, `evaluate.py`, `scripts/smoke_q3d.py`, `candidate_q3d/{export,diagnostics,losses}.py`, and the tests `test_q3d_{export,tokenizer,physiology_data,integration,repository_clean}.py`. For context I also read `teacher_resenc.py`, `teacher_features.py`, `pretrained.py`, the engine teacher groups and `config._base_config`.
- Evidence scope: SOFTWARE_ONLY. No patient training, downloads, GPU work or checkpoint fetch.

## Verdict

The worker's claim that the cleanup is **code-complete is not yet true end to end.** The rename to `teacher_supervised`, the student-only cold-load guard and freezing A/A3D are correct. However, the canonical AGENTS smoke currently **FAILS**, and 9 scoped tests fail. Some failures are stale tests in the worker's scope. Others come from concurrent config, preflight or pretrained work that is still in progress.

## Verified OK

1. **A/A3D frozen.** `git diff --stat` and `git status` show no change in `self_audit/candidate_a{,3d}`, `configs/candidate_a{,3d}` or `tests/test_candidate_a*`. `train.py` changed one line: the runner-map rename. A/A3D dispatch is untouched.
2. **The synthetic teacher is the same class.** `Q3DResEncTeacher(synthetic=True)` builds the same `ResidualEncoderUNet` encoder, `UNetDecoder` and `_BoundaryPath3D` as the real teacher, with widths (4,8,12,16,20,20) and 1 block per stage. It shares the forward, pad, crop and phase code. `pretrained._synthetic_teacher` routes through `_construct_resenc_teacher(..., synthetic=True)`.
3. **The `teacher_supervised` rename is consistent** across `train.py`, `evaluate.py`, `smoke_q3d.py`, `config.BASELINE_ARMS` and `baselines.{run,evaluate}_teacher_supervised*`. Export `_student_arm` rejects `teacher_supervised`, so the heavy baseline never produces a student export. This is correct.
4. **Student-only cold load.** `_assert_teacher_free_imports` and the subprocess guard now forbid `candidate_q3d.teacher_resenc`, `dynamic_network_architectures.*` and `nnssl.*`. I checked that importing `export`, `student` and `evaluation` in a fresh interpreter loads no teacher or DNA modules. `_construct_deploy_model` builds only `CandidateQ3DStudent`, drops the SDF and continuous heads, and runs a strict `load_state_dict`. `test_q3d_export.py` passes, including the new parametrized guard test.
5. **Removed tests.** The deleted MedNeXt checkpoint-dispatch, decoder_stride8 and stride4 hook tests covered paths the spec now removes (§3, §11, Migration). The replacements cover the ResEnc equivalents: one-forward capture, stage3/stage4 selection, rejection of `decoder_stride8`, and detach. Deleting `test_q3d_official_channels.py` matches removing the MedNeXt source. I found no removed test that covered a still-live contract.
6. **Alias and gradients (context, outside scope).** `UNetDecoder` registers `region_decoder.encoder` as an alias. `_teacher_parameter_groups` and `_teacher_head_parameters` split parameters by `id()`, so the groups are disjoint. T-a's `encoder.requires_grad_(False)` also freezes the alias, because it is the same tensors. Teacher feature outputs are detached (`test_feature_forward_is_detached...`).
7. **Provenance (context).** The real path `build_q3d_teacher` requires READY, the exact field set, SHA == the pinned `ac262d…`, the HF URL, the Table D7 dataset and overlap note, CC-BY-SA-4.0 and the dependency pins, and it re-hashes the file. I confirmed independently: the file SHA-256 is `ac262d3e8c226c79f9567fddc34d38e011284730ddf2294b5c6c319ce2f22bbd`, `dynamic_network_architectures` is 0.4.4, and `requirements.txt` pins `==0.4.4`. The strict 448-key test passes locally because the external checkpoint is present.
8. `scripts/check_repo.py` passes (q3d configs: 13).

## Findings

### F1 — HIGH, blocks acceptance: the canonical smoke fails at stage T
`python scripts/smoke_q3d.py --out <scratch>/smoke-v1` exits 1 with `status: FAILED`, `completed_stages: []` and:
`ValueError: Expected more than 1 spatial element when training, got input size torch.Size([1, 20, 1, 1, 1])`.

Cause:
- The smoke fixture is `shape_tzyx=(2,8,32,32)`.
- `config.NNFOUNDATION_DEFAULT_ENCODER_STRIDES` is isotropic (1,2,2,2,2,2), so the cumulative stride is 32 on every axis.
- `_pad_high` pads 8×32×32 to 32×32×32, so stage 5 is 1×1×1.
- InstanceNorm3d always uses input statistics, so it raises in eval too.

The same root cause fails `test_q3d_diagnostics.py` (3 tests: synthetic CLI, odd 17×19, o1). Unit tests hide this because `_small_resenc_teacher` uses a last stride of (1,1,1) and 64³ inputs.

Ownership: the smoke fixture shape is in this worker's scope. The default strides belong to config and Phase 0 (concurrent). Either side can fix it, but the AGENTS workflow command must pass before code-complete. Suggested fixes:
- (a) Have the smoke declare explicit synthetic `encoder_strides` (for example in-plane-only after stage 0), or use a larger fixture.
- (b) Make `Q3DResEncTeacher` fail closed with a clear message when any stage's padded extent product would be ≤ 1, rather than surfacing a raw InstanceNorm error.

Real ACDC (D≈8–10, H/W≈216–256) does not hit this: stage 5 is about 1×8×7. The defect is in synthetic and small inputs.

### F2 — MEDIUM, own scope: `smoke_q3d.py` still has deleted recipes
`_ALL_RECIPES` still lists `v1_stride4` and `v1_decoder_stride8`, `_recipe_config` still constructs them, and `--all-arms` help says "15". Both now raise inside `_recipe_config`, which is outside the per-recipe `try`:
- `v1_stride4` raises `KeyError: 4` from `{8:...,16:...}[4]`.
- `v1_decoder_stride8` raises `ValueError: nnFoundation feature_source must match the selected encoder stage`.

So `--all-arms` aborts with a traceback at the 12th recipe, after writing partial outputs. `--recipe v1_stride4` is also still an accepted CLI choice. Remove both recipes and update the count to 13. In `export.py:168`, the alias `"v1-stride4": "V1"` is dead code; it is harmless but should go.

### F3 — MEDIUM, own-scope tests are stale (4 failures in listed files)
- `test_q3d_integration.py::test_all_fifteen_recipes_select_declared_engine_effects` asserts 15 configs, but there are 13. It still has a `v1_stride4` branch at line 94.
- `test_q3d_integration.py::test_q3d_configs_are_nested_and_keep_real_provenance_not_run` expects `NOT_FOUND_OFFICIAL`, but configs now say `READY`. The intended contract must be decided with the config owner. Committing a READY absolute local path into canonical configs is a portability and provenance-policy question, not a code bug.
- `test_q3d_physiology_data.py::test_cache_cli_default_synthetic_teacher_path_is_explicit_and_runs_cpu`: the test config only renamed `architecture` and lacks `dependency_pins`/`encoder_strides`. It now fails with `nnFoundation teacher dependency pins do not match Phase 0`. That is correct fail-closed behaviour, but the fixture needs those fields.
- `test_q3d_tokenizer.py::test_p0_preflight_keeps_missing_weights_and_patient_diagnostics_not_run` expects `NOT_RUN`. The preflight now reports `ACTUAL_RESENC_MODULES_INSPECTED`, because `scripts/q3d_preflight.py:27-33` defaults to the absolute path `/Users/alvinluong/Self-Audit-runs/...`. The test result therefore depends on the machine (concurrent preflight work). Either pin the env var in the test, or drop the hard-coded default.
- Outside the listed files: `test_q3d_evaluation_export.py[v1_stride4-V1]` parametrizes a deleted config.

### F4 — LOW, expected until commit: repository_clean fails on untracked new modules
`test_full_pytest_suite_leaves_tracked_working_source_clean` snapshots tracked files only. `teacher_resenc.py`, `test_q3d_teacher_resenc.py`, `test_q3d_teacher_schedule.py`, `test_q3d_reduced_campaign.py` and `configs/candidate_q3d/teacher_supervised.json` are untracked, so collection fails with `ModuleNotFoundError: self_audit.candidate_q3d.teacher_resenc`. This resolves when root `git add`s them. Root should check that all five are in the commit.

### F5 — LOW, test gap: the strict-load test proves itself
`test_official_checkpoint_strict_encoder_load_and_ignore_decoder` hashes the file and passes that same digest to `load_nnfoundation_encoder`. The SHA check is therefore tautological; any file whose keys match would pass. Add `assert digest == "ac262d3e…"` (the pinned constant). Also, `load_nnfoundation_encoder` writes `hf_revision` into its metadata as a literal, without checking it against anything. The only real caller (`build_q3d_teacher`) pins the SHA first, so this is not exploitable today. Still, the function should either take the revision as a checked argument or check the SHA against the pinned constant itself.

### F6 — LOW, maintainability: the forbidden-import list is duplicated
The cold-load subprocess script re-implements `_assert_teacher_free_imports` inline. The two copies match today but can drift. The subprocess could call the helper instead.

### F7 — INFO, concurrent and pending: no channel for measured Phase-0 offsets
`Q3DResEncTeacher` accepts `phase_offsets`, but neither config nor `pretrained.build_q3d_teacher`/`_construct_resenc_teacher` passes it. The user's rule is: use the main-branch phase when the average 16-sample centroid offset is ≤ 2 voxels on every axis, otherwise use the measured offset. Calibration is pending. If the report exceeds the threshold, there is currently no config key or artifact-identity path to lock the measured offset, and the teacher would silently fall back to (0,0,0). This must be wired, or a fail-closed check added, before a real T run. I am not reporting it as a bug now, because calibration is explicitly pending.

### F8 — INFO, concurrent: z-strides and geometry ambiguity
`geometry-audit.json` reports a median TRAIN spacing of [1,1,1] (52/80 patients declare z = 1 mm; the rest declare 10 mm), with status `PASS_WITH_NATIVE_GEOMETRY_AMBIGUITY`. Applying the planner rule to that median gives isotropic strides, so D≈8 is replicate-padded to 32. That is 4× teacher compute, and stage-3 features have about one z-cell per 8 slices. This is a Phase-0 and config decision, outside cleanup scope; I flag it so the stride lock is made deliberately.

### Minor notes (no action required)
- `diagnostics.diagnose_checkpoint` now requires stage T for any teacher architecture outside {none, ema_student, candidate_a3d}. That fails closed in the safe direction.
- `losses.py` changed only a docstring.

### F1b — HIGH, own scope (`train.py`): the manufactured synthetic teacher breaks feature capture
`train._manufactured_teacher` replaces `teacher.forward` with `forward(self, image, geometry=None)`, which has no `return_features` parameter. `Q3DResEncTeacher.forward_with_features` calls `self.forward(x, geometry=..., return_features=True)`, which resolves to the patched instance attribute. I reproduced this on a 64³ input that avoids F1:
`TypeError: _manufactured_teacher.<locals>.forward() got an unexpected keyword argument 'return_features'`.

Every feature path goes through `engine._teacher_outputs` → `forward_teacher_with_features` → `teacher.forward_with_features`. That covers Q, J codeKD, C1, diagnostics and the smoke and `--synthetic-teacher-fixture` paths. So once F1 is fixed, the smoke will next fail at the first feature-consuming stage.

Fix: accept `return_features=False` and forward it, `original_forward(image, geometry=geometry, return_features=return_features)`. Keep the region-logit override, so that `region_logits` and features come from the same forward. With the old MedNeXt adapter the patch worked only because features came from a separate API. Add a regression test that calls `forward_with_features` on the manufactured teacher. SOFTWARE_ONLY; real teachers are unaffected.

## Test run (scoped, read-only)
`pytest -p no:cacheprovider` on export, tokenizer, physiology_data, integration, teacher_resenc, teacher_schedule, diagnostics, evaluation_export and repository_clean: **98 passed, 9 failed.** The failures are listed under F1, F3 and F4. Smoke `--recipe v1`: FAILED (F1), exit 1. F1b reproduced directly.

## Own scope vs concurrent
- Worker's own scope, needs a fix before claiming complete: F1b (`train.py` fixture `return_features`); F2; the stale tests in F3 (integration count and stride4 branch, physiology fixture, evaluation_export parametrize); the smoke fixture/strides side of F1.
- Concurrent or unfinished elsewhere: config default strides and READY provenance (F1, F3); preflight hard-coded path (F3); Phase-0 offset wiring (F7); geometry ambiguity (F8); committing untracked files (F4, root).

No concrete defects found in: gradient isolation, the alias param groups, export serialization or strict load, cold-load import isolation, or A/A3D immutability.
