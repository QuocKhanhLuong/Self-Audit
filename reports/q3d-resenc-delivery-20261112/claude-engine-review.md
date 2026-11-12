# Independent read-only review: Q3D ResEnc engine (task_d56b64517a81)

- Date: 2026-11-12
- Target worktree: `/Users/alvinluong/orca/workspaces/Self-Audit/q3d-resenc-20261112` (uncommitted worker diff on top of `37851c7`)
- Scope: `self_audit/candidate_q3d/{engine,artifacts,stages,baselines}.py` and tests `test_q3d_teacher_schedule`, `test_q3d_artifact_manifests`, `test_q3d_stage_artifacts`, `test_q3d_baselines`, `test_q3d_engine*`.
- Context read: `AGENTS.md`, `ACTIVE_PIPELINE.json`, all of `docs/CANDIDATE_Q3D_SPEC.md`, plus `pretrained.py` and `teacher_resenc.py` where the engine depends on them.
- Mode: read-only. I made no repo edits and downloaded nothing. I wrote probe scripts and temp outputs only under the Claude scratchpad. Everything below is SOFTWARE_ONLY evidence from CPU synthetic runs.

## Verdict

The T-a/T-b schedule, the disjoint optimizer groups, and the sealed teacher-quality fields work as specified. **The shared-T `teacher_supervised` evaluation path, which reports `independent_baseline=false` from T@8 with no J, does not work on a real campaign producer.** I confirmed two blocking bugs by running it (F1 and F2). Both bugs are in this worker's scope. I also found two stale or broken tests, which look like integration work that is still in progress (F5).

## Confirmed bugs (in scope)

### F1. A shared T artifact from a VQ producer always fails `teacher_supervised` evaluation because it carries `tokenizer_state` (blocking)
- Where: `stages.py` `export_stage_artifact`, through `_ARTIFACT_STATE_KEYS`, which includes `tokenizer_state`. The rejection happens in `baselines.py` `evaluate_teacher_supervised_baseline`: `if payload.get("tokenizer_state") is not None ... raise "must not carry a tokenizer or EMA student"`.
- Mechanism: in V1, V1-u and the other VQ arms, `_build_models` creates the tokenizer before T runs. `_capture_run_state` at the end of T stores `tokenizer.state_dict()`, which is untrained but non-None. The T export copies that through the allowlist and nothing strips it for stage T.
- Reproduced with a synthetic v1 run (`stage="T"`) followed by `evaluate_teacher_supervised_baseline(T.checkpoint.pt, ...)` using the `teacher_supervised` config. The call fails with `ValueError: teacher-supervised checkpoint must not carry a tokenizer or EMA student`. `stage_source_config(ts,"T") == stage_source_config(v1,"T")` was True and the data identity matched, so this check is the first one that fails.
- Impact: the campaign's `teacher_supervised@8` reuses the T from v1@8 (`scripts/q3d_campaign.py`, routed through `evaluate.py:93`), so that evaluation cannot succeed.
- Fix: in `export_stage_artifact`, set `snapshot["tokenizer_state"] = None` and drop `tokenizer_artifact` when `stage == "T"`. Alternatively, have the evaluator ignore `tokenizer_state` when `stage_artifact_stage == "T"`. The first option is cleaner because T is arm-independent.

### F2. The shared T artifact has no `metrics`, so the DEV-selection cross-check always fails (blocking, found after working around F1)
- Where: `stages.py` `_ARTIFACT_STATE_KEYS` leaves out `metrics`. `baselines.py` `evaluate_teacher_supervised_baseline` requires `payload["metrics"]` rows with `event == "development_selection"`, and raises `"has no DEV selection history"` if there are none.
- Reproduced by setting `tokenizer_state=None` in a copy of the probe checkpoint (with a fresh sidecar). Evaluation then fails with `ValueError: teacher-supervised checkpoint has no DEV selection history`.
- Fix: for `shared_t`, cross-check against `teacher_dev_history` / `best_teacher_dev_update` / `teacher_selected_dev_update`, which are already in the allowlist and sealed. Do not add the full mutable `metrics` list to the shared artifact.
- Test gap that let F1 and F2 through: `test_q3d_reduced_campaign.py:62` only checks the plan field `independent_baseline is False`. No test evaluates a shared T checkpoint produced by an engine run. Add one in `test_q3d_baselines.py`: v1 synthetic `stage="T"`, then `evaluate_teacher_supervised_baseline`, then assert `independent_baseline is False`, `stage_contract == "T0J"`, and that the teacher-quality fields are present.

### F3. Teacher-quality sealing fails open (medium; provenance)
- Where: `artifacts.py` `_teacher_quality_from_checkpoint` wraps `torch.load(weights_only=True)` in `except Exception: return None`. The loader treats `teacher_quality` and `teacher_quality_sha256` as optional for T.
- Effect: if a future state key ever makes the weights-only load fail, the T manifest is still written. It is silently missing `teacher_dev_fg_dice` / per-class / selected / terminal / `dev_curve_still_improving`, and loads without error. That breaks the spec's guarantee that the T manifest stores teacher quality.
- Today it works. The probe's engine-produced T manifest contained all six fields with a matching `teacher_quality_sha256`, and the RNG state is tensorized, so weights-only loading succeeds.
- Fix: for `stage == "T"`, make a load failure or missing quality keys a hard error in both write and load. Keep the tolerant path only for explicit legacy byte fixtures in tests.

### F4. The T source contract does not seal `training.warmup_fraction`, which drives the T-b LR schedule (low-medium; identity)
- Where: `engine.py` `_run_teacher_stage` calls `_schedule_lr(..., warmup_fraction=config["training"]["warmup_fraction"])`. `stages.py` `stage_source_config("T")` seals `teacher_warmup_fraction` but not `warmup_fraction`.
- Effect: two arms that differ only in `warmup_fraction` would share one T manifest while training different teachers. Every config currently uses 0.05, so this is latent.
- Fix: seal it, or better, add a dedicated `teacher_lr_warmup_fraction`. Related: `teacher_boundary_lr` is sealed and validated but the engine never reads it, because boundary parameters use `teacher_head_lr`. Either wire it in or remove it, so the sealed identity does not imply a knob that does nothing.

## Test failures observed (probably concurrent or unfinished work; not engine-logic bugs)

### F5. 2 of 55 scoped tests fail
I ran `pytest -q` on the 7 scoped test files: 53 passed, 2 failed, all in `tests/test_q3d_stage_artifacts.py`.
1. `test_stage_identity_and_source_configs_are_arm_independent`: `config.py:184 ValueError: feature_stride must be 8 or 16`. The test still builds `arm_config("v1", source_stride=4)` and `feature_source="decoder_stride8"`. Those are old-teacher paths that the ResEnc replacement correctly removed. The test is stale and should assert rejection instead of equal-T/different-Q. The production behaviour is correct.
2. `test_tqp_artifacts_are_shared_by_v1_and_v1u_joint_stages`: `TypeError: _manufactured_teacher.<locals>.forward() got an unexpected keyword argument 'return_features'`. The instance-level `forward` patch in `train.py:_manufactured_teacher` has the signature `(image, geometry=None)`. `Q3DResEncTeacher.forward_with_features` calls `self.forward(..., return_features=True)`. The fix belongs in `train.py`, outside my file scope: accept `**kwargs` / `return_features` and pass them through. The consequence is that the engine-level T→Q→P→J sharing test currently gives no coverage.

I could not tell from this terminal whether another worker owns `train.py` or the stale test. Re-check both after the other workers finish before treating them as defects of this delivery.

## Checklist verified (no bug found)

- **T-a (first 5%)**: `warmup_updates = ceil(8000*0.05) = 400`. `_teacher_stage("warmup")` calls `requires_grad_(False)` and `eval()` on the encoder. `_teacher_head_parameters` returns exactly the non-encoder parameters, `region_decoder` and `boundary_path`, with the `region_decoder.encoder` alias excluded by `id`. LR is a constant `teacher_head_lr = 1e-3`. The test confirms the warmup id set equals all non-encoder parameters.
- **T-b**: `_teacher_stage("pretrain")` re-enables the whole network. `_teacher_parameter_groups` assigns groups by parameter identity: `teacher_trunk` (encoder, 1e-4) and `teacher_heads` (everything else, 1e-3). The groups are disjoint even with the `decoder.encoder` alias. AdamW uses wd 1e-4, clipping covers only optimizer parameters, and the budget is 8000. O1 reuses the same disjoint grouping with `teacher_joint_lr_multiplier`.
- **Interpretation to confirm**: the T-b warmup is 5% of the T-b span (380 updates) using `training.warmup_fraction`. At the T-a→T-b boundary, the head LR drops from 1e-3 to about 2.6e-6, then ramps back up, and Adam moments are reset. The spec's "warmup LR 5% rồi cosine" is ambiguous, so confirm this is the intended reading.
- **Selected vs final**: `best_teacher_*` is updated with strict `>`. On completion: `teacher_dev_fg_dice` / `teacher_dev_per_class` / `teacher_selected_dev_update` are the selected checkpoint, `teacher_terminal_*` is the last evaluation (always run at `cursor == budget`), and `dev_curve_still_improving` is the last 3 actual points, strictly increasing, else `INSUFFICIENT`. The best state is reloaded and its SHA recorded. All of these are in `_ARTIFACT_STATE_KEYS`, sealed into the T manifest (`teacher_quality` plus a SHA, verified by the probe), and copied into J state by `import_stage_artifacts`.
- **Logging**: `_log_teacher_development` runs every 400 updates and at the final update, writes to `sys.stderr` only, and is the only `print` in the scoped files. JSON stdout is preserved; a test asserts `captured.out == ""`.
- **Shared `teacher_supervised` design**: the real `run_teacher_supervised` refuses to train standalone. For a shared T, the evaluator seals against `stage_source_config(...,"T")` and reports `independent_baseline=False` / `T0J`, with no extra T job. The design is correct but unreachable because of F1 and F2.
- **O1 and non-specific arms**: `_student_arm` still maps e1, o1, s1, v0, b1 and mean_teacher. O1 keeps its own Q/P family and online teacher optimizer, and S1 keeps separate Q/P with a shared T. No old teacher backend paths (MedNeXt, Tiny, `decoder_stride8`, stride4) remain in the four scoped files. The remaining legacy items are dead names only: `bottleneck_stride16` / `stride16` aliases in `_features_to_student_grid`, and the `{4:2}` index in `_teacher_feature_channels`. The `tiny_test3d` in `baselines.py:126` is the frozen A-3D teacher-free baseline's own smoke backbone, not a Q3D teacher.
- **Synthetic teacher**: `Q3DResEncTeacher(synthetic=True)` is the same class, built with a 6-stage (4,8,12,16,20,20) fixture. There is no TinyEncoder fallback, and the real path requires the pinned SHA, an existing file and strict encoder loading.
- **Serialization and aliases**: the state dicts contain both `encoder.*` and `region_decoder.encoder.*`. `deepcopy` and `torch.save` keep them as one shared object, and `Q3DResEncTeacher.load_state_dict` rejects divergent aliases. The engine's RNG state is tensorized, so `weights_only=True` loads work.

## Minor / efficiency (no action required)

- In T-a, `_development_fg_dice` calls `model.train(was_training)`, which puts the frozen encoder back in train mode until T-b starts. It has no numerical effect: InstanceNorm has no running stats and there is no dropout, and gradients stay off.
- The T export deep-copies the selected state twice (`teacher_state` and `best_teacher_state` as separate copies, about 2×400 MB for ResEnc-L). Each in-stage checkpoint carries the live state, the best state and Adam state, and is written twice (stage and root `latest.pt`) every 400 updates. That is a disk and I/O cost, not a correctness problem.

## Known-pending items (not reported as bugs)

- Phase offsets are all `[0,0,0]` in every teacher config. The phase-0 ERF calibration (rule: use the main phase if the mean centroid offset over 16 random samples is ≤2 voxels on every axis, otherwise use the measured offset) is pending until its report lands. The offsets are included in the T/Q/P source identity, so changing them later correctly invalidates sharing.
- The strict 448-key encoder load PASS, the checkpoint SHA `ac262d3e…2bbd`, HF revision `f443c093…7b2`, DNA 0.4.4 and nnssl pin `3d774e94…ed74` are the user's measurements. I cross-checked them against `checkpoint_inspection.json` (`PASS_STRICT_ENCODER`) and did not re-run the load.

## Evidence scope

All probes ran on CPU with synthetic data and are SOFTWARE_ONLY. There is no patient, GPU or efficacy evidence.
