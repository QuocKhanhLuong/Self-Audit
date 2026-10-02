# Current-main cleanup audit — 2026-10-02

**Decision: seven exact files are safe for deletion without code migration; three substantial legacy executable files can additionally be retired after the concrete helper extraction in Section 6. No source directory, model family, comparison baseline, test suite, dataset, checkpoint, freeze, or research evidence directory is approved for wholesale deletion.** This worker changed only this report; it performed no deletion, Git mutation, training, or model evaluation.

The user now requires fully no-GT learning: no manual masks/scribbles and no mask-supervised teacher; GT is allowed only for frozen independent evaluation. That changes which experiment should run next. It does not make supervised reference implementations, evaluation code, or shared model components disposable. In particular, the current v3 student directly uses a component from `self_audit`.

## 1. Source identity and audit method

| Item | Audited value |
|---|---|
| Authoritative main snapshot | `c31825a7f90df47f9f9382a9c9595e43f7f71236` (`origin/main`, merge PR #29, 2026-10-01) |
| Refresh receipt | Coordinator confirmed the fetch and remote verification; this worker independently resolved the local ref to the same SHA. The task's initial `0605e3b` is stale. |
| Open checkout | `234be0a9cc9d557d1263025f163ef28792da4423`, branch `QuocKhanhLuong/astra-pseudolabel-v3-audit` |
| Divergence | `git rev-list --left-right --count HEAD...c31825a` = **2 checkout-only / 39 main-only commits**; 534 changed paths between trees |
| Main checkout supplied by coordinator | `/private/tmp/self-audit-main-20261002`; this report's decisive source reads use immutable Git blobs, so concurrent checkout edits cannot silently change the evidence |
| Ownership | Only `reports/main_baseline_20261002/02_CLEANUP_AUDIT.md` in the original workspace |
| Evidence status | Source/reachability/history audit **COMPLETED**; static parsing **TESTED**; deletion **PROPOSED, NOT APPLIED**; runtime tests/training/Dice/speed **NOT RUN** |

No `.codegraph/` exists. Semble was used first for locating entrypoints, reproduction documentation, and provenance logic. Semble's checkout results were treated as navigation only, then checked against `git show c31825a:<path>`. Full-occurrence literal searches used `git grep` on that revision. All line references below mean the pinned **main** blob, not the older open checkout.

The audited tree contains 1,648 tracked paths: 118 under `src`, 84 under `scripts`, 24 under `configs`, 125 under `tests`, and 428 under `baseline`. All **536 Python files** under `src/`, `scripts/`, `tests/`, and `baseline/` parsed successfully with Python's AST parser. AST import enumeration was combined with explicit CLI, config, README, CI, freeze-binding, dynamic import, and history checks. An absent AST edge alone was never accepted as proof of obsolescence.

## 2. Exact safe deletion allowlist

These are file-level removals, not permission to delete their parent directories. Recheck the identities before applying them to a newer tree. All seven exact paths have **zero tracked content references** in the pinned main tree; searches for the unique scratch/log filenames also returned no references, and the scratch module has zero AST import edges. Generic `.gitkeep` ignore rules do exist for teacher/external placeholders and are discussed in the retain list. The current report naturally becomes a new documentary reference after publication.

| Exact path | Why obsolete and replacement | Dependency closure and evidence | Validation for application |
|---|---|---|---|
| `baseline/CUTS/src/datasets/tempCodeRunnerFile.py` | Unreferenced editor scratch copy of `ACDCOriginalStyle`; 6,313 bytes / 155 lines. Use retained `baseline/CUTS/src/datasets/acdc_original_style.py`, which adds explicit image-only handling, input validation, corrected OpenCV width/height sizing, and metadata. This is **not** a byte-identical duplicate. | `prepare_dataset.py:16,42-47` explicitly imports/constructs the retained module; `test_original_style_contract.py:53,132` explicitly loads that module. No scratch import, config selector, README/CI reference, or recorded freeze binding. The CUTS dataset directory does not dynamically discover Python modules; its globs enumerate data. Added in merge commit `fabe030` on 2026-10-01 (`git show -m --name-status` shows addition against both parents), with no dedicated historical execution contract found. | Confirm hash below and unchanged retained-loader imports; parse retained source; run `baseline/CUTS/tests/test_original_style_contract.py` in an appropriate test environment. Validate active v12 freeze after cleanup. Do not delete the retained loader, its configs, protocol document, tests, or experiment artifacts. |
| `baseline/.gitkeep` | Empty scaffold placeholder; 427 other tracked descendants already retain the directory. No replacement needed. | Added by `605e62b` to create baseline directories. Zero bytes, no consumer/binding; all actual baselines remain. | Assert zero-byte blob and a nonempty retained parent; exact-path diff review. |
| `baseline/CUTS/.gitkeep` | Empty scaffold placeholder; 222 other tracked descendants. No replacement needed. | Same scaffold commit, zero references/bindings. | Same zero-byte/retained-parent check. |
| `baseline/DFC/.gitkeep` | Empty scaffold placeholder; 50 other tracked descendants. No replacement needed. | Same scaffold commit. This removes no DFC implementation or historical evidence. | Same zero-byte/retained-parent check; retain `LEGACY_OUT_OF_SCOPE.md` and all comparison content. |
| `baseline/PICIE/.gitkeep` | Empty scaffold placeholder; 34 other tracked descendants. No replacement needed. | Same scaffold commit; no dependency or provenance content. | Same zero-byte/retained-parent check; preserve the PiCIE baseline. |
| `baseline/STEGO/.gitkeep` | Empty scaffold placeholder; 54 other tracked descendants. No replacement needed. | Same scaffold commit; no dependency or provenance content. | Same zero-byte/retained-parent check; preserve the STEGO baseline. |
| `baseline/STEGO/src/wget-log` | Empty download log, containing neither code nor observational evidence. No replacement needed. | Zero bytes; no consumer/binding. Last non-merge introduction is `d125179` (2026-09-20, recovered STEGO code). Removing this empty file does not remove the recovered source or its provenance. | Assert zero bytes and no new consumer, then exact-path diff review; no model test needed solely for an empty log. |

Expected deletion footprint: **7 files, 6,313 source bytes; 1 Python module and 6 empty files.** This cleanup removes ambiguity, not meaningful runtime cost. It provides no measured speed improvement or Dice improvement.

Pre-deletion SHA-256 identities:

```text
7210d539e3970dd5dcd6a7419ab5d95199ec482064af29b54de7bb5811831778  baseline/CUTS/src/datasets/tempCodeRunnerFile.py
e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855  baseline/.gitkeep
e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855  baseline/CUTS/.gitkeep
e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855  baseline/DFC/.gitkeep
e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855  baseline/PICIE/.gitkeep
e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855  baseline/STEGO/.gitkeep
e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855  baseline/STEGO/src/wget-log
```

The open `234be0a` checkout contains only the five `.gitkeep` candidates; the scratch loader and empty `wget-log` exist on current main. Applying a cleanup based only on the open checkout would miss the main-only files.

## 3. Retain list and concrete dependency closure

Every path outside the immediate allowlist is retained until its stated migration is complete. A historical document or compatibility test does not itself make an old training algorithm current: the distinction between actual runtime use, import-only compatibility, and historical reproduction is explicit in Section 6. The following table records the current dependency state rather than permanently vetoing retirement.

| Retained file/family | Reachability or preservation reason | Replacement/refactor requirement before any future removal |
|---|---|---|
| `scripts/train_self_audit_legacy.py` | Canonical `scripts/train_self_audit.py:26-36` imports eight compatibility/calibration names, but AST use analysis finds **zero internal calls/loads of those names** in the canonical runner. The real path is `trainer.train()` at line 228; `UnifiedTrainer` owns its own calibration method. `tests/test_checkpoint_binding.py:22,404,1098` and `test_calibration_lineage.py:939` exercise replay helpers; `test_unified_trainer.py:705` tests only the old CLI. README/legacy-guide references are historical. | **Conditional retirement recommended:** move the exact five-function helper closure into `src/self_audit/evaluation/checkpoint_calibration.py`, redirect compatibility exports and replay tests, retire the old CLI test/help, then delete the legacy runner. See Section 6. |
| `src/self_audit/training/train_annotation.py`, `train_auditor.py`, `finetune_joint.py` | `unified_trainer.py:99-119` imports live loss, validation, transition-cache, and rollout-counter functions from all three. `cache_validation_transitions.py:32`, Candidate C tests, and core regression tests also call them. Main's active milestone lists their CLI entrypoints explicitly. | Separate reusable functions from historical CLI wrappers, retain functions and API compatibility, then re-audit all importers. Whole-file deletion is currently unsafe. |
| `scripts/run_full_pipeline_legacy.sh`, `scripts/run_full_pipeline_legacy.ps1` | Standalone historical CLI interfaces, not invoked by the canonical wrappers. Their incoming references are documentation/help/error text and one error-message assertion. The old wrappers invoke the retained phase modules and use a distinct disconnected-process checkpoint-reload recipe. | **Conditional retirement recommended:** update current help/docs/tests, pin reproduction instructions to immutable `c31825a`, then delete both wrappers. No active training function needs to be extracted from these shell files. |
| `configs/self_audit_annotation.yaml`, `self_audit_auditor.yaml`, `self_audit_joint.yaml` | In addition to legacy runner defaults/tests/docs, **`shared_benchmark/spatial.py:125-162` loads and hashes all three together with `self_audit_full.yaml`** to establish the original 256-grid contract. Freeze generators explicitly bind them. | Any migration affects scientific hashes and grid provenance. Introduce a versioned protocol source, retain the original frozen configs, and validate old/new contracts; do not remove these as unused legacy YAML. |
| `src/self_audit/` (56 files) | Current README canonical model/trainer; reused by no-GT v3: `system_v3.py:162-165` imports/constructs `self_audit.models.annotation_expert.AnnotationExpert`; `annotation_expert.py:15` imports `DynamicWindowAttention` from `.dynamic_window`. Public package imports and all supporting modules must remain intact. | Supervised training is outside the user's requested learning protocol, but namespace-wide deletion breaks the no-GT student. A smaller deployment-only package would be a separate refactor with compatibility checks. |
| `src/self_audit_maskfree/` (34 files), maskfree scripts/configs/tests | Listed as Self-Audit core in the canonical environment contract. Shared benchmark depends on it: `firewall.py:7` imports `forbidden_reason`; `spatial.py:12-13` imports native geometry/resize helpers; `evaluate_cardiac_baseline_reference.py:26` imports volume/patient metrics; CUTS manifest tooling imports discovery. Its firewall file is included in adapter implementation hashing (`artifacts.py:631`). | Retain the full family for current work. Removing only the trainer would still need a separate caller/config/provenance review; deleting the package is unsafe. |
| `src/self_audit_pseudolabel/` (11 files); `train_pseudolabel_v3.py`, `train_student_v3.py`, `run_full_pipeline_v3.py`, `evaluate_pseudolabel_frozen.py`; `configs/pseudolabel_v3.json`; v3 tests | Main has a real workflow beyond this branch's two-file scaffold: teacher/student wrappers import `pipeline_v3`; it imports data, system, trainer, freeze, consistency; trainer imports evidence, evolution, losses. `__init__.py:3-15` uses lazy `import_module` exports, so AST-only package reachability misses public APIs. CI selects `tests/test_pseudolabel_*.py` and `tests/test_self_audit_*.py`. | Preserve the current no-GT research workflow and regression evidence. A poor/unmeasured quality result does not prove software obsolete. Do not replace it with this checkout's older scaffold. |
| `src/shared_benchmark/`, all `configs/adapter_v*_spec_source.json`, fixture JSON and evaluator config | `adapter.py:179-218` dispatches v4/v5 and retains the topology path used by v2/v3; `adapter_v5.py:15` itself depends on v4. `semantic_contract.py:20-33,249-260` binds different adapter versions to different grids. Tests cover permutation, topology, frozen fixtures, v4, and v5. | These are distinct immutable scientific contracts, not interchangeable aliases. Keep all specs and implementations required to inspect/replay historical outputs. |
| `scripts/validate_cardiac_benchmark*.py`, `scripts/regenerate*_freeze.py`, `scripts/historical_224_freeze_snapshot.py`, all `benchmark_freezes/` | Active v12 validator imports the **v7** validator (`validate_cardiac_benchmark_v12_historical_224_freeze.py:8,18-24`). Active v12 generator imports v7 (`regenerate_self_audit_cardiac_benchmark_v12_historical_224_freeze.py:13,22-30`). Historical v10/v11 validators use immutable Git-source snapshots through `historical_224_freeze_snapshot.py`. | Do not keep only the largest version number. Separate common validation/generation utilities in a future versioned change if desired, preserving all frozen evidence and snapshot verification. |
| `baseline/CUTS/` except its two allowlisted files | Active milestone comparison; retained original-style loader/config/protocol plus image-only scientific runner are distinct protocols. Diffusion/PHATE and analysis scripts are part of CUTS behavior and reproduction, not obsolete because the canonical Self-Audit model does not call them. | Keep both protocol identities explicit. Any algorithmic simplification needs a separately frozen comparison, not a deletion-only cleanup. |
| `baseline/DSS_US/`, `baseline/SGSCN/`, native infrastructure/configs/tests | Active scope includes both; `reports/ACTIVE_MILESTONE_SCOPE.md:37-47` documents blocked/native-reference/evaluator statuses. Protocol or data blockers do not imply dead code. | Retain source ledgers, locks, reference profiles, blocked profiles, and tests; protocol clarification is separate work. |
| `baseline/DFC/`, `baseline/STEGO/`, `baseline/PICIE/` except the exact empty allowlisted files; their runner/evaluation scripts/tests/configs | `reports/ACTIVE_MILESTONE_SCOPE.md:18-25` explicitly retains code/tests/freezes/artifacts for history; `LEGACY_OUT_OF_SCOPE.md:7-8` repeats this. Shared benchmark regression tests still exercise these historical contracts. | “Out of this milestone” is not evidence of safe baseline deletion. Keep their working reproduction boundary, including module-local imports and upstream provenance. |
| `baseline/CUTS/comparison/` | Vendored comparison implementations. Exact duplicates exist with recovered top-level baselines, but CUTS comparison scripts and upstream import roots are separate executable contexts. The user's retention rule also protects comparison baselines. | Deduplication requires an explicit import/path migration and reproduction verification. No subtree removal is proposed. |
| `scripts/train_rew.py`, `src/self_audit/training/rew_runner.py`, model/loss REW modules, `configs/self_audit_rew_16gb.yaml`, `tests/test_rew.py` | `docs.md:430`, `docs/rew_16gb.md:130-156`, and test CLI at `tests/test_rew.py:231` document/call this separate research baseline. | Retain as a supervised comparison/history path; it is not the newly requested no-GT experiment. No equivalent no-GT replacement has been demonstrated. |
| Other scripts, configs, tests, `environment.yaml`, `environments/`, dependency files, `README.md`, `docs.md`, `docs/` | Not proven obsolete. Current environment contract and CI affect active paths; scripts may be standalone CLIs rather than importable library nodes. | No removal based solely on lack of an incoming Python import. A future candidate needs its own exact closure and history evidence. |
| `reports/`, `wandb/`, `splits/`, all source snapshots, manifests, raw/processed data, weights/checkpoints, run configs and experiment outputs | Scientific evidence, reproducibility inputs, or user-owned artifacts. W&B metadata is explicitly used as environment-history evidence in `environments/self-audit-canonical/environment.json:18,63`. Ignored data/weights were not scanned or declared disposable. | Preserve. Git history is not a substitute for ignored local artifacts or large-file content. No `git clean`, wildcard removal, or evidence pruning is authorized by this report. |
| `checkpoints/teachers/{cinema,medsam2}/.gitkeep`, `external/{CineMA,MedSAM2}/.gitkeep`, `external/README.md`, `scratch/mock_split.json` | Teacher placeholders are one-byte tracked files explicitly accommodated by `.gitignore:108,117`; potentially populated local directories are not part of this source audit. `scratch/mock_split.json` is a nonempty 371-byte artifact without sufficient obsolescence proof. | These are **not** covered by the baseline empty-placeholder allowlist. Retain pending an artifact-specific decision. |

## 4. Current entrypoints and the v1/v2/v3 naming trap

| Family | Current entrypoint/selection | Cleanup interpretation |
|---|---|---|
| Canonical supervised Self-Audit | README:25-29, `scripts/train_self_audit.py --config configs/self_audit_full.yaml`; unified wrappers `.sh` / `.ps1`; M&Ms/joint profiles and Candidate C native wrapper | Keep as baseline/history and shared component provider. This is not a permitted training route for the user's fully no-GT objective. |
| Legacy phased supervised workflow | `train_self_audit_legacy.py`, legacy shell/PowerShell, three phase module CLIs and YAMLs | Distinct historical interface with live canonical dependencies; not a duplicate deletion target. |
| Maskfree | `scripts/train_maskfree.py`, `scripts/run_maskfree_full.sh`, `scripts/run_maskfree_acdc_mnms.sh`, `configs/maskfree_*` | Separate package/protocol and shared data/metric dependency; retain. |
| Cine pseudo-label v3 | `run_full_pipeline_v3.py:64-83` constructs teacher, frozen evaluation, optional student commands; `--train-student` controls the student stage | Retain and assess scientifically under the no-GT rules. Workflow completion itself makes no Dice claim. |
| Shared scientific CUTS | `scripts/run_cuts_scientific.py`; official-environment wrapper; historical-224 v12 freeze | Retain its exact preprocessing, raw-output, semantic-adapter, and provenance contracts. |
| Native comparison workflows | `baseline/DSS_US/scripts/run_native.py`, `baseline/SGSCN/scripts/run_native.py`, `scripts/native_protocol_readiness.py` | Keep readiness/blocking evidence and native protocols. |

There are **no tracked `system_v1.py`, `system_v2.py`, `pipeline_v1.py`, or `pipeline_v2.py` implementation counterparts** under current `src/self_audit_pseudolabel`, and the corresponding name searches in current src/scripts/tests/configs/README/CI returned no matches. V3 is not evidence that whole v1/v2 code directories remain to delete. Schema Version 1 in the unified supervised configuration, adapter v1/v2/v3, and benchmark freeze v1/v2/v3 are separate version axes.

Actual content duplication was checked by Git blob identity across live code/config/test files, excluding empty files and `__init__.py`. There are **13 duplicate-content groups**, all in baseline code/configs: DFC `demo_ref.py`; 11 STEGO files/configs shared between `baseline/CUTS/comparison/STEGO/` and `baseline/STEGO/`; and PiCIE/STEGO `cardiac_benchmark/output.py`. No duplicate-content group was found confined to `src/`, `scripts/`, `tests/`, or `configs/`. All 13 baseline groups are retained because equal bytes do not establish equivalent import roots or dispensable comparison history.

## 5. Git history and checkout differences that change the verdict

1. `6d2388d3565fee1f3b62beff45164d22288e660c` (2026-08-13, “lock repository to self-audit baseline”) already removed `src/models/s3r`, `src/distillation`, `src/teachers`, their obsolete training/config/test paths, and teacher-download/precompute scripts. README:6-8 describes that removal. Do not claim these still exist or manufacture a new deletion list from older reports.
2. `25f429a` (2026-09-09) introduced the unified training interface while preserving the legacy helper/reproduction path. The imports above show that this migration did not make the old files independently deletable.
3. `d4f503b` (2026-09-20) introduced the v3 scaffold; `103ce77` (2026-09-21) completed the workflow; `0605e3b` (2026-09-22) fixed supervision and split isolation. Main has 11 pseudo-label files versus only `__init__.py` and `system_v3.py` in this open checkout. An audit against the open checkout alone would miss `pipeline_v3`, freeze/evidence/training helpers, wrappers, and newer tests.
4. The checkout-only commits are `b028804` (Astra audit package) and `234be0a` (model-first flow/research). Its `tests/test_pseudolabel_v3_audit.py` is absent from current main, while main adds API-compatibility, complete-pipeline, integration, regression, and environment-contract tests. This is a tree difference, **not** a proposal to delete checkout-only audit evidence.
5. `fabe030` (2026-10-01) added the unused scratch loader alongside CUTS original-protocol work. Its retained replacement is explicitly selected by current code/tests.
6. `902d7a8` (2026-10-01) limits the active milestone to Self-Audit core, shared benchmark, CUTS, DSS-US, and SGSCN while explicitly retaining DFC/STEGO/PiCIE historical material. `c31825a` includes that policy plus the later native paper-protocol evidence update.

## 6. Meaningful retired-code migration option — recommended, not implemented

This option addresses removal of obsolete executable implementations rather than only empty-file hygiene. It has a bounded closure and removes **three files / 1,649 lines / 66,564 bytes** from the live scripts tree, while retaining roughly 399 lines of required replay/helper functions plus their imports/constants in a library module. This is a code-maintenance reduction; runtime speed and model accuracy remain unmeasured.

### Distinguish the dependency categories

| Candidate/dependency | Category | What that means for retirement |
|---|---|---|
| Canonical runner importing eight legacy names | **IMPORT_ONLY_COMPATIBILITY** | It currently prevents blind file deletion, but none of the eight names is called or loaded internally. Redirecting the export source closes this edge without routing canonical training through the historical algorithm. |
| `UnifiedTrainer` calling phase-module losses/validators | **ACTIVE_RUNTIME** | Keep these functions/modules. Retiring the legacy orchestrator does not make these modules dead. |
| Checkpoint/lineage tests calling five legacy helper functions | **HISTORICAL_REPLAY_API** | Move the tested behavior into a named library and preserve the scientific regression assertions. A test's old import path is not a reason to keep an obsolete executable. |
| Old CLI parser acceptance test | **RETIRED_INTERFACE_TEST** | Remove or replace the assertion that the retired CLI exists; retain and update canonical rejection/migration-guidance coverage. |
| Legacy shell references in README/docs/current-wrapper help | **HISTORY_OR_GUIDANCE_ONLY** | Point history to immutable source and guidance to a supported current command. These references need editing, not an indefinitely live old training wrapper. |
| Three phase YAMLs loaded/hashed by `shared_benchmark.spatial` | **ACTIVE_RUNTIME_AND_BOUND_PROVENANCE** | Keep at their existing paths. Their deletion is not part of this migration. |
| No `system_v1`/`system_v2` pseudo-teacher modules on main | **ABSENT_IMPLEMENTATION** | There is no v1/v2 teacher implementation to remove from this tree. Main's actual teacher calls the current v3 model; the older branch scaffold is a revision of that file, not an independently selected runtime version. |

### Exact extraction target and closure

Create **`src/self_audit/evaluation/checkpoint_calibration.py`** as a library for checkpoint-bound historical Phase-C replay/calibration, with an explicit module docstring stating that it is not the canonical trainer or a training CLI. Move these function bodies without altering their scientific semantics:

| Function in `scripts/train_self_audit_legacy.py` | Original lines | Local dependency |
|---|---|---|
| `_resolve_tau_accept` | 233-284 | Calls `_verify_calibration_for_phase_c` |
| `_verify_calibration_for_phase_c` | 287-340 | Uses existing checkpoint-binding and lineage primitives |
| `bind_post_training_checkpoint` | 343-370 | Uses `bind_evaluation_checkpoint` with historical phase checkpoint names |
| `_diagnostic_at_tau` | 373-426 | Uses retained `validate_phase_c` and `evaluate_audit_decomposition` |
| `run_post_training_calibration` | 466-676 | Calls `bind_post_training_checkpoint` and `_diagnostic_at_tau` |

Also move `SELECTION_BIAS_CAVEAT` and `TAU_PRECEDENCE`, and retain imports/re-exports for `collect_validation_transition_cache` and `sweep_thresholds`. The precise external function closure is:

- `self_audit.serialization.atomic_save_torch`;
- `self_audit.evaluation.audit_decomposition.evaluate_audit_decomposition`;
- `self_audit.audit.semantics.METRIC_SPACE_SLICE_PROXY`;
- `self_audit.evaluation.calibration_lineage.{COHORT_ROLE_CALIBRATION,CohortPolicy,build_expected_lineage,verify_calibration_lineage}`;
- `self_audit.evaluation.threshold.{load_calibration,save_calibration,select_threshold,sweep_thresholds}`;
- `self_audit.provenance.{CheckpointBinding,build_lineage,cohort_identity}`;
- `self_audit.training._utils.{bind_evaluation_checkpoint,bind_existing_evaluation_state,verify_bound_state}`;
- `self_audit.training.finetune_joint.{collect_validation_transition_cache,validate_phase_c}`;
- the existing `argparse`, `Path`, `Any`, NumPy, and torch imports/types.

Do **not** move `main`, `_parse_args`, `_apply_overrides`, `_architecture_signature`, `_assert_compatible_configs`, `_set_trainable`, `_make_loaders`, `_scheduler_and_amp`, `_log`, `_write_json`, or their training-only imports. Those belong to the retired orchestrator. The five extracted function bodies total 399 lines; copying the entire old script under another filename would fail the purpose of this migration.

**Preserve the calibration-policy boundary:** historical `bind_post_training_checkpoint:352-366` permits the explicit `phase_c_last.pt` fallback; current `UnifiedTrainer.run_post_training_calibration:3074-3095` requires a completed run and a selected `best.pt`, and forbids fallback. Keep the unified method unchanged and do not replace it with the historical replay function simply because their names overlap. Historical replay helpers may continue to interpret historical checkpoints, but they must not weaken the current runtime selection gate or be used to tune the requested no-GT experiment on evaluation GT.

### Consumer and documentation migration

1. In `scripts/train_self_audit.py:26-36`, re-export the same eight names from the new library (or retain direct imports for the two already-library functions). Preserve canonical `main -> UnifiedTrainer -> train` behavior. Update its module description and legacy-flag error at line 74 so neither directs users to a deleted script.
2. In `tests/test_checkpoint_binding.py:22,404,1098`, import from the new library. At lines 419-437, update the module object used by `monkeypatch`: the spies must patch the new library's globals, where the extracted function looks them up. Merely changing a function import while patching the old re-export module would silently invalidate the test.
3. Change `tests/test_calibration_lineage.py:939` to the new library, keeping all best-versus-last, bound-state, round-trip, and lineage assertions. These tests protect historical artifact interpretation and should not be deleted with the runner.
4. Retire `test_legacy_runner_accepts_legacy_flags` at `tests/test_unified_trainer.py:703-720`, because its only contract is the removed parser. Preserve the canonical negative-flag tests at 692-700 and 1288 onward; update the old-wrapper-name assertion at line 1299 to check current actionable guidance. Existing bounded canonical smoke tests continue to exercise the supported entrypoint.
5. Update help/error/comments in `scripts/run_full_pipeline.sh:15,74,104` and `scripts/run_full_pipeline.ps1:7,52`; these are references, not runtime calls to the old wrappers. Their execution targets remain `scripts/train_self_audit.py`.
6. Update `README.md:60,178-180`, `docs.md:231`, and `docs/legacy_reproduction.md`. Explain that new canonical supervised reproduction uses the unified command, while the historical live-weight and disconnected-process recipes can be inspected/replayed from immutable **`c31825a7f90df47f9f9382a9c9595e43f7f71236`**. Keep the historical recipe description and checkpoint semantics; remove commands that falsely imply deleted entrypoints exist in the current tree. Historical research reports and frozen evidence should not be rewritten.
7. Recheck all imports/callers, CLI help, CI collection, and frozen bindings. Only after these checks remove the three exact files below. Preserve original history; do not rewrite Git commits or regenerate old freezes to conceal a source change.

### Conditional deletion allowlist after the migration

| Exact path | Why the executable is obsolete | Replacement and closed dependency |
|---|---|---|
| `scripts/train_self_audit_legacy.py` | Superseded three-config training CLI; canonical training does not execute its algorithm. 1,085 lines / 41,035 bytes. | Unified trainer owns new canonical training; five-function replay closure lives in `checkpoint_calibration.py`; canonical exports/tests no longer import this script. |
| `scripts/run_full_pipeline_legacy.sh` | Superseded standalone three-process orchestrator; no canonical subprocess caller found. 351 lines / 14,658 bytes. | Current `run_full_pipeline.sh` / canonical Python runner for new runs; immutable revision for the old recipe; help/docs/test assertion migrated. |
| `scripts/run_full_pipeline_legacy.ps1` | PowerShell equivalent of the superseded three-process orchestrator; no canonical subprocess caller found. 213 lines / 10,871 bytes. | Current `run_full_pipeline.ps1` / canonical Python runner; immutable revision for historical replay; help/docs migrated. |

History supports this boundary: both legacy shell wrappers were last changed by `25f429a` (2026-09-09, unified-pipeline migration), while the legacy Python runner was last changed by `a6b3a75` (2026-09-11, checkpoint/report/resume hardening). Its later helper hardening is another reason to preserve the replay functions rather than discard their behavior with the training CLI.

These three files are absent from the active v12 freeze's 26 repository bindings. Older freeze contracts can still bind source paths affected by the migration (for example the canonical runner), so preserve those historical freezes and their immutable source rather than trying to make every historical freeze certify today's checkout.

**Still excluded:** the three phase YAMLs, phase module files, shared model files, baseline implementations, and all frozen evidence. The YAMLs are actually opened and hashed by active `load_self_audit_grid_spec`; migrating this contract would require a separate versioned protocol change and offers no necessary benefit to retiring the old executable runner. The phase modules provide live unified-training functions. No deletion of these files is needed to complete the three-executable retirement.

### Validation required for the substantive migration

Run the existing checkpoint-binding and calibration-lineage suites against the new library, the unified-trainer/console-progress/CLI and bounded runtime smoke suites against the canonical path, the shared benchmark suite, and active v12 validation. Keep the existing missing-best rejection and historical fallback tests both passing under their respective APIs. Verify that no live source/test import names `scripts.train_self_audit_legacy`, and that current help contains no executable command pointing at a retired wrapper; historical prose may name the old paths when explicitly pinned to the historical revision.

This is a substantive migration, so static parsing alone is insufficient. The coordinator owns implementation and behavioral verification; this worker has not performed them. A local CPU pass is software evidence only. User hardware target is **RTX 4080 Super 16GB**; no cleanup result proves CUDA speed or Dice on that device.

## 7. Validation receipt and application gate

### Completed in this report-only phase

- Resolved both refs and compared their trees/commit divergence; read all decisive sources from the immutable main SHA.
- Parsed **536/536** tracked Python files in the audited code/test roots without importing/executing model code; no syntax failures.
- Enumerated AST imports, then checked explicit script subprocess commands, lazy exports, filename-loading tests, YAML selectors, README/reproduction documentation, and CI selection.
- Searched all tracked content on main for the seven exact candidate paths and the unique scratch/log filenames: **zero** references; confirmed the retained ACDC loader's positive callers.
- Checked all tracked `benchmark_freezes/**/FREEZE_MANIFEST.json` repository-binding lists: **zero candidate bindings**. Active v12 contains **26** bound repository files, none in the allowlist.
- Read CUTS scientific `code_identity` file selection (`scripts/run_cuts_scientific.py:178-185`) and adapter implementation selection (`src/shared_benchmark/artifacts.py:619-637`); neither includes a candidate.
- Verified candidate byte counts, SHA-256 identities, scaffold/history origin, and retained directory contents.
- Compared duplicate Git blobs; retained comparison implementation copies rather than treating equal bytes as a complete dependency analysis.

These are source/static checks. They do not certify a canonical environment, runtime import success, CUDA execution, model quality, or a Dice threshold.

### Required when the coordinator applies cleanup

Apply only the seven paths on the current-main integration checkout, after ensuring each still matches this report. Retain the exact replacement loader and all listed contracts. For empty files, byte/consumer/parent checks and an exact diff are sufficient; do not add tests that merely assert an empty file was deleted.

For the scratch-module removal, the focused runtime regression is the existing original-style CUTS test; the scientific provenance check is the existing active v12 validator. Suggested commands from the integration checkout, using its selected test environment:

```sh
rtk proxy env PYTHONPATH=src python -m pytest -q baseline/CUTS/tests/test_original_style_contract.py
rtk proxy python scripts/validate_cardiac_benchmark_v12_historical_224_freeze.py
rtk git diff --check
rtk git diff --name-status
```

These commands are **NOT RUN by this worker**. If a relevant check already fails on unmodified current main, record that baseline failure and compare after cleanup; do not rewrite frozen evidence or silently exempt a candidate to make the gate pass. A new commit naturally changes repository identity; existing experiment receipts remain bound to their original identity and must not be relabeled.

The additional three-executable retirement follows Section 6 and requires its behavioral gate; it is not covered by the immediate seven-file static-only allowance. Broader extraction of phase/model functions would additionally require the transition-cache, Candidate C, pseudo-label, and relevant environment-contract tests and is not proposed here.

The coordinator owns deletion, combined-tree verification, commit, and publication. This report neither claims deletion completed nor infers fast/lightweight Dice >= 0.90 from code cleanup.
