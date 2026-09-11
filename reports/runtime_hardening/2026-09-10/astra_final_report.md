# Astra runtime-hardening report

**Astra gate: PASS for software correctness and publication. CUDA/RTX 4070 and production safety remain unverified.**

Audited base and fetched origin/main: `25f429a46255ceb59997ae5178939d5d40b2ea27`. Audit began 2026-09-10 UTC; current local date is 2026-09-11. Astra planned and reviewed; AGY implemented serialization, artifact IO, lifecycle and synthetic smokes through Orca. At the user's request, additional AGY and Claude workers ran disjoint work in parallel. After explicit AGY quota failures, Claude took over central resume/selection integration and independent resume regressions. Root inspected actual changes and independently reproduced the original failures before implementation.

The complete independent suite produced 897 passed and one stale test-message assertion failure. The runtime correctly refused the non-resumable checkpoint; Claude updated only that regex, and root reran the test: 1 passed. Thus all 898 distinct tests are verified, with zero remaining failures and no production/source/config changes after the complete suite. A second full-suite execution was intentionally avoided under the user's request to reduce repeated tests. Earlier focused wave results are in reviewer_gates.md; they are not added together to inflate the integrated count.

Implementation commit: `a6b3a752d2a1732203edc053f0b1889a0f089ebc`. Tested source content signature: `7681274e4c0b18291419a06f7abd31aafb73b10e8109d273ee9f5e6bb6a7c94a`. Final staging also removed trailing blank lines in two test files; no executable code changed.

## Scope and experiment identities

The canonical trainer remains one resolved config, one trainer, one global epoch timeline, canonical best.pt / last.pt, and one intended W&B identity across online resumed attempts. Offline W&B attempts explicitly cannot establish backend resume. Historical A/B/C entrypoints remain for backward reproducibility only. This work hardens runtime and provenance; no architecture, loss, A0, pretrained ConvNeXt baseline, audit stop-gradient, reject-to-HALT, scientific split, Proposal 2/3 or Dice tuning changes are included.

ACDC train/val, ACDC-to-frozen-external-M&Ms, and native M&Ms supervised training remain three separate experiments. Native M&Ms requires prepared disjoint train/val data. External M&Ms uses a frozen selected ACDC checkpoint and a threshold fixed before external evaluation; its labels do not enter training, checkpoint selection or calibration. Real factory dispatch, patient identity, four-class BG/RV/MYO/LV mapping `{0:0, 1:3, 2:2, 3:1}`, unknown-label rejection and depth-axis handling remain covered by dataset/protocol tests and the clean subprocess smoke. There are no dataset implementation changes in this patch.

## Exact canonical schedule

Epoch ranges below are zero-based and end-exclusive; human log epochs are one-based.

| Global epochs | Trainable modules | Base LR encoder / annotation / auditor | Annotation / audit weights | Rollout | Batch / augmentation |
|---|---|---|---|---|---|
| [0, 100) | Annotation including encoder | 3e-5 / 3e-4 / 0 | 1 / 0 | propagate_no_audit | 4 / enabled |
| [100, 120) | Auditor | 0 / 0 / 3e-4 | 0 / 1 | annotation_eval; adjacent_and_synthetic | 4 / disabled |
| [120, 130) | All | 1e-6 / 1e-5 / 1e-5 | 1 / 1 | threshold_gate; active_attempted | 2 / enabled |

Each interval has its own intended optimizer/scheduler reset at entry; an ordinary resume within an interval restores them. With U optimizer updates per epoch and N interval epochs, total steps T=U*N, warmup W=min(ceil(5*U), T). LR is base LR times max(s/W, 1e-8) during warmup, then 0.5*(1+cos(pi*clamp((s-W)/max(T-W,1),0,1))). Minimum ratio is 0. Accumulation is 1. Selection begins at zero-based epoch 120 and uses final_foreground_macro_dice, maximizing only defined finite observations. Calibration uses the selected ACDC checkpoint and ACDC validation data.

The only canonical YAML change is persistent_workers true → false in both ACDC and native M&Ms. This makes worker seeds reconstructible at epoch boundaries. It changes worker lifetime and throughput and may change the old run's random trajectory; it does not change augmentation probabilities or the scientific recipe. Old checkpoints are not silently certified under a changed source/config/RNG contract.

## Discovered defects and reproduction inventory

Base line references below name the audited commit, not moving worktree line numbers. Additional wave-local line references are explicitly labeled. Full pre-edit evidence is in audit_plan.md; detailed worker/source review is in parallel_claude_review.md and reviewer_gates.md.

| ID | Location and reproduction | Root cause / implemented direction | Regression coverage |
|---|---|---|---|
| R1 | Base training/_utils.py:1047; actual torch 2.4.1 first save raises uint32 KeyError | NumPy MT19937 keys became unsupported torch.uint32; store lossless int64 and explicitly restore NumPy uint32 | test_runtime_checkpoint.py; test_runtime_amp_resume_review.py |
| R2 | Base training/_utils.py:1164; Path metadata saves but weights_only=True reload fails | No recursive portable payload contract; normalize supported CPU tensors and builtins, reject unsupported custom objects/dtypes with paths | test_runtime_checkpoint.py |
| R3 | Base training/_utils.py:1187; mocked fsync ENOSPC replaced an old valid file | Durability exception swallowed; propagate file fsync/save/replace errors and clean temporary files without masking the primary error | test_runtime_checkpoint.py |
| R4 | Base training/unified_trainer.py:106; nested NaN array/Inf tensor/NumPy scalar/Path strict-JSON probes fail | Incomplete recursion and permissive JSON; shared strict converter, null for undefined/nonfinite metrics, never zero | test_artifact_io.py |
| R5 | Base training/unified_trainer.py:1491; injected first checkpoint failure leaves log event, no report and no finish | Missing lifecycle/commit ordering; atomic partial/failure artifacts, preserve completed validation, rethrow, nonzero CLI exit, failure finalization | test_trainer_lifecycle.py; commit integration |
| R6 | Base threshold.py:1421, transition_bank.py:1573, scripts/audit_checkpoint.py:1305, scripts/evaluate_external_mnms.py:328 | Direct final-file writes could truncate valid reports; shared atomic JSON writer | test_artifact_io.py; wave2_independent_probe.py; test_runtime_pipeline_smoke.py |
| R7 | Base unified_trainer.py:1739, scripts/cache_validation_transitions.py:114, scripts/train_self_audit_legacy.py:617 | Raw cache torch.save bypassed portability/atomic checks; shared safe atomic tensor writer | checkpoint/artifact tests; full CLI smoke |
| R8/R12 | Base unified_trainer.py:1271/1427; scripts/calibrate_threshold.py:61 | Safe-load failure fell back to unrestricted pickle, or loader explicitly requested it; weights_only=True with no unsafe fallback | checkpoint/binding/standalone calibration coverage |
| R9 | Base unified_trainer.py:1532; four-epoch injected final last-save failure gives BEST_LAST_FAILURE 3 False | Public best overwritten before last commit; immutable selected snapshot → committed last reference → public best alias; resume verifies and repairs | test_checkpoint_commit.py; test_canonical_best_alias.py; commit integration |
| R10 | Base training/_utils.py:1561; nonfinite values accepted by a later broad numeric branch | W&B converter leaked NaN/Inf and dropped nested data; shared strict conversion and truthful identity/finalization status | test_artifact_io.py; test_runtime_wandb_identity.py |
| R11 | Base unified_trainer.py:1344, canonical YAMLs; default persistent augmenting workers rejected by resume | Worker RNG not checkpointable; explicit canonical nonpersistent lifecycle and real multi-worker continuation coverage | test_runtime_resume.py; config/provenance guards |
| R13 | Base unified_trainer.py:1471; mocked CUDA missing-state restore silently no-ops by default | Exact GPU requirement bypassed; pass exact_cuda for CUDA resume and validate device-state count/type | test_runtime_amp_resume_review.py |
| R14 | Base unified_trainer.py:1377; real enabled CPU GradScaler rejects saved empty dictionary | Presence-only check confused disabled scaler with recoverable enabled state; reject incompatible effective scaler modes clearly | test_runtime_amp_resume_review.py; commit integration |
| R15 | Base unified_trainer.py:361/1524; real resume 1..5 leaves only suffix epoch rows (31 pass, 5 history failures in preliminary test) | Report history initialized empty on resume; retain and validate checkpoint-backed committed history | test_runtime_resume.py |
| R16 | Wave-local shared loader; nonfinite scheduler/scaler numeric fields accepted on load | Validation covered only model/optimizer tensors; validate all four training-state trees before acceptance | test_runtime_portability.py |
| R17 | Base unified_trainer.py:374; logger constructed before resume recovers run identity | Orphan new run and SDK random draws can affect continuation; delay init, persist truthful identity, preserve RNG around telemetry | test_runtime_wandb_identity.py; commit integration |
| R18 | Evaluation bind call path forwards CUDA map_location for the entire checkpoint (instrumented CPU spy) | Unused optimizer/RNG state materialized on GPU; stage full payload on CPU while preserving model device | test_runtime_portability.py; GPU peak effect not measured |

The audit also rejected intermediate worker defects before approval: W3 overwrote an existing failure artifact on resume refusal, W3 durable epoch count lagged at post-commit logging, helper snapshot/relocation paths could traverse symlinked directories, unreadable last.pt was initially treated as historical and skipped, and W&B resume was initially inferred from a matching requested ID rather than the SDK's explicit resumed flag. Corresponding failure-ownership, disk-reading logger, path-containment, unreadable-sibling and SDK identity regressions were added. These intermediate findings are not represented as defects in the original base commit. A final independent real-path probe also found that checkpoint epoch_history saved the current row before its commit/selection fields advanced, and resume read last_completed_validation from a nonexistent nested extra mapping. Those two findings caused another REVISE; the corrected commit/history path and validation restoration passed in the final independent suite. A stale GPU-only error-message assertion was corrected; mocked device tests now exercise that branch without claiming real GPU execution.

## Final implementation locations and invariants

| Boundary | Current implementation |
|---|---|
| Safe tensor payload and atomic file replacement | src/self_audit/serialization.py:226,302 |
| Strict JSON and atomic report publication | src/self_audit/artifact_io.py:44,116 |
| RNG capture/restore and finite training-state validation | src/self_audit/training/_utils.py:1053,1100,1123 |
| Checkpoint save / safe load | src/self_audit/training/_utils.py:1281,1395 |
| Canonical best alias verification and CPU-staged evaluation binding | src/self_audit/training/_utils.py:1524,1620,1669 |
| W&B identity and cleanup wrapper | src/self_audit/training/_utils.py:1756 |
| History, failure artifacts, strict resume, commit loop and calibration | src/self_audit/training/unified_trainer.py:495,655,1719,2299,2735 |
| Immutable selection reference / alias / relocation | src/self_audit/training/checkpoint_commit.py:392,444,471,514 |
| CLI initialization failure handling / frozen external report | scripts/train_self_audit.py:187; scripts/evaluate_external_mnms.py:200 |

- A failed new per-file serialization, file fsync or replace leaves the old target intact; cleanup does not replace the primary exception. Tensor payloads are CPU-staged and safely loadable without unrestricted pickle fallbacks.
- NumPy MT19937 keys roundtrip losslessly through int64 and explicit uint32 restoration, including cached Gaussian state. CUDA exact-resume state/count is required when resuming on CUDA. Effective scaler-mode mismatches fail clearly.
- A unique immutable selected snapshot is written before last.pt commits its reference. The public best alias is published afterwards. These are recoverable commits, not a claim of a multi-file atomic transaction: a crash can leave an alias stale; evaluation refuses it and verified resume repairs it.
- Committed selection metadata, observed validation Dice, snapshot epoch/model digest and reference hash must agree. Calibration binds that selected model. Undefined selection metrics cannot improve best or become numeric zero.
- Exact resume requires affirmative full-validation/resumability flags, matching source/config/cohort, a run identity, valid model digest, schedule-compatible counters and committed epoch history. The current row is proposed before last is saved and adopted only on successful checkpoint commit. Previously completed validation is restored from the actual top-level payload.
- Checkpoint commit and required JSON report commit are recorded separately. History's report_committed field is an observation at checkpoint capture time, so the current checkpoint row cannot assert that a later report write happened. Required report publication precedes completed-epoch W&B logging; required final research report publication precedes success finalization. Post-finish telemetry refresh is best effort and carries its own status.
- Exceptions propagate to a nonzero process exit. Failure artifacts preserve the stage, counters, last completed validation, provenance/config identity and cleanup state where the filesystem permits; previous failure artifacts are retained under distinct attempt paths.
- External M&Ms reports use the same standards-compliant atomic writer and frozen checkpoint binding. External labels cannot alter training, checkpoint selection or ACDC calibration.

## Runtime compatibility boundary

The observed failure was reproduced with actual Python 3.10.21 and PyTorch 2.4.1 on CPU: NumPy MT19937 keys became torch.uint32 and torch.save raised KeyError. A newer local PyTorch 2.13 installation did not represent the declared server's serialization support. The compatibility test environment is Python 3.10.21, torch 2.4.1, torchvision 0.19.1 and NumPy 1.26.4.

The final suite ran there, including weights_only=True loads and a real enabled CPU GradScaler roundtrip. Mocked CUDA tests establish device/state handling logic only. CUDA 12.1, RTX 4070 12 GB, GPU bf16 kernels, CUDA optimizer placement and memory peaks were not exercised locally. Official version/API references: [PyTorch previous versions](https://pytorch.org/get-started/previous-versions/), [2.4.1 storage implementation](https://raw.githubusercontent.com/pytorch/pytorch/v2.4.1/torch/storage.py), [2.4.1 GradScaler implementation](https://raw.githubusercontent.com/pytorch/pytorch/v2.4.1/torch/amp/grad_scaler.py). W&B backend behavior was mocked; [SDK resume semantics](https://docs.wandb.ai/models/runs/resuming) do not establish that an actual backend run resumed.

## Exact final test results and decision

| Root-run gate | Exact result |
|---|---|
| Environment preflight | 2 collection errors, no tests executed: missing scikit-image. Installed scikit-image 0.24.0 in the temporary environment, retaining torch 2.4.1 and NumPy 1.26.4. |
| Complete test suite | **897 passed, 1 failed, 0 skipped, 1 warning; 791.29 seconds** |
| Sole failure | test_runtime_resume.py:810 expected the old non-resumable/incomplete wording; actual ValueError correctly said resumable is False, expected True. |
| Exact failing test after one-line assertion correction | **1 passed; 1.70 seconds** |
| Distinct final coverage | **898 tests verified; 0 remaining failures. No production code changed after the full run.** |
| Clean subprocess ACDC training → calibration/diagnostics → bank → frozen external M&Ms | **PASS**, 13.755 seconds in full suite |
| External-label mutation leaves frozen model unchanged | **PASS**, 5.374 seconds in full suite |
| Separate native M&Ms supervised subprocess smoke | **PASS**, 2.635 seconds in full suite |
| Git diff whitespace check | **PASS** |

The one warning comes from test_unified_logging.py::test_canonical_learning_rate_keys deliberately advancing a scheduler in its key-inspection fixture; it is not evidence that the active training loop reverses optimizer/scheduler order. No skipped test is represented as GPU coverage.

Commands used by root:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:. OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /private/tmp/self-audit-torch241/bin/python -m pytest -q -p no:cacheprovider --junitxml=/private/tmp/self-audit-final-241.xml
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:. OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /private/tmp/self-audit-torch241/bin/python -m pytest -q -p no:cacheprovider tests/test_runtime_resume.py::test_max_steps_bounded_checkpoint_is_refused_for_resume --junitxml=/private/tmp/self-audit-targeted-resume-241.xml
```

Machine-readable evidence: test_results.json, full_suite_241.xml and targeted_resume_241.xml. Committed artifacts have trailing whitespace normalized; test outcomes and failure text are retained. Full-suite results include the exact-resume baseline and all five resume positions, both curriculum boundaries, enabled-scaler/RNG roundtrips, atomic/failure injection, mocked W&B and the three clean subprocess smokes. Orchestration attempt outcomes and quota/transport replacements are recorded separately in orchestration_outcomes.json; provider failures are not application failures.

**Unified trainer status:** complete for the canonical one-config/one-trainer/global-timeline interface and tested bounded end-to-end flow. **Change classification:** canonical CLI preserved; checkpoint/report schemas and resume acceptance are strengthened. Architecture, objectives and scientific recipe are preserved. Nonpersistent worker lifecycle changes throughput and can change the old stochastic trajectory; exact source/config compatibility is enforced rather than pretending old checkpoints are interchangeable. **Remaining software blockers:** none found by this audit. The real-GPU and real-data validation limits below remain.

## Commands

```bash
python scripts/train_self_audit.py --config configs/self_audit_full.yaml --device cuda
python scripts/train_self_audit.py --config configs/self_audit_full.yaml --device cuda --resume weights/self_audit_full/last.pt
python scripts/evaluate_external_mnms.py --config configs/self_audit_acdc_to_mnms.yaml --checkpoint weights/self_audit_full/best.pt --data-root preprocessed_data/mnm --split testing --tau-accept 0.0 --device cuda --output reports/external_mnms.json
python scripts/train_self_audit.py --config configs/self_audit_full_mnms.yaml --device cuda
```

The external command above fixes the existing protocol threshold at 0.0. To use ACDC-calibrated tau instead, verify the ACDC calibration artifact's checkpoint binding and substitute its tau_accept before inspecting external results. The external CLI accepts a frozen scalar and cannot calibrate on M&Ms. A short GPU first-save canary and data preparation requirements are in runtime_runbook.md. Standalone transition-bank export remains a separate explicit frozen diagnostic command.

## Remaining real-run risks

CPU unit/fault/synthetic integration tests do not establish production safety or scientific performance. A real server run is still needed for CUDA bf16 and kernel portability, 12 GB peak memory, actual data loading throughput, real-data calibration/diagnostics, online W&B delivery/resume and target-filesystem behavior. Mocked ENOSPC/fsync/replace failures do not simulate physical power loss. Parent-directory fsync is best effort where the OS/filesystem does not support it. Referenced immutable selected-best snapshots must be retained; automatic snapshot garbage collection is not implemented. Exact CPU continuation does not promise bitwise deterministic CUDA kernels with the canonical deterministic flag disabled.
