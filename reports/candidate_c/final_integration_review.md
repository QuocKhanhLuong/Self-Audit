# Candidate C final integration review — code PASS; GPU gates BLOCKED

Root is the sole Astra XHigh planner/final reviewer. The reviewed implementation and CPU integration gates PASS; required real GPU gates remain BLOCKED. The user subsequently authorized commit/push and will run the GPU checks personally. This changes publication authorization only; the GPU gates are still unverified and no scientific experiment-readiness claim is made. No claim of novelty, superiority, clinical safety or Dice improvement is supported by this implementation task.

## 1. Exact source revision

Base HEAD and refreshed origin/main: `9a612485b7ead1273082d7dd52879f8e9e7a3e83`. Implementation was committed as `d905637d7f9f32b56de02a03a80a157a82f1de0f` after the user explicitly authorized publication pending their own GPU runs. The combined source signature v2 is `394dd2438094b6eeab105d227d104f5c27c94771f3eb2a6ad43694ebe5552eed`; [reviewed_source_manifest.json](reviewed_source_manifest.json) lists the 23 changed production/config/test files and SHA256 hashes. Unrelated prior reports and artifacts are preserved.

## 2–3. Workers and responsibilities

See [orchestration_log.md](orchestration_log.md) for live dispatch lineage and [architecture_contract.md](architecture_contract.md) for exclusive file ownership. All implementation used Claude/AGY through Orca; root is the only planner. Initial extra planner was user-interrupted/fenced. Claude session quota was confirmed exhausted on the final red-team launch, so that stopped attempt was retried on AGY per user direction. AGY handled M&Ms through a custom Orca-managed terminal; its provider-default model is unverified. Core and strict FIX follow-up requested/effectively launched claude-opus-5. Other Claude workers used provider defaults.

| Worker stream | Implementation owner | Root diff decision | Evidence |
|---|---|---|---|
| C1/C2/C3 core | Claude Opus 5 | PASS after revisions | Explicit C1 numeric-only failure test, bounded true backtracking, strict tied-FIX protection; opus_core_report.md |
| M&Ms/native/external | AGY through Orca | PASS after revisions | Removed path heuristics and verification bypass; consistent frozen mode/settings declarations; native/external fixtures |
| Training exposure | Claude | PASS after revisions | Opt-in isolated curriculum; correct current-path and replay-failure counters; record invalidation |
| Diagnostics | Claude | PASS after revisions | Rejected/fallback historical metrics, actual invocation counts vs sample sums, unknowns preserved |
| Config/baselines | Claude | PASS after integration alignment | Six modes, schema/runtime limits, original baseline preserved |
| Runtime/provenance | Claude | PASS after revisions | Signed nondefault execution mechanism; ephemeral replay; fail-closed resume/calibration |
| Red team | AGY (Claude quota fallback) | PASS | Independent core/training/diagnostics review; M&Ms is self-review, separately reviewed by root |

Worker PASS is the root's actual-diff decision, supported by completed integration tests and red-team review. It does not waive required real-GPU gates. See [red_team.md](red_team.md) for scoped independence and findings.

## 4. Files changed

The exact paths and hashes are in the manifest. Production scope consists of core dynamic window/annotation/self-audit model and exports, unified config/model builder/trainer/joint trainer, centralized provenance, M&Ms dataset/config/external evaluator, audit_checkpoint diagnostics and new evaluation/candidate_c.py. Tests and reports are separate. Root integration only aligned min_regress_mass schema/runtime bounds and its rejection test, added num_fix_ties to aggregate counts, removed an unused import/duplicate comment, and corrected stale report bounds/CLI examples. After red-team completion, root removed the unused factual_delta_logits replay field/allocation and its sole synthetic fixture argument. No root algorithm implementation.

## 5. Implemented C1/C2/C3

C1 replays only the most recently accepted ordinary correction from recorded features, pretransition logits, previous audit input, raw and resolved turn/depth indices and all realized factual supports. Replay uses the original factual batch including inactive rows for numerical identity; only accepted active rows receive correction. Frozen functional parameters/buffers and detached inputs leave only coordinates differentiable. Numeric or record-state identity failure returns to current-state ordinary annotation with an explicit reason, followed by the same official Auditor.

C2 uses thresholded predicted REGRESS evidence to minimize weighted CE against the pretransition predicted class plus lambda mean squared support displacement in feature-pixel units. One gradient is evaluated at factual P. The proposal is normalized in pixel coordinates, projected into the domain and rho-pixel axis-wise trust box, then checked at most twice: full projected proposal and its midpoint toward P. For predicted FIX pixels, counterfactual argmax must equal the factual class and its class margin must retain margin_fraction times the factual margin; the final transferred argmax is checked again. All thresholded FIX pixels are protected, including exact ties and arbitrarily small margins. No pixel is excluded by the diagnostic tie threshold. No REGRESS, no finite gradient, infeasibility or non-improvement selects factual support exactly.

C3: `A_next_candidate = A_current + sigmoid(recorded_gate) * (B_counterfactual - B_factual)`. The gate is spatial and shared across classes; innovation and gate are detached during training. Identity fallback uses an explicit zero difference. The final Auditor sees detached inputs and retains strict deltaQ>tau; reject keeps state and HALTs. Accepted C/rollback clears ordinary history, so the following active turn is ordinary. No recursive or unbounded replay history.

## 6. Replay invariant

`U_frozen(r; P)` must match the stored factual candidate under atol=rtol=1e-5, and stored factual logits must still match retained current state. Parameter identity/version, dtype/device, mode and generation guard stale records. The independent archived-HEAD baseline comparison passed bit-for-bit on 325 tensors (274 state_dict entries and 51 output tensors), using the same seed, B=2, 32x32 CPU input and untrained ConvNeXt. See [baseline_parity.json](baseline_parity.json). Ambient autocast remains the same within one infer invocation; independent CUDA identity is NOT verified.

## 7. GT firewall

Runtime C takes predicted Auditor evidence only. The image query/replay receives no target. Existing oracle_accept is explicitly analysis-only; its GT acceptance rule never supplies inner-solver evidence. Training labels enter supervised losses outside inference. Diagnostics evaluate already-computed historical pre/factual/next tensors under separate PROPOSAL and RETAINED families, including rejected/fallback attempts; no GT metric is fed back into the model.

## 8. Gradient flow

Coordinate optimization uses autograd.grad, not backward or a parameter optimizer. Functional detached parameter/buffer mappings and cloned detached replay inputs isolate the solver from encoder, annotation parameters, Auditor and image gradients. Ordinary retained-state graphs still train. Auditor-only auxiliary exposure reuses exact collected shared features and cannot train the annotator. Records are local to infer and explicitly invalidated around optimizer updates. The independently rerun solver/runtime tests verify preservation of existing gradients, flags and state.

## 9. Compute and memory

Each executed C group uses one differentiable historical replay, one coordinate backward when eligible and at most two candidate replay checks, plus the unchanged official final Auditor. These replace the ordinary next-step annotator on eligible rows: at most three historical annotation forwards versus one ordinary forward, not three additional full-network passes. The encoder and Auditor are not re-evaluated inside the solver. C1 ordinary fallback also costs its ordinary annotation forward. Group size is the original factual batch, potentially larger than the surviving rows. Diagnostic actual work is deduplicated per invocation, separately from per-sample associated counts. Replay adds no learned parameters/buffers or checkpoint entries, but detached states/supports and the replay activation graph consume runtime memory. A direct B=1 FP32 record-storage probe (output 256², features 64², C=96, K=8, J=3, prior evidence present) found 6,291,456 expert-record bytes plus 786,432 accepted-evidence bytes after removal of the unused delta clone: 7,077,888 bytes (6.75 MiB) per row, saving exactly 1 MiB. [replay_memory_probe.json](replay_memory_probe.json) retains before/after storage accounting. This excludes solver activations and the live rollout. RTX 4070 peak memory and latency remain unmeasured. Raw geometry is opt-in and reduced per batch before cohort aggregation.

## 10–11. Dataset flows

Native ACDC and native M&Ms use separate configs/datasets under the same model/trainer. M&Ms contract is four classes BG/RV/MYO/LV with raw mapping {0:0, 1:3, 2:2, 3:1}, paired volumes and masks, patient disjoint splits, unknown-label refusal and explicit canonical depth axis 2. Native and frozen external protocols remain separate. External ACDC->M&Ms loads frozen weights and cannot tune the model, select best checkpoint, calibrate tau or tune solver parameters on M&Ms. External mode/settings declarations must agree across checkpoint config, runtime provenance, evaluation config and CLI; explicit null settings mean defaults, invalid modes fail, and unknown historical metadata cannot certify nondefault C. Root ran Candidate C plus predicted-history exposure through all three synthetic ACDC intervals (6 optimizer steps), best/last, ACDC calibration, transition export and frozen synthetic M&Ms evaluation. A separate native M&Ms run executed all three intervals and 6 steps; limited validation correctly kept completed=false and refused best.pt while writing last.pt. [synthetic_flow_receipt.json](synthetic_flow_receipt.json) records exact counters. Each bootstrap and auditor interval had 4 C attempts, zero replay failures and feasible attempts (2 ACDC, 1 native M&Ms); joint had zero eligible history under the unchanged Auditor. These are fixture observations, not effectiveness estimates.

## 12–13. Real GPU smoke gates

| Required gate | Status | Evidence |
|---|---|---|
| ACDC native RTX 4070 | BLOCKED / NOT RUN | SSH alias4070 read-only nvidia-smi failed exit255 Permission denied(publickey,password) |
| M&Ms native RTX 4070 | BLOCKED / NOT RUN | Same missing remote authentication; no local real dataset/CUDA |
| Frozen ACDC->external M&Ms real cases | BLOCKED / NOT RUN | No reachable real data or source-compatible frozen checkpoint/calibration |

Requested device environment remains CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1. CPU fixtures do not satisfy these gates.

## 14. Tests actually run

Root independently reproduced the numeric-only C1 tensor-identity bug before the fix; its regression now passes. On the quiescent combined tree before the final unused-record cleanup:

- Gate 1 PASS: 21 changed/new Python files compiled; 7 critical imports.
- Gate 2 / selected Gate 6 PASS: **735 passed, 0 skipped**, one existing scheduler-order warning from test_unified_logging, 498.26 seconds on PyTorch 2.4.1 CPU. One combined invocation over 20 focused files; no full suite. Exact argv/timing: [final_test_receipt.json](final_test_receipt.json); complete output: [final_pytest.log](final_pytest.log).
- Included actual core replay/coordinate/FIX/fallback/C3/Auditor tests, conditioning tests, unified trainer/config/logging, checkpoint/resume, Self-Audit core/audit/regressions, M&Ms/native/external CLI, provenance binding and calibration lineage. Resume checks did not silently skip.
- Independent baseline parity: 325 tensors bitwise equal vs archived HEAD, same initialization and input. [baseline_parity.json](baseline_parity.json).
- Additional all-three-interval Candidate C CPU fixture probe: separate ACDC and native M&Ms, 6 optimizer steps each, predicted-history exposure enabled; frozen external C tested from selected ACDC checkpoint. The coordinator initially asserted completed/best.pt for bounded native validation; the trainer correctly refused them. The harness assertions were corrected and rechecked on existing artifacts, with no retraining or production change. See [synthetic_flow_receipt.json](synthetic_flow_receipt.json).
- git diff --check PASS. All 23 source/config/test hashes and scoped source signature stayed unchanged through that test run. The pre-cleanup snapshot remains in pre_cleanup_source_manifest.json. Fresh fetch confirms HEAD == origin/main == the base above.

After the two-line unused record allocation/field removal and sole fixture keyword update, final-source verification also PASS:

- **184 passed, 0 skipped in 59.75 seconds**, covering C contracts, Candidate C runtime/training, exact resume, runtime checkpoint and Self-Audit core. [post_cleanup_test_receipt.json](post_cleanup_test_receipt.json) records argv and final source signature; [post_cleanup_pytest.log](post_cleanup_pytest.log) contains output. These tests overlap the 735; they are not 919 distinct tests.
- Recompiled all changed/new Python and rechecked 4 critical imports.
- Re-ran the all-three-interval Candidate C ACDC and native M&Ms fixture probe, including frozen selected ACDC best.pt, calibration and external M&Ms, on the final source. Both flows PASS. Native validation remains deliberately bounded and correctly refuses completed/best.pt.
- Re-measured replay record storage after removal. Final manifest source signature/hash verification and git diff --check PASS. No broader repeat was justified by removing an unconsumed runtime tensor.

## 15. Tests intentionally skipped

No 130-epoch training, full M&Ms training, large real-data validation, Dice/efficacy comparison, ablation sweep or full test suite after each worker. The complete suite is optional only after required gates pass, so it is not planned while real GPU gates remain blocked. Live W&B service logging is not tested; payload/serialization checks are software-only.

## 16. Scientific risks

Predicted FIX/REGRESS can be wrong; preserving predicted FIX classes is not proof of GT correctness. A cold/frozen bootstrap Auditor may yield little accepted history, so opt-in exposure logs actual occupancy and cannot guarantee useful conditioning gradients. C can repair only errors attributed to a previously accepted ordinary action; default 3-turn cap permits one restitution opportunity. Strict constraints and rho/lambda may yield frequent identity fallback. These are falsifiable questions for a later controlled experiment, not evidence to optimize away or a Dice result.

## 17. Engineering risks

CUDA/AMP/TF32 replay identity and RTX 4070 memory are unverified. Old last.pt exact resumes fail closed after config schema additions; resume from the original checkout or restart, no silent migration. Source-bound calibration must be recreated under the final code on ACDC before frozen external evaluation. Replay is ephemeral and not resumed mid-infer; checkpoints remain epoch-boundary artifacts. Red team identified no additional material code defect. Its M&Ms review was self-review, disclosed and supplemented by root actual-diff review; this limits reviewer independence and is not hidden. The redundant record field it agreed to remove is now removed and retested.

## 18. Decision

**Code integration: PASS. Publication: explicitly authorized by the user. Required real-GPU gates: NOT RUN by the agent.** Candidate C is ready for the next bounded GPU validation step; real controlled-experiment readiness remains unverified. The user said “puhs đi t tự chạy” after reviewing the blocked-GPU status, authorizing normal publication to main and taking responsibility for running those checks. This supersedes the earlier requirement to wait for GPU GREEN before pushing; it does not relabel any GPU check as passed.

The remaining enabling inputs are working authentication to the authorized RTX 4070 environment, its repository/runtime and actual ACDC/M&Ms data paths, and a source-compatible frozen ACDC checkpoint/calibration for external cases. The existing authentication attempt failed; no credentials, GPU status or data availability were invented. After access is restored, run only the already-requested bounded gates under `CUDA_DEVICE_ORDER=PCI_BUS_ID` and `CUDA_VISIBLE_DEVICES=1`. This implementation phase supports no novelty or superiority claim.

## Publication authorization record

Recorded at 2026-09-11T14:13:38.883808+00:00. Implementation commit: `d905637d7f9f32b56de02a03a80a157a82f1de0f`. Source hashes still match the passed test receipts. Reports are committed separately, preserving the original test timestamps and pre/post-cleanup source manifests. Normal push to origin/main is authorized; no force push. GPU checks remain the user's next step.
