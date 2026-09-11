# Candidate C Independent Read-Only Red Team Review

**Date:** 2026-09-11
**Reviewer:** Independent Candidate C Read-Only Red Team Worker
**Base Commit / Ref:** `9a612485b7ead1273082d7dd52879f8e9e7a3e83` (origin/main)
**Combined Quiescent Diff:** 15 tracked files (+4,146 / -71 lines)
**Deliverable Artifact:** `reports/candidate_c/red_team.md`

---

## Conflict of Interest & Role Disclosure
Per project orchestration instructions, the dispatched worker previously implemented the M&Ms stream before settling. Consequently, all findings regarding the M&Ms native and external evaluation flows in Dimension 7 are classified as **internal self-review** (which was independently audited and verified by the Astra root coordinator). The reviews of the C1/C2/C3 core architecture, training integration, runtime defaults, and diagnostics were conducted as a strictly independent red-team audit.

---

## 1. Executive Summary & Verdict

### Overall Verdict: **PASS** (Code & Architecture Implementation)
The reviewer did not identify a material contract violation in the inspected combined source. This is an evidence-bounded code review, not a proof that every input, device or execution path is correct.

| Evaluation Dimension | Status | Key Architectural Verification |
|---|---|---|
| **Dimension 1: C1 Replay & Invariance** | **PASS** | Exact batch-composition replay; numerical failure & stale records cleanly route to current-state ordinary annotation. |
| **Dimension 2: C2 Restitution Solver** | **PASS** | Functional detached parameter/buffer execution; strictly at most 1 backward and $\le 2$ candidate checks; genuine post-projection realized displacement halving; strict FIX protection across all thresholded pixels including ties. |
| **Dimension 3: C3 Transfer & Gating** | **PASS** | Exact zero identity subtraction; detached gate and detached Auditor inputs; strict $\Delta Q > \tau$ acceptance; restitution acceptance invalidates replay history to prevent recursion. |
| **Dimension 4: Runtime Defaults & A0 / Free Offsets** | **PASS** | Bitwise identity of default `window_mode="current"` against base; full reach ($0.92$) for free offsets matching structured reach. |
| **Dimension 5: Training & Rollout Exposure** | **PASS** | Default `predicted_history_exposure=false`; primary 130-epoch bootstrap objective preserved; records strictly invalidated across optimizer steps; honest attempt-based counters. |
| **Dimension 6: Diagnostics & Triplet Alignment** | **PASS** | Clean separation of JSON-safe diagnostics and raw geometry behind `capture_geometry=True`; invocation deduplication by scope/turn/record keys; honest historical triplet metrics. |
| **Dimension 7: M&Ms Native & External Flow** | **PASS** *(Self-Review)* | 4-class bijection $\{0: 0, 1: 3, 2: 2, 3: 1\}$; depth axis 2 canonical; fail-closed unknown label validation; strict frozen ACDC lineage enforcement. |

> [!IMPORTANT]
> **Separation of Software Readiness vs. Real-World Execution Gates:**
> This **PASS** verdict records the reviewer's assessment of the inspected source and configuration guards. It **does not** certify clinical utility, segmentation Dice improvements, novel hypothesis superiority, or hardware performance. The required real RTX 4070 GPU execution gates and real clinical dataset evaluations remain **BLOCKED / NOT RUN** due to external environment constraints.

---

## 2. Systematic Evaluation Across Protocol Dimensions

### Dimension 1: C1 Replay & Invariance
**Contract Invariant:** Replay must exactly re-execute the recorded ordinary transition with identical pre-state, features, audit input, turn/iteration indices, and batch composition, altering only realized sampling supports. Any C1 numerical failure or stale recorded state must cleanly route to the ordinary annotator on current state without crashing or masquerading as an identity C proposal.

* **Factual Batch Preservation:** Replay in `SelfAuditNet._turn_candidates` (`src/self_audit/models/self_audit_net.py:1267-1288`) selects `retained_group = state_full.index_select(0, record.indices.to(state_full.device))` and feeds the entire recorded batch into `_run_restitution_solver`. Inactive or halted rows are preserved during the replay forward pass to prevent kernel tiling or reduction order discrepancies under AMP/autocast.
* **Strict Numerical Tolerances:** In `_run_restitution_solver` (`src/self_audit/models/self_audit_net.py:346-359`), replay logits are compared against stored factual candidate logits via:
  $$\| \text{replay} - \text{factual} \| \le \text{atol} + \text{rtol} \times |\text{factual}|$$
  Max absolute and relative errors are logged, and `c1_passed` is computed per row. Current retained state is similarly verified against stored factual logits to guard against stale updates.
* **Fallback to Ordinary Annotator:** In `_turn_candidates` (`src/self_audit/models/self_audit_net.py:1309-1328`), any row where `c1_passed` is False or `stale` is True is immediately partitioned into `ordinary_positions`. It participates in Phase 2 ordinary `self.annotation_expert` forward pass on current state.
* **Honest Diagnostics:** Fallback rows receive `accepted_path="ordinary"`, explicit `fallback_reason="replay_failure"` or `"stale_record"`, `c1_passed=False` (`c1_failure_kind="numeric"` or `"record_state"`), and `coordinate_displacement=None` (`src/self_audit/models/self_audit_net.py:1380-1413`). They do not pretend to be identity restitution proposals.

---

### Dimension 2: C2 Restitution Solver Invariants
**Contract Invariant:** Optimize coordinate supports $Q$ using functional detached parameters; at most 1 coordinate backward; at most 2 candidate evaluations; genuine post-projection realized displacement halving; strict FIX protection across all thresholded pixels including ties; no ground truth in runtime loss.

* **Functional Detached Autograd Region:** `_run_restitution_solver` (`src/self_audit/models/self_audit_net.py:331-339`) operates inside `torch.inference_mode(False)` and `torch.enable_grad()`. `AnnotationExpert.replay` invokes `torch.func.functional_call` with `_frozen_state()`, where all parameters and buffers are detached (`src/self_audit/models/annotation_expert.py:551-617`). Only `variables = tuple(item.detach().clone().requires_grad_(True) ...)` receive gradients.
* **Parameter & Buffer Gradient Safety:** Gradients are computed using `torch.autograd.grad(total, variables, allow_unused=True)` (`src/self_audit/models/self_audit_net.py:451-454`). No `.grad` buffer on model parameters is created, modified, or cleared. Pre-existing parameter gradients in caller graphs remain intact (verified by adversarial micro-probe).
* **Search Budget:**
  - Coordinate backward pass: Exactly 1 (`evals["coordinate_backward"] += 1`).
  - Candidate checks: Clamped to `checks = min(int(config.max_backtracks), MAX_CANDIDATE_CHECKS)` where `MAX_CANDIDATE_CHECKS = 2` (`src/self_audit/models/self_audit_net.py:511`). Even direct runtime configuration cannot exceed 2 candidate forwards.
* **Genuine Backtracking:** Backtracking proposal 2 is generated via `_halve_towards_factual(factual_supports, projected)` (`src/self_audit/models/self_audit_net.py:245-257`). By halving the *realized* displacement between factual supports and already-projected points, it prevents saturated boundary components from stalling on repeated points.
* **Strict FIX Protection:**
  - Protected set: All pixels where `fix_raw >= float(config.fix_threshold)` (`src/self_audit/models/self_audit_net.py:376`).
  - Margin & Argmax Constraints: Every protected pixel must satisfy `candidate_margin >= required_margin` and `candidate_logits.argmax(dim=1) == winner` (`src/self_audit/models/self_audit_net.py:541-552`).
  - Exact Ties: Exact zero-margin ties have zero required margin; small positive margins retain their positive fractional margin requirement. Both also require exact class identity.
  - Zero Exclusions: `num_ties_excluded` is locked to 0 (`src/self_audit/models/self_audit_net.py:659`).
  - Double Check on Final Transfer: The convex mixture transfer `retained + gate * innovation` is explicitly verified against `winner` for all protected pixels (`src/self_audit/models/self_audit_net.py:620-642`). Any flip reverts the row to zero innovation and factual supports.
* **No Ground Truth Runtime:** The restoration target is `expert_record.annotation_logits.argmax(dim=1)` (`src/self_audit/models/self_audit_net.py:386`), reflecting the pre-transition prediction. Ground truth is entirely inaccessible during inference.

---

### Dimension 3: C3 Transfer & Gating
**Contract Invariant:** Innovation transferred as `current + sigmoid(recorded_gate) * (B_cf - B_factual)`; exact zero identity fallback; detached gate; detached Auditor inputs; strict $\Delta Q > \tau$; no recursive replay.

* **Innovation Formulation & Exact Fallback:** Innovation is computed as `innovation = torch.where(settled.view(-1, 1, 1, 1), best_logits - factual_logits, zero)` (`src/self_audit/models/self_audit_net.py:604`). When a row fails feasibility or improvement (`settled == False`), `innovation` is identically zero, yielding $B_{factual} - B_{factual} = 0$.
* **Detached Gate:** The gate is extracted as `gate = torch.sigmoid(expert_record.factual_update_gate).detach()` (`src/self_audit/models/self_audit_net.py:610`).
* **Auditor Isolation & Strict Decision:** In `SelfAuditNet.infer` (`src/self_audit/models/self_audit_net.py:1063-1085`), all inputs to the Auditor (`shared_active.detach()`, `previous_probs.detach()`, `candidate_probs.detach()`, `(candidate_probs - previous_probs).detach()`, entropies) are detached. Acceptance evaluates `delta_q > float(tau_accept)`. A rejected row immediately HALTs (`halt_now = rejected & (halt_turn < 0); active = accepted`).
* **Non-Recursive Replay Enforcement:** In `_update_records` (`src/self_audit/models/self_audit_net.py:1625-1669`), an accepted Candidate C or rollback action updates `previous_kind[global_index] = RECORD_KIND_CANDIDATE_C` and returns without recording an ordinary record. On the next turn, the row is deemed ineligible for replay (`previous_action_not_replayable`) and routed to the ordinary annotator.

---

### Dimension 4: Runtime Defaults & A0 / Free Offsets
**Contract Invariant:** Default `window_mode: "current"` must be bitwise identical to the base model; `feature_only` must zero both audit paths; `free_offsets` reach must span the full structured support range.

* **Bitwise Parity:** In `SelfAuditNet.__init__`, `window_mode` defaults to `"current"` (`src/self_audit/models/self_audit_net.py:844`). Independent parity verification against the archived base commit on 325 tensors (274 state dict entries and 51 output tensors) confirmed bitwise equality (`reports/candidate_c/baseline_parity.json`).
* **Feature-Only Ablation:** `AnnotationExpert.forward` (`src/self_audit/models/annotation_expert.py:373-408`) zeroes both the input projection audit channels (`audit_low = zeros(...)`) and disables the dynamic window conditioning map (`window_condition = None`).
* **Full Free Reach:** `DynamicWindowGenerator.free_support_reach` (`src/self_audit/models/dynamic_window.py:168-179`) calculates:
  $$\text{reach} = \text{max\_center\_displacement}\,(0.25) + \text{max\_radius}\,(0.55) + \text{max\_residual\_offset}\,(0.12) = 0.92$$
  The `free` offset mode rescales unit residuals by `free_support_reach` (`src/self_audit/models/dynamic_window.py:343`), ensuring the ablation control is not artificially hobbled by a $0.12$ ceiling.

---

### Dimension 5: Training & Rollout Exposure
**Contract Invariant:** Curriculum defaults must keep `predicted_history_exposure: false` and `predicted_history_weight: 0.1`; primary 130-epoch bootstrap objective must be preserved; records must be invalidated across optimizer steps; counters must reflect honest attempts.

* **Config Schema Defaults:** In `UnifiedRolloutConfig` (`src/self_audit/training/unified_config.py:532-548`), `predicted_history_exposure` defaults to `False` and `predicted_history_weight` defaults to `0.1`. Unknown keys are rejected.
* **Epoch & Step Invalidation:** `unified_trainer.py` (`src/self_audit/training/unified_trainer.py:1343-1488`) invokes `self.invalidate_candidate_c_records()`:
  1. At epoch start (line 1343)
  2. Prior to each batch (line 1364)
  3. After each optimizer step (line 1452)
  4. After partial/accumulation steps (line 1486)
  This calls `AnnotationExpert.invalidate_replay_records()`, bumping `_replay_generation` and altering `state_identity`, ensuring stale records fail closed with `StaleReplayRecordError`.
* **Attempt Counters:** In `finetune_joint.py` (`src/self_audit/training/finetune_joint.py:447-598`), `_candidate_c_attempted` identifies every turn that presented a replayable ordinary record, regardless of whether the solve succeeded, failed C1, or had no REGRESS mass. `candidate_c_attempts` properly forms the denominator. Exposure tracking respects the current path: previous-ordinary $\rightarrow$ current-C counts as solver exposure, whereas previous-C $\rightarrow$ current-ordinary counts as ordinary annotation exposure.

---

### Dimension 6: Diagnostics & Triplet Alignment
**Contract Invariant:** JSON-serializable diagnostic rows; geometry isolated behind `capture_geometry=True`; invocation deduplication by scope/turn/record keys; honest historical triplet metrics.

* **Schema Separation:** `infer` emits `candidate_c_diagnostics` as a list of pure Python/primitive numeric dicts. Large coordinate tensors are returned only under `candidate_c_geometry` when `capture_geometry=True` is requested (`src/self_audit/models/self_audit_net.py:1194-1196`).
* **Invocation Deduplication:** In `evaluation/candidate_c.py` (`src/self_audit/evaluation/candidate_c.py:340-362`), `_invocation_key` keys solver calls via `solver_invocation_id` or composite `(scope, turn, record_turn)`. Solver evals are aggregated once per batched invocation rather than erroneously multiplying work by surviving sample rows.
* **Historical Triplet Alignment:** `_row_triplet_indices` (`src/self_audit/evaluation/candidate_c.py:1260-1298`) aligns `transition_previous[record_turn]`, `transition_candidates[record_turn]`, and `transition_candidates[turn]`. True GT repair and destruction are scored across both `proposal` and `retained` families, ensuring unaccepted proposals are never mischaracterized as realized repairs.

---

### Dimension 7: M&Ms Native & External Flow *(Self-Review)*
**Contract Invariant:** Explicit 4-class mapping $\{0: 0, 1: 3, 2: 2, 3: 1\}$; canonical depth axis 2; patient-disjoint splits; paired volume validation; fail-closed unknown labels; strict frozen ACDC checkpoint lineage without M&Ms tuning.

* **Class Mapping:** `MNMSClassMapping` (`src/self_audit/data/mnms.py:33-110`) strictly enforces the bijection $\{0: 0, 1: 3, 2: 2, 3: 1\}$ (raw BG0 → unified BG0, raw LV1 → unified LV3, raw MYO2 → unified MYO2, raw RV3 → unified RV1). In `apply()`, any unrecognized raw integer label immediately raises a `ValueError`.
* **Path Sanitation:** `is_mnms_binary_path` (`src/self_audit/data/mnms.py:141-150`) inspects path parts for `mnm_binary` or `mnms_binary`, avoiding brittle substring matches while blocking contaminated binary derivatives.
* **Frozen Lineage & Anti-Tuning:** In `scripts/evaluate_external_mnms.py` (`scripts/evaluate_external_mnms.py:144-226`), `_inspect_checkpoint_dataset` verifies that checkpoints declare training dataset `acdc`. Checkpoints trained on M&Ms or bearing conflicting dataset claims are strictly rejected. Model configuration, window mode, and solver parameters are locked to the frozen source artifact.

---

## 3. Targeted Review of Coordinator Inquiry: `ExpertReplayRecord.factual_delta_logits`

The coordinator flagged that `ExpertReplayRecord.factual_delta_logits` is cloned and retained without active consumers. A thorough cross-codebase audit was performed:

1. **Current Usage:**
   - Declared in `ExpertReplayRecord` dataclass (`src/self_audit/models/annotation_expert.py:239`).
   - Materialized in `build_replay_record` (`src/self_audit/models/annotation_expert.py:540`) as `factual_delta_logits=_freeze_tensor(output.delta_logits)`.
   - Referenced in a mock synthetic record fixture in `tests/test_candidate_c.py:1496`.
2. **Consumption Audit:**
   - Neither `_run_restitution_solver`, `_run_direct_rollback`, `_turn_candidates`, `_diagnostic_row`, nor any evaluation/diagnostic utility ever reads `record.factual_delta_logits`.
   - The restitution solver reconstructs candidate counterfactual updates via `expert.replay` and measures C3 innovation against `factual_candidate_logits`, while applying `factual_update_gate`. Unscaled factual delta logits are entirely unreferenced.
3. **Memory Impact:**
   - At $256 \times 256$ spatial resolution and 4 classes (float32), retaining `factual_delta_logits` adds $\approx 1.05\,\text{MB}$ of detached heap allocation per active sample row per recorded turn.
4. **Integration Recommendation:**
   - **The inspected code supports removing the unused `factual_delta_logits` field, with targeted replay tests required after the patch.**
   - Because `ExpertReplayRecord` is strictly an ephemeral runtime structure that is never persisted in `state_dict` or serialized into disk checkpoints, removing this field does not change serialized checkpoint fields or model weights.
   - Removing it will clean up an unused $\approx 1.05\,\text{MB/row}$ allocation per turn, directly advancing the contract minimal-record commitment.

---

## 4. Blocked Infrastructure Assessment (External Gates)

The project runbook and architecture contract define required real GPU smoke gates. These gates could not be executed due to the following external environment blockers:

1. **Remote RTX 4070 SSH Authentication Failure:**
   - Command attempted by coordinator: `ssh -o BatchMode=yes 4070 nvidia-smi`
   - Outcome: **Exit 255 (`Permission denied (publickey,password)`)**
   - Impact: Remote GPU environment, driver configuration, and CUDA runtime are unreachable.
2. **Absence of Local GPU & Clinical Datasets:**
   - Local system: Apple Silicon macOS (arm64), Python 3.10.21, PyTorch 2.4.1 CPU. `torch.cuda.is_available()` is `False`.
   - Datasets: Full clinical preprocessed cohorts (`preprocessed_data/ACDC` and `preprocessed_data/mnm`) are absent from the local checkout.
3. **Implications for Certification:**
   - Synthetic CPU tests (including the independently reproduced root 735-test integration suite) verify software logic and algorithmic flows.
   - They **do not substitute** for the RTX 4070 GPU gates. Peak VRAM utilization, CUDA/AMP kernel tiling numerical precision, and clinical segmentation Dice remain unmeasured.

---

## 5. Residual Risks & Future Verification

1. **CUDA / AMP Numerical Replay Divergence:** While C1 replay tolerance ($\text{atol}=\text{rtol}=10^{-5}$) holds consistently on CPU, certain fused CUDA or TF32 kernels could exhibit subtle floating-point variations. In such events, the implementation fail-closed guard correctly routes samples to the ordinary annotator, but real GPU calibration will determine the empirical frequency of this fallback.
2. **Frequency of Identity Fallback Under Strict FIX:** Protecting all thresholded FIX pixels (including exact ties and tiny margins) strengthens theoretical safety guarantees, but may cause the solver to fall back to factual coordinates more frequently on noisy clinical images.
3. **Auditor Cold Start in Curriculum Rollout:** During initial bootstrap epochs with `predicted_history_exposure: true`, an untrained Auditor may produce unstable or near-uniform $\Delta Q$ scores. The code fallback and clamping safeguards prevent numerical blowups, but empirical utility depends on subsequent training dynamics.

---

## 6. Final Red Team Conclusion

No additional material defect was demonstrated in this review. Root integration test receipts are the authoritative executable evidence; this review does not establish GPU behavior or scientific effectiveness.

* **Code Correctness Verdict:** **PASS**
* **Publication / Deployment Gate:** **BLOCKED** pending authorized RTX 4070 access and clinical dataset validation.

## Root integration amendment
After this read-only review settled, root removed the unused factual_delta_logits declaration/allocation and its sole synthetic fixture argument. New replay/runtime/source-identity and synthetic-flow receipts in final_integration_review.md supersede pre-cleanup evidence where source hashes differ.
