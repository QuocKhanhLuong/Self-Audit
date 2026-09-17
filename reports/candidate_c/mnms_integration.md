# Candidate C: M&Ms Native and Frozen External Integration Report (Revision 2)

## 1. Executive Summary & Ownership

This report documents the final verification, hardening, and delivery of the M&Ms native supervised and frozen external evaluation flows under the user-selected Candidate C architecture for the Self-Audit project, incorporating all amendments requested in `reports/candidate_c/tasks/mnms_revision2.md`.

- **Worker Role:** Worker 2 — M&Ms native and frozen external integration
- **Active Task / Dispatch ID:** `task_15f845246264` / `ctx_2a83b4b1ce6d`
- **Coordinator Terminal:** `term_dc6a165c-3cfb-436b-afce-b65c544ae347`
- **Worker Terminal:** `term_9826093a-a378-4d20-943f-7a6a7b9e2b5f`
- **Ownership Scope:**
  - `src/self_audit/data/mnms.py`
  - `configs/self_audit_full_mnms.yaml`
  - `scripts/evaluate_external_mnms.py`
  - `tests/test_candidate_c_mnms.py`
  - `tests/test_candidate_c_pipeline_smoke.py`
  - `tests/test_external_mnms_evaluator.py` (stub metadata truthful fix per coordinator permission)
  - `reports/candidate_c/mnms_integration.md`
- **Deleted File:**
  - `src/self_audit/data/dataset_resolution.py` (removed standalone heuristics/uncalled helpers)

Shared core models (`src/self_audit/models/`), trainer implementations (`training/unified_trainer.py`, `finetune_joint.py`), configuration schemas (`unified_config.py`), and helper modules (`_utils.py`, `scripts/train_self_audit.py`) remain strictly unmodified.

---

## 2. Amendments Addressed (Revision 2)

1. **Unconditional `verify_model_config` Enforcement:**
   - Excised the `if hasattr(model, "num_classes")` conditional bypass in `scripts/evaluate_external_mnms.py`.
   - Both the live evaluation model config and the source checkpoint config (when present) are verified unconditionally against the live model via `verify_model_config(model, ...)`.
   - Updated `FakeModel` in `tests/test_external_mnms_evaluator.py` to truthfully declare `self.num_classes = 4`, eliminating teststub workarounds and preserving strict production checks.

2. **Strict Multi-Location Checkpoint Lineage & Disagreement Rejection:**
   - Eliminated `elif` precedence fallback in `scripts/evaluate_external_mnms.py`.
   - Systematically collects all declared occurrences of execution mode across:
     - `checkpoint.config.model.window_mode`
     - `checkpoint.config.window_mode`
     - `checkpoint.window_mode`
     - `checkpoint.provenance.model_identity.window_mode`
     - `checkpoint.model_identity.window_mode`
   - Rejects any internal disagreement with `ValueError`.
   - Systematically collects and validates all declared occurrences of `candidate_c` solver settings across config, top-level, and `model_identity` provenance, rejecting any conflicting parameter definitions.
   - Collects and rejects conflicting `window_mode` or `candidate_c` definitions between evaluation `config.model` and top-level `config`.

3. **Explicit `candidate_c: null` Defined as Default Settings & Invalid Mode Refusal:**
   - In evaluation configurations, an explicit `candidate_c: null` (or `None`) is recognized as an explicit request for documented default settings (`rho_feature_pixels: 1.0`, etc.) rather than absence.
   - If evaluation config declares `candidate_c: null` while the source checkpoint carries custom settings (e.g. `rho_feature_pixels: 2.0`), the mismatch is rejected with `ValueError` rather than silently inheriting.
   - Only genuinely absent keys (`"candidate_c"` not in config) inherit from the source checkpoint.
   - Explicit `window_mode: null`, empty strings, or non-string values are strictly rejected by `validate_window_mode`.

4. **Legacy Checkpoint Baseline Compatibility vs Non-Default Refusal:**
   - Checkpoints lacking execution mechanism metadata retain backwards compatibility for the baseline (`window_mode: "current"`).
   - Attempting to evaluate a legacy checkpoint under non-default runtime (`candidate_c`, `candidate_c_no_fix`, etc.) via CLI or evaluation config fails closed (`ValueError: Cannot evaluate legacy checkpoint lacking execution mechanism metadata under non-default mode...; source mechanism identity cannot be established`).

5. **Narrow, Accurate Scope for Native CLI Smoke Test:**
   - `tests/test_candidate_c_pipeline_smoke.py` docstring and assertions state its exact scope:
     - Executes native M&Ms training CLI (`scripts/train_self_audit.py` with `configs/self_audit_full_mnms.yaml`, `window_mode: "candidate_c"`, and `--max_steps 2`).
     - Bounded to interval 1 (`annotation_bootstrap`), verifying model construction, optimizer stepping, canonical `last.pt` generation (with model and optimizer state), and `pipeline_report.json`.
     - Explicitly acknowledges that it does NOT execute the full 3-interval curriculum or demonstrate an eligible solve (which occurs in interval 3 joint_self_audit under threshold_gate rollout).
     - Does NOT claim or test `best.pt` since `best.pt` is only saved after epoch validation.

---

## 3. Disjoint Scientific Protocols & Canonical Execution Commands

| Scientific Protocol Label | Source Training Data | Evaluation Target Data | Training Permitted | Calibration / Best Selection | Canonical Checkpoint Naming |
|---|---|---|---|---|---|
| `acdc_unified_native_v1` | ACDC (`preprocessed_data/ACDC`) | ACDC val/test | Yes (Unified Trainer) | Val split calibration | `best.pt`, `last.pt` |
| `mnms_unified_native_v1` | M&Ms (`preprocessed_data/mnm`) | M&Ms val/test | Yes (Unified Trainer) | Val split calibration | `best.pt`, `last.pt` |
| `acdc_frozen_to_mnms_external_v1` | ACDC (`best.pt`) | M&Ms `testing` split | **NO** (Strictly Frozen) | **NO** (Fixed $\tau$) | Evaluates source `best.pt` |

### Canonical Execution Commands (root corrected against the CLI)

Native ACDC baseline:
```sh
PYTHONPATH=.:src python scripts/train_self_audit.py \
  --config configs/self_audit_full.yaml --output_dir checkpoints/acdc_unified
```
For Candidate C native M&Ms, copy `configs/self_audit_full_mnms.yaml` to a run-specific config, set `model.window_mode: candidate_c`, and retain its native M&Ms dataset fields. Enable `training.rollout.predicted_history_exposure` only in the declared exposure experiment. The training CLI has no `--window-mode` flag.
```sh
PYTHONPATH=.:src python scripts/train_self_audit.py \
  --config /path/to/mnms_candidate_c.yaml --output_dir checkpoints/mnms_unified_candidate_c
```
Frozen ACDC-to-M&Ms external evaluation uses the external protocol config (which inherits the checkpoint's declared mode/settings), never the M&Ms native training config:
```sh
PYTHONPATH=.:src python scripts/evaluate_external_mnms.py \
  --config configs/self_audit_acdc_to_mnms.yaml \
  --checkpoint checkpoints/acdc_unified/best.pt \
  --output reports/external_mnms.json
```
These are operator commands, not claims that real training or evaluation ran. Source-compatible calibration and a trained frozen checkpoint are still required for scientific evaluation. The bootstrap smoke saves `last.pt`; complete eligible validation controls `best.pt` selection.

---

## 4. Test Verification & Evidence

All 39 tests across 4 test suites pass deterministically in under 10 seconds on CPU using `/private/tmp/self-audit-torch241/bin/python`.

### Test Execution Command & Output

```sh
PYTHONPATH=.:src /private/tmp/self-audit-torch241/bin/python -m pytest     tests/test_candidate_c_mnms.py     tests/test_candidate_c_pipeline_smoke.py     tests/test_external_mnms_evaluator.py     tests/test_mnms_wave2.py
```

```text
============================= test session starts ==============================
platform darwin -- Python 3.10.21, pytest-9.1.1, pluggy-1.6.0
rootdir: /Users/alvinluong/Self-Audit
plugins: anyio-4.15.1
collected 39 items

tests/test_candidate_c_mnms.py .........................                 [ 64%]
tests/test_candidate_c_pipeline_smoke.py .                               [ 66%]
tests/test_external_mnms_evaluator.py .                                  [ 69%]
tests/test_mnms_wave2.py ............                                    [100%]

============================== 39 passed in 9.67s ==============================
```

### Breakdown of Test Suites
1. **`tests/test_candidate_c_mnms.py` (25 tests):**
   - Four-class contract: $\{0: 0, 1: 3, 2: 2, 3: 1\}$, rejection of out-of-range labels, non-finite values, booleans.
   - Refusal of binary derivative datasets (`mnm_binary`, `mnms_binary`).
   - Depth axis $[H, W, Z] 	o [Z, H, W]$ normalization with default `depth_axis=None`.
   - Regression test loading legitimate M&Ms under path containing `acdc` and `mnms` with patient ID `patient001`.
   - Patient split disjointness.
   - Config and model construction parity.
   - External evaluator rejection of M&Ms checkpoints and non-testing splits.
   - Conflicting CLI `--window-mode` rejection.
   - Conflicting Candidate C solver parameter rejection.
   - String coercion rejection in solver settings.
   - **New Revision 2:** Checkpoint internal `window_mode` disagreement rejection.
   - **New Revision 2:** Evaluation config internal `window_mode` disagreement rejection.
   - **New Revision 2:** Checkpoint internal `candidate_c` parameter disagreement rejection.
   - **New Revision 2:** Explicit `candidate_c: None` in evaluation config rejected when checkpoint has custom settings.
   - **New Revision 2:** Rejection of invalid/null `window_mode`.
   - **New Revision 2:** Legacy checkpoint baseline enforcement (fails closed under non-default `candidate_c`).
   - End-to-end external evaluator subprocess smoke test with Candidate C.
   - Read-only resource audit test.
2. **`tests/test_candidate_c_pipeline_smoke.py` (1 test):**
   - Native M&Ms training CLI subprocess bootstrap smoke (`scripts/train_self_audit.py`).
   - Verifies optimizer step, canonical `last.pt`, and `pipeline_report.json`.
3. **`tests/test_external_mnms_evaluator.py` (1 test):**
   - External evaluation regression test verifying truthful `FakeModel` metadata under unconditional `verify_model_config`.
4. **`tests/test_mnms_wave2.py` (12 tests):**
   - Full wave 2 regression suite.

Total runtime: **9.67 seconds**, well within the 2-minute CPU budget.

---

## 5. Read-Only Hardware & Resource Audit

- **Compute Acceleration:** CPU only (`torch.cuda.is_available() == False`, 0 GPUs).
- **Environment:** PyTorch 2.4.1 environment at `/private/tmp/self-audit-torch241/bin/python`.
- **Integrity Guarantee:** All tests execute against synthetic volumetric fixtures ($16 	imes 16 	imes 4$ or $32 	imes 32 	imes 2$). No claims of real-data validation or GPU acceleration are made.

---

## 6. Summary of Changed Files

| File | Status | Description |
|---|---|---|
| `src/self_audit/data/mnms.py` | Modified | Default `depth_axis: int | None = None`; no path/name heuristics; protocol labels. |
| `configs/self_audit_full_mnms.yaml` | Modified | Candidate C solver block, `window_mode: "current"`, and exposure defaults (`predicted_history_exposure: false`, `predicted_history_weight: 0.1`). |
| `scripts/evaluate_external_mnms.py` | Modified | Unconditional `verify_model_config`; multi-declaration conflict rejection; explicit null handling; legacy checkpoint baseline enforcement; single-load reuse; plain dict CandidateC serialization. |
| `tests/test_candidate_c_mnms.py` | Modified | 25 tests covering four-class contract, binary refusal, legitimate path regression, solver conflict rejection, string rejection, legacy baseline enforcement, and external smoke. |
| `tests/test_candidate_c_pipeline_smoke.py` | **New** | Bounded native M&Ms unified synthetic training CLI bootstrap smoke test (`train_self_audit.py`). |
| `tests/test_external_mnms_evaluator.py` | Modified | Declared `self.num_classes = 4` on `FakeModel` fixture for truthful metadata verification. |
| `reports/candidate_c/mnms_integration.md` | **Revised** | Comprehensive integration report documenting all revision items, canonical CLI commands, and test evidence. |
| `src/self_audit/data/dataset_resolution.py` | **Deleted** | Excised uncalled heuristic helper. |
