# Wave 2.2 Implementation Report: Explicit Typed Entropy APIs

**Date:** 2026-09-08
**Worker:** AGY Implementer
**Task ID:** `task_36bce70a17f9`
**Dispatch ID:** `ctx_515c0d62b202`
**Coordinator Terminal:** `term_bb8fdfa8-8643-4694-8cda-d05efb24504b`
**Worker Terminal:** `term_4991158f-02ba-4966-80ad-dfb1b2b608f5`
**Scope:** Wave 2.2 ONLY (`reports/astra_impl_review/2026-09-08/wave2_entropy_task.md` & `execution_plan.md` W2.2)
**Status:** Completed and Verified locally

---

## 1. Executive Summary

1. **What was done:** Implemented explicit typed entropy APIs (`entropy_from_logits` and `entropy_from_probabilities`) in `src/self_audit/models/annotation_expert.py` and exported them in `src/self_audit/models/__init__.py`, completely eliminating heuristic range-based min/max dispatch. `AnnotationExpert.forward` now explicitly calls `entropy_from_logits`, `entropy_from_probabilities` strictly validates shape, finite values, range, and simplex sum without silent renormalization, and `MODEL_ENTROPY_VERSION = "2.0.0"` is exposed for lineage hashing without altering parameter tensors or checkpoint `state_dict` keys. Additionally, strengthened `test_validator_and_loader_parity_with_manifest` in `tests/test_dataset_split_safety.py` to loop all configured splits including test, use dataset/loader builders, and assert parity across paths, IDs, and all batch items.
2. **What was found and verified:** The preflight logit shift error of `0.2659366` identified in Astra's review (`evidence.md` probe with `z = [0, 0.1, 0.2, 0.3]`) was completely eliminated (shift error = `0.0`). All 13 dedicated regression tests in `tests/test_entropy_contract.py` pass cleanly (covering constant logit shift invariance, batch-composition invariance, uniform logits entropy ≈ 1.0, one-hot entropy = 0.0, probability/logit parity, simplex violation rejection, finite backward gradients, backwards-compatibility alias behavior, version metadata, and unchanged `state_dict` keys), and all 26 split safety tests pass.
3. **What's left / Next steps:** Work is strictly bounded to Wave 2.2. The full test suite passes (**175 passed, 1 pre-existing warning in 5.23s**), and all repository Python files byte-compile cleanly. Because this bug fix corrects entropy computation for logits with values inside `[0, 1]`, existing model checkpoints or calibrated thresholds (`tau`) may alter predictions; no Dice improvement is claimed and no backwards calibration compatibility is assumed. Waves 2.3 (Geometry) and Waves 3–4 remain unopened.

---

## 2. Root Cause Analysis and Architectural Solution

### 2.1 Root Cause Analysis

In the prior implementation of `annotation_entropy` (`src/self_audit/models/annotation_expert.py:18-21`):
```python
if logits_or_probs.min().detach() < 0 or logits_or_probs.max().detach() > 1:
    probs = logits_or_probs.softmax(dim=1)
else:
    probs = logits_or_probs / logits_or_probs.sum(dim=1, keepdim=True).clamp_min(eps)
```
This heuristic had three critical flaws:
1. **Value-Range Sensitivity for Logits:** When unnormalized model logits happened to have all values within `[0, 1]` (e.g., `z = [0.0, 0.1, 0.2, 0.3]`), the function treated them as probabilities, performing linear division rather than `softmax`.
2. **Logit Shift Non-Invariance:** Shifting logits by a constant (e.g., `z - 2`) forced `min < 0`, switching the code path to `softmax` and causing a severe shift error of `0.2659366`.
3. **Batch-Composition Coupling:** Because `min()` and `max()` were evaluated across the entire batch tensor, an extreme logit in one sample could alter how all other samples in the batch were normalized, violating per-sample independence.

### 2.2 Architectural Solutions

1. **`entropy_from_logits(logits, eps=1e-8) -> Tensor`:**
   - Computes normalized Shannon entropy strictly from logits using `F.softmax(logits, dim=1)` and `F.log_softmax(logits, dim=1)`.
   - Invariant to channel-wise constant shifts (`softmax(x + c) == softmax(x)`).
   - Invariant to batch composition (per-pixel channel operations).
   - Normalized by $\ln(C)$ so uniform logits produce $\approx 1.0$.

2. **`entropy_from_probabilities(probs, eps=1e-8, atol=1e-3) -> Tensor`:**
   - Strictly validates 4D shape `[B, C, H, W]`.
   - Validates finite values (rejects NaN and Inf with `ValueError`).
   - Validates valid probability range `[-atol, 1 + atol]` (rejects negative or $>1$ values).
   - Validates the per-pixel probability simplex $\sum_c p_c = 1.0 \pm \text{atol}$ (rejects unnormalized distributions rather than silently renormalizing bad inputs).
   - Uses the standard $0 \cdot \ln(0) \to 0$ limit convention (`probs * log(clamp_min(probs, eps))`), guaranteeing exact `0.0` for one-hot inputs.

3. **`annotation_entropy(x, eps=1e-8, *, input_type="logits", atol=1e-3) -> Tensor`:**
   - Backwards-compatibility wrapper with explicit input typing.
   - Defaults to `"logits"` for legacy callers.
   - Raises `ValueError` for invalid or unknown `input_type`; heuristic range-based dispatch is completely removed.

4. **Lineage Versioning & Checkpoint Compatibility:**
   - Defined `MODEL_ENTROPY_VERSION: str = "2.0.0"` in `annotation_expert.py` and exported from `models/__init__.py`.
   - Added class attribute `AnnotationExpert.entropy_version = MODEL_ENTROPY_VERSION`.
   - No parameter tensors or buffers were added, leaving `state_dict` keys 100% identical.

5. **Split Parity Regression Test Strengthening:**
   - In `tests/test_dataset_split_safety.py:test_validator_and_loader_parity_with_manifest`:
     - Loops all configured splits (`train`, `val`, `test`).
     - Constructs datasets and loaders using `build_patient_dataset` and `build_data_loader`.
     - Asserts case IDs and exact paths match `validate_dataset_splits` descriptors.
     - Iterates through all batches and asserts every individual batch case ID belongs to the expected split.

---

## 3. Detailed Changes by File

### 3.1 `src/self_audit/models/annotation_expert.py`
- Added `import math`.
- Defined `MODEL_ENTROPY_VERSION = "2.0.0"`.
- Implemented `entropy_from_logits(logits, eps=1e-8) -> Tensor`.
- Implemented `entropy_from_probabilities(probs, eps=1e-8, *, atol=1e-3) -> Tensor`.
- Updated `annotation_entropy(x, eps=1e-8, *, input_type="logits", atol=1e-3) -> Tensor` with explicit typed dispatch.
- Added `AnnotationExpert.entropy_version = MODEL_ENTROPY_VERSION`.
- Updated `AnnotationExpert.forward` line 110 to call `entropy_from_logits(annotation_logits)`.

### 3.2 `src/self_audit/models/__init__.py`
- Exported `MODEL_ENTROPY_VERSION`, `entropy_from_logits`, and `entropy_from_probabilities` alongside `annotation_entropy`.

### 3.3 `tests/test_dataset_split_safety.py`
- Imported `build_data_loader` and `build_patient_dataset`.
- Strengthened `test_validator_and_loader_parity_with_manifest` to test all configured splits (`train`, `val`, `test`), check record paths and IDs against validator output, and verify every sample in every DataLoader batch.

### 3.4 `tests/test_entropy_contract.py`
- Created 13 dedicated unit and regression tests verifying all pass criteria:
  - `test_constant_logit_shift_invariance`
  - `test_batch_composition_invariance`
  - `test_astra_review_probe_shift_error_zero`
  - `test_expert_forward_uses_logits_formula_for_unit_interval_logits`
  - `test_uniform_logits_and_probabilities_entropy_one`
  - `test_one_hot_probabilities_entropy_zero`
  - `test_valid_probabilities_parity_with_logits`
  - `test_invalid_probability_simplex_rejected`
  - `test_finite_gradients_logits_backward`
  - `test_finite_gradients_probabilities_backward`
  - `test_annotation_entropy_alias_contract`
  - `test_model_entropy_version_metadata`
  - `test_annotation_expert_state_dict_keys_unchanged`

---

## 4. Verification Commands and Outputs

### 4.1 Python Source Compilation Check
Executed:
```bash
rtk python3 -m py_compile $(find src tests scripts -name "*.py")
```
Output:
```
(Exit code 0, clean compilation across all Python files)
```

### 4.2 Focused Entropy Test Suite Execution
Executed:
```bash
rtk python3 -m pytest tests/test_entropy_contract.py -v
```
Output:
```
============================= test session starts ==============================
platform darwin -- Python 3.11.16, pytest-9.1.1, pluggy-1.6.0 -- /Users/alvinluong/miniforge3/bin/python3
cachedir: .pytest_cache
rootdir: /Users/alvinluong/Self-Audit
plugins: anyio-4.14.2
collecting ... collected 13 items

tests/test_entropy_contract.py::test_constant_logit_shift_invariance PASSED [  7%]
tests/test_entropy_contract.py::test_batch_composition_invariance PASSED [ 15%]
tests/test_entropy_contract.py::test_astra_review_probe_shift_error_zero PASSED [ 23%]
tests/test_entropy_contract.py::test_expert_forward_uses_logits_formula_for_unit_interval_logits PASSED [ 30%]
tests/test_entropy_contract.py::test_uniform_logits_and_probabilities_entropy_one PASSED [ 38%]
tests/test_entropy_contract.py::test_one_hot_probabilities_entropy_zero PASSED [ 46%]
tests/test_entropy_contract.py::test_valid_probabilities_parity_with_logits PASSED [ 53%]
tests/test_entropy_contract.py::test_invalid_probability_simplex_rejected PASSED [ 61%]
tests/test_entropy_contract.py::test_finite_gradients_logits_backward PASSED [ 69%]
tests/test_entropy_contract.py::test_finite_gradients_probabilities_backward PASSED [ 76%]
tests/test_entropy_contract.py::test_annotation_entropy_alias_contract PASSED [ 84%]
tests/test_entropy_contract.py::test_model_entropy_version_metadata PASSED [ 92%]
tests/test_entropy_contract.py::test_annotation_expert_state_dict_keys_unchanged PASSED [100%]

============================== 13 passed in 0.42s ==============================
```

### 4.3 Combined Wave 2 Focused Test Suite Execution
Executed:
```bash
rtk python3 -m pytest tests/test_entropy_contract.py tests/test_dataset_split_safety.py -v
```
Output:
```
============================== 39 passed in 0.50s ==============================
```

### 4.4 Full Project Test Suite Execution
Executed:
```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 rtk python -m pytest -q -p no:cacheprovider tests
```
Output:
```
........................................................................ [ 41%]
........................................................................ [ 82%]
...............................                                          [100%]
=============================== warnings summary ===============================
tests/test_self_audit_core.py::test_dynamic_window_coordinates_and_backward_are_valid
  /Users/alvinluong/Self-Audit/tests/test_self_audit_core.py:32: UserWarning: Converting a tensor with requires_grad=True to a scalar may lead to unexpected behavior.
  Consider using tensor.detach() first. (Triggered internally at /Users/runner/work/pytorch/pytorch/torch/csrc/autograd/generated/python_variable_methods.cpp:823.)
    assert float(coordinates.min()) >= -1.0

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
175 passed, 1 warning in 5.23s
```

---

## 5. Checkpoint & Calibration Impact Disclosure

> [!WARNING]
> Because `annotation_entropy` previously used linear normalization whenever all logit values fell in $[0, 1]$, fixing this bug to use standard log-softmax Shannon entropy changes the conditioning input to `AnnotationExpert` for such states.
>
> As a result:
> - Existing checkpoints trained with the broken entropy heuristic may produce slightly different predictions during recurrent turns.
> - Previously calibrated acceptance thresholds ($\tau_{\text{accept}}$) may require re-calibration on validation transitions.
> - No claim of improved Dice or automatic backwards calibration compatibility is made from this bug fix alone.

---

## 6. Scope Boundaries & Non-Claims

1. **Strictly Bounded Scope:**
   - Only `src/self_audit/models/annotation_expert.py`, `src/self_audit/models/__init__.py`, `tests/test_dataset_split_safety.py`, and new `tests/test_entropy_contract.py` were modified or added.
   - `train_auditor._entropy` was intentionally preserved and not modified (it operates purely on probabilities in a separate training routine).
   - No changes to network backbones, FPN, heads, objectives, loss formulations, optimizer settings, or state machines.
2. **Invariants Preserved:**
   - No commits or git pushes were executed.
   - No model training was executed.
   - No filesystem searches outside the repository were conducted.
