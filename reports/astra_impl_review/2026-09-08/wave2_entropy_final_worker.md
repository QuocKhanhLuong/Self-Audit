# Wave 2.2 Final Revision Report: Invalid-Logits Nonfinite Guard

**Date:** 2026-09-08
**Worker:** AGY Implementer
**Task ID:** `task_8c152dfedfeb`
**Dispatch ID:** `ctx_8ee228c3f2ce`
**Coordinator Terminal:** `term_bb8fdfa8-8643-4694-8cda-d05efb24504b`
**Worker Terminal:** `term_4991158f-02ba-4966-80ad-dfb1b2b608f5`
**Scope:** Wave 2.2 Final Guard ONLY (`reports/astra_impl_review/2026-09-08/wave2_entropy_review_notes.md` §Final)
**Status:** Completed and Verified locally

---

## 1. Executive Summary

1. **What was done:** Added an explicit finite-input validation guard (`if not torch.isfinite(logits).all(): raise ValueError(...)`) placed strictly BEFORE the $C \le 1$ shortcut in `entropy_from_logits` (`src/self_audit/models/annotation_expert.py`), ensuring that NaN, $+\infty$, and $-\infty$ inputs are rejected with a clear `ValueError` rather than masked as finite zeros by downstream numerical safety handlers (`nan_to_num` / `where`). Verified identical pre-shortcut finite validation in `entropy_from_probabilities`, and added 2 dedicated regression tests in `tests/test_entropy_contract.py`.
2. **What was found and verified:** All nonfinite logit inputs (NaN, $+\infty$, $-\infty$) across both multi-channel ($C=4$) and single-channel ($C=1$) tensors are now strictly rejected with `ValueError`, while valid extreme finite fp16 logits ($[65504, -65504, 0, 0]$) and fp16 one-hot probabilities continue to produce finite outputs and strictly finite gradients. All 19 entropy contract tests pass in 0.44s, all 45 combined Wave 2 focused tests pass in 0.55s, the full test suite passes (**181 passed, 1 pre-existing warning in 5.47s**), and all Python files byte-compile cleanly.
3. **What's left / Next steps:** Work on Wave 2.2 is completely finished with all numerical safety and input validation contracts satisfied. No model backbones, loss functions, geometry, or training pipelines were touched, and existing `state_dict` keys remain unchanged. Wave 2.3 (Geometry) and subsequent waves remain unopened.

---

## 2. Root Cause Analysis and Guard Placement

### 2.1 Root Cause

In the previous revision, `entropy_from_logits` used:
```python
log_probs_safe = torch.nan_to_num(log_probs, neginf=-1e30)
p_log_p = torch.where(probs > 0, probs * log_probs_safe, torch.zeros_like(probs))
```
While this successfully stabilized valid extreme finite logits (e.g. $[65504, -65504, 0, 0]$), it had the unintended side effect of masking genuinely invalid inputs:
- Passing `[NaN, 0, 0, 0]` or `[+Inf, 0, 0, 0]` caused `nan_to_num` and `where` to substitute finite zeros, returning `0.0` output but producing nonfinite backward gradients.
- Furthermore, if an input had $C \le 1$, the channel shortcut `if num_classes <= 1: return zeros` returned finite zero before checking whether the tensor contained NaNs or Infs.

### 2.2 Guard Placement

In both `entropy_from_logits` and `entropy_from_probabilities`:
```python
if logits.ndim != 4:
    raise ValueError(f"Expected 4D logits tensor [B, C, H, W], got shape {tuple(logits.shape)}")
if not torch.isfinite(logits).all():
    raise ValueError("Logits must contain only finite values (found NaN or Inf)")
num_classes = logits.shape[1]
if num_classes <= 1:
    return logits.new_zeros((logits.shape[0], 1, logits.shape[2], logits.shape[3]))
```
- Finiteness is validated immediately after the 4D shape check and **before** the $C \le 1$ shortcut.
- Any NaN, $+\infty$, or $-\infty$ entry in the tensor raises an explicit `ValueError`.
- Extreme finite values (such as fp16 max $65504.0$) evaluate as `isfinite() == True` and continue through the stable fp32 computation path.

---

## 3. Verification Commands and Outputs

### 3.1 Python Source Compilation Check
Executed:
```bash
rtk python3 -m py_compile $(find src tests scripts -name "*.py")
```
Output:
```
(Exit code 0, clean compilation across all Python files)
```

### 3.2 Focused Entropy Test Suite Execution
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
collecting ... collected 19 items

tests/test_entropy_contract.py::test_constant_logit_shift_invariance PASSED [  5%]
tests/test_entropy_contract.py::test_batch_composition_invariance PASSED [ 10%]
tests/test_entropy_contract.py::test_astra_review_probe_shift_error_zero PASSED [ 15%]
tests/test_entropy_contract.py::test_expert_forward_uses_logits_formula_for_unit_interval_logits PASSED [ 21%]
tests/test_entropy_contract.py::test_uniform_logits_and_probabilities_entropy_one PASSED [ 26%]
tests/test_entropy_contract.py::test_one_hot_probabilities_entropy_zero PASSED [ 31%]
tests/test_entropy_contract.py::test_valid_probabilities_parity_with_logits PASSED [ 36%]
tests/test_entropy_contract.py::test_invalid_probability_simplex_rejected PASSED [ 42%]
tests/test_entropy_contract.py::test_finite_gradients_logits_backward PASSED [ 47%]
tests/test_entropy_contract.py::test_finite_gradients_probabilities_backward PASSED [ 52%]
tests/test_entropy_contract.py::test_annotation_entropy_alias_contract PASSED [ 57%]
tests/test_entropy_contract.py::test_model_entropy_version_metadata PASSED [ 63%]
tests/test_entropy_contract.py::test_annotation_expert_state_dict_keys_unchanged PASSED [ 68%]
tests/test_entropy_contract.py::test_fp16_onehot_entropy_finite_and_gradients_finite PASSED [ 73%]
tests/test_entropy_contract.py::test_fp16_extreme_logits_finite_entropy_and_gradients PASSED [ 78%]
tests/test_entropy_contract.py::test_bfloat16_precision_stability PASSED [ 84%]
tests/test_entropy_contract.py::test_float64_precision_preserved PASSED  [ 89%]
tests/test_entropy_contract.py::test_nonfinite_logits_rejected PASSED    [ 94%]
tests/test_entropy_contract.py::test_nonfinite_rejected_before_channel_shortcut PASSED [100%]

============================== 19 passed in 0.44s ==============================
```

### 3.3 Combined Wave 2 Focused Test Suite Execution
Executed:
```bash
rtk python3 -m pytest tests/test_entropy_contract.py tests/test_dataset_split_safety.py -v
```
Output:
```
============================== 45 passed in 0.55s ==============================
```

### 3.4 Full Project Test Suite Execution
Executed:
```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 rtk python -m pytest -q -p no:cacheprovider tests
```
Output:
```
........................................................................ [ 39%]
........................................................................ [ 79%]
.....................................                                    [100%]
=============================== warnings summary ===============================
tests/test_self_audit_core.py::test_dynamic_window_coordinates_and_backward_are_valid
  /Users/alvinluong/Self-Audit/tests/test_self_audit_core.py:32: UserWarning: Converting a tensor with requires_grad=True to a scalar may lead to unexpected behavior.
  Consider using tensor.detach() first. (Triggered internally at /Users/runner/work/pytorch/pytorch/torch/csrc/autograd/generated/python_variable_methods.cpp:823.)
    assert float(coordinates.min()) >= -1.0

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
181 passed, 1 warning in 5.47s
```

---

## 4. Invariants & Scope Verification

1. **Strict File Ownership:** Only `src/self_audit/models/annotation_expert.py`, `tests/test_entropy_contract.py`, and this report were edited.
2. **Behavioral Invariants Preserved:**
   - Logit shift invariance and batch-composition invariance preserved.
   - Stable finite-extreme computations for fp16 ($[65504, -65504, 0, 0]$) and fp16 one-hot probabilities preserved with finite gradients.
   - Original tensor dtypes (`float16`, `bfloat16`, `float32`, `float64`) preserved on outputs and backward gradients.
   - `state_dict` keys for `AnnotationExpert` remain 100% identical.
3. **No External Mutations:** No training executed, no git commits/pushes, no outside-repo searches.
