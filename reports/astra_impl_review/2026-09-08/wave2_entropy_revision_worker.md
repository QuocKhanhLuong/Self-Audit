# Wave 2.2 Numerical Safety Revision Report: FP16/BFloat16 Precision Stability

**Date:** 2026-09-08
**Worker:** AGY Implementer
**Task ID:** `task_487bf40fd0e5`
**Dispatch ID:** `ctx_965b2b5684b0`
**Coordinator Terminal:** `term_bb8fdfa8-8643-4694-8cda-d05efb24504b`
**Worker Terminal:** `term_4991158f-02ba-4966-80ad-dfb1b2b608f5`
**Scope:** Wave 2.2 Revision ONLY (`reports/astra_impl_review/2026-09-08/wave2_entropy_review_notes.md`)
**Status:** Completed and Verified locally

---

## 1. Executive Summary

1. **What was done:** Resolved fp16/bfloat16 numerical instability in `entropy_from_logits` and `entropy_from_probabilities` (`src/self_audit/models/annotation_expert.py`) by executing internal calculations in float32 when receiving half-precision inputs while explicitly preserving the caller's output dtype, float64 precision, and gradient flow. Clamped pathological $-\infty$ in log-softmax products using safe numerical masking (`torch.where` and `torch.nan_to_num`), preventing `0.0 * -inf = NaN` during extreme logit differences or one-hot underflow. Added 4 dedicated numerical safety regression tests to `tests/test_entropy_contract.py`.
2. **What was found and verified:** Astra's fp16 reproduction probes (`p = [1, 0, 0, 0]` one-hot probability underflow producing NaN gradients and `x = [65504, -65504, 0, 0]` extreme logits producing NaN outputs) were completely eliminated: both now yield strictly finite outputs and strictly finite gradients. All 17 focused entropy contract tests pass in 0.70s, combined Wave 2 focused tests pass (43 tests in 0.54s), the full test suite passes (**179 passed, 1 pre-existing warning in 5.55s**), and all Python source files byte-compile cleanly.
3. **What's left / Next steps:** Work is strictly bounded to the numerical safety revision of Wave 2.2 without modifying network backbones, objectives, geometries, or training code. Existing checkpoints may still exhibit slight prediction differences due to the core W2.2 entropy bug fix (as disclosed previously), but half-precision inference and training are now fully protected against NaNs. Wave 2.3 (Geometry) and subsequent waves remain unopened.

---

## 2. Root Cause Analysis and Precision Stability Fix

### 2.1 Root Cause Analysis

In half precision (`torch.float16`):
1. **Clamp Epsilon Underflow:** The default epsilon `eps = 1e-8` is below the smallest subnormal representable number in fp16 ($\approx 5.96 \times 10^{-8}$). Consequently, `probs.clamp_min(1e-8)` underflowed to `0.0`, resulting in $\ln(0.0) = -\infty$. In IEEE floating-point arithmetic, $0.0 \times -\infty = \text{NaN}$, contaminating both forward outputs and backward gradients.
2. **Extreme Logit Difference Overflow:** For extreme fp16 logits such as $x = [65504.0, -65504.0, 0.0, 0.0]$, the difference $65504 - (-65504) = 131008$ exceeds the maximum representable float16 value ($65504.0$). During `log_softmax`, intermediate terms overflowed to $-\infty$, leading to $0.0 \times -\infty = \text{NaN}$.

### 2.2 Precision Stability Solution

1. **Dynamic Compute Precision:**
   ```python
   orig_dtype = x.dtype
   compute_dtype = torch.float32 if orig_dtype in (torch.float16, torch.bfloat16) else orig_dtype
   x_comp = x.to(compute_dtype)
   ```
   - Float16 and BFloat16 inputs are upcast to Float32 during internal operations, where $10^{-8}$ is fully normal and values up to $\pm 3.4 \times 10^{38}$ are representable.
   - Float64 inputs retain Float64 throughout.
   - The final result is explicitly cast back to `orig_dtype`, preserving the tensor contract and autograd graph.

2. **Safe Shannon Product Masking:**
   - In `entropy_from_logits`:
     ```python
     probs = F.softmax(logits_comp, dim=1)
     log_probs = F.log_softmax(logits_comp, dim=1)
     log_probs_safe = torch.nan_to_num(log_probs, neginf=-1e30)
     p_log_p = torch.where(probs > 0, probs * log_probs_safe, torch.zeros_like(probs))
     ```
   - In `entropy_from_probabilities`:
     ```python
     safe_probs = torch.clamp(probs_comp, min=eps, max=1.0)
     log_probs = torch.log(safe_probs)
     p_log_p = torch.where(probs_comp > 0, probs_comp * log_probs, torch.zeros_like(probs_comp))
     ```
   This guarantees that zero probabilities strictly contribute $0.0$ to entropy without triggering NaN through $-\infty$ multiplication or gradient evaluation.

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
collecting ... collected 17 items

tests/test_entropy_contract.py::test_constant_logit_shift_invariance PASSED [  5%]
tests/test_entropy_contract.py::test_batch_composition_invariance PASSED [ 11%]
tests/test_entropy_contract.py::test_astra_review_probe_shift_error_zero PASSED [ 17%]
tests/test_entropy_contract.py::test_expert_forward_uses_logits_formula_for_unit_interval_logits PASSED [ 23%]
tests/test_entropy_contract.py::test_uniform_logits_and_probabilities_entropy_one PASSED [ 29%]
tests/test_entropy_contract.py::test_one_hot_probabilities_entropy_zero PASSED [ 35%]
tests/test_entropy_contract.py::test_valid_probabilities_parity_with_logits PASSED [ 41%]
tests/test_entropy_contract.py::test_invalid_probability_simplex_rejected PASSED [ 47%]
tests/test_entropy_contract.py::test_finite_gradients_logits_backward PASSED [ 52%]
tests/test_entropy_contract.py::test_finite_gradients_probabilities_backward PASSED [ 58%]
tests/test_entropy_contract.py::test_annotation_entropy_alias_contract PASSED [ 64%]
tests/test_entropy_contract.py::test_model_entropy_version_metadata PASSED [ 70%]
tests/test_entropy_contract.py::test_annotation_expert_state_dict_keys_unchanged PASSED [ 76%]
tests/test_entropy_contract.py::test_fp16_onehot_entropy_finite_and_gradients_finite PASSED [ 82%]
tests/test_entropy_contract.py::test_fp16_extreme_logits_finite_entropy_and_gradients PASSED [ 88%]
tests/test_entropy_contract.py::test_bfloat16_precision_stability PASSED [ 94%]
tests/test_entropy_contract.py::test_float64_precision_preserved PASSED  [100%]

============================== 17 passed in 0.70s ==============================
```

### 3.3 Combined Wave 2 Focused Test Suite Execution
Executed:
```bash
rtk python3 -m pytest tests/test_entropy_contract.py tests/test_dataset_split_safety.py -v
```
Output:
```
============================== 43 passed in 0.54s ==============================
```

### 3.4 Full Project Test Suite Execution
Executed:
```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 rtk python -m pytest -q -p no:cacheprovider tests
```
Output:
```
........................................................................ [ 40%]
........................................................................ [ 80%]
...................................                                      [100%]
=============================== warnings summary ===============================
tests/test_self_audit_core.py::test_dynamic_window_coordinates_and_backward_are_valid
  /Users/alvinluong/Self-Audit/tests/test_self_audit_core.py:32: UserWarning: Converting a tensor with requires_grad=True to a scalar may lead to unexpected behavior.
  Consider using tensor.detach() first. (Triggered internally at /Users/runner/work/pytorch/pytorch/torch/csrc/autograd/generated/python_variable_methods.cpp:823.)
    assert float(coordinates.min()) >= -1.0

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
179 passed, 1 warning in 5.55s
```

---

## 4. Scope Boundaries & Invariants

1. **Strict Ownership:** Only `src/self_audit/models/annotation_expert.py`, `tests/test_entropy_contract.py`, and this report were edited or created in this revision.
2. **Preserved Contracts:**
   - Invariance to constant logit shifts and batch composition preserved.
   - Probability simplex validation preserved.
   - `AnnotationExpert.forward` wiring preserved.
   - `AnnotationExpert` `state_dict` keys 100% identical.
   - `train_auditor._entropy` and all other training code left untouched.
3. **No Training or Git Mutations:** No training was launched, and no git commits or pushes were performed.
