# Wave 1.3 Implementation Report: Volume Identity and Metric-Space Safety

**Date:** 2026-09-08
**Worker:** AGY Implementer
**Task ID:** `task_2a825730a0e0`
**Dispatch ID:** `ctx_7b3851306882`
**Coordinator Terminal:** `term_bb8fdfa8-8643-4694-8cda-d05efb24504b`
**Worker Terminal:** `term_4991158f-02ba-4966-80ad-dfb1b2b608f5`
**Scope:** Wave 1.3 ONLY (`reports/astra_impl_review/2026-09-08/wave1_volume_task.md`)
**Status:** Completed and Verified locally

---

## 1. Executive Summary

1. **What was done:** Eliminated implicit cohort pooling in threshold evaluation by requiring explicit case identity (`case_ids`) and omitting volume metrics (`volume_metrics_available = False`, `volume_macro_dice = None`, `volume_case_count = None`, with an explicit explanatory reason) when IDs are missing, unless `require_volume=True` is explicitly requested, which strictly halts with `ContractMismatchError`. Enforced metric-space safety by rejecting volume contracts (`volume_native`, `volume_resized`) at both the 2D slice collector boundary (`collect_validation_transition_cache`) and the transition cache boundary (`validate_and_normalize_transition_cache`), while introducing strict sufficient statistics bundle validation (`_validate_sufficient_statistics_bundle`) that verifies completeness, exact 2D/3D shapes, finite non-negative integer counts, supported class indices, and per-slice Q/stat consistency. Asserted distinct slice vs derived volume metadata (`slice_proxy` / `foreground_dice_exclude_v1` vs `volume_resized` / `foreground_dice_volume_resized_v1`) and enforced contract parity in `save_calibration`.
2. **What was found and verified:** All five mandatory pass criteria from `wave1_volume_task.md` were implemented and verified with comprehensive regression tests:
   - Two-case fixture: Two slices with stats `(1,0,0)` and `(0,9,9)` produce volume Dice $0.50$ (count 2) when supplied with `case_ids=['case1', 'case2']`, whereas without `case_ids` volume metrics are omitted (`volume_metrics_available: False`, `volume_macro_dice: None`, `volume_case_count: None`), completely preventing the erroneous pooled cohort score of $0.10$ (count 1).
   - Direct vs Replay grouped statistics parity: Case-grouped slice sums match direct case volume aggregation exactly.
   - Collector contract rejection: `collect_validation_transition_cache` rejects `volume_native` and `volume_resized` target contracts with `ContractMismatchError`.
   - Malformed statistics rejection: Partial bundles (missing FP/FN), NaN/Inf/negative counts, fractional floats (no silent int truncation), class mismatch, and Q/stat inconsistency are strictly rejected with `ContractMismatchError`.
   - Distinct metadata: Slice and derived-volume metric metadata are explicitly distinguished and asserted.
3. **What's left / Next steps:** Work within Wave 1.3 is complete. The full test suite passes (**136 passed, 1 pre-existing warning in 5.60s**), focused tests pass (**32 passed in 2.92s**), and all 61 Python files across `src/`, `tests/`, and `scripts/` compile cleanly with zero byte-compilation errors. Ready for coordinator review.

---

## 2. Root Cause Analysis and Architectural Fixes

### 2.1 Problem 1: Implicit Cohort Pooling Without Case Identity
- **Root Cause:** In the prior implementation of `threshold.py`, if sufficient statistics (`stats_initial`, `stats_candidate`) were present in the transition cache but `case_ids` were absent, the code fell back to summing all slice statistics into a single global pool, emitting `volume_macro_dice` and `volume_case_count = 1`. In an independent two-slice fixture where slice 1 has Dice $1.0$ (TP=1, FP=0, FN=0) and slice 2 has Dice $0.0$ (TP=0, FP=9, FN=9):
  - True 2-case volume Dice: $\frac{1}{2}(1.0 + 0.0) = 0.50$ ($N=2$).
  - Erroneous pooled cohort Dice: $\frac{2 \times 1}{2 \times 1 + 9 + 9} = \frac{2}{20} = 0.10$ ($N=1$).
  Missing identity is not evidence of a single patient volume.
- **Architectural Fix:**
  - In `_evaluate_threshold_core`: When `case_ids` are present, slices are grouped by unique case ID, slice TP/FP/FN counts are summed per case, case-level macro Dice is calculated, and the patient-level average is reported with `volume_metrics_available = True`, `volume_case_count = len(unique_cases)`, `volume_metric_space = "volume_resized"`, and `volume_metric_contract = "foreground_dice_volume_resized_v1"`.
  - When `case_ids` are missing:
    - If `require_volume=True`: Raises `ContractMismatchError("Volume scoring was explicitly requested (require_volume=True) but case_ids are missing; cohort pooling across unidentified slices is forbidden.")`.
    - Otherwise: Emits `volume_macro_dice = None`, `volume_per_class_dice = None`, `volume_case_count = None`, `volume_metrics_available = False`, and `volume_unavailable_reason = "Missing case_ids; cohort pooling across unidentified slices is forbidden"`.
  - In `validate_and_normalize_transition_cache`: Added `require_volume: bool = False`. If `require_volume=True` and `case_ids` are missing, raises `ContractMismatchError`. Normalizes `case_ids` to `list[str]` (supporting list, tuple, and 1D ndarray) while rejecting empty or sentinel (`"none"`, `"nan"`, `"null"`) identifiers.

### 2.2 Problem 2: Metric-Space Safety at 2D Slice Collector Boundary
- **Root Cause:** `collect_validation_transition_cache` in `src/self_audit/training/finetune_joint.py` iterates over 2D validation slices and evaluates target metrics per slice. If passed a contract specifying `metric_space == "volume_native"` or `"volume_resized"`, it previously stamped that contract onto 2D slice $Q$ values.
- **Architectural Fix:**
  - Added strict metric-space contract validation at entry of `collect_validation_transition_cache`:
    ```python
    if target_contract.metric_space != METRIC_SPACE_SLICE_PROXY:
        raise ContractMismatchError(
            f"collect_validation_transition_cache operates strictly on 2D slices and requires "
            f"metric_space == '{METRIC_SPACE_SLICE_PROXY}'. Got contract '{target_contract.name}' "
            f"with metric_space == '{target_contract.metric_space}'."
        )
    ```
  - Added identical contract check in `validate_and_normalize_transition_cache`: If `expected_contract` or cache `metric_contract` has `metric_space != METRIC_SPACE_SLICE_PROXY`, raises `ContractMismatchError`.

### 2.3 Problem 3: Complete and Aligned Sufficient Statistics Bundle Validation
- **Root Cause:** Transition caches previously permitted partial bundles (e.g. TP and FP without FN), negative counts, non-integer values (which might be truncated silently to int), NaN/Inf, mismatched shapes, or statistics whose implied Dice did not match the corresponding $Q$ slice scores.
- **Architectural Fix:**
  - Implemented `_validate_sufficient_statistics_bundle`:
    - Completeness: Both initial and candidate stats must contain complete `{metric_prefix}_tp`, `{metric_prefix}_fp`, `{metric_prefix}_fn`.
    - Dimension: Initial stats must be exact 2D $(N, C)$ arrays; candidate stats must be exact 3D $(N, T, C)$ arrays.
    - Class range: Number of classes $C$ must satisfy $C > \max(\text{contract.classes})$.
    - Value types and ranges: Must be finite, non-negative, and exact integers (rejects NaN, Inf, negative counts, and floats with non-zero fractional parts; rejects float truncation).
    - Parity: Implemented vectorized per-slice Q/stat parity check (`_check_stats_scores_consistency`) verifying that $\frac{2 \cdot \text{TP}}{2 \cdot \text{TP} + \text{FP} + \text{FN}} == Q$ within $10^{-5}$ float tolerance (handling zero-denominator empty slices according to contract empty policy).

### 2.4 Problem 4: Distinct Slice vs Derived Volume Metadata & Calibration Persistence
- **Root Cause:** Persisted calibration outputs could mislabel 2D slice calibration as `volume_native`.
- **Architectural Fix:**
  - Distinct metadata fields in calibration results:
    - Slice replay: `metric_space = "slice_proxy"`, `metric_contract = "foreground_dice_exclude_v1"`
    - Derived volume: `volume_metric_space = "volume_resized"`, `volume_metric_contract = "foreground_dice_volume_resized_v1"`
  - `save_calibration`: Added strict assertion verifying `selected_row.get("metric_space") == metric_space`. Rejects stamping `volume_native` onto slice cache calibration results.

---

## 3. Detailed Changes by File

### 3.1 `src/self_audit/training/finetune_joint.py`
- Added imports for `METRIC_SPACE_SLICE_PROXY` and `ContractMismatchError` from `self_audit.evaluation.contracts`.
- In `collect_validation_transition_cache`:
  - Enforced `target_contract.metric_space == METRIC_SPACE_SLICE_PROXY`; raises `ContractMismatchError` if a volume contract (`volume_native`, `volume_resized`) is supplied.
  - Stamped `metric_space = target_contract.metric_space` explicitly into the emitted transition cache dictionary.

### 3.2 `src/self_audit/evaluation/threshold.py`
- Added imports for `METRIC_SPACE_SLICE_PROXY`, `METRIC_SPACE_VOLUME_RESIZED`, and `FOREGROUND_DICE_VOLUME_RESIZED_V1_CONTRACT`.
- Added helper `_validate_sufficient_statistics_bundle` and `_check_stats_scores_consistency` to validate TP/FP/FN bundles.
- In `_evaluate_threshold_core`:
  - When `case_ids` are present: groups slice TP/FP/FN by case, computes per-case macro Dice, and sets `volume_metrics_available = True`, `volume_case_count = len(unique_cases)`, `volume_metric_space = "volume_resized"`, `volume_metric_contract = "foreground_dice_volume_resized_v1"`.
  - When `case_ids` are absent: if `require_volume=True`, raises `ContractMismatchError`; otherwise sets `volume_macro_dice = None`, `volume_per_class_dice = None`, `volume_case_count = None`, `volume_metrics_available = False`, and `volume_unavailable_reason = "Missing case_ids; cohort pooling across unidentified slices is forbidden"`.
- In `validate_and_normalize_transition_cache`:
  - Added `require_volume: bool = False`.
  - Validates `metric_space == METRIC_SPACE_SLICE_PROXY` (or rejects mismatch).
  - Normalizes `case_ids` into `list[str]` (supporting list, tuple, and 1D ndarray), validating length matches slice count $N$ and rejecting empty/sentinel values.
  - Invokes `_validate_sufficient_statistics_bundle` whenever statistics keys are present.
- In `evaluate_threshold` and `sweep_thresholds`:
  - Added `require_volume: bool = False` argument and forwarded it to validation and evaluation core.
- In `save_calibration`:
  - Enforces `selected_row.get("metric_space") == metric_space`.

### 3.3 `tests/test_metric_contract_and_replay.py`
Added Section 11 with 5 dedicated regression tests matching all mandatory criteria:
1. `test_w1_3_two_case_fixture_volume_identity_and_no_cohort_pooling`: Verifies the independent two-case fixture (IDs -> 0.50 / count 2; no IDs -> volume scores omitted, `volume_metrics_available=False`, reason present, never cohort pooled to 0.10; `require_volume=True` raises `ContractMismatchError`).
2. `test_w1_3_direct_grouped_statistics_parity`: Verifies that grouping per-slice TP/FP/FN by case ID and computing volume Dice matches direct case volume calculation.
3. `test_w1_3_collector_rejects_volume_target_contracts`: Verifies that `collect_validation_transition_cache` rejects `volume_native` and `volume_resized` contracts with `ContractMismatchError`.
4. `test_w1_3_stats_bundle_validation_and_rejections`: Verifies rejection of partial bundles (missing FN), NaN counts, Inf counts, negative counts, fractional float counts, class dimension mismatches, and Q/stat score inconsistency, while confirming valid $T=0$ and blank-to-hallucination transitions pass.
5. `test_w1_3_distinct_slice_and_derived_volume_metric_metadata`: Verifies distinct metadata fields for slice proxy vs derived volume resized, and asserts `save_calibration` rejects `volume_native` mismatches.

---

## 4. Verification Commands and Outputs

### 4.1 Python Source Compilation Check
Executed:
```bash
rtk python3 -m py_compile $(find src tests scripts -name "*.py")
```
Output:
```
(Exit code 0, clean compilation across all 61 Python files)
```

### 4.2 Focused Test Suite Execution
Executed:
```bash
rtk python3 -m pytest tests/test_metric_contract_and_replay.py -v
```
Output:
```
============================= test session starts ==============================
platform darwin -- Python 3.11.16, pytest-9.1.1, pluggy-1.6.0 -- /Users/alvinluong/miniforge3/bin/python3
cachedir: .pytest_cache
rootdir: /Users/alvinluong/Self-Audit
plugins: anyio-4.14.2
collecting ... collected 32 items

tests/test_metric_contract_and_replay.py::test_p0_01_sign_reversal_eliminated PASSED [  3%]
tests/test_metric_contract_and_replay.py::test_direct_vs_cache_replay_parity PASSED [  6%]
tests/test_metric_contract_and_replay.py::test_mixed_accept_prefix PASSED [  9%]
tests/test_metric_contract_and_replay.py::test_volume_aggregation_patient_grouping PASSED [ 12%]
tests/test_metric_contract_and_replay.py::test_contract_mismatch_rejection PASSED [ 15%]
tests/test_metric_contract_and_replay.py::test_blank_hallucination_tracking PASSED [ 18%]
tests/test_metric_contract_and_replay.py::test_no_feasible_threshold PASSED [ 21%]
tests/test_metric_contract_and_replay.py::test_forged_contract_rejection PASSED [ 25%]
tests/test_metric_contract_and_replay.py::test_strict_cache_rejects_metadata_only PASSED [ 28%]
tests/test_metric_contract_and_replay.py::test_cli_calibrate_threshold_roundtrip PASSED [ 31%]
tests/test_metric_contract_and_replay.py::test_collector_sweep_multistage_integration PASSED [ 34%]
tests/test_metric_contract_and_replay.py::test_astra_bypass_1_stats_candidate_cannot_bypass_state_scores PASSED [ 37%]
tests/test_metric_contract_and_replay.py::test_astra_bypass_2_missing_q_previous_rejected PASSED [ 40%]
tests/test_metric_contract_and_replay.py::test_astra_bypass_3_wrong_shape_q_previous_rejected PASSED [ 43%]
tests/test_metric_contract_and_replay.py::test_astra_bypass_4_fabricated_nan_actual_delta_rejected PASSED [ 46%]
tests/test_metric_contract_and_replay.py::test_t_max_zero_valid_cache PASSED [ 50%]
tests/test_metric_contract_and_replay.py::test_initial_q_previous_mismatch_rejected PASSED [ 53%]
tests/test_metric_contract_and_replay.py::test_state_discontinuity_rejected PASSED [ 56%]
tests/test_metric_contract_and_replay.py::test_score_bounds_out_of_range_rejected PASSED [ 59%]
tests/test_metric_contract_and_replay.py::test_sweep_neutral_margin_mismatch_rejected PASSED [ 62%]
tests/test_metric_contract_and_replay.py::test_cache_metric_space_mismatch_rejected PASSED [ 65%]
tests/test_metric_contract_and_replay.py::test_bogus_statistics_rejected PASSED [ 68%]
tests/test_metric_contract_and_replay.py::test_wave1_rev2_attack_probe_1_normalized_bypass_rejected PASSED [ 71%]
tests/test_metric_contract_and_replay.py::test_wave1_rev2_attack_probe_2_unsupported_cache_schema_version_rejected PASSED [ 75%]
tests/test_metric_contract_and_replay.py::test_wave1_rev2_cache_schema_version_strict_validation PASSED [ 78%]
tests/test_metric_contract_and_replay.py::test_wave1_rev2_valid_collector_t0 PASSED [ 81%]
tests/test_metric_contract_and_replay.py::test_wave1_rev2_collector_and_sweep_blank_data PASSED [ 84%]
tests/test_metric_contract_and_replay.py::test_w1_3_two_case_fixture_volume_identity_and_no_cohort_pooling PASSED [ 87%]
tests/test_metric_contract_and_replay.py::test_w1_3_direct_grouped_statistics_parity PASSED [ 90%]
tests/test_metric_contract_and_replay.py::test_w1_3_collector_rejects_volume_target_contracts PASSED [ 93%]
tests/test_metric_contract_and_replay.py::test_w1_3_stats_bundle_validation_and_rejections PASSED [ 96%]
tests/test_metric_contract_and_replay.py::test_w1_3_distinct_slice_and_derived_volume_metric_metadata PASSED [100%]

============================== 32 passed in 2.92s ==============================
```

### 4.3 Full Project Test Suite Execution
Executed:
```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 rtk python -m pytest -q -p no:cacheprovider tests
```
Output:
```
........................................................................ [ 52%]
................................................................         [100%]
=============================== warnings summary ===============================
tests/test_self_audit_core.py::test_dynamic_window_coordinates_and_backward_are_valid
  /Users/alvinluong/Self-Audit/tests/test_self_audit_core.py:32: UserWarning: Converting a tensor with requires_grad=True to a scalar may lead to unexpected behavior.
  Consider using tensor.detach() first. (Triggered internally at /Users/runner/work/pytorch/pytorch/torch/csrc/autograd/generated/python_variable_methods.cpp:823.)
    assert float(coordinates.min()) >= -1.0

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
136 passed, 1 warning in 5.60s
```

---

## 5. Unresolved Concerns and Non-Claims

1. **Synthetic Fixture Disclaimer:**
   > [!IMPORTANT]
   > All verification and tests in Wave 1.3 were performed using synthetic tensors, mock collectors, and programmatic transition caches. No claims of clinical patient protocol validation or real-checkpoint verification are made from these synthetic test suites.

2. **Preservation of System Invariants:**
   - Strict global gate, stop-gradient, training objectives/losses, proposer, feedback, data/splits, weights, and interpolation remain unchanged.
   - No architectural refactors or training runs were triggered.
   - Git working tree remains uncommitted and unpushed.

3. **Status of Subsequent Waves:**
   - Wave 2 (Geometry, Entropy, Patient-Split isolation) and later waves remain unopened and untouched.
