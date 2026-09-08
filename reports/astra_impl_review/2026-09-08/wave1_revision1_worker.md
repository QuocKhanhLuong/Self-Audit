# Wave 1 Revision Implementation Report: Strict Transition Cache Normalization and Anti-Bypass Hardening

**Date:** 2026-09-08
**Worker:** AGY Implementer
**Run ID:** `run_688c52658c21`
**Task ID:** `task_e06e6582ab7f` (Wave 1 Revision following `task_0cbe9cbfc7c7` / `task_ff3aa716046b`)
**Dispatch ID:** `ctx_5a9fc41bd585`
**Scope:** Wave 1 REVISION ONLY (Resolving Astra 4 bypasses, strict transition cache normalization boundary, and negative regressions)
**Status:** Completed and Verified locally

---

## 1. Executive Summary

1. **What was done:** Resolved all 4 independent verification bypasses identified during Astra's independent Gate B review of Wave 1 by introducing a unified, centralized normalization and validation boundary `validate_and_normalize_transition_cache` in `src/self_audit/evaluation/threshold.py`, eliminating mixed-metric fallback arithmetic across all evaluation paths, and wiring strict neutral margin and metric space enforcement through the calibration CLI.
2. **What was found and verified:** All 4 Astra bypass probes are now strictly rejected at the cache boundary before replay execution:
   - `stats_candidate: []` incomplete statistics are rejected, and state score presence is mandatory;
   - Missing `q_previous` raises `ContractMismatchError`;
   - Incompatible array shapes (e.g. `q_previous` shape `(1, 2)` vs `q_candidate` shape `(1, 1)`) raise `ContractMismatchError`;
   - Inconsistent validity masks (fabricated NaN deltas for finite state scores) raise `ContractMismatchError`.
   Added 11 new focused regression tests in `tests/test_metric_contract_and_replay.py` (22 total in file), covering all 4 bypasses, $T=0$ valid caches, state continuity violations, out-of-bounds scores, and neutral margin/metric space conflicts. The complete project test suite passes cleanly: **126 passed, 1 warning** with zero py_compile errors.
3. **What's left / Next steps:** Work is strictly bounded to the Wave 1 revision. No Git commits or pushes have been executed, no training loops were run, and Wave 2 has not been opened. The working tree is prepared for Astra's independent re-evaluation.

---

## 2. Astra Review Findings & Root-Cause Resolutions

During independent Gate B evaluation of the initial Wave 1 implementation, Astra tested cache base `initial=.5, delta_q=1, actual_delta=.133333333, contract=foreground_dice_exclude_v1` against 4 adversarial probe variants:

| # | Probe Variant | Pre-Revision Behavior | Root Cause | Post-Revision Behavior |
|---|---|---|---|---|
| **1** | `stats_candidate: []` | **ACCEPTED**; final=`0.633333333` | Presence of key alone was treated as `has_state_scores`, but `cand_scores` was None, triggering fallback mixed arithmetic `final += delta_turn`. | **REJECTED**: `validate_and_normalize_transition_cache` requires non-empty, well-formed 3D sufficient statistics and strictly requires explicit state scores (`q_candidate`, `q_previous`). `evaluate_threshold` unconditionally forbids mixed delta addition when a contract is present. |
| **2** | `q_candidate: [[.4]]`, missing `q_previous` | **ACCEPTED**; `harmful_total=0` | Parity verification guard was `if q_previous is not None and cand_scores is not None:`. When `q_previous` was omitted, the check was skipped silently. | **REJECTED**: `q_previous` is mandatory under contract validation; omission raises `ContractMismatchError`. |
| **3** | `q_previous` shape `(1, 2)`, candidate `(1, 1)` | **ACCEPTED**; `harmful_total=0` | Parity verification checked `if prev_scores.shape == quality.shape:` as a conditional guard that silently skipped parity on mismatch rather than raising. | **REJECTED**: Normalization boundary strictly requires 2D array shapes `(N, T)` matching across all transition fields. In `evaluate_threshold`, any shape mismatch immediately raises `ContractMismatchError`. |
| **4** | `prev=.5, cand=.4, actual_delta=NaN` | **ACCEPTED**; `harmful_total=0` | Mask check `both_fin = np.isfinite(diff) & np.isfinite(actual)` evaluated to empty boolean mask, hiding degradation and fabricated NaN delta. | **REJECTED**: Validity mask consistency strictly enforces that whenever both $q_{\text{previous}}$ and $q_{\text{candidate}}$ are finite, `actual_delta_dice` must be finite and match their difference within $10^{-4}$; conversely, if either state is NaN, actual delta must be NaN. |

---

## 3. Architecture of Strict Validation Boundary

All cache validation and normalization has been centralized into `validate_and_normalize_transition_cache` in `src/self_audit/evaluation/threshold.py`, which is invoked unconditionally before threshold sweeping and replay:

```python
def validate_and_normalize_transition_cache(
    transitions: Mapping[str, Any],
    expected_contract: MetricContract | str | None = None,
    expected_metric_space: str | None = None,
    expected_neutral_margin: float | None = None,
) -> dict[str, Any]:
```

### Core Invariants Enforced:

1. **Contract Metadata Validation:**
   - Caches must contain a valid `metric_contract`.
   - Contract is validated via `validate_contract`, ensuring all mandatory schema keys exist and rejecting forged contracts.
   - Validates that contract name and properties match canonical specifications.

2. **Dimension and Shape Normalization:**
   - Cohort size $N$ derived from 1D `initial_dice` of shape `(N,)`.
   - $T$ turns derived from 2D transition arrays.
   - All transition fields (`delta_q`, `actual_delta_dice`, `q_previous`, `q_candidate`) must be 2D arrays of exact shape `(N, T)`. Flat 1D arrays or ragged/mismatched dimensions are rejected with `ContractMismatchError`.
   - Explicitly supports empty transition cohorts ($T=0$) with shape `(N, 0)` for zero-turn evaluations.

3. **Mandatory State Scores:**
   - Both `q_previous` and `q_candidate` are strictly required.
   - Merely providing sufficient statistics keys (e.g. `stats_candidate: []` or empty arrays) does not bypass the requirement for valid state scores.

4. **Numerical Score Bounds:**
   - Finite Dice scores in `initial_dice`, `q_previous`, and `q_candidate` must reside within $[-10^{-6}, 1.0 + 10^{-6}]$.
   - Finite policy deltas `delta_q` must be strictly finite numbers (no NaN or Inf).

5. **Finite/Undefined Mask Consistency:**
   - For every transition $(i, t)$:
     - If both $q_{\text{previous}}[i, t]$ and $q_{\text{candidate}}[i, t]$ are finite, `actual_delta_dice` must be finite and satisfy $|(q_{\text{cand}} - q_{\text{prev}}) - \Delta| \le 10^{-4}$. Fabricated NaNs are rejected.
     - If either $q_{\text{previous}}[i, t]$ or $q_{\text{candidate}}[i, t]$ is NaN, `actual_delta_dice` must be NaN. Fabricated finite deltas are rejected.

6. **State Machine Invariants:**
   - **Turn 0 Invariant:** $q_{\text{previous}}[:, 0]$ must equal `initial_dice[:]` (both NaN or both finite within $10^{-4}$).
   - **State Continuity Invariant:** For turns $t \ge 1$, $q_{\text{previous}}[:, t]$ must equal $q_{\text{candidate}}[:, t-1]$ (both NaN or both finite within $10^{-4}$). Discontinuous state transitions are rejected.

7. **Sufficient Statistics Validation:**
   - If sufficient statistics are present, they must be dictionaries containing valid 3D integer/float arrays of shape `(N, T, num_classes)` with non-negative counts. Empty lists `[]` or corrupted structures raise `ContractMismatchError`.

8. **Metric Space & Neutral Margin Compatibility:**
   - Verified against `expected_metric_space` and `expected_neutral_margin` (within $10^{-7}$). Mismatches raise `ContractMismatchError`.

---

## 4. Anti-Bypass Hardening in Evaluation and CLI

### 4.1 Evaluation Pipeline (`src/self_audit/evaluation/threshold.py`)
- In `evaluate_threshold`:
  - When a dictionary `transitions` is supplied with a contract, it is normalized through `validate_and_normalize_transition_cache`.
  - When a contract is active, fallback mixed delta addition (`final += delta_turn`) is strictly blocked. If `cand_scores` or `q_previous` is None under a contract, `evaluate_threshold` raises `ContractMismatchError`.
  - Shape mismatches between `prev_scores` and `quality` unconditionally raise `ContractMismatchError` (no conditional skipping).
  - Fabricated NaNs in `actual_delta` are caught and raise `ContractMismatchError`.

### 4.2 Module Exports (`src/self_audit/evaluation/__init__.py`)
- Exported `validate_and_normalize_transition_cache` in `__init__.py` and included it in `__all__`.

### 4.3 Calibration CLI (`scripts/calibrate_threshold.py`)
- Wired `--neutral_margin` argument through to `sweep_thresholds`.
- Persisted `selected["neutral_margin"] = args.neutral_margin` in calibration artifact.
- Cleanly catches `ContractMismatchError`, `NoFeasibleThresholdError`, and `ValueError`, emitting an informative message to stderr and exiting with code 1.

---

## 5. Verification and Test Evidence

### 5.1 Focused Metric Contract & Regression Suite (`tests/test_metric_contract_and_replay.py`)

All 22 tests in `tests/test_metric_contract_and_replay.py` pass cleanly:

| Test Case | Description | Result |
|---|---|---|
| `test_p0_01_sign_reversal_eliminated` | Canonical exclude prevents false-positive acceptance under legacy 1.0 inflation | **PASSED** |
| `test_direct_vs_cache_replay_parity` | Exact agreement between direct model inference and cached replay | **PASSED** |
| `test_mixed_accept_prefix` | Halting and state-freezing across variable acceptance depths | **PASSED** |
| `test_volume_aggregation_patient_grouping` | Patient-wise grouping and 3D volume aggregation | **PASSED** |
| `test_contract_mismatch_rejection` | Core contract attribute mismatch rejection | **PASSED** |
| `test_blank_hallucination_tracking` | Blank $A_0$ transition tracking and hallucination counting | **PASSED** |
| `test_no_feasible_threshold` | `NoFeasibleThresholdError` under unachievable safety constraints | **PASSED** |
| `test_forged_contract_rejection` | Tampered/forged contract schema rejection | **PASSED** |
| `test_strict_cache_rejects_metadata_only` | Metadata-only cache rejection without state scores | **PASSED** |
| `test_cli_calibrate_threshold_roundtrip` | CLI subprocess roundtrip, JSON null compliance, and metric space check | **PASSED** |
| `test_collector_sweep_multistage_integration` | End-to-end collector -> cache -> sweep -> select pipeline | **PASSED** |
| `test_astra_bypass_1_stats_candidate_cannot_bypass_state_scores` | **Bypass 1:** Rejection of `stats_candidate: []` attempting mixed arithmetic | **PASSED** |
| `test_astra_bypass_2_missing_q_previous_rejected` | **Bypass 2:** Rejection of missing `q_previous` | **PASSED** |
| `test_astra_bypass_3_wrong_shape_q_previous_rejected` | **Bypass 3:** Rejection of shape mismatch `(1, 2)` vs `(1, 1)` | **PASSED** |
| `test_astra_bypass_4_fabricated_nan_actual_delta_rejected` | **Bypass 4:** Rejection of fabricated NaN delta for finite states | **PASSED** |
| `test_t_max_zero_valid_cache` | Valid normalization and evaluation for zero-turn cache ($T=0$) | **PASSED** |
| `test_initial_q_previous_mismatch_rejected` | Rejection of Turn 0 invariant violation ($q_{\text{prev}}[:, 0] \neq \text{initial}$) | **PASSED** |
| `test_state_discontinuity_rejected` | Rejection of sequential continuity violation ($q_{\text{prev}}[:, t] \neq q_{\text{cand}}[:, t-1]$) | **PASSED** |
| `test_score_bounds_out_of_range_rejected` | Rejection of out-of-bounds Dice scores ($> 1.0$ or $< 0.0$) | **PASSED** |
| `test_sweep_neutral_margin_mismatch_rejected` | Rejection of cache neutral margin mismatch against contract | **PASSED** |
| `test_cache_metric_space_mismatch_rejected` | Rejection of cache metric space mismatch against contract | **PASSED** |
| `test_bogus_statistics_rejected` | Rejection of negative or malformed sufficient statistics | **PASSED** |

**Command Output:**
```
$ PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest -v -p no:cacheprovider tests/test_metric_contract_and_replay.py
============================= test session starts ==============================
platform darwin -- Python 3.11.16, pytest-9.1.1, pluggy-1.6.0 -- /Users/alvinluong/miniforge3/bin/python
rootdir: /Users/alvinluong/Self-Audit
plugins: anyio-4.14.2
collecting ... collected 22 items

tests/test_metric_contract_and_replay.py::test_p0_01_sign_reversal_eliminated PASSED [  4%]
tests/test_metric_contract_and_replay.py::test_direct_vs_cache_replay_parity PASSED [  9%]
tests/test_metric_contract_and_replay.py::test_mixed_accept_prefix PASSED [ 13%]
tests/test_metric_contract_and_replay.py::test_volume_aggregation_patient_grouping PASSED [ 18%]
tests/test_metric_contract_and_replay.py::test_contract_mismatch_rejection PASSED [ 22%]
tests/test_metric_contract_and_replay.py::test_blank_hallucination_tracking PASSED [ 27%]
tests/test_metric_contract_and_replay.py::test_no_feasible_threshold PASSED [ 31%]
tests/test_metric_contract_and_replay.py::test_forged_contract_rejection PASSED [ 36%]
tests/test_metric_contract_and_replay.py::test_strict_cache_rejects_metadata_only PASSED [ 40%]
tests/test_metric_contract_and_replay.py::test_cli_calibrate_threshold_roundtrip PASSED [ 45%]
tests/test_metric_contract_and_replay.py::test_collector_sweep_multistage_integration PASSED [ 50%]
tests/test_metric_contract_and_replay.py::test_astra_bypass_1_stats_candidate_cannot_bypass_state_scores PASSED [ 54%]
tests/test_metric_contract_and_replay.py::test_astra_bypass_2_missing_q_previous_rejected PASSED [ 59%]
tests/test_metric_contract_and_replay.py::test_astra_bypass_3_wrong_shape_q_previous_rejected PASSED [ 63%]
tests/test_metric_contract_and_replay.py::test_astra_bypass_4_fabricated_nan_actual_delta_rejected PASSED [ 68%]
tests/test_metric_contract_and_replay.py::test_t_max_zero_valid_cache PASSED [ 72%]
tests/test_metric_contract_and_replay.py::test_initial_q_previous_mismatch_rejected PASSED [ 77%]
tests/test_metric_contract_and_replay.py::test_state_discontinuity_rejected PASSED [ 81%]
tests/test_metric_contract_and_replay.py::test_score_bounds_out_of_range_rejected PASSED [ 86%]
tests/test_metric_contract_and_replay.py::test_sweep_neutral_margin_mismatch_rejected PASSED [ 90%]
tests/test_metric_contract_and_replay.py::test_cache_metric_space_mismatch_rejected PASSED [ 95%]
tests/test_metric_contract_and_replay.py::test_bogus_statistics_rejected PASSED [100%]

============================== 22 passed in 2.05s ==============================
```

### 5.2 Full Repository Test Suite

```
$ PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest -q -p no:cacheprovider tests
........................................................................ [ 57%]
......................................................                   [100%]
126 passed, 1 warning in 4.89s
```

### 5.3 Static Bytecode Compilation

```
$ python -m py_compile $(find src scripts tests -name "*.py")
# Exited 0 with no errors or warnings
```

---

## 6. Commitments and Scope Discipline

- **Wave 1 Bounded:** Work strictly addressed Wave 1 transition cache normalization and anti-tampering verification. No Wave 2 changes or tasks were initiated.
- **No Training Loss / Architecture Alteration:** The model state machine, architecture, and training loss calculations (`multiclass_dice`, `build_transition_targets`) remain preserved bit-identical.
- **No Git Commits / Remote Operations:** No git commits or pushes were performed.
- **Handoff:** The patch is fully verified locally and ready for Astra independent re-evaluation.
