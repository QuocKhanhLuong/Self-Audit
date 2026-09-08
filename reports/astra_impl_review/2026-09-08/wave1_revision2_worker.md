# Wave 1 Revision 2 Implementation Report: Eliminating Trust Shortcuts and Enforcing Cache Schema Versioning

**Date:** 2026-09-08
**Worker:** AGY Implementer
**Run ID:** `run_688c52658c21`
**Task ID:** `task_ffae34f6055e`
**Dispatch ID:** `ctx_2fb49876baaf`
**Coordinator Terminal:** `term_bb8fdfa8-8643-4694-8cda-d05efb24504b`
**Worker Terminal:** `term_4991158f-02ba-4966-80ad-dfb1b2b608f5`
**Scope:** Wave 1 Revision 2 ONLY (Resolving the two independent coordinator blocker probes: `_normalized: True` trust shortcut removal and exact `cache_schema_version` enforcement)
**Status:** Completed and Verified locally

---

## 1. Executive Summary

1. **What was done:** Resolved both independent verification blockers identified in coordinator messages `msg_af1d81ab7821` and `msg_81ca5a09334f`: (1) eliminated every trust shortcut based on caller-supplied `_normalized` across `validate_and_normalize_transition_cache` and `evaluate_threshold`, ensuring public boundaries revalidate strictly and delegate to a private internal evaluation core `_evaluate_threshold_core`; (2) introduced supported constants `CACHE_SCHEMA_VERSION = 1` and `SUPPORTED_CACHE_SCHEMA_VERSIONS = (1,)`, enforced exact `cache_schema_version` at strict entry (rejecting missing, unknown, bool, or malformed versions), updated `collect_validation_transition_cache` to emit it, wired the CLI (`scripts/calibrate_threshold.py`) through the same boundary, and updated genuine positive fixtures.
2. **What was found and verified:** Both adversarial probe attacks were run directly and confirmed rejected with `ContractMismatchError`:
   - Attack Probe 1 (`validate_and_normalize_transition_cache({"_normalized": True})` and `evaluate_threshold(0.0, transitions={"_normalized": True})`) is strictly rejected with `ContractMismatchError: Cached transitions missing required 'cache_schema_version'. Expected version 1.`;
   - Attack Probe 2 (valid $Q_{\text{previous}}/Q_{\text{candidate}}$ cache with `cache_schema_version: 999`) is strictly rejected with `ContractMismatchError: Unsupported cache_schema_version 999; supported versions: (1,)`;
   - Malformed versions (`True`, `False`, `"1"`, `1.0`, `None`), collector $T=0$ zero-turn cohorts, and blank slice data were verified through 5 new comprehensive regression tests.
3. **What's left / Next steps:** Work is strictly bounded to the two blockers within Wave 1. No new abstractions, objectives, or architectures were added, no training or commits/push were executed, and Waves 2–4 remain unopened. The workspace is verified, cleanly passing the full project test suite (**131 passed, 1 warning in 5.15s**) with clean python byte compilation.

---

## 2. Root Cause Analysis and Architectural Fixes

### Blocker 1: Removal of `_normalized` Trust Shortcuts
- **Prior Flaw:** In Revision 1, `validate_and_normalize_transition_cache` contained `if transitions.get("_normalized") is True: return dict(transitions)`, and `evaluate_threshold` contained `if not transitions.get("_normalized"): ... validate`. Any caller or adversarial probe supplying `{"_normalized": True}` could bypass all schema, contract, state-score, and parity checks.
- **Resolution:**
  - Removed all checks for `_normalized` from `validate_and_normalize_transition_cache` and `evaluate_threshold`.
  - Removed `"_normalized": True` from the normalized cache dictionary emitted by `validate_and_normalize_transition_cache`. No boolean marker replaces it.
  - Separated public boundaries from private core execution:
    - Public `validate_and_normalize_transition_cache` validates unconditionally on every invocation.
    - Public `evaluate_threshold` validates `transitions` strictly at entry via `validate_and_normalize_transition_cache` and rejects untrusted inputs.
    - `sweep_thresholds` normalizes and validates `transitions` once at its public boundary, then invokes private `_evaluate_threshold_core` for each threshold in the grid without redundant boundary revalidation.

### Blocker 2: Strict Cache Schema Versioning
- **Prior Flaw:** `CACHE_SCHEMA_VERSION = 1` was defined but never enforced in `validate_and_normalize_transition_cache`, allowing unversioned or invalid schema versions (such as `cache_schema_version: 999`) to be accepted as valid caches.
- **Resolution:**
  - Exported `CACHE_SCHEMA_VERSION = 1` and `SUPPORTED_CACHE_SCHEMA_VERSIONS = (CACHE_SCHEMA_VERSION,)` in `threshold.py` and `self_audit.evaluation`.
  - Added strict version validation at the entry of `validate_and_normalize_transition_cache`:
    - If `cache_schema_version` is missing in strict mode: raises `ContractMismatchError: Cached transitions missing required 'cache_schema_version'. Expected version 1.`
    - If `cache_schema_version` is a boolean (e.g. `True` / `False`), non-integer, or malformed: raises `ContractMismatchError: Invalid cache_schema_version ...: version must be an integer`
    - If `cache_schema_version` is not in `SUPPORTED_CACHE_SCHEMA_VERSIONS`: raises `ContractMismatchError: Unsupported cache_schema_version ...; supported versions: (1,)`
  - Normalized cache output includes `"cache_schema_version": CACHE_SCHEMA_VERSION`.
  - Updated collector `collect_validation_transition_cache` in `src/self_audit/training/finetune_joint.py` to emit `"cache_schema_version": CACHE_SCHEMA_VERSION` in its `provenance` metadata.
  - Wired CLI `scripts/calibrate_threshold.py` to validate through `validate_and_normalize_transition_cache` at load time before running sweeps.
  - Updated genuine positive test fixtures in `test_self_audit_hardening.py` and `test_metric_contract_and_replay.py` to declare `cache_schema_version`.

---

## 3. Verification Commands and Outputs

### 3.1 Direct Execution of Coordinator Attack Probes

```bash
PYTHONPATH=src python3 -c '
from self_audit.evaluation.threshold import validate_and_normalize_transition_cache, evaluate_threshold
import numpy as np

# Attack Probe 1
try:
    validate_and_normalize_transition_cache({"_normalized": True})
    print("PROBE 1: ACCEPTED (FAIL)")
except Exception as exc:
    print(f"PROBE 1: REJECTED with {type(exc).__name__}: {exc}")

# Attack Probe 1 with version 1 but no keys
try:
    validate_and_normalize_transition_cache({"_normalized": True, "cache_schema_version": 1})
    print("PROBE 1b: ACCEPTED (FAIL)")
except Exception as exc:
    print(f"PROBE 1b: REJECTED with {type(exc).__name__}: {exc}")

# Attack Probe 2
valid_cache_999 = {
    "cache_schema_version": 999,
    "metric_contract": "foreground_dice_exclude_v1",
    "initial_dice": np.array([0.5]),
    "delta_q": np.array([[0.1]]),
    "actual_delta_dice": np.array([[0.1]]),
    "q_previous": np.array([[0.5]]),
    "q_candidate": np.array([[0.6]]),
}
try:
    validate_and_normalize_transition_cache(valid_cache_999)
    print("PROBE 2: ACCEPTED (FAIL)")
except Exception as exc:
    print(f"PROBE 2: REJECTED with {type(exc).__name__}: {exc}")

# Downstream evaluate_threshold with Probe 1
try:
    evaluate_threshold(0.0, transitions={"_normalized": True})
    print("DOWNSTREAM 1: ACCEPTED (FAIL)")
except Exception as exc:
    print(f"DOWNSTREAM 1: REJECTED with {type(exc).__name__}: {exc}")

# Downstream evaluate_threshold with Probe 2
try:
    evaluate_threshold(0.0, transitions=valid_cache_999)
    print("DOWNSTREAM 2: ACCEPTED (FAIL)")
except Exception as exc:
    print(f"DOWNSTREAM 2: REJECTED with {type(exc).__name__}: {exc}")
'
```

**Output:**
```
PROBE 1: REJECTED with ContractMismatchError: Cached transitions missing required 'cache_schema_version'. Expected version 1.
PROBE 1b: REJECTED with ContractMismatchError: Cached transitions do not declare a metric_contract. Unversioned transition caches are rejected by default. Expected contract: foreground_dice_exclude_v1
PROBE 2: REJECTED with ContractMismatchError: Unsupported cache_schema_version 999; supported versions: (1,)
DOWNSTREAM 1: REJECTED with ContractMismatchError: Cached transitions missing required 'cache_schema_version'. Expected version 1.
DOWNSTREAM 2: REJECTED with ContractMismatchError: Unsupported cache_schema_version 999; supported versions: (1,)
```

### 3.2 CLI Boundary Attack Probes

```bash
PYTHONPATH=src python3 -c '
import subprocess, tempfile, torch, numpy as np

with tempfile.NamedTemporaryFile(suffix=".pt") as f1, tempfile.NamedTemporaryFile(suffix=".pt") as f2, tempfile.NamedTemporaryFile(suffix=".json") as out:
    torch.save({"_normalized": True}, f1.name)
    p1 = subprocess.run(["python3", "scripts/calibrate_threshold.py", "--transitions", f1.name, "--output", out.name], capture_output=True, text=True)
    print("CLI PROBE 1 returncode:", p1.returncode)
    print("CLI PROBE 1 stderr:", p1.stderr.strip())

    valid_cache_999 = {
        "cache_schema_version": 999,
        "metric_contract": "foreground_dice_exclude_v1",
        "initial_dice": np.array([0.5]),
        "delta_q": np.array([[0.1]]),
        "actual_delta_dice": np.array([[0.1]]),
        "q_previous": np.array([[0.5]]),
        "q_candidate": np.array([[0.6]]),
    }
    torch.save(valid_cache_999, f2.name)
    p2 = subprocess.run(["python3", "scripts/calibrate_threshold.py", "--transitions", f2.name, "--output", out.name], capture_output=True, text=True)
    print("CLI PROBE 2 returncode:", p2.returncode)
    print("CLI PROBE 2 stderr:", p2.stderr.strip())
'
```

**Output:**
```
CLI PROBE 1 returncode: 1
CLI PROBE 1 stderr: Error: Cached transitions missing required 'cache_schema_version'. Expected version 1.
CLI PROBE 2 returncode: 1
CLI PROBE 2 stderr: Error: Unsupported cache_schema_version 999; supported versions: (1,)
```

### 3.3 Focused Unit & Regression Tests

```bash
PYTHONPATH=src:. pytest tests/test_metric_contract_and_replay.py tests/test_self_audit_hardening.py
```

**Output:**
```
============================= test session starts ==============================
platform darwin -- Python 3.11.16, pytest-9.1.1, pluggy-1.6.0
rootdir: /Users/alvinluong/Self-Audit
plugins: anyio-4.14.2
collected 39 items

tests/test_metric_contract_and_replay.py ...........................     [ 69%]
tests/test_self_audit_hardening.py ............                          [100%]

============================== 39 passed in 3.65s ==============================
```

### 3.4 Full Project Test Suite

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest -q -p no:cacheprovider tests
```

**Output:**
```
........................................................................ [ 54%]
...........................................................              [100%]
=============================== warnings summary ===============================
tests/test_self_audit_core.py::test_dynamic_window_coordinates_and_backward_are_valid
  /Users/alvinluong/Self-Audit/tests/test_self_audit_core.py:32: UserWarning: Converting a tensor with requires_grad=True to a scalar may lead to unexpected behavior.
  Consider using tensor.detach() first. (Triggered internally at /Users/runner/work/pytorch/pytorch/torch/csrc/autograd/generated/python_variable_methods.cpp:823.)
    assert float(coordinates.min()) >= -1.0

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
131 passed, 1 warning in 5.15s
```

### 3.5 Python Byte Compilation

```bash
python3 -m py_compile src/self_audit/evaluation/threshold.py src/self_audit/evaluation/__init__.py src/self_audit/training/finetune_joint.py scripts/calibrate_threshold.py scripts/cache_validation_transitions.py tests/test_metric_contract_and_replay.py tests/test_self_audit_hardening.py
```

**Exit code:** `0` (clean compilation across all modified and test files).

---

## 4. Scope and File Modifications Summary

All changes were strictly contained within the requested ownership boundaries:

1. `src/self_audit/evaluation/threshold.py`:
   - Removed all `_normalized` bypass checks and output markers.
   - Enforced exact `cache_schema_version` checking at entry; rejected missing, unknown, bool, and malformed versions.
   - Refactored evaluation simulation into private `_evaluate_threshold_core`; public `evaluate_threshold` revalidates strictly at the boundary; `sweep_thresholds` normalizes once and calls `_evaluate_threshold_core`.
   - Exported `CACHE_SCHEMA_VERSION`, `SUPPORTED_CACHE_SCHEMA_VERSIONS`, `json_safe`, and `validate_and_normalize_transition_cache` in `__all__`.
2. `src/self_audit/evaluation/__init__.py`:
   - Imported and exported `CACHE_SCHEMA_VERSION` and `SUPPORTED_CACHE_SCHEMA_VERSIONS`.
3. `src/self_audit/training/finetune_joint.py`:
   - Imported `CACHE_SCHEMA_VERSION` and emitted `"cache_schema_version": CACHE_SCHEMA_VERSION` in `collect_validation_transition_cache` provenance metadata.
4. `scripts/calibrate_threshold.py`:
   - Wired CLI load path to run `validate_and_normalize_transition_cache(..., strict_contract=True)`, ensuring the CLI entry point uses the exact same boundary.
5. `tests/test_self_audit_hardening.py`:
   - Updated positive transition fixtures to declare `cache_schema_version: 1`.
6. `tests/test_metric_contract_and_replay.py`:
   - Updated positive transition fixtures to declare `cache_schema_version: CACHE_SCHEMA_VERSION`.
   - Added 5 new regression tests in Section 10 (`test_wave1_rev2_attack_probe_1_normalized_bypass_rejected`, `test_wave1_rev2_attack_probe_2_unsupported_cache_schema_version_rejected`, `test_wave1_rev2_cache_schema_version_strict_validation`, `test_wave1_rev2_valid_collector_t0`, `test_wave1_rev2_collector_and_sweep_blank_data`).
