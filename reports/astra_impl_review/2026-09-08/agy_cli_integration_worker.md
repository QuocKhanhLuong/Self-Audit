# AGY CLI Integration Worker Report — Proposal 1 Subprocess Regressions

- **Date:** 2026-09-08
- **Implementer:** AGY via Orca (`task_d2e068c8d1f7` / `ctx_084f9722dc62` on `term_66f7758d-b4f2-491f-b8bc-1301a16748eb`)
- **Coordinator Terminal:** `term_bb8fdfa8-8643-4694-8cda-d05efb24504b`
- **Ownership:** Exclusive ownership of `tests/test_proposal1_cli_integration.py` and `reports/astra_impl_review/2026-09-08/agy_cli_integration_worker.md`. No production files modified; no overlap with bank exporter (W4) or producer/calibration files (W3.1/W3.2).

---

## 1. Executive Summary

Implemented actual public-CLI subprocess end-to-end regression tests across the Proposal 1 pipeline (`cache_validation_transitions.py` -> `calibrate_threshold.py` -> `audit_checkpoint.py`) using temporary synthetic NPY volumes, manifest, and tiny CPU-only saved checkpoint. Verified that positive matching binds exact weight digests and metric contracts, same-filename replaced weights fail evaluation before any calibrated report is emitted, CLI tau overrides cannot bypass invalid calibration, and authorized disjoint evaluation cohorts succeed under strict patient disjointness. All 4 focused integration tests pass; full test suite achieves 426 passed with 3 pre-existing/in-flight dependency failures in concurrent worker files.

---

## 2. Regression Scope and Implementation

Implemented in `tests/test_proposal1_cli_integration.py`:

1. **`test_cli_pipeline_positive_matching_end_to_end`**:
   - Executes `scripts/cache_validation_transitions.py` via `subprocess.run`: validates dataset splits, binds checkpoint state, collects validation transitions, stamps lineage with exact `state_digest`, and outputs `transitions.pt`.
   - Executes `scripts/calibrate_threshold.py`: sweeps tau grid, verifies lineage completeness and contract semantics, and emits schema-v2 `calibration.json` with `validity: "diagnostic_only"`.
   - Executes `scripts/audit_checkpoint.py`: loads model, binds evaluation checkpoint, derives expected lineage from live objects, verifies calibration lineage, executes GT firewall and probe diagnostics, and emits `audit_report.json`.
   - Asserts:
     - `diagnostic_schema_version == 1`
     - `evidence_class == "diagnostic_only"`
     - `tau_accept_source == "calibration_artifact"`
     - `threshold_resolution["lineage_verified"] is True`
     - `checkpoint_binding["state_digest"] == expected_state_digest`
     - `checkpoint_binding["checkpoint_sha256"] == expected_file_sha256`
     - `expected_calibration_lineage["semantics"]["metric_contract"] == "foreground_dice_exclude_v1"`
     - `metric_space == "slice_proxy"`
     - `cohort_policy["role"] == "calibration"`
     - `gt_firewall["gt_firewall/passed"] is True`
     - `required_keys_present["ok"] is True`

2. **`test_cli_pipeline_replaced_weights_fails_before_calibrated_report`**:
   - Calibrates against original weights `checkpoint.pt`.
   - Replaces weights at `checkpoint.pt` with a different randomly initialized state (same filename).
   - Invokes `scripts/audit_checkpoint.py`.
   - Asserts non-zero exit code (`returncode != 0`), `LineageMismatchError` in stderr (`refusing to apply its threshold`), and asserts that no report artifact was emitted.

3. **`test_cli_tau_override_cannot_bypass_invalid_calibration`**:
   - Targets mismatched weights with `--tau_accept 0.15` and `--allow_tau_override`.
   - Asserts non-zero exit code and verifies that CLI tau override cannot bypass lineage verification of an invalid calibration artifact before report generation.

4. **`test_cli_disjoint_independent_evaluation_cohort_positive`**:
   - Calibrates on original `val` split (`patient002`).
   - Evaluates on a disjoint manifest where `val` contains `patient003` (strictly patient-disjoint from `patient002`).
   - Passes `--cohort_role independent_evaluation`, `--authorized_cohort_signature <auth_sig>`, and `--authorized_cohort_split val`.
   - Asserts exit code 0, `cohort_policy["role"] == "independent_evaluation"`, `calibration_lineage_verification["verified"] is True`, and `calibration_lineage_verification["cohort"]["patients_disjoint"] is True`.

---

## 3. Exact Outputs and Verification

### 3.1 Focused Integration Suite
```
$ PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:src OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 rtk pytest -v -p no:cacheprovider tests/test_proposal1_cli_integration.py
============================= test session starts ==============================
platform darwin -- Python 3.11.16, pytest-9.1.1, pluggy-1.6.0 -- /Users/alvinluong/miniforge3/bin/python
rootdir: /Users/alvinluong/Self-Audit
plugins: anyio-4.8.0
collected 4 items

tests/test_proposal1_cli_integration.py::test_cli_pipeline_positive_matching_end_to_end PASSED [ 25%]
tests/test_proposal1_cli_integration.py::test_cli_pipeline_replaced_weights_fails_before_calibrated_report PASSED [ 50%]
tests/test_proposal1_cli_integration.py::test_cli_tau_override_cannot_bypass_invalid_calibration PASSED [ 75%]
tests/test_proposal1_cli_integration.py::test_cli_disjoint_independent_evaluation_cohort_positive PASSED [100%]

============================== 4 passed in 15.65s ==============================
```

### 3.2 Full Repository Suite
```
$ PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:src OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 rtk pytest -q -p no:cacheprovider tests
3 failed, 426 passed, 1 warning in 82.57s
```

All 3 failed tests are in existing / concurrent worker files outside this task's ownership:
1. `tests/test_calibration_lineage.py::test_changed_scoped_source_invalidates_the_artifact`: In-flight regex expectation mismatch during W3.2 refactoring of `source_content_signature`.
2. `tests/test_metric_contract_and_replay.py::test_cli_calibrate_threshold_roundtrip`: Wave 1 test that fabricates a legacy transition cache without a `lineage` block (previously reported by W3.1 in `msg_042855adb497`).
3. `tests/test_remediation_evaluation.py::test_diagnostic_cli_consumes_the_saved_tau_under_the_documented_precedence`: Passes no expected lineage to `resolve_tau_accept`, intentionally fail-closed per coordinator review note `msg_ddbedc5c0e10`.

### 3.3 Source Compilation and Whitespace Cleanliness
```
$ python -m compileall -q src scripts tests; echo compileall_rc=$?
compileall_rc=0

$ git diff --check; echo diff_check_rc=$?
diff_check_rc=0
```

---

## 4. Coordination and Concurrent Interfaces

- **W3.2 Bug Notification:** During execution of `scripts/calibrate_threshold.py`, identified an unhandled `NameError: name '_REPO_ROOT' is not defined` and missing `MEASUREMENT_SOURCE_FILES` introduced in `calibration_lineage.py`. Immediately reported to coordinator (`msg_23fead47bdd2`) and W3.2 terminal (`msg_dc9b2ae946d3`). W3.2 concurrently removed the obsolete `measurement_code_identity` call from `src/self_audit/evaluation/threshold.py:1323`.
- **Interface Stability:** All Proposal 1 CLI scripts (`cache_validation_transitions.py`, `calibrate_threshold.py`, `audit_checkpoint.py`) were invoked strictly through their public command-line arguments without internal monkeypatching or synthetic bypasses.
- **Invariants Preserved:** No production code, architectures, loss objectives, configs, tracked splits, or existing tests were modified or weakened. All temporary test artifacts use isolated pytest `tmp_path` fixtures.
