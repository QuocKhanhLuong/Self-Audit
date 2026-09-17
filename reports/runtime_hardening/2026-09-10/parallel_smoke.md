# Parallel AGY Bounded End-to-End Smokes Delivery

**Worker:** Parallel AGY Worker (`term_0b602e97-c36d-4c9c-ab7c-cd01f3f7b7af`)
**Task ID:** `task_fe9b43cd3244` (continuation from `task_bd2b9879a663`)
**Dispatch ID:** `ctx_ee963b6e78e5` (continuation from `ctx_d65d64afe46d`)
**Run ID:** `run_6f2a154bf6dc`
**Base HEAD:** `25f429a46255ceb59997ae5178939d5d40b2ea27`
**Reference Environment:** `/private/tmp/self-audit-torch241/bin/python` (Python 3.10.21, PyTorch 2.4.1 CPU)
**Execution Flags:** `PYTHONPATH=src:. PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1`

---

## 1. Executive Summary & Revision Resolution

Following review checkpoint feedback from Astra, the bounded CLI integration smoke suite has been updated, cleaned, and verified directly against unpatched production code.

### 1.1 Key Adjustments Made
1. **Withdrew "No-Mocks" Claim:** Clarified that the smoke tests use isolated, deterministic synthetic test fixtures (generated via local `np.random.default_rng` seeds, scaled model dimensions, and short 3-interval schedules). Dataloaders, model inference, metric evaluators, and filesystem serializers run real unmocked logic.
2. **Removed Compatibility Shim:** With `compute_config_signature` now correctly imported from `self_audit.training.unified_trainer` in `scripts/train_self_audit.py`, `tests/runtime_smoke_shim/sitecustomize.py` and its directory were completely deleted. Tests run against unpatched `PYTHONPATH=src:.`.
3. **Subprocess Portability:** Subprocesses strictly invoke `sys.executable` (rather than hardcoded environment paths), guaranteeing honest reporting of the active test environment.
4. **Explicit CPU Flag:** Added `--device cpu` explicitly to both external evaluation subprocess invocations in `test_external_labels_cannot_tune_model`.
5. **Deterministic Generators:** All synthetic volumes and masks across ACDC and M&Ms fixtures now use local `np.random.default_rng` instances with fixed seeds.

Zero production files or existing test files were modified.

---

## 2. Test Execution & Evidence

### 2.1 Test Results

Test suite: `tests/test_runtime_pipeline_smoke.py`
Target command:
```bash
rtk proxy env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:. OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  /private/tmp/self-audit-torch241/bin/python -m pytest tests/test_runtime_pipeline_smoke.py -v
```

Execution output:
```text
============================= test session starts ==============================
platform darwin -- Python 3.10.21, pytest-9.1.1, pluggy-1.6.0 -- /private/tmp/self-audit-torch241/bin/python
cachedir: .pytest_cache
rootdir: /Users/alvinluong/Self-Audit
plugins: anyio-4.15.1
collecting ... collected 3 items

tests/test_runtime_pipeline_smoke.py::test_bounded_pipeline_e2e_acdc_to_mnms PASSED [ 33%]
tests/test_runtime_pipeline_smoke.py::test_external_labels_cannot_tune_model PASSED [ 66%]
tests/test_runtime_pipeline_smoke.py::test_native_mnms_tiny_config_smoke PASSED [100%]

============================== 3 passed in 21.29s ==============================
```

### 2.2 Test Coverage Matrix

| Test Function | Target CLI Entrypoints | Verified Contract & Invariants |
| :--- | :--- | :--- |
| `test_bounded_pipeline_e2e_acdc_to_mnms` | `scripts/train_self_audit.py`<br>`scripts/export_transition_bank.py`<br>`scripts/evaluate_external_mnms.py` | Full 3-interval schedule executed end-to-end (`annotation_bootstrap`, `auditor_training`, `joint_self_audit`). Produced `best.pt`, `last.pt`, `validation_transitions.pt`, `calibration.json`, and `pipeline_report.json`. Validated strict standard JSON, completion flags, hash matching between `best.pt` and artifact bindings, transition bank export, and external M&Ms evaluation under calibrated $\tau$. |
| `test_external_labels_cannot_tune_model` | `scripts/evaluate_external_mnms.py` | Executed against synthetic M&Ms test set, then re-executed with zeroed ground-truth masks with `--device cpu` explicitly set. Model bytes and live state digest verified bit-for-bit invariant before and after inference. |
| `test_native_mnms_tiny_config_smoke` | `scripts/train_self_audit.py` with `configs/self_audit_full_mnms.yaml` | Bounded native M&Ms training execution verifying config parsing, patient-disjoint split validation, and strict JSON output under PyTorch 2.4.1. |

---

## 3. Verified System Invariants

1. **Strict JSON Parsing:** All generated artifacts (`pipeline_report.json`, `calibration.json`, `transition_bank.json`, `external_mnms.json`) parse strictly under `read_json_artifact` with `allow_nan=False` and explicit rejection of non-standard constants (`NaN`, `Infinity`).
2. **Deterministic Checkpoint Binding:**
   - SHA-256 of `weights/best.pt` matches `checkpoint_sha256` recorded in `pipeline_report.json:checkpoint_binding`.
   - `export_transition_bank.py` records the exact same `checkpoint_sha256` in `generation.checkpoint`.
   - `evaluate_external_mnms.py` records the exact same `checkpoint_sha256` in `checkpoint_binding`.
3. **Calibration & Protocol Lineage:**
   - Post-training calibration runs only after the complete run commits without errors.
   - `calibration.json` records `tau_accept`, `neutral_margin`, and full roundtrip-verified lineage.
   - `pipeline_report.json:calibration.calibrated_tau` is identical to `calibration.json:tau_accept`.
4. **External Independence & Label Isolation:**
   - `evaluate_external_mnms.py` enforces `evidence_class == "independent_external_evaluation"`, `dataset == "mnms"`, and `training_dataset == "acdc"`.
   - Ground-truth masks are consumed solely by `evaluate_volume_cohort` for metric calculation and never modify live weights.
   - Checkpoint bytes and live state digest remain bit-for-bit invariant across evaluation runs.
5. **Class Mapping and Shape Invariants:**
   - External M&Ms evaluation preserves `raw_to_acdc` (`{0: 0, 1: 3, 2: 2, 3: 1}`) mapping.
   - Uniform volume grids and network input grids (`[32, 32]`) are strictly validated and tracked in the external evaluation report.

---

## 4. Deliverables

* [`tests/test_runtime_pipeline_smoke.py`](file:///Users/alvinluong/Self-Audit/tests/test_runtime_pipeline_smoke.py): Portable real subprocess CLI integration test suite using `sys.executable` and local deterministic generators.
* [`reports/runtime_hardening/2026-09-10/parallel_smoke.md`](file:///Users/alvinluong/Self-Audit/reports/runtime_hardening/2026-09-10/parallel_smoke.md): This durable delivery report.
