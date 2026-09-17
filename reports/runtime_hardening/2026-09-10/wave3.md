# Wave 3 Delivery: Lifecycle and Truthful Commit Reporting

**Execution Date:** 2026-09-10
**Task ID:** `task_4a01df1ac1a4`
**Dispatch ID:** `ctx_f8f76dcbfc37`
**Coordinator Handle:** `term_dc6a165c-3cfb-436b-afce-b65c544ae347`
**Assignee Handle:** `term_a273cb57-bbde-468e-97ed-39e2bc19c164`
**Status:** Completed (Astra REVISE Gate fully resolved)
**Target Environment:** PyTorch 2.4.1 (Python 3.10.21, macOS Darwin)

---

## 1. Executive Summary & Root Cause Analysis

Wave 3 establishes exception-safe lifecycle handling, truthful commit reporting, per-epoch progress persistence, failure artifact schema compliance, and telemetry finalization across `UnifiedTrainer` and `scripts/train_self_audit.py`.

### 1.1 Astra REVISE Gate Resolutions

1. **Independent `finally` Cleanup (Item 1):**
   - *Issue:* Outer `try` blocks catching metadata errors could still skip `logger.finish` if earlier payload preparation (e.g. `str(exc)`) threw an exception.
   - *Fix:* Telemetry cleanup (`logger.set_summary` and `logger.finish`) is enclosed in a dedicated `finally:` block completely decoupled from payload preparation. Protected `str(exc)` with fallback `<unformattable ...>`. Primary exceptions always propagate without masking.
2. **Strict Provenance Integrity for `config_signature` (Item 2):**
   - *Issue:* Weakening `config_signature` to catch exceptions globally and return `None` undermined provenance verification during normal execution.
   - *Fix:* Restored strict `config_signature` property during normal execution, throwing if configuration is malformed. Exceptions during signature generation are caught defensively only within `record_failure`.
3. **Unconditional Failure Artifact Preservation (Item 3 & Coordinator Review):**
   - *Issue:* Pre-existing `failure.json` from earlier runs or unhandled resume failures (e.g. `ValueError`) could be overwritten by subsequent failure writers.
   - *Fix:* In both `UnifiedTrainer.record_failure` and `scripts/train_self_audit.py:_write_pre_init_failure`, whenever canonical `failure.json` already exists in `report_dir`, write to `report_dir / f"failure_attempt_{uuid.uuid4().hex[:8]}.json"` unconditionally, irrespective of exception type or stage. Pre-existing failure bytes remain unmodified.
4. **Failure Artifact Aliases (Item 4):**
   - *Issue:* Consumers and review gates requested canonical aliases matching historical conventions.
   - *Fix:* Added `global_epoch` (alias for zero-based current epoch vs explicit count `completed_epochs`), `git_sha` (alias for `git_commit`), and `config_identity` (alias for `config_signature`). All are populated on post-config init failure when known, with unavailable values explicitly `None` (JSON `null`).
5. **Durability Order & Truthful Validation / Checkpoint Rows (Item 5):**
   - *Issue:* If checkpointing failed after validation, reports lacked explicit status distinguishing completed validation from uncommitted checkpoints. Furthermore, `completed_epochs` was previously incremented after the W&B logging call.
   - *Fix:* Validation row is appended to `self.report["epochs"]` immediately with `validation_complete=True` and `checkpoint_committed=False`. Once checkpoint and progress-report commits succeed, `checkpoint_committed=True` is committed, and `completed_epochs` is incremented BEFORE the W&B adapter call.
6. **Report Ordering & Best-Effort Telemetry Refresh (Item 6 & Coordinator Review):**
   - *Issue:* Moving `logger.finish(exit_code=0)` before the final report write caused a regression: if writing the final report failed, W&B was already finalized with success, preventing truthful failure handling.
   - *Fix:*
     1. Required final research report (`pipeline_report.json`) commits FIRST with telemetry snapshot marked `"finalization_status": "pending"`. Any I/O or serialization failure propagates to the trainer failure handler and exits with `exit_code=1`.
     2. `logger.set_summary` and `logger.finish(exit_code=0)` execute next.
     3. Best-effort atomic telemetry-status refresh executes post-finish, capturing any degraded logger status (`"finalization_status": "finalized"`). Failure of this refresh is treated as nonfatal telemetry degradation and does not invalidate the completed training commit.
7. **Explicit Diagnostics Execution / Skip when Calibration Disabled (Item 7):**
   - *Issue:* When calibration was disabled, requested diagnostics were skipped without reporting reasons or running with fixed tau.
   - *Fix:* Implemented `run_independent_diagnostics`, executing requested diagnostics binding selected `best.pt` with configured fixed tau (`tau_accept`), or explicitly reporting `status: "skipped", reason: "diagnostics_disabled"`.
8. **Subprocess Test Fixture Parity (Item 8):**
   - *Issue:* Subprocess tests used undersized synthetic arrays and did not inherit environment or specify timeouts.
   - *Fix:* Updated `test_subprocess_injected_checkpoint_failure_exits_nonzero_and_writes_failure_json` to specify `image_size=32`, `depth_axis=2`, input shape `(32, 32, 2)`, inherited `env={**os.environ, "PYTHONPATH": "src:."}` and `timeout=60`.

---

## 2. Technical Modifications Summary

- **`src/self_audit/training/unified_trainer.py`:**
  - Placed `record_failure` cleanup in `finally:`, decoupled from payload preparation.
  - Unconditionally preserves existing `failure.json` and writes `failure_attempt_<uuid>.json`.
  - Added aliases `global_epoch`, `git_sha`, `config_identity` to failure artifacts.
  - Per-epoch row tracks `validation_complete`, `checkpoint_committed`, `last_checkpoint_committed_epoch`.
  - `self.completed_epochs` incremented strictly before W&B logging.
  - Commits `pipeline_report.json` before `logger.finish(exit_code=0)`, followed by best-effort post-finish telemetry refresh.
  - Added `run_independent_diagnostics` for calibration-disabled execution.
  - Strict `config_signature` property preserved.
- **`scripts/train_self_audit.py`:**
  - Removed duplicate mid-file imports, keeping clean imports at module top.
  - Unconditionally preserves pre-existing `failure.json` in `_write_pre_init_failure`.
  - Added failure artifact aliases and post-config signature resolution.
- **`tests/test_trainer_lifecycle.py`:**
  - Added 8 focused regression tests (24 tests total):
    1. `test_metadata_preparation_failure_still_finishes`: verifies `finally` cleanup executes even on unstringable exceptions.
    2. `test_ownership_refusal_preserves_old_failure_json`: verifies canonical bytes are untouched and attempt artifact written.
    3. `test_logger_failure_after_durable_report_has_correct_committed_counters`: verifies `completed_epochs` is incremented before logging stage failure.
    4. `test_final_telemetry_failure_report_truthful`: verifies degraded finish error is captured in durable report.
    5. `test_diagnostics_when_calibration_disabled`: verifies independent diagnostics execution with fixed tau.
    6. `test_final_report_failure_prevents_success_finish`: verifies failure during final report write prevents `finish(exit_code=0)` and triggers `exit_code=1`.
    7. `test_pre_init_failure_preserves_canonical_failure_on_resume_value_error`: verifies pre-existing `failure.json` is preserved on resume `ValueError`.
    8. `test_record_failure_preserves_canonical_failure_when_file_exists`: verifies `record_failure` preserves pre-existing `failure.json` across general runtime exceptions.
  - Updated subprocess test fixture with resolution 32, depth axis 2, inherited env, and timeout.

---

## 3. Verification & Test Suite Execution

Executed with `/private/tmp/self-audit-torch241/bin/python` under PyTorch 2.4.1:

| Test File | Status | Duration | Total Items |
|---|---|---|---|
| `tests/test_trainer_lifecycle.py` | 24 / 24 PASSED | 39.89s | 24 items |

All 24 lifecycle, failure schema, cleanup `finally`, unconditional failure artifact preservation, report ordering, telemetry finalization, and subprocess tests passed cleanly.
