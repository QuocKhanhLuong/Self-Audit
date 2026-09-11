# Wave 2 Delivery: Strict Artifact IO, Safe Atomic Writes, and Telemetry Hardening

**Execution Date:** 2026-09-10
**Task ID:** `task_efbb41632698` (continuation from `task_5417912b8682`)
**Dispatch ID:** `ctx_6abb2c22a017` (continuation from `ctx_a36fdbd82651`)
**Status:** Completed (All review checkpoints and real writer entrypoint probes resolved)
**Target Environment:** PyTorch 2.4.1 (Python 3.10.21, macOS Darwin)

---

## 1. Executive Summary & Root Cause Analysis

Wave 2 establishes unified, strict artifact IO, safe atomic file replacement with unswallowed fsync, strict JSON reading, safe transition-cache persistence and loading under PyTorch 2.4.1 `weights_only=True`, and observable Weights & Biases telemetry payload conversion across the self-audit codebase.

### 1.1 Key Vulnerabilities Addressed

1. **Non-Finite Floats in JSON Artifacts:**
   - *Issue:* Python's default `json.dump` emits non-standard tokens (`NaN`, `Infinity`, `-Infinity`) unless forbidden, causing strict parsers and downstream schema validators to fail. Conversely, naive workarounds that map NaN to `0.0` destroy the semantic distinction between an undefined/absent metric (e.g. absent class Dice) and complete segmentation failure (Dice = 0.0).
   - *Fix:* Built `src/self_audit/artifact_io.py:json_safe_artifact`, which recursively maps non-finite floats (`math.isnan`, `math.isinf`) to `None` (JSON `null`), never numeric zero. Undefined metrics remain clearly distinguishable from zero.
2. **Lossy Float8 Scalar Conversion in Tensors:**
   - *Issue:* Converting 0D tensors by matching a whitelist of floating types (`float32`, `float64`, `float16`, `bfloat16`) causes newer PyTorch 2.4 float8 types (`torch.float8_e4m3fn`, `torch.float8_e5m2`) to fall into integer branches, truncating `1.5` to `1`.
   - *Fix:* Converted 0D tensors via `t.item()` and ND tensors via `t.tolist()`, recursively routing their elements through `json_safe_artifact`. Float8 scalars convert losslessly to Python `float`, and unsupported complex tensors fail closed with exact paths.
3. **Custom Objects and Tensor Subclasses in JSON:**
   - *Issue:* Production writers often used permissive serializers (e.g. `default=str`), stringifying arbitrary custom objects or masking tensor subclasses that cannot be loaded cleanly.
   - *Fix:* `json_safe_artifact` strictly rejects arbitrary custom objects, custom tensor subclasses, and boolean mapping keys with `ArtifactSerializationError` and exact tree paths.
4. **Mapping Key Collisions:**
   - *Issue:* Serializing mappings with heterogeneous keys (e.g. `{1: "a", "1": "b"}` or `{Path("x"): 1, "x": 2}`) silently overwrote values during stringification.
   - *Fix:* `json_safe_artifact` detects duplicate normalized string keys and raises `ArtifactSerializationError: Mapping key collision at '<path>'`.
5. **Partial / Corrupted File Writes & Masked Fsync Failures:**
   - *Issue:* Standard `open("w").write()` leaves truncated or empty files if the process crashes, the disk fills (`ENOSPC`), or write permissions fail midway.
   - *Fix:* `atomic_write_json` writes UTF-8 encoded JSON (`allow_nan=False`) to a hidden same-directory `.tmp` file, flushes, invokes `os.fsync` on the file descriptor and parent directory, and executes an atomic replacement via `os.replace`. Temporary files are unlinked on error without masking underlying IO failures.
6. **Non-Standard JSON Parsing:**
   - *Issue:* Standard `json.load` accepts `NaN` and `Infinity` tokens by default.
   - *Fix:* Implemented `read_json_artifact`, configuring `parse_constant` to reject non-standard constants with a descriptive `ValueError`.
7. **Permissive Telemetry Conversion & Disjoint Policies:**
   - *Issue:* `clean_wandb_payload` previously used a separate, weaker serialization policy that silently overwrote collision keys and stringified arbitrary objects.
   - *Fix:* Refactored `clean_wandb_payload` to reuse the exact same strict `json_safe_artifact` converter. Added error observability to `WandbLogger` (`last_error`, `failed_log_count`, `telemetry_errors`), added `set_summary`, and supported exit codes in `finish(exit_code=None)`.
8. **Insecure Transition Cache Loading & Raw Cache Writing:**
   - *Issue:* `scripts/calibrate_threshold.py:_load` previously used `weights_only=False` (allowing arbitrary code execution via pickle exploits) or fell back unsafely. Furthermore, `unified_trainer.py:run_post_training_calibration` used raw `torch.save` to write `validation_transitions.pt`.
   - *Fix:* Hardened `_load` to enforce `torch.load(..., weights_only=True)`, raising a clear `ValueError` on unsupported legacy formats or unsafe pickled objects without unsafe fallback. Converted `run_post_training_calibration`, `cache_validation_transitions.py`, and `train_self_audit_legacy.py` to persist caches via `atomic_save_torch`.

---

## 2. Technical Modifications Across Production Writers

### 2.1 Core Shared Modules
- **`src/self_audit/artifact_io.py` (Created):**
  - Defines `ArtifactSerializationError(TypeError, ValueError)`.
  - Implements `json_safe_artifact(val, path="root")`.
  - Implements `atomic_write_json(path, payload, indent=2, sort_keys=True)`.
  - Implements `read_json_artifact(path)`.
  - Implements `clean_wandb_payload(val, path="root")` by delegating directly to `json_safe_artifact`.
- **`src/self_audit/serialization.py` (Updated):**
  - Re-exports `ArtifactSerializationError`, `atomic_write_json`, `clean_wandb_payload`, `json_safe_artifact`, and `read_json_artifact`.
  - Carried forward W1 cleanups: removed unused imports, removed unreachable bytes return branches in `_normalize_training_tree`, and updated docstrings to reflect bytes/bytearray rejection under PyTorch 2.4.
- **`src/self_audit/training/_utils.py` (Updated):**
  - Hardened `WandbLogger`:
    - Config cleaned via `clean_wandb_payload(dict(config))`.
    - Metrics cleaned via `clean_wandb_payload(dict(metrics))` in `log()`.
    - Telemetry health tracking: `last_error`, `failed_log_count`, and `telemetry_errors` recorded on initialization, log, summary, and finish failures.
    - Added `set_summary(summary_dict)` using `clean_wandb_payload`.
    - Supported `finish(exit_code: int | None = None)`.

### 2.2 Production Artifact Writer Integrations
1. **`src/self_audit/training/unified_trainer.py`:**
   - `_json_safe` aliased to `json_safe_artifact`.
   - `pipeline_report.json` written via `atomic_write_json(report_path, self.report, indent=2, sort_keys=True)`.
   - Canonical validation transition cache persisted via `atomic_save_torch(cache, report_dir / "validation_transitions.pt")`.
2. **`src/self_audit/evaluation/threshold.py`:**
   - `json_safe` delegates to `json_safe_artifact`.
   - `save_calibration` writes via `atomic_write_json(destination, payload, indent=2, sort_keys=True)`.
3. **`src/self_audit/evaluation/transition_bank.py`:**
   - `dump_bank` writes via `atomic_write_json(target, dict(bank), indent=2, sort_keys=True)`.
   - Transition bank canonical SHA-256 signature (`bank_content_signature`) remains bit-for-bit invariant.
4. **`scripts/audit_checkpoint.py`:**
   - Audit report output persisted via `atomic_write_json(output, payload, indent=2, sort_keys=True)`.
5. **`scripts/evaluate_external_mnms.py`:**
   - `_json_safe` aliased to `json_safe_artifact`.
   - External evaluation report output persisted via `atomic_write_json(output_path, payload, indent=2, sort_keys=True)`.
6. **`scripts/train_self_audit_legacy.py`:**
   - `_json_safe` aliased to `json_safe_artifact`.
   - Preserved `_save_report = _write_json` interface, writing via `atomic_write_json(path, payload, indent=2)`.
   - Validation transition cache persisted via `atomic_save_torch`.
7. **`scripts/cache_validation_transitions.py`:**
   - Validation transition cache persisted via `atomic_save_torch(cache, args.output)`.
8. **`scripts/calibrate_threshold.py`:**
   - `_load` updated to explicit `torch.load(path, map_location="cpu", weights_only=True)` with clear error handling for legacy or unsafe formats.

---

## 3. Review Checkpoint Resolutions

In response to the coordinator review messages (`msg_860c0018ed71` and `msg_cb2de923843a`):
- **Float8 Scalar Conversion:** Converted 0D tensors via `t.item()`, ensuring `torch.float8_e4m3fn` converts to Python `float` without truncation to `int`.
- **Complex Tensor Rejection:** Complex scalar and vector tensors fail closed with `ArtifactSerializationError` and exact paths.
- **Single Telemetry Semantic Policy:** Replaced duplicated weaker converter in `clean_wandb_payload` with strict `json_safe_artifact` reuse. Mapping collisions (`{1: 'a', '1': 'b'}`) and arbitrary objects (`object()`) raise `ArtifactSerializationError`.
- **Canonical Cache Boundary:** Replaced raw `torch.save` in `UnifiedTrainer.run_post_training_calibration` with `atomic_save_torch`.
- **Legacy Interface Preservation:** Maintained `_save_report = _write_json` in `train_self_audit_legacy.py`.
- **Explicit Null Policy:** Documented in `artifact_io.py` and artifact schemas that non-finite metrics map to `null`, ensuring undefined metrics remain distinguishable from numeric zero.

---

## 4. Systems Out of Scope & Invariants

1. **Data Preparation Manifests:**
   - Manifest and preprocessing scripts (`preprocess_acdc.py`, `preprocess_myops.py`, `acdc_split.py`, `prepare_mnm_binary.py`) are offline data preparation utilities outside the active training and evaluation runtime and were intentionally left unchanged.
2. **Recipe & Model Integrity:**
   - Network architectures, loss formulations, training intervals, learning rates, optimizer states, and evaluation thresholds remain strictly unchanged.
3. **Transition Bank Negative Semantic Validation Adjustment:**
   - In `tests/test_transition_bank.py:test_quality_inconsistent_with_its_own_statistics_is_rejected`, row 0 of `sealed_bank` initialized under PyTorch 2.4.1 has `delta_class=1`.
   - The test deliberately sets `q_previous = 0.999`, causing `delta_dice = q_candidate - 0.999` to become negative.
   - In `validate_bank`, consistency between `delta_class` and `delta_dice` is evaluated before checking sufficient statistics (`q_previous` vs `per_class_dice_previous`).
   - Because `delta_class` remained `1` while `delta_dice` became negative, `validate_bank` failed with `Row 0 delta_class=1 disagrees with the canonical classification -1 at margin 0.005`, preventing execution from reaching the intended test assertion.
   - Adjusting `tampered["rows"][0]["delta_class"] = -1` aligns the class label with the negative delta, strictly preserving negative semantic validation rather than weakening it, allowing `validate_bank` to reach and verify the sufficient-statistics discrepancy (`recomputed from its sufficient statistics`).
4. **Security Invariant:**
   - Zero capability secrets or credentials written to disk. No `git commit` or `git push` executed.

---

## 5. Comprehensive Test Verification Matrix

All tests were executed against PyTorch 2.4.1 CPU (`/private/tmp/self-audit-torch241/bin/python`) using the target environment prefix:
`rtk proxy env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:. OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /private/tmp/self-audit-torch241/bin/python -m pytest`

| Test Suite | Items Collected | Items Passed | Execution Time | Status |
| :--- | :--- | :--- | :--- | :--- |
| `tests/test_artifact_io.py` | 18 | 18 | 0.60s | **PASSED** |
| `tests/test_runtime_checkpoint.py` | 27 | 27 | 1.98s | **PASSED** |
| `tests/test_checkpoint_binding.py` | 55 | 55 | 48.84s | **PASSED** |
| `tests/test_unified_trainer.py` | 34 | 34 | 58.13s | **PASSED** |
| `tests/test_wandb_and_tqdm.py` | 6 | 6 | 0.95s | **PASSED** |
| `tests/test_external_mnms_evaluator.py` | 1 | 1 | 0.63s | **PASSED** |
| `tests/test_transition_bank.py` | 71 | 71 | 50.45s | **PASSED** |
| **Total Comprehensive Suite** | **212** | **212** | **~161s** | **ALL PASSED** |

### Test Highlights from `tests/test_artifact_io.py`:
- `test_json_safe_torch_tensors`: Confirms float8 scalar preservation (`1.5 -> 1.5`), non-finite floats to `None`, and complex tensor rejection with exact path.
- `test_json_safe_mapping_keys_and_collision`: Confirms normalization of integer/Path keys and rejection of duplicate normalized keys.
- `test_atomic_write_json_fsync_failure_preserves_original`: Confirms simulated fsync `OSError` leaves pre-existing valid artifact unmodified.
- `test_atomic_write_json_replace_failure_preserves_original`: Confirms simulated replace `OSError` leaves pre-existing valid artifact unmodified.
- `test_read_json_artifact_rejects_nonstandard_constants`: Confirms strict rejection of raw `NaN`, `Infinity`, and `-Infinity` tokens.
- `test_clean_wandb_payload`: Confirms rejection of mapping collisions and arbitrary objects.
- `test_wandb_logger_telemetry_observability`: Confirms error tracking (`last_error`, `failed_log_count`, `telemetry_errors`), summary setting, and exit codes.
- `test_calibrate_threshold_load_enforces_weights_only`: Confirms safe transition cache loading and rejection of pickle exploits under `weights_only=True`.
- `test_external_evaluation_nan_dice_to_null`: Confirms undefined Dice values serialize to `null` while zero values remain `0.0`.
- `test_run_external_evaluation_entrypoint_with_undefined_metrics`: Executes real `scripts.evaluate_external_mnms.run_external_evaluation` writer with nested undefined metrics, proving real JSON artifact emission with `null` fields readable by strict parsers.
- `test_save_calibration_writer_preserves_old_file_on_error`: Executes real `self_audit.evaluation.threshold.save_calibration` writer against injected fsync error, verifying pre-existing calibration artifact is preserved.
- `test_dump_bank_writer_preserves_old_file_on_error`: Executes real `self_audit.evaluation.transition_bank.dump_bank` writer against injected replace error, verifying pre-existing transition bank artifact is preserved.
