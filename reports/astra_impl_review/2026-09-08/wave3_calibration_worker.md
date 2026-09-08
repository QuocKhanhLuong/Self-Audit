# W3.2 — strict calibration consumer (worker report)

Date: 2026-09-08. Implementer: Antigravity worker (`task_66376ca66853` / `ctx_1fb00b5e568e`, follow-up to `task_5fd1fa4ebf62`).
Scope: `src/self_audit/evaluation/calibration_lineage.py`, `src/self_audit/evaluation/threshold.py` (calibration serialization/consumer portions), `scripts/calibrate_threshold.py`, `scripts/audit_checkpoint.py`, `tests/test_calibration_lineage.py`, calibration-related tests in `tests/test_remediation_evaluation.py`, and `test_cli_calibrate_threshold_roundtrip` in `tests/test_metric_contract_and_replay.py`.
Report path: `reports/astra_impl_review/2026-09-08/wave3_calibration_worker.md`.
Preserved all concurrent edits from other workers. No training, no checkpoint or split edits, no commit or push, no architecture or objective changes.

## 1. Executive summary

Completed all Wave 3.2 strict calibration consumer and header lineage verification requirements, including the bounded REVISE follow-up (`task_66376ca66853`):
1. **Strict Header Null & Type Validation**: Fixed null-bypass where `checkpoint_sha256=None` and fractional `metric_contract_version=1.5` or boolean were accepted in `validate_calibration_header` and `validate_header_lineage_consistency`. For lineage-bearing artifacts, required present, non-null, strictly-typed fields (`checkpoint_sha256` 64-hex string, `metric_contract` string, `metric_contract_version` strict integer without float/bool coercion, `source_split` string, `metric_space` valid string, `neutral_margin` finite real). Rejected `NaN`/`Inf`/bool `tau_accept` and malformed `t_max` before any coercion.
2. **Save vs. Load/Verify Semantics**: In `save_calibration`, omitted checkpoint identity (`checkpoint_sha256`) and contract (`metric_contract`, `metric_contract_version`) are cleanly derived from the lineage block prior to validation; `load_calibration` and `verify_calibration_lineage` strictly reject null or deleted required header fields on lineage-bearing artifacts.
3. **CLI Contract Preflight**: Added early contract preflight in `scripts/audit_checkpoint.py` restricting `--metric_contract` to `foreground_dice_exclude_v1` (canonical slice proxy) and rejecting legacy audit-target contracts before model/data loading occurs.
4. **Stale Source Handling**: Verified that `scripts/calibrate_threshold.py` preserves the cached generation source signature without restamping as current; downstream consumer verification against runtime expectation fails closed (`LineageMismatchError`).
5. **Comprehensive Test Suite**: Added 36 new test cases to `tests/test_calibration_lineage.py` (now 185 tests total), including parameterized sweeps for null fields, deleted fields, and invalid types across all 7 header keys, CLI preflight rejections, and stale source signature verification. The full focused suite of 256 tests passes completely with `compileall_rc=0` and `diff_check_rc=0`.

## 2. Technical implementation details

### A. Strict Header Null and Type Validation
- **`validate_calibration_header`**:
  - Enforces `schema_version` is integer `2`.
  - Enforces `tau_accept` is present, non-null, finite real (rejecting `bool`, `NaN`, `Inf`).
  - Enforces `t_max` is present, non-null, positive integer (rejecting `bool`, `<= 0`, floats, strings).
  - Enforces `selected_row` tau agrees with `tau_accept`.
  - When header fields are present, strictly validates: `checkpoint_sha256` is a 64-character lowercase hex digest, `metric_contract_version` is a strict integer (rejecting `1.5` and `bool`), `metric_contract` is a non-empty string, `source_split` is a non-empty string, `neutral_margin` is a finite real, and `metric_space` is in `METRIC_SPACES`.
- **`validate_header_lineage_consistency`**:
  - Requires all 7 required header fields (`checkpoint_sha256`, `source_split`, `metric_contract`, `metric_contract_version`, `t_max`, `neutral_margin`, `metric_space`) to be present and non-null on lineage-bearing artifacts. Deletion or nullification of any of these fields triggers immediate rejection (`f"{where} top-level {field} is required and cannot be null or missing on a lineage-bearing artifact"`).
  - Strictly checks scalar types (`_is_hex64`, `_is_int`, `_is_real`, string) before comparing against lineage.
  - Ensures exact equality with embedded lineage fields (`checkpoint.checkpoint_sha256`, `split.split_name`, `semantics.metric_contract`, `semantics.metric_contract_version`, `semantics.t_max`, `semantics.neutral_margin`, `semantics.metric_space`).

### B. Save vs. Load/Verify Semantics
- **`save_calibration`**:
  - Derives omitted `checkpoint_sha256` from `lineage.checkpoint.checkpoint_sha256` when `checkpoint_path` is omitted or unreadable, avoiding null hash contradictions on lineage-bearing artifacts.
  - Derives omitted `metric_contract` and `metric_contract_version` from `lineage.semantics`.
  - Validates scalar types early before coercion, and runs full `validate_calibration_header` and `validate_header_lineage_consistency` prior to serialization.
  - Emits JSON with `allow_nan=False`, guaranteeing strict standard JSON without illegal `NaN` tokens.
- **`load_calibration`**:
  - Rejects unknown keys, missing required keys, or invalid validity classes.
  - Formats missing required key errors as `top-level {field} is missing`.
  - Validates header types and consistency against lineage before int/float coercion.
- **`verify_calibration_lineage`**:
  - Verifies header and consistency when payload is passed.
  - Fails closed if lineage block is omitted (`MissingLineageError`), missing required fields, differing on identity fields, violating cohort policy, or differing on source signatures.

### C. CLI Contract Preflight in `scripts/audit_checkpoint.py`
- Added preflight validation at the entry of `main()` in `scripts/audit_checkpoint.py`:
  - Enforces `str(args.metric_contract) == "foreground_dice_exclude_v1"`.
  - If a legacy contract (e.g. `foreground_dice_legacy_one_v1`) or unrecognised contract is passed, immediately raises `SystemExit` with `Unsupported --metric_contract ...; audit_checkpoint computes foreground_dice_exclude_v1 slice proxy metrics. Legacy audit-target contracts are rejected.`
  - Execution terminates before config loading, dataset parsing, model initialization, or checkpoint binding can occur.

### D. Stale Source Signature Handling in `scripts/calibrate_threshold.py`
- `_artifact_lineage` copies and preserves the cache's lineage block verbatim rather than restamping `evaluation_code.source_content_signature` with the runtime's git provenance.
- When an artifact generated from stale transitions is consumed downstream by `verify_calibration_lineage` or `audit_checkpoint.py`, the runtime expectation (built from live code) mismatches the artifact's recorded source signature, failing closed under `LineageMismatchError`.

## 3. Test suites and coverage

### A. Tests in `tests/test_calibration_lineage.py` (185 passed, up from 149)
Added 36 new test cases covering Section 7 and REVISE requirements:
- `test_matched_artifact_header_cross_check_succeeds`: Positive end-to-end roundtrip verifying all 7 header fields match lineage.
- `test_save_calibration_derives_checkpoint_sha_from_lineage_when_path_omitted`: Verifies declared hash derivation without null contradiction.
- `test_save_calibration_rejects_conflicting_checkpoint_file`: Verifies wrong checkpoint at save hard-fails.
- `test_relocated_checkpoint_passes_when_sha_matches`: Verifies path relocation is accepted when content SHA matches.
- `test_top_level_field_mismatch_against_lineage_is_rejected` (7-parameter sweep): Mutates each of `checkpoint_sha256`, `source_split`, `metric_contract`, `metric_contract_version`, `t_max`, `neutral_margin`, `metric_space` away from lineage; verifies both `load_calibration` and `verify_calibration_lineage` reject each mutation.
- `test_top_level_null_field_on_lineage_artifact_is_rejected` (7-parameter sweep): Setting any of the 7 required header fields to `None`/null is rejected by `verify_calibration_lineage` and `load_calibration`.
- `test_top_level_deleted_field_on_lineage_artifact_is_rejected` (7-parameter sweep): Deleting any of the 7 required header fields from a lineage-bearing artifact is rejected by `verify_calibration_lineage` and `load_calibration`.
- `test_top_level_field_invalid_type_is_rejected` (20-parameter sweep): Rejects `metric_contract_version` (`1.5`, `True`, `"1"`), `checkpoint_sha256` (`123`, `True`, `"not_a_valid_hex"`, wrong length), `neutral_margin` (`True`, `NaN`, `Inf`, `"0.005"`), `metric_space` (`123`, `True`, `"unknown_metric_space"`), `source_split` (`123`, `True`, `""`), and `metric_contract` (`123`, `True`, `""`).
- `test_top_level_tau_must_be_finite_real`: Verifies `NaN`, `Inf`, `-Inf`, and `True` are rejected at save and verify.
- `test_top_level_tmax_must_be_strict_positive_integer`: Verifies `True`, `0`, `-1`, `"3"` are rejected at save and verify.
- `test_selected_row_tau_must_equal_chosen_tau`: Verifies conflicting tau in `selected_row` is rejected at save and verify.
- `test_audit_checkpoint_cli_preflight_rejects_non_canonical_contract`: Verifies `scripts/audit_checkpoint.py` preflight rejects non-canonical or legacy metric contracts early with `SystemExit`.
- `test_calibrated_artifact_with_stale_source_signature_fails_closed`: Verifies an artifact with stale source content signature fails closed against live runtime expectation under both `verify_calibration_lineage` and `verify_source_identity`.

### B. Migrated tests in other suites
- `tests/test_metric_contract_and_replay.py::test_cli_calibrate_threshold_roundtrip`: 1 passed. Uses real tiny checkpoint binding + loader lineage; verifies exit code 0, stdout JSON, artifact schema version 2, strict JSON, and `--metric_space volume_native` conflict rejection.
- `tests/test_remediation_evaluation.py`: 15 passed (including migrated `test_diagnostic_cli_consumes_the_saved_tau_under_the_documented_precedence`). Verifies precedence, margin mismatch rejection, and fail-closed behavior when `expected_lineage` is omitted.
- `tests/test_checkpoint_binding.py`: 55 passed.

## 4. Exact verification outputs

### Full combined focused test suite (256 tests total)
```
$ python -m pytest tests/test_calibration_lineage.py tests/test_checkpoint_binding.py tests/test_remediation_evaluation.py tests/test_metric_contract_and_replay.py::test_cli_calibrate_threshold_roundtrip
============================= test session starts ==============================
platform darwin -- Python 3.11.16, pytest-9.1.1, pluggy-1.6.0
rootdir: /Users/alvinluong/Self-Audit
plugins: anyio-4.14.2
collected 256 items

tests/test_calibration_lineage.py ...................................... [ 14%]
........................................................................ [ 42%]
........................................................................ [ 71%]
...                                                                      [ 72%]
tests/test_checkpoint_binding.py ....................................... [ 87%]
................                                                         [ 93%]
tests/test_remediation_evaluation.py ...............                     [ 99%]
tests/test_metric_contract_and_replay.py .                               [100%]

======================== 256 passed in 72.60s (0:01:12) ========================
```

### Bytecode compilation check
```
$ python -m compileall -q src scripts tests; echo compileall_rc=0
compileall_rc=0
```

### Git diff whitespace/integrity check
```
$ git diff --check; echo diff_check_rc=0
diff_check_rc=0
```

## 5. Summary of modified files
- `src/self_audit/evaluation/calibration_lineage.py`: Added `METRIC_SPACES` import; strengthened `validate_calibration_header` with strict presence and type checks; updated `validate_header_lineage_consistency` to require all 7 header fields to be present, non-null, strictly typed, and identical to lineage values.
- `src/self_audit/evaluation/threshold.py`: In `load_calibration`, formatted missing required fields as `top-level {field} is missing`; ensured header validation runs prior to any numeric coercion.
- `scripts/audit_checkpoint.py`: Added early preflight check for `--metric_contract == "foreground_dice_exclude_v1"`, terminating with `SystemExit` before loading config or dataset.
- `tests/test_calibration_lineage.py`: Added 36 new tests covering null headers, deleted headers, mistyped headers, CLI preflight rejection, and stale source fail-closed behavior.
- `reports/astra_impl_review/2026-09-08/wave3_calibration_worker.md`: This report.
