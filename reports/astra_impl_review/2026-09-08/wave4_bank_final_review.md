# W4 — Frozen transition bank final validator closure and review

Worker: AGY (Orca dispatches `ctx_0ab9fbe2228d` [task `task_4017b559c3ee`] and follow-up `ctx_7d31d78ea17e` [task `task_e81001310856`]).
Date: 2026-09-08.
Ownership: Owned ONLY `src/self_audit/evaluation/transition_bank.py` (`_stats_from_row` and validator closure), `tests/test_transition_bank.py`, and this report `reports/astra_impl_review/2026-09-08/wave4_bank_final_review.md`.
Constraints: No commit, no push, no training, no architecture/objective/split/checkpoint changes, Proposal 2/3 unopened.

---

## Executive Summary

1. **What was done**: Following Root independent inspection (`wave4_bank_review_notes.md` and review messages `msg_3aaa82a819b9`/`msg_154fe37e479e`), we hardened `transition_bank.py` across all semantic validator requirements, specifically updating `_stats_from_row` to require non-negative counts (`TP/FP/FN >= 0`) and exact declared class key sets (rejecting extra, missing, and normalized alias/colliding keys like `"01"`), and expanded `tests/test_transition_bank.py` to 71 passing tests with negative regression tests after integrity resealing.
2. **What was found**: Negative counts proportionally preserve Dice scores under the standard formula `(2 * -tp) / (2 * -tp + -fp + -fn) == (2 * tp) / (2 * tp + fp + fn)`, proving that recomputing Q alone is insufficient without explicit count non-negativity and exact class key set checks. Additionally, GT-free rollout claims remain strictly scoped to the proposal generation API and model inference, since upstream loaders can optionally consume reference GT via `foreground_only=True` slice filtering.
3. **What is left**: Bounded diagnostic verification of the export path and validator is complete (71/71 tests PASS); no real bank or trained model is claimed, avoiding overclaims beyond the synthetic fixture scope.

---

## Root Independent Inspection and Validator Hardening

The Root independent inspection confirmed that while transport integrity (content hashing) prevented in-flight tampering, semantic inconsistencies across bank metadata, contracts, and row values were accepted once `bank_content_signature` was updated. We systematically implemented and verified all 10 validator requirements:

### 1. Row `empty_policy` must equal contract
- **Diagnosis**: `_EVALUATED_REQUIRED_FIELDS` omitted `empty_policy`, and `_validate_rows` never compared row `empty_policy` against `contract.empty_policy`. A row with `empty_policy="legacy_one"` under an `exclude` contract was accepted once resealed.
- **Remedy**: Added `empty_policy` to `_EVALUATED_REQUIRED_FIELDS`, enforced `_require_nonempty_str` and strict agreement `str(row["empty_policy"]) == contract.empty_policy` in both `_validate_rows` and `validate_bank`.

### 2. Strict integers for `metric_contract_version` and `integrity.num_rows`
- **Diagnosis**: `validate_bank` utilized `int(...)` conversions on `metric_contract_version` (both row-level and header-level) and `integrity["num_rows"]`. In Python, `isinstance(True, int)` evaluates to `True`, and `int(1.5)` or `int(True)` coerces to `1`. Consequently, boolean `True` or fractional `1.0`/`1.5` values were accepted.
- **Remedy**: Converted all version and row counter validations to `_require_int(..., where)`, which rejects boolean instances (`isinstance(value, bool)`) and requires exact integer types (`isinstance(value, int)`).

### 3. `delta_class` strict integer or null
- **Diagnosis**: `row["delta_class"]` was evaluated via `row["delta_class"] != expected_class`. When `expected_class == 1`, passing `True` or `1.0` passed the equality check due to Python numeric coercion. Furthermore, undefined quality rows did not strictly enforce that `delta_class` must be `None`.
- **Remedy**: When delta is defined, `delta_class` is validated via `_require_int(row["delta_class"], ...)` against `expected_class`. When delta is undefined, `row["delta_class"]` is strictly required to be `None` (`BankValidationError` is raised if not null).

### 4. Always-accept source `accepted` must be `True`
- **Diagnosis**: `_validate_rows` previously only validated `accepted` for `ROLLOUT_SELF_AUDIT` (`accepted == gate["accept"]`). Rows with `SOURCE_ALWAYS_ACCEPT_PREFIX` could declare `accepted=False` without rejection.
- **Remedy**: Added explicit enforcement: if `row["source"] == SOURCE_ALWAYS_ACCEPT_PREFIX`, `_require_bool(row["accepted"])` must be `True`.

### 5. `halted_after` must match on-policy rejection
- **Diagnosis**: `halted_after` was merely checked for boolean type, without verifying if the value actually aligned with whether the sample halted. An on-policy accepted row could set `halted_after=True`, or an on-policy rejected row could set `halted_after=False`.
- **Remedy**: In `_validate_rows`, `expected_halted = bool(row["source"] == SOURCE_ON_POLICY and not row["accepted"])`. If `halted_after != expected_halted`, `BankValidationError` is raised.

### 6. Bound checkpoint `state_digest == file_state_digest` and model identity match
- **Diagnosis**: `validate_bank` checked that `state_digest` and `file_state_digest` each matched a 64-hex regex, but never verified that they equaled each other. Additionally, `generation["model_identity"]` was checked only for `isinstance(..., Mapping)`, never comparing it with `checkpoint["model_identity"]`.
- **Remedy**: Enforced `checkpoint["state_digest"] == checkpoint["file_state_digest"]`. Enforced that both `generation["model_identity"]` and `checkpoint["model_identity"]` are mappings and `dict(live_model_identity) == dict(binding_model_identity)`.

### 7. Bound cohort non-empty mapping and schema validation
- **Diagnosis**: For bound exports, `generation["cohort"] = {}` was accepted because `{}` satisfies `isinstance(..., Mapping)`.
- **Remedy**: Enforced that `cohort` is a non-empty mapping. Validated required cohort provenance structure: `split_name` (must match `generation["split_name"]`), `membership_signature` (must be a valid 64-hex SHA-256 digest), `coverage` (must be `MEMBERSHIP_COVERAGE` = `"membership_only_not_content_hash"`), and `num_records` (strict positive integer). Cohort identity is validated against the existing provenance schema without duplicating broad lineage.

### 8. Header metric fields agreement with contract definition
- **Diagnosis**: In `validate_bank`, `evaluation` provenance only checked `contract.name == evaluation["metric_contract"]`. Header fields `metric_contract_version`, `metric_space`, `empty_policy`, and `neutral_margin` were unvalidated against `contract`.
- **Remedy**: Added all four fields to `evaluation` required keys and asserted:
  - `_require_int(evaluation["metric_contract_version"]) == contract.version`
  - `str(evaluation["metric_space"]) == contract.metric_space`
  - `str(evaluation["empty_policy"]) == contract.empty_policy`
  - `abs(float(evaluation["neutral_margin"]) - float(contract.neutral_margin)) <= 1e-12`

### 9. Source identities required for fresh schema (no `generation_code={}`)
- **Diagnosis**: `_check_source_identity` silently returned if `generation_code` was empty or missing `source_content_signature`. Thus, `generation_code={}` was accepted even for bound exports.
- **Remedy**: Updated `_check_source_identity(block, where, *, is_bound, schema_version)`: for fresh schemas (`schema_version >= 1`) or bound exports, `block` must be a non-empty mapping, and `source_content_signature` and `source_content_signature_known` are required.

### 10. Historical unknown producer preservation
- **Diagnosis**: Checkpoints saved before provenance recording was introduced report `producer_git_sha == "unknown"` with `producer_recorded == False`.
- **Remedy**: Verified that `checkpoint["producer"]` preserves and accepts `"producer_git_sha": "unknown"`. `_check_source_identity` allows `signature == "unknown"` when `known is False`, ensuring historical checkpoints remain loadable and verifiable without model or objective changes.

### 11. Sufficient Statistics Count Validity and Exact Class Key Sets (`_stats_from_row`)
- **Diagnosis**: `_require_int` validates only that values are integer types without boolean coercion, but does not enforce non-negativity. Negative counts across `tp`, `fp`, and `fn` preserve Dice scores identically due to proportional cancellation `(2 * -tp) / (2 * -tp + -fp + -fn) == (2 * tp) / (2 * tp + fp + fn)`, bypassing quality recomputation checks. Furthermore, `int(key)` in earlier drafts silently normalized string aliases (e.g. `"01"` -> `1`), ignored extra classes (e.g. `"99"`), and permitted alias collisions.
- **Remedy**: Hardened `_stats_from_row` to require:
  1. Every count must be strictly non-negative (`count >= 0`), raising `BankValidationError` on negative counts.
  2. Exact declared class key set: keys must be strings exactly matching `str(c)` for declared classes, rejecting any extra keys, missing keys, non-string keys, or normalized alias/colliding keys (`"01"`).

---

## GT-Free Claim Scoping Disclosure

A critical design inquiry concerned whether transition bank generation can be claimed as strictly ground-truth-free.

1. **Generation API Scope**: `generate_on_policy_proposals(model, images, ...)` takes raw image tensors and explicit string/int identifiers (`patient_ids`, `case_ids`, `slice_indices`). It refuses dataset batch mappings with `TypeError` because PyTorch batch dicts carry `"mask"`. Inside `model.infer`, proposals, residual updates, auditor logits, delta-Q, and gating decisions are computed without ground-truth access.
2. **Upstream Loader Selection Scope**: In volume slice datasets (`VolumeSliceDataset` in `src/self_audit/data/common.py`), the optional constructor argument `foreground_only: bool = False` inspects `np.any(mask[slice_index] > 0)` to omit empty slices. While inactive and omitted by default training builders, if a user loader enables `foreground_only=True`, slice selection itself consumes reference masks before `generate_on_policy_proposals` is called.
3. **Formal Scope Boundary**: The GT-free claim is therefore **strictly scoped to the proposal generation API and model inference rollout**. An end-to-end GT-free claim across an entire data pipeline requires verifying that the upstream data loader does not perform GT-dependent slice filtering. Both the module docstring and the `generate_on_policy_proposals` docstring have been updated with explicit `SCOPE NOTE` disclosures.

---

## New Regression Negative Tests (Section 14)

All 17 new regression tests in `tests/test_transition_bank.py` deliberately mutate a valid bank, call `reseal(tampered)` (recomputing `bank_content_signature`), and assert that `validate_bank` rejects the semantic violation rather than failing on a transport hash mismatch:

| Test | Target / Mutation | Prior Draft | Hardened Validator Result |
|---|---|---|---|
| `test_row_empty_policy_must_equal_contract` | `empty_policy="legacy_one"` or missing key | ACCEPTED | Rejected: `disagrees with the bank contract empty policy` or `missing required field` |
| `test_metric_contract_version_strict_integer_no_coercion` | Row/header `metric_contract_version` = `True` or `1.0` | ACCEPTED | Rejected: `must be an integer, got True` / `got 1.0` |
| `test_integrity_num_rows_strict_integer_no_coercion` | `integrity.num_rows` = `True` or float `len` | ACCEPTED | Rejected: `Bank integrity num_rows must be an integer` |
| `test_delta_class_strict_integer_or_null` | Defined row `delta_class=True`/`1.0`; undefined row `delta_class=0` | ACCEPTED | Rejected: `delta_class must be an integer` / `must be null when delta is undefined` |
| `test_always_accept_source_must_have_accepted_true` | Always-accept row with `accepted=False` | ACCEPTED | Rejected: `has source 'always_accept_prefix' but accepted is false` |
| `test_halted_after_must_match_on_policy_rejection` | Accepted row `halted_after=True`; rejected row `halted_after=False` | ACCEPTED | Rejected: `halted_after=... does not match on-policy rejection` |
| `test_bound_checkpoint_state_digest_must_equal_file_state_digest` | Bound checkpoint `file_state_digest` != `state_digest` | ACCEPTED | Rejected: `state_digest ... does not match file_state_digest` |
| `test_bound_live_model_identity_must_match_binding` | Live `model_identity` != binding `model_identity` | ACCEPTED | Rejected: `Live model_identity does not match checkpoint binding model_identity` |
| `test_bound_cohort_empty_mapping_or_invalid_structure_rejected` | `cohort={}`, split mismatch, invalid hash, or content coverage claim | ACCEPTED | Rejected: `empty or null cohort`, `split_name disagrees`, `not a full SHA-256 digest`, `coverage must be 'membership_only_not_content_hash'` |
| `test_header_metric_fields_must_agree_with_contract` | Header version, space, empty_policy, or neutral_margin altered | ACCEPTED | Rejected: `disagrees with contract` |
| `test_source_identities_required_no_empty_generation_code` | `generation_code={}` on bound or demo export; missing signature | ACCEPTED | Rejected: `generation_code must be a non-empty mapping` / `missing required source_content_signature` |
| `test_preserve_unknown_historical_producer` | Bound checkpoint with `producer_git_sha="unknown"`, `recorded=False` | N/A | Succeeded: validates and preserves historical unknown producer |
| `test_negated_sufficient_statistics_counts_rejected_after_reseal` | Negated `tp`, `fp`, `fn` counts (preserves Dice via cancellation) | ACCEPTED | Rejected: `must be non-negative, got -...` |
| `test_extra_class_key_in_sufficient_statistics_rejected` | Extra undeclared class key (e.g. `"99"`) in counts | ACCEPTED | Rejected: `unrecognized class key '99'` |
| `test_missing_class_key_in_sufficient_statistics_rejected` | Missing declared class key in counts mapping | ACCEPTED | Rejected: `missing count for declared class '...'` |
| `test_alias_class_key_in_sufficient_statistics_rejected` | String alias class key (e.g. `"01"`) instead of exact declared `"1"` | ACCEPTED | Rejected: `unrecognized class key '01'` |
| `test_colliding_alias_class_key_in_sufficient_statistics_rejected` | Normalized colliding key (e.g. `"1"` and `"01"` both present) | ACCEPTED | Rejected: `unrecognized class key '01'` |

---

## Verification and Execution Outputs

### Environment Facts
```text
python=3.11.16 pytorch=2.13.0 pytest=9.1.1 numpy=2.4.6
interpreter=/Users/alvinluong/miniforge3/bin/python
cwd=/Users/alvinluong/Self-Audit
```

### Focused Regression Suite
```text
$ PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest -q -p no:cacheprovider tests/test_transition_bank.py
71 passed in 12.55s
```

### Software Verification vs Real Artifact Scope Disclosure

> [!IMPORTANT]
> **Verification Boundary & Non-Overclaim Disclosure**:
> The 71 passed tests verify the **software diagnostic path, schema enforcement, and validator integrity contracts** against synthetic fixtures, simulated rollouts, and tiny test datasets. This review does **not** claim that a production clinical transition bank or trained model checkpoint weights have been generated or validated. Full artifact certification will occur during production rollout when trained checkpoints and clinical cohorts are processed through this verified bank pipeline.

### Whitespace and Syntax Hygiene
```text
$ python -c "
for p in ['src/self_audit/evaluation/transition_bank.py', 'tests/test_transition_bank.py', 'reports/astra_impl_review/2026-09-08/wave4_bank_final_review.md']:
    with open(p) as f:
        for i, l in enumerate(f, 1):
            if l.rstrip('\r\n') != l.rstrip(): print(f'{p}:{i} trailing whitespace')
            if '\t' in l: print(f'{p}:{i} tab')
"
# Result: 0 trailing whitespace, 0 tabs.

$ python -m py_compile src/self_audit/evaluation/transition_bank.py tests/test_transition_bank.py
# Result: COMPILE_OK (exit code 0).
```

### Concurrent Repository Context
The full repository test suite contains pre-existing concurrent failures in unrelated modules (`test_calibration_lineage.py`, `test_metric_contract_and_replay.py`, and `test_remediation_evaluation.py`) owned by parallel workers in W3.2. All transition bank tests pass completely (71/71).

---

## Files Modified

| File | Status | Description |
|---|---|---|
| `src/self_audit/evaluation/transition_bank.py` | Modified | Added GT-free scoping documentation; added `empty_policy` to `_EVALUATED_REQUIRED_FIELDS`; enforced strict integers for versions and num_rows; enforced strict integer/null `delta_class`; enforced `always-accept` acceptance invariant; enforced `halted_after` on-policy rejection matching; enforced checkpoint state digest equality; enforced live-vs-binding model identity equality; enforced bound cohort structure and SHA-256 signature; enforced header metric contract parameter agreement; enforced non-empty source identities for fresh schemas; preserved historical unknown producers; enforced non-negative counts (`>= 0`) and exact declared class key sets (rejecting extra, missing, alias, and colliding keys). |
| `tests/test_transition_bank.py` | Modified | Added Section 14 containing 17 negative regression tests exercising all validator closure requirements against resealed banks. Set `later["halted_after"]=True` in `test_bank_rejects_a_fabricated_transition_after_a_halt`. Total tests expanded from 54 to 71. |
| `reports/astra_impl_review/2026-09-08/wave4_bank_final_review.md` | New | This comprehensive final review report with GT-free scoping disclosure, validator closure documentation, and execution verification. |
