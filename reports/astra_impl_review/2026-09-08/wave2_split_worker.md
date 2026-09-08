# Wave 2.1 Implementation Report: Authoritative Effective Split Records and Safety

**Date:** 2026-09-08
**Worker:** AGY Implementer
**Task ID:** `task_3945e271726f`
**Dispatch ID:** `ctx_3b7cbf0792c7`
**Coordinator Terminal:** `term_bb8fdfa8-8643-4694-8cda-d05efb24504b`
**Worker Terminal:** `term_4991158f-02ba-4966-80ad-dfb1b2b608f5`
**Scope:** Wave 2.1 ONLY (`reports/astra_impl_review/2026-09-08/wave2_split_task.md` & `reports/astra_impl_review/2026-09-08/wave2_split_review_notes.md`)
**Status:** Completed and Verified locally

---

## 1. Executive Summary

1. **What was done:** Implemented a single authoritative split resolver (`resolve_effective_acdc_splits` in `data/acdc.py`) returning an immutable `EffectiveSplits` bundle with a deterministic SHA-256 signature (`compute_split_signature`), ensuring complete parity between dataset loading (`ACDCDataset`) and preflight split validation (`validate_dataset_splits`). Enforced strict manifest authority over directory tags (flagging manifest-vs-tag conflicts as hard errors), eliminated silent set-deduplication in `read_split_manifest` (catching duplicate list entries and cross-split case duplication), upgraded `discover_acdc_records` to safely deduplicate recursive traversals of identical file pairs while failing real file collisions, and asserted joint cross-split validation against ED/ES patient leakage.
2. **What was found and verified:** All mandatory pass criteria from `wave2_split_task.md` and all four coordinator review notes from `wave2_split_review_notes.md` were verified with 26 dedicated regression tests in `tests/test_dataset_split_safety.py`:
   - Legitimate summary metadata (`n_patients`, `num_patients`, `n_volumes`, `num_volumes`, `seed`, `created_at`) is safely tolerated, and tracked manifest `splits/acdc_patient_split_seed42.json` parses cleanly without modification.
   - All membership representations (`train_cases`, `train_volumes`, `train`, nested `splits`) are evaluated jointly; identical redundant sets are accepted, and conflicting representations are rejected rather than silently relying on first-key-wins.
   - Extension stem collisions (e.g. `case1.npy` and `case1.npz` in the same directory) are detected and rejected before creating dictionaries.
   - `validate_dataset_splits` outputs strictly JSON-serializable string-path descriptors for `effective_records`, matching `ACDCDataset` identities and paths while preventing CLI/report serialization crashes.
   - Controlled ED-train / ES-validation leakage for the same patient fails strictly before model training (tested under manifest, directory tags, and Astra's conflict probe).
   - Missing manifest entries, extra discovered cases, duplicate list entries, cross-split case duplication, and distinct file collisions with identical case IDs are rejected with explicit errors.
   - Recursive raw NIfTI traversal preserves specific split tags without false duplicate case reports, and discovery reordering preserves record identities and hash signatures.
3. **What's left / Next steps:** Work is strictly bounded to Wave 2.1. The full project test suite cleanly passes (**162 passed, 1 pre-existing warning in 4.95s**), focused tests cleanly pass (**26 passed in 0.49s**), and all Python source files byte-compile with zero errors. Content-hash duplicate detection was explicitly deferred as allowed by the specification. Waves 2.2 (Entropy), 2.3 (Geometry), and Waves 3–4 remain unopened.

---

## 2. Coordinator Review Notes Resolution

The four specific guidance items from `reports/astra_impl_review/2026-09-08/wave2_split_review_notes.md` were implemented and verified as follows:

| Review Note | Issue Identified | Resolution Implemented | Test Verification |
| :--- | :--- | :--- | :--- |
| **1. Summary Metadata & Tracked Manifest** | Independent probe `read_split_manifest('splits/acdc_patient_split_seed42.json')` failed on summary key `n_patients`. Tracked manifest must not be rewritten. | Added `SUMMARY_METADATA_KEYS` in `data/common.py` (`n_patients`, `num_patients`, `n_volumes`, `num_volumes`, `seed`, `created_at`, `notes`, `description`, etc.) to permit legitimate scalar summary metadata without treating them as unrecognized split keys. Tracked manifest was left untouched. | `test_tracked_manifest_parse_regression`: loads `splits/acdc_patient_split_seed42.json`, verifying 160 train cases and 40 val cases parse cleanly with no cross-split leakage. |
| **2. Multi-Representation Validation** | Need to validate every membership representation (`train_cases`, `train_volumes`, `train`, `training`, nested `splits`), accepting redundant identical sets but rejecting conflicting memberships (no first-key-wins). | Updated `read_split_manifest` to collect cases from all matching split representation keys; asserts uniqueness within each list, compares lists across keys, accepts identical redundant representations, and raises `ValueError("Conflicting representations...")` on any mismatch. | `test_manifest_redundant_identical_representations_accepted` and `test_manifest_conflicting_representations_rejected`. |
| **3. Stem Collision Detection in `_paired_files`** | Stem-based dictionary indexing silently overwrote same-stem files with different extensions (`.npy` vs `.npz`). | Upgraded `_paired_files` in `data/acdc.py` to check for collision before dictionary insertion: `if stem in images: raise ValueError(f"Ambiguous file stem collision: found multiple image files with stem '{stem}'")`. | `test_paired_files_stem_collision_rejected`. |
| **4. JSON-Safe Validator Output** | `validate_dataset_splits` returned raw dataclasses in `effective_records`, causing serialization issues for downstream CLI/report callers. | Converted `effective_records` in `validate_dataset_splits` to return pure JSON-safe dicts with string paths (`image_path`, `mask_path`, `case_id`, `patient_id`, `split`, etc.). | `test_validator_result_is_strictly_json_serializable`: asserts `json.dumps(result)` succeeds without error. |

---

## 3. Root Cause Analysis and Architectural Fixes

### 3.1 Root Cause Analysis

1. **Split Authority Inversion:** In previous code, `resolve_acdc_records` returned directory-tagged records (`explicit = [r for r in records if r.split ...]`) before inspecting an explicit manifest, allowing conflicting directory tags to silently override the manifest.
2. **Validator / Loader Divergence:** `validate_dataset_splits` re-implemented split logic separately by inspecting the manifest or calling `patient_level_split` directly without using actual selected records, allowing fixtures (like Astra's `patient001_ED` training / `patient001_ES` validation probe) to pass the validator while leaking in the dataset loader.
3. **Masking of Duplicates:** `read_split_manifest` used `set(values)`, silently hiding duplicate entries within a manifest split list or across splits. Similarly, `discover_acdc_records` used `if case_id in seen: continue`, silently swallowing real collisions where two distinct files on disk shared the same case ID.
4. **Traversal Duplication:** Raw NIfTI discovery using `root.rglob` traversed files twice (once from root and once from child split folders like `training/`), risking loss of tags or false duplicate reporting.

### 3.2 Architectural Solutions

1. **Authoritative Effective Resolver (`resolve_effective_acdc_splits`):**
   - Encapsulates all split resolution and validation in `src/self_audit/data/acdc.py`.
   - Returns immutable `EffectiveSplits(splits=..., signature=..., strategy=..., manifest_path=...)`.
   - `ACDCDataset` and `validate_dataset_splits` both delegate to this single resolver.
2. **Strict Manifest Governance:**
   - When a manifest is supplied, it is checked first. If the file is missing, it fails immediately with `FileNotFoundError`.
   - Cases declared in the manifest must match discovered cases exactly: missing cases or extra discovered cases raise `ValueError`.
   - If a record has a directory tag that contradicts its manifest assignment, `resolve_effective_acdc_splits` halts with `ValueError("Manifest-vs-tag conflict...")`.
3. **Strict Duplicate Rejection in Manifests:**
   - `read_split_manifest` validates uniqueness within each split list (rejecting `["case1", "case1"]`) and across split lists.
   - Validates all membership representations without first-key-wins.
4. **Collision vs Safe Traversal Deduplication in Discovery:**
   - `discover_acdc_records` tracks file pair identities `(resolved_image_path, resolved_mask_path)`.
   - When an identical file pair is encountered again (e.g. from recursive traversal), it upgrades an untagged record with a specific tag (e.g. `"train"`), while failing on conflicting tags.
   - When different file paths share the same `case_id`, it fails immediately as a real collision.
   - In `_paired_files`, stem collisions between different extensions are detected prior to dictionary mapping.
5. **Deterministic Split Signature (`compute_split_signature`):**
   - SHA-256 hash computed over canonically sorted split names and canonically sorted case IDs: `";".join(f"{split}=" + ",".join(sorted(case_ids)))`.
   - Independent of file discovery order or dictionary key ordering.
   - Exposed on both `validate_dataset_splits` and `ACDCDataset.split_signature`.

---

## 4. Detailed Changes by File

### 4.1 `src/self_audit/data/common.py`
- Defined `CANONICAL_SPLITS = ("train", "val", "test")`, `SUPPORTED_SPLIT_ALIASES`, and `SUMMARY_METADATA_KEYS`.
- Added `canonicalize_split_alias(alias)` helper to validate and map aliases (`"training"` -> `"train"`, `"validation"` -> `"val"`, `"testing"` -> `"test"`) while failing on unsupported aliases.
- Added `compute_split_signature(splits)` helper to compute deterministic SHA-256 hash signatures.
- Updated `read_split_manifest(path)`:
  - Supports summary metadata keys without treating them as unrecognized split keys.
  - Checks and validates all membership representations (`{split}_cases`, `{split}_volumes`, `{split}`, nested `splits`).
  - Accepts redundant identical representation sets; rejects conflicting representations.
  - Enforces duplicate checking within lists and across split lists.
  - Runs `validate_patient_split` to ensure no cross-split patient leakage.

### 4.2 `src/self_audit/data/acdc.py`
- Defined `EffectiveSplits` frozen dataclass with indexing and `.get()` helpers.
- Updated `_paired_files`:
  - Detects and rejects stem collisions across different file extensions (`.npy` vs `.npz`).
- Updated `discover_acdc_records`:
  - Tracks `pair_identity = (image_path.resolve(), mask_path.resolve())`.
  - Upgrades untagged records when visited from tagged subdirectories.
  - Fails on conflicting tags for the same file.
  - Fails on real collisions (different file paths sharing the same `case_id`).
  - Returns deterministically sorted records.
- Implemented `resolve_effective_acdc_splits`:
  - Rejects empty record lists.
  - Enforces alias normalization on `train_split`, `val_split`, `test_split`.
  - Manifest strategy: verifies file existence, exact case alignment, rejects tag conflicts, verifies required `train` and `val`, validates patient disjointness.
  - Tagged strategy: rejects mixed tagged/untagged ambiguity, verifies required `train` and `val`, validates patient disjointness.
  - Fallback strategy: deterministic `patient_level_split` for untagged records.
- Updated `resolve_acdc_records` to delegate to `resolve_effective_acdc_splits`.
- Updated `ACDCDataset.__init__` to use `resolve_effective_acdc_splits` and record `self.split_signature`.

### 4.3 `src/self_audit/training/_utils.py`
- Updated `validate_dataset_splits` to delegate entirely to `resolve_effective_acdc_splits`.
- Added robust import fallback (`self_audit` / `src.self_audit`).
- Exposes `split_signature`, `effective_identities`, and `effective_records`.
- Converts `effective_records` to JSON-safe dictionary structures with string paths.
- Exposes `content_hash_duplicate_detection: {"executed": False, "status": "deferred"}`.

### 4.4 `src/self_audit/data/__init__.py`
- Exported `CANONICAL_SPLITS`, `SUPPORTED_SPLIT_ALIASES`, `SUMMARY_METADATA_KEYS`, `canonicalize_split_alias`, `compute_split_signature`, `EffectiveSplits`, `resolve_effective_acdc_splits`, and `resolve_acdc_records`.

### 4.5 `tests/test_dataset_split_safety.py`
- Created 25 dedicated unit and integration tests covering all 7 pass criteria and all 4 coordinator review points.

---

## 5. Verification Commands and Outputs

### 5.1 Python Source Compilation Check
Executed:
```bash
rtk python3 -m py_compile $(find src tests scripts -name "*.py")
```
Output:
```
(Exit code 0, clean compilation across all Python files)
```

### 5.2 Focused Test Suite Execution
Executed:
```bash
rtk python3 -m pytest tests/test_dataset_split_safety.py -v
```
Output:
```
============================= test session starts ==============================
platform darwin -- Python 3.11.16, pytest-9.1.1, pluggy-1.6.0 -- /Users/alvinluong/miniforge3/bin/python3
cachedir: .pytest_cache
rootdir: /Users/alvinluong/Self-Audit
plugins: anyio-4.14.2
collecting ... collected 26 items

tests/test_dataset_split_safety.py::test_controlled_ed_train_es_val_leakage_manifest_fails_before_training PASSED [  3%]
tests/test_dataset_split_safety.py::test_controlled_ed_train_es_val_leakage_tags_fail_before_training PASSED [  7%]
tests/test_dataset_split_safety.py::test_controlled_ed_train_es_val_astra_probe_tag_vs_manifest_conflict PASSED [ 11%]
tests/test_dataset_split_safety.py::test_validator_and_loader_parity_with_manifest PASSED [ 15%]
tests/test_dataset_split_safety.py::test_validator_and_loader_parity_tag_only PASSED [ 19%]
tests/test_dataset_split_safety.py::test_validator_and_loader_parity_untagged_deterministic_fallback PASSED [ 23%]
tests/test_dataset_split_safety.py::test_manifest_missing_discovered_cases_rejected PASSED [ 26%]
tests/test_dataset_split_safety.py::test_manifest_extra_discovered_cases_rejected PASSED [ 30%]
tests/test_dataset_split_safety.py::test_manifest_duplicate_list_entries_rejected PASSED [ 34%]
tests/test_dataset_split_safety.py::test_manifest_duplicate_across_splits_rejected PASSED [ 38%]
tests/test_dataset_split_safety.py::test_real_duplicate_files_collision_rejected PASSED [ 42%]
tests/test_dataset_split_safety.py::test_real_duplicate_nifti_files_collision_rejected PASSED [ 46%]
tests/test_dataset_split_safety.py::test_manifest_vs_tag_conflict_fails_clearly PASSED [ 50%]
tests/test_dataset_split_safety.py::test_mixed_tagged_and_untagged_ambiguity_fails PASSED [ 53%]
tests/test_dataset_split_safety.py::test_optional_test_absent_accepted PASSED [ 57%]
tests/test_dataset_split_safety.py::test_missing_required_train_or_val_rejected PASSED [ 61%]
tests/test_dataset_split_safety.py::test_configured_aliases_respected PASSED [ 65%]
tests/test_dataset_split_safety.py::test_unsupported_alias_rejected PASSED [ 69%]
tests/test_dataset_split_safety.py::test_raw_nifti_recursive_traversal_deduplication PASSED [ 73%]
tests/test_dataset_split_safety.py::test_discovery_order_invariance_signature PASSED [ 76%]
tests/test_dataset_split_safety.py::test_tracked_manifest_parse_regression PASSED [ 80%]
tests/test_dataset_split_safety.py::test_manifest_redundant_identical_representations_accepted PASSED [ 84%]
tests/test_dataset_split_safety.py::test_manifest_conflicting_representations_rejected PASSED [ 88%]
tests/test_dataset_split_safety.py::test_paired_files_stem_collision_rejected PASSED [ 92%]
tests/test_dataset_split_safety.py::test_validator_result_is_strictly_json_serializable PASSED [ 96%]
tests/test_dataset_split_safety.py::test_manifest_unrecognized_split_keys_rejected PASSED [100%]

============================== 26 passed in 0.49s ==============================
```

### 5.3 Wave 1 Regression Test Suite Execution
Executed:
```bash
rtk python3 -m pytest tests/test_metric_contract_and_replay.py -v
```
Output:
```
============================== 32 passed in 2.78s ==============================
```

### 5.4 Full Project Test Suite Execution
Executed:
```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 rtk python -m pytest -q -p no:cacheprovider tests
```
Output:
```
........................................................................ [ 44%]
........................................................................ [ 88%]
..................                                                       [100%]
=============================== warnings summary ===============================
tests/test_self_audit_core.py::test_dynamic_window_coordinates_and_backward_are_valid
  /Users/alvinluong/Self-Audit/tests/test_self_audit_core.py:32: UserWarning: Converting a tensor with requires_grad=True to a scalar may lead to unexpected behavior.
  Consider using tensor.detach() first. (Triggered internally at /Users/runner/work/pytorch/pytorch/torch/csrc/autograd/generated/python_variable_methods.cpp:823.)
    assert float(coordinates.min()) >= -1.0

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
162 passed, 1 warning in 4.95s
```

---

## 6. Unresolved Concerns and Non-Claims

1. **Synthetic Fixture Disclaimer:**
   > [!IMPORTANT]
   > All verification and tests in Wave 2.1 were performed using controlled programmatic fixtures, mock directories, and synthetic manifests. No claim of real raw-dataset or clinical checkpoint validation is made from these tests.

2. **Deferred Capabilities:**
   - Content-hash duplicate detection is optional under `wave2_split_task.md` and has been explicitly deferred (`content_hash_duplicate_detection: {"executed": False, "status": "deferred"}`).
   - Geometry metadata guards (Wave 2.3) and typed entropy APIs (Wave 2.2) remain untouched and deferred to their respective wave tasks.

3. **Preservation of System Invariants:**
   - No architecture, model, or loss changes were introduced.
   - Tracked manifests under `splits/` were not modified.
   - No commits or pushes were executed.
