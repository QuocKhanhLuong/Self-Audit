# W3.1 producer source identity revision (worker report)

Date: 2026-09-08. Implementer: Antigravity worker taking over from stopped quota-exhausted Claude (`task_0133cd36806b` / `ctx_e7957f772da8`).
Scope: `src/self_audit/provenance.py`, `tests/test_checkpoint_binding.py`, and `reports/astra_impl_review/2026-09-08/wave3_source_identity_worker.md`.
Preserved all concurrent edits from other workers. No objective, model, config, or split changes.

## 1. Root REVISE items addressed

### A. Fixed `SOURCE_SIGNATURE_EXCLUDED_DIRS` to include `src/self_audit/data` without raw-data walks
- **Issue**: `SOURCE_SIGNATURE_EXCLUDED_DIRS` previously included `"data"` globally. Because directory exclusion matched directory basenames everywhere under a root, `src/self_audit/data` was skipped during traversal. Core dataset and preprocessing code (`acdc.py`, `common.py`, `mnms.py`, `transforms.py`, `__init__.py`) was therefore omitted from `source_content_signature`.
- **Fix**: Removed `"data"` from `SOURCE_SIGNATURE_EXCLUDED_DIRS`. Top-level raw data directories (`data/`, `preprocessed_data/`) are kept out of the walk by explicit scoping to `SOURCE_SIGNATURE_ROOTS = ("src/self_audit", "scripts", "configs")`. An additional guard prunes any directory named `"data"` whose relative path is not `"src/self_audit/data"`, guaranteeing no raw-data walks even if an operator places a data directory under another root.

### B. Included active `.sh` and `.ps1` scripts wrappers
- **Issue**: `SOURCE_SIGNATURE_SUFFIXES` only included `(".py", ".yaml", ".yml")`, ignoring active orchestration wrappers in `scripts/` (`run_full_pipeline.sh` and `run_full_pipeline.ps1`).
- **Fix**: Updated `SOURCE_SIGNATURE_SUFFIXES = (".py", ".yaml", ".yml", ".sh", ".ps1")`. Edits to active shell and PowerShell wrappers in `scripts/` are now hashed into the signature.

### C. Bumped `SOURCE_SIGNATURE_VERSION` to 2
- Bumped `SOURCE_SIGNATURE_VERSION = 2` (per coordinator guidance in `msg_a52404860ce5`). This ensures older signatures (which omitted `data` and wrappers) are never mistakenly compared against version 2 signatures as if they shared the same scope contract. No invalid historical calibrations are grandfathered.

### D. Directory and file read errors yield unknown / False (fail-closed)
- **Issue**: `_scoped_source_files` previously had `try: entries = sorted(directory.iterdir()) except OSError: continue`. If a directory was unreadable, it silently continued, collected partial files from remaining directories, and marked the partial signature `known=True`.
- **Fix**: Removed `except OSError: continue` in `_scoped_source_files`. Unreadable directories raise `OSError` (e.g. `PermissionError`). In `source_content_signature`, any `OSError` during file listing or content reading immediately fails closed, returning `source_content_signature = "unknown"` with `source_content_signature_known = False`.

### E. Missing required scope roots fail closed explicitly
- **Issue**: `_scoped_source_files` previously had `if not start.is_dir(): continue`. If a required root (e.g. `configs` or `scripts`) was missing, it skipped that root and computed a partial signature marked known.
- **Fix**: `_scoped_source_files` now explicitly raises `FileNotFoundError(f"Missing required scope root: {relative_root}")` when any root in `SOURCE_SIGNATURE_ROOTS` is missing. Callers like `source_content_signature` catch `OSError` and fail closed, returning `source_content_signature = "unknown"` with `source_content_signature_known = False`.

## 2. Regression tests in `tests/test_checkpoint_binding.py` (55 tests, up from 50)

All existing tests were preserved and updated for version 2 / 6 scoped files. Five new dedicated temp-tree regression tests were added:

1. **`test_temp_tree_regression_data_common_edit`**:
   - Verifies modifying `src/self_audit/data/common.py` changes `source_content_signature`.
   - Verifies reverting `src/self_audit/data/common.py` restores the exact baseline signature.
   - Verifies modifying raw data or adding raw data files in top-level `data/` does not change the signature (no raw-data walks).
2. **`test_temp_tree_regression_wrapper_edits`**:
   - Verifies modifying `scripts/run_full_pipeline.sh` changes the signature, and reverting restores baseline.
   - Verifies modifying `scripts/run_full_pipeline.ps1` changes the signature, and reverting restores baseline.
3. **`test_temp_tree_regression_report_and_test_only_invariants`**:
   - Verifies adding or editing files under `reports/` and `tests/` leaves `source_content_signature` strictly unchanged.
4. **`test_temp_tree_regression_monkeypatched_unreadable_directory`**:
   - Monkeypatches `Path.iterdir` to raise `PermissionError` on a scoped directory (`src/self_audit/data`).
   - Verifies `_scoped_source_files` does not swallow the error and raises `OSError`.
   - Verifies `source_content_signature` returns `"unknown"` with `known=False`.
5. **`test_temp_tree_regression_missing_scope_roots_fail_closed`**:
   - For each required root in `SOURCE_SIGNATURE_ROOTS`, removes the directory from a temp-tree repo.
   - Verifies `_scoped_source_files` explicitly raises `FileNotFoundError`.
   - Verifies `source_content_signature` and `git_source_provenance` fail closed with `"unknown"` and `known=False`.

## 3. Exact verification outputs

### Focused checkpoint binding tests
```
$ PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest -q -p no:cacheprovider tests/test_checkpoint_binding.py
.......................................................                  [100%]
55 passed in 11.29s
```

### Python bytecode compilation check
```
$ python -m compileall -q src/self_audit/provenance.py tests/test_checkpoint_binding.py; echo rc=$?
rc=0
```

### Scoped source files inspection on real repository
```
$ python -c "from src.self_audit.provenance import _scoped_source_files, _REPO_ROOT; files = _scoped_source_files(_REPO_ROOT); print('Total:', len(files)); [print(f.relative_to(_REPO_ROOT)) for f in files if 'data' in str(f) or f.suffix in ('.sh', '.ps1')]"
Total: 57
scripts/run_full_pipeline.ps1
scripts/run_full_pipeline.sh
src/self_audit/data/__init__.py
src/self_audit/data/acdc.py
src/self_audit/data/common.py
src/self_audit/data/mnms.py
src/self_audit/data/transforms.py
```

## 4. Invariants preserved

- Existing interfaces preserved: `source_content_signature`, `git_source_provenance`, `state_digest`, `file_sha256`, and schema shapes.
- Legacy checkpoints without producer record remain unknown (`producer_recorded=False`, `producer_source_content_signature=None`).
- Checkpoint strict state binding and `weights_only=True` load compatibility intact.
- No commit, no push, no training runs.
