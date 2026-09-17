# Wave 1 Delivery: Checkpoint Portability, RNG Restoration, and Atomic Durability

**Execution Date:** 2026-09-10
**Status:** Completed (Revised with all edge-probe fixes)
**Target Environment:** PyTorch 2.4.1 (Python 3.10.21, macOS Darwin)
**Local Environment:** PyTorch 2.13.0 (Python 3.11.16, macOS Darwin)

---

## 1. Overview & Root Cause Analysis

Wave 1 addresses critical checkpoint serialization, deserialization, RNG stream fidelity, and durability defects identified in the audit plan and coordinator probes:

### 1.1 Root PyTorch 2.4.1 Serialization Probes & Revisions
1. **Unsafe Containers (`set` and `frozenset`):**
   - *Issue:* `builtins.set` and `builtins.frozenset` fail with `Unsupported global: GLOBAL builtins.set` under PyTorch 2.4.1 default `weights_only=True`.
   - *Fix:* Explicitly reject `set` and `frozenset` in both training (`_normalize_training_tree`) and metadata (`_normalize_metadata_tree`) trees with `CheckpointSerializationError` and exact dotted/indexed paths.
2. **String and Byte Subclasses (`torch.torch_version.TorchVersion`):**
   - *Issue:* `torch.__version__` is an instance of `torch.torch_version.TorchVersion` (inheriting from `str`). `torch.load(..., weights_only=True)` fails with `Unsupported global: GLOBAL torch.torch_version.TorchVersion`.
   - *Fix:* Recursively normalize string subclasses to pure builtin `str(val)`.
3. **`bytes` and `bytearray` Rejection:**
   - *Issue:* `bytes` and `bytearray` trigger `WeightsUnpickler error: Unsupported global: GLOBAL _codecs.encode` and `builtins.bytearray` under `weights_only=True` in PyTorch 2.4.1.
   - *Fix:* Strictly reject `bytes` and `bytearray` in both training and metadata normalization with `CheckpointSerializationError` and exact path reporting.
4. **Mapping Key Normalization & Collision Detection:**
   - *Issue:* Mapping keys like `np.int64(2)` fail under `weights_only=True` with `Unsupported global: GLOBAL numpy.core.multiarray.scalar`. Furthermore, root-level mapping keys previously bypassed normalization, and collisions between coerced keys (e.g. `Path` vs `str`) could silently overwrite entries.
   - *Fix:* Implemented `_normalize_mapping_key`:
     - Normalizes `int` and `np.integer` keys to builtin `int` (preserving optimizer integer parameter IDs).
     - Normalizes `str`, `Path`, and `os.PathLike` keys to builtin `str`.
     - Strictly rejects boolean (`bool`, `np.bool_`), floating-point (`float`, `np.floating`), and custom object keys.
     - Detects and rejects duplicate normalized keys with `CheckpointSerializationError: Mapping key collision`.
     - Applied recursive key normalization and collision detection at both root payload level (`normalize_checkpoint_payload`) and nested mapping levels.
5. **Reserved-Key Override Prevention:**
   - *Issue:* In `save_checkpoint`, checking reserved keys before normalizing `extra` allowed non-string keys such as `Path("epoch")` to evade `set(payload).intersection(extra)` and overwrite critical fields like `epoch=-999`.
   - *Fix:* Normalize `extra` via `normalize_metadata_tree` before performing the reserved key intersection check.
6. **Custom Tensor Subclass Rejection:**
   - *Issue:* Custom `torch.Tensor` subclasses (e.g., `class CustomTensor(torch.Tensor): pass`) retain their subclass across `detach().cpu().clone()`, creating checkpoints that fail under `weights_only=True`.
   - *Fix:* Fail closed with `CheckpointSerializationError` for any tensor where `type(val) is not torch.Tensor and type(val) is not torch.nn.Parameter` in both training and metadata normalization.
7. **`rng_state` Recursive Validation:**
   - *Issue:* `rng_state` could potentially bypass recursive validation if treated as a special opaque object.
   - *Fix:* Processed `rng_state` through `_normalize_training_tree` in `normalize_checkpoint_payload`, validating all internal containers and tensors.
8. **Public Self-Normalizing API (`atomic_save_torch`):**
   - *Issue:* Unnormalized payloads passed directly to `atomic_save_torch` could write non-compliant checkpoints.
   - *Fix:* `atomic_save_torch` now self-normalizes via `normalize_checkpoint_payload` before writing to disk using internal primitive `_atomic_save_torch_raw`.

---

### 1.2 RNG Restoration & PyTorch 2.4 Operator Hardening
1. **NumPy MT19937 Keys Dtype Kind Validation & Lossless Restoration:**
   - *Issue:* `_restore_rng_state` previously accepted object arrays (`dtype=object`) and complex arrays (`dtype=complex`) with lossy truncation/casting.
   - *Fix:* Enforce NumPy dtype kind `keys_arr.dtype.kind in ('i', 'u', 'f')`. Strictly reject object (`'O'`), complex (`'c'`), boolean (`'b'`), and string (`'U'/'S'`) dtypes before any comparison or cast. Real floats are verified to be finite and integral (`keys == np.floor(keys)`).
2. **PyTorch 2.4.1 `torch.uint32` Operator Limitation (`lt_cpu`):**
   - *Issue:* In PyTorch 2.4.1, evaluating `(keys < 0)` directly on a `torch.uint32` tensor raises `RuntimeError: "lt_cpu" not implemented for 'UInt32'`.
   - *Fix:* Converted tensors directly to CPU NumPy via `keys.detach().cpu().numpy()` prior to validation, allowing unsigned 32-bit operations without invoking unsupported PyTorch operators.
3. **Non-Truncating Scalar Validation for `pos` and `has_gauss`:**
   - *Issue:* Calling `int(pos)` or `int(has_gauss)` silently truncated fractional floats (e.g., `1.9` -> `1`).
   - *Fix:* Explicitly reject `bool`, `np.bool_`, non-finite floats (`NaN`, `Inf`), and fractional floats before casting. Enforce $0 \le pos \le 624$ and $has\_gauss \in \{0, 1\}$.
4. **`cached_gaussian` Validation:**
   - Enforce that `cached_gaussian` is a real, non-boolean, finite float.
5. **Scalar Progress Counters:**
   - Hardened `_require_integer` to reject booleans (`bool`, `np.bool_`), floats, and negative numbers for `epoch`, `global_step`, and `optimizer_step`.

---

### 1.3 Durability & Durability Invariants
1. **Unswallowed Fsync:**
   - `atomic_save_torch` flushes and executes `os.fsync(handle.fileno())`. Any `OSError` (e.g., `ENOSPC`) propagates immediately, the temporary file is unlinked, and the pre-existing target checkpoint remains untouched.
2. **Directory Fsync:**
   - Best-effort directory fsync on the parent folder to persist directory entry metadata.
3. **Non-Masking Cleanup:**
   - Cleanup failures in `finally` do not mask primary serialization or IO errors.
4. **Nonfinite Metadata Sentinel Policy:**
   - Explicit nonfinite sentinel policy without zeroing: `NaN`, `Inf`, `-Inf`, and `None` are preserved in metadata (`config`, `provenance`, `extra`) as standard Python primitives without alteration.

---

## 2. Modified & Created Files

1. **`src/self_audit/serialization.py` (New):**
   - `CheckpointSerializationError` (inherits from `TypeError` and `ValueError`).
   - `SAFE_TENSOR_DTYPES` frozenset.
   - `_normalize_training_tensor`, `_normalize_training_tree`, `_normalize_metadata_tree`.
   - `_normalize_mapping_key` with key validation, type coercion, and collision detection.
   - `normalize_checkpoint_payload` with root-level key normalization and collision detection.
   - `_atomic_save_torch_raw` and `atomic_save_torch` self-normalizing safe public API.
2. **`src/self_audit/training/_utils.py` (Modified):**
   - Updated `_require_integer` to reject `bool` and `np.bool_`.
   - Updated `_rng_state` to store MT19937 keys as lossless `int64` tensors on CPU.
   - Hardened `_restore_rng_state` with dtype kind validation (`('i', 'u', 'f')`), object/complex rejection, non-truncating `pos`/`has_gauss` validation, `cached_gaussian` validation, and CPU NumPy conversion bypassing `lt_cpu`.
   - Updated `save_checkpoint` to normalize `extra` before checking reserved keys and route persistence through `atomic_save_torch`.
   - Added `exact_cuda` parameter to `load_checkpoint`.
3. **`tests/test_runtime_checkpoint.py` (New):**
   - 27 dedicated regression tests covering all Wave 1 requirements and edge-probe fixes.
4. **`reports/runtime_hardening/2026-09-10/wave1.md` (New):**
   - Detailed delivery report with root cause analysis and exact test evidence.

---

## 3. Test Verification Matrix & Honest Collection Evidence

### Exact Test Invocations and Results

#### Target Environment: Python 3.10.21 / PyTorch 2.4.1 CPU
Target command prefix:
`rtk proxy env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:. OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /private/tmp/self-audit-torch241/bin/python -m pytest`

1. **Dedicated Runtime Hardening Suite:**
   ```
   Command: rtk proxy env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:. OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /private/tmp/self-audit-torch241/bin/python -m pytest tests/test_runtime_checkpoint.py -v
   Result:  collected 27 items, 27 passed in 1.93s
   ```

2. **Coordinator Target Suite (`test_runtime_checkpoint.py`, `test_checkpoint_binding.py`, `test_unified_trainer.py`):**
   ```
   Command: rtk proxy env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:. OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /private/tmp/self-audit-torch241/bin/python -m pytest tests/test_runtime_checkpoint.py tests/test_checkpoint_binding.py tests/test_unified_trainer.py
   Result:  collected 116 items, 116 passed in 108.23s
   ```
   *Test Count Reconciliation:*
   - `tests/test_runtime_checkpoint.py`: 27 passed (includes 14 new regression tests)
   - `tests/test_checkpoint_binding.py`: 55 passed
   - `tests/test_unified_trainer.py`: 34 passed
   - **Total for this 3-file suite: 116 passed** (previously 102 when `test_runtime_checkpoint.py` had 13 tests).

3. **Full 4-Suite Execution (including `test_self_audit_hardening.py`):**
   ```
   Command: rtk proxy env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:. OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /private/tmp/self-audit-torch241/bin/python -m pytest tests/test_runtime_checkpoint.py tests/test_checkpoint_binding.py tests/test_unified_trainer.py tests/test_self_audit_hardening.py
   Result:  collected 128 items, 128 passed in 109.10s
   ```
   *Test Count Breakdown:*
   - `tests/test_runtime_checkpoint.py`: 27 passed
   - `tests/test_checkpoint_binding.py`: 55 passed
   - `tests/test_unified_trainer.py`: 34 passed
   - `tests/test_self_audit_hardening.py`: 12 passed
   - **Total across 4 files: 128 passed** (the earlier reference to 114 was composed of 102 + 12).

#### Local Environment: Python 3.11.16 / PyTorch 2.13.0
```
Command: rtk env PYTHONPATH=src:. /Users/alvinluong/miniforge3/bin/pytest tests/test_runtime_checkpoint.py -v
Result:  collected 27 items, 27 passed in 2.66s
```

---

## 4. Invariants & Scope Boundaries

1. **Preservation of Model Architecture and Recipes:**
   - No modifications made to network architectures, loss formulations, training recipes, dataset splits, or post-training calibration contracts.
2. **Scope Isolation:**
   - All changes strictly isolated to Wave 1 checkpoint serialization, RNG restoration, durability, and corresponding regression tests.
3. **No Capability Leaks:**
   - Zero capability secrets or credentials written to disk. No `git commit` or `git push` executed.
