# Parallel Runtime Portability & Shared Loader Hardening Report

## Executive Summary

As part of Wave 4 implementation in the runtime-hardening audit for repository `Self-Audit` (base `25f429a46255ceb59997ae5178939d5d40b2ea27`), we implemented the shared-loader runtime portability hardening, canonical nonpersistent worker configurations, and fail-closed finite state validation without clobbering disjoint Wave 3 trainer lifecycle work.

### Owned Files Modified / Created
- `src/self_audit/training/_utils.py`: Enhanced finite validation, added public reusable `validate_checkpoint_finite_state`, updated `load_checkpoint` to fail-closed on non-finite components, and updated `bind_evaluation_checkpoint` and `bind_existing_evaluation_state` to stage full checkpoint payloads on CPU while preserving model target device.
- `configs/self_audit_full.yaml`: Set `persistent_workers: false` to resolve R11 exact resume limitation.
- `configs/self_audit_full_mnms.yaml`: Set `persistent_workers: false` to resolve R11 exact resume limitation.
- `tests/test_runtime_portability.py`: 20 comprehensive unit tests covering all portability, map_location, finite-state negative cases, and config contracts.
- `parallel_portability.md`: This delivery report.

---

## Technical Details

### 1. Shared Loader CPU Staging (`bind_evaluation_checkpoint` & `bind_existing_evaluation_state`)
- **Problem**: When evaluating checkpoints or binding existing evaluation state with `map_location=device` (e.g. GPU), `torch.load` allocated the entire checkpoint dictionary—including large optimizer state buffers, momentum tensors, and RNG states—directly on the target GPU. On memory-constrained devices (e.g., 12GB GPUs), this caused memory bloat or OOM during evaluation and calibration.
- **Remediation**:
  - Both `bind_evaluation_checkpoint` and `bind_existing_evaluation_state` now invoke `load_checkpoint` with `map_location="cpu"`.
  - For `bind_evaluation_checkpoint`: The model's target device is preserved. `model.load_state_dict(payload["model"])` copies the CPU weights into the model parameters in-place on the model's existing device; it leaves the model on its existing model device and does not move a CPU model to the requested `map_location`. Unused optimizer and RNG states remain strictly on CPU memory and never allocate GPU VRAM.
  - For `bind_existing_evaluation_state`: `load_checkpoint` stages the payload on CPU, computes the digest without allocating GPU memory, and verifies live model state equality without touching the live model.

### 2. Reusable Finite-State Validation Contract (`validate_checkpoint_finite_state`)
- **API Contract**:
  ```python
  def validate_checkpoint_finite_state(payload: Mapping[str, Any]) -> None:
      """Validate that model, optimizer, scheduler, and scaler states contain only finite values.

      Inspects tensors and numeric scalars across all present training state components
      ('model', 'state_dict', 'optimizer', 'scheduler', 'scaler').
      Raises FloatingPointError if any non-finite (NaN or Inf) tensor or scalar is detected.
      Does not sanitize or mutate any training state.
      """
  ```
- **Coverage**:
  - Model tensors (under `'model'` or `'state_dict'` or `nn.Module`): checks all tensors for NaN and Inf.
  - Optimizer state: checks both tensor buffers (e.g., Adam `exp_avg`, `exp_avg_sq`) and numeric scalars in `param_groups` (e.g., `lr`, `eps`, `betas`, `weight_decay`).
  - Scheduler state: checks numeric scalar fields and lists (e.g., `base_lrs`, `_last_lr`, `gamma`).
  - Scaler state: checks numeric scalars (e.g., `scale`, `growth_factor`, `backoff_factor`).
  - `_finite_tree` supports `torch.Tensor`, `np.ndarray`, `float`, `np.floating`, `complex`, `Mapping`, and collections (`list`, `tuple`, `set`).
- **Fail-Closed Integration**:
  - Integrated directly into `load_checkpoint`: unconditionally validates all present training state components before returning or restoring. Checkpoints containing corrupted optimizer/scheduler/scaler states fail closed with `FloatingPointError` even if `optimizer=None` or `scheduler=None` was passed.
  - Available for import and use by direct resume in `unified_trainer.py` or root orchestrator.

### 3. Canonical Nonpersistent Worker Settings (R11 Remediation)
- **Problem**: In `configs/self_audit_full.yaml` and `configs/self_audit_full_mnms.yaml`, `dataloader.persistent_workers` was set to `true`. Resuming with augmenting persistent workers failed strict execution-config identity or was rejected because dataloader worker RNG states cannot be serialized.
- **Remediation**:
  - Updated `persistent_workers: false` in both `configs/self_audit_full.yaml` and `configs/self_audit_full_mnms.yaml`.
  - Preserved all other parameters: `num_workers: 2`, `pin_memory: true`, `prefetch_factor: 2`.
  - Verified with `load_unified_config` that both configurations parse and validate without error.

---

## Verification & Test Results

### 1. Focused Tests on Python 3.11 / PyTorch 2.13
- `tests/test_runtime_portability.py`:
  `20 passed in 1.50s`
- `tests/test_runtime_checkpoint.py`:
  `27 passed in 2.85s`
- `tests/test_checkpoint_binding.py`:
  `55 passed in 12.62s`

### 2. Focused Tests on Compatibility Python 3.10 / PyTorch 2.4.1 (`/private/tmp/self-audit-torch241/bin/python`)
- `tests/test_runtime_portability.py`:
  `20 passed in 1.09s`
- `tests/test_runtime_checkpoint.py`:
  `27 passed in 2.20s`

All tests passed with zero failures or warnings.
