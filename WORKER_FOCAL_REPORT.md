# Worker Report: B8 Supervised Focal Loss & Adaptive LR Scheduler

## Executive Summary
This report documents the implementation of the authorized full-model `ema_full` B8 features:
1. Opt-in `supervised_loss` with Focal Loss + present-foreground Dice.
2. Opt-in `lr_scheduler` with ReduceLROnPlateau (`train_plateau`) driven strictly by supervised TRAIN loss.
3. Checkpoint/resume support retaining complete scheduler state dicts, step counters, and partial metric accumulations with strict schema validation.
4. Comprehensive test suite ensuring mathematical equivalence (`gamma=0 == CE`), ignored-pixel invariance, extreme logit stability, strict resume validation, and bit-exact uninterrupted vs. resumed parity.

All existing default behaviors and numerics remain 100% backward compatible.

---

## Implemented Components & Files

### 1. `self_audit/losses.py`
- Implemented `validate_supervised_loss_config(config)`:
  - Supports default/omitted/`"ce_dice"` as well as `"focal_dice"`.
  - Validates `focal_dice` keys (`name`, `gamma`, `focal_weight`, `dice_weight`), ensuring finite non-negative values and non-zero weight sums.
- Implemented `focal_loss(logits, labels, gamma=2.0)`:
  - Multi-class focal loss across all 4 classes without class/alpha reweighting.
  - Denominator normalized by valid pixels (`labels != 255`).
  - Strict validation of labels: values must be in `{0, 1, 2, 3}` or `255` (invalid labels like `4` are rejected).
  - Exact branch for `gamma=0` returning `F.cross_entropy(logits, labels, ignore_index=255)`.
- Updated `supervised_loss(logits, labels, config=None)`:
  - Calls `focal_loss` directly when `config["name"] == "focal_dice"`.
  - Computes canonical present-foreground Dice identically to v2 formula.
  - Combines `focal_weight * focal_loss + dice_weight * dice_loss`.

### 2. `self_audit/lr_schedule.py`
- Implemented `validate_lr_scheduler_config(config)`:
  - Validates exact train_plateau parameters: `interval_steps`, `factor`, `patience`, `threshold`, `threshold_mode`, `cooldown`, `min_lr_factor`, `eps`.
- Implemented `validate_scheduler_state_dict(state_dict, ...)`:
  - Enforces exact 15-key PyTorch `ReduceLROnPlateau` schema.
  - Validates immutable configuration settings (`factor`, `patience`, `threshold`, `mode="min"`, `cooldown`, `eps`, `min_lrs`, `default_min_lr`).
  - Validates typed integer counters (`last_epoch == num_updates`, `num_bad_epochs`, `cooldown_counter`, `metric_count`).
  - Verifies step, warmup, and interval consistency: `expected_updates = (step - warmup) // interval`.
  - Enforces zero metric sum for empty accumulation window (`metric_count == 0 => metric_sum == 0.0`).
  - Enforces `best == +inf` before first update, and finite positive float thereafter.
  - Enforces `mode_worse == +inf`.
  - Validates optimizer LR matches scheduler `_last_lr` within `[min_lr, initial_lr]`.
- Implemented `TrainPlateauScheduler`:
  - Instantiates dual `ReduceLROnPlateau` instances for AdamW body and SGD head.
  - Sets `min_lr = initial_lr * min_lr_factor` preserving exact head/body LR ratio.
  - Accumulates metric strictly after warmup (`step >= c.warmup_steps`).
  - Evaluates mean loss and steps schedulers after each complete `interval_steps` block, after student and EMA finalization.

### 3. `self_audit/engine.py`
- Preserved canonical `DEFAULT` dictionary without default injected keys; allowed `OPTIONAL_FIELDS = {"supervised_loss", "lr_scheduler"}` in `validate_config`.
- Restricts `focal_dice` and `train_plateau` to `method == "ema_full"` and `window["mode"] == "none"`.
- `supervised_step`: accepts `loss_config=None`, forwarding to `supervised_loss`.
- Checkpointing: saves scheduler state dicts in `weights.pt` and metadata in `MANIFEST.json`; keeps `None` when scheduler is not enabled.
- Resume: executes strict validation prior to mutating model or optimizer weights.
- Step logging: logs `body_lr`, `head_lr`, `next_body_lr`, `next_head_lr`, `monitored_metric`, `scheduler_updated`, `lr_reduced`, and detailed `lr_scheduler` diagnostics in `steps.jsonl`.

### 4. `tests/test_focal_adaptive.py`
Added 16 comprehensive unit and regression tests:
- `test_gamma0_equals_cross_entropy`: verifies exact match between focal (gamma=0) and cross entropy.
- `test_ignored_pixel_zero_effect_and_gradient`: verifies zero gradient and invariant loss on label 255.
- `test_extreme_logits_finite`: verifies finite loss and gradients with extreme logits (+/- 1e4).
- `test_empty_and_malformed_loss_config_rejected`: tests empty support, label 4 ontology rejection, shape mismatch, negative gamma, zero weights, and mode restrictions.
- `test_lr_scheduler_config_validation`: tests invalid scheduler configs and mode restrictions.
- `test_plateau_warmup_timing_and_reductions`: verifies no accumulation during warmup, reduction after patience, ratio preservation, and minimum LR bounds.
- `test_resume_state_missing_or_tampered_rejected`: tests state dict tampering, negative counters, nonfinite sums, last_epoch mismatch, and optimizer LR mismatch.
- `test_parameterized_warmup_state_rejections`: 6 parameterized tests covering negative inf best, NaN LR, nonzero empty window sum, out-of-bounds LR, and mode_worse.
- `test_uninterrupted_vs_resumed_parity_focal_adaptive`: parameterized across steps 5 (middle of window) and 7 (after forced decay at 6), with `interval=2, patience=0, threshold=1.0, threshold_mode='abs'`, asserting forced decay at step 6 and verifying bit-exact student, teacher, full optimizer states, and complete scheduler states.
- `test_old_defaults_unchanged`: verifies default recipes produce null scheduler checkpoints and identical original behavior.

---

## Test Evidence & Provenance

- **Python Interpreter**: `/Users/alvinluong/miniforge3/bin/python` (Python 3.11.16)
- **Repository Integrity**: `python scripts/check_repo.py` -> `{"status": "PASS", "architecture": "dinov3_convnext_tiny_native_decoder_v2", "supervision": "semi_supervised; dense TRAIN subset incl audit; not no-GT"}`
- **Test Suite Results**:
  - `pytest tests/test_focal_adaptive.py`: 16 passed in 1.81s
  - `pytest -q`: **125 passed, 1 warning in 10.72s** (0 failures)
- **Source SHA-256 Digest**: `e3b5b9469d5378b50677b7225eddd16421d4de856ee6d883b4dfc719f98905d2`
- **Git Branch**: `QuocKhanhLuong/b8-focal-adaptive-20261031`
- **Files Modified/Created**:
  - `self_audit/losses.py`
  - `self_audit/lr_schedule.py`
  - `self_audit/engine.py`
  - `tests/test_focal_adaptive.py`
  - `WORKER_FOCAL_REPORT.md`
