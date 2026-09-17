# Candidate C Console Progress & Per-Epoch Metric Telemetry Fix

## 1. Executive Summary

This report documents the resolution and revision of per-epoch metrics and progress bars during native Candidate C training in the Self-Audit codebase.

### Root Causes Identified
1. **Explicit Runner Flag Suppression**: The native runner script `scripts/run_acdc_mnms_candidate_c.sh` explicitly passed `--no_tqdm` to `scripts/train_self_audit.py` in both the ACDC and M&Ms training invocations (introduced upstream in commits `50381e9..0fe756c`), suppressing console progress bars completely by default.
2. **Transient Completed Bars, Truncated Postfix, and Premature Halt Counting**: In `train_epoch` (`src/self_audit/training/unified_trainer.py`), `tqdm` was configured with `leave=False`, causing completed epoch bars to vanish from terminal history. When breaking on bounded `max_steps`, iterable `tqdm` deferred its increment until after yield completion, leaving `n=0` on `max_steps=1` despite consuming the batch. Additionally, breaking immediately on `self.optimizer_step >= max_steps` bypassed `pbar.set_postfix`, dropping telemetry for the final processed batch.
3. **Anonymous Metric Summaries & Output Stream Hygiene**: `_train_impl` previously printed an anonymous `train_loss`, `primary_metric`, and `opt_step` without explicit semantic naming or phase-specific validation metrics (e.g. Dice, real on-policy AUROC / fallback accuracy, local and combined fix/regress F1, harmful acceptance rate). Because the runner script already executes under `python -u`, lack of explicit flushing was not the demonstrated cause of missing output for that runner; however, adding `flush=True` improves general stream and buffering hygiene across arbitrary pipes and non-interactive execution contexts.

---

## 2. Changes Implemented

### A. Native Runner Script (`scripts/run_acdc_mnms_candidate_c.sh`)
- Removed default `--no_tqdm` flags from both native ACDC and M&Ms training invocations.
- Preserved `python -u` unbuffered execution and `tee` logging unchanged.
- Preserved GPU device allocation (`CUDA_VISIBLE_DEVICES=1`), batch size (`BATCH_SIZE=8`), schedule overrides, and independent datasets.
- User CLI opt-out `--no_tqdm` remains fully supported across all entrypoints.

### B. Progress Bar & Training Postfix (`src/self_audit/training/unified_trainer.py`)
- **Preserved Completed Bars**: Configured `tqdm` in `train_epoch` with `leave=True` so completed epoch progress bars remain visible in the terminal log.
- **Accurate Progress Step Counting**: Iterates `train_loader` directly with `total=len(train_loader)` and calls `pbar.update(1)` on batch completion, ensuring bounded execution (e.g. `max_steps=1`) accurately records `n=1, total=4` rather than leaving `n=0`.
- **Robust Progress Bar Closing**: Wrapped batch execution in a `try...finally` block invoking `if hasattr(pbar, "close") and callable(pbar.close): pbar.close()` to cleanly close the bar on normal exit, early bounded halts, and unhandled exceptions.
- **Running Loss Telemetry**: Added running annotation loss (`ann_loss`) and audit loss (`audit_loss`) to the postfix using existing running accumulation formulas (sample-weighted for Phase A, batch-mean for Phase B/C). Divisor arithmetic parity is validated in `test_unified_logging.py`.
- **Final Step Postfix Preservation**: Configured `pbar.set_postfix(postfix)` and `pbar.update(1)` immediately upon batch execution before checking `self.optimizer_step >= int(max_steps)`, guaranteeing bounded smoke halts capture terminal step telemetry.

### C. Explicit Persistent Console Epoch Summary (`src/self_audit/training/unified_trainer.py`)
- Implemented `format_epoch_summary(epoch, interval, train_stats, val_stats)` providing explicit, phase-aware metric formatting across canonical schedule phases:
  - **Phase A (`annotation_bootstrap`, Epochs 1–100)**: Emits `train_loss`, `val_loss`, `val_macro_foreground_dice`, and `initial_dice` (only if measured via headroom evaluation; omitted when unmeasured).
  - **Phase B (`auditor_training`, Epochs 101–120)**: Emits `train_loss`, `val_loss`, and the real verbatim source from `resolve_primary_metric` (`on_policy_auroc` or fallback `on_policy_improve_regress_accuracy`). Emits `primary_metric=N/A` if undefined or missing. Formats `on_policy_fix_f1`, `on_policy_regress_f1`, `combined_fix_f1`, and `combined_regress_f1`. Auditor accuracy is never mislabeled as Dice.
  - **Phase C (`joint_self_audit`, Epochs 121–130)**: Emits `train_loss`, `initial_foreground_macro_dice`, `final_foreground_macro_dice`, `net_gain`, `harmful_acceptance_rate`, and `beneficial_rejection_rate`.
  - **Non-fabrication Contract**: Non-finite floats (`math.isfinite` check) and malformed objects are formatted as `"N/A"` rather than dumping raw objects, raising, or fabricating `0.0000`.
- Emits summary with `flush=True` in `_train_impl`, ensuring summary visibility even when `--no_tqdm` disables bars.

---

## 3. Console Output Before & After (Canonical Phases)

> [!NOTE]
> The examples below are synthetic illustrations of console appearance, not measured training results.

### Phase A: Annotation Bootstrap (Canonical Epochs 1–100)

**Before:**
*(Progress bar was completely hidden due to `--no_tqdm`, or transiently deleted on completion due to `leave=False`)*
```text
[annotation_bootstrap] Epoch 001/130 train_loss=0.4567 primary_metric=0.7654 opt_step=12
```

**After:**
```text
Epoch 001/130 [annotation_bootstrap]: 100%|██████████| 10/10 [00:02<00:00, 4.80it/s, loss=0.4567, avg=0.4567, ann_loss=0.4567, opt_step=10]
[annotation_bootstrap] Epoch 001/130 train_loss=0.4567 val_loss=0.3891 val_macro_foreground_dice=0.7654 initial_dice=0.6800 opt_step=10
```
*(If headroom diagnostics are unmeasured, `initial_dice` is cleanly omitted without fabricating 0.0000)*.

---

### Phase B: Auditor Training (Canonical Epochs 101–120)

**Before:**
```text
[auditor_training] Epoch 101/130 train_loss=0.2345 primary_metric=0.8123 opt_step=45
```

**After:**
```text
Epoch 101/130 [auditor_training]: 100%|██████████| 10/10 [00:03<00:00, 3.20it/s, loss=0.2345, avg=0.2345, audit_loss=0.2345, opt_step=45]
[auditor_training] Epoch 101/130 train_loss=0.2345 val_loss=0.2876 on_policy_auroc=0.8123 on_policy_fix_f1=0.7410 on_policy_regress_f1=0.6920 combined_fix_f1=0.7250 combined_regress_f1=0.6800 opt_step=45
```
*(If AUROC is unmeasured and accuracy fallback was selected, emits `on_policy_improve_regress_accuracy=...`; if undefined, emits `primary_metric=N/A`)*.

---

### Phase C: Joint Fine-Tuning (Canonical Epochs 121–130)

**Before:**
```text
[joint_self_audit] Epoch 121/130 train_loss=0.1500 primary_metric=0.7400 opt_step=100
```

**After:**
```text
Epoch 121/130 [joint_self_audit]: 100%|██████████| 10/10 [00:04<00:00, 2.45it/s, loss=0.1500, avg=0.1500, ann_loss=0.1000, audit_loss=0.0500, opt_step=100]
[joint_self_audit] Epoch 121/130 train_loss=0.1500 initial_foreground_macro_dice=0.6500 final_foreground_macro_dice=0.7400 net_gain=0.0900 harmful_acceptance_rate=0.0350 beneficial_rejection_rate=0.0200 opt_step=100
```

---

## 4. Test Verification & Results

Environment:
- Python: `/private/tmp/self-audit-torch241/bin/python`
- Environment flags: `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=src:.`

### Focused Suites
1. **Console Progress Suite (`tests/test_unified_console_progress.py`)**:
   - `test_candidate_c_runner_defaults_no_tqdm_flag_removed`: Verified `--no_tqdm` absent from runner script.
   - `test_cli_user_opt_out_no_tqdm_remains_supported`: Verified CLI parser flag preservation.
   - `test_train_epoch_tqdm_configured_with_leave_true`: Verified `leave=True` on progress bar.
   - `test_train_epoch_progress_bar_closes_on_exception`: Verified `pbar.close()` called in `finally` block.
   - `test_train_epoch_postfix_running_losses` (3 intervals): Verified running loss telemetry keys and no Dice fabrication.
   - `test_train_epoch_max_steps_preserves_final_postfix`: Verified final step postfix and `n=1, total=4` preserved on halt.
   - `test_train_epoch_real_tqdm_progress_count_on_bounded_steps`: Verified real `tqdm` with `io.StringIO` displays `n=1, total=4`, and `"1/4"`.
   - `test_format_epoch_summary_phase_a_annotation_headroom`: Verified Phase A explicit metrics (Epoch 001/130).
   - `test_format_epoch_summary_phase_a_omits_unmeasured_initial_dice`: Verified unmeasured omission.
   - `test_format_epoch_summary_phase_b_auditor`: Verified Phase B canonical epoch (Epoch 101/130) and verbatim `on_policy_auroc`.
   - `test_format_epoch_summary_phase_b_fallback_accuracy`: Verified fallback `on_policy_improve_regress_accuracy`.
   - `test_format_epoch_summary_phase_b_undefined_source`: Verified `undefined` -> `primary_metric=N/A`.
   - `test_format_epoch_summary_phase_c_joint`: Verified Phase C canonical epoch (Epoch 121/130) metrics.
   - `test_format_epoch_summary_nan_metrics_marked_na_never_zero`: Verified NaN metrics formatted as `N/A`.
   - `test_format_epoch_summary_nonfinite_and_malformed_marked_na`: Verified infinite and malformed values formatted as `N/A`.
   - `test_no_tqdm_mode_disables_bars_but_preserves_printed_summary`: Verified production `_train_impl` print path under `disable_tqdm=True`.
   **Outcome:** 18 passed in 0.88s.

2. **Telemetry and W&B Logging Suite (`tests/test_unified_logging.py`)**:
   **Outcome:** 11 passed in 8.73s.

3. **Combined Focused Suite**:
   ```sh
   rtk env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=src:. /private/tmp/self-audit-torch241/bin/python -m pytest tests/test_unified_console_progress.py tests/test_unified_logging.py
   ```
   **Outcome:** 29 passed in 9.61s.

---

## 5. Scope of Changes

Files modified:
1. `scripts/run_acdc_mnms_candidate_c.sh`: Removed `--no_tqdm` default flags from ACDC and M&Ms invocations.
2. `src/self_audit/training/unified_trainer.py`: Added direct `train_loader` iteration with `update(1)`, verbatim primary metric source reporting, nonfinite/malformed `N/A` handling, and persistent epoch summary formatting.
3. `tests/test_unified_console_progress.py`: Comprehensive test suite verifying leave=True, safe close, real tqdm count, canonical phase formatting, and production `_train_impl` output.
4. `reports/candidate_c/console_progress_fix.md`: Durable outcome report.


## 6. Astra Final Review — 2026-09-12

**PASS after revision.** AGY implemented the four owned files through Orca; Astra inspected the actual diffs and independently verified the final tree against base `ca62f90` on latest main. Review corrected the Auditor metric source label, bounded tqdm counting, unavailable-value formatting, production summary test, and preserved loader completion semantics. The accepted final Orca dispatch was `ctx_2e0706c385c3`; release retained the external worker terminal with no process action.

Independent final validation (PyTorch 2.4.1, CPU):

```sh
rtk proxy env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:. OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /private/tmp/self-audit-torch241/bin/python -m pytest tests/test_unified_console_progress.py tests/test_unified_logging.py tests/test_unified_trainer.py -q
```

- **63 passed in 82.52s**, with one scheduler-order warning from the existing `test_canonical_learning_rate_keys` fixture.
- `py_compile` passed for the trainer, console tests and training entrypoint; critical imports passed.
- Independent real tqdm probe: bounded execution reported `1/4` for one processed batch/optimizer step; full execution reported `4/4` for four processed batches/steps.
- `bash -n` passed. Executing the actual runner in an isolated directory with a Python command shim captured both ACDC and M&Ms commands without `--no_tqdm`; it performed no model training.
- Trainer, test and runner SHA-256 digests were unchanged throughout final validation.

No model forward, loss, acceptance rule, dataset protocol, checkpoint schema, or W&B metric computation was added or changed. GPU training, real-data evaluation and the complete test suite were intentionally not run for this console fix. Already-running training processes do not reload these changes; checkpoint source-provenance checks remain unchanged.
