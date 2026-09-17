# Wave 4: best/last commit protocol and exact-resume integration

Central worker: Claude. Environment for every count below:
`/private/tmp/self-audit-torch241/bin/python` (Python 3.10.21, torch 2.4.1, CPU),
`PYTHONPATH=src:.`, `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1`, `-p no:randomly`.
No commits, no push, no long training. All shell commands `rtk` / `rtk proxy` prefixed.

Owned in this dispatch: `src/self_audit/training/unified_trainer.py`, the single
`_utils.py` alias-guard correction, `tests/test_runtime_checkpoint_commit_integration.py` (new),
small updates to `tests/test_trainer_lifecycle.py`, `tests/test_canonical_best_alias.py` and the
stale GPU RNG assertion in `tests/test_runtime_checkpoint.py`, and this report.
`tests/test_runtime_resume.py` and `parallel_resume.md` belong to the resume worker and were not
touched; `tests/test_runtime_pipeline_smoke.py` is frozen; `tests/test_unified_trainer.py` is
another worker's (its hand-built resume fixtures are being adapted to the stronger metadata
contract there, not here); canonical YAMLs and the rest of `_utils.py` belong to the portability
worker. Root owns the final full suite and the final reports.

## Focused test counts

Focused invocation for the corrected paths:

```
PYTHONPATH=src:. PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  /private/tmp/self-audit-torch241/bin/python -m pytest \
  tests/test_runtime_checkpoint_commit_integration.py tests/test_canonical_best_alias.py \
  -q -p no:randomly
```

**39 passed in 151.66s.**

| File | Result |
|---|---|
| `tests/test_runtime_checkpoint_commit_integration.py` (26 tests after the correction) | part of the **39 passed in 151.66s** above |
| `tests/test_trainer_lifecycle.py` (fixture update only) | **24 passed in 44.17s** |
| `tests/test_canonical_best_alias.py` (13 tests) | part of the same **39 passed** run |
| `tests/test_runtime_checkpoint.py::test_cuda_rng_state_handling` | **1 passed** |

One final integrated suite is root's.

## 0. Alias guard now fails closed

`_require_committed_best_alias` previously printed a warning and continued when an existing
sibling `last.pt` could not be safe-loaded. That inferred "historical, no committed selection"
from unreadable bytes — but a corrupted `last.pt` could equally commit a selection the alias
contradicts, which is exactly the stale alias the guard exists to catch. It now raises, naming
the repair options. A non-mapping payload raises too. Only the *absence* of a sibling, or a
readable sibling with no `best_reference` key, is historical compatibility.

Judgement, revised from the previous report: the compatibility cost is real — a legacy `last.pt`
holding `Path` metadata fails `weights_only=True` (audit R2) and now blocks evaluation of a
`best.pt` beside it. That is the correct trade: the operator can evaluate a standalone copy in a
clean directory, and silently skipping verification is not a safe default.
Test: `test_unreadable_sibling_last_fails_closed`.

## 0b. Final bounded correction (Astra REVISE)

Two bugs root reproduced through the actual `_trainer` fixture, plus the remaining strict
provenance acceptance, are fixed here.

### Bug 1 — the saved history tail was a pre-commit row

`epoch_history` was built from `self.report["epochs"]` *before* the commit fields were advanced,
so the row inside `last.pt` recorded `checkpoint_committed=False` and the previous
`completed_epochs` while the durable report said `True` and the current count. On resume that
stale tail persisted into the final report.

The complete committed row is now **proposed before** the `last.pt` save — with
`checkpoint_committed=True`, `status="checkpoint_committed"`, `completed_epochs` and
`last_checkpoint_committed_epoch` at the proposed epoch, and every selection field
(`best_selection_status`, `selection_metric`, `selection_metric_value`, `best_metric`,
`best_epoch`, `best_checkpoint_hash`) — and the history written into the checkpoint is the earlier
rows plus that proposed row. It is adopted into the in-memory row (`row.update(committed_row)`)
only once the save has succeeded. Nothing is patched after the fact to claim a state the bytes do
not contain.

The distinction between the two commits stays truthful: the proposed row carries
`report_committed=False`, and that flag flips to `True` only after the atomic
`pipeline_report.json` write succeeds. So a restored row for an epoch whose report *did* commit in
an earlier attempt still reads `report_committed=False` — the checkpoint cannot certify the later
JSON commit, and the row is not rewritten to pretend otherwise.

The history is also normalized with exactly the report's semantics (`_json_safe`) before it is
written into the checkpoint, so the checkpoint and `pipeline_report.json` describe the same rows —
including "an undefined metric is an explicit null, never a number and never zero". Without that
the checkpoint kept `NaN` where the report wrote `null`, and a restored row differed from the
baseline it is supposed to reproduce (found by the field-by-field comparison below, which is
exactly what an epoch-id-list assertion would have missed).

Test: `test_saved_history_tail_is_the_committed_row_and_survives_resume` — crashes entering epoch
3 of 3, asserts the saved tail's committed fields against the payload counters and interval, then
resumes and compares each restored row **field by field** against the first attempt's durable
report rows (with `report_committed` asserted separately), not merely the epoch id list.

### Bug 2 — `last_completed_validation` was read from the wrong place

`save_checkpoint` merges `extra` into the payload, so the saved validation sits at the payload top
level. Resume read `payload["extra"]`, found nothing, and left `last_completed_validation` as
`None` — losing exactly the evidence a failure report written right after resume has to show. It
now reads the top-level value and deep-copies a mapping.

Test: `test_resume_restores_the_saved_completed_validation` — crash, resume, assert the restored
validation matches the saved Dice, then force a failure immediately after resume and assert
`failure.json` carries it.

### Strict provenance acceptance, before any output mutation

Every gate below runs before the selection block, so before any alias repair, relocation or other
output-directory mutation:

- `resumable`, `incomplete_epoch` and `validation_complete` must each be **affirmatively** right
  (`True`/`False`/`True`). An absent flag is unknown, and unknown is not permission.
- Saved **and** current source content signature must each be a non-empty, non-`UNKNOWN` string —
  fail closed when either is unavailable, rather than bypassing the comparison as before.
- `run_id` must be a non-empty string.
- `provenance.state_digest` must be present and equal `state_digest` of the payload's stored
  tensors.
- `epoch` must lie within `[1, schedule.total_epochs]`, and `optimizer_step <= global_step`.
- The restored history's tail row must agree with the payload on `global_step`,
  `optimizer_step`, `completed_epochs` and `last_checkpoint_committed_epoch`, must be marked
  `checkpoint_committed`, and must name the interval and index the schedule assigns to that epoch.
- A committed best reference may not name an epoch later than `completed_epoch`.
- The selected-best snapshot must record a `selection_metric`, and its own
  `last_completed_validation[final_foreground_macro_dice]` — the actual observed measurement, not
  a duplicated counter — must match the selected `best_metric`. An undefined observation is
  refused rather than read as a value; undefined stays undefined.

No negative provenance test was weakened. Fixtures were regenerated only where the stronger
contract requires real generated checkpoint metadata.

The fail-closed unreadable-sibling trade-off in §0 is approved; standalone `best.pt` evaluation in
a directory with no sibling `last.pt` remains supported and tested.

## 1. The commit protocol

Per-file atomicity is not a transaction. Writing the public `best.pt` first meant a subsequent
`last.pt` failure left the old `last.pt` naming bytes that had already been overwritten (audit
**R9**). The commit block now runs, per epoch:

1. If selection is active and the metric improves, the candidate is saved with the ordinary
   `save_checkpoint` to a unique, never-reused
   `output_dir/selected_best/epoch_<n>_<uuid4hex12>.pt` (`extra.role="selected_best"`).
2. `make_best_reference` produces the reference (relative path + sha256 + epoch + metric).
3. `last.pt` is committed with `best_reference` plus selection metadata **derived from the
   reference**, so `best_metric`/`best_epoch`/`best_checkpoint_hash` cannot drift from the
   snapshot they describe.
4. `last_checkpoint_committed_epoch` and the live `self.best_*` advance, then
   `publish_best_alias` republishes `best.pt` from the hash-verified snapshot
   (`failure_stage="best_alias"`).

Failure accounting: a crash before or during step 3 leaves the previous `last.pt` referencing its
own untouched snapshot; a crash during or after step 4 leaves `last.pt` authoritative and the
alias stale, which resume repairs from the reference. Nothing depends on in-process rollback,
which cannot run under `SIGKILL`. Nothing is ever deleted, so an older `last.pt`'s reference stays
resolvable.

Tests: `test_commit_writes_snapshot_then_last_then_alias`,
`test_alias_failure_leaves_last_authoritative_and_resume_repairs_it`,
`test_failed_last_commit_preserves_the_previously_referenced_snapshot`.

### Selection metric

Selection happens **only** in the threshold-gated interval (`rollout == "threshold_gate"`), only
at or after `checkpoint.best_selection_min_epoch`, only under full validation, and only on
`final_foreground_macro_dice`. An absent Dice raises rather than falling back to
`primary_metric`, which is a different quantity in the other intervals. An undefined Dice —
`None` or `NaN` — cannot improve the best: status becomes `skipped_nonfinite_metric`, no snapshot
is written, `best_metric` stays at the `-inf` sentinel, and `selection_metric_value` is an
explicit null, never `0`.

Only `-inf` (or an absent value) is a valid "nothing selected" sentinel — at commit time and on
both resume branches. A finite value, `+inf` or `NaN` with no verifiable selected state is
refused, because it would suppress or corrupt every future comparison.

Tests: `test_gated_selection_refuses_a_missing_dice`,
`test_undefined_dice_cannot_improve_the_best[None, nan]`,
`test_explicit_absent_selection_with_finite_metric_refuses_resume`,
`test_explicit_absent_selection_with_sentinel_resumes_and_can_still_select`.

**Fixture consequence, not a production relaxation.** Bootstrap-only synthetic schedules relied on
the old non-gated fallback. Rather than weaken the guard, the two fixtures in
`tests/test_trainer_lifecycle.py` and the one in the new integration file were changed to the real
gated interval shape — `objective: retained_final_annotation`, `trainable: all`,
`transition_population: active_attempted`, `rollout: threshold_gate`,
`annotation_weight: 1.0`, `audit_weight: 1.0`, `reset_optimizer: true`, all three LRs positive.
That is the only combination `schedule.py` accepts for a gate and the only validator that emits
the Dice. All 24 lifecycle tests pass on it, unchanged otherwise.

### `last.pt` keys

`best_reference` (canonical, exported as `COMMITTED_BEST_REFERENCE_KEY`), `best_metric`,
`best_epoch`, `best_checkpoint_hash`, `best_checkpoint_path`, `selection_metric`,
`selection_metric_value`, `best_selection_status` ∈ {`selected`, `not_improved`,
`skipped_nonfinite_metric`, `not_gated`}, `resolved_schedule`, `epoch_history`, `role`,
`wandb_identity`, plus everything that was already there.

## 2. Resume

- **Both unrestricted `weights_only=False` fallbacks are gone.** `resume_from_checkpoint` and the
  sibling read now go through the shared `load_checkpoint(map_location="cpu", restore_rng=False)`,
  so safe deserialization and finite model/optimizer validation apply and cannot be bypassed.
- **CPU staging.** The payload is staged on CPU, so a resumed GPU run does not materialize a
  second copy of the optimizer/RNG tensors on the device;
  `Optimizer.load_state_dict` then casts each state tensor onto its parameter's device through
  the supported API. The same staging already applies at both evaluation bind entrypoints. No GPU
  memory figure is claimed — none was measured locally.
- **RNG.** Restored last, with `exact_cuda=(self.device.type == "cuda")`, so a GPU resume requires
  the CUDA generator state and count instead of silently leaving the device stream un-restored.
- **Scaler / AMP.** An empty `{}` saved state against an enabled live scaler raises an actionable
  AMP-mode message instead of torch's opaque `source state dict is empty`; a populated saved state
  against a disabled live scaler also raises rather than silently discarding the saved scale.
  Canonical bfloat16 keeps scaling disabled on both sides, so canonical resume is unaffected.
- **Strict counters.** `_strict_counter` replaces `int(payload.get(key, 0))` for `epoch`,
  `global_step` and `optimizer_step`: a missing key, a bool, a float, a numeric string or a
  negative value each raise rather than being coerced or defaulted to `0`.
  Test: `test_resume_rejects_invalid_counters[epoch, global_step, optimizer_step]`.
- **Selection recovery**, three branches:
  - `best_reference` present — `resolve_best_reference` verifies the snapshot by hash;
    `best_metric`, `best_epoch` and `best_checkpoint_hash` are all **required** and cross-checked
    against it; the snapshot must carry `run_id` and `config` (both now required, not
    "if present"), its own epoch via the strict counter, a measured `best_metric` equal to the
    reference metric, a `selection_metric` equal to the configured Dice when recorded, and a
    `provenance.state_digest` equal to `state_digest` of its stored tensors. Relocation goes
    through `relocate_best_reference`, and `best_alias_matches_reference` decides whether to
    republish the alias.
  - `best_reference` explicitly `None` — refuses anything but the `-inf` sentinel, then resets.
  - Neither key — the legacy sibling-hash lineage path is kept verbatim, so historical negative
    provenance gates still fire.

Tests: `test_tampered_selection_metadata_refuses_resume[best_metric, best_epoch,
best_checkpoint_hash]`, `test_corrupted_snapshot_refuses_resume`,
`test_missing_snapshot_refuses_resume`, plus the alias-repair test above.

### Verified progress history

`last.pt` carries `epoch_history` — the committed per-epoch rows — and resume restores
`self.report["epochs"]` from it before anything can truncate it. It is validated, not trusted: a
missing history at `epoch > 0`, a wrong length, a non-mapping row, or a row whose recorded epoch
is not its 1-based position is a refusal. A resumed final report therefore contains epochs
`[1..N]`, not only the new attempt's rows.

Tests: `test_resumed_report_preserves_the_committed_epoch_history` (crash at epoch 3 of 3, resume,
report and durable `pipeline_report.json` both `[1,2,3]`),
`test_resume_refuses_a_truncated_or_missing_history`.

### Worker lifecycle

The resolved configuration is honoured verbatim. An earlier revision forced
`persistent_workers=False` inside `get_loaders`; that was reverted, because silently overriding an
explicit user config contradicts the provenance the checkpoint records and would disarm the
negative resume guard. The canonical YAMLs already resolve to non-persistent workers, and the
guard that refuses an augmenting persistent-worker checkpoint is untouched, so such a
configuration is rejected at resume rather than quietly reinterpreted. The YAML values themselves
are the portability worker's to change.

## 3. Report ordering (W3 carryover)

The atomic `pipeline_report.json` now publishes the **proposed** `completed_epochs` (`epoch + 1`)
together with `checkpoint_committed=True` and `status="checkpoint_committed"`, and the live
`self.completed_epochs` advances only after that write succeeds, with the proposed row/report
values restored on failure. A post-commit reader — the W&B adapter, or a crash investigator — no
longer sees a durable report carrying the previous count.

The required final report is still written before a successful `finish`, and the post-finish
telemetry refresh remains explicitly best-effort.

Every committed row and every checkpoint also carry `resolved_schedule`: interval name and index,
`rollout_mode`, `trainable`, `objective`, `transition_population`, the annotation/audit weights,
the three configured LRs, batch size, accumulation steps and `augment`.

`test_post_commit_log_sees_the_advanced_count_on_disk` uses a mocked adapter that reads
`pipeline_report.json` and `last.pt` **off disk** inside `log()` and cross-checks, per epoch:
`completed_epochs`, checkpoint epoch, interval name and index, schedule weights, `rollout_mode`,
all three LRs against the real `train/lr_*` keys, the primary metric, the `val/initial_dice` and
`val/final_dice` family against the measured row keys (an undefined metric must stay absent, never
fabricated as `0`), the committed selection metric and value, and that the checkpoint history is
`[1..epoch]`.

## 4. W&B identity

`__init__` no longer starts a run. It builds a *disabled* `WandbLogger` placeholder — a complete
no-op with the full attribute surface, so every existing call site keeps working — and stores the
settings. `start_logging()` runs at the top of `_train_impl`, i.e. after any resume, and
constructs the real logger with `run_id=self.run_id` and `resume="allow"` when `self.is_resumed`.
So a resumed attempt asks to continue the recovered identity instead of opening an orphan run.

`identity_summary` is stored on `self.wandb_identity` and appears in `telemetry_summary`, in
`report["wandb_identity"]` and in every checkpoint's `extra`. `_refresh_wandb_identity` re-reads
the live summary for the final/failure report, so a cached `finish_status="not_finished"` can no
longer sit next to `finalization_status="finalized"`; `telemetry_summary` prefers the live summary
for the same reason.

`finalization_status` comes from the logger's actual `finish_status` —
`finalized` / `finish_failed` / `not_finalized` / `no_logger` — never an assumed value. An adapter
that raises out of `log`/`set_summary`/`finish` instead of recording its own error is recorded
through `_record_logger_adapter_error`, so a lost telemetry error cannot read as clean.

Offline semantics are the wrapper's: `resume` is forwarded only in online mode, and
`backend_resume_performed` is `True` only on an explicit SDK `run.resumed`, never from a matching
id. An offline resumed attempt is a new local run reusing an id, and `resume_limitation` says so.

Every telemetry call — initialization, per-epoch log, summary, finish — is wrapped in
`_preserving_rng()`, which snapshots and restores the Python/NumPy/torch streams. Checkpoint RNG is
captured at save time, before post-commit telemetry, so a mocked adapter that draws from the
global streams cannot change training or resumed continuation. Data augmentation RNG semantics are
untouched.

Tests: `test_logger_initialization_is_delayed_until_after_resume`,
`test_resumed_attempt_requests_the_recovered_run_identity`,
`test_finalization_status_reports_a_failed_finish`,
`test_telemetry_cannot_consume_the_training_random_streams` (a drawing adapter versus a quiet one:
identical model tensors and identical checkpointed Python/NumPy/torch RNG).

## 5. Calibration binding

`run_post_training_calibration` and `run_independent_diagnostics` bind
`output_dir/best.pt`, which is now the published view of the committed selection, and the
`_utils` guard verifies that binding against the sibling `last.pt`'s `best_reference` before the
model is loaded. So calibration measures the actual committed selection and refuses a stale alias
rather than silently switching to different weights; the repair is a resume, not an evaluation-time
rewrite.

## 6. Stale GPU-only assertion

`tests/test_runtime_checkpoint.py::test_cuda_rng_state_handling` asserted
`match="does not match available devices"`. The implementation raises
`Checkpoint contains N CUDA RNG state(s), but M device(s) are available`, so the regex could never
match — and, living inside the CUDA-only arm, it could not fail on a CPU-only host. It now asserts
the real message with both counts. The same branch is covered under mocked CUDA in
`tests/test_runtime_amp_resume_review.py`, so it runs everywhere.

## Limits

- No GPU was available: CUDA 12.1 / RTX 4070 / bf16 execution, OOM behaviour and GPU peak memory
  are not measured here. The CUDA paths are covered by mocks and by `exact_cuda` requirements.
- Exact-resume continuity across all schedule positions and `workers=2` is the resume worker's
  file, deliberately not duplicated here.
- No architecture, loss, pretraining, gate, split or recipe change was made.
