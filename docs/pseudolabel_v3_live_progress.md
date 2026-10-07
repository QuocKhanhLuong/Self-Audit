# Live v3 training progress

The full runner, standalone teacher and standalone student show a tqdm bar for
**every processed training batch**. Bars include elapsed time, batch rate and
remaining time within the current epoch. Empty student batches advance progress
but never advance the optimizer. Native teacher export and student inference also
show per-patient prediction/write progress.

## Flags

- `--log-every 50` (default): flush a structured metric record on the first batch,
  every 50th batch, and the last batch of each epoch. The first/last record is not
  duplicated if it is also on the interval. Use `--log-every 10` for more frequent
  detail, or `--log-every 1` for every batch
- `--no-progress`: disable tqdm bars, while keeping periodic records and the
  existing epoch summaries
- `--log-every 0`: disable sampled JSONL/console/W&B training records. Combine
  with `--no-progress` for the previous epoch-only terminal cadence
- Existing `--wandb --wandb-mode online` or `offline` flags still apply only to
  the full runner. It owns **one** W&B run and receives sampled teacher/student
  records while their subprocess is training. Final histories are not replayed
  into W&B. Evaluation metrics still arrive only after independent evaluation

For the next run, keep the same data, split, scientific configuration and
training options, and append e.g.:

```bash
--log-every 50
```

No dependency lock changes are needed: tqdm is already in the canonical lock.
On an existing noncanonical environment missing it, install tqdm in that same
isolated environment. Do not replace a working Blackwell/CUDA 12.8 environment
with the canonical CUDA 12.1 lock just to enable logging.

## Files and metric meanings

For full runner output `RUN`:

| File | When written |
| --- | --- |
| `RUN/pipeline.log` | Live forwarded subprocess output; carriage-return bar refreshes become separate lines in this file |
| `RUN/teacher/train_metrics.jsonl` | Sampled teacher events, flushed immediately |
| `RUN/student_<profile>.metrics.jsonl` | Sampled student events, flushed immediately |
| `RUN/teacher/train_metrics.json` | Existing complete batch history, after teacher training |
| `RUN/student_<profile>.json` | Existing complete successful-update history and coverage, after student training |

Each JSONL row has schema `self_audit.progress.v1`, a `stage`, and a `metrics`
object. The same row is emitted as `[V3_METRIC] { ... }` on the console. Flushing
retains completed records if a later stage fails or the process is interrupted;
this is telemetry, **not a resumable checkpoint** or power-loss durability promise.
Existing final scientific artifacts and source/freeze checks remain authoritative.

Common fields:

- `epoch` and `step` retain the existing zero-based indices; `batch` and
  `global_batch` are one-based processed-batch counts
- `batches_per_epoch` honors `--max-train-batches` or
  `--student-max-train-batches`; it is not the unbounded loader size
- `learning_rate`, `samples_per_second`, `batches_per_second`, elapsed seconds,
  `epoch_eta_seconds` and `training_eta_seconds`
- ETA uses observed wall time, including loading and logging. Training ETA
  estimates the remaining **training batches of this teacher or student** only;
  it does not include export, freezing, validation or later pipeline stages.
  Early estimates are noisy and assume similar future batch cost

Teacher fields:

- All existing loss components: `total`, `reconstruction`, `prototype`,
  `semantic_seed`, `motion_photo`, `motion_smooth`
- `accepted_regions`, `candidate_regions` and `accepted_{bg,rv,myo,lv}_regions`
  count image-only raw-evidence regions accepted for the region-level loss.
  They are **counts, not accuracy or pixel coverage**
- `decoded_valid_pixels/fraction`, `decoded_foreground_pixels/fraction`, and
  `decoded_{bg,rv,myo,lv}_pixels/fraction` describe a diagnostic decode from the
  **same pre-update teacher encoding/evidence**, before temporal export rejection.
  There is no second neural forward and no GT access. These are not the region
  loss targets or the final frozen export coverage. At logging intervals only,
  the diagnostic decode and detached reductions add a small amount of work;
  no GPU overhead percentage has been measured

Student fields:

- `loss` for batches with a real optimizer update; `skipped_empty` explicitly
  identifies empty batches, which have no fabricated loss
- `optimizer_steps` counts actual updates separately from processed batches
- `target_valid_pixels/fraction`, `target_foreground_pixels/fraction` and
  `target_{bg,rv,myo,lv}_pixels/fraction` count the frozen training pseudo-label
  support in this batch

All coverage fractions divide by **all spatial pixels in the batch**, including
UNKNOWN pixels in the denominator. Foreground is RV + MYO + LV. UNKNOWN is
excluded from accepted class counts. These support metrics do not measure
Dice, correctness or calibration. GT remains confined to independent evaluation.
Gradient norms are not collected.

Export bars count two slice passes per patient (prediction and consistency/write).
The final NIfTI save remains part of export; its compression and final freeze
hashing have no advance ETA. A 100% slice bar is not a completed pipeline.

## Updating safely

An already running Python process cannot acquire these changes. Do not edit or
pull over its checkout: v3 hashes source and can reject mid-run source changes.
Keep that checkout and its environment for its checkpoint inference/export.
Use a separate clean checkout for the updated code and a **new `--out` directory**.
Never bypass a source/environment identity mismatch on an old checkpoint.

The review patch is based on `d4d30519233c1da691a358a57d57de3f7566a35f`.
After obtaining the patch, a separate checkout can be prepared as follows:

```bash
# Run from the existing Git repository; this does not modify its working files.
git worktree add -b feat/v3-live-progress ../Self-Audit-live-progress \
  d4d30519233c1da691a358a57d57de3f7566a35f
cd ../Self-Audit-live-progress
git apply --check /absolute/path/self-audit-v3-live-progress.patch
git apply /absolute/path/self-audit-v3-live-progress.patch
export PYTHONPATH="$PWD/src"
# Activate the SAME compatible environment intended for the next run.
python scripts/run_full_pipeline_v3.py --help
```

This does not push or merge anything. If the local base revision differs, inspect
and reconcile the patch before applying it. Record Git revision, patch hash,
exact command and environment versions with research results.

## Tests

```bash
PYTHONPATH=src:. python -m pytest -q \
  tests/test_pseudolabel_v3_progress.py \
  tests/test_pseudolabel_full_pipeline.py tests/test_pseudolabel_wandb.py \
  tests/test_pseudolabel_v3_checkup.py tests/test_pseudolabel_v3_regressions.py
```

The new tests cover immediate JSONL flush and interruption retention; bounded,
skipped and cross-epoch counts/ETA; exact teacher/student numerical equivalence
with logging on/off; no reference reads during training; unchanged pseudo export
arrays; single teacher forward; live subprocess delivery before child exit;
carriage-return forwarding; W&B history de-duplication; and CLI propagation.
