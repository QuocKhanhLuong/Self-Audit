# Legacy resume/checkpoint fixture review — tests/test_unified_trainer.py

Scope: audit the hand-built resume/checkpoint fixtures in `tests/test_unified_trainer.py`
against the strict runtime contract specified in `wave4_final_revision_task.md`, and adapt
only stale fixture construction. No production source was touched. No test was skipped,
xfailed, or relaxed; no validator was monkeypatched away.

## 1. Reproduction before any change

```
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:. \
  /private/tmp/self-audit-torch241/bin/python -m pytest tests/test_unified_trainer.py \
  -k "resume_lineage or strict_resume_failures or output_relocation or uninterrupted_vs_resumed" \
  -q -p no:randomly
```

Result: **4 failed, 30 deselected in 8.16s.**

| Test | Observed failure |
| --- | --- |
| `test_resume_lineage_before_and_after_boundary` | `required flag 'resumable' is absent` |
| `test_strict_resume_failures` | regex `Cannot resume from non-resumable or incomplete checkpoint` did not match actual `'resumable' is False, expected True` |
| `test_output_relocation_safety_and_clobber_prevention` | `required flag 'validation_complete' is absent` |
| `test_uninterrupted_vs_resumed_weights_and_optimizer_ordinary_and_reset_boundaries` | `required flag 'validation_complete' is absent` |

## 2. Classification: stale fixtures, not production defects

All four failures are stale fixture construction. Each fixture was written against the
pre-hardening contract, in which an absent resumability flag was treated as permission
(`if resumable is False or incomplete_epoch is True` at base commit `25f429a`,
`unified_trainer.py:1274-1279`). The current contract requires each of `resumable`,
`incomplete_epoch` and `validation_complete` to be *affirmatively* present and correct,
and additionally requires non-empty source identity, non-empty `run_id`, a provenance
`state_digest` matching the stored tensors, an epoch inside the configured schedule,
`optimizer_step <= global_step`, and an `epoch_history` describing exactly epochs
`1..N` whose tail row agrees with the payload counters and the schedule
(`unified_trainer.py:495-570`, `1736-1836`).

The old fixtures supplied none of that metadata, so they could not certify an exact
resume point. Production is correct here; the fixtures were the stale artefact.

**No concrete production defect was found in this review.** The two bugs named in
`wave4_final_revision_task.md` (stale `last.pt` tail row; `last_completed_validation`
read from `payload['extra']`) are already repaired in the central source and are now
additionally covered from this file at the unit level — see §4.

One non-defect note for root: the refusal *message* for a non-resumable checkpoint
changed from the single `Cannot resume from non-resumable or incomplete checkpoint`
string to per-flag messages. The corresponding negative assertion was retargeted to the
current message rather than relaxed, and the case was split so both "flag says no" and
"flag is absent" are asserted separately.

## 3. The unit-fixture metadata builder

A single labelled helper, `_unit_fixture_commit_extra()`, was added near the top of the
file. Its docstring states plainly that these checkpoints are hand-built unit fixtures
whose counters are synthesized, and that real end-to-end continuation coverage lives in
`tests/test_runtime_resume.py`.

The metadata it produces is internally coherent rather than merely sufficient:

- **Identity** — `run_id`, `source_signature`, `config_signature`/`recipe_signature` are
  read off the live trainer, so they agree with what the resuming trainer computes. Model
  digest is not synthesized at all: `save_checkpoint` stamps `provenance.state_digest`
  from the tensors it actually writes, so the digest gate is satisfied by real bytes.
- **Certification flags** — `resumable=True`, `incomplete_epoch=False`,
  `validation_complete=True`, plus `completed_epoch`.
- **Schedule** — `active_interval`, `interval_name` and `interval_index` are derived from
  the live `Schedule` at the recorded epoch, so a fixture cannot claim an interval the
  configured schedule does not put that epoch in.
- **`epoch_history`** — one row per epoch `1..N`, each carrying its schedule-derived
  interval and index, and monotone `global_step`/`optimizer_step` interpolated so the
  final row lands exactly on the payload counters (and so `optimizer_step <= global_step`
  holds row by row). The tail row is marked `checkpoint_committed=True` with
  `completed_epochs` and `last_checkpoint_committed_epoch` equal to the recorded epoch,
  and `report_committed=False`, preserving the truthful distinction between the
  checkpoint commit and the later JSON report commit.
- **Observed validation** — `last_completed_validation` carries the actual selection
  metric key `final_foreground_macro_dice`, so a selection claim can be cross-checked
  against an observation rather than a duplicated counter.

`**overrides` are applied last, so a negative test can corrupt exactly one field and
still reach the gate it is aiming at.

## 4. Per-test changes

### `test_resume_lineage_before_and_after_boundary`
Fixture now carries full commit metadata for epoch 99. Assertions were **added**, not
relaxed: the restored history must be exactly epochs 1..99, its tail must report
`checkpoint_committed=True`, `completed_epochs=99`, `last_checkpoint_committed_epoch=99`
and the saved counters, the trainer's own `completed_epochs` /
`last_checkpoint_committed_epoch` must agree, and `last_completed_validation` must be
restored from the payload top level — direct unit coverage of the
`payload['extra']` regression named in the wave-4 task.

### `test_strict_resume_failures`
Every pre-existing refusal is preserved. Case 1 keeps the non-resumable smoke refusal
(retargeted to the current per-flag message) with every *other* field a complete unit
commit, so the refusal can only originate in the flags. Added:

- **1b** — omitting each of `resumable`, `incomplete_epoch`, `validation_complete` in turn
  must be refused with `required flag '<name>' is absent` (unknown is not permission).
- **4** — an epoch of `total_epochs + 1` must be refused as outside the configured
  schedule, and `optimizer_step=5` with `global_step=3` must be refused.
- **5** — a history truncated to fewer rows than completed epochs must be refused, and a
  tail row with `checkpoint_committed=False` must be refused.

Cases 2 (loader cardinality mismatch) and 3 (config `num_classes` mismatch) keep their
original expected refusals; they only gained the certification metadata needed to reach
those gates at all, since the flag gate now runs first.

### `test_output_relocation_safety_and_clobber_prevention`
Both hand-built `last.pt` fixtures now carry full commit metadata alongside the existing
legacy `best_checkpoint_hash` lineage fields. Case A (safe relocation retains best
lineage), Case B (`FileExistsError` on a dirty target) and Case C (`Stale or replaced best
checkpoint detected`) are unchanged in intent and still assert the same refusals.

### `test_uninterrupted_vs_resumed_weights_and_optimizer_ordinary_and_reset_boundaries`
The step-1 and epoch-100 boundary checkpoints now carry full commit metadata. The exact
parity assertions (identical model digest and optimizer step between the uninterrupted
and resumed runs; fresh auditor-only optimizer at the 100 boundary) are unchanged.

### Not changed
`test_cohort_descriptor_detects_same_count_changed_patient_fixtures` calls
`compare_cohort_descriptors` directly and never constructs a checkpoint, so the stronger
resume contract does not reach it. Its eight negative cases are untouched. No other
hand-built resume metadata remains in the file.

## 5. Focused verification after the change

Same interpreter and environment as the reproduction:

```
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:. \
  /private/tmp/self-audit-torch241/bin/python -m pytest tests/test_unified_trainer.py \
  -k "resume_lineage or strict_resume_failures or output_relocation or uninterrupted_vs_resumed" \
  -q -p no:randomly
```

Result: **4 passed, 30 deselected in 16.45s.**

`git diff --check` on `tests/test_unified_trainer.py` is clean. Diff stat: 295 insertions,
29 deletions, one file.

## 6. Coverage explicitly deferred

Whole-file execution of `tests/test_unified_trainer.py` was **not** run here. The
coordinator directed that the remaining 30 deselected tests are covered by root's single
complete integrated suite, so whole-file coverage for this file is deferred to that run.
