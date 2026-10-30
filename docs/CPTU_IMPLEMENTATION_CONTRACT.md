# Scalar transfer-utility implementation contract

Source base: `d93c4dea4f42895297ec5941bb938b47140aa9e4`. This is an
implementation contract, not patient-data validation or pilot success.

## Shared interfaces

`apply_compact_update(..., config=..., utility_probe=None)` keeps its existing
return keys and adds selector diagnostics. Config `selector` defaults to
`original`. New selectors are restricted to `scalar_guard`; existing joint and
plain behavior remains available without a new selector. `same_batch_utility`
is an optional, explicitly named control using the same utility algorithm.

`utility_probe` is a mapping with detached tensors `x`, `targets`, `valid`,
`temperature`, `coefficient`, `confidence`, and optional JSON-safe `metadata`.
It is sampled exactly once per transaction. Missing probe is a zero decision,
not a fallback to the proposal patient. The controller owns frozen copies,
functional evaluation, the raw-teacher confidence mask before temperature,
the common valid-pixel denominator, and reference/trial KD diagnostics.
Teacher target generation remains in the trainer/cache. No new teacher is
constructed in the controller.

Defaults: `selector_dice_tolerance=0.0` (exact Dice ties),
`utility_gain_tolerance=1e-7`, `utility_tie_tolerance=1e-7`,
`probe_seed=42017`, `probe_batch_size=8`, `max_trials=8`. Guard epsilon/FP
budget remain zero; existing guard numeric tolerance remains `1e-6`.
These numerical tolerances are distinct and must not be silently relaxed.

Original selection retains strict objective comparison and its historical
handling of zero: do not add a zero competitor to original/larger_tie.
Larger-tie first resolves original, then selects the largest feasible scalar
with foreground Dice equal to that original choice within Dice tolerance.
Utility includes zero with gain zero, requires gain above its gain tolerance,
and breaks tied gains by smaller displacement norm. Zero reuses the reference.
The fixed eight-scale grid is unchanged. Log audit stream forward batches,
nonzero trial evaluations, reference/probe forward counts and timing separately;
eight weight trials can require many streamed audit minibatch forwards.

Probe sampling uses a separate NumPy Generator, excludes every proposal
patient in the current transaction, reads image-only TRAIN-U records, and
uses the same spatial/intensity augmentation and cached target alignment as
proposal sampling. The existing L/U/audit RNG consumption remains unchanged.
Same-batch control explicitly reuses the already sampled proposal tensors.

Exact resume saves model, optimizer, all relevant global RNG states,
proposal/data and probe Generator states, completed optimizer-step counts,
coverage/counters, and immutable protocol identities. Restore RNG after model
construction/loading. Legacy checkpoints continue loading/exporting; exact
resume without required RNG metadata must fail clearly. A saved total budget
cannot be silently extended. Bounded stop/resume may use an explicit stop step
within that declared total, with the identical next decision on continuation.

## Ownership during implementation

- Controller worker: `self_audit/compact_control.py`, optional new
  `self_audit/compact_utility.py`, `tests/test_compact_selectors.py`.
- Trainer worker: `self_audit/compact_training.py`, `compact_artifacts.py`,
  `compact_probe.py`, `train.py`, CLI plumbing in `engine.py`, and new tests
  for probe/resume/CLI. Coordinate any data-loader edits with the coordinator.
- Runner worker: new `scripts/*transfer_utility*`,
  `configs/transfer_utility/`, `docs/TRANSFER_UTILITY_RUNBOOK.md`, and new
  runner/diagnostics tests. No training launch or edits to core trainer.
- Evidence worker: `reports/transfer_utility/preflight/` and an independent
  archived-trial replay/oracle script; all other source remains read-only.
- Coordinator: integration, existing source-of-truth docs, protocol lock,
  independent verification, publication and authorized server execution.

All workers share this worktree; preserve other edits, do not commit or push,
and report exact commands/results plus remaining limitations. No worker starts
patient training. The coordinator locks pilot budget and evaluation rule before
launch, after bounded profiling and without inspecting new pilot VAL outcomes.
