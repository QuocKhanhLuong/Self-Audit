# Compact terminal follow-up — 2026-09-15

The default mask-free console now uses tqdm: one bar per training epoch and
one summary line for elapsed time, the three losses and selection NLL. Long
preparation/checkpoint phases show elapsed time; phases under 0.5 seconds stay
hidden. Batch completion updates the counter on every batch even when the
diagnostic JSONL snapshot is throttled. Resumed bars start at the saved batch
cursor. Errors and final report paths remain visible.

Detailed file/header/hash/heartbeat/metric events continue to the existing
JSONL journals. They are no longer printed by default. The final CLI result
does not dump the complete report JSON. `MASKFREE_PROGRESS=verbose` or
`--progress verbose` restores diagnostic console output. `tqdm>=4.66` was added
to `requirements-maskfree.txt` (the checked local environment has 4.70.0).

## Validation metric boundary

**Superseded later on 2026-09-15:** the user explicitly enabled isolated
reference Dice every epoch and clarified native validation locations. See
`epoch_validation_20260915.md` and the current operator guide. The text below
records the earlier compact-display change, before that authorization.

Per-epoch GT validation is **not implemented by this console change**. The
trainer currently has no val Dice/IoU/HD95; its audit NLL is explicitly labelled
selection NLL. The console shows `val Dice=--`, rather than substituting
pseudo-label agreement. The existing isolated reference evaluator runs after
the final complete prediction freeze.

The latest user request asks for per-epoch Dice, which requires a change from
the original final-freeze timing and an actual validation image/mask mapping.
Those two points were requested from the user while the independent console
fix proceeded. No validation paths or reference scores were invented. A
read-only follow-up by `/root/luna_eval` confirmed the current evaluator's
requirements and output schema for later integration.

## Checks

* Focused progress + trainer checks: **8 passed in 4.36 seconds**, including
  exact resume and identical model states with/without console telemetry.
* A TTY smoke exposed a missing newline after a delayed phase; corrected using
  the installed tqdm close behavior. The focused regression for that fix:
  **1 passed in 1.33 seconds**.
* Actual Python CLI was exercised both with redirected output and with a PTY,
  on synthetic ACDC CPU fixtures, physical batch 2, bounded to one optimizer
  group. Exit code **2 / partial**, with a single epoch summary and no printed
  affine, hash or full metric table. This is not a production batch fallback
  or a hardware validation result.
* Shell syntax and `git diff --check` passed. No GPU/SSH task was launched.

Raw redirected smoke: `runs/maskfree150/software_checks/compact_tqdm_cli_20260915/terminal.log`.
Scientific configs, loss functions, data discovery and checkpoint resume
guards are unchanged. The source hash guard still requires matching source
when resuming a checkpoint; this console build should be used for a fresh run.
