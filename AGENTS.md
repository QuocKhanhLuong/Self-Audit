# Current source of truth

Read ACTIVE_PIPELINE.json and docs/METHOD.md. The only trainer is train.py,
backed by self_audit/. This v2 is semi-supervised; do not claim no-GT training.
All audit labels count in the declared dense TRAIN-patient budget. No U/VAL/TEST
mask may influence training, pseudo-targets, label-subset selection or gates.

Do not revive frozen-Q/CineMA-bank trainers or combine their checkpoints with v2.
Archive: archive/pre-semisupervised-20261020 at 924265f7df604d3dca84dda49578b1a0cf28d821.
The original exact layer reference remains there, unchanged. Native v2 is a new
architecture/policy, not a parity claim against that reference.

Keep one shared state timeline: supervised reference -> fixed EMA targets ->
fixed candidates -> local solve -> actual functional trials -> commit -> EMA
once. Never feed an audited direction through AdamW afterward. No live state
may change during rejected trials. A changed objective or denominator is a new
experimental factor. EMA predictions are not independent truth.

All methods share the same labeled/U schedules, inputs, initialization and
pseudo-class normalization in a matched comparison. Keep source IDs unique.
Audit Brier includes negatives and marks absent positive classes explicitly.
It does not certify patient Dice. Exposed VAL is development, not confirmation.

Run python scripts/check_repo.py and python -m pytest -q before delivery. Report
actual tests, source/weight hashes, file paths and missing evidence. No patient
or GPU success may be inferred from manufactured tests. Do not run full patient
training, download/accept licenses, change active jobs, or delete local datasets.
Use isolated worktrees; never git clean -fdx around user data or checkpoints.
