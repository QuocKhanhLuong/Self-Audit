# PG K8 A4 padding recovery

Status: **LOCAL_VALIDATION_PASSED_PENDING_DEPLOYMENT**. The production campaign remains stopped;
no resume has been attempted for this failure.

A4 stopped at 2026-11-16 23:27:08 UTC with a positive-class-mass validation
error on its first post-warmup step. The durable checkpoint is update 800,
epoch 10, SHA-256 `1c557ec57ad4156977155571d20c6388e239fe5a4cd630ca9440b20716237866`,
with 60 committed epoch rows. S1/S2/S3 completed 10,000/8,000/4,000 updates.

An isolated replay restored this checkpoint and reproduced the original error.
The next sampled keys were L patient041/frame 10, U anchor patient099/frame 11,
and U partner patient090/frame 2. The first mixed PG packet had shape
`[1,4,16,224,154]`, FP32, and 435,408 valid voxels out of 551,936.
Fusion, teacher and clean-student probabilities each had 115,200 zero-mass
voxels, all invalid; no valid voxel had zero class mass. The original validator
rejected these padded cells before applying the validity mask.

Luna implemented the narrow PG correction and regression tests. The fix keeps
finite/nonnegative checks global, requires positive mass on valid support,
and leaves zero-mass invalid probabilities at zero. The PG code coordinator reviewed the final source and test hashes with no
blocking findings and cleared the stopped-boundary deployment after the
measured validation results.
Local validation passed 28 reliability tests, 211 PG tests, repository checks
and a fresh unique two-update CPU smoke. The isolated patched GPU replay
completed update 801 with the original configuration, every existing optimizer
state advancing one step, and the production checkpoint unchanged. This is
bounded replay evidence; production recovery has not yet been demonstrated.

Original checkpoint, failure log, queue, epoch rows, configuration and shared
stage receipts are archived under
`code_updates/a4-probability-20261116T233344Z/`, with `PRESERVED_SHA256.json`.
The configuration, split manifests and shared artifacts remain pinned. The
matching JSON receipt records the measured evidence and outstanding checks.

All four arm exports, finalization and committed final TEST receipts remain
outstanding. The named PG automation stays enabled. M&Ms remains
preparation-only; historical tracks and runs are unchanged.

Diagnostic setup failures and the prior accidental reused-directory smoke
invocation are explained and preserved in the archive diagnostics manifest.
The fresh smoke exited zero and has unique epoch rows; a stale result from the
failed reused-directory invocation is not used as evidence.
