# FUSE semi K8 delivery and full-label archive

## Track A: semi K8

Track A used the locked ACDC L8 patients `patient007`, `patient010`,
`patient011`, `patient024`, `patient026`, `patient041`, `patient077`, and
`patient099`. The run exposed their 16 ED/ES TRAIN masks; U contained 2,132
image-only frames across TRAIN80. VAL18 and TEST50 stayed disjoint and retained
their pinned native geometry. The access audit blocked 144 disallowed mask
paths and recorded zero non-L8 TRAIN mask reads; this is data-contract evidence.

The run completed T, J, VAL selection, export, and its single TEST transaction
on 2026-11-13 at 17:53:56 UTC. T used 8,000 labeled-frame exposures (4,000
optimizer steps, physical batch 2); J completed 100 epochs / 8,000 updates.
`TEST_ONCE/TEST_TRANSACTION.json` is committed on attempt 1. The deployment
choice was student, selected by VAL18 under the locked 0.01 margin. TEST was not
used for selection or tuning. See [Track A results](track-A/README.md) for the
VAL18/TEST50 comparison, patient-paired analysis, checkpoint identities, and
receipt hashes.

## M&Ms transfer status

M&Ms geometry passed under the pinned `mnms_open345_mm_polar_v1` policy, and
target L8 is locked. The current transfer PLAN (`observed_utc`
2026-11-13 18:03:00 UTC) records Track B as `FAILED` in `PROFILE_TJ`, before a
profile receipt was produced. It records the exact error as `ValueError: source
mask contains fractional labels`, raised at
`self_audit/candidate_fuse/mnms_data.py:525` during zero-shot teacher evaluation
on development VAL data. B target training and TEST are `NOT_RUN`; B0 is
optional and `NOT_RUN`. The launch receipt is an older 15:42:23 UTC snapshot
showing the queue waiting for A. The A-only monitor did not launch or modify B.

## Separate full-label comparison

The completed full-label result remains a separate all-TRAIN80-label run. Its
paired results and provenance are preserved in
[the full-label report](full-label/README.md). The full-label checkpoint and
matching result bundle are already archived on Drive with verified SHA-256 in
`DRIVE_RECEIPT.json`; this delivery uploaded no large model artifacts.

## Software evidence

The repository software checks and CPU smoke results are retained in
`check_repo.txt`, `pytest.txt`, `pytest_focused.txt`, and
`SOFTWARE_ONLY_SMOKE.json`. They establish software behavior only, not patient
efficacy or clinical benefit.
