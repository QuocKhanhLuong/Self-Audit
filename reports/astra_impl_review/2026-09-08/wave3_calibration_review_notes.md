# W3.2 independent draft review — REVISE, 2026-09-08

Not a final finding on completed code: worker is still implementing. Root read the full
initial calibration_lineage.py and ran a CPU synthetic boundary probe with a real tiny
SelfAuditNet checkpoint saved by save_checkpoint, bound by bind_evaluation_checkpoint,
and expected metadata built from the actual synthetic loader using build_expected_lineage.
No training or real-checkpoint validation.

Four independent mutations of the copied artifact lineage returned `verified=True`:

| Mutation | Actual draft | Required |
|---|---|---|
| Remove cohort | True | Reject missing required measurement extent |
| Remove evaluation_code | True | Reject missing calibration-run provenance |
| Change observed_samples to 1 and covers_full_split to False | True | Reject incompatible calibration cohort/protocol |
| lineage_schema_version=True | True | Reject boolean schema version |

Root probe used the same runtime weights/loader on both sides, changed only the artifact
copy, and invoked the public verify_calibration_lineage boundary. The synthetic dataset
has no real patient provenance; this probe proves these validation gaps only.

Requests sent in `msg_def50d30662e`: enforce required recorded fields and schema types;
validate exact cohort extent beyond the full split membership signature; do not allow
require_disjoint_patients=False to claim overlapping patients are independent; distinguish
report-only changes from functional code changes using a scoped source-content identity.
Producer owner received the source-signature dependency in `msg_25b5aa618660`.

These are correctness gates, not novelty contributions. Do not bypass old tests to retain
legacy deployable behavior. Legacy inspection must remain explicitly unverified.

Additional code finding sent in `msg_ddbedc5c0e10`: the draft
audit_checkpoint.resolve_tau_accept skipped verification when expected_lineage was omitted
and returned a usable tau with lineage_verified=False to preserve isolated precedence
tests. Requested fail-closed at this resolver boundary whenever an artifact is consulted;
a later build_report boolean guard is not a replacement. Existing tests may be migrated
to valid tiny runtime lineage, not retained with a deployable compatibility bypass.

## Header versus nested lineage — independent reproduction, 12:16 UTC

Loaded the current `tests/test_calibration_lineage.py` through importlib and called its
`producer.__wrapped__(Path(tempdir))` to create a real tiny checkpoint, binding, loader,
and artifact. `verify_calibration_lineage(artifact, runtime_lineage)` returned True on the
unchanged positive control. Deep-copied the artifact and mutated exactly one header field:

| Mutation | Actual before correction |
|---|---|
| checkpoint_sha256 = f repeated 64 times | ACCEPTED, verified=True |
| t_max = 999 | ACCEPTED, verified=True |
| source_split = test | ACCEPTED, verified=True |
| metric_contract = audit_target_legacy_one_v1 | ACCEPTED, verified=True |
| tau_accept = NaN | ACCEPTED, verified=True |

Command environment: PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src OMP_NUM_THREADS=1
MKL_NUM_THREADS=1 python; temporary synthetic fixture only. Root read save_calibration,
load_calibration and verify_calibration_lineage: nested lineage was checked without
cross-checking the duplicated public header. Assigned to AGY calibration replacement
`ctx_f9d2b9acf721` via `msg_105a1c849df3` and follow-up reproduction. Gate C remains REVISE.
