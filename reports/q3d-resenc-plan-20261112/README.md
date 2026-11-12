# Q3D ResEnc — locked six-run PLAN

ACDC, Protocol A, seed 42. **PLAN ONLY / patient training NOT_RUN.** [PLAN.json](PLAN.json), [acceptance and lineage proof](ACCEPTANCE.json). Previous shared/reduced plans were not overwritten.

| Execution priority | K | T | Q | P | J settings | J count |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 8 | 1 × 8,000 updates | 1 × 1,000 updates | 1 cache pass | v1, v1u, c1, c0 | 4 |
| 2 | 5 | 1 × 8,000 updates | 1 × 1,000 updates | 1 cache pass | v1 | 1 |
| 3 | 10 | 1 × 8,000 updates | 1 × 1,000 updates | 1 cache pass | v1 | 1 |
| Total | | 3 | 3 | 3 | | 6 |

Every J has 8,000 optimizer updates. The runtime traverses dependencies in experiment order, completing K8 before K5 and K10; the storage order of `stage_jobs` groups producers and is not the execution order. `ACCEPTANCE.json.execution_order` records the actual dependency traversal. V1/V1u@8 share T/Q/P; C1@8 shares T/P and pins Q as optional source metadata. C0 has no teacher producer. Producer source-config hashes are fixed; future checkpoint/artifact hashes are null/NOT_RUN because no patient job has run.

Teacher: pinned nnFoundation ResEnc-L encoder, new decoder and heads. All real teacher recipes pass independent provenance validation against the actual downloaded checkpoint and committed phase receipt. Job blockers are **only missing manifest inputs**, not pretrained provenance.

## Dataset evidence

K8 TRAIN bytes rehashed to `e524c7531079996232fe71cdfe982a46a5950d49d91a8203eee396b0e67434a3`; actual schema `self_audit.data.v2`, role `training`, label budget 8. Existing `label_subset` selector reproduces nested L5 ⊂ L8 ⊂ L10 from the locked eligible IDs, seed42. No patient was reselected based on model results.

K5/K10 manifest bytes and DEV18 bytes are not available locally; their historical expected hashes do not substitute for a fresh file check. Thus the plan retains missing inputs as NOT_RUN. The expected shared DEV hash is `ba46934afe8ea5c47dd798019bee233c97c98320b358e2d40165e87a651dc7df`. Patient image/header availability and raw geometry consistency were not certified by this metadata-only plan.

## Comparison references

`teacher_supervised@8`: evaluate the shared T region head, **0 J**, `independent_baseline=false`; evaluation NOT_RUN.

`a3d_teacher_free@8`: **NOT_RUN**, reference only, no job. Existing `K8_PROVENANCE_RECEIPT.json` hashes to `6aa1c4b8772b2c8f1a1962debfebc5dcb046f99eb922535fc0a307c7332395a8` and explicitly says `INFO_HANDOFF_ONLY_NOT_RUN`. No completed K8 checkpoint and equivalent training budget were proven.

Secondary arms and mean_teacher remain implemented/tested but **NOT_PLANNED**. K5/K10 are label-count curves only; the VQ/teacher/physiology decisions apply only at K8. TEST and all patient metrics are NOT_RUN. No seed variance is estimated from this one-seed plan.
