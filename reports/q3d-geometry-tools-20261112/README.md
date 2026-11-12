# Q3D geometry audit and repair tooling

Base: `ba35a056b714ec2cd8f838324c77cca3e6117fe2`. The user asked to complete
independent software work while the original ACDC data are being restored.
After the data became available, the original-header audit completed and
triggered the Phase A stop condition. This delivery does not certify corrected
patient geometry or replace the historical phase receipt and campaign plan.

## Phase A stop: original headers conflict

The read-only audit used nibabel 5.5.0 on `vast-gpu`, reading original CINE
headers and hashing their bytes without loading voxel or GT arrays. All 148
requested patients were covered. Full patient-level spacing and affine tables:
[CSV](header-audit.csv), [JSON](header-audit.json),
[reproducible audit script](audit_native_headers.py).

| Cohort | Headers read | pixdim versus selected-affine conflicts |
| --- | ---: | ---: |
| TRAIN80 | 80 | 52 |
| DEV18 | 18 | 11 |
| TEST50 | 50 | 49 |
| Total | 148 | 112 |

For example, `patient010` has native XYZ pixdim `[1.5625, 1.5625, 10]`,
but coded sform (`sform_code=2`) is `diag(-1,-1,1,1)`, whose spatial axis
lengths are `[1,1,1]`. Its qform contains the pixdim scale but has code 0.
Selecting that uncoded qform instead would be a new geometry interpretation;
the audit does not make that substitution. The other 36 patients have both
form codes 0; nibabel constructs a fallback affine, which this strict repair
workflow does not certify as authoritative. Raw headers remain unchanged.

The new download therefore does not resolve the physical-geometry blocker.
The requested stop condition prevents manifest repair and all downstream
Phase B–D execution. TEST50 original headers were inspected, but no prepared
TEST geometry manifest was available, so its manifest comparison is explicitly
`NOT_RUN_NO_TEST_MANIFEST`. TEST labels were not opened.

The restored K5/K8/K10 manifests retain the same TRAIN80 cohort, L8 matches
the SHA-verified historical K8 manifest, and their labeled sets satisfy
**L5 ⊂ L8 ⊂ L10**. These are restored provisional manifests, not corrected
geometry artifacts; their SHA values appear in the JSON audit and must not
replace the frozen campaign pins. No patient was reselected.

## Evidence boundary

| Work | Evidence |
| --- | --- |
| Header validation, geometry-only repair, lineage checks | SOFTWARE_ONLY; generated NIfTI fixtures |
| Recompute native pooling schedule and prepare canonical ERF probe | SOFTWARE_ONLY; no new patient-derived schedule or measured phase |
| Reject invalid ACDC geometry before Q3D voxel/GT readers | SOFTWARE_ONLY; metadata fixtures and reader sentinels |
| Bounded GPU profile command and prerequisite checks | SOFTWARE_ONLY; no GPU timing or patient update |
| Audit all original TRAIN80/DEV18/TEST50 headers | READ_ONLY_ORIGINAL_HEADER_AUDIT; completed, BLOCKED_HEADER_GEOMETRY |
| Corrected TRAIN80/K5/K8/K10/DEV18 manifests and their SHA-256 | NOT_RUN |
| Revalidate L5 subset L8 subset L10 on corrected manifests | NOT_RUN |
| New ERF measurement and locked teacher stride/phase | NOT_RUN |
| L8 GT physiology with corrected millimetre geometry | NOT_RUN |
| PLAN v2 (3 T, 3 Q, 3 P, 6 J) | NOT_RUN; do not invent manifest or artifact hashes |
| Real GPU profile: 20 T + 20 J(v1) optimizer updates | NOT_RUN |
| Campaign launch, efficacy and clinical conclusions | NOT_RUN |

## Preparation audit

The current ACDC preparation source obtains geometry from the original cine
affine and rejects unknown spatial units unless an explicit unit declaration
is provided. It does not contain a silent `(1,1,1)` fallback. The historical
K8 manifest nevertheless contains 52 unit-spacing patient entries. Those IDs
match the historical unknown-unit header population. This evidence identifies
the affected artifacts; it does **not** identify the exact historical writer
that introduced the wrong geometry. The untracked historical bridge script
was not recovered. The restored K10 manifest now matches the historical full
TRAIN source SHA `4982655478334405a538c5ced3fe5093c4305c0c9ba7c758a6f54e20247a1cb5`.

The direct header check establishes the mechanism in current preparation:
`prepare_acdc.py` takes `cine.affine`; `native_geometry_zyx` derives spacing
from its column lengths. Nibabel chooses the coded unit-scale sform over the
uncoded pixdim-scaled qform. The baseline preparation does not compare those
affine lengths against header pixdim. The new Q3D-only tooling rejects this
conflict before writing a corrected manifest. Frozen baseline sources were
not modified.

Historical receipts include cine-versus-frame affine discrepancies and an
explicit millimetre declaration. A matching array shape alone cannot prove
that an arbitrary prepared array has the original cine's voxel ordering.
The repair input therefore needs explicit source mapping provenance as well
as valid header geometry. An unknown unit cannot silently become millimetres.

The old manifest was also used for A-3D K8 runs. Their SDF/mm-dependent
results **may be affected** by its geometry. This is a protocol note, not a
claim about a measured effect or completed run; no baseline was rerun on
patients and no baseline source was changed.

The new engine metadata guard was also exercised against the historical K8
manifest with SHA-256
`e524c7531079996232fe71cdfe982a46a5950d49d91a8203eee396b0e67434a3`:

```text
SOFTWARE_ONLY metadata guard: BLOCKED: ACDC patient 'patient010' has invalid geometry: geometry spacing [1,1,1] is not accepted as authoritative
```

Only manifest JSON was read for that check; no patient image, mask or original
header was accessed. The manifest-based engine route checks both protocol-B
views before constructing either loader. A caller-supplied `Q3DData` object
may already have been materialized externally; its metadata are checked
before further engine access, not retroactively before the caller's reads.

## Pending protocol amendment

After all original headers pass, geometry may be replaced in new output
manifests only. Patient IDs, split membership, selector, image hashes, mask
hashes and labels must remain identical. The amended manifests need receipts
binding old and new bytes and proving that only geometry changed. The
historical manifest, phase receipt and plan remain preserved.

The teacher topology must then be recomputed from corrected TRAIN median
native spacing and median native shape using the pinned nnU-Net pooling
rule. No resampling or crop is introduced. The ERF experiment retains seeds
42–57 and the predeclared 2-voxel decision rule. Only its new measured receipt
can authorize new stride/phase values in the recipe and spec §3–§4.

GPU profiling remains downstream of the geometry/phase checks. Stage J
requires genuine hash-verified T/Q/P artifacts and passed production gates;
a profile must not manufacture a codebook or bypass a failed gate. Timing
the first 20 T updates measures T-a warmup only. T-b, Q, P and evaluation
costs cannot be claimed as measured from that sample. Profile output belongs
in an isolated temporary directory, with measurements retained separately.

The current config validator still pins the historical phase receipt and old
manifest contracts. They are intentionally not relaxed while Phase A is
blocked. Registering the measured new phase/manifest pins in config validation
is part of the pending Phase B lock, so the new profile utility is **not
ready to execute against current real recipes**. Its software tests do not
constitute that integration or a profile measurement.

The user subsequently authorized campaign launch after all required gates and
the bounded GPU profile pass: K8 V1, then K5 V1, K10 V1, and finally the
registered K8 V1-u/C1/C0 comparisons. A server window must show live training
logs. [Launch request](LAUNCH_REQUEST.json) preserves that authorization and
order; it is not a runnable PLAN or evidence that a server session was started.
No further launch confirmation is needed once the prerequisites pass. The
current geometry stop still prevents profiling and campaign execution.

## Validation

All checks below are **SOFTWARE_ONLY**. No patient training or GPU measurement
was performed. The original-header audit above is separate read-only data evidence.

- [Full pytest stdout, verbatim](pytest.stdout.txt): `593 passed, 2 skipped in 187.31s (0:03:07)`.
- [check_repo stdout](check_repo.stdout.txt): `PASS`.
- [All-arm smoke stdout](smoke.stdout.txt) and [summary](smoke-summary.json): all 13 recipes completed.
- Default `OMP_NUM_THREADS` was unset, with 8 parent Torch threads.
- The full-suite clean-copy regression passed; its isolated committed repo had
  empty `git status --porcelain` after running the child suite. The parent
  worktree's Git status was also identical before and after the checks.
- Frozen A/A-3D sources/configs and historical phase/plan artifacts have no diff.
  Earlier focused baseline validation passed 184 tests.

The synthetic tokenizer gate was NOT_READY and its explicit smoke-only
override was logged. V1 and V1-u consumed identical T/Q/P artifact hashes;
their omega vectors differed as expected, with mean 1 and the same patient
cohort. These results do not establish real-data gate passage.

[Execution metadata](validation.execution.json) records commands and environment.
[Independent review findings and resolutions](REVIEW.md) records the defects
fixed before the final suite. Pre-existing untracked research/download files
were preserved; no cleanup of unrelated files was used to claim a clean repo.
