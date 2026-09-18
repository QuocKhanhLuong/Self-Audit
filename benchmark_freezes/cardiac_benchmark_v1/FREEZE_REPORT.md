# Cardiac benchmark v1 scientific-freeze report

## Identity and scope

- proposed freeze ID: `cardiac-benchmark-v1-6feede910fdcb9c9`
- scientific payload SHA-256:
  `6feede910fdcb9c97543568f855cc5a4c6fd782dda0df151e117aaeb42ad1110`
- repository commit bound by the payload:
  `826a11ca9c37fd024c9d7d83df594f8b051ec9ca`
- execution branch: `huann/cuts-dfc`, whose HEAD equalled `origin/main` at
  freeze start; worktree was clean before artifacts were created.

This is a local, uncommitted **pre-implementation freeze artifact set**.  It
cannot be an accepted official scientific data freeze until the artifacts are
reviewed/committed and real image-only manifests plus physical GT-isolation
evidence are attached in a new freeze revision.

## Verified source-of-truth behavior

The bound FreeMask sources retain patient-level assignment, split seed `42`,
deterministic BLAKE2b ranking, official membership before 70/15/15 fallback,
all acquired frame enumeration, all Z-target enumeration, and endpoint
replicated `[z-1,z,z+1]` context.  `Info.cfg`, masks, annotation paths, and
annotation-selected file names are excluded by the image-only firewall.  No
ED/ES or foreground/mask filter is part of generation membership.

## Real-data status

| Dataset | Image-only root | Shared scientific manifest | Patient/sample/source-hash results |
|---|---|---|---|
| ACDC | unavailable in this execution context | **DEFERRED** | not fabricated or inspected |
| M&Ms | unavailable in this execution context | **DEFERRED** | not fabricated or inspected |

No real materialization command was run.  When a separately mounted image-only
root is available, the only permitted path is:

```text
python scripts/prepare_shared_benchmark_manifest.py --image-root <IMAGE_ONLY_ROOT> --dataset acdc --output benchmark_freezes/<new-freeze>/data/acdc.shared_manifest.json
python scripts/prepare_shared_benchmark_manifest.py --image-root <IMAGE_ONLY_ROOT> --dataset mnms --output benchmark_freezes/<new-freeze>/data/mnms.shared_manifest.json
```

That later invocation must validate hashes, patient disjointness, all-frame ×
all-Z inventory, cross-consumer identity, and physical isolation before it can
change either dataset to `PASS`.

## Spatial and consumer contract

- target grid: `[224,224]`, from both bound FreeMask YAML files.
- whole FOV: true; crop: none.
- value resampling: FreeMask `masked_resize`, recorded as
  `masked_area_normalized_convolution`, never bilinear.
- grid SHA-256:
  `7c9d33fed0facbbabe736a5216bc599b65f1b465e3e26a3d19c624cf9be57949`.
- deterministic golden resize fixture SHA-256:
  `7f697a8d3b3cd2c0ddb8c944cde5deaf5cf4a6e992fd328d79edcee56a84166c`.

CUTS consumes the shared manifest in primary `CUTS-2D` and declared
`CUTS-2.5D` sensitivity mode.  DFC consumes the same central target record in
`DFC-Direct-2D-Default-MinL3`.  Their intensity normalizations remain
method-specific; membership, source identity, target grid, frame and Z do not.
Real-manifest cross-consumer identity is deferred with the real manifests.

## GT firewall

Schema/root-bounding firewall: **PASS_LOCAL**.  Fixture promotion, forbidden
field injection, and root traversal rejection are covered by the shared suite.
Physical GT isolation: **DEFERRED** because no separately mounted real
image-only root was available.  No claim of OS-level unmounting is made.

## Adapter v1 pre-implementation freeze

`adapter_v1_spec.json` freezes `cardiac_adapter_v1` as deterministic,
non-learned, GT-free, producer-neutral, permutation-invariant,
connected-component naming/merging only.  It uses 4-connected components,
4-neighbour boundary adjacency, strict topology-only enclosure, no region
splitting, no intensity resolution, no orientation resolution, and explicit
VOID (`4`) / validity / coverage semantics.  Exact ties become VOID; no raw
cluster numeric ID, method name, patient, dataset, or raster order may choose a
semantic role.

The fixture set has 14 synthetic image-only topology cases, including
permutation, split-background, K=3/DFC-MinL3, missing/ambiguous anatomy,
disconnected same-ID components, constant intensity, unavailable orientation,
non-square, and high-K cases.  Its SHA-256 is
`134540a8e9c561b3312473f36f5e8f6ea28a2cbcf31453e76692d7474dc214a5`.

## Validation

| Command/gate | Result |
|---|---|
| `tests/shared_benchmark` | 9 passed |
| CUTS cardiac P0 regression | 8 passed |
| DFC cardiac regression | 8 passed |
| freeze validator | pass |
| mutation of copied bound contract | rejected |
| `compileall src/shared_benchmark` | pass |
| `git diff --check` | pass |

No GT metrics, GT-based selection, split/inventory change, method-core change,
or adapter implementation occurred.

## Readiness

| Gate | State |
|---|---|
| Code/config freeze | READY TO COMMIT |
| Shared data-contract freeze | READY TO COMMIT |
| ACDC scientific manifest | DEFERRED |
| M&Ms scientific manifest | DEFERRED |
| Shared grid | READY TO COMMIT |
| CUTS consumer | READY LOCAL |
| DFC consumer | READY LOCAL |
| GT schema firewall | PASS LOCAL |
| Physical GT isolation | DEFERRED |
| Adapter input/rule/fixture freeze | READY TO COMMIT |
| Adapter implementation authorization | YES FOR FIXTURE-ONLY IMPLEMENTATION |
| Overall scientific benchmark freeze | NOT READY |
