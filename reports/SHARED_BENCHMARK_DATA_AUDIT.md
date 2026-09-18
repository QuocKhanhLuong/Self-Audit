# Shared benchmark data audit

## 1. Repository identity and scope

Audited repository: `Self-Audit` outer worktree only.  The relevant FreeMask
implementation is `src/self_audit_maskfree/`, not the older supervised
`src/self_audit/` pipeline.  The pinned FreeMask revision
`96c32b10fc7b8e09b48822e10ae9eb6cc149e253` exists and is an ancestor of the
audited `HEAD`; no differences under `src/self_audit_maskfree/` or the two
mask-free YAMLs were found between that revision and `HEAD`.

This is a read-only code/configuration audit.  No data root was inventoried,
no GT was opened for generation decisions, and no scientific manifest or
benchmark result is claimed.

## 2. Authoritative sources actually executed by FreeMask

| Concern | Authoritative executable location | Verified behavior |
|---|---|---|
| Run configuration | `configs/maskfree_acdc_150.yaml`, `configs/maskfree_mnms_150.yaml`, `src/self_audit_maskfree/config.py` | Strict `MaskfreeConfig`; configured `seed: 42`, `image_size: 224`, `depth_axis: 2`, and `protocol: auto`. `image_size` is config-driven (dataclass default is 128), not globally hard-coded to 224. |
| Discovery and split | `src/self_audit_maskfree/data/discovery.py` | Image-only source enumeration, patient identity, all-frame/all-slice records, patient split, hashes, native geometry checks, and manifest identity. |
| Firewall | `src/self_audit_maskfree/data/firewall.py` | Refuses annotation directories, mask-like names, ACDC `Info.cfg`, and `diagnosis.csv`; the discovery data layer cannot read them. |
| Decoding/context/preprocessing | `src/self_audit_maskfree/data/geometry.py`, `src/self_audit_maskfree/data/dataset.py` | Canonical NIfTI/NPY read, `[z-1,z,z+1]` stack, native-role partition then fit-only normalization then masked-area resize. |
| Observation partition | `src/self_audit_maskfree/data/partition.py` | Study-grid-scoped fit/select/verify spatial partition; this is internal FreeMask observation logic, not a shared benchmark split. |
| Trainer consumption | `src/self_audit_maskfree/trainer.py` | Calls `discover_dataset(...)` in `_setup`, trains only `split="train"`, and writes the in-memory result to a run-local `data_manifest.json`. Finalization constructs datasets for train/dev/test. |
| Standalone materialization | `scripts/prepare_maskfree_data.py` | Calls the same `discover_dataset`, then writes `manifest_<dataset>.json` plus inventory files. It is the correct Case-B producer. |
| Isolated reference evaluation | `src/self_audit_maskfree/evaluation/reference.py`, `epoch_reference.py`, `freeze.py` | Separate GT-reading evaluator after frozen-prediction validation; M&Ms ED/ES metadata is read only there. |

The tracked `splits/acdc_patient_split_seed42.json` and its producer
`scripts/acdc_split.py` are **not** FreeMask-authoritative.  They serve the
older supervised pipeline (`configs/self_audit_*.yaml`), are ACDC-only,
preprocessed ED/ES-only, train/val (80/20), and have no FreeMask test split.
They must not be reused for this benchmark.

## 3. Exact split semantics

`discover_dataset(root, dataset, seed=42, protocol="auto", depth_axis=2)`:

1. lists allowed image containers in sorted filesystem order while excluding
   forbidden paths;
2. derives `patient_id` from the source filename after known frame/time/series
   suffix stripping (not from pixels, labels, or source hash);
3. discovers every source frame, deduplicates only byte-identical cross-file
   re-exports, verifies compatible source grids within each study, then takes
   the sorted unique patient set;
4. calls `assign_splits(patients, dataset, seed, official)`;
5. assigns that one patient split to all retained source frames and all their
   Z-slice records.

The split is therefore patient-level and patient-disjoint.  A patient's
frames, source exports, and slices cannot cross train/dev/test unless the
discovery identity heuristic itself conflates or fails to conflate patients;
the implementation explicitly strips M&Ms `_tNN` so time exports share a
patient ID.  The exact fallback ratios are `train=.70`, `dev=.15`, `test=.15`.
The seed is actually consumed in `_hash_fraction(dataset, str(seed), patient)`
(BLAKE2b-64), followed by deterministic sort `(hash_fraction, patient_id)`.
Thus seed 42 is implemented, but it is a manifest-generation parameter, not a
literal `split_seed` field in the YAML schema.

Dataset-specific official-cohort handling is deliberately different:

| Dataset | Official rule before fallback | Consequence |
|---|---|---|
| ACDC | A `training` folder is treated as source location, not an official train cohort. An explicitly found `testing`/`test` folder is preserved as test; otherwise all patients receive the 70/15/15 rank-hashed split. | Current `test` is normally a deterministic held-out subset, not an official fresh ACDC test cohort. |
| M&Ms | If any official validation/dev folder exists, official `Training`/`Validation`/`Testing` memberships are preserved as train/dev/test. Only unassigned patients are rank-hashed into remaining required splits. | Native official folders are used when present; this is not an independently randomised M&Ms split. |

The policy sees only patient ID, seed, and applicable path-derived official
folder membership.  It does not read masks, diagnosis, `Info.cfg`, frame
annotations, image content, source hash, or GT availability.  The manifest
records this in `split_provenance.annotation_inputs=[]` and
`content_fingerprint_used=false`.

## 4. Meaning of test and sample inventory

`test` is membership in the above manifest split, not an ED/ES or
heart-containing-slice filter.  The concrete lineage is:

`dataset root -> allowed image source -> patient -> source-frame volume_id -> every Z slice unit_id -> [z-1,z,z+1] context`.

* Rank-4 NIfTI: every acquired index on axis 3 is emitted.  `frame_selection_rule`
  is `all_acquired_frame_indices_image_only`.
* Rank-3 source: frame index 0 is emitted.  ED/ES names are merely source
  naming; they do not cause selection.
* ACDC `Info.cfg` is specifically forbidden, so it cannot identify ED/ES.
  M&Ms phase metadata is only opened by the isolated evaluator to locate GT
  frames; it never enters discovery/generation.
* Every target Z index `0..depth-1` is emitted.  There is no mask-derived
  foreground/heart slice filter and no mid-ventricular-only policy.
* Context is `[clip(z-1,0,depth-1), z, clip(z+1,0,depth-1)]`; endpoints repeat
  the centre plane.  The central target is channel 1.  “central slice” means
  target Z of this stack, not the anatomical middle slice.
* Exact content-identical exported frames may be removed in favour of a
  canonical source; this is image-only deduplication and its evidence is
  stored in `duplicates`.

Generation membership is intentionally independent of annotation availability.
Evaluation eligibility is a separate later question: the reference evaluator
may be unable to find a matching mask/declared annotated M&Ms frame and records
that fact after prediction freeze.  This separation is necessary and should be
retained in the shared contract.

## 5. ACDC and M&Ms inventory behavior

| Field | ACDC | M&Ms |
|---|---|---|
| Source discovery | Allowed `.nii(.gz)`/`.npy` below root, excluding GT/annotation paths; rank 3/4 accepted. | Same generic decoder/discovery. `_tNN` suffixes are removed for patient identity. |
| Patient identity | Filename stem with known frame/series suffixes stripped. | Same, including time suffix stripping to keep temporal exports together. |
| Frames | Every rank-4 frame or rank-3 source frame. No `Info.cfg` ED/ES read. | Every acquired frame. Official ED/ES metadata is not used in generation. |
| Slices/context | Every depth-axis-2 slice; clipped 3-slice context. | Same. |
| Split | Explicit test folder if present; otherwise 70/15/15 image-only deterministic patient split. `training` is not official train. | Official Train/Validation/Test folders preserved when validation folders exist; fallback only for unassigned patients. |
| Annotation discovery | None. `Info.cfg` and mask names/trees forbidden. | None in generator. Masks/annotation paths forbidden. |
| Evaluation subset | Only isolated evaluator later matches source/re-export to reference. | Isolated evaluator reads official ED/ES indices only after freeze because sparse labels cannot define generator membership. |
| Native geometry | NIfTI affine/orientation/zooms/unit retained if valid; NPY is stored-grid only. | Same. |
| Target grid | Configured square `image_size` (checked templates: 224); no crop. | Same. |

An ED/ES-only *preprocessed* root can be ingested as images, but discovery marks
`annotation_selected_cohort` because its upstream selection may have depended
on masks.  Such an artifact is not sufficient evidence for a shared scientific
cohort without resolving that limitation.

## 6. Existing manifest/config artifacts

| Artifact | Producer / consumer | GT content / share safety | Status |
|---|---|---|---|
| `splits/acdc_patient_split_seed42.json` | `scripts/acdc_split.py`; older supervised configs | Split names only, but generated from preprocessed ED/ES supervised inventory; unsafe for FreeMask/CUTS/DFC contract | Tracked, no content hash field, not FreeMask-used. |
| Run-local `runs/maskfree150/<dataset>/<run_id>/data_manifest.json` | `MaskfreeTrainer._setup`; trainer/run artifacts | Image-only FreeMask manifest schema v2; contains source paths/hashes and no GT paths | Reusable only if an actual completed run artifact is supplied and validated; none is present in this worktree. |
| `manifest_<dataset>.json` | `scripts/prepare_maskfree_data.py`; externally supplied consumers only | Same image-only FreeMask schema v2 | Correct Case-B source, but no generated ACDC/M&Ms artifact is checked in. `manifest_id` is recorded; the file itself has no separately stored file SHA. |
| `inventory_<dataset>.json/.md` | `prepare_maskfree_data.py` | Image-only readiness/limitations | Companion only; not a manifest and no frozen file hash. |
| CUTS `cuts.cardiac.shared-manifest.v1` | `baseline/CUTS/scripts_cardiac/build_shared_manifest.py` | Intended translation of FreeMask discovery, no GT fields | No scientific instance; separate schema/hash and pinned-SHA assumption. |
| DFC `dfc-cardiac-shared-manifest-v1` | fixture helper only | Schema rejects direct prohibited keys, but fixture-only producer is not a FreeMask converter | No scientific instance; incompatible schema. |

Conclusion: Case A is not available in this checkout.  Case B must call the
outer repository's `self_audit_maskfree.data.discovery.discover_dataset` once
per real image-only dataset root with seed 42, then preserve its byte content
and a recorded SHA-256.  Consumers must read that canonical artifact or a
hash-bound projection generated from it, not independently rediscover/split.

## 7. Actual spatial/preprocessing policy

FreeMask preserves the entire stored FOV: inverse-transform metadata explicitly
states `crop=None` and `whole_field_of_view_no_image_informed_crop`.  There is
no mask/ROI crop.  NIfTI orientation is read as affine axis codes but arrays are
not reoriented or resampled into a common physical orientation.  Within a study,
mixed stored shape, depth axis, orientation, affine availability, source format,
spatial-unit declaration, or affine corner displacement above `1e-4` voxels
fails discovery.  Original per-source affine remains in records.  NPY has no
native affine/spacing and is explicitly stored-grid-only.

FreeMask order is exactly:

`native [z-1,z,z+1] -> study role mask -> fit-pixel 0.5/99.5 clip + mean/std -> zero role-excluded values -> masked normalized-area resize to [image_size,image_size]`.

`masked_resize` uses adaptive average pooling numerator/denominator (area-like
normalized convolution) and masks use `nearest-exact`; it is not ordinary
bilinear.  It processes the stack jointly in the sense that fit statistics are
across all three channels and the same XY role support applies to each context
plane, while the value pooling is per channel.  Labels exported back use
`nearest_exact`; probability inverse resize is `bilinear_then_renormalize`.
The inverse transform is fully described per unit.  Target 224x224 follows the
two committed benchmark YAMLs, but a caller can configure another multiple of
8 size, so “224” is a pinned benchmark configuration rather than an invariant
of the loader.

## 8. Cross-method preprocessing matrix

| Property | FreeMask | CUTS cardiac layer | DFC cardiac layer | Must match for benchmark? |
|---|---|---|---|---|
| Patient/sample membership | Generated FreeMask v2 manifest | Intended translated manifest | Separate fixture schema only | **Yes** |
| Split policy/seed | Executable 70/15/15/official policy, seed configurable (42 templates) | Requires 42 but translator has own schema | Requires 42 but no FreeMask importer | **Yes** |
| Target target-slice inventory | All frames/all Z | Uses manifest rows | Fixture record central slice | **Yes** |
| FOV/crop | Whole FOV, no crop | Whole extracted plane; no crop code | Whole selected plane fixture | **Yes** |
| Shared target grid | Configured 224 template; masked-area policy | Optional `target_hw`; no required 224 | record-declared grid, fixture identity | **Yes: exact declared target grid and transform, currently not achieved** |
| Context | 3 slice | 2D centre or 2.5D stack | Primary 2D centre only | Target slice/indices yes; input channel context may differ if declared |
| Intensity normalization | Stack-wide fit-only clip/mean/std for training; full-stack deployment variant | Stack 0.5/99.5 clip then [0,1], no z-score | Centre-only clip then z-score | No; method-specific but frozen/disclosed |
| Image resize interpolation | Masked adaptive-area normalized convolution | Bilinear `align_corners=False` | Bilinear `align_corners=False` | **Yes if a common input grid is claimed; otherwise mismatch must be explicit** |
| Geometry | Affine/orientation/spacing/provenance and inverse transform | Carries selected fields but no full transform | Fixture schema requests metadata but loader ignores much | **Yes: preserve metadata and declared transform** |
| GT access during generation | Firewall | schema-level key rejection, no real process firewall | schema-level rejection/fixture decoder | **Yes; physical execution firewall still needs shared test** |

## 9. CUTS audit

CUTS adds a cardiac shell, not a shared manifest consumer yet:

* `CUTS-2D` extracts the central plane and is the appropriate primary
  method-faithful-ish adaptation. `CUTS-2.5D` feeds the 3-plane stack and is an
  input-context sensitivity, not original CUTS.
* `dataset.py` uses the manifest context indices and clips boundaries correctly.
  It normalizes all three stack planes with shared stack percentiles and scales
  to `[0,1]`; 2D then takes the centre.  Resize is ordinary bilinear.
* `train_stage1.py` trains train only, validates on dev only, and does not load
  test in those loaders.  `export_latents.py` can export any requested split.
* `cluster_kmeans.py` creates per-sample PHATE + KMeans K=10 raw integer maps,
  preserving IDs and retrying only seeds 1 then 2 after an exception.
* It has no semantic adapter in the core.  Its current raw bundle hashes output
  files, but does not enforce a common semantic freeze.

Mismatches/blockers: (1) CUTS serializes a distinct `cuts...v1` translation
rather than consuming FreeMask v2; (2) `source_path` is absolute while a
canonical contract should have root-relative identity plus declared root; (3)
its required records omit FreeMask `volume_id`, frame/stored shape,
native/stored grid status, source-format, full spatial transform, and source
frame hash; (4) target grid is optional and its bilinear interpolation differs;
(5) source-reader accepts NPZ although FreeMask does not; (6) process-level
annotation path firewall is weaker than FreeMask's path gate; (7) no test proves
all-frame/all-slice parity against a real/fixture FreeMask v2 manifest.

## 10. DFC audit: code versus its documents

| Field | Actual DFC code | Contract judgment |
|---|---|---|
| P0.1a schema | `dfc-cardiac-shared-manifest-v1`, separate fixture builder; no FreeMask conversion | **MISMATCH** |
| Split seed/patient disjointness | Validator requires 42 and one split/patient | **MATCH structurally; unresolved against real FreeMask artifact** |
| Central target | `load_primary_2d` reads only `volume[z]`; neighbours are metadata | **MATCH for declared DFC-2D target semantics** |
| Frame/axis decoding | Fixture loader assumes `.npy`, rank 3 and Z is axis 0 | **MISMATCH** with FreeMask configurable depth axis 2/NIfTI/rank-4 frame decoding |
| Normalization | Central-only p0.5/p99.5 then centre z-score | **Allowed method-specific difference**, but must be frozen/disclosed |
| Grid | Record `shared_grid.shape`; bilinear resize | **MISMATCH** to FreeMask masked-area transform; only synthetic fixture support |
| Geometry | Schema requires fields, but loader only returns metadata; no affine-aware decoder/inverse operation | **MISMATCH** |
| Scientific/fixture guard | Rejects mock/fixture, checks source hashes/root/qualification/identity flag | **MATCH in intent**, **UNRESOLVED** because no real producer/launch wrapper executes it |
| Source hash | Validates fixture/scientific schema source SHA-256 | **MATCH in schema**, **MISMATCH** with direct FreeMask v2 field layout |
| Seed derivation | Versioned SHA-256 sample seed | **MATCH**, method-specific and deterministic |
| Primary MinL3 | Immutable `DFC-Direct-2D-Default-MinL3`, input 1, 100 channels, 1000 iters | **MATCH** with documented P0 policy |
| GT firewall | Prohibited exact keys only | **NEEDS FIX**: no robust recursive substring/path firewall and no actual pipeline entry point to prove GT-inaccessible execution |

The DFC documents accurately call real P0.1b/grid/adapter unresolved, but their
claimed “FreeMask-compatible manifest” is a design claim rather than current
consumer behavior.  The raw DFC core ends at an anonymous `[H,W]` `int32`
partition and is appropriately separate from semantic naming.

## 11. Existing tests and missing shared tests

FreeMask tests materially defend image-only discovery/firewall, all frame/slice
enumeration, patient disjointness, deterministic manifest ID, M&Ms time suffix
identity, boundary stacks, native affine/spacing behavior, mixed-grid rejection,
masked resizing, inverse transform records, freeze gates, and dev reference
split binding.  Relevant files include `tests/test_maskfree_data.py`,
`test_maskfree_firewall.py`, `test_maskfree_validation_split.py`,
`test_geometry_safety.py`, and `test_maskfree_epoch_reference.py`.

CUTS `baseline/CUTS/tests/cardiac/test_p0.py` covers mock-manifest rejection in
scientific mode, profile channel shape, train/dev isolation, core one-step
parity, PHATE retry and raw-bundle mutation.  DFC cardiac tests cover fixture
schema/central-only behavior, freeze mutation, seed/runner parity, and bounded
smoke.  None validates a shared real-or-synthetic FreeMask v2 artifact through
both consumers.

Create future shared tests under `tests/shared_benchmark/`, rather than
duplicating them in baselines: canonical-manifest schema/hash; FreeMask-to-common
projection; patient/sample equality across consumers; all-frame/all-slice and
boundary-context parity; affine/depth-axis/rank-4 decoding; exact target-grid
transform declaration; image-only prohibited paths/fields; fixture-versus-
scientific gate; raw/semantic freeze and evaluator separation.

## 12. Proposed authoritative common manifest contract

Use FreeMask `maskfree150.data.v2` as the source discovery artifact, plus a
new hash-bound common projection only if required for stable public schema.
Do not mutate or replace the source artifact.  The projection must reference
`source_manifest_id`, `source_manifest_sha256`, source schema and discovery
contract hash, and must be generated deterministically from every source
record.  It needs at least:

* `schema_version`, `contract_version`, `manifest_id`, `manifest_sha256`,
  `fixture_or_scientific`, `dataset`, `split_seed`, `split_provenance`;
* `patient_id`, `study_id`, `volume_id`, `sample_id`/`unit_id`, `split`,
  `source_id`, root-relative source path, `source_hash`, `frame_fingerprint`;
* `frame_index`, `frame_axis`, `slice_index`, `depth_axis`, `context_indices`,
  `frame_selection_rule`, and duplicate lineage;
* source format/dtype, full stored shape, native spatial shape, `native_hw`,
  axis semantics, native affine, orientation, spacing/unit validity,
  native-grid-export status, and study-grid compatibility;
* a versioned shared spatial-transform declaration: no crop, source FOV,
  target H/W, values/masks interpolation, forward/inverse behavior.  The
  actual 224 value must be pinned in this contract, not inferred by consumers;
* explicit image-only lineage (`mask_inputs_used=false`, annotation sidecars
  read=0), source roots/hashes, and `evaluation_eligibility` held separately
  from generation membership.

The existing FreeMask source manifest already supplies much of this except a
common target-grid transform (it is materialized in per-unit loader metadata),
explicit context indices, a `fixture/scientific` flag, and an externally
recorded file SHA.  Those are the minimal companion/projection additions.

## 13. Blockers and eventual changes (not made here)

**BLOCKED:** No real ACDC or M&Ms image-only root, frozen scientific
`maskfree150.data.v2` manifest, inventory receipt, source-manifest SHA-256, or
verified 224 shared-transform artifact exists in this worktree.  Therefore no
sample count, actual cohort, native orientation reliability, or scientific
grid equivalence can be asserted.

When implementation is authorised, likely changes are:

* new project-level `src/shared_benchmark/manifest.py` and
  `tests/shared_benchmark/test_manifest_contract.py`;
* `scripts/prepare_maskfree_data.py` (or a separate shared-materialization
  command) only to emit the hash-bound projection/receipt from the existing
  discovery result;
* `baseline/CUTS/src/cardiac_benchmark/{manifest.py,dataset.py}` and
  `scripts_cardiac/build_shared_manifest.py` to consume the canonical artifact
  and its spatial policy rather than create a parallel one;
* `baseline/DFC/src/cardiac_benchmark/{manifest.py,dataset.py}` to replace its
  NPY/axis-0 fixture-only decoder with the canonical image-only decoder or a
  proven equivalent; and
* `baseline/DFC/config/cardiac/dfc_direct_2d_minl3.yaml` only if it needs an
  explicit reference to the frozen shared contract (not a new DFC grid).

No such implementation was performed by this audit.

## 14. Evidence-based readiness verdict

| Item | Verdict | Evidence |
|---|---|---|
| Shared manifest contract | **NEEDS FIX** | FreeMask v2 discovery is authoritative, but no frozen scientific artifact/common projection is present and baseline schemas diverge. |
| Patient split contract | **READY** | Executable patient-level, seed-consumed split policy exists; external scientific artifact still needs materialization. |
| Test inventory | **READY** | All acquired frames and all Z slices are image-only enumerated; evaluation eligibility remains separate. |
| Shared grid | **NEEDS FIX** | FreeMask config requests 224, but no common frozen transform artifact exists and CUTS/DFC use bilinear paths. |
| CUTS integration | **NEEDS FIX** | Intended translation is not canonical consumption and loses/changes contract details. |
| DFC integration | **NEEDS FIX** | Current loader is NPY/axis-0 fixture-only and has an incompatible manifest schema. |
| GT firewall | **NEEDS FIX** | FreeMask firewall is strong; CUTS/DFC need a shared physical execution/firewall contract and tests. |

## 15. Implementation update (2026-09-18; local/fixture evidence only)

The identified common data-layer gap is now implemented without changing
FreeMask discovery or split code.  `src/shared_benchmark/manifest.py` projects
only completed `maskfree150.data.v2` discovery output into
`shared_benchmark_manifest.v1`; it never assigns patients itself.  The
projection is canonicalized by sample ID, has a root-relative source locator,
source content SHA-256, explicit frame/z/context identity, native/stored
geometry facts, and a byte-stable scientific payload hash.  Local roots and
timestamps are receipts rather than scientific hash inputs.

`src/shared_benchmark/spatial.py` reads the two pinned FreeMask configs,
records their config hashes, and reuses FreeMask `masked_resize` for the
whole-FOV shared value transform.  It does not substitute bilinear resize.
CUTS and DFC now consume this record/grid contract; each retains its declared
method-specific intensity normalization before shared spatial resampling.
DFC remains central-slice-only in its primary 2D profile and CUTS retains its
2D and declared 2.5D sensitivity profiles.

The shared firewall rejects annotation/GT/metric/oracle-shaped fields before
hash validation, permits only root-bounded relative image locators, and makes
a fixture/scientific flag immutable under validation.  Scientific validation
requires a separately supplied image-only root and matching source hashes.
This is a schema/launch-contract firewall; a deployment must still prove that
GT storage is physically unmounted or otherwise inaccessible.

Local validation passed: `tests/shared_benchmark` (6),
`baseline/CUTS/tests/cardiac/test_p0.py` (8), and
`baseline/DFC/tests/cardiac` (8).  No real ACDC or M&Ms roots were available,
so no scientific manifests, counts, grid receipts, or GT metrics were made.

### Updated readiness

| Item | Verdict | Evidence and limit |
|---|---|---|
| Shared manifest contract | **READY locally** | Versioned canonical projection and hash/freeze tests pass; real materialization is deferred. |
| Patient split contract | **READY** | Projection consumes FreeMask discovery/split output and has no split implementation. |
| Test inventory | **READY** | All discovered frames x Z records and endpoint context are tested. |
| Shared grid | **READY locally** | Config-provenanced whole-FOV masked-area contract is shared by both consumers. |
| CUTS integration | **READY locally** | CUTS 2D/2.5D consume shared records; focused regression suite passes. |
| DFC integration | **READY locally** | DFC central 2D consumes shared records; focused regression suite passes. |
| GT firewall | **PARTIAL** | Schema/root/hash firewall tests pass; physical server isolation is deployment evidence still required. |
| Real ACDC scientific manifest | **DEFERRED** | No real image-only root materialized. |
| Real M&Ms scientific manifest | **DEFERRED** | No real image-only root materialized. |
