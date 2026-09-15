# Luna data-layer delivery

## Scope

The W3 data layer now treats one acquired Z slice of one source frame as one
unit. Discovery enumerates every frame in rank-4 image sources and every slice
in every retained volume. A stable `volume_id` identifies source path plus
frame; `unit_id` appends the Z index. Patient split assignment uses the
patient identity only, including M&Ms `_tXX` suffix stripping.

## Correctness changes

- Duplicate collapse is limited to cross-file frames with identical native
  geometry and identical image content. Unrelated same-patient series survive.
- A study with mixed native shape, affine, or orientation fails closed with
  `MixedStudyGeometryError`; discovery does not approximate a role map across
  incompatible grids.
- `native_hw` excludes the declared depth axis. Native affine availability is
  reported separately from millimetre spacing. Spacing is emitted only for
  finite positive spatial zooms with a recognised NIfTI header unit, including
  nibabel's `meter` and `micron` labels.
- Source bytes are fingerprinted once per selected file for exact-resume
  provenance. The source/content fingerprints do not seed patient splits or
  role partitions.
- Verification intensities are materialised only from
  `load_verification_unit` after the authoritative W7
  `export.validate_freeze(..., require_complete=True)` gate. Hash-only or
  arbitrary receipts are rejected.
- Dataset iteration passes its already selected record into the builder,
  avoiding an all-record index rebuild for every Z/T item.

## CPU evidence

The focused data and firewall tests were exercised with synthetic NIfTI/NPY
fixtures using the repository's CPU Python environment. After the coordinator
added the legitimate `spacing` metadata key, the canonical run passed without
monkeypatching:

```text
test_loads_without_masks_and_refuses_annotation_paths: passed
test_discovery_fails_closed_on_mixed_native_study_grid: passed
test_mnms_time_suffixes_share_one_patient_split_identity: passed
test_selection_and_verify_mutation_cannot_change_the_fitting_view: passed
test_geometry_splits_and_the_freeze_gate_on_sealed_verification: passed
firewall study-wide mutation, freeze gate, and fit-only bank checks: passed

Canonical result: **10 passed** (`test_maskfree_data.py` and
`test_maskfree_firewall.py`). Python compilation of the owned data, script, and
test modules also passed.
```

The canonical retained pytest artifacts belong under
`reports/maskfree150/software_checks/`. These are software fixtures only; no
real-data, CUDA, GPU, or clinical metric is claimed.

## Integration note

The fitting metadata emitted by `_unit_record` includes `volume_id`,
`num_slices`, and `spacing`; the coordinator must include those keys in the
`FittingView.metadata` whitelist. The full `source_hash` remains in the
manifest and `TrainingUnit.record` export identity for exact resume and is
intentionally excluded from fitting metadata. Declared `spacing_mm`, spatial
unit, and XYZT units are likewise retained in the export record. The exporter
must continue to group by `volume_id`, never by patient, when assembling all
enumerated units.
Finalization must provide W7's required unit, checkpoint, and nuisance artifact
requirements so the complete freeze proof is present before verification.
