# W3 delivery — image-only ACDC / M&Ms data layer

Verdict: **PASS** for the software contract, with the limitations below.
No real ACDC or M&Ms root exists in this checkout, so every number here is
evidence about the data layer's behaviour on synthetic CPU fixtures, not
evidence about real cardiac data. Real-root inventory is **NOT INSPECTED**.

Worker: Claude Opus through Orca (model identity as exposed by this session:
`claude-opus-5`). Code only, on the existing `main`. No worktree, no GPU, no
remote host, no commit, no push, no staging.

## Owned files delivered

| Path | Status | Purpose |
| --- | --- | --- |
| `src/self_audit_maskfree/data/__init__.py` | new | public API surface |
| `src/self_audit_maskfree/data/firewall.py` | new | mask / annotation path refusal |
| `src/self_audit_maskfree/data/geometry.py` | new | native geometry, masked resampling, inverse transform |
| `src/self_audit_maskfree/data/discovery.py` | new | `discover_dataset`, splits, dedup, protocol resolution |
| `src/self_audit_maskfree/data/partition.py` | new | study-level fit/select/verify partition + guard band |
| `src/self_audit_maskfree/data/dataset.py` | new | `ImageOnlyDataset`, verification gate, deployment view |
| `scripts/prepare_maskfree_data.py` | new | manifest + readiness inventory CLI |
| `tests/test_maskfree_data.py` | new | 3 focused checks |
| `reports/maskfree150/W3_delivery.md` | new | this report |

No file owned by another worker was read-modified. `src/self_audit_maskfree/contracts.py`
was consumed as frozen and not edited.

## API as implemented

Contract signatures, honoured exactly:

```python
discover_dataset(root, dataset, *, seed=42, protocol='auto', depth_axis=2) -> dict
ImageOnlyDataset(manifest, split='train', image_size=128, seed=42)   # __len__, __getitem__ -> TrainingUnit, unit_ids
load_verification_unit(manifest, unit_id, freeze_manifest, *, image_size=128, seed=42) -> ScoringView
load_full_input(manifest, unit_id, *, image_size=128) -> (Tensor[3,H,W], record)
```

Additional exact helpers this package exports, as W3 is required to document:

```python
build_training_unit(manifest, unit_id, *, image_size=128, seed=42) -> TrainingUnit
build_partition(height, width, *, study_id, seed=42) -> RolePartition
partition_identity(*, study_id, height, width, seed) -> str
validate_freeze_receipt(freeze_manifest, *, manifest, partition_id) -> dict
masked_resize(values[C,H,W], support[H,W], out_hw) -> (values, support)
nearest_resize_mask(mask, out_hw) -> mask
inverse_transform_record(...) -> dict
read_geometry(path) -> dict
save_manifest(manifest, path) -> Path ;  load_manifest(path) -> dict
assert_image_only(path) ;  forbidden_reason(path) ;  is_image_only_path(path)
```

`ImageOnlyDataset` additionally exposes `patient_ids`, `unit_index(unit_id)`,
`partition_ids()` and `fingerprint()`. **`fingerprint()` is the value W5 should
store in resume state** — it is a hash over `(manifest_id, split, image_size,
seed, [(unit_id, partition_id)…])`, so a changed cohort, split, resolution or
partition fails a resume closed rather than silently migrating.

### Keys W7 / W5 need from `TrainingUnit.record`

`unit_id, study_id, patient_id, dataset, split, manifest_id, protocol,
protocol_reason, partition_id, partition_spec, image_size, native_hw, depth,
depth_axis, slice_index, frame_index, frame_selection_rule, source_format,
native_geometry, native_geometry_reason, spacing_valid, orientation, zooms,
native_affine, export_grid, native_grid_export, inverse_transform,
normalization, support_counts, capabilities, cohort_provenance,
acquisition_id, mask_inputs_used, deployment_only`

`inverse_transform` is the export contract: `native_shape`, `depth_axis`,
`slice_index`, `frame_index`, `crop` (always `None` — whole FOV, no
image-informed crop), `forward_resize`, `inverse_resize` (nearest for labels,
bilinear-then-renormalize for probabilities, support-weighted for validity) and
`native_grid_export`. **When `native_grid_export` is `False` the only honest
destination is the stored grid**; W7 must not emit a native-grid NIfTI claim
for those records.

### Freeze receipt contract required by `load_verification_unit`

This is the one interface W3 had to define itself; **W7 should align
`freeze_predictions` output with it or tell me to change it**:

```json
{"dataset": "acdc", "manifest_id": "<data manifest id>",
 "partition_ids": ["<partition id>", "..."],
 "files": {"<absolute path>": "<sha256>"}}
```

All four keys are required. Validation checks dataset match, manifest-id match,
membership of this unit's `partition_id`, and that every listed file exists with
the recorded SHA-256. Anything else raises `FreezeReceiptError` and the sealed
observations stay sealed. Extra keys are ignored, so W7 may add fields freely.

## Design decisions and why

**Study-level role partition (changed after the coordinator's follow-up).** The
fit/select/verify map is seeded from `(study_id, native H×W, seed)` and
explicitly *not* from `unit_id`. Every slice and every cine time of one study
shares one XY role mask, so a voxel sealed as `verify` in one unit cannot
reappear as fitting context when the neighbouring slice becomes the centre.
`partition_spec["role_mask_scope"] == "study_grid"` records this. Duplicate
re-exports are excluded from the unit list entirely, so they cannot smuggle a
second role map for the same acquisition.

**Order of operations is the leak surface.** Partition on the native grid →
normalize from fitting pixels only → zero withheld voxels → resample per role.
Normalizing first would let a withheld intensity move the mean/std the fitting
view is expressed in; resampling first would let it bleed across the block
boundary. Check 2 exists to hold this order.

**Masked resampling.** Values and their role mask are resampled as a normalized
convolution, `pool(values·mask) / pool(mask)`, via `adaptive_avg_pool2d`, so a
resized pixel is a mean of same-role native pixels only. An output pixel keeps
a role only when ≥ 0.5 of its native footprint carried it (`SUPPORT_FLOOR`).
Independent nearest resizing of three native masks can still round two roles on
to one output pixel, so a fixed priority `verify > select > guard > fit` is
applied afterwards; the fitting support can never win a contested pixel.

**Protocol.** Cine eligibility is detected honestly (`n_frames ≥ 8`, positive
frame duration with a real time unit, valid native affine). It resolves to
`spatial_predictive` regardless, because `ObservationModel.supports_temporal is
False` and no fit-time temporal transport model exists; the reason string is
persisted in the manifest and in every unit record. An explicit
`protocol='cine_predictive'` request raises `ProtocolUnavailableError` rather
than mislabelling a spatial fit. Cine is never manufactured from ED/ES exports.

**Frame choice.** Multi-frame acquisitions use frame index 0
(`frame_selection_rule = "first_acquired_frame_index_0_image_only"`). ED/ES
would require `Info.cfg`, which is annotation-derived and forbidden here.

**Splits.** Patients whose files sit in an official `testing`/`test` folder keep
that membership. Otherwise patients are ranked by `blake2b(dataset|seed|
patient_id)` and cut at quantiles (70/15/15), which keeps tiny fixture cohorts
non-empty and fully deterministic. `split_provenance.official_test_membership`
is `False` in that case and the manifest carries
`synthetic_split_not_a_fresh_test_set`. The split is never described as
official or as a previously untouched cohort when it is not.

**Deduplication.** Within a patient, a 4-D raw acquisition wins; its single-frame
re-exports are recorded in `manifest["duplicates"]` with `duplicate_of` and are
not counted as records. With no raw cine present, ED/ES exports share one
`acquisition_id` and carry `acquisition_member_index`, so nobody can later count
them as two independent acquisitions.

**Firewall.** `firewall.py` refuses any path traversing `masks/`, `labels/`,
`gt/`, `annotations/`… any file named `Info.cfg` or `diagnosis.csv`, and any
name containing `_gt.`, `_mask`, `_label`, `_seg.`, `groundtruth`… Every read in
the package goes through `assert_image_only`. **Limitation: the check is
path-shaped.** A mask stored under an innocuous name would not be caught by the
name rule; the mitigation is that discovery only enumerates files it selected
itself and the loaders refuse paths that did not come from discovery.

## Focused checks (3, as instructed)

Command:

```
PYTHONPATH=.:src /Users/alvinluong/miniforge3/bin/python -m pytest tests/test_maskfree_data.py -q
...                                                                      [100%]
3 passed in 0.97s
```

1. `test_loads_without_masks_and_refuses_annotation_paths` — ACDC-shaped tree
   with raw 4-D cine, ED/ES re-exports, `_gt` masks and `Info.cfg`. Asserts no
   annotation path appears anywhere in the manifest, the 12 re-exports collapse
   to 6 counted records, cine is detected (6) yet the protocol still resolves to
   `spatial_predictive`, and a full unit still loads after `chmod 000` on every
   mask file and `Info.cfg`. Also loads a preprocessed ED/ES `.npy` tree and
   asserts `annotation_selected_cohort` + `stored_grid_export_only` are
   disclosed and `native_grid_export is False`.
2. `test_selection_and_verify_mutation_cannot_change_the_fitting_view` — every
   withheld voxel (select ∪ verify ∪ guard) is overwritten with `9.9e4` across
   all slices and all 10 cine frames; the fitting image, context, support,
   normalization statistics and support counts are bitwise identical, while the
   selection view *does* change (positive control). Then the study-level case:
   a study with two units has one shared `partition_id`, both source files are
   corrupted, and **both** units' fitting views are re-checked. Also asserts the
   deployment view is flagged `deployment_only`, `training_use_permitted=False`,
   `withheld_intensity_prediction_claim=False`.
3. `test_geometry_splits_and_the_freeze_gate_on_sealed_observations` — splits
   non-empty and patient-disjoint across train/dev/test, `manifest_id` stable
   across two discoveries, native affine preserved and the inverse-transform
   record round-trips `72×68 → 128×128 → 72×68` with nearest labels. The freeze
   gate is checked against four bad receipts (empty, wrong manifest id, wrong
   partition, tampered SHA-256) before one valid receipt releases a
   `ScoringView(role='verify')` that is disjoint from both fit and select
   supports. `ImageOnlyDataset.load_verification_unit` raises `PermissionError`.

### Executed CLI evidence (synthetic fixture)

```
PYTHONPATH=.:src python scripts/prepare_maskfree_data.py \
  --root <scratch>/acdc_fixture --dataset acdc --output <scratch>/out
dataset=acdc manifest_id=cfcfe41cfb2368c44d7c3ed48181dafa
protocol=spatial_predictive (6 acquisition(s) are cine-eligible, but no fit-time
  temporal transport model is implemented; resolved to the weaker spatial
  appearance experiment rather than claiming cine_predictive)
records=6 patients=6 duplicates_collapsed=12 splits={'train': 4, 'dev': 1, 'test': 1}
limitation: cine_detected_temporal_transport_unimplemented
limitation: synthetic_split_not_a_fresh_test_set
EXIT=0

python scripts/prepare_maskfree_data.py --root /nonexistent/ACDC --dataset acdc \
  --output <scratch>/out2 --allow-missing-root
NOT INSPECTED: dataset root /nonexistent/ACDC does not exist on this machine
EXIT=0          # exit 2 without --allow-missing-root
```

W1 interoperability probe (not a new test, a direct call):

```
ImageOnlyDataset(manifest, split='train', image_size=128) -> 4 units
ObservationModel().fit(hypothesis, unit.fitting)
ObservationModel().score(fitted, unit.selection)
  -> role=select count=3313 normalized_nll=1.1191 total=1.1191 available=True
```

Measured support on a 72×68 native grid, 8-px blocks, 2-px guard:
`fit=1976, select=928, verify=992, guard_dropped=1000` (fit ≈ 40% of the FOV).
The guard band is the expensive term at this block size. The constants are the
contract's; flagging the cost so it is a chosen trade-off, not a surprise.

## Limitations, honestly stated

- **No real data inspected.** `preprocessed_data/ACDC/training` and
  `preprocessed_data/mnm` are not present in this checkout, and the remote host
  was not contacted. Real-root readiness, native metadata and actual cine
  availability remain **UNKNOWN**. All evidence above is synthetic-fixture
  software evidence.
- **`cine_predictive` is unavailable**, by design, until a fit-time temporal
  transport model exists. Detection is implemented; the experiment is not run.
- **`.npy` roots are stored-grid only.** Native-grid export is `unavailable`,
  not approximated. Those cohorts also carry `annotation_selected_cohort`,
  because their ED/ES frame selection required masks.
- **Firewall is path-shaped** (see above).
- **Patient identity is parsed from file names** (`_4d`, `_sa`, `_la`, `_ED`,
  `_ES`, `_frameNN` suffixes stripped). A root with a different naming
  convention needs the rule extended; it will not silently mis-group, but it
  may treat one patient as several.
- **One unit per acquisition** (centre slice, first frame). This matches the
  contract's unit definition. Multi-slice sampling would be a contract change
  and is not made unilaterally.

## Cross-worker notes for the coordinator

1. **W7** — please confirm or amend the freeze-receipt schema above, and honour
   `inverse_transform` / `native_grid_export` when choosing NIfTI vs stored-grid
   export.
2. **W5** — `ImageOnlyDataset.fingerprint()` is the intended resume hash;
   `partition_ids()` gives the per-unit partition identity for the resume state.
3. **Pre-existing failure, not mine, W1-owned.** With `PYTHONPATH=.:src`,
   `tests/test_maskfree_observation.py::test_target_blind_fit_and_positive_score_sensitivity`
   fails: `RuntimeError: Boolean value of Tensor with more than one value is
   ambiguous` at `tests/test_maskfree_observation.py:122`, from
   `assert fitted_clean.parameters == fitted_corrupt.parameters` comparing dicts
   holding tensors. The other 3 tests in that file pass. I did not touch W1's
   files. That module also has no `sys.path` bootstrap, so it only imports when
   `PYTHONPATH` is set or another test module inserts `src` first — the repo has
   no `conftest.py` or `pyproject.toml`, and adding one is not mine to own.
