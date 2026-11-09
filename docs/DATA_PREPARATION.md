# ACDC and original M&Ms preparation

The preparation commands create the same hash-pinned `self_audit.data.v2`
manifest consumed by `train.py`. They do not download data or infer a split.
Each patient is assigned to exactly one source split, and a mask is read only
when the patient is selected into labeled TRAIN or when `--evaluation` explicitly
authorizes a development cache.

The same manifest supports both tracks. A retains its native 2D/2.5D cache and
in-plane distance convention. A-3D reads a complete native `[D,H,W]` frame with
the recorded `spacing_zyx_mm` and `affine_zyx`; it does not crop or resample in
the reference protocol. Its SDF is therefore computed from the full frame. A
cropped or incomplete geometry record fails closed instead of treating a
structure outside the crop as absent.

## ACDC

ACDC uses the repository ontology directly: `0=background`, `1=RV`, `2=MYO`,
`3=LV`, `255=ignore`. The locked JSON must contain patient-level `train`, `val`
and optionally `test` arrays. The official ACDC folder naming convention is
`patientXXX/patientXXX_4d.nii.gz`, `patientXXX_frameXX.nii.gz`, and
`patientXXX_frameXX_gt.nii.gz`; the preparation code checks the frame image
against the 4-D cine voxel-for-voxel and checks the mask affine.

```bash
python prepare_acdc.py --root /data/ACDC/training \
  --split /data/splits/acdc_locked.json --out /data/prepared/acdc \
  --labels 80 --seed 42 --evaluation
```

`--labels` is a dense TRAIN patient budget. It may equal the number of TRAIN
patients for a fully labeled control. The resulting `train.json` includes an
image-only `unlabeled` list for any remaining TRAIN patients; `development.json`
contains only explicit validation patients. Spatial units are validated from
each NIfTI header and converted to millimetres. If a source header has no
usable unit, pass `--spatial-unit mm|m|micron`; the override is recorded in
`PREPARATION.json`. No unit is silently assumed.

## Original M&Ms (2020), not M&Ms-2

The original 2020 M&Ms challenge provides LV, MYO and RV labels at ED and ES.
Public mirrors differ in filenames and access restrictions, so this repository
requires an index CSV rather than guessing a directory layout. The CSV has one
row per patient and these columns:

```text
patient_id,split,image,ed_frame,es_frame,axis_order,mask_axis_order,mask_ed,mask_es,vendor
```

`axis_order` must be the explicit native NIfTI order `XYZT`; `mask_axis_order`
must be `XYZ` for a 3-D ED/ES mask (or `XYZT` for an all-frame mask). The
preparer intentionally rejects arbitrary permutations rather than silently
guessing an orientation. For an all-frame mask use `mask` instead of
`mask_ed,mask_es`. Frame numbering is declared with `--frame-base 0` or
`--frame-base 1`. Spatial units are read from each NIfTI header and normalized
to millimetres (`mm`, `meter`, or `micron`). If a mirror has no valid header
unit, pass `--spatial-unit mm|m|micron`; the override is recorded in
`PREPARATION.json` and is an operator declaration, not an inferred fact. For
the original M&Ms source ontology pass the mapping explicitly:

```json
{"0":"background","1":"LV","2":"MYO","3":"RV"}
```

This source-label example follows the original challenge participant
implementation's explicit `1=LV, 2=MYO, 3=RV` convention
([MnMsCardiac README](https://github.com/MarioProjects/MnMsCardiac/blob/master/README.md)).
The challenge paper is the primary dataset description
([Campello et al., IEEE TMI 2021](https://diposit.ub.edu/server/api/core/bitstreams/f2f7dc43-becc-4554-910d-360ed7fd339c/content)).
The preparation remaps the declared source map to the native ontology
(`RV=1`, `MYO=2`, `LV=3`) and records it in `PREPARATION.json`; it never
assumes that M&Ms label 1 has the same meaning as ACDC label 1.

```bash
python prepare_mnms.py --index /data/mnms/index.csv --root /data/mnms \
  --out /data/prepared/mnms --labels 120 --seed 42 \
  --source-class-map '{"0":"background","1":"LV","2":"MYO","3":"RV"}' \
  --frame-base 0 --axis-order XYZT --mask-axis-order XYZ \
  --dataset-release original_2020 --evaluation
```

Rows without both masks are image-only TRAIN records. Their mask columns are
never opened. Test rows are never materialized by the preparation command;
create a separately authorized frozen evaluation manifest if test evaluation is
approved. The resulting geometry records preserve native affine, voxel spacing
and `T,Z,Y,X` cache order. Sheared/non-orthogonal affine, inconsistent mask
affine, non-integer labels, unknown labels, and mismatched frame geometry fail
closed.

The optional `vendor` column is copied into patient metadata for A-3D subgroup
reports. Missing values remain `unknown`; they are not inferred from filenames
and do not affect training or split assignment.

The canonical challenge page is https://www.ub.edu/mnms/ and the challenge
paper is Campello et al., *Multi-Centre, Multi-Vendor and Multi-Disease Cardiac
Segmentation: The M&Ms Challenge*, IEEE TMI 40 (2021), DOI
10.1109/TMI.2021.3090082. `--dataset-release original_2020` is a required
operator declaration and is recorded in the manifests; file contents cannot
reliably identify whether a mirror is original M&Ms or M&Ms-2. The command
therefore does not claim to detect that distinction automatically. M&Ms-2 is
not accepted unless a separately audited preparation protocol is added.
