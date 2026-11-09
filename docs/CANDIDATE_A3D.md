# Candidate A-3D

This is the 3D track of Candidate A. It keeps the typed measurement code, K1
selection and code-only decoder, but changes the image context and distance
protocol. A-3D is an implementation track, not an efficacy claim.

## Tensor and geometry contract

The prepared cache stores an image as `[T,D,H,W]` and a labeled mask as
`[D,H,W]`. One update loads one annotated frame and adds batch/channel axes:

```text
x       [1, 1, D, H, W]  float32 normalized native frame
y       [1, D, H, W]     int64 labels BG=0, RV=1, MYO=2, LV=3
geometry {
  shape:           (D,H,W)
  spacing_zyx:     (dz,dy,dx)  # millimetres; manifest field is spacing_zyx_mm
  affine_zyx:      4x4 native affine in millimetres
  context:         "full_volume"
}
```

The teacher and student receive the same `x` and geometry. A-3D uses native
full-volume frames for its reference protocol: no crop, resampling, data-level
padding or patch-only SDF target. Internal teacher padding is declared below.
If the source cannot provide this geometry,
preparation fails closed. Any future patch protocol must be a separately named
experiment with an explicit out-of-domain target rule.

The 3D bank is finite and reproducible from config. A REGION query identifies a
class, normalized 3D region and scale and returns the class fraction in that
region. A BOUNDARY query identifies a class, normalized physical point and scale
and returns the clipped signed distance in millimetres divided by fixed `r`.
Distance is negative inside the class and positive outside. An absent class has
REGION target `0` and BOUNDARY target `+1`; a query outside the declared full
volume is invalid and is never silently clamped. Anisotropic spacing is used by
the continuous voxel-face distance convention: only interfaces between two
voxels inside the declared full volume are anatomical surfaces. The outer FOV
is not treated as background, so a class touching the FOV does not acquire an
artificial boundary. A full-volume class with no interface has target `-1` and
status `full_occupied_no_boundary`; an absent class has target `+1` and status
`absent`. This is intentionally distinct from a voxel-centre EDT protocol.

The packet records IDs, finite quantized levels, padding and all geometry fields
used by decoding. Its cost is measured from serialized bytes, including `R0`
and metadata; a float tensor without a finite value budget is not a code. In the
current codec the fixed header is 140 bytes (including the `full_volume`
context marker), the checksum is 8 bytes, and each
query contributes 4 ID bytes plus `ceil(value_bits/8)` level bytes:
`bits = 8 * (148 + N * (4 + ceil(value_bits/8)))`. The header stores
`shape`, runtime `spacing_zyx` and the 4x4 affine as float32 values. Those fields
are included at fixed width: their numeric values do not change packet length,
but serialization and decode validate the values that the decoder actually uses.

## Teacher

The reference backend is a local hash-pinned MedNeXt-L 3D k3 adapter. When the
official local factory is used, it must implement
`nnunet_mednext.create_mednext_v1(num_input_channels=1, num_classes=4,
model_id="L", kernel_size=3, deep_supervision=False)`. The current adapter
accepts only the verified native MedNeXt-v1 stage sequence and refuses generic
`forward_features`/`encode` fallbacks or segmentation logits as boundary
features. The backbone's original region segmentation path is retained. A separate
high-detail boundary path receives shared trunk features and outputs:

```text
region_logits   [B,4,D,H,W]
boundary_logits [B,4,D,H,W]
sdf             [B,4,D,H,W]
```

Teacher input padding is `replicate`, with every spatial axis at least 32 and a
multiple of 16; logits are cropped back to the native volume. This is the
`external_minimum_bottleneck_2x2x2` policy, not a hidden data crop. Teacher loss
is CE+Dice for both segmentation outputs plus normalized SDF MAE.
Pretraining, boundary warm-up and alternating joint updates are separate stages.
Within a student rollout the teacher output is one detached snapshot. Frozen
teacher is a required ablation. A missing or random local checkpoint is recorded
as untrained and cannot be called a strong or pretrained teacher. Run the
optional adapter contract check against an already available clean local source:

```bash
python scripts/check_mednext_adapter.py \
  --source-repo /path/to/MedNeXt \
  --out /new/mednext_cpu_verification.json
```

This check uses random weights and a CPU backward pass; it does not establish
segmentation quality. RefineFormer3D Standard is only an adapter slot;
this repository contains no verified adapter, weights or result for it.

## Student and decoder

The reference student is a stride-one Conv3d encoder with a global context token
and a local typed query head. A REGION head pools the exact declared 3D query
footprint; a BOUNDARY head samples the declared point/neighbourhood. Query
embeddings encode type, class, coordinates and scale. `local_queries=false` is a
global-only ablation and is not equivalent to the reference representation.

The value head predicts all bank values, while the policy sees feature context,
the current code and the remaining budget. K1 interleaves separate REGION and
BOUNDARY teacher proposal quotas with student proposals, random exploration and
STOP. Trials add the quantized student value using the same decoder state and
charge incremental packet bytes. The policy target is detached and normalized
over the tried set. Inference uses the full feasible bank, never teacher, GT or
utility labels.

The decoder accepts only `(query IDs, quantized values, query coordinates,
geometry)`. It does not accept `F_loc`, global context, an image, teacher output
or a skip connection. Full-volume output is decoded in chunks to control memory.
The measured `rollout_ms` includes encoder, value head, policy and full-volume
decoding together. Packet serialization, packet roundtrip and decoding the
roundtripped packet are timed separately. `total_ms` sums those measured phases;
it excludes checkpoint loading and file I/O. Per-component encoder/value-head/
policy timing is not currently instrumented.

## Train, evaluate and export

Use the same preparation commands as A, then choose the A-3D recipes:

Start from `configs/backbone3d.example.json`, replace the source path with an
already available clean MedNeXt checkout at the pinned commit, and add a local
weights path plus SHA-256 only when verified weights are actually present.
`weights: null` is an explicitly untrained pretraining initialization, not a
pretrained-teacher claim.

```bash
python train.py --config configs/candidate_a3d/teacher_pretrain.json \
  --data /prepared/acdc/train.json --data-sha256 "$TRAIN_SHA" \
  --validation-data /prepared/acdc/development.json \
  --validation-data-sha256 "$DEV_SHA" --model-config /private/mednext3d.json \
  --out /runs/a3d-pretrain --device cuda --allow-full-run

python train.py --config configs/candidate_a3d/teacher_warmup.json \
  --data /prepared/acdc/train.json --data-sha256 "$TRAIN_SHA" \
  --validation-data /prepared/acdc/development.json \
  --validation-data-sha256 "$DEV_SHA" --teacher-checkpoint /runs/a3d-pretrain/final \
  --teacher-checkpoint-sha256 "$PRETRAIN_SHA" --out /runs/a3d-warmup \
  --device cuda --allow-full-run

python train.py --config configs/candidate_a3d/full.json \
  --data /prepared/acdc/train.json --data-sha256 "$TRAIN_SHA" \
  --validation-data /prepared/acdc/development.json \
  --validation-data-sha256 "$DEV_SHA" --teacher-checkpoint /runs/a3d-warmup/final \
  --teacher-checkpoint-sha256 "$WARMUP_SHA" --out /runs/a3d-joint \
  --device cuda --allow-full-run

python scripts/export_candidate_a.py \
  --checkpoint /runs/a3d-joint/final --checkpoint-sha256 "$JOINT_SHA" \
  --out /exports/a3d-student
python scripts/candidate_a3d_infer.py --checkpoint /exports/a3d-student \
  --checkpoint-sha256 "$EXPORT_SHA" --image /input/frame.npy \
  --geometry-json /input/geometry.json --out /predictions/a3d-example \
  --device cuda
python evaluate.py --checkpoint /exports/a3d-student \
  --checkpoint-sha256 "$EXPORT_SHA" --data /prepared/acdc/development.json \
  --data-sha256 "$DEV_SHA" --out /runs/a3d-evaluation --device cuda
```

The A-3D evaluator freezes and hashes predictions before opening reference
masks. It reports pooled and per-patient per-class Dice/F1/precision/recall,
3D surface metrics in millimetres, actual code bytes/query counts and measured
timing. If a manifest declares vendor, the same scores are shown by vendor;
missing vendor values are not invented.

Evaluation uses a discrete voxel-surface convention: 6-connected binary
erosion with outside-FOV background, followed by anisotropic voxel-centre EDT
in millimetres. Foreground touching the FOV therefore contributes a surface.
`surface_dice` uses a fixed 1 mm tolerance and counts surface voxels equally
(it is not area-weighted); ASSD and HD95 use the concatenated bidirectional
surface distances. This evaluation protocol differs from the internal
continuous-face SDF used by teacher/query targets above. A full foreground
volume has an evaluation surface even though its target status is
`full_occupied_no_boundary`. These quantities must not be reported as the
same boundary definition. Ignore-labelled reference volumes skip boundary
metrics; absent-class cases carry explicit statuses.

The independent implementation review is recorded in
[README.md](../reports/candidate-a3d-review-20261110/README.md) and
[REVIEW.md](../reports/candidate-a3d-review-20261110/REVIEW.md).

## Review checklist and evidence boundary

| ID | Required contract | Current evidence expectation |
|---|---|---|
| A3D-01 | True `[B,1,D,H,W]` track | shape tests and data-flow log; not a renamed Conv2d |
| A3D-02 | Original region path plus separate boundary path | teacher forward/state tests |
| A3D-03 | Local ROI/point queries | locality tests under changed local voxels |
| A3D-04 | 3D mm targets | cube, absent class, anisotropic spacing and full-volume tests |
| A3D-05 | Student values and finite bytes | packet roundtrip and cost tests |
| A3D-06 | Code-only decoder | fixed-code feature-cache deletion test |
| A3D-07 | Same-state K1 utility | STOP, cost, detach and candidate-renormalization tests |
| A3D-08 | No label/teacher gradient leak | backward and split-provenance tests |
| A3D-09 | Student-only deployment | export/load/infer without teacher or GT |
| A3D-10 | Fair teacher comparisons | matched config/seed/budget table |
| A3D-11 | Full timing and bytes | phase timing and serialized packet receipt |
| A3D-12 | Bounded claims | separate fixtures, checkpoints and patient results |

At the current software revision, the synthetic contract receipt records 117
passing tests and 88 synthetic matrix jobs. These checks cover shapes,
geometry, serialization, gradients, export and pipeline lifecycle only. They do
not establish teacher quality or segmentation efficacy. Locked ACDC/M&Ms patient
runs, verified pretrained MedNeXt weights, CUDA profiling, SOTA/96 Dice and
novelty evidence remain `NOT_RUN`/`UNKNOWN`; the code does not report them as
results.
