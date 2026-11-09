# Candidate A: implementation contract

This is the active training method. See [RUNBOOK.md](RUNBOOK.md) for commands,
[DATA_PREPARATION.md](DATA_PREPARATION.md) for ACDC/original M&Ms, and
[EXPERIMENTS.md](EXPERIMENTS.md) for component ablations.

## Tensors and geometry

- Cached cine: `[T,Z,Y,X]`, normalized by per-cine 0.5/99.5 percentiles.
- Teacher/student input: identical `[1,3,H,W]`, three neighboring slices with
  replicated end slices. Target: central `[1,H,W]` dense mask.
- Ontology: BG=0, RV=1, MYO=2, LV=3. Source M&Ms semantic IDs are remapped explicitly.
- Teacher outputs: region and boundary logits `[1,4,H,W]`, normalized SDF
  `[1,4,H,W]`. Branch processing is separate; the boundary path retains native detail.
- Student predicted measurement values: `[1,Q]`. The reference rollout supports
  one image at a time; full-volume evaluation loops over frames/slices.

Native in-plane spacing and orthogonal affine are required. No resize, ROI from
GT, or inferred isotropic spacing is used. Horizontal flip applies to both image
and label; lengths remain unchanged. SDFs are **2D in-plane** physical distances
to observed pixel-cell interface segments, with half-pixel edges and anisotropic
spacing. They are negative inside, positive outside, normalized by the fixed
`radius_mm`, then clipped to `[-1,1]`. No FOV edge is invented as an anatomical
boundary. Absent classes give REGION=0 and BOUNDARY=+1; full-FOV classes give
BOUNDARY=-1. Measurement targets require dense labels and reject ignore=255.

## Bank, code and decoder

`BankConfig` defines a reproducible finite bank by type, foreground class,
normalized grid center and scale. REGION averages class occupancy in a fixed
FOV fraction. BOUNDARY samples a fixed grid location. IDs and descriptors are
independent of image/encoder/labels. The bank fingerprint pins this protocol.

The student predicts raw bounded values and its policy chooses unused affordable
queries or STOP. Fixed `baseline_ids` form R0 and consume budget. The code contains
only **student** quantized values and IDs. Teacher or GT values never replace them.

The packet uses a 57-byte header, 8-byte checksum, 32-bit query ID and
`ceil(value_bits/8)` bytes for each value. Its complete cost is
`8 * (65 + N * (4 + ceil(value_bits/8)))` bits, including shape, float32 spacing,
count, bank fingerprint, precision, protocol metadata and padding. This format is
byte padded, not entropy compression. Four-bit and eight-bit levels each occupy
one value byte. `incremental_cost` measures actual packet growth.

The query decoder reads IDs, dequantized values, native shape, spacing and pixel
coordinates only. It has no encoder features, skip connection or raw-image bypass.
A learned null token supports an explicitly empty R0. Pixel queries are decoded
in chunks. Tokens are canonicalized so packet serialization preserves output.

## K1 and gradient flow

Teacher REGION probabilities and boundary SDF provide per-type normalized
teacher/student discrepancies. Discrepancy ranks trial proposals only. Each enabled
teacher branch has a separate `teacher_topk` quota; branch, student and random
streams are interleaved and deduplicated before adding STOP. The reference limit
fits all quotas plus STOP. For each feasible
trial, the same decoder/state evaluates current code and code plus one quantized
student value. Utility is TRAIN CE+Dice improvement minus `cost_lambda * added_bits`;
STOP utility is exactly zero. Targets are detached and policy CE is renormalized
on the tested candidates. Actual rollout actions always use the full feasible
bank plus STOP. Inference never computes labeled utilities or needs the teacher.

Student loss is CE + present-foreground Dice plus independently weighted REGION
MAE, BOUNDARY MAE/2 and policy CE. Uniform quantization uses STE: forward values
are exact finite dequantized levels; backward reaches the value head/encoder.
Discrete IDs receive no segmentation gradient. Policy CE learns the selection
policy. Ground-truth measurement losses supervise all configured measurements,
not encoder feature matching. No student loss updates teacher parameters.

Teacher loss sums CE+Dice on both branches and normalized SDF MAE. Teacher stages:
pretrain/load, warm-up with shared encoder frozen, then alternating teacher/student
updates. One detached teacher snapshot is fixed for each student rollout. A frozen
teacher arm skips teacher updates. Backbone pretraining is recorded separately
from cardiac segmentation pretraining; neither implies measured SOTA.

## Epochs, provenance and measurements

An epoch visits every annotated TRAIN frame/slice once in deterministic shuffled
order. Data/augmentation RNG is separate from action sampling. Shared parameter
initialization is preserved across bank-size ablations where shapes agree.
Training uses only the declared dense TRAIN budget; U images are currently unused.
The fully labeled setting permits an empty U list. No validation/test labels feed
training targets, policy utilities, label subset selection or stopping rules.

`metrics.csv`/`metrics.jsonl` log each epoch/split/branch/class with Dice, F1,
precision, recall, support, TP/FP/FN and absent-class status. TRAIN rows are online
predictions, not a fixed final-epoch model evaluation. Development predictions are
frozen and hashed before references are opened. Teacher-only stages evaluate the
teacher branches; joint stages evaluate student-only predictions. Null represents
undefined metrics. Per-class binary Dice and F1 are mathematically identical.
The main metric columns use pooled voxel counts; validation's separate
`macro_patient_*` fields average patient scores without mixing them with pooled
TP/FP/FN. Each patient's frame scores are averaged first.

Checkpoint source and payload hashes are verified. Teacher stage transfer requires
the same exact TRAIN manifest and budget. Student exports contain only student
weights/configuration and have no teacher-weight dependency. Optimizer/RNG resume
is not implemented. Final-epoch selection is fixed; development logs do not select
a checkpoint. Boundary evaluation is explicitly 2D and reports empty-class cases.
If a reference slice contains ignore pixels (255), its boundary metrics are null
with status `ignored_reference_pixels`; voxel metrics still exclude those pixels.
Dense TRAIN masks must contain only 0..3 and are checked before any update.
Inference timing includes the reference packet roundtrip verification path; it is
measured, not inferred from parameter count.

The legacy native-v2 and initial Candidate A step-based trees are recoverable via
[MIGRATION.md](MIGRATION.md). Their historical validation receipts do not certify
this revision. Current delivery tests/receipts are linked from the cleanup report.
No patient/GPU result, 96 Dice or novelty certification is claimed.
