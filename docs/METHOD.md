# Active methods and protocol boundary

The repository now has two explicit Candidate tracks. They share the typed
measurement-code idea, the K1 utility target, epoch accounting, provenance
checks and student-only deployment, but they do not share an implicit geometry
definition or a checkpoint namespace.

| Track | Config schema | Input to teacher/student | Geometry and distance | Status |
|---|---|---|---|---|
| A | `self_audit.candidate_a.v2` | `[B,3,H,W]` three-slice context | native 2D `shape`, `spacing_yx_mm`; 2D physical distance | reference implementation |
| A-3D | `self_audit.candidate_a3d.v1` | `[B,1,D,H,W]` native full-volume frame | runtime `shape_zyx`, `spacing_zyx`, `affine_zyx`; manifest stores `spacing_zyx_mm`; 3D physical distance | implementation track; efficacy not measured |

`train.py` and `evaluate.py` dispatch by the declared method/config schema. A
checkpoint from one track cannot be loaded into the other. The default recipes
use only the dense labeled TRAIN budget; image-only U records are not targets.
An explicitly selected A-3D semi-supervised profile additionally reads
image-only TRAIN U for detached EMA consistency, never for masks, SDF ground
truth or K1 utility labels. Development and test masks are opened only after the
relevant frozen prediction artifact is written and hashed, and never choose a
checkpoint or stop training.

## Shared method contract

The teacher has a shared trunk and separate region and boundary processing. Its
region branch keeps the backbone's segmentation path; its boundary branch has a
detail-preserving block and predicts segmentation plus a signed-distance field.
The teacher loss is CE plus Dice for each branch and normalized SDF MAE. A fixed
teacher snapshot proposes useful query IDs, but the value written into the code
is always predicted and quantized by the student.

The finite bank uses IDs `(type, class, location/region, scale)`. `R0`, query IDs,
quantized values, padding and image-dependent metadata all count toward the
packet budget. K1 trials use the same decoder state before and after adding one
unused feasible query, the actual quantized student value and a bit cost. STOP is
valid from step zero. The policy target is detached and renormalized on the
tried candidates. Inference selects from the feasible bank without teacher,
ground truth or utility computation.

The decoder receives only the serialized code, query coordinates and declared
public geometry. It never receives encoder features, the source image, a teacher
tensor or a hidden cache. Decoding is chunked for memory, while timing reports
the full encoder, query head, policy, packet roundtrip and mask decoder path.

## Optional A-3D semi-supervised profile

`configs/candidate_a3d/semi_supervised.json` is an opt-in A-3D joint-training
profile. It does not redesign the measurement bank, signed-distance target,
K1, packet or decoder. Labeled TRAIN frames keep the existing segmentation,
measurement and detached utility targets. An image-only TRAIN U update supplies
only a consistency target: a frozen EMA copy produces weak-view region,
boundary and soft-SDF outputs, while the online student receives an
intensity-only strong view (gamma sampled from `[0.7, 1.5]` plus Gaussian noise
with standard deviation `0.03`). Confidence gating uses threshold `0.7` and
background weight `0.1`; the consistency weight ramps linearly for 10 epochs.
No U mask is opened, and U never creates a ground-truth SDF or K1 utility
label.

The EMA copy is a student self-EMA helper present in every arm, including
`teacher_free`. It is not the dual-branch teacher and does not turn
`teacher_free` into a teacher-guided arm. The dual-branch teacher remains the
labeled-query proposal mechanism. EMA parameters are detached, updated after
the online step, and reset for each new pretrain/warm-up/joint stage; a stage is
not an optimizer/RNG resume.

Each semi-supervised update samples one labeled frame and one U frame. The
deterministic U cycle covers all U patients and their native cine frames during
an epoch; labeled frames repeat through balanced cycles. `steps_per_epoch: 0`
means the runner resolves one fixed update per labeled TRAIN patient (80 on the
prepared ACDC protocol) and reuses that resolved count for every few-shot `k`.
The original M&Ms plan records its own resolved count rather than reusing the
ACDC value.
Thus the old shorthand of 100 epochs with four updates at `k=2` is not this
profile: the current ACDC setting is 100 x 80 updates for every `k`.

## A track

A visits every annotated TRAIN frame/slice once per epoch. The three-slice input
is an explicit 2D/2.5D context; its SDF and boundary metrics remain in-plane.
The reference bank contains REGION fractions and BOUNDARY signed distances, and
the student uses the local 2D feature contract described in
[CANDIDATE_A.md](CANDIDATE_A.md). This track is retained as the software baseline
for ablations and for comparison against A-3D.

## A-3D track

A-3D reads one native full-volume frame at a time. The default implementation
does not crop or resample before creating targets; unsupported cropped inputs
must fail closed rather than treating a truncated structure as absent. Both
networks receive the same `[B,1,D,H,W]` tensor and the same native geometry.
The optional semi-supervised schedule above changes only the source of an
additional U consistency term; it does not change this native geometry or the
labeled A-3D contract.

The reference teacher backend is a local, hash-pinned MedNeXt-L 3D k3 adapter
using the verified native MedNeXt-v1 stage sequence and original region path; it
does not fall back to generic feature APIs or logits-as-features. A separate
high-detail boundary branch adds segmentation and SDF outputs. External padding
uses replicate mode with a minimum spatial size of 32 and multiple of 16, then
cropping back to the native volume. A random or missing checkpoint is recorded
as untrained and never described as a strong teacher.
RefineFormer3D Standard is an adapter slot for a separately verified local
backend; no unverified weights or performance claim is bundled.

The A-3D student uses a stride-one Conv3d encoder with global context plus a
typed local query head. REGION queries pool the declared 3D footprint; BOUNDARY
queries sample the declared physical point/neighbourhood. `local_queries=false`
is a named global-only ablation, not the reference implementation. Its decoder
still reads code and 3D geometry only. A-3D signed distances use anisotropic
runtime `spacing_zyx` in millimetres (from manifest `spacing_zyx_mm`) and full-volume masks, with explicit empty and out-of-domain
cases. The packet includes the 3D geometry fields whose changes affect decoding.

The A-3D acceptance checklist is in [CANDIDATE_A3D.md](CANDIDATE_A3D.md).
It separates implementation checks from real ACDC/M&Ms efficacy. No result in
this repository certifies SOTA, 96 Dice or novelty.

## Data and evidence

ACDC and original M&Ms preparation is documented in
[DATA_PREPARATION.md](DATA_PREPARATION.md). Native class maps, ED/ES frames,
patient splits, affine, spacing, units and source hashes are explicit. Every
report must identify track, split, aggregation (pooled and per-patient), vendor
grouping when declared, bit budget, realized bytes/query count and measured
inference time. Manufactured smoke fixtures validate software contracts only.
