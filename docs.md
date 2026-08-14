# Self-Audit living technical specification

This document specifies the locked Self-Audit baseline currently implemented
in this repository. The former S3R, distillation, and teacher namespaces and
their related tracked documentation have been removed from the active tree.

## Locked data and prediction contract

The current experiment contract is:

```text
ACDC = primary development dataset
M&Ms = external/domain-shift dataset

GT = volumetric
model input = 2.5D
prediction unit = 2D center slice
final annotation = stacked 3D volume
self-audit v1 = slice-level transition audit
refinement loop = threshold-controlled
T_max = hard cap only
```

ACDC and M&Ms are not merged for the first baseline. The normal domain-shift
experiment trains on ACDC only and evaluates on M&Ms only. Both adapters expose
the same sample dictionary:

```python
{
    "image": Tensor[B? , 3, H, W],  # one sample is [3,H,W]: z-1, z, z+1
    "mask": Tensor[H, W],           # GT for the center slice z
    "case_id": str,
    "patient_id": str,
    "slice_idx": int,
    "num_slices": int,
    "spacing": tuple[float, ...],  # physical spacing, or explicit unit fallback
    "spacing_known": bool,         # false means metrics are pixel-space only
    "spacing_units": "mm" | "pixel",
}
```

The semantic class order is fixed for ACDC-compatible experiments:

```text
0 Background
1 RV
2 MYO
3 LV
```

M&Ms uses an explicit mapping layer. The documented default mapping for the
common M&Ms raw order `(background, LV, MYO, RV)` is:

```python
{0: 0, 1: 3, 2: 2, 3: 1}
```

Callers can and should provide a release-specific mapping when the raw label
contract differs. Unknown raw labels fail loudly; they are never silently
treated as ACDC ids.

For every volume, the common pipeline is:

```text
raw NIfTI/NPY volume
→ canonical orientation / [Z,H,W] shape normalization where needed
→ volume-wise percentile clipping (0.5, 99.5 by default)
→ volume-wise z-score normalization
→ in-plane H/W resize to 256x256 by default
→ neighboring-slice construction [z-1,z,z+1]
→ center-slice mask
```

Through-plane spacing is preserved. No isotropic pseudo-3D resampling is used.
At the boundaries, `z=0` produces `[z0,z0,z1]` and the last slice produces
`[z-1,z,z]`. Geometric augmentation is applied to all three image channels
and the center mask through the same flip/rotation operation. Split utilities
validate patient identity and reject a patient appearing in more than one
split.

## Implemented package

The clean namespace is `src/self_audit/`. Legacy S3R, distillation, teacher,
and shared legacy source namespaces were removed rather than mixed into the
baseline.

### Data

- `self_audit.data.common`: volume loading, optional NIfTI support, orientation
  and depth-first conversion, clipping/z-score, in-plane resize, 2.5-D
  construction, patient splits, and the shared `VolumeSliceDataset` contract.
- `self_audit.data.acdc`: paired current `.npy` layout and raw ACDC NIfTI
  discovery, ACDC identity mapping, and `ACDCDataset`.
- `self_audit.data.mnms`: M&Ms discovery, `MNMSClassMapping`, explicit raw-to-
  ACDC mapping, and `MNMSDataset`.
- `self_audit.data.transforms`: geometry-preserving flips and 90-degree
  rotations.

Absent roots, missing pairs, shape mismatches, unknown classes, and invalid
split manifests raise explicit errors. No synthetic fixture is returned by a
dataset loader.

### Annotation model

`self_audit.models.self_audit_net.SelfAuditNet` implements:

```text
[B,3,H,W]
  → ConvNeXt-Tiny multi-scale features
  → lightweight FPN
  → H_shared [B,96,H/4,W/4] (96 is the baseline setting)
  → initial head A0_logits [B,4,H,W]
  → one shared AnnotationExpert over recurrent turns
  → final soft annotation logits [B,4,H,W]
```

`ConvNeXtTinyEncoder` calls `timm` with `convnext_tiny`, `in_chans=3`, and
ImageNet pretrained weights when `pretrained_encoder: true`. The constructor
also has an explicitly non-pretrained dependency-light fallback for synthetic
tests when timm or checkpoints are unavailable; this fallback is not a claim
of ImageNet pretraining.

The `AnnotationExpert` receives `H`, the current soft logits `A_t`, entropy,
previous local audit evidence, and turn/iteration embeddings. It uses the same
weights on every turn and sets:

```text
depth = min(turn_index + 1, 3)
A_candidate = A_t + sigmoid(update_gate) * delta_logits
```

It never converts the recurrent state to a hard argmax mask.

### Dynamic-window operator

`self_audit.models.dynamic_window.DynamicWindowAttention` is the core expert
operator. For each query it predicts bounded:

```text
center displacement dx,dy
anisotropic radius rx,ry
orientation theta
K residual offsets (K=8 by default)
```

The canonical support is transformed by center/scale/rotation, residual offsets
are added, coordinates are clamped to `[-1,1]`, and `grid_sample` obtains the
sparse support values. Ordinary fixed square windows, Swin attention, custom
CUDA, routers, MoE, and true-3D attention are not used. The previous audit map
is projected into the generator state; no rule says that uncertainty must
produce a larger window.

### Counterfactual auditor

`self_audit.models.auditor.CounterfactualAuditor` sees a transition rather
than an isolated segmentation score:

```text
stopgrad(H), P_previous, P_candidate, delta_P,
entropy_previous, entropy_candidate
```

It contains one small projection and approximately three residual blocks, with
no second image encoder. Its outputs are local `[B,3,H,W]` logits for
`FIX/UNCHANGED/REGRESS` and global `[B,1]` signed `DeltaQ`.

`self_audit.audit.targets` uses GT only while training:

```text
previous wrong → candidate correct = FIX
previous correct → candidate wrong = REGRESS
otherwise = UNCHANGED
```

The global target is candidate multiclass Dice minus previous multiclass Dice.
The audit loss includes local cross-entropy and a signed pairwise ranking term;
it is not an absolute-confidence loss.

`self_audit.audit.counterfactual.CounterfactualGenerator` starts from real
model probabilities. Its default sampling mix is 40% positive local repair,
40% controlled negative perturbation, and 20% hard/neutral mixed transition.
Positive repairs sample a strength uniformly in 30–60% and move only a
connected partial error region toward GT; `A_positive` is never replaced by
GT. Negative operations have separate localized implementations for erosion,
dilation, boundary displacement, hole insertion, false island, component
deletion, and semantic class swap. Each sample exposes `valid`, `edit_mask`,
strengths, source/target classes, and measured GT-derived DeltaDice metadata.
Hard/neutral samples combine spatially separate local repair and regression
edits, retry within the configured `epsilon_neutral`, and report
`neutral_satisfied`, `retry_count`, and the closest actual DeltaDice when the
tolerance cannot be met. Requested operation names are never used as training
labels: Phase B/C targets are always measured from the GT-derived transition.
On-policy expert transitions are also supported. Candidates stay finite,
non-negative, and normalized in soft probability space.

`self_audit.audit.gate.ThresholdGate` applies the non-differentiable global
decision:

```python
if audit.delta_q > tau_accept:
    state = candidate
else:
    state = previous
    halt = True
```

The baseline is `tau_accept=0` and `T_max=3`. The cap protects against runaway
inference; the self-audit path can stop after zero or one accepted turns and is
not required to execute three turns. Pixel-wise selective blending is not in
v1.

### Training phases and configurations

- `configs/self_audit_annotation.yaml` / `training/train_annotation.py`:
  Phase A trains encoder, FPN, initial head, and shared expert with Dice+CE on
  the initial and intermediate soft states. It does not run audit decisions.
- `configs/self_audit_auditor.yaml` / `training/train_auditor.py`: Phase B
  freezes annotation parameters and trains the auditor on every adjacent
  on-policy transition `A0→A1`, `A1→A2`, ..., `A_(T-1)→A_T`, plus one
  synthetic transition around every `A_t`. The total audit loss is averaged
  across the explicit transition list. Every inference turn carries a
  batch-aligned `transition_active_masks[t]`, so halted rows are excluded from
  later audit losses and cannot contaminate mixed-batch training.
- `configs/self_audit_joint.yaml` / `training/finetune_joint.py`: Phase C
  uses low learning rates and the complete threshold-controlled flow. The
  annotation loss updates encoder/FPN/heads/expert through retained final
  annotation behavior; `lambda_audit` scales a separate local-plus-global
  audit loss whose `H`, `A_previous`, and `A_candidate` inputs are detached.
  Therefore audit gradients update only Auditor parameters. Per-sample
  `accepted_count`, `halt_turn`, `num_attempted_turns`, and `final_active` are
  exposed for evaluation. The accept/reject gate remains non-differentiable and
  `tau_accept`/`t_max` live under `audit`, not in the model constructor.
- `configs/self_audit_acdc_to_mnms.yaml`: ACDC-only training and M&Ms external
  domain-shift evaluation; it does not merge datasets.

`training._utils.build_model_from_config()` validates the supported model
keys explicitly. `image_size`, batch size, epochs, and data roots remain
data/training settings and are never forwarded to `SelfAuditNet`; unknown
model keys fail loudly instead of being silently ignored. The M&Ms default
mapping `{0: 0, 1: 3, 2: 2, 3: 1}` assumes raw order Background, LV, MYO, RV.
Before evaluating a release, inspect one real mask with:

```bash
python scripts/inspect_mnms_mask.py --mask /path/to/mnms-mask.nii.gz
```

The command prints unique raw labels, the configured mapping, and mapped
labels. Unknown labels remain errors.

Pretrained encoder parameters are placed in a lower-LR optimizer group than
new heads. Phase A/B/C entrypoints now include validation loops, `last.pt` and
metric-selected `best.pt` checkpoints, strict compatible resume loading,
warmup-cosine AdamW scheduling, safe CPU/CUDA AMP, gradient accumulation and
non-finite loss/gradient guards. DataLoader worker-only options are passed only
when workers are enabled. `stage_weights` defaults to `[0.5, 0.7, 0.8, 1.0]`
for `[A0,A1,A2,A3]`; a matching YAML list is honored, otherwise the documented
prefix plus `1.0` for additional states is used.

`scripts/cache_validation_transitions.py` and
`scripts/calibrate_threshold.py` provide validation-only DeltaQ/DeltaDice
caching and threshold sweeps. Calibration reports final Dice, net gain,
harmful acceptance, beneficial rejection, mean attempted/accepted turns, and
acceptance rate without using test data. The required preflight command is:

```bash
python scripts/self_audit_preflight.py --config configs/self_audit_annotation.yaml
```

It validates the runtime, split integrity, one real DataLoader batch, model,
loss/backward, finite gradients, and checkpoint-directory writability without
starting training. `--max_steps` and `--max_val_batches` are available on the
Phase A/B/C entrypoints for bounded real-data smoke runs.

### Evaluation

`self_audit.evaluation.volume_inference` constructs 2.5-D inputs, runs slice
inference, and reconstructs `[Z,H,W]` labels. Deployable modes are:

```text
initial_only
always_accept_refinement
self_audit
```

`oracle_accept` exists only as an explicit analysis mode on `SelfAuditNet.infer`
when a caller supplies GT; `infer_patient_volume` rejects it so GT cannot enter
the deployable inference path. `evaluation.metrics` reports Dice, HD95, ASSD,
precision, recall, per-class values, and dependency-free transition AUROC,
AUPRC, correlation, FIX F1, and REGRESS F1 helpers. NIfTI input is
canonicalized before deliberate `[Z,H,W]` conversion; preprocessed NPY uses the
configured `depth_axis` (the checked-in ACDC convention is `[H,W,Z]`, hence
`depth_axis: 2`). Unknown
spacing is marked `spacing_known=false` and HD95/ASSD are labeled pixel-space;
physical metrics require real spacing metadata.

## Patch verification record (2026-08-14)

The previous S3R/distillation/teacher implementation was removed in the
baseline commit history. This hardening patch did not restore or rewrite that
legacy tree; it remains recoverable with Git history and is not a dependency of
the active Self-Audit implementation.

Verified with PyTorch 2.6.0 and pytest 7.4.4 in the available Anaconda
environment:

```text
/Users/alvinluong/opt/anaconda3/bin/python -m pytest -q tests/test_self_audit_*.py
53 passed
/Users/alvinluong/opt/anaconda3/bin/python -m pytest -q
56 passed
/Users/alvinluong/opt/anaconda3/bin/python -m py_compile $(find src scripts tests -name "*.py")
passed
```

The default `python` on this machine is a separate Miniforge environment
without pytest, so the literal pytest command fails before collection with
`No module named pytest`; the alternate interpreter above is the verified test
environment. The focused suite covers all four baseline YAML constructors,
adjacent transitions, mixed-batch active masks, all concrete counterfactual
operations, sampled positive repair strength, hard-neutral metadata and actual
DeltaDice, normalization, checkpoint compatibility, gradient separation on a
real SelfAuditNet, dynamic-window conditioning, and threshold halting.

The real ACDC preflight passed without downloading pretrained weights:

```text
python=3.12.7 pytorch=2.6.0 cuda_available=False timm_available=True
split_integrity=ok batch_image_shape=(1,3,256,256) batch_mask_shape=(1,256,256)
finite_gradients=True checkpoint_dir_writable=weights/self_audit
PREFLIGHT_OK
```

The validated split report is `train: 160 cases / 80 patients / 1526 slices`,
`val: 40 cases / 20 patients / 376 slices`, with no patient overlap. The
checked-in manifest has no separate test split, so `test_available=false` is
reported instead of inventing one.

Bounded real ACDC training smoke results were also verified: Phase A completed
two optimizer steps on CPU at `64x64` with two validation batches; Phase B and
Phase C each completed one optimizer step at `32x32` with one validation batch.
All three printed finite losses, validation metrics, and checkpoint paths.
No long training job was launched.

The ACDC loader smoke covers one patient/volume and first/middle/last slices:
`patient001_ED`, volume `(10,224,224)`, slices `0/5/9`, resized image
`[3,256,256]`, mask `[256,256]`, with labels `[0]` on the first slice and
`[0,1,2,3]` on the middle/last slices. The checked-in NPY data has no physical
spacing metadata, so `spacing_known=false`, explicit unit spacing is used only
for pixel-space metrics, and no physical HD95/ASSD result is claimed. M&Ms is
not present at `data/MnMs`; no real M&Ms smoke was claimed.

## Tests and verification

Synthetic tests cover boundary construction, both dataset contracts and M&Ms
mapping, patient isolation, encoder/FPN shapes, dynamic coordinate bounds and
backward, residual updates and shared depth, auditor detachment, transition
targets, counterfactual validity, threshold early halt, volume reconstruction,
per-sample mode bookkeeping, checkpoint/resume helpers, numerical edge cases,
and AMP-compatible metrics. Real-dataset integration tests are intentionally
not faked.

`.github/workflows/self-audit-ci.yml` runs the focused Self-Audit tests and
source compilation on CPU with no dataset or pretrained-weight download.

Recommended commands in the repository environment are:

```bash
python -m pytest -q tests/test_self_audit_*.py
python -m pytest -q
```

The current machine's default Python has a broken/incomplete PyTorch install
and no pytest; an alternate local environment contains PyTorch but also lacks
pytest. A source compilation check is still available:

```bash
python -m py_compile $(rg --files src scripts tests -g '*.py')
```

## Remaining engineering blockers before real ACDC training

1. The default shell Python is incomplete; use a complete environment (the
   verified Anaconda interpreter or the CI environment) and make the real
   ConvNeXt-Tiny ImageNet weights available before production Phase A.
2. Run the full-resolution GPU job only after choosing a safe batch size from
   `scripts/self_audit_memory_smoke.py`; no long training has been started.
3. The checked-in ACDC manifest is train/validation-only at the volume level;
   keep test reporting separate until a labeled test manifest is supplied.
4. Preprocessed NPY spacing is explicitly unknown, so publishable physical
   HD95/ASSD remains blocked until real spacing metadata is available.
5. M&Ms remains external-domain evaluation only and is blocked until
   `data/MnMs` exists and one raw mask is verified with
   `scripts/inspect_mnms_mask.py`.

The research architecture and its prohibitions remain unchanged.
