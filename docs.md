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
edits, retry within the configured `epsilon_neutral`, and report the closest
actual DeltaDice when the tolerance cannot be met. On-policy expert transitions
are also supported. Candidates stay finite, non-negative, and normalized in
soft probability space.

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
  across the explicit transition list.
- `configs/self_audit_joint.yaml` / `training/finetune_joint.py`: Phase C
  uses low learning rates and the complete threshold-controlled flow. The
  annotation loss updates encoder/FPN/heads/expert through retained final
  annotation behavior; `lambda_audit` scales a separate local-plus-global
  audit loss whose `H`, `A_previous`, and `A_candidate` inputs are detached.
  Therefore audit gradients update only Auditor parameters. The accept/reject
  gate remains non-differentiable and `tau_accept`/`t_max` live under `audit`,
  not in the model constructor.
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
new heads. The current training entrypoints expect real data and fail clearly
when it is absent.

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
AUPRC, correlation, FIX F1, and REGRESS F1 helpers.

## Patch verification record (2026-08-14)

The previous S3R/distillation/teacher implementation was removed in the
baseline commit history. This patch did not restore or rewrite that legacy
tree; it remains recoverable with Git history and is not a dependency of the
active Self-Audit implementation.

Verified in the available Anaconda environment with PyTorch 2.6.0 and pytest
7.4.4:

```text
/Users/alvinluong/opt/anaconda3/bin/python -m pytest -q tests/test_self_audit_*.py
33 passed
```

The literal command requested from the default `python` resolved to the
miniforge environment and failed before collection with
`No module named pytest`; the same glob was then run successfully with the
alternate interpreter shown above.

The complete repository suite also passed:

```text
/Users/alvinluong/opt/anaconda3/bin/python -m pytest -q
36 passed
```

The suite includes baseline YAML construction, adjacent transition extraction,
all concrete counterfactual operations, sampled positive repair strength and
validity, hard-neutral actual DeltaDice reporting, probability normalization,
Phase-C gradient separation, dynamic-window audit conditioning, and threshold
halt behavior. Source compilation also passed:

```text
/Library/Frameworks/Python.framework/Versions/3.13/bin/python3 -m py_compile $(rg --files src scripts tests -g '*.py')
```

The repository's default Python 3.12 environment has an incomplete PyTorch
installation and no pytest; the alternate environment above was used for the
actual test runs. No real-data metric is inferred from unit tests.

The synthetic end-to-end smoke passed with random `[2,3,64,64]` input through
real `SelfAuditNet` inference and Phase-C backward:

```text
finite_inference=true
candidate_transitions=1
accepted_turns=[]
halted_turns=[0]
phase_c_transitions=1
finite_gradients=true
```

The real preprocessed ACDC path exists and its loader smoke passed for
`patient001_ED`, volume shape `(10,224,224)`, first/middle/last slices
`0/5/9`, resized image `[3,256,256]`, mask `[256,256]`, labels `[0]` on the
first slice and `[0,1,2,3]` on the middle/last slices. NPY data had no physical
spacing metadata, so explicit unit spacing `(1.0,1.0,1.0)` was reported.
M&Ms data was not present at `data/MnMs`; no M&Ms real-data smoke was claimed.

## Tests and verification

Synthetic tests cover boundary construction, both dataset contracts and M&Ms
mapping, patient isolation, encoder/FPN shapes, dynamic coordinate bounds and
backward, residual updates and shared depth, auditor detachment, transition
targets, counterfactual validity, threshold early halt, and volume
reconstruction. Real-dataset integration tests are intentionally not faked.

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

## Remaining TODOs before real ACDC training

1. Use a complete training environment for the declared Python 3.10 stack and
   confirm the real ConvNeXt-Tiny ImageNet checkpoint is available; the
   verified unit-test environment intentionally used `pretrained_encoder=false`
   to avoid network access.
2. Run the real ACDC one-patient/volume/slice smoke and confirm patient-level
   train/val/test manifests. The repository currently has preprocessed ACDC
   files, but the checked-in manifest still requires validation for the
   intended release and a labeled ACDC test split is required for three-way
   reporting.
3. Run preprocessing/data integration checks on real NIfTI orientation and
   spacing metadata, then confirm patient counts and no overlap.
4. Run Phase A before Phase B/Phase C and calibrate `tau_accept` on validation
   transitions. Report harmful acceptance, beneficial rejection, net Dice,
   and mean accepted turns.
5. M&Ms evaluation remains blocked until the external M&Ms root is present and
   its raw label order is verified with `inspect_mnms_mask.py`.
