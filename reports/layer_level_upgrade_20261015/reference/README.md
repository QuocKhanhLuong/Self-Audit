# Self-Audit: layer/function-level strong-student prototype

This package is a concrete upgrade of the previous strong-student proposal. It includes model code, mathematical proofs, tests and integration boundaries. It is **not** a trained ACDC model or a production Self-Audit patch.

## What is implemented

| Location | New function / layer | Property tested/proved |
|---|---|---|
| Input | `CenterDifferenceInput` | Starts exactly at normalized repeated-center inference; learns neighbor differences |
| Decoder/head | Four-scale residual decoder + `compose_logits` | Exact coarse/detail logit separation; bounded native residual |
| Loss | `evidence_distribution`, `soft_ce`, `set_loss` | No semantic target invented at zero support; no hidden CE-to-KL 'fix' |
| Optimizer primitive | `project_direction`, `guarded_sgd_step` | Parameter-space projection and actual finite-step guard on a fixed batch |
| Self-learning primitive | `FrozenImageEnergy`, `refine_distribution` | Detached image-derived proposals with a quantified proxy-energy descent bound |

`THEORY.md` gives proofs, a positive conditional class-correction result, counterexamples and scope. `TEST_LOG.txt`, `TEST_RESULTS.xml`, `FULL_MODEL_CHECK.json` and `PROBE_RESULTS.json` record actual checks.

## Verified scope

47 CPU tests passed. A full-size random torchvision ConvNeXt-Tiny with the new decoder ran forward/backward on odd spatial dimensions and preserved the native output shape. The instantiated model has **32,890,558 trainable parameters**, of which 27,818,592 are encoder parameters. These numbers belong to the tested random encoder configuration, not an inspected DINOv3 checkpoint. The official DINOv3 native-stage wrapper is implemented from the inspected API but no official pretrained checkpoint was loaded in this session.

No patient images, masks, real teacher features or pretrained weights were used. No ACDC training, GPU profiling, generalization evaluation, commit or remote push occurred. The full-model smoke has zero optimizer steps; operator tests include tiny manufactured parameter updates. All included code is new prototype code, not a bundled official DINOv3 source or model weight file.

## Run locally

Use a separate environment or the existing compatible research environment. The recorded environment was Python 3.13.5, torch 2.10.0+cpu and torchvision 0.25.0+cpu. Requirements are in `requirements.txt`; do not upgrade an active training environment blindly.

```bash
cd SelfAudit_LayerUpgrade_20261015
PYTHONPATH=. python -m pytest -q --junitxml=TEST_RESULTS.xml
PYTHONPATH=. python verify_full_model.py
PYTHONPATH=. python probe_report.py
```

## Construct the real pretrained student

Obtain the official model access/license and load the backbone in the existing approved local DINOv3 checkout. Record the repository commit, checkpoint digest and upstream lineage. Do not replace it with a task-finetuned medical/segmentation head while retaining a self-supervised claim.

```python
import hashlib
from pathlib import Path
import torch
from layer_upgrade import LayerGuidedStudent, OfficialConvNeXtStages

repo = Path('/actual/local/dinov3')
weights = Path('/actual/approved/dinov3_convnext_tiny_weights.pth')
# Set this from a previously reviewed provenance record, not a new tautological
# hash of whichever file happens to be present.
expected_sha256 = 'REVIEWED_64_HEX_DIGEST'
assert len(expected_sha256) == 64
h = hashlib.sha256()
with weights.open('rb') as stream:
    for chunk in iter(lambda: stream.read(1024 * 1024), b''):
        h.update(chunk)
if h.hexdigest() != expected_sha256:
    raise ValueError('unreviewed backbone checkpoint')

backbone = torch.hub.load(
    str(repo), 'dinov3_convnext_tiny', source='local', weights=str(weights)
)
student = LayerGuidedStudent(OfficialConvNeXtStages(backbone), detail_bound=1.0)
```

This loading example was API-checked against official documentation, not executed here. The wrapper intentionally uses raw native stages, not resized common-grid intermediate outputs. It excludes the classification head and does not concatenate online teacher features into the student inference path.

## Integration into Self-Audit without destroying existing runs

1. Add the model and objectives in a new opt-in `experiments/` path. Preserve `main@4d58f3a` artifacts and historical source snapshots. Do not overwrite current checkpoints or frozen banks. The old 0.35M checkpoint is NOT shape-compatible with this architecture.
2. Add an explicit architecture identifier and source/model/preprocessing hashes to each new recipe. Register a new loader, rather than letting `load_student` instantiate the historical tiny network for a new checkpoint.
3. Supply image-only normalized [0,1] adjacent slices. Historical banks store z-scored intensities: do not pass them to this input adapter without an explicitly recorded transform. All three slices need coherent normalization. No GT ROI or label-derived phase selection.
4. Retain native-to-padded-to-feature pixel transforms. `logits` is native-cropped; `logits_padded` obeys the exact block identity. Cached targets must be mapped through their valid support to the padded coarse grid. Never resize integer UNKNOWN labels as probabilities or supervise padded regions as BG. Teacher structures do not acquire new fine information when interpolated.
5. Semantic targets use original source provenance. Reconstruct or store class mass and support from the SAME positive transport. Do not infer the numerator from nearly uniform Q by dividing uncertain quantities. Apply the returned valid mask before any semantic loss. Source labels still require correctness evaluation.
6. Warm up new decoder/projection layers, then unfreeze encoder with explicit learning-rate policy. No learning rate or number of epochs has been validated for this prototype. First compare ordinary versus split decoder on the SAME strong backbone and source targets.
7. `guarded_sgd_step` updates ONLY its supplied parameter block. Start with the shared decoder and keep all other parameters fixed during this substep. Closures must run deterministically in eval mode with frozen targets. A subsequent unrestricted AdamW step on the same parameters can invalidate the guard. Restoring weights alone is insufficient when rejecting a stateful optimizer step.
8. `refine_distribution` runs with gradients detached and produces proposals. Its graph/unary/anchors remain fixed within each solve. Appearance models should be obtained from declared image/source information and frozen before comparing proposals. The example spatial graph is not anatomical truth or validated cardiac motion.
9. An image-grounded self-learning experiment can construct proposals from a weak view, freeze them, transport them to a strong view and use weighted native CE/consistency. Admission weights and handling of conflicts must be explicit; do not automatically certify all proposals because their image energy fell. With no semantic evidence, permit masked feature learning without fabricating named masks.
10. Preserve the exposed-VAL development status and untouched external evaluation. Report raw/gated scores, classwise correct coverage, PR metrics, positive RV support, guard-rejection rates, image-energy change, parameter update norms, and cost. Do not use total accepted background coverage as the principal success metric.

## Not included / not proved

No end-to-end dataset trainer, automatic high-confidence pseudo-label authorization, native-grid geometry adapter for every original artifact, EMA controller, learned correspondence, threshold selection or pretrained checkpoint is silently bundled. Runtime integration needs real artifact/geometry checks on the user's machine. There is no proof of improved Dice or domain generalization. This prototype proves selected operator contracts, not clinical safety or semantic correctness.

## File map

- `layer_upgrade/model.py`: full residual multiscale dense decoder and pretrained-backbone adapter.
- `layer_upgrade/layers.py`: identity-preserving MRI adapter and exact cell projection.
- `layer_upgrade/losses.py`: detached evidence/set/feature objectives.
- `layer_upgrade/guarded.py`: small-guard QP and verified additive update.
- `layer_upgrade/refine.py`: image-energy and detached mirror steps.
- `THEORY.md`: full derivations and failure cases.
- `ASTRA_INTEGRATION.md`: bounded integration and ablation specification.
