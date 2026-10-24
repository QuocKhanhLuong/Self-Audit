# CineMA teacher provenance and upstream contract

Audit date: 2026-10-24. Scope: the selected M&Ms SAX seed-0 segmentation
teacher for an opt-in native teacher-memory pilot. This is a source/provenance
audit; no weights were downloaded, no model was executed, and no patient or GPU
result was produced by this audit. The canonical semi-supervised trainer and its
label budget remain governed by `ACTIVE_PIPELINE.json` and `docs/METHOD.md`.

## Identity and integrity

| Item | Pinned value |
| --- | --- |
| Official source | `mathpluscode/CineMA` |
| GitHub revision | `c10daa1d93f0ea28d8b9ad9206b0f673d25805c1` |
| Hugging Face revision | `b1251ee50423bceeca84c080782fc3bc7756dea6` |
| Checkpoint | `finetuned/segmentation/mnms_sax/mnms_sax_0.safetensors` |
| Expected checkpoint bytes | `509604320` |
| Expected checkpoint SHA256 (LFS OID) | `ee11e5392cc89f0a03f7cd2c171e87de0771b839d414835cb59044077679acb8` |
| Config | `finetuned/segmentation/mnms_sax/config.yaml` |
| Config bytes | `441` |
| Config SHA256 | `aab63af2dc5b1592fd76faa209b37d84b846c7bcf09639a6fb8f7469a31fbea1` |
| Config Git blob OID | `50684bc09883e9a102e3402854d0631395d6deab` |

The checkpoint digest is VERIFIED as published LFS metadata, not a locally
computed weight-file digest. The config digest was computed from fetched bytes.
Verify the installed checkpoint bytes against the expected digest before use.
Git blob OIDs and Xet hashes are different identifiers and are not replacements
for the LFS SHA256. [Pinned HF file metadata](https://huggingface.co/api/models/mathpluscode/CineMA/tree/b1251ee50423bceeca84c080782fc3bc7756dea6/finetuned/segmentation/mnms_sax?expand=true)
and [sibling configuration](https://huggingface.co/mathpluscode/CineMA/raw/b1251ee50423bceeca84c080782fc3bc7756dea6/finetuned/segmentation/mnms_sax/config.yaml).

## Exact model, input, and feature tap

The sibling config specifies ConvUNetR **Base**, one SAX input channel, four
output channels, XYZ spacing `[1,1,10]` mm and XYZ patch size `[192,192,16]`.
Base means encoder embedding dimension 768, depth 12, heads 12. Encoder patch
and scale factors are `[4,4,1]` and `[2,2,1]`; convolution channels are `[64,128]`
with two blocks. Decoder channels are `[32,64,128,256,512]`, patch factor
`[2,2,1]`, scale factor `[2,2,1]`. [Base definition](https://github.com/mathpluscode/CineMA/blob/c10daa1d93f0ea28d8b9ad9206b0f673d25805c1/cinema/vit.py#L807).

Construct the segmentation model using the unmodified upstream
`cinema.segmentation.convunetr.get_model(config)`, then load the complete
safetensors state dictionary with `strict=True`. Reject missing, unexpected,
or incompatible keys. `from_finetuned(..., revision=...)` uses the same strict
PyTorch default; `from_pretrained` is a different MAE transfer path and must not
substitute for the selected segmentation checkpoint. Set `eval()`, freeze all
parameters, and disable gradient checkpointing for frozen inference.
[Loader source](https://github.com/mathpluscode/CineMA/blob/c10daa1d93f0ea28d8b9ad9206b0f673d25805c1/cinema/segmentation/convunetr.py#L488).

The tensor is `{"sax": float32[B,1,X,Y,Z]}`; one volume/frame per sample.
SimpleITK returns array axes `Z,Y,X`; upstream reverses those axes to `X,Y,Z`
before inserting channel and batch dimensions. Spatial axis 3 is the SAX slice
axis, not time. Timeframes are handled independently. Output logits have shape
`[B,4,X,Y,Z]`. [Dataset ordering](https://github.com/mathpluscode/CineMA/blob/c10daa1d93f0ea28d8b9ad9206b0f673d25805c1/cinema/segmentation/dataset.py#L117)
and [inference example](https://github.com/mathpluscode/CineMA/blob/c10daa1d93f0ea28d8b9ad9206b0f673d25805c1/cinema/examples/inference/segmentation_sax.py#L125).

Class indices are `0=BG, 1=RV, 2=MYO, 3=LV`. Original M&Ms labels 1/2/3 are
LV/MYO/RV and are remapped upstream; do not apply that original M&Ms remapping
a second time to teacher outputs. [Named labels](https://github.com/mathpluscode/CineMA/blob/c10daa1d93f0ea28d8b9ad9206b0f673d25805c1/cinema/__init__.py#L14),
[M&Ms remapping](https://github.com/mathpluscode/CineMA/blob/c10daa1d93f0ea28d8b9ad9206b0f673d25805c1/cinema/data/mnms/__init__.py#L15).

A non-mutating forward hook on `model.decoder_dict["sax"]` captures the decoder
output immediately before `model.pred_head_dict["sax"]`. It is a full-resolution
`[B,32,192,192,16]` feature tensor; the final head is a 1x1 Conv3d from 32 to 4.
The hook must return `None`, capture exactly once per patch, and be cleared or
removed reliably. Keep the normal upstream model forward as the logits path.
[Decoder/head construction](https://github.com/mathpluscode/CineMA/blob/c10daa1d93f0ea28d8b9ad9206b0f673d25805c1/cinema/segmentation/convunetr.py#L395) and
[direct decoder-to-head call](https://github.com/mathpluscode/CineMA/blob/c10daa1d93f0ea28d8b9ad9206b0f673d25805c1/cinema/segmentation/convunetr.py#L482).

Runtime acceptance requires hook-on/off native logits parity and
`pred_head_dict["sax"](captured_feature)` parity on the identical patch, device,
dtype, evaluation state, and autocast policy. Record maximum absolute difference,
relative tolerance, finiteness, feature shape and class channel count. These
checks remain NOT RUN in this source-only audit. A toy substitute cannot establish
real-checkpoint strict-load or logit parity.

## Intensity preprocessing and forbidden GT crop

The training-data preparation performs: physical resampling; XY cropping around
the ED **ground-truth LV** bounding-box center; per-frame 3D percentile clipping
at **0.95 and 99.5 percentiles**; `sitk.Normalize`; `sitk.RescaleIntensity` to
`[0,1]`; then uint8 conversion of `image * 255`. At dataset/inference time,
`ScaleIntensityd` again maps to `[0,1]`, before end-padding with zeros. This is
not a final z-score input. Percentiles are computed after the upstream LV crop,
independently for ED and ES. [M&Ms preprocessing](https://github.com/mathpluscode/CineMA/blob/c10daa1d93f0ea28d8b9ad9206b0f673d25805c1/cinema/data/mnms/preprocess.py#L95),
[clipping and scaling](https://github.com/mathpluscode/CineMA/blob/c10daa1d93f0ea28d8b9ad9206b0f673d25805c1/cinema/data/sitk.py#L246),
[uint8 storage](https://github.com/mathpluscode/CineMA/blob/c10daa1d93f0ea28d8b9ad9206b0f673d25805c1/cinema/data/sitk.py#L452),
[evaluation transforms](https://github.com/mathpluscode/CineMA/blob/c10daa1d93f0ea28d8b9ad9206b0f673d25805c1/cinema/segmentation/dataset.py#L210).

Calling upstream `mnms_preprocess` or `acdc_preprocess` on U/VAL/TEST is forbidden:
it reads GT to determine cropping geometry. Image-only raw inputs need an explicit,
versioned replacement. Full-FOV percentile clipping changes the percentile domain
relative to the historical cropped training inputs; direct float scaling also
omits historical uint8 quantization. Record those choices as pilot deviations,
not exact preprocessing parity. Do not choose normalization or crop behavior by
inspecting U/VAL/TEST masks.

## Image-only physical geometry contract

Recommended safe path: read the image and its NIfTI physical metadata; preserve
the acquisition axis order and direction; create a `[1,1,10]` mm grid using only
the image; resample scalar intensity with linear interpolation; compute the
frozen image-only intensity transform; end-pad to at least `[192,192,16]`; tile
the complete volume with fixed patches and 50% overlap. No training crop or GT
localization is required. Upstream evaluation already averages patch softmax
probabilities and takes their log. Feature overlap fusion must separately be
specified and use identical patch coordinates. [Upstream tiled inference](https://github.com/mathpluscode/CineMA/blob/c10daa1d93f0ea28d8b9ad9206b0f673d25805c1/cinema/segmentation/train.py#L148).

A geometry finding requires care: upstream `resample_spacing_3d` preserves the
direction matrix but changes origin componentwise by `(target_spacing-spacing)/2`.
For non-identity directions, that is not the physical half-voxel shift implied
by the stated edge alignment. With direction matrix `D`, a direction-aware
edge-aligned grid uses `origin_new = origin_old + D @ ((spacing_new-spacing_old)/2)`.
A center-aligned alternative is valid if explicitly frozen and recorded. Never
silently drop the direction, origin or spacing, or resize the native image to a
fixed tensor shape. This formula is DERIVED from physical image-grid semantics;
it is an intentional correction, not upstream numerical parity.
[Upstream origin handling](https://github.com/mathpluscode/CineMA/blob/c10daa1d93f0ea28d8b9ad9206b0f673d25805c1/cinema/data/sitk.py#L194) and
[SimpleITK physical-image semantics](https://simpleitk.readthedocs.io/en/master/fundamentalConcepts.html).

Store the input and resampled size, spacing, origin and direction; axis
permutations; tile offsets and weights; padding extent; and a valid-support mask.
Restore continuous teacher logits/probabilities and features onto the native
image reference grid with an explicit physical transform. Renormalize interpolated
probabilities over valid classes; never treat unmapped support as confident BG.
Apply the student's image-only crop to both student inputs and teacher memory
using the same native pixel coordinates. For label visualization only, use
nearest-neighbor label resampling. Validate non-square images, flipped/oblique
directions, anisotropic spacing, padding, and impulses or coordinate ramps;
shape agreement alone does not establish alignment.

## Training provenance and ACDC claim boundary

REPORTED: CineMA pretraining used 74,916 unlabeled UK Biobank cine CMR studies;
the paper describes separate task/dataset fine-tuning. The selected model-card
row identifies M&Ms SAX segmentation seed 0. These sources support the reported
lineage **UK Biobank pretraining -> M&Ms segmentation fine-tuning**, with no
reported ACDC training stage for this selected checkpoint. They are not a
checkpoint-specific executed training manifest. [Paper v2, sections 2.1 and 4.1](https://arxiv.org/html/2506.00679v2)
and [pinned model card](https://huggingface.co/mathpluscode/CineMA/raw/b1251ee50423bceeca84c080782fc3bc7756dea6/README.md).

VERIFIED SOURCE: the M&Ms training implementation reads `train_metadata.csv` and
`val_metadata.csv` under its M&Ms data directory. This supports the intended
training path; it does not prove which files produced released weight bytes.
The repository-wide HF `datasets: mathpluscode/ACDC` tag is not evidence that the
M&Ms checkpoint was trained on ACDC, nor is the M&Ms filename proof of patient
non-overlap. [M&Ms training loader](https://github.com/mathpluscode/CineMA/blob/c10daa1d93f0ea28d8b9ad9206b0f673d25805c1/cinema/segmentation/mnms/train.py#L26).

**ACDC patient-identity overlap is UNKNOWN.** The examined release provides no
cross-dataset patient crosswalk or checkpoint-bound dataset manifest. Different
reported cohorts are not proof of disjoint identities. This is not evidence that
ACDC overlap exists. A claim that ACDC was a clean unseen external evaluation for
this teacher remains BLOCKED until source manifests or explicit author evidence
resolve the question. Do not substitute an ACDC fine-tuned checkpoint.

This provenance limit does not block an explicitly labeled, bounded engineering
strict-load/adapter smoke with the user-selected M&Ms weights and image-only TRAIN
inputs. It does block upgrading that smoke into an unseen-ACDC efficacy claim.
Teacher semantic quality, cache alignment, runtime cost, and downstream patient
benefit each require their own recorded evidence. No archived bank trainer is
restored by this audit.
