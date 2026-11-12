# ResEnc-L geometry audit

This is a read-only Phase 0 audit. It uses only the JSON metadata in the locked ACDC K8 TRAIN manifest; no NIfTI/NumPy image, mask, validation/test data, download, repository edit, or training run was performed.

## Geometry result

Manifest:

`/Users/alvinluong/Self-Audit-runs/vast-a3d-semi-20261111/team-k8-ablations/manifests/k8_train.json`

Fresh SHA-256: `e524c7531079996232fe71cdfe982a46a5950d49d91a8203eee396b0e67434a3`.

The manifest has 80 TRAIN patients (8 labeled L8 and 72 unlabeled). Reading `geometry.by_patient[id]` only:

- Median TRAIN spacing in declared Z,Y,X order: **[1.0, 1.0, 1.0] mm**.
- Median native TRAIN shape in declared Z,Y,X order: **[9, 256, 216] voxels**.
- Largest native volume by voxel count: **patient057**, shape **[8, 512, 428]**, 1,753,088 voxels, spacing [1, 1, 1] mm.
- L8-only sensitivity reference: median [1, 1, 1] mm; largest labeled volume is patient026, [10, 256, 216].

The median is not a guarantee of isotropic native geometry: 28 of 80 TRAIN entries have z spacing 10 mm. The full spacing table and calculation provenance are in `geometry-audit.json`.

## Locked lineage retained

The audit preserves the earlier K8 proof: K8 manifest SHA `e524c753...`, FEWSHOT plan SHA `ad8abf69...`, seed 42, and the existing `self_audit.data.label_subset` selector reproduce the exact K5/K8/K10 sets with `K5 ⊂ K8 ⊂ K10`. Historical K5, K10, and DEV bytes were unavailable and were not rehashed.

## Pinned nnUNet rule and proposal

The primary source is nnUNet commit [`c04f7ec8`](https://github.com/MIC-DKFZ/nnUNet/commit/c04f7ec8bbffaf3ebc3b26aa2c89042e8a8e368b). Its ResEnc planner specifies encoder blocks beginning `(1, 3, 4, 6, 6, 6)`, decoder blocks `(1, 1, 1, 1, 1)`, and the six-stage ResEnc example uses features `(32, 64, 128, 256, 320, 320)` with an initial all-axis stride followed by repeated pooling. The topology helper pools only when spacing and minimum feature-map constraints permit it and pads the patch to powers-of-two divisibility.

Applying the pinned `get_pool_and_conv_props` rule to the manifest's median metadata (`spacing=[1,1,1]`, `patch_size=[9,256,216]`, minimum feature-map edge 4) gives pool counts `[1,5,5]`, divisibility `[2,32,32]`, and padded diagnostic patch `[10,256,224]`. The exact six-stage result in Z,Y,X stride vectors is:

```text
[(1,1,1), (2,2,2), (1,2,2), (1,2,2), (1,2,2), (1,2,2)]
```

The result is exact for the median native metadata diagnostic and is locked for the declared B=1 native high-side padding policy. It must not be extrapolated to other datasets or undocumented batching modes. The pretraining artifact is no-resample, so a resampling-style plan must not be silently substituted.

## Pretraining lock

The inspected external checkpoint is:

`/Users/alvinluong/Self-Audit-runs/q3d-resenc-20261112/checkpoint_final.pth`

SHA-256 `ac262d3e8c226c79f9567fddc34d38e011284730ddf2294b5c6c319ce2f22bbd`, 409,579,442 bytes, Hugging Face revision `f443c09319988e833b3d9589bf95909e4443d7b2`. The strict encoder inspection is `PASS_STRICT_ENCODER` with architecture `ResEncL`, 448 encoder keys, and no missing/unexpected keys. Its adaptation plan is `Dataset000_Pretrain`, `nnsslPlans/noresample`, Z-score normalization, and recommended pretrain/downstream patch size [192,192,192]. This locks the artifact for encoder initialization; it is not an ACDC result and no fine-tuning was run here.

The checkpoint's registry architecture uses all-axis strides `[(1,1,1),(2,2,2),(2,2,2),(2,2,2),(2,2,2),(2,2,2)]`; the native-median planner diagnostic used by the probe is `[(1,1,1),(2,2,2),(1,2,2),(1,2,2),(1,2,2),(1,2,2)]`. Encoder parameter shapes are compatible, but these are not identical architectures. The probe therefore measures a shape-compatible native geometry adaptation, not the exact pretraining forward operator.

The machine-readable record, including source URLs and all stop-gates, is [geometry-audit.json](geometry-audit.json).

## Empirical phase/ERF preflight

The external CPU probe is [erf_probe.py](erf_probe.py), receipt [erf_probe.json](erf_probe.json). It instantiated the exact six-stage stride result above, loaded the 448 encoder keys from the checkpoint, and used 16 deterministic random Z-score inputs with logical shape [33,97,97]. Because the external residual block combines ceil-style stride convolution with floor-style AvgPool skip paths, the logical input is high-side padded according to the full teacher cumulative stride [2,32,32], producing model input [34,128,128]. Padding uses the teacher's actual `replicate` mode after Z-score. The selected logical stage-3 cell is [8,6,6] with exact nominal phase [16,48,48]. It differentiated the signed mean over all stage-3 channels and computed absolute-gradient centroids.

The decision gate is absolute mean centroid offset per axis, threshold 2 voxels; the largest individual sample is reported but is not the gate. The final teacher-context result is **PASS**: signed/absolute mean offset [0.6721, 1.9470, 1.7222] voxels in Z,Y,X. Maximum individual absolute offset was 5.1300 voxels and is reported only, never used as a gate. Checkpoint pin, Torch 2.14.0, and dynamic-network-architectures 0.4.4 provenance are recorded in the JSON. The earlier full-zero-padding, stage-3-padding, and even-shape receipts are preserved as non-calibration diagnostics because their preprocessing/context/center did not match the final teacher path. This is synthetic CPU architecture evidence only; the bounded shape can truncate the full effective field and no GPU or patient-volume claim follows.
