# CUTS ACDC original-style protocol

This protocol ports the upstream CUTS data path to ACDC.  It is not the
patient-disjoint Self-Audit cardiac benchmark and must be reported separately.

`ACDCOriginalStyle` enumerates every annotated ED/ES slice in native ACDC
training folders.  Each slice is resized with preserved aspect ratio, centred
and padded to 224x224, then independently min--max normalized to `[-1, 1]`.
The loader returns native four-class labels because upstream `main.py` expects
the dataset interface `(image, label)` and writes labels during its evaluation
export.  Its unsupervised reconstruction and contrastive losses do not consume
labels.  Enumeration itself checks for the corresponding GT file, so this
protocol is label-backed and must not be called image-only.

## Commands

Run from `baseline/CUTS/src` with the CUTS environment:

```bash
python main.py --mode train --config ../config/acdc_original_style_seed1.yaml
cd scripts_analysis
python generate_diffusion.py --config ../../config/acdc_original_style_seed1.yaml
python export_diffusion_persistent.py --config ../../config/acdc_original_style_seed1.yaml
```

The first command trains and exports latent maps for the full dataset, as does
upstream CUTS.  The second writes every Multiscale-PHATE level.  The third
derives one anonymous, GT-free raw partition with upstream
`get_persistent_structures`; it reads only `image`, `labels_diffusion`, and
`granularities_diffusion`, and writes no labels.

The upstream `run_metrics.py` entries named `*-diffusion-best` select the
highest score across diffusion levels using GT.  They are diagnostic oracles,
not valid runtime outputs, and are not part of this protocol. Even the
multi-class persistent metric uses GT-guided relabeling. Neither metric path
may supply predictions to a GT-free adapter.

## Audit fixes and separate image-only variant

`config/acdc_image_only_seed1.yaml` selects `image_only: true`. In this mode
enumeration and sample loading never probe or read GT files. A zero dummy
label maintains the upstream dataset pair interface; exported latent NPZs
omit `label` entirely. Setting `no_label: true` alone does not enable this mode.
ED/ES still comes from `Info.cfg`; this is image-only with phase metadata,
not phase discovery without annotations. The loader currently expects `.nii`.

The original-style config keeps random slice 7:3 and automatic CPU test/export
after training. The image-only config also keeps that split but explicitly sets
`export_after_train: false`; export requires a separate `--mode test` request.
Neither variant is patient-disjoint, and test/export covers the full dataset.
For the shared cardiac benchmark use its separate manifest/split/loader;
do not reuse this checkpoint as evidence of patient-disjoint evaluation.
The 224 canvas remains a project choice, not a claim of exact paper reproduction.

New latent files carry patient/frame/z, preprocessing identity, image-only
status, and config/checkpoint hashes. Diffusion propagates these fields, uses
image dimensions rather than GT dimensions, and omits GT in image-only mode.
Persistent export carries this metadata into its manifest. Image-only export
rejects missing provenance. Historical files without metadata remain usable
only in legacy mode, without a new provenance or end-to-end GT-free claim.

Checkpoint selection remains minimum weighted unsupervised validation loss.
Training refuses an existing checkpoint; latent and persistent exports refuse
nonempty output directories. Choose a fresh run path rather than overwrite
historical evidence. Diffusion skips an existing file only after matching its
source/config/seed provenance and validating its hierarchy. Its separate seed
is explicitly 0 to preserve the upstream default; failed hierarchies are rejected.

The multi-class metrics pixel-diffusion guard is corrected. IoU counting and
persistent cluster storage use int64 to support 224x224. Historical scores
computed with int16 are not silently considered equivalent to corrected scores.
Original metric conventions remain: per-slice aggregation, GT-present classes
for Dice, and GT-guided/oracle scoring. They are not adapter evaluation metrics.

Synthetic regression checks (no training or diffusion backend execution):

```bash
PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider baseline/CUTS/tests/test_original_style_contract.py
```
