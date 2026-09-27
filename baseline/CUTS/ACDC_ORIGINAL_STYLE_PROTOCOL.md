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
not valid runtime outputs, and are not part of this protocol.
