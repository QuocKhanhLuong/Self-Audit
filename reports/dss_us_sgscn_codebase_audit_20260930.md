# Static code audit: DSS-US and SGSCN

**Date:** 2026-09-30
**Scope:** static inspection only. No package installation, import, model load, training,
inference, diffusion, ACDC-data read, raw-artifact write, or evaluator run was performed.

| Repository | Audited commit | License finding | Static verdict |
|---|---:|---|---|
| alexaatm/UnsupervisedSegmentor4Ultrasound (DSS-US) | d4ac44c60df18b921c590796f6994a4c8ac0726c | No LICENSE* file in the cloned tree | **Conditional GO only as a clean reimplementation** |
| osmond332/Spatial_Guided_Self_Supervised_Clustering (SGSCN) | 592efb6e72ceeef15c8be0630a4673eda5dce6f5 | GPL-3.0, LICENSE | **Conditional GO as an isolated GPL baseline** |

The clones were shallow scratch copies under .scratch/ and are not part of the
benchmark source tree. This report deliberately does not choose hyperparameters or
provide a run plan.

## Decision summary

DSS-US has a usable GT-free *algorithmic core*: DINO feature extraction, per-image
spectral oversegmentation, and a dataset-level K-means stage that assigns anonymous
semantic cluster IDs. However, the upstream code must not be copied or vendored
without permission because no license is supplied. Its stock project layout and
sweep entrypoint also mix an image producer with an oracle evaluator and W&B. A
separate clean reimplementation is required before it can become a Self-Audit
producer.

SGSCN's official demo is GT-free in the narrow sense: it opens only the input images,
re-initializes and optimizes a CNN separately for each image, then takes argmax.
It is not a usable raw-partition producer yet. It writes a randomly colored RGB
visualization instead of the integer partition; it has no seed control; it writes
inside the input directory; and its optional context-loss path hard-codes CUDA.
It must remain separate from DFC because it is GPL-3.0 and scientifically differs
from the frozen DFC profile.

Neither baseline is ready to run on the benchmark as checked.

## DSS-US

### What the code actually does

The primary pipeline is an eight-step image workflow:

~~~text
image files
  -> DINO ViT feature keys
  -> affinity / Laplacian eigenvectors
  -> per-image K-means eigensegments
  -> component boxes and component features
  -> one K-means over boxes from the whole supplied set
  -> anonymous semantic map
  -> optional DenseCRF
~~~

Evidence:

- configs/defaults.yaml:36-49 sets segments_num=15, clusters_num=15, propagates
  them into spectral K, the box-cluster count, and CRF class count.
- pipeline/pipeline.py:141-218 calls feature extraction and eigen extraction.
  pipeline.py:232-249 calls per-image multi-region segmentation.
  pipeline.py:304-420 extracts boxes/features, fits clusters, and forms the
  semantic segmentation.
- extract/extract.py:1015-1063 uses MiniBatchKMeans with random_state=seed
  across every extracted box in the supplied set, then saves that assignment with
  the boxes. This is the source of cross-image cluster-ID consistency.
- extract/extract.py:1067-1112 maps each per-image segment ID through the fitted
  box-cluster ID, preserving background as 0.
- extract/extract.py:1189-1230 applies an optional DenseCRF to the semantic
  map. Its dependency is SimpleCRF, named in req_pip.txt:1.

The segmentation mechanism does not need GT. The upstream default
pipeline_steps/defaults.yaml:1-10 has eval: False; evaluation is reached only
behind cfg.pipeline_steps.eval in pipeline/pipeline.py:422-431.

### GT/oracle boundary

The repository is **not** GT-safe by construction:

- Dataset documentation asks for gt_dir: labels
  (configs/dataset/README.md:1-16), and pipeline setup resolves gt_dir when it
  is present (pipeline/pipeline.py:62-85).
- With pipeline_steps.eval=True, pipeline.py:423-431 calls evaluate.
  That evaluation constructs EvalDataset and invokes
  evaluate_dataset_with_remapping.
- evaluation/dataset.py:12-62 opens a same-stem GT PNG.
  evaluation/segm_eval.py:39-87 calculates an IoU-based match and then remaps
  the prediction to GT labels. It is oracle evaluation, not a valid adapter or
  producer operation.
- configs/sweep/num_clusters.yaml varies clusters_num across 6, 9, 12, 15; the
  run script invokes pipeline_sweep_subfolders (run_pipeline.sh:1-15) and the
  sweep implementation optimizes/logs mIoU. That configuration-selection route
  cannot be used.

A GT-free producer is possible only when it owns its image list and output
directory, provides no GT path, hard-disables both eval and only_eval, and does
not invoke any upstream sweep entrypoint. The stock code does not enforce those
conditions.

### Input and output incompatibilities

- ImagesDataset uses cv2.imread then cv2.COLOR_BGR2RGB
  (extract/extract_utils.py:28-45). It consumes ordinary image files, rather
  than NIfTI/ACDC records. A gray PNG read by OpenCV is expanded to three
  channels, but this behavior must be made explicit in any MRI wrapper.
- Preprocessing is ToTensor plus ImageNet normalization for norm=imagenet
  (extract/extract_utils.py:890-945); the mean/std are 3-channel.
  The current ACDC protocol's slice normalization is therefore not directly the
  DSS-US input contract.
- get_model obtains the model with
  torch.hub.load('facebookresearch/dino:main', name)
  (extract/extract_utils.py:113-143). configs/model/dino_vits8.yaml leaves
  checkpoint empty. Thus a stock run has a mutable remote dependency, not a
  checkpoint hash.
- At 224x224 and patch size 8, feature extraction retains H // P by W // P
  tokens (extract/extract.py:129-155), i.e. a 28x28 patch grid. It silently
  truncates non-divisible bottom/right pixels at extract.py:137-139.
- The initial eigensegment map is saved as a grayscale PNG at patch resolution
  (extract.py:609-629). The semantic map is also stored as uint8 PNG
  (extract.py:1089-1112). The CRF branch is the candidate full-resolution
  output, but the stock code provides neither an [224,224] int32 artifact
  contract nor hashes, manifest, source identity, or a non-empty-output refusal.
- The per-image eigensegmentation has an internal border-majority background
  relabeling rule (extract.py:617-626). It is unsupervised, but it is a
  substantive producer rule and must be retained or explicitly versioned.

### Reproducibility and dependency findings

extract/extract.py:29 calls utils.set_seed(1) on import. That helper seeds
NumPy/Python/Torch/CUDA and sets cuDNN deterministic/benchmark flags
(extract_utils.py:504-517). The dataset-level K-means receives the configured
seed (extract.py:1015-1019). This is promising but insufficient for a sealed
baseline: the source has no run receipt, output inventory, checkpoint hash, or
repeat-hash check.

The upstream environment is Linux-oriented and large: conda.yaml:1-14 names
Python 3.11 and many packages; it pins CUDA toolkit 11.8 at line 64.
req_pip.txt adds SimpleCRF, spectralnet, sewar, and torchsummary.
SimpleCRF and the repository's dino2_models import are integration risks.
No package was installed during this audit.

### DSS-US verdict

**Do not vendor, execute, or modify this unlicensed source as benchmark code.**
The method is still worth pursuing as a paper-faithful clean reimplementation:
the evidence above identifies the GT-free core and the oracle-only code that
must be excluded. It should be reported as **DSS-US reimplementation**, never
as an official-code port, unless the authors grant permission under a license.

## SGSCN

### What the official demo actually does

demo_final.py runs a new CNN for every input/test/* image:

- It loads an image with cv2.imread, normalizes uint8 RGB/BGR values by 255,
  creates a new model, and uses SGD with momentum 0.9
  (demo_final.py:82-119).
- The network is a 3x3 convolution, nConv-1 3x3 layers, then a 1x1 head,
  all with 100 channels by default (demo_final.py:41-65).
- Each optimization step creates the current pseudo-target from the model's own
  per-pixel argmax (demo_final.py:143-166). No GT, mask, support image, or
  pre-trained checkpoint is read in this demo.
- Its losses are cross-entropy against that self-target plus L1 differences
  between adjacent logits (demo_final.py:104-165).
- With --center, it adds a per-channel spatial variance term around the channel
  center (demo_final.py:127-140,160-165).

So it is a per-image, direct clustering producer in the same broad family as
DFC, but it is not merely a DFC configuration.

### Critical output and protocol defects

1. **No integer partition is persisted.** The final integer img_target exists
   only in RAM (demo_final.py:173-180). The sole saved output is a 3-channel
   random-color PNG (lines 180-185). There is no raw cluster map, ID list,
   metadata, sample ID, hash, manifest, or seal.

2. **RGB output is not losslessly invertible.** Colors are created by unseeded
   np.random.randint(255, size=(100,3)) at line 119. The file does not store
   that palette, and color collisions are possible. Re-reading the visualization
   cannot safely recover the argmax IDs.

3. **The output count is neither fixed nor bounded below by minLabels.**
   minLabels=3 is described as “minimum number of labels” (lines 21-38), but
   the code breaks only after observing nLabels <= minLabels (lines 156-171).
   A step can therefore cross from more than three active IDs to two or one and
   then stop. If it never crosses, it exits at maxIter with more labels. The
   report’s prior statement that SGSCN guarantees at least three labels was
   wrong.

4. **No seed is set.** random is imported but no Python, NumPy, Torch, CUDA, or
   cuDNN seed/configuration occurs in the repository. Model initialization and
   palette differ across runs.

5. **It mutates the input root.** It creates input/result/ and writes
   visualizations there (lines 82-85,183-185), which is incompatible with a
   read-only dataset and sealed producer output root.

6. **The advertised context-loss path is GPU-only.**
   center.get_coordinate_tensors calls .cuda() unconditionally
   (src/center.py:4-11). Therefore --center fails on CPU and is not
   device-agnostic. The default is --center=False (demo_final.py:37-38),
   although the README describes the context loss as a method contribution.

7. **Input contract differs from ACDC.** The demo requires image files under
   input/test/*, reads them as 3-channel images, and has no NIfTI handling,
   manifest, grayscale declaration, shape validation, or slice provenance.

### Relationship to current DFC

Current Self-Audit DFC is a controlled direct-clustering baseline. Its frozen
primary configuration is 1-channel, 100 channels, 2 convolutions, 1000
iterations, minLabels=3, and equal similarity/spatial weights
(baseline/DFC/src/cardiac_benchmark/dfc_runner.py:13-39). It derives a
per-sample seed before model creation, checks input shape/finiteness, writes an
int32 [H,W] raw map, and records stop state (dfc_runner.py:72-115).

SGSCN differs at the scientifically relevant points: stock defaults are 50
iterations, spatial weight 5, optional center loss, and 3-channel input. Its
official source is GPL-3.0, whereas DFC is maintained as its own attributed
baseline. SGSCN must therefore live in a distinct baseline/SGSCN provenance
and source boundary; adding its code to the DFC package would both blur method
identity and contaminate that package with GPL-3.0 code.

### SGSCN verdict

**The official source is usable as an isolated GPL-3.0 reference, but cannot be
run as a benchmark producer unchanged.** Its final argmax, rather than its RGB
visualization, is the only meaningful raw partition. Any future port must be
reported as an official-code-derived ACDC port, with source changes, device
behavior, seeds, and raw output contract all made explicit.

## Cross-baseline comparison

| Property | DSS-US | SGSCN |
|---|---|---|
| Producer granularity | global semantic K-means after per-image eigensegments | separate CNN optimization per image |
| Output in stock code | grayscale PNG maps and .pth intermediates | RGB visualization only |
| Stable global IDs | yes, after global box K-means | no; IDs are fresh CNN channels per image |
| GT used by core partition | no | no |
| GT/oracle code present | yes; evaluator/remapping and metric sweep | no in demo_final.py |
| License-safe direct port | no: no license | yes, under GPL-3.0 with isolation |
| Determinism supplied | partial seed utility plus seeded K-means | absent |
| Current status | clean reimplementation candidate | isolated official-code port candidate |

## Audit limits

This is a source audit, not a scientific result. It cannot establish ACDC
accuracy, MRI transferability, GPU/VRAM behavior, output determinism, or the
quality of the MYO ring. Those questions require a later pre-registered smoke
run with a GT firewall and separate evaluation.
