# Frozen SSL guidance — experimental implementation

Implementation date: 2026-10-11. Based on Self-Audit commit
`9e84fbfc0b819016004aaa0289f4b024f4e004c8`. This is a separate opt-in package;
it does not change v3/CQA configuration, training or artifact schemas.

For the current end-to-end command, mandatory native-data preflight, frozen
held-out exports and separate reference scoring, use the
[full pipeline runbook](stable_guidance_full.md). This page describes the method
and individual prepare/train/export stages; its verification section records
the original implementation environment.

## What is implemented

1. **Frozen CineMA MAE adapter:** only the official `pretrained/cinema.safetensors`
   hash `a841b7e9300d60ea8feb73e3a7d8c60ddedc1d1d49ec1049ebfcfa337c5d5590`
   is accepted. A checkpoint merely renamed `cinema.safetensors` is not enough.
   The MAE is eval/no-grad; no segmentation checkpoint or student/EMA replaces it.
2. **Full-field feature cache:** native cine TZYX is tiled in XYZ with fixed,
   overlapping 192x192x16 windows. Every native position is covered, including
   padded small volumes and the far image border. Fused XYZ tokens are mapped
   back to the native-coordinate low-resolution feature grid. A seeded Gaussian
   projection reduces 768 channels to the configured 64 for a bounded cache.
   This is not PCA fitted on evaluation data or a trainable semantic projection.
3. **Experimental anatomy readout:** deterministic multiscale clustering of
   frozen feature plus intensity, component wall/cavity hypotheses, orientation
   and adjacency evidence, weighted conflicting votes, then soft anchors. The
   clustering is over the full patient cine; local hypotheses are per slice.
   Cine context enters through the subsequent temporal graph. This is **not** a
   joint global anatomical optimizer, learned anatomy head or recovered SOTA.
4. **Whole-cine sparse graph:** image/feature-weighted spatial neighbors and
   mutual local temporal matches with best/runner-up margin and photometric
   veto. No fixed-coordinate temporal assumption, cross-patient graph or
   unregistered cross-slice propagation. Ambiguous matches contribute no edge.
5. **Anchored bank:** `Q_new = alpha R0 + (1-alpha) P Q`, fixed P and R0. The solver
   reports residual and error bound `residual/alpha`; a failed solve cannot be
   frozen as converged. It separately propagates anchor support for reliability.
6. **Student:** multiscale image encoder and decoder. Feature distillation works
   from the first batch even with all-UNKNOWN semantic output. Semantic target
   weights are detached. Classes require accepted support in at least two TRAIN
   patients; supported classes can train without waiting for all classes. No
   confidence-based student feedback changes the fixed bank. BG-only batches do
   not update the semantic decoder. The checkpoint is fixed last epoch, not best
   by reference Dice. An entirely unsupported run is `REPRESENTATION_ONLY` and
   cannot export named predictions. Partial semantic support is reported as such.
7. **Freeze/export:** input, checkpoint, source, split and stage-artifact hashes;
   separate TRAIN bank and held-out student export; native TZYX label arrays,
   UNKNOWN=255, exact native XYZ T shape/affine in the manifest. No evaluator or
   reference-mask reader is called. This artifact schema is separate from v3;
   do not pass it into a v3 evaluator as if schemas were interchangeable.

## No-GT scope and limits

The local pipeline never reads reference masks, Info.cfg, manual ED/ES selection,
ROI prompts, disease labels, or GT-driven cluster mapping/checkpoint selection.
It discovers raw full cine, not CineMA's preprocessed ACDC variant. Anonymous
clusters are not mapped with Hungarian matching against reference masks.

General anatomy priors are explicit, so this is not prior-free semantic discovery.
A ring-shaped non-heart object can satisfy these priors. A fragmented/open wall,
an absent cavity or a single merged component may still yield no semantic anchor.
The readout is an experimental baseline, NOT a claimed solution to those cases.
Importantly, failed readout no longer disables feature learning.

CineMA MAE pretraining is documented as self-supervised, but **its complete upstream
preprocessing/annotation lineage has not been independently audited here**.
Artifacts record `SSL_DECLARED_PREPROCESSING_NOT_FULLY_AUDITED`. The known MAE
weight hash does not prove the whole ancestry is annotation-free. Do not claim
"no annotations anywhere in all upstream stages" without completing that audit.
Known fine-tuned and unknown weight hashes fail closed.

No upstream data preprocessing or center-by-mask crop is reused. We preserve
native sampling and do not resample to the pretraining voxel spacing; native
spacing is recorded. This distribution mismatch and the 16-pixel transformer
stride/interpolated feature resolution can affect quality. Full-field tiling
can be substantially slower than ROI inference. The student sees native images;
its low-resolution pseudo-targets do not magically provide fine manual contours.

Numerical convergence is not semantic accuracy. Uniform anchors produce uniform
solutions; background-only anchors can produce background-only solutions. The
solver proof is for fixed P/R0, not neural SGD or an adaptive teacher. Scores are
reliability heuristics, not calibrated probabilities of anatomical correctness.

## Environment

The existing v3 canonical environment is unchanged. Use an isolated Python 3.11
CineMA environment with an already-compatible CUDA PyTorch/torchvision pair.
The official CineMA package has pinned scientific dependencies, so do not install
it into a running v3 environment. Example in an isolated environment:

```bash
python3.11 -m venv --system-site-packages .venv-stable-guidance
source .venv-stable-guidance/bin/activate
python -c 'import torch; print(torch.__version__, torch.cuda.is_available())'
python -m pip install 'git+https://github.com/mathpluscode/CineMA.git@c10daa1d93f0ea28d8b9ad9206b0f673d25805c1'
python -m pip install 'huggingface_hub>=0.25,<2' 'pytest>=8,<10'
```

The PyTorch/torchvision pair must already be appropriate for the server. There
is no silent CPU fallback when CUDA was requested. This install recipe is based
on upstream pyproject/API inspection, not an installation validated on the user's
GPU. Network access is needed initially. Local `--checkpoint ... --teacher-config
... --offline` works for the same pinned official weights. No credentials are
written to the repository or command line.

## Run, in the repository root

First run the focused tests (CineMA weights and nibabel not required for them):

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=src \
  python -m pytest -q tests/test_self_audit_stable_guidance.py
```

Prepare the **same deterministic four-TRAIN-patient selection** as the v3 pilot
for matching split/seed. Images must be original full cine, not ED/ES-only files.
`--patients 0` means the complete locked TRAIN cohort, not all discovered images.

```bash
RUN="$PWD/runs/stable_guidance/pilot_$(date +%Y%m%d_%H%M%S)"
python scripts/run_stable_guidance.py prepare \
  --dataset acdc --root "$PWD/data/ACDC/training" \
  --split-manifest splits/acdc_patient_split_seed42.json \
  --config configs/stable_guidance_experimental.json \
  --patients 4 --seed 42 --device cuda --tile-batch 1 \
  --out "$RUN/bank"
```

**Exit 2 with a valid FROZEN.json means the feature bank was completed but semantic
support is insufficient.** It is not a successful four-class segmentation run.
Inspect it; feature training is still deliberately allowed. Do not join prepare
and train with `&&` when testing representation learning under insufficient
semantic support. An exception or absent/invalid FROZEN.json is a failed stage,
not a license to train from a partial directory.

```bash
python scripts/run_stable_guidance.py inspect --bank "$RUN/bank"
python scripts/run_stable_guidance.py train \
  --bank "$RUN/bank" --out "$RUN/student" \
  --epochs 10 --batch-size 8 --width 32 --lr 0.0003 \
  --seed 42 --device cuda --threads 4
```

Teacher `--tile-batch` batches whole 3D windows; student `--batch-size` batches
2.5-D slices within a patient. They are intentionally independent. Neither
changes the number of graph nodes or retained regions per image. We have not
benchmarked an optimal batch size/VRAM setting. CPU clustering and graph work may
be a bottleneck; larger GPU batches do not fix that automatically.

Outputs include `REPORT.json`, `FROZEN.json`, per-patient features/anchors/Q/
reliability/labels and a student `last.pt` plus training history. No stage overwrites
an existing directory. No implicit resume or self-relabel cycle is implemented.

Independent held-out prediction export, without teacher adaptation or references:

```bash
python scripts/run_stable_guidance.py export \
  --student "$RUN/student" --dataset acdc --root "$PWD/data/ACDC/training" \
  --split-manifest splits/acdc_patient_split_seed42.json \
  --split val --out "$RUN/val_predictions" --device cuda --batch-size 8
```

A representation-only checkpoint yields only UNKNOWN. A partial semantic model
may export only classes permitted by its TRAIN support and returns exit 2 to
prevent claiming complete four-class readiness. Export probabilities/softmax
confidence are not calibrated by hidden references. No Dice is computed here.

## Historical implementation verification (2026-10-11)

See `reports/stable_guidance_20261011/VERIFICATION.json`. Focused CPU tests cover
linear-solve agreement and contraction receipts, nonconvergence, flat/uniform
negative controls, finite gradients without semantic labels, detach behavior,
mutual temporal matching, full-field tile geometry/batch invariance, hash
integrity, split isolation and prepare/train/reload/UNKNOWN inference roundtrip.
The provider and image-reader in the roundtrip are **explicit synthetic mocks**.
There is no mock/provider option in the production CLI.

Not executed in that initial verification: actual CineMA import/checkpoint inference, native nibabel
NIfTI loading, full existing repository regression suite, real ACDC/M&Ms training,
segmentation accuracy or CUDA performance. The initial local environment could not resolve
external download hosts and had neither nibabel nor CineMA installed. Adapter
API/geometry were inspected against pinned upstream source, not tested on the
503 MB checkpoint. Synthetic tests are not evidence of anatomical recovery.

## Required scientific comparisons

Keep the old CQA pilot as a negative result. Compare same pretrained/readout:
static anchors without propagation, spatial graph, matched temporal graph,
direct distillation, and the later transition-audit extension. Recurrent
correction, adaptive EMA teacher, DINOv3 adapter and student-proposed bank edits
are **not implemented** in this change. Neither pretrained distillation nor the
classical anchored solver is claimed as a new contribution.

Freeze recipe/selection before opening independent reference evaluation. Report
UNKNOWN and missing patients/classes, not only supported-pixel accuracy. Do not
select graph thresholds or teacher checkpoints with validation/test Dice and
call the resulting model-selection process annotation-free.

Primary sources used for adapter/contract inspection:
- https://github.com/mathpluscode/CineMA/blob/c10daa1d93f0ea28d8b9ad9206b0f673d25805c1/cinema/mae/mae.py
- https://huggingface.co/mathpluscode/CineMA/blob/main/pretrained/cinema.safetensors
- https://huggingface.co/mathpluscode/CineMA/blob/main/pretrained/config.yaml
- https://arxiv.org/abs/2203.08414
- https://arxiv.org/abs/2209.11359
- https://dennyzhou.github.io/papers/LLGC.pdf
