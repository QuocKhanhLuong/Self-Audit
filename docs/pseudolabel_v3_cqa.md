# Counterfactual Quotient Auditing (CQA) — experimental research decision

Date: 2026-10-08. Implementation base: `f4c38e040e1faf7391396915a57976ebda9441f6`.

**Decision:** use CQA as the next concrete no-manual-mask research hypothesis,
not a claim of established originality, recovered ACDC accuracy or publication
readiness. The old configurations remain unchanged. This implementation is
opt-in through `configs/pseudolabel_v3_cqa_experimental.json`.

## Problem and proposed contribution

Anonymous appearance clusters need not be complete anatomical objects. A wall
or cavity split into multiple confident components can minimize the image
objectives while yielding no enclosure pair. With no accepted foreground seeds,
semantic training remains disabled. More anonymous-loss optimization need not
resolve this mismatch.

CQA changes the unit of prediction from a single anonymous component to a
**quotient of a confidence-supported partition**: several adjacent atoms can
represent one physical proposal. Before accepting an edit, an image-based
auditor compares keeping an interface with removing it. Its detached decisions
also supervise the anonymous network through same/different-region relations,
without requiring class names or an anatomical seed first.

The proposed paper question is:

> Can image-evidence-audited partition edits, fed back as permutation-invariant
> relational supervision, reduce semantic-bootstrap starvation without increasing
> unsupported anatomy or collapsing to all-merge assignments?

The contribution hypothesis is the coupling of **audited quotient construction
and pre-semantic relational learning**, not region merging, reconstruction,
consistency, cross-fitting, or the word counterfactual by itself.

## Implemented flow

```text
image-only cine -> existing appearance/motion encoder -> anonymous q
  -> existing confidence-gated connected atoms (void remains void)
  -> keep-interface versus merge-atoms image-risk audit
  -> bounded quotient proposals -> unchanged anatomy evidence and readiness
  -> unchanged named semantic learning / UNKNOWN / frozen pseudo-labels
  -> unchanged deployment student, only after foreground support exists

                       detached audit relations
                                  |
                                  v
                   coassignment loss on anonymous q
                   (active before anatomical seeds)
```

### 1. Local image-risk comparison, without semantic self-certification

For every adjacent atom pair, fit scalar intensity means on one checkerboard
fold and evaluate squared image prediction error on the other. Repeat in the
opposite direction. Compare the merged model with the fine partition:

`delta = held-out MSE(merged mean) - held-out MSE(original atom means)`.

The current image and genuine adjacent frames supply witnesses, each normalized
by its own image mean and standard deviation. Both fold directions must pass on
all witnesses. Replicated endpoints do not count. The 90th-percentile intensity
jump along the disappearing interface is an additional boundary veto.

A merge requires small excess risk and a weak interface. A keep decision requires
large excess risk and a strong interface. The intermediate region abstains.
Thresholds are explicit engineering settings, NOT calibrated correctness
probabilities or significance levels. Flat witnesses do not authorize merging.

The present implementation uses **fixed-grid** neighbouring images, not learned
warps. Motion can therefore cause conservative rejection. Checkerboard folds
and neighbouring frames are correlated; this is not statistically independent
validation and does not establish a causal effect. The proposal network has
seen the image. Do not describe this as independent causal certification.

### 2. Anti-chaining quotient construction

Process candidate merges in risk order with physical-position tie-breaking.
Every interface that would disappear must have positive image evidence. A
positive path cannot erase an untested, ambiguous or negative edge. Re-evaluate
the entire proposed union against **all original constituent atoms**, rather
than only comparing with the latest intermediate merge. This limits cumulative
intensity drift. Edges, merges and group sizes are bounded explicitly.

Only existing confident pixels are grouped. There is no hole filling, added
pixel, cross-void bridge, manual ROI or ring template. Feature pooling is
recomputed for each physical group. The original confidence, validity, void and
pixel ownership are retained; a one-hot quotient is not new confidence.

### 3. Learn edit relations, not anatomical labels

Let `q_bar_a` denote the mean anonymous assignment vector in original atom `a`.
For audited relation `s_ab` (1=merge, 0=keep), train with binary cross entropy of

`p_same(a,b) = dot(q_bar_a, q_bar_b)`.

The positive and negative strata receive equal weight within each sample. By
default both strata must exist before a sample contributes to this loss. Missing
or one-sided relations yield differentiable zero, not a manufactured negative or
an all-merge target. Image decisions and atom masks are detached; gradients flow
only through anonymous assignments and their feature network.

This objective is invariant to a common permutation of anonymous prototype
channels. The quotient implementation is invariant to atom-ID renumbering
with physical support fixed. **Neither is a proof of invariance to arbitrary
changes in the partition's granularity.** The existing information and image
losses remain active; no all-collapse guarantee follows from this objective.

### 4. Anatomy permission remains separate

The semantic head, prototype bank, named anatomical classes and topology do not
choose merges. Quotient proposals still pass the original image/topology,
orientation, repeated-TRAIN readiness, confidence, bank agreement and dense
ownership gates. A useful grouping is not proof that the structure is a heart.
Semantic updates and exports remain blocked when support is insufficient.

## What changed in code

- `src/self_audit_pseudolabel/cqa_v3.py`: bounded pure partition audit, quotient
  adapter, risk comparison, diagnostics and differentiable relation loss.
- `losses_v3.py`: optional validated CQA settings in `BootstrapLossConfig`.
- `trainer_v3.py`: run the quotient adapter after connected components and add
  relation loss before semantic readiness; retain original training guards.
- `configs/pseudolabel_v3_cqa_experimental.json`: explicit opt-in recipe.
- `tests/test_pseudolabel_cqa_v3.py`: mechanism, gradient and adapter tests.

No new learned head or pretrained weights are introduced. The existing model
state shape is unchanged. The existing pipeline persists the resolved nested
configuration in teacher artifacts and hashes package source files; inference
uses the same stateless audit. Do not reinterpret or resume an old artifact as a
new recipe by bypassing lineage checks.

## Prior art and defensible scope

| Work | Established idea | What must not be claimed as new here |
| --- | --- | --- |
| Nock & Nielsen, Statistical Region Merging (2004) | Statistical image-region merging | Merging or intensity-based evidence itself |
| IIC, ICCV 2019 | Label-free clustering via paired-view mutual information | Anonymous clustering / anti-collapse objectives |
| PiCIE, CVPR 2021 | Invariant/equivariant unsupervised clustering | Augmentation consistency |
| STEGO, ICLR 2022 | Distillation of feature correspondences for segmentation | Correspondence or affinity distillation alone |
| CUTS, MICCAI 2024 | Contrastive patch representations and multigranular segmentation | Multiscale grouping as the whole contribution |
| Pixel-level Counterfactual Contrastive Learning for Medical Image Segmentation (2026 preprint) | Counterfactual image synthesis and dense representation learning | First counterfactual medical segmentation method |

Primary sources:

- https://ieeexplore.ieee.org/document/1335450
- https://arxiv.org/abs/1807.06653
- https://arxiv.org/abs/2103.17070
- https://arxiv.org/abs/2203.08414
- https://arxiv.org/abs/2209.11359
- https://arxiv.org/abs/2603.17110

CQA edits the *partition*, rather than generating a counterfactual medical image.
It contrasts fine and merged prediction risks and distils the resulting edit
relations before semantic seeds exist. This is a candidate distinction, not an
exhaustive absence-of-prior-art proof. A strong classical region-merging baseline
with the same gates is essential. If assembly alone explains every gain, the
contribution is mainly engineering and the stronger learning claim must be dropped.

The no-GT contract means **no manual segmentation masks or GT-driven selection
in training**, not absence of all prior information. Handwritten anatomical and
orientation priors still provide class meaning. Never call this prior-free
anatomy discovery. No atlas, segmentation-pretrained model or manual ROI is added.

## First run: bounded TRAIN-only pilot

Do not modify a checkout while an old run is using it. Use a clean checkout after
that run ends, or a separate checkout, and a fresh output directory.

```bash
git pull --ff-only origin main
PYTHONPATH=src python -m pytest -q tests/test_pseudolabel_cqa_v3.py

python scripts/run_bootstrap_pilot_v3.py \
  --dataset acdc --root "$PWD/data/ACDC/training" \
  --split-manifest splits/acdc_patient_split_seed42.json \
  --config configs/pseudolabel_v3_cqa_experimental.json \
  --out "$PWD/runs/pseudolabel_v3/cqa_$(date +%Y%m%d_%H%M%S)" \
  --device cuda --patients 4 --epochs 3 --batch-size 1 --threads 4
```

This GPU/real-data command is supplied for execution, **not reported as executed
here**. The existing pilot uses TRAIN images only and does not train the student
or evaluate reference masks. Preserve its failure criteria.

Monitor `cqa_atoms`, `cqa_edges`, `cqa_merge_relations`, `cqa_keep_relations`,
`cqa_chain_vetoes`, `cqa_no_genuine_neighbor_samples`, `cqa_flat_witness_samples`,
`cqa_relation`, and the existing enclosure/seed/readiness/decoded metrics.
Counts describe support, not accuracy. A zero relation loss may mean insufficient
audit targets, not convergence. All-void atoms remain all-void: CQA cannot recover
missing confidence-supported pixels by inventing them.

## Ablations and acceptance criteria

Compare the original bootstrap, CQA assembly with `distill_weight=0`, and full
CQA, using the same locked TRAIN patients, seeds and budgets. Include a standard
region-merging-plus-identical-gates baseline. Report compute overhead as well as
support; repeated CPU/GPU synchronization and graph construction may be costly.

First require multi-patient anatomical support, actual foreground semantic
updates and valid frozen exports. Test flat/noise/open-ring, wrong-neighbour,
orientation and partition-permutation controls. Next freeze the recipe and
checkpoint-selection rule before independent GT evaluation. Do not pick the
best configuration by validation/test Dice and call that selection label-blind.

Report class-wise coverage, missing patients/classes, UNKNOWN fraction, Dice
with rejected foreground counted as errors, and coverage-versus-quality rather
than supported-pixel-only accuracy. Improvement must survive assembly-only and
classical-merging controls; a coverage increase alone is not a success.

## Verification performed in this change

See `reports/cqa_20261008/VERIFICATION.json` for exact local scope/environment.
44 focused tests pass in a noncanonical CPU environment. They include actual
teacher pooling/decoding modules and gradient paths, but not full-pipeline
training, the full repository suite, CUDA, or real ACDC/M&Ms evaluation.

A manufactured fixed partition with a split wall and split cavity has zero
single-component enclosure pairs. CQA merges two relations, preserves strong
interfaces and produces one enclosure pair; an open void seam is not filled.
These supplied regions are mechanism fixtures, **not a network trained from
random initialization to segment anatomy**. The free-logit optimization test
only verifies that relation gradients can reduce a fixed relation objective.

Real-data recovery, specificity of semantic naming, end-to-end checkpoint
roundtrip under this recipe, runtime on the target GPU and publication-level
novelty remain unverified. Large motion, low-confidence seams, different tissues
with similar intensity, and non-heart rings are explicit risks, not solved cases.
