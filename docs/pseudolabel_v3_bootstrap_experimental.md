# Experimental two-stage image-only teacher bootstrap

This is an **opt-in research recipe**, separate from the unchanged
`configs/pseudolabel_v3.json` baseline. It addresses two demonstrated software/
method-design limitations: named BG supervision can start before any foreground
support exists, and one anonymous prototype can contain disconnected places that
cannot safely share a semantic label. It does not establish real ACDC/M&Ms
foreground recovery, anatomy accuracy, or target-GPU performance.

Use `configs/pseudolabel_v3_bootstrap_experimental.json` explicitly. Never modify
an active run's checkout/environment or reuse its output directory. Changed
source and recipe require fresh teacher artifacts; do not override checkpoint
source/configuration checks to reinterpret old results.

## Stage 1: anonymous image-region learning

The teacher keeps image reconstruction and registration. Its original anonymous
soft assignments q additionally receive:

- Edge-aware spatial total variation: encourage neighbouring assignments to
  agree when the input image has little local contrast; no periodic wrap
- Soft region reconstruction: reconstruct local image intensities from each
  region's input-derived mean, so reconstruction explicitly trains q rather
  than bypassing it through the appearance decoder
- Existing information loss for assignment sharpness/diversity, with its weight
  explicitly reduced from 0.1 to 0.01 in this experimental configuration

The declared exploratory weights are 0.3 spatial continuity, 1.0 region
reconstruction and edge scale 0.25. Images are normalized per slice for these
losses. No coordinates, circle/ring template, foreground mask, manual stroke,
segmentation target or reference-derived ROI enters the loss. TV alone can merge
useful regions; image distortion alone need not produce spatially coherent ones.
These weights were explored on manufactured images and are **not calibrated
against real cardiac accuracy**.

All named-class semantic CE, including BG, and prototype-bank updates stay off
until the gate opens. Semantic-head gradients are set to None on unsupported
updates, preventing AdamW decay/momentum from moving an ostensibly frozen head.
Outputs remain UNKNOWN=255 during warmup. Neither a low total loss nor a timer
can turn on RV/MYO/LV supervision.

This is established unsupervised-segmentation machinery rather than a novelty
claim: [Kim, Kanezaki and Tanaka](https://arxiv.org/abs/2007.09990) combine feature
clustering and spatial continuity; [W-Net](https://arxiv.org/abs/1711.08506) couples
segmentation and image reconstruction with a different normalized-cut objective.
The implementation here is not a reproduction of either complete method.

## Spatial components, without prototype-wide label propagation

For this recipe only, hard anonymous assignments are split into four-connected
components. Each component gets its own pooled image/features/motion descriptor
and semantic logits. A ring and an unrelated speck sharing a prototype ID cannot
borrow each other's semantic evidence. The original q remains the input to all
anonymous clustering/spatial/reconstruction losses.

Small, capped, exactly tied and below-threshold assignments enter an explicitly
ineligible void channel before component construction. Uncertain pixels cannot
close a ring or bridge regions used as readiness evidence. Padded channels are ineligible. There is no hole filling, pixel addition,
manual ROI or semantic template in component construction. Components are ordered
by size and physical raster position, so anonymous-ID permutations do not pick
an anatomical winner. Defaults are minimum 4 pixels, maximum 256 retained
components, and original assignment probability >=0.70 at both component and
pixel levels. Dense allocation has explicit grid/batch bounds; overflow pixels
abstain instead of receiving a label. Counts describe observations, not accuracy.

The existing whole-region connectedness/enclosure/orientation rules then apply
to each actual spatial component. Dense validity additionally requires the
pixel's owning component to survive every region gate and the dense class to
agree with that owner. Bilinear interpolation cannot lend accepted anatomy to
rejected components, tiny specks or void. Temporal consistency still only rejects;
it never replaces probabilities or manufactures foreground.

`bootstrap.regions.mode="prototype"` is available as an explicit representation
ablation. It retains the whole-anonymous-ID representation and does not apply the
component adapter. The same original-assignment-confidence and owner-validity
guards still apply to this bootstrap ablation. The checked-in experimental
recipe uses `"components"`.

## Stage transition: repeated training-only candidate support

The gate uses detached input evidence, never semantic-head confidence or the
prototype bank as proof. It requires:

1. Raw confidence/margin support for a connected internal wall and its enclosed
   connected cavity, under the existing topology rules
2. Sign-free wall/cavity intensity contrast relative to local within-region
   variation, plus interface-edge enrichment against surrounding image gradients
3. Corroborating input evidence on at least one genuine neighbouring frame;
   replicated temporal endpoints do not count as independent frames
4. Stable physical wall/cavity masks on repeated visits to distinct training
   sample identities, across multiple training patients

Default permission criteria are at least 5 successful updates, 4 distinct
samples across 2 patients, 2 stable observations per sample, and role-mask IoU
>=0.8. Five updates is only an opportunity floor, **not evidence**. Local defaults
are at least 8 pixels per wall/cavity, contrast >=1 and edge enrichment >=1.25.
These are engineering safeguards, not calibrated probabilities of correctness.
Every proposed wall/cavity pixel must also retain original assignment confidence
>=0.70; a high component mean cannot hide an uncertain seam. Raw fixed-grid
neighbour corroboration is deliberately conservative under large
motion; no learned warp is treated as independent certification.

A bounded, deterministic identity-based anchor set preserves observations across
full epochs. It reserves patient representatives where capacity allows and uses
SHA256 identity ranks rather than an LRU window that could forget every sample
before the next epoch. Failed support or unstable masks reset that sample's
streak. Only locked TRAIN identities can advance readiness; cohort/identity
checks occur before optimizer work.

Once permission opens, **current-image eligibility is still required** for each
semantic update and prediction, including BG. If prototype agreement removes all
foreground from an image, its remaining BG supervision is also suppressed.
Existing confidence, margin, prototype-class agreement, neural-class agreement,
dense confidence/support and temporal-rejection gates remain. Unsupported pixels
are UNKNOWN. No fallback forces anatomy and no student guard is weakened.

The global permission latch stays open, but a loss of local support suppresses
semantic/bank updates and output on that image. Repeated candidate support is
not proof that a detected structure is clinically the heart.

## Checkpoint, inference and diagnostics

Teacher checkpoints include the readiness state, bounded masks, a versioned
activation receipt, locked training cohort, complete experimental configuration
and evidence settings. The freeze binds the checkpoint hash and a scalar
bootstrap summary. Missing or incompatible state fails closed. Checkpoint-only
export restores readiness; inference cannot accumulate observations or advance
it, including when exporting held-out patients.

`run_summary.json` records `bootstrap.stage` as `anonymous_warmup` or `semantic`.
Training records include candidate/accepted class support, readiness and successful
updates; export reports retain raw/accepted/decoded region and pre/post temporal
pixel support. See [foreground diagnostics](pseudolabel_v3_foreground_diagnostics.md)
for the distinction between zero teacher support, valid zero-Dice evaluation, and
student training refusal. A completed freeze is not a successful segmentation.

## First real-data check: small TRAIN-only pilot

Before another full-cohort run, use the dedicated pilot. It deterministically
selects four identities from the original TRAIN split, keeps all frames/slices
of each selected patient, and runs three complete epochs. A global batch cap is
intentionally absent: it could visit only one patient and make repeated,
multi-patient readiness impossible. Planning refuses more than 5,000 optimizer
updates before training starts; the planned count is printed explicitly.
The original patient split is never modified. The selected split and resolved
configuration are copied into the fresh pilot directory.

```bash
python scripts/run_bootstrap_pilot_v3.py \
  --dataset acdc --root "$PWD/data/ACDC/training" \
  --split-manifest splits/acdc_patient_split_seed42.json \
  --config configs/pseudolabel_v3_bootstrap_experimental.json \
  --out "$PWD/runs/pseudolabel_v3/pilot_$(date +%Y%m%d_%H%M%S)" \
  --device cuda --patients 4 --epochs 3 --batch-size 1 --threads 4
```

Add `--dry-run` to inspect selection and the exact update budget without training
or writing outputs. The pilot never runs independent reference evaluation or
student training. `pilot.log` captures teacher output; `PILOT_REPORT.json` uses
verified frozen artifacts and requires all four support checks:

- Bootstrap permission actually activated
- At least one optimizer update used accepted foreground supervision
- Frozen foreground appears on at least two selected training patients
- RV, MYO and LV each have nonzero final exported support

Exit code 2 and `STOP_INSUFFICIENT_SUPPORT` mean stop here and inspect the report;
do not retry student training, lower confidence gates, or launch the full cohort
just to force a pass. The old completed run remains untouched. Exit code 0 and
`SUPPORT_OBSERVED_NOT_QUALITY_CERTIFIED` establish only that support exists, not
that it is correct anatomy or that a full run is ready for deployment. These
criteria use no reference labels and do not select thresholds from Dice.

A four-patient pilot is not a representative accuracy estimate. Real-data
recovery and target-GPU behavior still require actual execution and review.

## Running an explicitly experimental comparison

In a fresh checkout and compatible existing environment, keep data/split, seed,
budget and output destinations explicit. Example:

```bash
python scripts/run_full_pipeline_v3.py \
  --dataset acdc --root "$PWD/data/ACDC/training" \
  --split-manifest splits/acdc_patient_split_seed42.json \
  --config configs/pseudolabel_v3_bootstrap_experimental.json \
  --out "$PWD/runs/pseudolabel_v3/bootstrap_$(date +%Y%m%d_%H%M%S)" \
  --teacher-epochs 3 --batch-size 1 --threads 4 --device cuda \
  --log-every 50
```

This command has **not** been validated on the user's GPU or real data here.
Compare against `configs/pseudolabel_v3.json` using a separate fresh output and
matched budget. Do not choose thresholds from validation/test reference masks.
The experimental standalone teacher config defaults to 3 epochs (the legacy
config remains 1). An explicit one-epoch budget is warmup-only: the sampler visits
each sample once per epoch, so it cannot meet the two-observation criterion.
The CLI warns about such an insufficient repeat budget; more epochs alone are
not permission to assign anatomy.

Student training remains opt-in; add it only deliberately, expecting an explicit
stop if the frozen training split still has zero foreground. Additional epochs
are not guaranteed to unlock the gate.

## Evidence and limits

Software tests cover frozen semantic parameters/bank during warmup, an explicit
state-machine transition, current-image rejection after activation, class-local
component support, ownership under upsampling, absence of GT reads, train-only
readiness, bounded stable history, checkpoint restoration and inference immutability.
Resolver/state-machine fixtures supply synthetic anonymous regions and therefore
are **not evidence that a network learned them**.

Separately, matched image-only learning probes use manufactured images and random
initialization. The small-teacher matrix uses 128x128 inputs, 12 anonymous
prototypes, 600 updates and seeds 42/7 for annulus/constant/noise/open-ring images.
The original recipe found zero qualifying annulus candidates in both seeds,
including component-local diagnostic reassessment. The experimental warmup
objectives plus component-local assessment found qualifying candidates in both
seeds; original whole-prototype assessment still found none. At the final confidence-aware endpoint, neither recipe accepted any of its six
negative-control runs. Earlier sampled diagnostics precede the final confidence
fix and are retained as historical traces, not final-gate time-series proof.
This isolates candidate support on manufactured images, not segmentation accuracy.

A separate four-frame/two-manufactured-sequence full-trainer trajectory from
random initialization activated readiness at update 26 and first changed named
semantic parameters at update 27. Two subsequent updates lost local support and
correctly abstained. A checkpoint-initialized continuation also
activated, but lost local support on 20 subsequent updates before recovering;
those updates correctly abstained. Stability and generalization are not guaranteed.
A bounded default-width check (24/48/16/64, 12 prototypes, seed 42, 600 updates)
also accepted the annulus endpoint and rejected noise. A complete default-width
cohort transition is still untested. Exploratory tuning preceded these matched
checks and used the same manufactured-image family, so this is not held-out
hyperparameter validation. The separate delivered validation bundle contains
the learning-probe scripts, seeds and full histories; those are not repository
benchmark artifacts. This checkout contains the contract tests and pilot runner.

Still required before calling a reported real-data problem resolved: match the
run log to its saved configuration and teacher artifacts, run the declared recipe
on real image-only training data, inspect support before opening frozen independent
evaluation, and verify target-GPU behavior. No real-data accuracy, heart identifiability, clinical
utility, novelty or SOTA claim follows from these probes.
