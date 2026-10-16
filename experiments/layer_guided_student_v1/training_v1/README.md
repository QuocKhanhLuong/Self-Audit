# Astra student training v1

This package connects the exact layer-level architecture to an actual optimizer
loop. Use [the current RTX 4070 handoff](../../../docs/prompts/ASTRA_TRAIN_4070_LATEST.md)
for execution. The old `integration/` package remains a separate,
representation-only scaffold with its original schema and admission rules.

## Executed model and inputs

`scripts/train_astra_student.py` explicitly constructs `LayerGuidedStudent`
(`layer_guided_convnext_tiny_v1`) using approved local official DINOv3
ConvNeXt-Tiny weights. It never enters the historical `GuidanceStudent` runner.
The baseline CLI/default configuration is unchanged.

The six `layer_upgrade/*.py` files retain their pinned reference bytes from
`f4b0fe7d9e49ee398354e20ff0981c818840f5d4`. The architecture keeps the slice-difference
adapter, four raw encoder stages, top-down decoder, stride-16 feature projection,
stride-4 coarse head and exact D/U detail factorization. Ordinary AdamW updates
the parameters with gradients. Guard and refiner are explicitly off.

One batch contains contexts from one patient at its native H/W. The bank's
historical image preprocessing is full-cine 1st/99th percentile clipping followed
by z-score. The adapter then applies one min/max transform per cine, forms
`[previous_z, center_z, next_z]` with replicated endpoints, and the model pads
bottom/right to multiples of 32. No ROI, ED/ES selector, GT or VAL/TEST reader is
part of this training loop.

## Versioned supervision policy

The opt-in policy is **`frozen_historical_q_reliability_v1`**. This is a documented
extension beyond the original scaffold's semantic-admission-off contract; it
does not change that contract or silently claim anatomically verified evidence.

1. Verify the externally pinned ORIGINAL `FROZEN.json`, full inventory, all
   payload hashes, split disjointness, teacher identity and four reviewed
   cardinal-v1 producer hashes. Recheck each patient payload before/after reading
   and the full bank at completion. Production runner inputs must be disk paths.
2. Preserve the historical class permissions: a class needs hard-label support
   in at least two TRAIN patients. Counts must match the payload when loaded.
   For stored distribution `Q`, let `w = reliability * allowed[argmax(Q)]`.
3. Form mass `m = w * Q` and support `s = w`. Apply the same positive pixel-cell
   overlap transport to both, on the registered native/padded stride-4 grid.
   Normalize `target = transport(m) / transport(s)` only for positive support;
   renormalize its class sum for floating-point consistency. Zero support stores
   uniform probability and has exactly zero loss weight.
4. Preserve the historical pre-transport semantic activation test: the batch
   must have some `argmax(Q) != BG` with positive weight. An inactive batch has
   no semantic objective or coarse-head optimizer update. An active batch uses
   all its positive supported target cells, including supported background.
5. Apply weighted soft CE directly to `outputs['coarse_logits']`. Do not resize
   native logits and call that the reference coarse objective.

This policy consumes **full historical Q**, not conditional Z and not hard
labels. `UNKNOWN=255` in historical exports is an abstention; it is never
replaced by BG. A hard-unknown pixel can still have a soft Q/reliability target,
as in the historical learner. No new confidence cutoff or Q-to-Z inversion is
introduced. The named targets remain pseudo-evidence of unverified quality.

The legacy bank predates `orientation_policy` and `teacher.source` fields.
Missing orientation metadata is accepted only with the exact four pinned
producer hashes; a conflicting field is rejected. The receipt names this basis
`historical_schema_pinned_producer_without_policy_field` and explicitly sets
anatomical orientation certification false. `provider@source_commit_expected`
is a source identifier derived from existing teacher metadata. No missing field
is filled with an anatomical truth claim.

The separate feature objective **`full_native_feature_cosine_v1`** transports
64-channel cached CineMA features to stride 16, averages over covered cells and
uses only fully native, nonzero feature targets. It is an unmasked-view cosine
loss. It does not invoke or claim the reference masked-view objective.

## Scope and deviations

| Item | Contract in this version |
| --- | --- |
| New optimizer loop | AdamW, one learning rate, clipping, deterministic seed/order; no scheduler, AMP or accumulation |
| Semantic source | Frozen historical Q/reliability, with historical permissions/activation; quality not certified |
| Feature objective | Full-view cosine, explicitly separate from masked feature learning |
| Native detail | **No native-detail loss. Exact split-head detail output remains zero/untrained.** Shared decoder and coarse path learn; do not claim learned fine boundaries |
| Guard/refiner | Off; no finite-step guard or refinement guarantee |
| Recipes | Pilot: 32 updates; full: 50 epochs, no step cap; batch size 1, seed 42 |
| Historical baseline | Preserved; new batch size/schedule/input geometry are not a matched scientific comparison |
| Checkpoint | New `layer_guided.training_checkpoint.v1`; optimizer snapshot is audit-only |
| Resume | Not implemented; full run starts afresh from the same approved pretrained weights |
| Inference | Dense argmax plus separate gated labels, confidence and entropy; no automatic evaluation/export |

The optional existing `layer_guided_convnext_tiny_conventional_v1` control uses
its existing conventional-head composition; its native detail can receive
coarse-loss gradients. Default recipes use the exact split head. The runner's
`NOT_TRAINED_NATIVE_DETAIL` label means no **native-detail objective** in either
arm; the zero-detail statement above is specific to the exact split head.

Learning rates and loss weights are starting settings, not validated convergence
choices. `semantic_loss = target_entropy + semantic_kl` (up to roundoff) is
logged so the target's nonzero entropy floor is not mistaken for optimizer
failure. These are minibatch losses, not held-out accuracy measurements.

## Artifacts and reload

`STARTUP.json` records class, architecture, parameter counts, pretrained identity,
weights hash, bank hash, source/config hashes and runtime/device. `metrics.jsonl`
records every actual update; `REPORT.json` appears only after final input/source
verification and final checkpoint save. A directory with no successful report
is incomplete even if it contains earlier checkpoints.

The checkpoint loader requires the externally pinned manifest hash, exact
inventory and current source/config hashes, then reconstructs the same approved
backbone and loads all tensors strictly. Preserve the code snapshot used for the
run. Positive optimizer steps with no semantic updates remain
`REPRESENTATION_ONLY`; named inference refuses that status. A semantic-trained
checkpoint is `PSEUDO_SEMANTIC_TRAINED_NOT_QUALITY_CERTIFIED`.

`gt_used_for_training=false` is scoped to **this training loop**. Upstream bank
and backbone lineage is inherited from their pinned receipts; this flag is not
a new audit of all upstream pretraining data or a global supervision-free claim.

`training_v1.checkpoint.predict_cine` accepts a historically preprocessed TZYX
cine. For raw NIfTI, call `self_audit_guidance.io.read_cine` first. It then applies
the same per-cine min/max/context transform as training. Dense output is always
four-class argmax; gated output additionally applies declared confidence/margin
thresholds and class permissions, retaining 255 on rejection. Neither output
changes targets or training gates. Returned arrays are TZYX, not on-disk NIfTI
XYZT. Any future exporter must explicitly preserve affine and axis mapping.

## Local evidence

See [the verification and review receipt](../../../reports/astra_training_integration_20261016/README.md).
The manufactured end-to-end test uses the real bank adapter, optimizer loop,
checkpoint writer/loader and prediction path with only the external pretrained
constructor replaced by a small encoder. It proves interfaces and update/reload
behavior; it is not a real DINO, patient, GPU, convergence or anatomy result.
