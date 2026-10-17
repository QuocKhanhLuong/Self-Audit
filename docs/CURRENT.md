# Current implementation contract — 2026-10-17

This document and `ACTIVE_PIPELINE.json` supersede historical handoffs.

## Implemented and enabled

`train.py --phase coarse` constructs the approved local official
`dinov3_convnext_tiny` via `LayerGuidedStudent`. It uses the center-difference
adapter, four native encoder stages, residual top-down decoder, stride-16
feature projection, stride-4 coarse head and exact coarse/detail factorization.
The full-size random architecture check recorded 32,890,558 parameters. The
production constructor must load approved pretrained weights and records its
actual identity/count in STARTUP.json; the count alone is not identity proof.

Training uses `frozen_historical_q_reliability_v1`: original full Q,
reliability, class permissions and batch-activation policy. Mass and support
are transported together with pixel-overlap geometry; zero support has zero
semantic loss weight. Q is not converted to conditional Z. Images retain the
historical percentile clipping/z-score before per-cine min/max and slice
contexts. No patient-label reader exists in the optimizer loop.

The checkpoint schema is `layer_guided.training_checkpoint.v1`. Reload checks
source/config/bank/pretrained provenance and strict tensors. Optimizer state is
saved for audit, but exact resume is not implemented. A new full run starts
from the approved pretrained initialization, not by relabeling a pilot as resume.

## Implemented but not enabled

The current objective is full-view feature cosine plus coarse semantic CE.
Because the detail output head starts at zero and has no loss path, exact split
native logits remain the lifted coarse logits. Native boundaries are not learned.
The class/source gradient guard and frozen image-energy refiner are reference
operators only. Masked learning, temporal fusion, EMA semantic target updates,
native image-derived detail supervision and validated semantic self-learning
are not enabled in this training policy.

`--phase native` is rejected rather than pretending those features are complete.
The conventional-head control remains an internal experimental API for matched
research; it is not an alternative selected silently by the canonical command.

## Current evidence

The imported integration report recorded 125 CPU software tests, including
original operators, 17 differential parity cases and a manufactured end-to-end
optimizer/checkpoint test. That is historical evidence for the imported source,
not a claim that patient training, official DINO loading or GPU validation ran
in the cleanup environment. See CI and the current cleanup receipt for fresh
checks. Historical small-CNN Dice is not evidence about the new DINOv3 student.

## Source boundaries

Current model/runtime: `experiments/layer_guided_student_v1/`.
Compatibility I/O: `src/self_audit_guidance/io.py` only. Its cardinal orientation
metadata is legacy information; the current model must not treat it as newly
verified anatomical truth. The underlying bank lineage remains explicitly pinned.

No legacy trainer, bank producer, threshold sweep or automatic VAL scoring is
retained in the active command surface. Regenerating a historical bank requires
an explicitly selected archive worktree; cached, hash-verified source banks
can be consumed without running the old model.
