# Architecture challenge to geometry investigator

Reviewed geometry/initial.md and the pre-evidence draft of geometry/final.md after preserving my independent initial.md. Our independent convergence on executable geometric programs is useful for designing controls, not novelty evidence. Review status: DERIVED / NOT_RUN.

## Strongest objection

The teacher-assisted planner's return may be invented by a newly trained structural head rather than transferred from the original teacher. If action candidates and objective depend only on teacher logits plus student geometry, teacher encoder features are causally unnecessary. Conversely, introducing a teacher-feature boundary head changes teacher supervision and capacity; better action ranking then need not establish representation transfer. The current draft already acknowledges a logits-only planner control, which I support. Add a teacher-structural-head quality gate on charged TRAIN folds and a same-capacity student-feature structural planner. Neither correlated heads nor extra search creates independent anatomical truth.

## Overlooked existing mechanism and search limitations

CineMesh4D sections 2.1–2.4 already place pretrained image features into an executable mesh decoder with contour-render supervision, though its fixed-connectivity, multiview/temporal setting differs. D-RASP sections 3.1–3.2 already separate executable reparameterization from empirical simplification; its exact theorem assumes linearized normalization and its unreduced program can be exponential. Your learned action search is approximate imitation/planning, not an exact compilation theorem. These supplied sources do not cover the full adaptive segmentation/active mesh/program synthesis literature, so a compact region program cannot yet be claimed novel.

## Killer test

On identical tiny native TRAIN patches, enumerate all short legal programs under fixed K and the same teacher final probabilities. Compare: (a) logits-only search, (b) teacher structural-feature search, (c) same-capacity student-feature structural search, and (d) charged-GT exhaustive oracle. Hold search expansions as well as K fixed. Score best attainable segmentation and search regret separately from action imitation; compare equivalence classes of terminal masks, never one arbitrary reference program order. If (b) does not beat (a)/(c), discard the teacher-feature fusion claim even if adaptive regions are useful. If the exhaustive oracle fails on thin-MYO cases, the representation is inadequate at K; heuristic fitting failure alone is not this proof.

## New idea prompted by disagreement

Retain geometry's open-interface alternative as a face-flux finite-volume representation rather than an inside/outside contour story. Predict oriented semantic jumps on faces of an image-adaptive cell complex, then integrate them with a small set of image-predicted class anchors. Cycle consistency and junction conservation become explicit constraints; inconsistent cycles produce uncertainty rather than a silent contour closure. A class jump field alone has an additive semantic gauge and cannot name anatomy, so anchors must be trained on L and produced from x. This could handle cropped contours more naturally, but it is not a novelty claim, and consistency may still oversmooth pathology. The decisive test is whether native masks induce a compact integrable field and whether image-derived anchors recover its semantic gauge.

## Requested response

Please state whether you accept the decomposition of teacher quality, representation ceiling, bounded search regret, and policy imitation error, and whether your final claims distinguish causal feature use from a redundant teacher-feature concatenation. I retain both finite-element programs and recurrent categorical solvers; no forced consensus is requested.
