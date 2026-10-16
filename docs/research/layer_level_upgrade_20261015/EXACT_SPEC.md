# Exact implementation contract — layer upgrade v1

This is the implementation index for the preserved prototype, not a replacement for its source or proofs. Reference root: `reports/layer_level_upgrade_20261015/reference/`. Use `docs/prompts/ASTRA_LAYER_UPGRADE_EXACT.md` for the execution task. The original 19 files are preserved byte-for-byte; only the test filename receives a `.txt` transport suffix.

## Model contract

| Item | Reference setting |
|---|---|
| Input | NCHW, three adjacent z slices, center channel index 1, finite [0,1] |
| Input normalization | mean (0.485,0.456,0.406), std (0.229,0.224,0.225) |
| Context adapter | bias-free Conv2d(2,3,3,padding=1), zero weights |
| Encoder interface | four raw native stages; channels (96,192,384,768), strides (4,8,16,32) |
| Padding | bottom/right replicate to multiple of 32; no implicit semantic evidence on padding |
| Decoder channels | (96,128,192,256), shallow to deep |
| Decoder refinement | two ResidualUnit blocks per level; each has two bias-free 3x3 convolutions, GroupNorm(8), GELU; additive skip fusion |
| Coarse head | 1x1 from shallow decoder to four logits, stride 4 |
| Feature projector | 1x1 from encoder stage index 2 to 64 channels |
| Detail path | raw padded three-channel image -> 32 channels; fuse upsampled shallow decoder; residual refinement; zero-initialized four-channel detail head |
| Detail bound | registered fixed buffer b=1.0; raw residual mapped through b*tanh |
| Composition | disjoint-cell mean D, repeat U, k=4; l=Ua+(I-UD)r |
| Native result | crop padded logits to original H/W; retain padded logits for the exact identity |
| Output keys | logits, logits_padded, coarse_logits, raw_detail, bounded_detail, features, native_valid, native_hw |
| Random smoke encoder | torchvision convnext_tiny(weights=None, stochastic_depth_prob=0.0) |
| Random smoke count | 32,890,558 total = 27,818,592 encoder + 5,071,966 other |
| Official pretrained runtime | requires supplied approved DINOv3 object and checkpoint provenance; not exercised in the reference tests |

For input (1,3,65,97), logits are (1,4,65,97), padded logits (1,4,96,128), coarse logits (1,4,24,32), feature projection (1,64,6,8), validity (1,1,96,128). The valid mask is true only on the original rectangle. The proof applies before cropping. The native image branch is not the normalized backbone branch.

## Symbol and proof map

| Module | Required symbols | Contract / proof |
|---|---|---|
| layers.py | CenterDifferenceInput | repeated-center identity at initialization; bias-free neighbor differences can learn |
| layers.py | block_mean, block_lift, zero_mean_detail, compose_logits | DU=I, D(I-UD)=0; exact DIRECT detail-gradient exclusion for coarse guidance |
| model.py | OfficialConvNeXtStages, ResidualUnit, LayerGuidedStudent, build_random_convnext_demo | stage shapes, residual decoder, initialization and state dict; no automatic pretrained fallback |
| losses.py | evidence_distribution | nonnegative transported class mass/support; min_support=1e-6; valid mask; detach sources |
| losses.py | soft_ce, set_loss | no implicit resize; detached weights/targets; original summed numerator and clamped denominator |
| losses.py | masked_feature_loss | aligned structural mask; detached teacher; caller must actually mask the student input |
| guarded.py | project_direction | Euclidean projection of descent direction into g_c^T d>=0; <=8 guards, float64 reference solve |
| guarded.py | guarded_sgd_step | selected-parameter additive step, fixed deterministic closures, finite loss/objective checks, backtracking and rollback |
| refine.py | FrozenImageEnergy, image_grid_graph, refine_distribution | frozen summed quadratic graph/anchor energy plus linear unary; detached log-space mirror descent |

Preserve exact public defaults and validation behavior from the source, not just the table. Keep originals as an executable comparison oracle.

## Optimizer and energy details

`project_direction` normalizes nonzero guard rows without changing their halfspaces. Default tol is 1e-9; Gram pinv rtol is 1e-12. The returned vector is cast back to the candidate dtype; finite-step verification remains necessary. Zero is a valid fallback/stall, not evidence of useful learning.

`guarded_sgd_step` uses default max_backtracks=16, atol=1e-10, and halves the trial step. It checks both the objective and all frozen guard losses. It changes only supplied unique trainable parameters, does not populate `.grad`, and has no optimizer state. No subsequent unguarded update may be included in the same non-harm claim.

Energy uses each undirected edge once, or counts a repeated edge repeatedly in BOTH energy and degree. Defaults lambda=0.2, beta=0.5, spatial sigma=0.2 are prototype constants, not validated MRI settings. With weighted dmax and wmax, L=2*lambda*dmax+beta*wmax. Mirror refinement defaults to 3 steps, eta=0.5/L when L>0 and eta=1 otherwise; an explicit eta must satisfy 0<eta<=1/L for L>0. Float64 numerical nonincrease tolerance is 1e-10. Extreme logits causing loss of positive support are rejected rather than clipped.

## Non-negotiable scope limits

1. Exact coarse-logit identities do not imply pooled probabilities agree; softmax and D do not commute. Shared encoder updates can affect both branches indirectly.
2. The unbounded decomposition can represent every logit field. The implemented bounded residual intentionally restricts this claim; a wrong high-margin coarse label can resist correction.
3. Fixed proxy nonincrease is not class accuracy, validation non-harm, or protection of future batches. Wrong sources can be protected.
4. Image-energy descent is not Dice descent. The conditional semantic result assumes an image unary truly favors the true class; that assumption is not automatically observable.
5. Missing information supplies no semantic permission. No target masks, GT-selected ROI/phase, UNKNOWN-to-BG conversion, source-bank rewriting, or external-cohort fitting is part of this design.
6. No complete trainer, deployed admission rule, motion model, EMA controller, recurrent auditor or pretrained weights were bundled. These need separate integration decisions; do not invent them to claim exact reproduction.

## Acceptance protocol

Run the exact 47 tests, random full-model check and mathematical probes from a new verified copy. Then differential-test the port with identical states/tensors under a separate import identity. Preserve shape, gradient, initialization, rollback and failure semantics. Require original tolerances on matched CPU tests; report environment-dependent deviations instead of editing the oracle. Only clock times, JUnit metadata and other incidental execution fields are excluded from numerical parity.

If a defect is found, keep the reference intact, record a counterexample and introduce an explicitly versioned change. Integration scaffolding may add loaders/recipes, not silently alter these functions. Exact reproduction is the first gate; patient training and claims about generalization are later, separately authorized gates.
