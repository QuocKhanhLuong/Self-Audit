# Failure attribution: separate the objects and the evidence

**We have not localized the first failing stage of this particular ACDC run.** We have proved a geometry limitation and a loss/export ambiguity, corrected geometry behind an opt-in flag, and implemented measurements that can distinguish competing causes once 4070 artifacts are accessible. Neither software finding establishes how much of the reported Dice failure it caused.

## Actual source-level information flow

| Stage | Operation at the reported source | What it does and does not establish |
|---|---|---|
| Named proposals | Frozen feature/intensity clustering; wall/cavity enclosure and contrast; patient-right adjacent cavity as RV hypothesis | Anatomical priors, not a learned supervised segmentation head; non-heart lookalikes remain possible |
| R0 | Accumulated votes V and mass m give `(1-exp(-m))*V/m + exp(-m)*uniform` | Repeated correlated multiscale votes increase strength; no independence is asserted |
| Q and hard bank | Fixed row-normalized graph, unit self-loops, up to four spatial and two mutually matched adjacent-time edges; restart solve; probability, margin and reliability gates | Hard acceptance can reject a node with nonuniform useful soft targets; numerical convergence does not establish anatomy |
| Permissions | For each class, at least two TRAIN patients must have hard accepted pixels | A patient-level permission derived from hard labels, not an anatomical guarantee |
| Training weights | `w_i = reliability_i * allowed[argmax Q_i]` | **No hard-label-validity mask and no training reliability threshold.** Hard UNKNOWN can receive nonzero soft supervision |
| Semantic objective | Interpolate native logits to Q's grid; weighted soft CE for all four target components | Permission masks pixels by the winning target class; it does not delete disallowed class components from a retained soft target |
| Batch activation | Any permitted, positive-weight foreground argmax activates the batch | Active batches can also train weighted BG pixels; all BG/inactive batches contribute semantic loss zero |
| Student export | Native softmax, probability and margin gates, predicted-class permission; representation-only checkpoint short-circuits to UNKNOWN | Export has no propagated reliability gate; native predictions are not constrained to hard bank support |

On inactive batches semantic decoder/head gradients are set to `None` before AdamW, preventing their optimizer updates. The shared encoder can still change from feature loss, so latent semantic predictions can change. With a fixed bank, permissions cannot become newly authorized. A representation-only checkpoint always exports UNKNOWN. Conversely, a hard-bank support obstruction alone cannot establish zero soft learning or zero neural generalization.

The run is fixed stable guidance, not an active transition auditor. No result here is attributed to unexecuted CQA, refinement or recurrent self-labeling.

## Competing hypotheses and discriminating observations

| Hypothesis | Observation needed | Current verdict | What would weaken/falsify it as the main cause |
|---|---|---|---|
| Empty/wrong named anchors | TRAIN raw R0/support; fixed-teacher, frozen exposed-VAL R0 scoring and provenance | BLOCKED; plausible, not established | Useful named anchors on VAL, broad TRAIN permissions, failure appears only later |
| Graph destroys usable anchors | Same anchors before/after propagation on a common grid, class changes and protected evaluation | BLOCKED | Comparable or better Q with failure introduced downstream |
| Student cannot fit usable soft targets | Same active batches: H, KL, CE, weights; coarse-grid fidelity and encoder gradients | BLOCKED | Small weighted KL and high common-grid fidelity despite poor anatomy/export |
| Coarse/native transfer mismatch | Unchanged checkpoint native output versus the fixed projection control | Structural ambiguity PROVED; actual contribution BLOCKED | Native/projected outputs closely agree on the fixed diagnostic cohort |
| Gates reject useful predictions | Unthresholded argmax and each veto, with unchanged class mapping and thresholds | BLOCKED | Ungated predictions remain anatomically poor; accepted pixels are already wrong |
| Cardinal orientation suppresses RV proposals | Actual coded headers, left-axis values, wall/cavity opportunities, skipped RV checks and RV support | Source defect PROVED; run-level contribution BLOCKED | No relevant oblique/missing-orientation opportunity, or corrected geometry does not add justified RV candidates |

The reported low precision among the sparse predictions makes “only excessive abstention” insufficient as an established explanation. It does not identify whether wrong labels came from anchors, graph, student or export. Lowering a threshold could increase both coverage and error. No threshold was swept or selected.

`no_orientation_slices` in the historical readout is misleadingly named: it increments **per accepted wall/cavity hypothesis per scale**, not once per unique time/slice. Its raw value must be interpreted alongside wall/cavity opportunities and unique-slice counts; it is not directly an orientation prevalence estimate. Stored R0 does not uniquely recover raw vote masses or contradictory source histories. Retrospective raw-vote evidence is missing; the opt-in geometry path records total vote mass, conflicting-vote pixels and the explicit dependence caveat.

## Finding 1: a valid oblique header can prevent RV proposals

Plain language: a continuous left direction does not need to align with image rows or columns. The old rule requires a normalized affine-axis component of at least 0.9. A valid 45° rotation has maximum component `1/sqrt(2) < 0.9`; 30° and 60° also fail. The source then skips the RV candidate loop even when a wall/cavity hypothesis exists. This does not suppress the already-added MYO/LV votes in that same loop.

For a native affine A in NIfTI RAS coordinates, world-left coordinate is `-world_x`. Bilinear `align_corners=False` uses pixel-center map

```
x_native = (x_bank + 1/2) * W_native/W_bank - 1/2
y_native = (y_bank + 1/2) * H_native/H_bank - 1/2.
```

For centroids of two candidate regions in the same native z plane, the translation, z and half-pixel offset cancel in their difference. Therefore

```
left(RV) - left(LV)
 = -A[0,0]*(W_native/W_bank)*Delta_x_bank
   -A[0,1]*(H_native/H_bank)*Delta_y_bank.
```

This follows directly by substituting the pixel-center map into the affine and subtracting. A negative result places the RV candidate on the patient's right under this declared prior. The implementation uses the equivalent `(y,x)` covector, preserving anisotropic shapes and affine spacing/shear. Comparing unscaled bank indices could reverse the relation when the x/y scales differ.

`world_left_v2` selects coded sform, otherwise coded qform; records both and their agreement; rejects uncoded fallback orientation, singular/nonfinite geometry and an acquisition plane with no meaningful numerical left projection. It never assumes a default side. Conflicting coded forms are disclosed, with sform authoritative, rather than silently averaged. Header anatomical correctness remains an assumption.

Regression checks cover identity, flips, axis permutations, 30/45/60/90° rotations, anisotropic bank/native shapes, translations, invalid headers, out-of-plane direction and real nibabel roundtrips. A manufactured ring/adjacent-cavity fixture reaches the RV vote loop at 45° with the repaired rule but not the old rule. Aligned geometry reproduces the old readout numerically. A correct direction still does not prove the adjacent object is RV or that the ring is myocardium.

## Finding 2: CE near one is compatible with exact target fitting

For an active batch with the actual weights w and denominator `D=max(sum w,1e-8)`, define

```
CE_w = -sum_i,c w_i Q_ic log p_ic / D
H_w  = -sum_i,c w_i Q_ic log Q_ic / D
KL_w =  sum_i,c w_i Q_ic (log Q_ic - log p_ic) / D.
```

Adding the latter two cancels `Q log Q` pointwise and gives `CE_w=H_w+KL_w`, including zero target entries using `0 log 0=0`. At `p=Q`, KL is zero while CE is the target entropy. For `Q=(.1,.1,.1,.7)`, entropy is approximately 0.940448. Thus the reported CE near 1.017 is not by itself evidence of optimizer failure. This numerical example is not the run's measured target distribution.

The implementation computes all terms after the same native-logit interpolation, with the same detached float32 targets, permission-adjusted weights, activation predicate and denominator as `guidance_loss`. Inactive batches report zero for all objective terms and retain pre-activation entropy separately. Epoch means include inactive zeros and weight batches equally, matching the old aggregation; pooling pixels or excluding inactive batches would answer a different question. Per-class target mass and CE contributions are distinct from argmax support.

A read-only fixed-batch autograd diagnostic measures feature and semantic gradient norms and their cosine on enc0/enc1/enc2 before clipping. It does not update parameters or populate `.grad`. These final-checkpoint gradients cannot reconstruct historical gradients. If all TRAIN banks become available, the fixed NumPy batch schedule can reconstruct eligible update counts; these must agree with the checkpoint's recorded semantic-update counter. Eligibility still does not prove a finite nonzero semantic parameter change at every step.

## Finding 3: unchanged coarse loss can conceal changed native predictions

Let D be the linear native-to-bank bilinear interpolation. The semantic objective depends on logits L only through DL. Therefore for any perturbation N with `DN=0`, it has exactly the same value at `L+N` as at L. This is a statement about the objective on logits; it is not a proof that the CNN can realize every such N.

A complete finite example uses a 2x2 native grid and 1x1 bank grid. Set each native pixel to `log(.1,.1,.1,.7)`. Add `[[20,-20],[-20,20]]` only to RV logits. Bilinear reduction averages all four pixels, so the perturbation cancels exactly. The coarse target, probability and CE are unchanged, while RV replaces LV at two of four native argmax pixels. The executed float32 check tolerates floating-point roundoff. Both predictions remain generated from the same fixed class ontology; UNKNOWN is not converted to background.

The predetermined control is **interpolate logits to the bank grid → softmax → bilinear probability transport to native**. It is implemented only in diagnostics, alongside native probabilities, coarse fidelity, veto counts, and native/projected disagreements. It uses the run's existing thresholds and permissions. It is not a chosen deployment policy, does not establish true fine boundaries, and is not selected by VAL Dice. Shared CNN parameters and feature loss can affect all native pixels; sparse interpolation support does not imply a fraction of pixels was never trained.

## What is repaired and what remains conditional

Implemented: opt-in world geometry and noninvasive diagnostic logging/inspection. No representation, class mapping, graph, bank solver, default loss, default gate, default export or patient split was changed. Geometry preparation creates a fresh frozen bank and records its policy; it cannot reinterpret an existing bank. Feature and semantic learning retain the old objective unless diagnostics are explicitly enabled, which only records extra values.

Not implemented as a repair: a new native objective, changed export policy, graph replacement, clamping, threshold tuning, candidate-cap adjustment or learned grounding. The run evidence does not yet distinguish those interventions. A classical clamped/harmonic comparator would only be warranted after demonstrating that usable anchors are degraded by propagation; it would remain a baseline with erroneous-anchor and unreachable-component caveats, not a novelty claim.
