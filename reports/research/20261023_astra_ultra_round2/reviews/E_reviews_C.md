# E independent review of C endpoint headroom

Reviewed `workers/C_headroom.md` SHA256 `ac39a44686ce505080faa7b13f2834a0140b49faf7d395a98888f9e84406d4be`, the endpoint-headroom receipt, exact-arithmetic counterexample receipt and proposed transaction schema. Reviewer E / astra_design. **PASS for the stated finite-endpoint and evidence-identification claims.** No material correction is required. This approval does not create guard headroom, patient evidence, implementation parity or compute authorization.

## Independent checks

I re-read the immutable `8960b6b:results/PATIENT_METRICS.csv`, verified its SHA256 `5f3b8c49d8505bba052fdfb56a40097b0fdceb608e74afa1b5cb6f6f98b8e695`, and independently recomputed all three whole-patient oracle/constant/class-relaxation values. I also recomputed both RV correctness-coupling intervals, both BG-to-RV overlap bounds, and the rational risk values/eight rejected finite scales of C's manufactured example. Receipt: [E_review_C.json](../evidence/E_review_C.json), script: [E_review_C.py](../probes/E_review_C.py). All checks pass; float values agree within 1e−14 and integer/rational checks agree exactly. These are local table/algebra calculations, not prediction replay.

The whole-patient oracle scores are 76.8233%, 81.9047%, and 82.1007%. The block/full oracle equals the full constant model on these 18 saved patient scores. The four-model oracle adds .1960 percentage points. The RV overlap bounds reproduce [2,578,49,850] and [30,708,68,302] block-wrong/comparator-correct pixels for supervised and full, respectively. The joint correction/degradation endpoints obey x−y=b−a; they are not independent confidence intervals.

## Challenges and disposition

1. **Does zero endpoint headroom rule out a useful guard? No.** C correctly restricts its theorem to a selector choosing one of the retained complete final outputs per patient. A new scalar/joint policy, retrained model, subpatient routing, pixel ensemble or parameter interpolation lies outside that set. The zero block/full oracle gain is sample-specific hindsight, not a population or algorithmic ceiling. This distinction is explicit in C's headline, derivation, table and disposition.

2. **Can independent class maxima be called an attained segmentation? No.** C labels them a score-selection relaxation and explains possible overlap/inconsistent multiclass decisions. The whole-patient selector has a coherent exclusive output; the per-class relaxation need not. The RV-targeted oracle correctly retains and reports its MYO/LV collateral rather than recombining favorable class scores into an invented predictor.

3. **Do marginal confusion changes identify repairable update errors? No.** The patient-stratified coupling intervals are sharp given those correctness marginals on a common ground-truth domain, but they do not recover voxel coordinates, an admissible selector, source-class responsibility, or transaction effects. C explicitly retains the common-domain assumption and distinguishes partial identification from statistical uncertainty. Its final-model differences are not interpreted as theta_s versus a pseudo trial.

4. **Does the trajectory witness prove realizability under the fixed v2 code? No.** C expressly limits it to valid probability trajectories and does not equate a probability segment with a NativeStudent parameter line or CE gradient. This is sufficient to falsify an inference made from endpoint summaries alone without additional dynamics assumptions. Preserve that qualification in any synthesis: it is not a constructed second real training history with the same source/data/weights, nor an estimated prevalence of harmful patient updates.

5. **Are numerical decision receipts equivalent to independently replayable evidence? No.** C's three evidence levels correctly separate recomputing a recorded scalar guard decision, reproducing a frozen transaction from retrievable states/inputs/candidates, and evaluating independent policies. Original-unit A,b,e do not certify nonlinear risks. Hashes without retrievable bytes establish claimed identity rather than replay. The schema includes rejected/zero/unrepresentable options and labels P as the live driver, so it avoids counting shadow S/J outcomes as commits.

6. **Does repeated same-audit ordering establish policy utility? No.** C's current-audit portfolio property stays within its measured finite option set. Its schema requires fresh policy histories for total outcome claims. E's sequential probe supplies a separate executed witness: current-audit J superiority can reverse on another support and EMA makes the later candidate supports diverge. The reports are complementary, not substitutes for unseen-patient testing.

## Synthesis constraint

Use C's result to reject the narrow story that switching between already trained block/supervised endpoints can recover full EMA on the saved cohort. Do not use it to approve or kill a new guard: that requires the same-state mechanism record and subsequent independently evolving policy comparison. Retain the exposed-VAL, one-subset/one-initialization and source-recipe confounds in the final discussion. No further breadth or patient execution is needed to support C's current bounded conclusions.
