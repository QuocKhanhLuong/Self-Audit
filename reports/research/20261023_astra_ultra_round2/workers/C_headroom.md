# C — endpoint headroom is not guard headroom

Status: **new aggregate arithmetic and algebraic proof; no new patient evidence**. This report reads immutable experiment `8960b6baac4c6600553d900c303f192c2fcfa4f0`, with report worktree HEAD `ccfde9bb99b6050e00ad94fbd6d57fcf9afc8b3f`; production source remains `637f38c`. No patient arrays, weights, remote artifacts, GPU, training, or inference were accessed. Old reports and production files were not edited.

## New finding and its exact limit

**EMA-full exceeds EMA-block on the stored patient foreground score for all 18 patients. Consequently, even a label-informed oracle that chooses one of those two complete final predictors separately for each patient has zero headroom over always choosing EMA-full.** Among all four completed predictors, that oracle adds only **0.1960 percentage points** over the best constant predictor. This is a statement about a finite collection of final predictions, not a ceiling on any retrained guard.

Conversely, final confusion differences cannot identify whether a particular pseudo update harmed TRAIN Brier, whether a scalar/joint guard would reject it, or whether rejecting it would improve eventual RV Dice. Final supervised and EMA-block models are different training histories, not the two endpoints of a single pseudo transaction.

The phrase “guard can repair harmful RV updates” is therefore an **unmeasured hypothesis**, with three separate missing links: locating a harmful pseudo update; showing the guard distinguishes it on admissible TRAIN support; and showing the changed trajectory improves final RV without unacceptable collateral/compute cost. Current aggregate results establish none of those three.

## Recomputed input and selection unit

The only patient-derived input is `8960b6b:results/PATIENT_METRICS.csv:1–289`, SHA256 `5f3b8c49d8505bba052fdfb56a40097b0fdceb608e74afa1b5cb6f6f98b8e695`: 288 rows = four methods ×18 patients ×four classes. The script rechecks common patient IDs and equal reference marginals for every method/patient. Counts are pooled across a patient's reference frames; stored Dice is the mean of reference-present **3D frame** Dice, with each frame ratio pooling Z,H,W. These endpoints inherit the evaluator's absent-reference omission policy; no unavailable per-frame support was reconstructed.

Let d[p,c,m] be that saved Dice for patient p, foreground class c and completed model m. Define

    s[p,m] = (d[p,RV,m] + d[p,MYO,m] + d[p,LV,m]) / 3.

A whole-patient selector chooses **one complete model output for all frames/pixels/classes of patient p**. Its score is the mean of s[p,g(p)] over the18 patients. This selection unit preserves a coherent exclusive multiclass segmentation. It is an endpoint selector, not a training update rule.

For a fixed available model set M, the exact retrospective upper bound for this selector family is

    O_patient(M) = mean_p max_{m in M} s[p,m].

Proof: every s[p,g(p)] is at most the per-patient maximum; choosing an argmax independently for each patient attains equality. The model-choice oracle needs inspected reference Dice. It is **not an admissible training, model-selection or deployment policy** and must never be trained/tuned against these exposed VAL labels.

## Exact retrospective bounds

| Available final predictors | Best constant foreground % | Whole-patient ORACLE % | ORACLE gain over best constant, pp | Oracle choices |
|---|---:|---:|---:|---|
| EMA-block, supervised | 75.4727 | 76.8233 | +1.3506 | 11 supervised /7 block |
| EMA-block, EMA-full | 81.9047 | 81.9047 | 0 | 18 full |
| All four completed predictors | 81.9047 | 82.1007 | +0.1960 | 17 full /1 extra-supervised |

Even perfect whole-patient selection between block and supervised remains **5.0814 pp below** the completed EMA-full foreground score. This falsifies a narrow rescue story: selecting the existing block-versus-supervised final model per patient cannot recover the full-EMA endpoint on this cohort. It does **not** falsify training a new method whose outputs differ from both endpoints.

For a random whole-patient selector with the same mixing probabilities across patients and independent of their labels/scores, expected cohort score is a convex combination of the constant-model scores. It cannot exceed the best constant score. An image-conditioned selector might exploit predictable heterogeneity, but these aggregate metrics contain no admissible predictor of that heterogeneity and no independent test of such a selector. Best-constant choice here is also retrospective descriptive comparison, not untouched confirmation.

The looser quantity

    O_class_relaxation(M) = mean_{p,c} max_{m in M} d[p,c,m]

is **77.6292%, 82.4141%, 82.8457%** for the three respective model sets. It upper-bounds the whole-patient selector because it permits a different model per patient/class. It is only a **score-selection relaxation**: independently selected RV/MYO/LV masks may overlap, and the saved counts cannot establish a consistent exclusive multiclass reconciliation. Reconciliation can create a different predictor outside this family. These numbers are neither achieved segmentations nor bounds on arbitrary pixelwise ensembles, weight interpolation, new models, or joint-guard retraining.

Selecting whole-patient predictors retrospectively for RV alone also changes the other classes. Among all four models, the RV oracle uses15 full/3 extra-supervised and yields RV74.5083%, MYO79.5309%, LV90.7948%; compared with always-full, it improves RV0.8834 pp but reduces MYO1.1428 and LV0.6207 pp. This is another **oracle diagnostic**, not a recommendation to target those patients/classes using VAL labels.

Evidence: [C_endpoint_headroom.json](../evidence/C_endpoint_headroom.json), generated directly from immutable CSV by [C_endpoint_headroom.py](../probes/C_endpoint_headroom.py). No inference, parameter interpolation, probability averaging or hypothetical predicted masks were performed.

## What the count tradeoffs actually show

The following differences are comparator minus EMA-block; counts are aggregate pixels across the retained reference frames. The Dice difference is separately the mean of patient frame-mean Dice.

| Comparator | Class | ΔTP | ΔFP | ΔFN | Patient Dice Δ pp | Patients better/worse |
|---|---|---:|---:|---:|---:|---:|
| supervised | RV | −12,704 | −87,941 | +12,704 | +8.4918 | 13/5 |
| supervised | MYO | −9,486 | +5,244 | +9,486 | −3.1562 | 4/14 |
| supervised | LV | −14,443 | −14,652 | +14,443 | +0.7050 | 9/9 |
| EMA-full | RV | +29,829 | −87,630 | −29,829 | +21.3657 | 18/0 |
| EMA-full | MYO | +2,953 | −463 | −2,953 | +0.3860 | 9/9 |
| EMA-full | LV | +3,868 | −12,668 | −3,868 | +3.5850 | 15/3 |

For10 of18 patients, block has lower RV Dice but higher MYO Dice than supervised. For9 of18, block has lower RV but higher MYO than full. Thus “RV harm” and “MYO benefit” coexist at the endpoint; they are not two independently removable training components identified by these data.

Supervised improves RV Dice over block even though it has fewer RV true positives: the much larger false-positive reduction matters. This cautions against equating harmful RV updates solely with RV false negatives, low RV confidence, or a candidate whose pseudo class is RV. Shared parameters and softmax coupling can alter all foreground classes. The audit's one-vs-rest positive/negative support is relevant, but final confusion alone does not identify which audit term would detect these errors.

The stronger full result changes parameter scope, optimizer state, clipping and LR versus block; it is not an isolated guard or encoder-scope ablation. Those source differences were established in round1 and remain unchanged here: `8960b6b:self_audit/engine.py:133–145,168–200` and `self_audit/decoder_control.py:13–29`.

## Additional partial-identification result: endpoint error overlap is unknown

Even paired patient confusion matrices do not provide voxelwise agreement between methods. Within one patient and one true class, let N be the reference count, a the number correct under block and b the number correct under a comparator. The number x of block-wrong/comparator-correct pixels satisfies the sharp marginal bounds

    max(0,b−a) <= x <= min(b,N−a).

The reverse-change count y satisfies

    max(0,a−b) <= y <= min(a,N−b),    x−y = b−a.

These follow by bounding the intersection of the two correct sets between max(0,a+b−N) and min(a,b). Both extremes can be attained by rearranging their unobserved overlap within the common ground-truth domain. Summing separately over patients yields sharper bounds than first pooling patients. This is a **partial-identification interval, not a confidence interval**; it assumes the stored matrices refer to the same patient/reference domain, as their matching marginals and source protocol indicate.

| Comparator | True RV: block-wrong /comparator-correct count | True RV: block-correct /comparator-wrong count | Known net comparator TP gain |
|---|---:|---:|---:|
| supervised | [2,578,49,850] | [15,282,62,554] | −12,704 |
| EMA-full | [30,708,68,302] | [879,38,473] | +29,829 |

For the100,452 BG pixels predicted RV by block, the number that supervised predicts correctly as BG is bounded by **[67,687,100,452]**; for full it is **[65,801,100,452]**. These use the analogous intersection bound for block's RV prediction set and the comparator's BG prediction set within true BG, stratified by patient. The coordinates and actual overlap remain unknown.

Thus the tables imply substantial endpoint correction opportunities under other **already trained models**, but cannot reveal a pixel mask that a legal selector could detect, a candidate source responsible for those errors, or a training step whose rejection would remove them. Even exact cross-model prediction overlap would still be endpoint evidence, not a transaction counterfactual.

## Non-identifiability proof: same endpoint, opposite guard behavior

Consider a manufactured audit with one RV pixel and one BG pixel. Let u and v be their RV probabilities and let the two absent foreground probabilities both equal δ=0.01. Each pixel's four-class probability is

    (p_BG,p_RV,p_MYO,p_LV) = (1−p_RV−2δ,p_RV,δ,δ).

All probabilities below are positive and sum to one. The balanced audit risks, including the implementation's one-half Brier factor, are

    R_RV(u,v) = ((1−u)^2+v^2)/4,
    R_MYO = R_LV = δ^2/2 = 0.00005.

| Probability state | (u,v) | RV Brier | Mean foreground Brier | Hard confusion |
|---|---|---:|---:|---|
| Supervised reference S | (0.8,0.1) | 0.0125 | 0.0042 | one correct RV, one correct BG |
| Proposal A | (0.9,0.05) | 0.003125 | 0.001075 | identical |
| Proposal H | (0.6,0.2) | 0.05 | 0.0167 | identical |

A zero-budget finite guard accepts S→A and rejects S→H despite identical hard confusion at every endpoint. Along the harmful probability path

    S_t=(0.8−0.2t,0.1+0.1t),
    R_RV(S_t)=0.0125(1+t)^2 > R_RV(S), for every t>0.

All eight scales1,1/2,...,1/128 reject even with the configured1e−8 risk tolerance. The saved hard counts therefore cannot determine a would-be guard verdict even on the same audit, much less on disjoint validation patients.

A stronger temporal construction gives **the same initial and final soft probabilities and the same number of pseudo steps**. History1 follows eight increasing-risk subdivisions S→H; history2 follows eight decreasing-risk subdivisions S→A. An unobserved later labeled step returns each to S. All intermediate hard confusions are also identical. History1 contains eight harmful pseudo steps; history2 contains none. The endpoint cannot distinguish them, nor can an aggregate count of eight pseudo commits. A compensating labeled step is relevant because the implemented guard only constrains the extra pseudo displacement, not the preceding or future supervised steps.

This proves non-identifiability for unrestricted valid probability trajectories under the stated risk rule. **It does not prove the exact paths are produced by NativeStudent, per-class pseudo-CE gradients, the fixed optimizer or the actual patients.** The constructed probability interpolation is not silently identified with a parameter-space line. Source-shaped realizability is a separate question; exact rational arithmetic checks only the risks, simplex/hard decisions and finite verdicts. Since no time-indexed TRAIN audit states/candidates are in the endpoint table, that table supplies no assumptions that would exclude either trajectory class. Attributing the real run requires additional executed evidence, not choosing one illustrative history.

The independent read-only subagent `/root/astra_empirical/nonidentifiability_proof` supplied the initial algebra; C verified and strengthened it with equal pseudo-step counts and exact rational checks in [C_endpoint_nonidentifiability.py](../probes/C_endpoint_nonidentifiability.py). Output: [C_endpoint_nonidentifiability.json](../evidence/C_endpoint_nonidentifiability.json). These are manufactured proof witnesses, not extra patient observations.

## Why labeled replay and a simple comparator are still missing

1. **The final supervised model is not theta_s for a block transaction.** After the first different update, future supervised gradients, body AdamW moments, EMA targets and accepted candidates may all differ. Equal seed/data order does not make final-model subtraction a pseudo-update intervention. See `8960b6b:self_audit/engine.py:150–202`.
2. **Final confusions cannot replay either model.** They omit probabilities, cross-model pixel overlap, feature/Jacobian responses, model/teacher/optimizer states and candidate directions. Probability averaging or weight interpolation cannot be scored from them; even raw endpoint probabilities would not recover the missing training history.
3. **A legal simple selector must use admissible TRAIN evidence.** A matched scalar aggregate direction with the same finite guard can test whether joint coefficients add value beyond uniform shrinkage. It needs the same frozen theta_s, teacher q, U support and audit measure; `m*lambda` matches the joint objective on alpha=a·1. This scalar transaction comparator is not the same object as the retrospective whole-patient oracle above.
4. **Using labeled audit data can itself explain improvement.** A matched audit-gradient or extra-labeled replay can test direction-family/label-use effects; `supervised_extra` is a coarse control with different scope/loss/optimizer. Its final score cannot be substituted for a same-state audit-gradient intervention. No VAL/TEST label may determine the replay choice, coefficients, threshold, patient subset or stopping decision.
5. **Lower selected current-audit risk is not patient headroom.** A future finite portfolio that includes fallback/scalar/joint and selects the minimum feasible observed mean risk has a straightforward order property on that very audit. Repeated adaptively reused TRAIN support, Brier-to-Dice mismatch and divergent policy trajectories prevent interpreting that order as a generalization guarantee or final-quality improvement.

## Minimal sufficient future transaction evidence

The proposed machine-readable specification is [C_transaction_schema.json](../evidence/C_transaction_schema.json), explicitly marked `PROPOSED_SCHEMA_ONLY_NOT_PATIENT_EVIDENCE`. It separates three levels:

| Question | Minimum sufficient retained objects | Still not established |
|---|---|---|
| Recompute the declared finite-guard decision | Before/after group risks or exact weighted Brier sufficient statistics; group support/normalization; budgets/tolerances; finiteness; every scale/alpha; exact acceptance rule | Truth of recorded outputs, parameter realizability, Dice or generalization |
| Reproduce a same-state P/S/J transaction | Retrievable student/buffer/optimizer/teacher/RNG state or exact replay dependencies; supervised reference; frozen TRAIN input/labels/validity; teacher q; ordered candidates/parameter block; A,b,e/regularization; functional trials and commit hashes | On-policy trajectory benefit after choosing a different update |
| Estimate full-policy utility | Independently evolving matched fresh arms with initial-state/schedule receipts; declared fixed final endpoint; frozen patient evaluation and support policy; repeated subset/init blocks and compute | Untouched confirmation unless evaluation patients were prospectively independent |

For group c, sufficient arithmetic witnesses for actual risk are the original-unit weighted sums

    R_c = 0.5*(sum w*y^2 − 2*sum w*y*p + sum w*p^2),

with weights and support frozen across options. A local moment certificate A,b,e is not sufficient for actual nonlinear finite-step risk. A hash without retrievable state/input/direction bytes is an identity claim, not independent replay. Saving only accepted trials hides the rejected/nonexistent/zero/unrepresentable distinctions and cannot characterize guard activity.

Every transaction must declare the trajectory-driving policy. In a P-reference diagnostic, shadow S/J outcomes are never live commits; only P commits and EMA updates once. Finite risk violations by unguarded P are diagnostic outcomes, not guarded-contract failures. A future full S/J run has its own teacher and references after the first divergence and estimates a total policy effect. Keep those estimands separate in files and claims.

## Claim disposition

| Claim | Disposition from this round |
|---|---|
| Some completed-model endpoints improve RV over block with class tradeoffs | Supported by reaggregated saved counts/Dice |
| A whole-patient selector among block/full can beat full on this stored foreground score | Refuted within that exact finite selector family: full wins18/18 |
| Joint guard can recover the5.0814pp gap beyond the block/supervised endpoint oracle | Unknown; oracle does not bound retrained outputs |
| Final block errors identify harmful RV pseudo updates | Not identifiable from final aggregates |
| Guard rejection would repair eventual RV Dice | Not identifiable; needs same-state mechanism evidence followed by on-policy evaluation |
| Current-audit scalar/joint portfolio ordering proves safe SSL or clinical benefit | Unsupported scope extension |
| Inspecting retrospective oracle choices justifies a VAL-trained selector | Prohibited by the data-access contract |

Both new Python probes executed successfully using only standard-library arithmetic and immutable Git blobs. No production fix or patient/GPU result is claimed. The substantive next deliverable is an auditable same-state transaction record, not another interpretation of the existing four final scores.

## Independent challenge

E reviewed the substantive draft SHA256 `ac39a44686ce505080faa7b13f2834a0140b49faf7d395a98888f9e84406d4be` and independently reread the immutable CSV. Its separate probe reproduced all three constant/patient-oracle/class-relaxation values, both RV coupling intervals, both BG→RV overlap bounds, and all rational risk/eight-scale verdicts. Float agreement was within1e−14; integer/rational agreement was exact. No material correction was requested. [E_reviews_C.md](../reviews/E_reviews_C.md) records challenges to the selector unit, retraining ceiling, multiclass coherence, trajectory realizability and evidence-level boundaries, with PASS limited to this report's stated finite-endpoint and identification claims. It does not confer patient efficacy or compute authorization.
