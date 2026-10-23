# A reviews F — generated-direction and effective-step accounting

**Verdict: ACCEPT within the stated affine-interface, manufactured-existence scope.** The generated-direction loophole from round 1 is closed at the production `make_candidates` / `controlled_head_step` interface. A complete NativeStudent/EMA training history realizing the fixture remains unproved and is explicitly excluded by F.

Reviewed `workers/F_realizable.md`, `probes/F_realizable_replay.py`, the original selected search fixture and saved replay receipt. Independently reconstructed three variants with [`../probes/A_review_F.py`](../probes/A_review_F.py): the original selected four-U-pixel fixture, a hard-coded copy of the rounded two-U-pixel fixture, and an independently reconstructed lifted default-LR representation. Complete outcomes and source/candidate accounting are retained in [`../evidence/A_review_F.json`](../evidence/A_review_F.json); terminal output is [`../evidence/A_review_F.log`](../evidence/A_review_F.log).

Command: `rtk proxy python reports/research/20261023_astra_ultra_round2/probes/A_review_F.py`. No discovery search was rerun, no F artifact was edited, and no patient/GPU execution occurred. This review is `EXECUTED_MANUFACTURED_CPU`.

## 1. Candidate accounting survives independent reconstruction

For all three replayed variants:

- An exact parameter copy of the student supplies teacher softmax probabilities. The probe does not assign arbitrary q values to manufacture a direction.
- `make_candidates` emits canonical unique source IDs `pseudo_class_1`, `pseudo_class_2`, accepted class counts `[0,1,1,0]`, and a common accepted denominator `Z=2`.
- Independent differentiation of the complete accepted hard-CE loss matches the sum of generated source displacements. Maximum parameter discrepancy is `2.22e-16` in the original and rounded fixtures, and `6.94e-18` in the lifted fixture.
- Candidate generation leaves the student parameters exactly unchanged. Joint and scalar policies start from separate exact copies of the same reference and use the same generated directions.
- Audit positives are RV=2, MYO=1, LV=1. This is not an absent-class loophole, and audit/U features are distinct. No U reference mask enters generation.

These checks substantiate the actual-generator claim. The head has two or three manufactured input features, accepted by the canonical affine interface; they are not demonstrated outputs of the current 32-channel NativeStudent feature extractor. F explicitly preserves that limitation. Likewise, an identical teacher/student pair is a valid stipulated input to the transaction seam, not proof that a specific supervised-plus-EMA history reaches it.

## 2. The finite guard truly accepts a same-audit Dice decrease

All three independently replayed variants have initial RV Dice `2/3`, accepted joint scale `.5`, and final RV Dice `.5`. The rounded fixture reproduces local coefficients `[.4264018107484825,.12385989160017911]`, actual risks `[.12300715598201488,.13649537584982982,.1665798751595983]`, and strict improvement of every actual Brier group by more than the declared `1e-8` tolerance. Thus the effect does not depend on accepting a tolerated positive Brier regression.

The first full-scale proposal is rejected; the accepted half-scale proposal is evaluated through the real affine weights and softmax. This is neither a statement about only local-linearized risk nor an arbitrary external direction. RV TP remains one, while an additional false positive decreases hard Dice. Per-class balanced probability error and argmax overlap need not order updates identically.

F's exact aggregate scalar uses the same summed generated CE direction, the same actual predicate/backtracks, and ridge `2*lambda`, which is the correct `alpha=beta*1` restriction. It also loses RV Dice on this selected fixture. That result correctly prevents a claim that scalar allocation generally repairs the Brier/Dice mismatch. Plain aggregate preserving RV Dice here cannot establish a method ranking because another actual Brier group worsens and the example was selected for existence.

## 3. Default raw LR lifting is algebraically correct and narrowly interpreted

For `r=eta0/eta`, the lifted features `f'=[sqrt(r)f,sqrt(r-1)]`, weights `W'=[W/sqrt(r),0]`, and unchanged bias preserve initial logits. An affine CE-gradient source has audit logit response proportional to `eta*(f_a'f_u+1)`; the additional constant feature plus trainable bias gives

`eta*(f'_a' f'_u+1)=eta0*(f_a' f_u+1)`.

The independent replay matches source-wise audit logit effects within `1.33e-15`. Candidate probabilities, common denominator, effective local moments and finite affine trial paths are consequently preserved up to roundoff. The default raw LR `.001` variant therefore reproduces the same Dice drop with maximum feature magnitude about `247.6`.

This is an exact effective-step reparameterization, not a new ordinary-scale counterexample drawn from native features. Raw LR alone does not bound the function displacement when features can scale. F explicitly says this, and neither synthesis nor a figure caption should omit the large-feature qualification. The present review does not add a native feature-range, realistic activation, or EMA-reachability claim.

## 4. Search and policy scope

F retains original selected trial 292, all attempted search rows and fixed postprocessing. This review replays the selected case rather than inferring population frequencies from it. One success among a filtered, first-success search is existence evidence; `1/292` and `1/64` must not become empirical failure-rate estimates. The lifted and rounded variants are algebraic/dtype robustness checks, not independent randomly sampled events.

The guard's documented claim remains intact: actual current TRAIN-audit Brier protection is narrower than Dice protection. This counterexample rejects a stronger universal implication at the canonical generator/control seam. It does not demonstrate patient harm, poor generalization, typical default-LR behavior, or a defective implementation of the stated Brier contract.

Disposition: **ACCEPT**. No production change follows from this review. A separate patient replay would require provenance-complete frozen artifacts and authorization; none was performed.
