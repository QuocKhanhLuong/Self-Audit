# D — round-two claim and theorem audit

**Status: bounded failure characterization; no new-method, population-safety or patient-efficacy claim.**
Date: 2026-10-23. Report HEAD: ccfde9bb99b6050e00ad94fbd6d57fcf9afc8b3f. Canonical production source: 637f38ced217fe1db2347a87924695533056d8f0. No production/configuration changes, patient/GPU work, data/weight downloads or Git writes.

This round asks what follows from normalization, joint versus scalar allocation, and an actual-outcome fallback. It does not repeat the round-one prior-art inventory. The important distinction is between a mathematical optimum, a local numerical candidate, an accepted nonlinear endpoint, and an unseen-patient outcome.

## Claim-to-theorem map

Use fixed reference theta_s, fixed candidate matrix D, fixed TRAIN audit A and weights. Let E_c(alpha) be the local quadratic group risk, F(alpha)=mean_c E_c(alpha)+lambda*||alpha||²/2, and R_c(alpha) the actual nonlinear audit risk at theta_s+D alpha. J optimizes over the feasible coefficient box; S restricts alpha=a*1.

| Claim | What establishes it | Required conditions | What does not follow |
|---|---|---|---|
| Numerical normalization preserves the QCQP. | Positive affine objective transformation and positive constraint scaling preserve ordering/feasibility. | Fixed finite positive scales; include ridge and budgets correctly; preserve box and group weights. | Same floating-point output, same stopping behavior, exact optimum, nonzero motion or better patient learning. |
| Exact joint local allocation is no worse than exact S. | Feasible-set inclusion gives F(alpha_J*) <= F(a_S*1). | Same moments/reference/budgets; exact scalar penalty m*lambda; feasible global solutions. | Lower unregularized mean E, lower actual R, a larger accepted displacement or post-guard dominance. |
| Keeping S as a local incumbent prevents numerical local regression. | Return a feasible candidate only if its original-unit F is no worse than the checked incumbent. | Independently verified feasibility/value and faithful selection. | The selected J ray remains as good after nonlinear backtracking. S must survive as an actual endpoint to obtain that stronger claim. |
| Choosing the best accepted actual endpoint dominates included candidates on observed mean risk. | Finite-set minimization, not a new optimization theorem. | Same reference/support/risk; finite faithful evaluations; retain zero and specified S/J endpoints; choose by the same actual mean; deterministic tie policy. | Componentwise dominance over S/J, population dominance, a better future EMA trajectory or clinical safety. |
| Two/three final candidates require only a two/three-model generalization bound. | False without independence or an adaptive-validity argument. | A hypothetical fixed finite class independent of fresh i.i.d. evaluation data supports a union bound. | Reused TRAIN data become independent because only a few final endpoints remain. |

All four valid deterministic statements are standard optimization/set-selection facts. Their implementation and limits can be valuable research evidence, but are not first-ever theory.

## 1. Which normalization is an equivalent problem?

Write the local risk difference g_c(alpha)=E_c(alpha)-e_c-epsilon_c <= 0. For constants s>0 and t_c>0 fixed during a transaction,

    F_tilde(alpha) = [F(alpha)-F(0)] / s
    g_tilde_c(alpha) = g_c(alpha) / t_c

has the identical feasible set and argmin. The subtractive constant does not affect the minimizer. Dividing the whole objective scales lambda as well; dividing only the data-fit part changes the relative ridge strength. Scaling constraints includes their budget and offsets.

A trap is normalizing every class by its own scale and then taking an unweighted mean of normalized E_c: that changes the class tradeoff. Separate row scaling of constraints is harmless mathematically; replacing the objective's class weights is not. Candidate-column rescaling also changes the box/penalty unless every induced transformation is included.

Equivalent exact problems need not have equivalent finite precision solver behavior. SciPy's SLSQP ftol participates in checks on Lagrangian gradient, constraint violations, step size and objective changes; normalizing quantities can therefore change termination. This is a reason to qualify execution, not evidence of a new loss or scientific improvement. [Official SciPy documentation, accessed 2026-10-23](https://docs.scipy.org/doc/scipy/reference/optimize.minimize-slsqp.html)

Preserve independent checks in original risk units. Report original F, group violations, comparison with S/zero and actual effective parameter changes. Zero-budget feasible sets may lack constraint qualification; normalization does not restore it. KKT residuals are meaningful only where the required multipliers/conditions exist. A solver success flag is not an exact-optimum certificate.

A's stress results should establish the numerical behavior and delimit failures; the algebra above alone cannot establish that a normalization works across real moment scales.

## 2. Local J versus S: four distinct gaps

The exact restriction alpha=a*1 has a_c=1^T A_c 1, b_c=b_c^T 1 and scalar ridge **m*lambda**. With this penalty, the scalar feasible ray lies inside the joint box. Therefore exact local regularized optimum ordering is immediate.

Four gaps prevent upgrading that inequality:

1. **Metric:** regularized F and unregularized mean E rank candidates differently when coefficient penalties differ.
2. **Model:** probability linearization omits residual-weighted output curvature, even for affine logits followed by softmax.
3. **Globalization:** J and S produce different rays; the same backtracking grid may accept different scales or reject one entirely.
4. **Measurement:** the guard controls current balanced TRAIN Brier, not Dice, calibration, unseen patients or future training.

Adding a feasible S incumbent to the local optimizer can close a numerical F-value gap. It closes none of the last three gaps. Selecting J because its local F is better and then discarding S can still yield a worse accepted actual endpoint.

Local-model proposal and nonlinear acceptance are established safeguards. Fletcher–Leyffer's filter SQP accepts progress in objective or constraint violation, whereas this repository requires classwise risk limits and mean-risk acceptance; it is therefore precedent for the architecture of safeguarding, not a convergence theorem transferable to this transaction. [Fletcher and Leyffer, Mathematical Programming 91, 239–269 (2002), institutional primary record](https://discovery.dundee.ac.uk/en/publications/nonlinear-programming-without-a-penalty-function/)

The most informative artifact in this round is B's [finite softmax counterexample](B_geometry.md#3-exact-softmax-reversal-from-the-same-state-and-directions). At uniform probabilities and two manually specified logit directions, the unique exact J=(1,0) has lower local regularized objective **and** lower local mean Brier than S=(1,1). Yet J increases the true-class actual risk at every positive scale; the eight default guard trials all reject, while S accepts at full scale and improves every observed class. The whole-box gradient certificate proves local optimality without requiring constraint qualification. Positive guard tolerance and a longer finite grid can admit sufficiently small harmful J steps, so the executed rejection claim retains its exact grid/tolerance. D independently checked the rational local moments and certificate; B owns the execution receipt. This separates the claims without relying on a solver defect. The directions were not generated by make_candidates, so neither pseudo-CE-family prevalence nor patient frequency follows.

## 3. Actual-endpoint portfolio: exact but limited dominance

Call the additional policy **F**, to avoid confusing it with the existing plain-pseudo P arm. From the same theta_s, run the fully specified finite S and J procedures functionally, without mutating live state. Let C contain the reference endpoint and every retained guard-accepted endpoint from those procedures. Define

    theta_F = argmin_{theta in C} mean_c R_c(theta).

If both S and J return their first accepted endpoint, C contains those endpoints; the statement does not silently cover untested steps or every point on either ray. A policy that evaluates all backtracking points is a different, more expensive portfolio.

By construction, measured mean R(theta_F) is no larger than the measured mean at any included endpoint, including the reference. Every selected nonzero endpoint also inherits the same classwise reference-relative budget/tolerance check. This is not componentwise dominance over S or J: the lowest-mean endpoint may have higher risk for an individual class than another accepted endpoint.

If comparisons use a ranking tolerance, the mean-risk inequality inherits that tolerance; if an objective different from mean actual risk ranks candidates, the claimed dominance changes accordingly. Ties, finite handling, tolerance, endpoint retention and trial budget must be frozen. Commit exactly the measured endpoint, then update EMA once.

F is a distinct proposed selection policy and costs both searches. Its observed-audit advantage is partly guaranteed by choosing the winner on the selection data; that advantage alone is not efficacy evidence. It cannot be attributed to J's local optimizer or used to prove J dominates S. Its future model/teacher state and future candidates diverge after commitment.

These are **same-transaction endpoints** from one supervised reference and frozen audit/candidate tuple. C's retrospective final-model selector instead chooses among complete trained predictors for each exposed VAL patient. Its oracle bounds only whole-patient routing within that fixed final-predictor library; it is not a bound on retrained guard policies, subpatient routing or future training trajectories. The two endpoint comparisons identify different objects, and neither establishes unseen-patient improvement.

## 4. Adaptive TRAIN selection is not population protection

The audit contributes to local coefficients, trial rejection and prior training history. Calling this validation, making the current supervised patient different, or retaining only two final candidates does not remove that dependence.

Dwork et al. distinguish ordinary adaptive holdout reuse from mechanisms that support valid repeated queries. Their Thresholdout guarantee requires protected i.i.d. holdout access through a specified noisy/thresholded, budgeted mechanism; direct repeated TRAIN-risk access does not satisfy it. No such mechanism exists in the current transaction. [Generalization in Adaptive Data Analysis and Holdout Reuse, arXiv v2, 2015-09-25, §4.1](https://arxiv.org/abs/1506.02629v2)

For contrast, a directly derived finite-class bound makes the missing assumption visible. For M models fixed independently of n i.i.d. patient examples, and patient loss in [0,1], Hoeffding plus a union bound gives uniform deviation

    epsilon = sqrt(log(2*M/delta)/(2*n)).

An empirical minimizer then satisfies population risk <= best candidate population risk + 2*epsilon with probability at least 1-delta. [Hoeffding, JASA 58(301), 13–30 (1963)](https://doi.org/10.2307/2282952)

Setting M=2 or 3 is invalid when the final endpoints were constructed through the same audit transcript. A fresh independently sampled audit after the entire candidate family is frozen would be an additional protocol, not a theorem about the current one. n is the number of independent sampling units, not pixels, crops, trial forwards or repeated uses of a patient. A genuinely independent evaluation can estimate a frozen selected policy's performance; it need not and should not be silently fed back into the transaction.

## 5. What the next result can identify

| Evidence | Identifiable conclusion | Claim excluded |
|---|---|---|
| Original-unit optimum/feasibility comparisons across moment scales | Whether numerical normalization executes the unchanged local problem more reliably. | New objective or patient benefit. |
| Same-state S/J with identical candidates, scalar penalty and guard, and exact or independently qualified comparable optimizer accuracy | Effect of coefficient freedom on the specified accepted-update rule, including scale/guard mediation. If J stalls numerically while S is analytic, this instead compares the implemented procedures. | Allocation-freedom attribution without solver qualification, or a full training-policy effect. |
| Realizable nonlinear reversal | Local optimum ordering does not imply accepted actual-risk ordering. | Failure prevalence in pseudo-CE candidates or medical data. |
| Same-state F versus S/J with explicit cost | Observed-audit best-endpoint selection behavior and overhead. | Independent evidence of generalization or intrinsic J superiority. |
| Frozen fresh policy comparisons at matched label/compute budgets | Empirical practical utility under that protocol, if actually executed. | Distribution-free or clinical guarantee. |

Recommended framing remains **failure characterization plus numerical qualification of an existing constrained update rule**. Continue only if the study distinguishes objective correctness, effective learning and frozen patient outcomes. No architecture or broad new algorithm claim is justified by these deterministic observations.

## Evidence and review status

This report makes no independent model/probe execution claim. A and B own the manufactured stress/counterexample receipts. Relevant primary-source metadata and access scope are in [D_sources.json](../evidence/D_sources.json). [D's independent review of B](../reviews/D_reviews_B.md) records the exact reviewed hashes, mathematical checks and challenge resolution.

C reviewed draft SHA256 8209c72e8d71984c4d5b5ed70bfb6ed37bdbda358c6e72a79afc0272842b3ea3. Both requested clarifications were accepted: the evidence table now conditions the coefficient-freedom interpretation on optimizer qualification, and section 3 explicitly separates transaction endpoints from retrospective final-model routing. See [C's review](../reviews/C_reviews_D.md).
