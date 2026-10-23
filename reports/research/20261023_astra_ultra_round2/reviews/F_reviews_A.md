# F cross-review of A — numerical claims and certificate boundaries

Status: **ACCEPT within the stated manufactured numerical scope.** Independent review and final revision check completed; both precision challenges were accepted and resolved. Initial reviewed A script SHA256 `2cfd1a370fdc036a5de957f14cb662170105a88a209a8b090f002f1a11efe071`; initial receipt SHA256 `df6f14a1c37bb8995f2b6d3669ce9a6a2cb4d55893b228de5452c3a2297fa7ac`. Evidence under review: `workers/A_numerics.md`, `probes/A_solver_qualification.py`, `evidence/A_solver_qualification.json`.

## Verdict

A's bounded result survives independent challenge: scaling recovers the saved scalar false stalls but can still produce an original-unit infeasible candidate and worse zero fallback. The suite is evidence for a qualification requirement, not a universally improved replacement solver. I found no disagreement in saved represented-point feasibility labels, but two uses of mathematical exactness required tighter treatment. No production repair or patient-effect claim is warranted.

## Independent checks actually run

`probes/F_review_A.py` uses `Fraction.from_float` to interpret every saved float64 coefficient and candidate exactly as a rational number, then independently evaluates each group excess and objective. It checks **1,078 retained points across all 114 cases**, including original, normalized raw/returned, aggregate and coordinate rays, strict/tolerant incumbents, and available box/grid references. It does not rationalize all grid candidates or claim a global optimum.

For the initial Decimal70 receipt there were zero boolean disagreements on represented-point exact-budget and declared-tolerance feasibility. The independent 1,078-point check was then rerun against the final rational-validator receipt, again with zero disagreements. It confirms **17 normalized raw candidates** that pass the declared 1e-10 rule but violate the exact represented budget, and **one raw candidate** that exceeds 1e-10. The distinction between the 17 tolerance-only points and the 7 exceeding A's narrower roundoff screen matters; neither count is an empirical patient-risk rate.

Read-only focused reruns of A's functions independently reproduce:

- `scalar_response_-6`: original alpha 0, normalized alpha 0.7, high-precision scalar reference about 0.6999993000007.
- `scalar_zero_budget_singleton`: normalized alpha about 5.1958e-12, exact represented excess about 1.3498e-23, admissible under declared slack but mathematically outside the zero-budget set; scalar reference 0.
- `m2_near_collinear_3`: normalized raw excess 1.182295056e-9, exceeding fixed 1e-10; the probe correctly rejects and returns zero, while a feasible aggregate ray remains available.

The replay imports A's module without running its artifact-writing `__main__`, so A-owned evidence is not overwritten. Outputs: `evidence/F_A_rational_validation.json` and `evidence/F_A_rounded_gram_check.json`.

## Precision challenge 1: Decimal70 is high precision, not exact polynomial arithmetic

Initial A validator lines 46-72 converts floats using `Decimal.from_float` but multiplies/sums under a 70-digit context. Float-to-Decimal inputs are exact, yet subsequent operations can round; finite precision alone cannot certify every cancellation-sensitive sign. Accordingly, calling that procedure literally exact represented-point arithmetic was too strong in general, even though F's rational check found no disagreement on this finite suite.

A accepted the request to use exact rational arithmetic for concrete-point feasibility/objective comparisons, retaining Decimal only for square-root reference approximation and readable output, with rational postchecks. This closes the certificate-wording gap for the represented candidate values. It still cannot certify pre-rounding moments, exact model risk, or arbitrary multivariate optimality.

## Precision challenge 2: a floating Gram matrix is not necessarily exactly PSD when reinterpreted as rational

I additionally checked all principal minors of the saved rounded A matrices using exact rational arithmetic. There are **17 negative principal minors across eight cases**. For example, B's finite-head case has group 1/2 determinant approximately −5.421e-22; the near-collinear family contains similarly tiny rounding defects across scales. These matrices were indeed formed from valid finite V/r by float64 Gram calculations; the finding is ordinary rounding, not evidence of deliberately inconsistent input construction.

However, “generated from squared responses” and “every represented rounded QCQP is exactly convex” are distinct claims. Interpreting rounded A as exact can lose PSD, so a formal convex-QCQP/strict-uniqueness or box-stationarity certificate must not be transferred automatically to all rounded cases. Concrete-point feasibility and objective comparisons remain valid for their stated rounded problem, and positive objective/constraint scaling still preserves that problem regardless of this minor convexity defect. A states this boundary explicitly in the final dossier and independently reproduces the 17 negative minors in eight cases. Its separate exact-rational Sylvester audit also establishes positive definiteness of all 114 represented ridge-regularized mean objective Hessians; this does not make all rounded constraints convex or certify the numerical optimizer. The finding does not negate the observed original-unit rejection or justify dropping the adverse case.

## Normalization and feasibility review

The actual normalization divides the **whole** baseline-subtracted objective, including ridge, by one positive objective scale, and divides each **whole** constraint, including budget, by its positive scale. Gradients follow those scales. The box and within-case ridge/budget values remain unchanged. This preserves mathematical minimizers/feasible sets in exact arithmetic. The coherent V/r/ridge scaling sweep is separately distinguished from the fixed-ridge sweep, which changes the optimization problem.

Preserving the mathematical problem does not preserve an SLSQP stopping tolerance expressed in different units. A correctly retains the original-unit 1e-10 postcheck and refuses the near-collinear candidate. A's optional 128*epsilon summand allowance is a diagnostic heuristic, not a formal floating-error theorem, and is not used to inflate the returned feasibility budget.

The original source's `e+delta <= e+epsilon+tol` arithmetic can suffer baseline cancellation; A records it separately and checks the baseline-subtracted expression. This is a stronger interpretation of the intended original-unit contract, not a claim that every implementation-level floating comparison is bitwise identical.

## References, incumbents and finite-step limits

- The 39 scalar cases have analytic interval/clamp formulas, with a high-precision numerical square root where needed and a represented-point feasibility recheck. That is stronger than a grid but should not be mislabeled an arbitrary-precision proof of all coefficients.
- A box-face point can be a useful numerical optimum reference only under its stated objective/feasibility qualifications. The finite grid supplies known feasible comparison points, not certified global optima. A positive gap to a known feasible point establishes avoidable loss; a zero gap does not prove global optimality.
- The aggregate ray correctly uses `M*lambda`; otherwise it would not be the joint regularizer restricted to equal coefficients. Keeping that ray among admissible candidates proves only a minimum-over-a-finite-set local-objective comparison under the same chosen feasibility rule.
- Exact-budget and tolerance-compatible incumbents have different admissible sets. One must not claim the exact-budget selection always dominates a tolerance-only original candidate under an unchanged contract.
- None of these local comparisons orders the nonlinear finite guarded outcomes. A correctly retains B's counterexample and recognizes that evaluating additional actual candidates would change the decision policy and compute. F's separately generated-direction example also shows that even strict actual Brier improvements do not imply hard Dice protection.

## Counts and unresolved scope

The stable initial A dossier records **seven**, not preliminary eight, original objective gaps above 1e-10 after tightening references to its strict budget check; the revision history preserves that change. A also identified that its initial `numerical_failures` summary omitted an original `SOLVER_FALLBACK_ZERO` status and corrected that metadata in the final receipt. No scenario was added or removed to improve the result.

The remaining limitation is scientific, not a missing synthetic test: no transaction moments from the retained patient jobs are available here, so these stress cases cannot explain their empirical outcomes or estimate operational solver failure frequency. Production changes, exact nonlinear safety outside the audited support, and efficacy remain unproved.

## Final artifact binding and disposition

The final read-only revision check reviewed these exact SHA-256 values:

- `workers/A_numerics.md`: `7c951a903ceed22de332a7b75f1f91e8a96a8c20df70ace85608257c466b398d`.
- `probes/A_solver_qualification.py`: `009a113dc9c1a54b2e3ccf0054f0d5c83dee6f3d2076b7aff45aca7d95b86a2b`.
- `evidence/A_solver_qualification.json`: `5dab6c4badf431633445f29bba19cb761a77d9fb9b0ec08106b90386e09b68bb`.

The arithmetic and rounded-PSD qualifications are incorporated, the adverse normalization case is retained, and the final independent rational check covers the same 1,078 represented points with zero classification disagreements. **ACCEPT** does not extend to universal solver improvement, exact pre-rounding moments, all multivariate optima, actual finite endpoint dominance or patient efficacy. No production edit is requested.

Read-only synthesis check: the A/F sections of `DECISION_BRIEF.md` at SHA-256 `8ca4783718eecb5133f252dc3a89a0b8ce164a626e9a0e1ce2399decd94b2274` preserve these numerical and manufactured-existence boundaries; no material correction was identified.
