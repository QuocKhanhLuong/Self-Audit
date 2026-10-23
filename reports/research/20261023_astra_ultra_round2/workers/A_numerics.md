# Worker A — numerical qualification, round 2

Initial stable dossier, 2026-10-23. Task `/root/astra_source`; narrow analytical challenge delegated to `/root/astra_source/data_contract`. All changes are research artifacts in this round's owned paths. Production, configs, baselines, round 1, patient jobs, GPU jobs, weights and Git publication are untouched by this worker.

## Result

**Equivalent numerical scaling recovers the small-response failures, but is not a complete replacement solver.** In 114 manufactured valid squared-response cases, including 39 scalar cases with a high-precision closed-form reference, the original solver has ten scalar coefficient errors above `1e-3`; the scaled solver has none. However, one large-response near-collinear case makes the scaled SLSQP report success while violating the fixed original-unit feasibility tolerance. The independent check rejects it and the standalone scaled method returns zero, materially worse than a feasible analytic ray.

An analytic aggregate-ray incumbent can prevent an avoidably worse **local objective** relative to that ray, provided both are evaluated with exactly the same ridge, budgets and feasibility rule. This is a finite-candidate selection property, not a global-optimization certificate or finite nonlinear guard dominance. B's independently established local-optimum/finite-guard counterexample directly blocks the latter interpretation.

## Reproduction and scope

- Worktree HEAD: `ccfde9bb99b6050e00ad94fbd6d57fcf9afc8b3f`.
- Canonical method source: `637f38ced217fe1db2347a87924695533056d8f0`; compared experimental source remains `8960b6baac4c6600553d900c303f192c2fcfa4f0`.
- Executed command: `rtk proxy python reports/research/20261023_astra_ultra_round2/probes/A_solver_qualification.py`.
- Probe: [`../probes/A_solver_qualification.py`](../probes/A_solver_qualification.py).
- Machine-readable full cases, vectors, moments, raw candidates, failures, reference qualifications and original-unit checks: [`../evidence/A_solver_qualification.json`](../evidence/A_solver_qualification.json).
- Terminal output: [`../evidence/A_solver_qualification.log`](../evidence/A_solver_qualification.log).
- Script SHA-256: `009a113dc9c1a54b2e3ccf0054f0d5c83dee6f3d2076b7aff45aca7d95b86a2b`.
- Production `self_audit/control.py` SHA-256: `6bedb4165e85b4cc1e507d73858e491a4b7e668601604f216a4472280b723068`.
- Round-1 tree digest before and after is identical: `58bfd186e37f71739eac1b7c5d79c92a66054770a14db7e6a49e285244665f67`. The script asserts preservation.
- Environment: Python 3.11.16, NumPy 2.4.6, SciPy 1.17.1. The retained numerical loop took approximately 0.47 seconds; this excludes interpreter/import overhead and is not a training/GPU timing claim.

All findings below are `EXECUTED_MANUFACTURED_CPU` or `DERIVED` mathematical consequences of the stated problem. There is no patient-distribution estimate, likelihood estimate for these failures, DINO/GPU execution, or production repair. Source behavior being qualified is `self_audit/control.py:22-61`, with original tolerances `1e-10` feasibility and `1e-12` objective-improvement postcheck. Nonlinear actual-trial checks at `:205-224` are outside this numerical qualification.

## Exact problem preserved within each comparison

Every case starts from explicit finite `V_c` and `r_c`:

`A_c = V_c' V_c`, `b_c = V_c' r_c`, `e_c = ||r_c||²/2`.

Thus the construction starts from valid squared-response vectors rather than arbitrary unrelated matrices and alignments. Moment construction is float64, and the independent checker evaluates those rounded moment entries and coefficient values using exact `Fraction.from_float` rational arithmetic. It does not pretend to recover exact real moments before float64 rounding. Baselines in the saved suite are at most `.5`; large responses do not imply large baseline Brier risk. Except for B's finite-head construction, these are abstract response moments and need not be realizable by the exact current EMA generator.

**Rounding qualification, accepted from F's independent review:** constructing `A` as a floating Gram matrix does not ensure that its rounded entries, subsequently treated as exact rationals, remain PSD. Exact principal-minor checks found 17 negative minors across eight saved cases, including near-collinear cases and B's rounded finite-head moment array. These are ordinary rounding discrepancies, not evidence that the underlying `V,r` construction was intentionally inconsistent. Therefore this report does not claim that every represented constraint is exactly convex. Pointwise rational feasibility and objective comparisons remain well defined. A separate exact-rational Sylvester check found the ridge-regularized mean objective Hessian positive definite in all 114 cases. This supports the objective's strict convexity for these represented problems, but does not turn numerical box-stationarity residuals into formal optimizer certificates or repair constraint convexity.

The independent moment audit is [`../probes/A_moment_audit.py`](../probes/A_moment_audit.py), with complete minors and Hessian checks in [`../evidence/A_moment_audit.json`](../evidence/A_moment_audit.json). Reproduce with `rtk proxy python reports/research/20261023_astra_ultra_round2/probes/A_moment_audit.py`. It reads the saved suite and changes no solver/case. Its input-file hash binds it to that receipt.

For a coefficient vector `x`, define the baseline-subtracted original-unit quantities

`d_c(x) = -b_c' x + (x' A_c x)/2`,

`f(x) = mean_c d_c(x) + lambda ||x||²/2`.

The proposed research-only numerical formulation uses `f(x)/s_f` and `(epsilon_c-d_c(x))/s_c`, where all `s_f,s_c` are strictly positive and recorded per case. The entire ridge term is divided by `s_f`; the entire budget/constraint expression is divided by its `s_c`; gradients receive the corresponding scaling. The box `[0,1]^M`, ridge value and risk budgets are otherwise identical to the original solver **within each case**. Subtracting the constant `mean(e)` and dividing by positive scales preserve the mathematical minimizers and exact feasible set.

SLSQP's `ftol=1e-12` is retained in scaled coordinates. Consequently its success flag alone is deliberately insufficient. Every candidate is independently checked for finite values, box membership, unscaled `d_c(x)-epsilon_c <= 1e-10`, and original-unit `f(x)<=1e-12`. The scaled method's failed check yields zero. No scaled tolerance is translated into a larger permitted original-unit budget.

The receipt distinguishes three notions:

1. **Exact-budget feasibility of the represented point:** rational maximum excess `<=0`, plus exact box membership. This is the rule used for saved known-feasible references and the strict incumbent.
2. **Roundoff-sized diagnostic:** a recorded `128*machine_epsilon` allowance based on absolute summand scale. This is descriptive, not a theorem/certificate or replacement feasibility rule.
3. **Declared numerical feasibility:** original-unit excess `<=1e-10`, matching the source's intended postcheck tolerance. This can admit an exactly infeasible point, especially with zero budgets. Such points are explicitly marked.

The receipt also repeats the production-style `e+d <= e+epsilon+1e-10` arithmetic for comparison. Ranking uses baseline-subtracted exact rational objective values to avoid constant-baseline cancellation; 70-digit Decimal strings are also saved for readability.

## References and deliberately limited comparators

**Scalar reference.** The exact one-dimensional interval bound is computed in high-precision Decimal arithmetic, followed by the clamped quadratic minimizer. Conversion to float is rounded downward along the nonnegative feasible ray, then rechecked. For scalar cases this is the full constrained mathematical solution, numerically represented with the stated precision.

**Aggregate ray.** Restrict `x=beta*1`, with `0<=beta<=1`. Its moments are `a_c=1' A_c 1`, `b_ray,c=1' b_c`, and the penalty is `M*lambda*beta²/2`. The `M` factor is essential: using scalar ridge `lambda` would change the restriction of the current vector objective. Coordinate rays similarly retain their exact squared direction norm in the ridge term.

**Box-face reference.** For `M<=4`, enumerate all `3^M` free/lower/upper box-face patterns, solve their linear stationarity systems and record a relative box KKT residual. A numerical minimizer of the unconstrained box quadratic is also a constrained minimizer only when its exact-budget rational check passes. This is an exhaustively enumerated numerical reference, not a formal arbitrary-precision optimality certificate. It does not assert general QCQP KKT regularity; degenerate quadratic constraints remain possible.

**Grid comparator.** For two-source cases, a `129 x 129` grid supplies only a finite feasible point after exact-budget rechecking. It is explicitly **not** a certified optimum, nor does its failure to improve imply optimality. The reported gaps are against the best saved exact-budget feasible point among the available comparators, not against an invented global optimum for every case.

## Executed suite and results

The 114 cases cover source-response magnitudes, coherent whole-problem scaling, fixed-ridge scaling, positive risk budgets, zero-response sources, a zero-budget singleton feasible set, independent/collinear/nearly-collinear/opposed columns, a wholly redundant group, a group forcing one coordinate to zero, and 3/4-source seeded examples. All arrays and seeds are retained. These deterministic stress cases are not a random representative sample of patient training.

| Measurement | Original solver | Equivalent scaled solver |
|---|---:|---:|
| Cases | 114 | 114 |
| Scalar cases with coefficient error > `1e-3` | 10 / 39 | 0 / 39 |
| Returned objective gap > `1e-10` to a saved exact-budget feasible point | 7 | 1 |
| Largest such gap | `2.44999755000245e-7` | `0.0043569347037887745` |
| Returned points failing declared original-unit feasibility | 0 | 0 |
| Raw scaled candidates rejected by original-unit postcheck | n/a | 1 |
| Reported numerical failure or fallback | 1 | 1 |

Seventeen raw scaled candidates pass the declared `1e-10` rule but are not exactly feasible at their represented float point; seven exceed the narrower roundoff diagnostic. This does not mean seventeen material risk failures: excess magnitudes are recorded and may be extremely small. It means that numerical tolerance must not be described as exact feasibility. Scaling reduces some stopping failures but does not erase ordinary numerical feasibility issues.

### Recovered false stall

`scalar_response_-6`: three identical groups with `V=1e-6`, `r=.7`, `A=1e-12`, `b=7e-7`, `e=.245`, and `lambda=1e-6`, zero budgets. The exact scalar alpha is `.6999993000007`. Original returns alpha 0 with `NUMERICAL_FEASIBLE`; scaled returns `.7`. The original's avoidable objective loss is `2.44999755e-7`; the scaled objective gap is approximately `2.45e-19`. This is not a feasibility conflict.

### Separating numerical scale from mathematical ridge shrinkage

In `scalar_uniform_-4_fixedFalse`, both `V` and `r` are scaled by `1e-4`, and ridge is also multiplied by `1e-8`. The entire original objective/constraints therefore scale together, preserving the exact coefficient `.5217391304347826`. Original returns zero; scaled recovers that coefficient. The same coefficient persists across the coherent scale sweep, exposing scale-sensitive stopping independently of a changed ridge tradeoff.

In the paired `scalar_uniform_-4_fixedTrue`, the moments receive the same scaling but ridge remains `.3`. That is a genuinely different mathematical problem: the exact coefficient is about `8e-9`, and the exact objective improvement is only `9.6e-18`. Both solvers return zero. Calling this a major numerical failure based solely on “no update” would conflate expected ridge shrinkage with a meaningful missed opportunity. The receipt keeps coefficient error and original-unit objective gap separate.

### Retained normalization failure

`m2_near_collinear_3`: nearly identical response columns with the response scale `1e3`. Scaled SLSQP reports success in six iterations, but its raw candidate has maximum original-unit excess `1.182295056e-9`, exceeding the unchanged `1e-10` threshold. The probe rejects it and returns zero. The exact aggregate ray is feasible and improves the objective by about `.004356934703`; a feasible coordinate ray improves it slightly further. The original solver also obtains a useful point within declared tolerance.

This counterexample is the reason the result is **not “replace SLSQP with normalized SLSQP.”** Positive scaling preserves the mathematical problem, but a scaled stopping rule does not by itself preserve the original-unit numerical feasibility guarantee. Keeping the final unscaled check is mandatory, and an independent feasible incumbent can prevent this avoidable zero fallback. A generic repair of this failure is outside the round's scope.

### Correct zero under degenerate zero budgets

`scalar_zero_budget_singleton` has groups `(r,V)=(1,1)` and `(0,1)`, both with zero budget. The second constraint is `alpha²/2<=0`, so the mathematical feasible set is exactly `{0}`. Both numerical solvers return tiny positive values near `5.196e-12`; their excess is around `1.35e-23`, well inside the declared numerical tolerance, but they are not exactly feasible. The strict scalar reference and exact-budget incumbent return zero.

This zero is not a false stall, despite the unconstrained mean objective derivative being negative. It also prevents an unqualified QCQP KKT-residual requirement: the constraining quadratic has zero gradient at the only feasible point. The box minimizer `.4999995` is correctly marked infeasible and never used as a constrained reference.

## Incumbent selection: what is proved, what is only tested

The probe retains original, normalized, zero, exact aggregate-ray, and coordinate-ray candidates. It selects the lowest **same original objective** among candidates passing the **same chosen feasibility rule**. Both policies are saved:

- A tolerance-compatible incumbent, using original-unit `1e-10` feasibility. It improves the original solver by more than `1e-12` in 11 cases. Original is worse than the exact aggregate ray by more than `1e-12` in eight cases.
- An exact-budget incumbent, using the rational exact-budget check on rounded moments. This correctly returns zero for the singleton case and never quietly promotes a tolerance-only point to a mathematically feasible point.

For either policy, if the aggregate ray belongs to its admissible finite set, the selected local objective cannot be worse than that ray's; the script asserts the comparison in exact rational arithmetic. This is a direct minimum-over-a-set fact. It does not imply global optimality, better coefficients under another ridge, or dominance after evaluating the real model. The analytic ray is an incumbent, not a surrogate declaration that the joint problem has been solved.

B's exact finite-head case is included as `B_finite_head_local_box_optimum`: both solvers recover local optimum `(1,0)` with objective delta `-.0156245`; the aggregate ray `(1,1)` has worse local objective `-.0119260833333`. B's separate actual-softmax experiment shows that the locally preferred update can fail all configured finite trials while the scalar ray passes. Therefore selecting only by the best local objective cannot guarantee actual guarded improvement; evaluating additional real candidates would be a separately declared policy change with additional compute.

## Qualification decision

The round strengthens the numerical claim beyond one anecdotal scalar failure and establishes a bounded adverse case for the proposed normalization. It supports a **research qualification requirement**: coherent scale-invariance tests, exact scalar references, original-unit feasibility checks, retained feasible incumbents, and explicit handling of tolerance-only points. It does not authorize a production solver patch, prove all multivariate optima, establish clinical significance of tiny local gaps, or explain retained patient results without recorded transaction moments.

Before any future authorized implementation: specify whether the numerical contract permits `1e-10` slack or requires a stricter represented-point budget, preserve the same mathematical ridge/budgets, and test both recovered stalls and the retained large-response failure. Judge finite-step behavior independently. No new architecture or long training run is justified by this qualification alone.

## Revision record

- v1: initial 114-case evidence and dossier. Before this stable version, reference qualification was tightened from a diagnostic roundoff allowance to exact Decimal original-budget feasibility; this changed the original gap count from preliminary eight to seven. All current counts use the retained exact-reference rule. Independent F review pending.
- v2: accepted F's arithmetic-precision challenge: Decimal70 is high precision but not literally exact product/sum arithmetic. Switched represented-point feasibility and objective ranking to exact rational arithmetic; all retained classifications/counts stayed unchanged. F independently checked 1,078 represented points with zero classification disagreements. Corrected summary bookkeeping to count the original solver's one `SOLVER_FALLBACK_ZERO`, updated script hash and timing. Scalar square roots remain explicitly high-precision numerical references, not symbolic exact arithmetic.
- v3: accepted F's floating-Gram challenge and independently verified 17 negative exact principal minors in eight rounded case arrays. Added the separate exact-rational moment audit; all 114 regularized objective Hessians are positive definite, while no universal exact-convexity claim is made for rounded constraints. A independently replayed F's original, rounded and lifted actual-generator fixtures; review is `reviews/A_reviews_F.md`.
