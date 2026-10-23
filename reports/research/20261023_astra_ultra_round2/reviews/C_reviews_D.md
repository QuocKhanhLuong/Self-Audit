# C independent review of D claim map

Initial inspected file: `workers/D_claims.md`, SHA256 `8209c72e8d71984c4d5b5ed70bfb6ed37bdbda358c6e72a79afc0272842b3ea3`, 2026-10-23. C checked the deterministic claims against their stated observation units and C's independently computed endpoint selector/coupling bounds. No patient compute, production edits, or new literature search was performed for this review; D owns its cited primary-source verification.

## Initial disposition

No material theorem overclaim found. The draft correctly separates regularized local objective, nonlinear finite outcome, selected observed mean risk, and unseen/future-policy outcomes. It also explicitly excludes broad novelty. Two clarifications were requested and sent to D before final disposition.

1. **Numerical-procedure effect versus allocation freedom.** The evidence table at initial line89 says a same-state S/J comparison identifies the effect of allowing separate coefficients. If S is solved analytically but J terminates suboptimally, observed differences combine allocation freedom with numerical reliability. Require numerically qualified/exact-enough solutions for the narrow freedom claim; otherwise call it a comparison of the implemented S/J procedures. The earlier A qualification discussion is good but the table should carry the condition directly.

2. **Two meanings of endpoint must not merge.** D's section3 concerns functionally evaluated endpoints from the same supervised reference and frozen TRAIN audit/candidate tuple. C's retrospective endpoint selector concerns complete final models after different training histories, chosen per exposed VAL patient. The latter's finite-family oracle is not an upper bound on retrained guard policies or subpatient routing; the former's empirical mean-risk ordering says nothing about the final-model oracle or unseen performance. Add this firewall explicitly when cross-linking round2 findings.

## Checks that passed

- Positive objective/constraint rescaling keeps the exact problem only when ridge, offsets, budgets, box and group weighting are transformed consistently. Finite precision behavior remains a separate empirical question.
- The scalar restriction uses `m*lambda`; feasible-set inclusion orders the **regularized** optimum, not necessarily unregularized or actual risk. This prevents a common wrong inference.
- A local scalar incumbent alone does not preserve scalar's accepted nonlinear endpoint. The proposed actual portfolio needs to retain/re-evaluate the specified scalar endpoint under the same actual measure.
- A minimum over a finite set that includes zero and feasible S/J endpoints orders only the selected observed **mean** risk. D correctly excludes componentwise superiority, future EMA trajectory ordering, patient Dice and population safety; ranking tolerance weakens the exact inequality accordingly.
- First-accepted endpoint and best over all tested scales are distinct policies/costs. D does not silently claim coverage of untested rays or scales.
- Fixed finite-class generalization needs candidate independence from an i.i.d. evaluation sample; the number of final retained candidates does not erase their adaptive TRAIN construction. Independent units are patients, not pixels or repeated queries.
- D frames the contribution as failure characterization/numerical qualification, without calling standard finite-set selection a new theorem or inferring actual patient prevalence from manufactured counterexamples.

## Reconciliation with C arithmetic

C's valid retrospective whole-patient oracle over block/full is exactly the full-model score because full exceeds block's stored foreground score for18/18 patients. All-four oracle adds0.1960pp over best constant. These quantities use VAL labels after the fact and are restricted to already produced complete predictors. They neither constrain a newly trained guard's achievable quality nor demonstrate that any TRAIN transaction detects/removes the observed RV errors.

C's marginal coupling intervals demonstrate that even the spatial overlap of endpoint errors is not fully identified by saved confusions. The algebraic same-hard-endpoint counterexample gives opposite audit Brier verdicts, but is explicitly not a proof of actual pseudo-gradient realization. D's evidence map is compatible with both restrictions.

## Resolution

D accepted both clarifications. Final inspected SHA256 is `4763ccae138763b9bfb39a65d8c2aafe228af8180c22c1b4b99b8332493045ef`. The revised evidence table explicitly requires exact or independently qualified comparable optimizer accuracy for the coefficient-freedom interpretation; analytic S versus stalled J is called an implemented-procedure comparison. Section3 now separates same-transaction endpoints from final trained-model whole-patient routing and disclaims retraining/subpatient/unseen-patient implications.

The added B counterexample paragraph retains its fixed-grid/tolerance and manually specified-direction limitations; it does not claim candidate-generator or patient prevalence. D owns its mathematical cross-check of that example, so C does not relabel it as independent patient/probe execution here.

Final disposition: **PASS for the stated narrow deterministic/evidence claim map.** No unresolved material finding remains. This is neither a novelty certification nor an implementation/patient-utility endorsement. No request to expand the claim or perform patient work was made.
