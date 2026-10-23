# D reviews B — independent mathematical and novelty challenge

Reviewer: D / astra_prior_art
Date: 2026-10-23
Target: [B_theory.md](../workers/B_theory.md)
Reviewed draft SHA256: bb321c6cc3a75b881ae3a870f73f8d3f792285b81e0190e88d079e82f969a78d
Source anchor: 637f38ced217fe1db2347a87924695533056d8f0

## Verdict

**Accept the mathematical scope, with the three wording/control clarifications below. No blocking mathematical error found.** This is read-only reasoning and source review; D did not rerun B's probes and does not independently claim their execution receipts.

The strongest useful conclusions are that numerical feasibility is different from useful allocation, that zero-budget tangency can prevent every finite nonzero move, and that decreasing every observed balanced Brier group need not improve even same-audit hard Dice. These are valid limitations to emphasize. They are diagnostic facts and examples, not evidence of a new optimization theory or patient benefit.

## Independent checks

- The quadratic expansion and restricted Gauss–Newton interpretation agree with the source response moments. Positive ridge yields a unique mathematical minimizer on the nonempty compact feasible box; numerical SLSQP success does not certify that minimizer.
- The zero-budget ray formula supports the stated strict/weak-alignment distinctions. The two-group singleton feasible-set example does fail KKT stationarity: at zero, the constraining quadratic has zero derivative while remaining active gradients cannot balance the negative objective derivative. Thus a generic “log KKT residuals” demand needs qualification.
- The finite softmax obstruction is valid: the correct-class denominator is strictly larger than four for nonzero displacement because the four logit changes sum to zero and are not equal. Shrinking a step does not give strict feasibility when the first derivative vanishes and curvature has the harmful sign.
- The Brier/Dice construction demonstrates a logical nonimplication. Its manually constructed candidate is clearly distinguished from an unlabeled-CE-generated direction, so it does not purport to estimate active-pipeline failure frequency.
- The FNR/FPR constants agree with the code's half-Brier and half-positive/half-negative normalization for the stated single-slice populations. B correctly avoids turning a risk decrease into monotone error-count improvement.
- Balanced squared-loss properness analysis has the right population optimizer. It is a counterexample to general calibration claims, not a fitted model of the full patient-specific multiclass audit.
- The aggregate-scalar restriction correctly requires m*lambda. Direct audit-gradient control changes candidate family and differs from supervised-extra's CE+Dice/full-scope/AdamW recipe; B labels both differences.
- Audit reuse defeats an independence argument. Within-transaction patient separation does not establish unseen-risk, volume-wide, cumulative-trajectory, EMA or Dice protection.

## Challenges sent to B and resolution

| ID | Challenge | B response | Resolution |
|---|---|---|---|
| D-B-1 | Explicitly classify Gram/Gauss–Newton reduction, convexity, finite acceptance invariant, compensation and scalar restriction as standard optimization facts or diagnostic examples; do not let new counterexamples imply first-ever theory. | B accepted and agreed to link D prior-art audit and add explicit no-novelty language. | RESOLVED in revised file; directly verified. |
| D-B-2 | “First-order sign cone” could be mistaken for the repository's aggregate sign arm; specify the full classwise derivative-feasibility cone. | B accepted this exact distinction. | RESOLVED in revised file; directly verified. |
| D-B-3 | Aggregate-scalar and direct-audit controls can themselves legitimately reject. Protection alone is not success; compare useful progress, label use and compute. | B accepted and agreed to state useful-progress/label/compute requirements. | RESOLVED in revised file; directly verified. |

The novelty challenge is supported by [GEM, NeurIPS 2017](https://papers.neurips.cc/paper/2017/file/f87522788a2be2d171666752f97ddebb-Paper.pdf), [OGD, AISTATS 2020](https://proceedings.mlr.press/v108/farajtabar20a/farajtabar20a.pdf), [CPO, ICML 2017](https://proceedings.mlr.press/v70/achiam17a/achiam17a.pdf), and [DS3L, ICML 2020](https://proceedings.mlr.press/v119/guo20i/guo20i.pdf). They are collisions at different levels, not an assertion that current code is equivalent to any one of them. The detailed mechanism distinctions are in [D_prior_art.md](../workers/D_prior_art.md).

## Reciprocal correction to D's own report

B's constraint-qualification example exposed an overbroad test suggestion in D's draft. D revised its requirement to use KKT residuals only where meaningful multipliers/qualification exist, with reference-optimum gaps or feasible-control comparisons otherwise. D also adopted B's exact m*lambda scalar restriction. This reciprocal correction is recorded because review should change conclusions when warranted.

## Revision verification

Verified revised B_theory.md SHA256 e8556f1abb1b2748f7db359f646a5efc250033729cdd221a1e4dc0bedcb020ff. Directly read the new non-novelty paragraph (line 28), classwise-cone distinction (line 56), and legitimate-rejection/progress paragraph (line 121), plus B's response ledger. All D-B-1 through D-B-3 are resolved. Final verdict: ACCEPT AS A BOUNDED THEORY/FAILURE-MODE AUDIT; no empirical efficacy or publication-novelty clearance follows.
