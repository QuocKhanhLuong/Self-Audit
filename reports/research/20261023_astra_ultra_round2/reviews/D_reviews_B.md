# D independent review of B's local-to-finite geometry

**Disposition: ACCEPT after clarification, within the stated manufactured-controller scope.** No patient efficacy, generated pseudo-CE prevalence or new-theorem claim is supported or requested.

Date: 2026-10-23. Initial report reviewed: `workers/B_geometry.md`, SHA256 `3621c4ddd934f720fbd49c684684415055489326a61e588d3bee2e6c7ba4fff8`. Final clarified report inspected: SHA256 `53224dc470d221cdf0f2db6d8a7cc49edd78b3005db71945eaec1c53a8d0863c`.

Evidence inspected: `evidence/B_geometry.json`, SHA256 `963d0c6f926a9724d83a93d375b78f42d0323a4cf0a0ceaadd3ca814e90c6162`; probe inspected: `probes/B_geometry.py`, SHA256 `fc4323bcdfafd59b122951933874174b995c5496f9464d68438e9b7b7ea11da6`. D checked that both recorded production-file hashes match the current files. D did **not** rerun the model probe. B's saved execution is distinguished from D's independent exact rational arithmetic in [D_B_review.json](../evidence/D_B_review.json).

## Main challenge: does the reversal depend on a defective solver?

No. D independently reconstructed the foreground response rows from the stated uniform softmax, labels and logit directions. Their averaged moments are

    mean A = [[1/96, -7/960], [-7/960, 17/3200]]
    mean b = (1/48, -1/120).

At J=(1,0), the regularized gradient is `(-1/96+lambda, 1/960)`. At lambda=1e-6 these signs certify the minimum of the convex objective over the entire box: the negative-gradient coordinate is at its upper endpoint and the positive-gradient coordinate at its lower endpoint. J also satisfies the classwise quadratics, so it is the unique constrained optimum. This argument needs no constraint qualification or multiplier recovery.

The aggregate scalar has numerator `1/80`, curvature `11/9600` and ridge `2*lambda`; its unconstrained optimum exceeds 1 and S=(1,1) is feasible, making S exact too. The local mean-risk gap is

    mean E(S) - mean E(J) = 71/19200 > 0.

The regularized gap is still larger by lambda/2. Thus the counterexample is not an artifact of comparing different penalties or conflating regularized and raw local metrics.

Along J, the true-class probability is `1/(exp(t)+1+2*exp(-t/2))`. Its denominator is strictly convex with its unique minimum 4 at t=0, so every nonzero positive step worsens the true-class risk. This establishes the nonlinear mechanism independently of the recorded solver result. The receipt records eight default-tolerance rejections, then unchanged reference state; S accepts at full scale and lowers each observed class risk. The unmodified production solver gives approximately (1,8.67e-19) and the same default rejection. Its agreement is supporting execution evidence, not the optimum proof.

## Qualifications checked

- **Zero-budget regularity:** the normal-cone condition is necessary and sufficient without CQ. The zero-budget example with feasible set {0} correctly shows why a true optimum may lack KKT multipliers. A present valid multiplier certificate remains sufficient in this convex problem. No generic KKT residual is silently used at singular feasible points.
- **Comparator penalty:** restricting alpha=a*1 changes the scalar ridge to m*lambda. B uses this in both the formula and executable comparator. Feasible-set inclusion orders regularized f; the separate consistent example shows why raw local mean risk need not have the same ordering.
- **Finite grid/tolerance:** strict worsening at every positive J scale does not imply rejection at every positive numerical tolerance. B correctly retains the default eight-point/tolerance claim and distinguishes the zero-tolerance argument. Near-optimal perturbations have their own tested-grid scope; sufficiently small safe steps may exist for them.
- **Realizability:** the probabilities, JVPs and production head guard are real finite softmax computations. The directions are manually supplied and are not shown to lie in the output family of make_candidates. The artifact therefore refutes an unrestricted controller implication, not a theorem restricted to the generated pseudo-CE family.
- **Execution isolation:** the inspected probe temporarily mocks only coefficient return when passing exact certified coefficients, verifies moments, restores the mock and separately invokes the unmodified solver. This is a disclosed mechanism test; it is not an end-to-end training run.
- **Portfolio:** minimum observed mean over a common-reference finite endpoint set is an elementary selection property. It does not order every class against each arm, preserve future trajectories, establish patient Dice or justify a population claim. B's valid-probability example and trial-cost accounting preserve these limits.
- **Novelty:** the variational condition, symmetry argument, set inclusion, remainder logic and finite-set minimum are standard facts. The useful result is their concrete separation under this controller's finite guard. No first-ever method/theorem claim follows from the construction.

## Challenge and reply ledger

| D challenge | B reply and inspected resolution | Status |
|---|---|---|
| Could local regularized improvement be mistaken for raw Brier improvement? | Section 2 supplies the exact penalty identity, a consistent default-ridge counterexample and the m*lambda comparator; the principal reversal improves both local metrics. | Resolved. |
| Does a finite affine softmax example establish occurrence for actual EMA/pseudo-CE candidates? | Provenance and sections 3–4 explicitly identify manually constructed directions, with no make_candidates-family or patient-frequency claim. | Resolved; actual candidate-family evidence remains uncollected. |
| Can local-gap/remainder ordering be transported to unequal returned scales? | Section 5 now explicitly compares local/actual risks at the same coefficient endpoints and requires recomputation at s_J*alpha_J and s_S*alpha_S after unequal backtracking. D inspected final lines 130–132. | Resolved. |

No blocking issue remains for the bounded report. A useful next scientific question is whether representative generated transactions exhibit the identified failure and whether correcting it improves a frozen policy at matched cost. The current proof and CPU receipts do not answer that question.
