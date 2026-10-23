# B — joint-guard mathematical audit

Status: **the finite TRAIN-audit guard is coherent within its declared scope; allocation benefit is unestablished, and a numerical allocation defect must be resolved before interpreting method comparisons.** No new module is recommended. The next justified work is a numerical repair with matched controls, not an efficacy claim.

## Provenance and execution scope

- Worker: `/root/astra_theory`, with an independent algebra challenge from child `/root/astra_theory/rank_audit`.
- Read current `AGENTS.md`, `ACTIVE_PIPELINE.json`, `docs/METHOD.md`, `docs/EXPERIMENTS.md`, and the actual control/training implementation.
- Audited worktree: `/Users/alvinluong/Self-Audit-research-20261023`, source commit `637f38ced217fe1db2347a87924695533056d8f0`. Compared the control/theory paths to experimental `8960b6baac4c6600553d900c303f192c2fcfa4f0`. The experimental branch adds image-validity masking to candidate selection; the quadratic solve and finite acceptance logic in these files are unchanged.
- Used the scientific-critical-thinking skill for evidence separation. Earlier Astra memory supplied workflow/claim-boundary context only; the old no-GT mechanism is not the object of these proofs.
- Executed only manufactured float64 CPU probes. No patient images, patient labels, pretrained weights, training jobs, downloads, GPU execution, production edits, commits or pushes.
- Probe: [`../probes/B_counterexamples.py`](../probes/B_counterexamples.py). Receipt: [`../evidence/B_counterexamples.json`](../evidence/B_counterexamples.json). Receipt includes exact source hashes and package versions: Python 3.11.16, Torch 2.14.0, NumPy 2.4.6, SciPy 1.17.1.

## Findings that change the research decision

| Finding | Status | Consequence |
|---|---|---|
| A feasible exact finite guard bounds only the evaluated extra step on the present TRAIN audit measure | Proven from acceptance rule, subject to numerical evaluation | Preserve this narrow claim; it is not a patient-Dice or generalization theorem |
| SLSQP can return `NUMERICAL_FEASIBLE` at zero despite a substantially better feasible scalar solution | Reproduced with actual affine-head probability Jacobian | Repair/validate allocation before treating a stalled joint arm as evidence against its mathematical objective |
| Every foreground audit Brier risk can improve while same-audit hard RV Dice worsens | Executed counterexample, joint guard accepts | The limitation is intrinsic to the endpoint, not just unseen-patient distribution shift |
| A locally accepted direction can harm a class for every nonzero finite scale | Proven and executed for an exact affine softmax head | Backtracking does not guarantee useful motion without stronger descent/slack assumptions |
| Balanced one-versus-rest squared risk is generally not a proper posterior-calibration objective | Derived | Do not interpret risk or its decrease as calibrated correctness or pseudo-label truth |
| Independent per-source allocation does not replace an aggregate scalar control | Derived comparison design | Add aggregate direction + the same exact guard, with the correct restricted ridge penalty |
| PSD of each supplied Gram does not ensure valid squared-response moments | Executed malformed-input example; internal producer is consistent | Tighten the standalone solver contract if it is exposed/reused; this is not evidence of an active pipeline failure |

## 1. What is mathematically sound

The Gram/Gauss–Newton reduction, convexity argument, finite acceptance invariant, compensation examples and scalar restriction below are established optimization facts or illustrative diagnostics applied to this implementation. They are not claims of novel theory. The separate [D prior-art audit](D_prior_art.md) locates the relevant gradient-constraint, function-response, constrained-step and safe-SSL literature; novelty requires a narrower demonstrated distinction and matched empirical value.

At a fixed supervised reference state `theta_s`, fixed audit labels/weights, and fixed candidate matrix `D=[d_1,...,d_m]`, define

`r_c = sqrt(W_c)(y_c-p_c(theta_s))`, `V_c = sqrt(W_c) J_c D`.

Then `E_c(alpha)=1/2 ||r_c-V_c alpha||^2 = e_c-b_c^T alpha+1/2 alpha^T A_c alpha`, with `A_c=V_c^T V_c`, `b_c=V_c^T r_c`, `e_c=||r_c||^2/2`. This is exactly the construction in [`self_audit/control.py:120-129`](../../../../self_audit/control.py#L120). It is a linearized probability-response model, not a second-order Taylor expansion of the original nonlinear risk: it retains the Gauss–Newton term and omits residual-weighted output curvature.

For nonnegative budgets, `alpha=0` is feasible. The box is compact; every `E_c` is convex. A positive ridge makes the objective strongly convex with Hessian `mean_c A_c + lambda I`, hence the mathematical minimizer exists and is unique. Rank deficiency does not invalidate this result. The source's careful qualification that SLSQP supplies a checked numerical candidate rather than an exact optimizer is correct ([`docs/METHOD.md:66-80`](../../../../docs/METHOD.md#L66), [`control.py:22-61`](../../../../self_audit/control.py#L22)).

The one-direction solution in `scalar_allocation` is an exact specialization for consistent moments: intersect the groupwise feasible upper bounds with `[0,1]`, then clip `mean(b)/(mean(a)+lambda)` to that interval. With negative alignment its rationalized root avoids subtractive cancellation ([`control.py:64-80`](../../../../self_audit/control.py#L64)). Clipping coordinates of the multivariate unconstrained ridge solution is not an equivalent joint solve.

Candidate normalization is also consistent: each class uses the same accepted-pixel denominator, so summing frozen directions equals one ordinary SGD pseudo step on that same block ([`control.py:132-159`](../../../../self_audit/control.py#L132)). It does not equal a subsequent AdamW step, and the active implementation correctly commits the selected displacement directly.

### Rank, conditioning and moment consistency

The common nullspace is `K=intersection_c ker(V_c)`. Directions in K change none of the local risks or alignments. Ridge chooses a minimum-norm coefficient representative. Individual rank-deficient groups need not imply a rank-deficient aggregate: `V_1=[1,0]`, `V_2=[0,1]` already give positive-definite `mean(A)`.

For stacked normalized response matrix V, the objective condition number is `(sigma_max(V)^2+lambda)/(sigma_min(V)^2+lambda)`. This controls curvature conditioning, but does not make SLSQP's absolute stopping rules invariant to objective scale. Well-conditioned scalar problems can still stop at zero.

A necessary and sufficient algebraic consistency condition for arbitrary supplied moments is that the augmented matrix `[[A_c,b_c],[b_c^T,2e_c]]` be PSD. It is a Gram matrix of `[V_c,r_c]`; conversely any PSD matrix admits a Gram factorization. In particular, `b_c` must be orthogonal to `ker(A_c)`, and `b_c^T A_c^+ b_c <= 2e_c`. Current validation checks only symmetry/PSD of A and nonnegative e. The probe supplies `A=0,b=1,e=.5`, which passes and yields a claimed local squared risk `-.5`. This is an API-contract gap, **not** evidence that `response_moments` generates such malformed inputs.

## 2. The zero-budget boundary is a real constraint, not automatically a numerical bug

Along a nonnegative coefficient ray `alpha=t u`,

`E_c(tu)-E_c(0) = -t b_c^T u + t^2 ||V_c u||^2/2`.

Thus, with zero budget: positive `b_c^T u` permits a sufficiently small interval; negative alignment forbids every positive t; zero alignment permits the ray only if `V_c u=0`. The full **classwise derivative-feasibility cone**, defined by `b_c^T u>=0` for every c, is therefore too permissive at a tangent boundary. This mathematical cone is not the repository's aggregate `sign` arm. For example, `V=I,r=(-1,0)` has a first-order admissible ray `(0,t)`, but its quadratic risk increases by `t^2/2`; the feasible set in the nonnegative box is only zero.

Slater/KKT regularity is not guaranteed either. With one coefficient, `V_1=V_2=1`, `r_1=1,r_2=0`, and zero budgets, group 2 forces `alpha=0`. It is the unique optimum, yet the group-2 constraint has zero gradient there and the objective derivative is negative; no nonnegative inequality multipliers satisfy stationarity. This does not contradict existence/uniqueness. It rules out silently invoking a generic constraint qualification or treating missing/ill-conditioned multipliers as proof of infeasibility.

## 3. Actual finite safety and its sufficient assumptions

The head path evaluates `theta_s + s sum_j alpha_j d_j` with the actual affine-head softmax; decoder scope uses a functional call with cloned buffers. Acceptance requires finite measured risks, each class at or below reference plus budget/tolerance, and nonincreasing mean risk up to tolerance ([`control.py:177-228`](../../../../self_audit/control.py#L177), [`decoder_control.py:48-83`](../../../../self_audit/decoder_control.py#L48)). A rejected head trial does not mutate live head parameters; a successful trial is copied once. The engine updates EMA only after the final student state ([`engine.py:163-192`](../../../../self_audit/engine.py#L163)).

The resulting guarantee is a direct acceptance invariant. It requires the audited function, inputs, masks, labels, weights, parameter block, denominator and reference state to remain fixed; deterministic prediction; no hidden optimizer transform after evaluation; exact matching of the committed and evaluated state; and no mutation of optimizer/EMA/buffer/RNG state by rejected trials. Native deterministic eval modules and these transaction paths support the intended contract. `eval()` alone would not prove determinism or absence of side effects for an arbitrary future custom module. Numerical acceptance bounds computed risks; it does not establish an exact-real-arithmetic inequality beyond the declared numerical tolerance.

If the weighted probability remainder is R, the exact-vs-local risk difference is `-<r-V alpha,R> + ||R||^2/2`, whose absolute value is bounded by `||r-V alpha|| ||R|| + ||R||^2/2`. The documented bound is correct. It is not a nonzero-step existence theorem.

**Executed finite-step obstruction.** For one audit pixel with class-1 truth, start from equal probabilities `(1/4,1/4,1/4,1/4)` and an exact head-logit direction `(1,0,-1/2,-1/2)`. The correct-class derivative is zero, while both other foreground probabilities decrease. The local QCQP chooses `alpha≈1`. Yet

`p_1(t) = 1/(exp(t)+1+2 exp(-t/2)) < 1/4` for every nonzero t,

by strict convexity of exp and the zero mean of the four logit changes. Correct-class risk therefore increases for every positive finite scale. All 12 actual trials reject at zero tolerance; the head remains bitwise unchanged. At the default eight-trial minimum scale `1/128`, the class-1 risk increase is approximately `2.149e-6`, also above the default `1e-8` tolerance. No more accurate JVP or more local optimization rescues this ray.

A sufficient positive-step condition is strict negative directional derivative for every active zero-budget group and for the mean risk, with continuous derivatives near the reference. Inactive positive-budget constraints admit a neighborhood by continuity. This yields some positive feasible radius; a **finite** predetermined backtracking grid still may not reach it. With weak/zero directional derivatives, higher-order signs matter. Positive tolerances may eventually accept small harmful motion; that is the stated approximate invariant, not strict descent.

## 4. Brier improvement does not even guarantee same-audit Dice improvement

The four manufactured pixels have truth `[RV,RV,BG,BG]`. Initial probabilities are:

| Channel | Pixel 1 | Pixel 2 | Pixel 3 | Pixel 4 |
|---|---:|---:|---:|---:|
| BG | .46 | .46 | .88 | .88 |
| RV | .52 | .52 | .10 | .10 |
| MYO | .01 | .01 | .01 | .01 |
| LV | .01 | .01 | .01 | .01 |

The proposed endpoint has RV `[.80,.48,.10,.10]`, MYO=LV=`.005`, and BG equal to the remaining mass. It is exactly represented by a finite 1x1 head displacement on four one-hot features. `joint_guard` chooses `alpha=1, scale=1`; all three measured risks fall:

`(.0601,.00005,.00005) -> (.0413,.0000125,.0000125)`.

Pixel 2 changes from correct RV to BG, so RV Dice falls from **1 to 2/3**. This probe uses a manually constructed candidate direction, not a generated unlabeled CE gradient; its purpose is to falsify a logical implication of the guard, not estimate failure frequency of the candidate generator.

Some absolute bounds are possible but weak. For a present class with both positive and negative audit pixels, a false negative implies `p_c<=1/2` and a multiclass argmax false positive implies `p_c>=1/4`. For the exact present-class weighting in the code this gives `FNR_c<=16 R_c` and `FPR_c<=64 R_c`, clipped at 1. For an absent class, `FPR_c<=32 R_c`. These bounds refer to that fixed slice and the corresponding pixel population; a batch-averaged risk is not a per-patient bound without its normalization factor. They do not convert a decrease in R into a decrease in either error count. Dice further depends on foreground prevalence. Native patient/frame endpoints remain indispensable.

### Class balancing is a task-specific surrogate

The risk places half its mass on positives and half on negatives when both exist ([`control.py:83-117`](../../../../self_audit/control.py#L83)). In a binary population with prevalence pi and posterior eta(x), the minimizer of this class-balanced squared risk is

`p*(x) = (1-pi) eta(x) / [(1-pi) eta(x) + pi (1-eta(x))]`,

not eta(x), except in special cases such as pi=1/2. For an uninformative feature with pi=.1, the true posterior .1 has balanced half-Brier .205; p=.5 has .125. Ordinary prevalence-weighted half-Brier instead rises from .045 to .125. This is not a defect if the intended objective is balanced error control. It prohibits interpreting the balanced score as posterior calibration, source correctness, or independent teacher truth. The actual multiclass simplex and patient-specific weights add further coupling; the binary derivation is a precise counterexample to any general properness claim.

## 5. The solver issue is about useful motion and valid comparisons, not a safety breach

The source uses SLSQP from zero with absolute `ftol=1e-12`, then checks feasibility and improvement against zero ([`control.py:55-61`](../../../../self_audit/control.py#L55)). Such checks do not detect a false stall at zero. SciPy documents that ftol enters checks on objective changes and step size as well as optimality/violation criteria ([official SLSQP documentation](https://docs.scipy.org/doc/scipy/reference/optimize.minimize-slsqp.html); online documentation was v1.18.0, while the executed environment is v1.17.1).

The independent B probe uses actual probabilities and affine-head derivatives, not arbitrary incompatible moments. At uniform four-class probabilities and true class 1, take logit direction `h*(-1,3,-1,-1)`. For `h=1e-6`, the exact scalar solution is `.2291666141493176`, while the joint routine returns **0, `NUMERICAL_FEASIBLE`**. The attainable objective gain is `2.625867454e-8`; the returned gain is zero. At h from `1e-5` through `1e-2`, both routines choose 1 in the same probe. A's separate CE-derived head probe independently reaches the same qualitative failure.

This is a real software/interpretation problem, but does **not** show every patient update fails, and does not invalidate the finite guard. Its magnitude/frequency on real directions remains unmeasured. It can affect `joint_guard`, `joint_no_class`, and `joint_no_guard` while the exact `independent_guard` scalar path is unaffected by this stopping mechanism, confounding a comparison attributed to joint interactions. Feasible-zero returns must be distinguished from a mathematically optimal zero or an actual nonlinear rejection.

Minimum numerical repair requirements, without prescribing a production change in this report: scale objective/constraints consistently while preserving the same optimum; compare the m=1 case over relevant dynamic ranges against the closed form; retain fixed-source multi-direction tests with degenerate and near-collinear responses; verify feasibility in original units; log a meaningful optimality gap or an independent comparison to known feasible controls. Rejecting all updates is safe but cannot be called useful allocation.

The existing `pseudo_accepted` counter increments when `scale>0` ([`engine.py:181,190`](../../../../self_audit/engine.py#L181)). A meaningful utility receipt should additionally log effective committed parameter/response norm and objective gain: positive scale alone does not prove a representable nonzero parameter change or learning benefit.

## 6. Minimum controls and parameterization caveats

The essential missing baseline restricts the joint coefficient vector to `alpha=a*1`, `a in [0,1]`. It uses the same reference, teacher, accepted mask, denominator, candidate directions, scope and pseudo LR. Its moments are `a_c=1^T A_c 1`, `b_c=b_c^T 1`, and it applies the same exact finite class guard. To be literally the restricted joint problem, its scalar ridge is **m*lambda**, because `||a*1||^2=m*a^2`. Using lambda instead is a legitimate separately named tuning choice, not an exact restriction. This control asks whether independent class allocation beats selecting one step size for the ordinary aggregate pseudo direction. It is absent from the nine named arms ([`docs/EXPERIMENTS.md:41-49`](../../../../docs/EXPERIMENTS.md#L41)).

A second informative control uses `d_audit=-eta grad_B mean_c R_c(theta_s)` and the same exact finite class guard on the same scope. It measures the value of spending the same audit labels on a direct update. This changes the candidate family and should be labeled as such. `supervised_extra` is not equivalent: it uses CE+Dice, the full supervised scope, clipping and body AdamW ([`engine.py:62-73,159-161`](../../../../self_audit/engine.py#L62)). Label access and actual compute both need accounting.

Both proposed controls can legitimately return zero or fail every finite trial. Neither is guaranteed to improve every class or produce useful motion. Compare realized progress, effective displacements, patient endpoints and label/compute use; satisfying a protection rule by rejecting everything is not a success criterion for any arm.

Coefficient ridge is representation-dependent. Duplicating a column lets a total coefficient s split across k copies, reducing its minimum ridge penalty from `lambda*s^2/2` to `lambda*s^2/(2k)` and enlarging the box-representable total range. Unique class grouping prevents deliberate duplication, but semantic near-collinearity remains. Scaling candidates by h changes A by h^2 and b by h while lambda stays fixed. In an unconstrained, unsaturated scalar regime, `alpha=h*b0/(h^2*a0+lambda)`, so the committed displacement is proportional to `h^2/lambda` as h approaches zero. Pseudo LR and ridge are therefore coupled even with a perfect solver. This is distinct from SLSQP's false stall. Do not compare different normalizations or scopes and attribute the result solely to the joint allocation mechanism.

## 7. Audit adaptation and the strongest defensible claim

Audit labels are TRAIN optimization data. A different patient within one transaction does not make the audit independent of the model: previous steps can have trained on that patient, and earlier audit decisions affect the current state and EMA. Choosing alpha and testing several scales on these labels is adaptive optimization. The source correctly states this ([`docs/METHOD.md:131-142`](../../../../docs/METHOD.md#L131)).

There is no distribution-free unseen-risk implication. A head can have an audit-visible feature and an unseen-only feature; an update can improve every audit prediction while arbitrarily worsening the unseen feature's predictions. Identical behavior on the observed support cannot distinguish such models. A fresh independent patient sample and a predeclared uniform-complexity/stability or testing argument would be additional assumptions, not consequences of the deterministic guard. Millions of correlated pixels cannot replace patient-level sampling units.

The correct claim remains: **the accepted extra displacement satisfies the configured measured TRAIN group-risk limits relative to its own supervised reference.** It does not bound the supervised step, other slices of the same patient, the future AdamW trajectory, EMA predictions, cumulative rotating-audit debt, unobserved patients, external domains, hard Dice or clinical outcomes. The documentation already makes most of these limits explicit; the new counterexamples show why they are necessary.

## 8. Explicit kill criteria

1. **Numerical stop:** Do not interpret a joint-vs-control efficacy experiment until the small-scale scalar mismatch is fixed or demonstrated irrelevant by an independently checked moment/solver audit. Manufactured fixes alone do not establish patient benefit.
2. **Mechanism stop:** If matched aggregate-scalar+finite-guard and direct-audit controls explain the gains, drop the claim that classwise joint allocation is the causal contribution. Do not add modules to rescue the interpretation.
3. **Utility stop:** If a repaired method mainly returns zero or tiny unrepresentable updates, report a safe but ineffective policy. Separate no accepted pseudo pixels, exact/local infeasibility, numerical stalls and finite-trial rejection.
4. **Endpoint stop:** Better TRAIN Brier with worse native patient Dice/FP/FN is a negative result, not partial proof of anatomy improvement. Freeze evaluation and retain subgroup/absent-class outcomes.
5. **Protocol stop:** A change in label budget, denominator, audit weights, batch/crop sampling, parameter scope, optimizer convention or finite guard is a new experimental factor. Repeated exposed-VAL tuning is development, not confirmation.

## Commands and receipts

All shell commands used the required `rtk` prefix. Key reproducible command:

```sh
rtk proxy python reports/research/20261023_astra_ultra/probes/B_counterexamples.py
```

Result: **PASS**, four probe families, five physical scale cases. Source comparison used:

```sh
rtk proxy git diff 637f38ced217fe1db2347a87924695533056d8f0 8960b6baac4c6600553d900c303f192c2fcfa4f0 -- self_audit/control.py self_audit/decoder_control.py docs/METHOD.md
```

No patient/GPU execution was performed. Full repository checks are coordinated centrally by the parent; this worker does not claim their result.

## Independent review response

D reviewed draft SHA256 `bb321c6cc3a75b881ae3a870f73f8d3f792285b81e0190e88d079e82f969a78d` and accepted the algebra/claim limits subject to three clarifications. B accepted all three: (1) explicitly marked standard optimization facts and diagnostic examples as non-novel and linked D's prior-art audit; (2) distinguished the full classwise derivative-feasibility cone from the repository's aggregate `sign` arm; (3) stated that both additional controls can legitimately reject and must be judged on useful progress with label/compute accounting. These are report clarifications; no production changes or patient claims result.
