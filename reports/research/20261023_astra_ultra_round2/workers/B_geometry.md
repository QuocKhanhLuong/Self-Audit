# B — local joint geometry versus actual finite-guard outcomes

**Answer to the round-two question: yes.** From the same finite affine softmax head, audit labels and two fixed candidate directions, the unique exact local joint optimum can fail every finite guard trial while the exact aggregate scalar comparator succeeds and strictly lowers every observed class risk. This reversal does not require a numerical solver defect. It is a controller-level manufactured counterexample, not a generated pseudo-CE example or evidence about its frequency on patient data.

## Provenance and scope

- Worker `/root/astra_theory`; independent algebra challenge from `/root/astra_theory/rank_audit`; early independent fraction/gradient check by D (`/root/astra_prior_art`).
- Research HEAD `ccfde9b`; production source `637f38ced217fe1db2347a87924695533056d8f0`; experimental comparison pin `8960b6baac4c6600553d900c303f192c2fcfa4f0`. Worktree `/Users/alvinluong/Self-Audit-research-20261023`.
- Previous reports and production files were preserved. Only the owned round-two report/probe/evidence paths were written. No patient/GPU execution, pretrained weights, downloads, commits or pushes.
- Executable: [`../probes/B_geometry.py`](../probes/B_geometry.py). Receipt: [`../evidence/B_geometry.json`](../evidence/B_geometry.json), including probe and production-file SHA256, Python/Torch/NumPy/SciPy versions.
- The finite examples use genuine softmax probabilities, production `response_moments` and the production `controlled_head_step` acceptance/commit logic. Candidate directions are **manually constructed**, not obtained from `make_candidates`. Exact coefficients are supplied through a temporary in-process solver mock, removed after each call; the reversal is also reproduced with the unmodified production SLSQP solver. These experiments establish a mathematical/controller distinction, not a restricted pseudo-CE-family theorem.
- The geometry, normal-cone condition and selection facts below are standard convex-analysis and evaluation arguments, not claimed new theorems or publication novelty. See the [round-one prior-art audit](../../20261023_astra_ultra/workers/D_prior_art.md) and round-two D claim review.

## 1. Exact aggregate-ray comparator and a necessary-and-sufficient criterion

Write `E_c(alpha)=e_c-b_c^T alpha+alpha^T A_c alpha/2`, `A_c=V_c^T V_c`, and

`F={alpha in [0,1]^m : E_c(alpha)<=e_c+epsilon_c for every c}`,

`f(alpha)=mean_c E_c(alpha)+lambda ||alpha||^2/2`, with `lambda>0` and nonnegative budgets. The exact joint optimum is J. The restricted scalar comparator S has `alpha=a*1` and therefore **scalar ridge m*lambda**, not lambda. This preserves the same objective, directions, accepted-pixel denominator and coefficient box. The implemented moment and scalar formulas are at [`control.py:64-80,120-129`](../../../../self_audit/control.py#L64).

Let `h_c=1^T A_c 1`, `k_c=b_c^T 1`. The scalar feasible set is `[0,tau]`, where tau is the minimum of 1 and each nontrivial nonnegative quadratic root. Consistent moments imply `h_c=0 => k_c=0`. For `h_c>0`, the upper root is `(k_c+sqrt(k_c^2+2h_c epsilon_c))/h_c`, with a cancellation-safe equivalent when k is negative. Thus

`a_S=clip(mean(k)/(mean(h)+m*lambda),0,tau)`.

Set `x_S=a_S*1`, `g=grad f(x_S)`. Then **S equals J if and only if**

`g^T(x-x_S)>=0 for every x in F`,

equivalently `-g` belongs to the normal cone of the **actual feasible set** F at x_S. This is the variational inequality for convex differentiable minimization and needs no constraint qualification. The scalar optimum only checks feasible motion along the common ray; it says nothing about transverse feasible directions.

A strict-benefit witness is any feasible x with `g^T(x-x_S)<0`: convexity keeps the segment feasible and a sufficiently small movement strictly improves f. Conversely, if J differs from S, strict convexity gives `f(J)<f(S)` and J itself supplies such a witness. This is a necessary-and-sufficient characterization, though checking every feasible direction is not generally a cheap numerical certificate.

Under a suitable constraint qualification, the normal-cone condition can be expanded into nonnegative multipliers for the active group quadratics and box faces. Such KKT multipliers remain sufficient for optimality in this convex problem even without a constraint qualification, but need not exist at a true optimum. Example: one coefficient, `q_1(a)=a^2/2-a`, `q_2(a)=a^2/2`, zero budgets. The second constraint forces a=0, yet stationarity would require `-1/2-mu_1-lower_multiplier=0`, which is impossible with nonnegative multipliers. A box-only projected-gradient test is not a general certificate for the quadratic feasible set. A's round-two solver audit is the appropriate place for numerical certification; this report does not silently replace it with a generic KKT tolerance.

Useful special cases:

- If f and F are invariant under a transitive coordinate-permutation group, averaging any feasible point over the group stays feasible and cannot increase f. Uniqueness then forces J to have equal coordinates: S=J. This is a real no-benefit condition, not an assumption that joint allocation always helps.
- If F itself lies on the common ray, the scalar problem already is the full problem.
- If S is interior to both box and all quadratic inequalities, S=J exactly when every component of g is zero. The scalar first-order condition only gives `sum(g)=0`; any nonzero transverse gradient supplies an improving feasible direction.
- A whole-box certificate can bypass active-constraint regularity: if J is box-optimal and also satisfies the class constraints, it is globally optimal over F. The main counterexample uses this stronger, directly checkable certificate.

## 2. Objective inclusion is not automatically mean-Brier inclusion

Feasible-set inclusion proves `f(J)<=f(S)`. It does **not** generally prove `mean E(J)<=mean E(S)`, because the coefficient penalty can be different. Precisely,

`mean E(J)-mean E(S) = f(J)-f(S) + lambda (||S||^2-||J||^2)/2`.

The excess is at most `m*lambda/2`, using the box bounds and exact objective inclusion. This makes the possible discrepancy small in absolute local-risk units at the default ridge, but does not make the ordering logically valid.

An executed consistent Gram example at the default `lambda=1e-6` has `J=(.64,.5)`, `S=(.605,.605)`,

| Quantity | J | S |
|---|---:|---:|
| Mean local squared risk | 1.6596e-7 | 1.44435e-7 |
| Regularized objective | 4.9576e-7 | 5.1046e-7 |

The receipt contains explicit V and r; the full unconstrained/box optimum is feasible for the zero-budget group. This is a small manufactured quadratic case, not a patient estimate. Every comparison below reports which of these two quantities is being ordered.

## 3. Exact softmax reversal from the same state and directions

Use one audit pixel with class-1 truth and uniform four-class probabilities. A zero affine 1x1 head on feature 1 realizes this exactly. Let the two candidate **logit** changes be

`d1=(1,0,-1/2,-1/2)`, `d2=(-4/5,1/10,7/20,7/20)`.

These mean-zero logit directions give probability derivatives dz/4 at the reference. The three foreground weighted residual/response rows are

`r=(3/4,-1/4,-1/4)`,

`V = [[0,1/40],[-1/8,7/80],[-1/8,7/80]]`.

Here each row denotes its own class group with a single observed pixel; absent foreground groups measure false positives. The initial risks are `(9/32,1/32,1/32)`. Use zero budgets and `lambda=1e-6`.

### Local joint optimum is exact, not merely numerically plausible

The averaged moments satisfy

`mean(b)=(1/48,-1/120)`,

`mean(A)=[[1/96,-7/960],[-7/960,17/3200]]`.

At `J=(1,0)`, the objective gradient is `(-1/96+lambda,1/960)`. Its first component is negative at the upper box face; its second is positive at the lower face. For every point in the box, the gradient's inner product with displacement from J is nonnegative. Convexity certifies global box optimality, and positive ridge makes it unique. J is classwise feasible: local risks are `(9/32,1/128,1/128)`. No KKT regularity assumption is needed.

For S, the aggregate direction is `(1/5,1/10,-3/20,-3/20)`. Its mean alignment is `1/80`, mean squared response is `11/9600`, and scalar ridge is `2lambda`. The unconstrained scalar optimum exceeds 1 and every group permits a=1, so **S=(1,1)**.

| Local quantity | J | S |
|---|---:|---:|
| Mean local risk | 19/192 = .098958333333 | 657/6400 = .10265625 |
| Regularized objective | .098958833333 | .10265725 |

J beats S in **both** local quantities here. The raw local gap is `71/19200`; this reversal is not explained by the ridge-ordering caveat.

### Yet the finite guard rejects J and accepts S

Along J, the actual correct-class probability is

`p_1(t)=1/[exp(t)+1+2exp(-t/2)] < 1/4` for every nonzero t.

Strict convexity of exp proves the inequality because the four logit shifts have mean zero and are not all equal. Hence the correct-class risk increases for every positive finite step. The correct-class **first-order** response along J is zero; adverse curvature is omitted by the local probability model.

The production guard at its defaults, eight backtracks and tolerance `1e-8`, rejects every tested J scale, including `1/128`; that last trial's class-1 excess is approximately `2.149e-6`. The zero-tolerance twelve-trial check also rejects all scales. At exact J, the analytical conclusion holds for every positive real scale, not just this finite grid. Positive tolerance with a sufficiently extended grid can eventually permit a tiny harmful step; that would not be strict classwise improvement.

S is accepted at full scale by the same source guard:

| Actual risk | Retained J outcome | Accepted S outcome |
|---|---:|---:|
| Class 1 | .28125 | .264251973494 |
| Class 2 | .03125 | .022604914953 |
| Class 3 | .03125 | .022604914953 |
| Mean | .114583333333 | .103153934467 |

The retained J head is bitwise unchanged; no RNG change occurs. Thus a locally superior J can yield a strictly worse **post-guard current-audit outcome** than S. The unmodified production solver independently returns `(1,8.67e-19)` and reaches the same rejection, so a perfect local numerical repair would not remove this geometry.

This is not a fragile equality-only effect over the prescribed finite grid. Feasible perturbations `alpha=(1,a2)` with `a2=1e-5,1e-4,1e-3,.005` have regularized objective gaps from the exact optimum of approximately `1.04e-8,1.04e-7,1.04e-6,5.27e-6`; all also reject every default trial. For these perturbed directions a sufficiently small positive step can become safe because the correct-class first derivative is now favorable. The claim for them is therefore limited to the tested grid.

## 4. Genuine equality and strict-benefit cases also exist

The report does not infer that S always wins.

- **No local benefit:** `V=I`, `r=(.4,.4)`, `lambda=.1` gives `J=S=(4/11,4/11)` by symmetry. An exact finite-head counterpart uses two one-hot pixel features and two distinct directions, each applying `(-.1,.3,-.1,-.1)` to only one of the two equally weighted class-1 pixels. The complete objective/constraints are symmetric under exchanging pixels and directions. J=S=(1,1), and both source-guard endpoints agree exactly. The directions are distinct rather than duplicated copies of one column.
- **Strict local benefit:** `V=I`, `r=(.9,.1)`, `lambda=.1` gives `J=(9/11,1/11)` and `S=(5/11,5/11)`, with regularized objectives `.0372727273` versus `.1827272727`.
- **Strict actual finite benefit:** On the same uniform four-class one-pixel head, use `d1=(-.1,.3,-.1,-.1)`, `d2=(.3,-.1,-.1,-.1)`. The exact box certificate gives J=(1,0), S=(1,1); both pass at full scale. Actual mean risk is `.090864833963` for J versus `.095240810926` for S. This is again an illustrative manual-direction case, not efficacy evidence.

No-benefit, strict-benefit and ordering-reversal cases all fit the broad controller model. Their relative frequency on real generated candidate directions is an empirical question requiring preserved, representative transactions and patient evaluation.

## 5. A limited regime in which finite rejection cannot reverse the comparison

If the **probabilities**, not merely logits, are affine in candidate coefficients over all trial points, the local risk equals the actual risk exactly. With exact feasible optimizers, both S and J satisfy the group budgets at scale 1. Also `f(alpha)<=f(0)` implies `mean E(alpha)<=mean e-lambda||alpha||^2/2`, so the mean guard passes too, even with positive group budgets. Therefore both accept their full local coefficients in this idealized regime; the specific finite-rejection reversal is impossible. Regularized-versus-raw ordering still needs the distinction in section 2.

An affine **logit** head with softmax does not satisfy that assumption; the main counterexample is already such a head. More generally, certified remainder bounds and sufficient slack can guarantee acceptance: if each local risk plus its valid nonlinear error bound stays below its actual guard threshold, and the analogous mean inequality holds, scale 1 passes. If the local mean gap also exceeds the sum of valid J/S remainder bounds, actual mean ordering follows. The current source does not supply such global error certificates; measuring the actual functions remains necessary.

These slack and ordering statements compare the **same coefficient endpoints** in their local and actual evaluations, typically both proposed full-scale endpoints. They do not transfer unchanged to different returned backtracking scales. For returned coefficients `s_J alpha_J` and `s_S alpha_S`, the local gaps and remainder bounds must be recomputed at those respective coefficients.

## 6. Selecting between S and J: what it buys and what it changes

Consider a **research proposal**, not an implemented new method: from one immutable reference, compute S and J from the same fixed directions; evaluate each with the same functional finite checks; let a rejected arm return the reference. Include the reference itself and choose the returned endpoint with minimum measured actual mean audit risk. Then, deterministically,

`mean R(selected)<=min(mean R(returned S),mean R(returned J),mean R(reference))`.

Every selected class still respects its budget/tolerance relative to the common reference. This is **current-audit mean dominance among the evaluated returned endpoints**, not classwise dominance over each arm. A valid four-class probability illustration makes the distinction concrete: from uniform base, S endpoint `(.2,.6,.1,.1)` gives risks `(.08,.005,.005)`; J endpoint `(.1,.65,.15,.1)` gives `(.06125,.01125,.005)`. The lower-mean J endpoint worsens class 2 relative to S while both stay below the reference budgets. These endpoints illustrate the selector logic only; they are not claimed to be solver outputs.

The comparator is each arm's first accepted endpoint, matching the existing guard. Taking the minimum over **all** evaluated scales, searching additional directions, or evaluating S after committing J would be a different policy. Sequential commits would destroy the shared-reference comparison.

This selector is not cost-free and does not create a new learning guarantee:

- If J's moments are already available, S's scalar moments follow algebraically with no additional JVP. A standalone S baseline can instead need only one aggregate-direction decoder JVP rather than m; matching the portfolio's cost against an unoptimized scalar baseline would conceal that advantage.
- Evaluating both candidate rays adds actual trial forwards, potentially up to two backtracking grids. The reversal fixture uses eight failed J trials plus one successful S trial, versus one S trial alone. Record actual counts and time; sharing the same input/labels does not make the compute equal.
- It increases adaptive use of the same TRAIN audit measure. No additional unique patient labels are necessarily read, but label-query/forward counts and selection policy have changed. This is a new declared experimental arm, not an unnoticed implementation detail.
- The selection objective is actual mean risk, whereas J was chosen with a coefficient penalty. That is an explicit two-stage policy choice, not a theorem that the original regularized QCQP already solves the finite problem.
- All candidates must remain shadow states. Commit once and update EMA once after selection. The mathematical comparison certifies neither future states nor the EMA trajectory.
- The selector can still reject both arms, overfit the audit support, hurt a different slice/patient, or reduce hard Dice. Round one's same-audit Brier/Dice counterexample already blocks a Dice guarantee. Independent external evaluation and novelty attribution are unchanged obligations.

## 7. Decision and falsification gates

1. Keep the exact aggregate scalar + identical finite guard as the mandatory reference. Matching scalar ridge to `m*lambda` is essential for a literal restricted-objective claim.
2. Report four separate quantities: local regularized objective, local mean risk, actual accepted group/mean risks, and final patient endpoints. Never use `f_J<=f_S` as shorthand for all four.
3. A numerically successful or near-optimal local solve is insufficient evidence of useful finite updates. Log exact/near-optimal gaps separately from actual rejection and effective displacement.
4. If a portfolio is evaluated, compare against the individually optimized S and J costs and state its extra selection explicitly. A current-audit minimum is a mechanical property, not patient efficacy or novelty.
5. Do not use manufactured reversal prevalence, large local objective gaps, or favorable finite examples to select patient hyperparameters. First reproduce the candidate-family geometry on an independently declared transaction set under the same label/provenance rules.

## Execution receipt

```sh
rtk proxy python reports/research/20261023_astra_ultra_round2/probes/B_geometry.py
```

Result: **PASS** — three consistent local quadratic cases, three exact finite-head cases, four feasible near-optimal perturbations, and one portfolio counterexample. CPU float64 only; no patient or GPU claim. The probe temporarily replaces only the numerical coefficient-return function within its own process when checking analytically certified candidates, then uses the unchanged source guard. It also tests the unmodified source solver in each finite case. Production files and all prior report bytes remain untouched by this worker.

## Independent D review response

D reviewed initial report SHA256 `3621c4ddd934f720fbd49c684684415055489326a61e588d3bee2e6c7ba4fff8` and independently checked the exact averaged moments, box-gradient signs, scalar optimum and local-gap fractions. No blocking mathematical issue was found. Earlier challenges to distinguish regularized objective from raw mean risk and manual directions from `make_candidates` output are explicitly addressed above. D's final precision request was accepted: section 5 now states that nonlinear remainder/slack ordering applies at the same coefficient endpoints, and must be recomputed when returned backtracking scales differ. This is a report clarification, not a production or patient-result change.
