# What scientific problem would make Candidate C necessary?

Research date: 2026-09-12. Source revision: `9726e6d519c33035489ca7a5e4681a046493d89e`. This is a formulation and falsification analysis, not a proposed production patch or a result from trained models. Candidate C remains the selected subject. Statements labelled **Derivation** are deductions below; empirical premises remain unverified.

## 1. Replace the module question with a decision question

“Can we improve Dynamic Window Attention?” confounds three separate questions:

1. Does the system encounter accepted corrections containing useful fixes and harmful regressions?
2. Can its predicted evidence distinguish those changes sufficiently well to support another action?
3. Does changing the spatial support of the **recorded previous computation** offer an advantage over cheaper actions given the same evidence?

The paper-level question should be:

> Under noisy transition evidence, can intervention on a recorded correction computation recover beneficial changes more reliably than output rollback or correction of the current state, at matched preservation constraints and computational cost?

This makes geometry and historical replay hypotheses to test. It does not treat either as an intrinsically valuable architectural feature. The strongest competing formulation is frozen-network test-time auxiliary-variable refinement, exemplified by [f-BRS, §3](https://arxiv.org/abs/2001.10331) and [G-BRS, §2](https://openaccess.thecvf.com/content/CVPR2022/papers/Lin_Generalizing_Interactive_Backpropagating_Refinement_for_Dense_Prediction_Networks_CVPR_2022_paper.pdf). Candidate C needs to establish why its particular intervention variable and temporal reference matter within that family.

## 2. Define the objects before attributing meaning to them

For one accepted ordinary transition, let

\[
r=(H,A,E_{\mathrm{in}},t,J,\theta,P,B,g),\qquad
B=U_\theta(r;P),\qquad B^Q=U_\theta(r;Q).
\]

Here `A` is the pre-transition logit state; `B` the factual candidate retained after acceptance; `H` the recorded feature tensor; `P` the sequence of realized sampling coordinates across internal iterations; and `g` the recorded class-shared sigmoid update gate. Frozen replay holds the state, parameters, buffers and conditioning fixed while overriding coordinates. Let `a=argmax A`, `b=argmax B`, and let `Y` denote GT **only in mathematical evaluation definitions**.

\[
F=\{x:a(x)\ne Y(x),\ b(x)=Y(x)\},\quad
R=\{x:a(x)=Y(x),\ b(x)\ne Y(x)\}.
\]

These sets are disjoint. The remaining `UNCHANGED` category includes persistent errors and wrong-to-different-wrong changes; it does not mean the logits or labels were literally unchanged. Runtime substitutes thresholded predicted sets `F_hat,R_hat` from the Auditor, not these oracle sets. See [target definitions](../../../src/self_audit/audit/targets.py) and [current solver](../../../src/self_audit/models/self_audit_net.py).

The research objective is

\[
\min_{Q\in\mathcal T_\rho(P)}
\underbrace{\frac{\sum_x w_R(x)\,\mathrm{CE}(B^Q(x),a(x))}
{\sum_xw_R(x)}}_{L_R(Q)}
+\lambda D(Q,P),
\quad
\arg\max B^Q(x)=b(x),\ m_x(B^Q)\ge\gamma m_x(B)
\quad(x\in\widehat F).
\]

`T_rho` is the bounded coordinate box intersected with the sampling domain; `D` is mean squared displacement in feature pixels; `m_x` is the winner-versus-runner-up logit margin. This is a predicted-label preservation constraint, not a GT-correctness constraint.

The implemented approximation computes one gradient at `P`, then checks a full projected step and at most its half step, falling back to `P`. It does **not** project onto the FIX feasible set or solve the constrained problem to optimality. The official Auditor subsequently judges

\[
N=S+g\odot(B^{Q_{\rm selected}}-B),
\]

where `S` is the current retained state. It retains authority over acceptance and rejection-to-HALT. The exact code-level qualifications are in [02](02_mechanism_analysis.md).

## 3. Five deductions that narrow the contribution

### D1. An oracle rollback ceiling changes the question

**Derivation.** Define `n_oracle(x)=a(x)` on `R` and `b(x)` elsewhere. On every true REGRESS pixel, `a(x)=Y(x)` by definition. Therefore this action repairs all `R` and changes no pixel in `F` or its complement outside `R`.

Thus no geometry method can exceed oracle local hard rollback on the narrow endpoint “fraction of previous true regressions repaired while preserving previous true fixes.” This is not an available inference baseline: its mask uses GT. The shipped predicted, gated logit rollback is also not this oracle. The deduction instead identifies where a contribution could exist: **robustness to incorrect evidence, useful changes outside the restoration target, and the cost–harm–benefit trade-off**.

If the practical claim is only that coordinates reproduce previous labels on a correctly identified region, the much simpler mechanism already explains it. A reviewer should require both same-evidence rollback and the separate oracle ceiling.

### D2. C3 can preserve a repair target's class while suppressing the repair itself

**Derivation.** When `S=B` exactly,

\[
N=(1-g)B+gB^Q.
\]

If the same class strictly wins at both endpoints, it wins at every convex interpolation with a shared class-independent `g`. Ties and finite precision require explicit checks. This explains a preservation property; it says nothing about whether a new class will become the winner on a REGRESS pixel.

For binary or pairwise restoration margin `B_r-B_w=-a<0` and `B^Q_r-B^Q_w=b>0`, the repaired ordering survives C3 iff

\[
g>\frac{a}{a+b}.
\]

In multiclass segmentation it must beat **every** competitor, not just the original winner. A factual margin of `-0.2`, counterfactual margin of `+0.3`, and `g=0.2` give final margin `-0.1`: successful counterfactual repair, unsuccessful transferred repair. Report `B^Q`, the transferred `N`, and the finally accepted state separately.

The inner expert already gates its delta. Consequently the derivative of the transferred action includes a product of inner and outer gates; if the inner gate is locally constant, its delta pathway is attenuated by `g²` at the factual point. The ordinary gate was not explicitly trained as a restitution gain. This is a testable coupling, not proof that removing it would improve the method.

### D3. The current solver is a three-choice probe; λ is a screening parameter

**Derivation.** `grad_Q D(Q,P)|_(Q=P)=0`. Therefore the single initial direction is independent of `lambda`. Under default normalization, the full and half candidate coordinates are likewise independent of it. Changing `lambda` can change which proposal passes the improvement check; it cannot rotate the search direction or continuously tune its initial magnitude.

Furthermore, rejecting both checked points does not certify infeasibility. Consider `P=(0,0)`, unit box, `L=(q1-1)²+(q2-1)²`, and protected margin `m(q)=0.1-2q1` with floor `0.05`. Descent along `(1,1)` proposes `(1,1)` and `(0.5,0.5)`, both infeasible. Yet `(0,1)` is feasible and lowers `L` from 2 to 1. A feasible direction exists but this bounded solver cannot find it.

Positive constraint slack can make halving restore feasibility even in a linear model. Only at an active constraint with a direction pointing strictly outward will every positive step violate the first-order condition. Constraint-aware gradient projection is itself established optimization machinery; compare [GEM, Eqs. 6–8](https://papers.neurips.cc/paper/2017/file/f87522788a2be2d171666752f97ddebb-Paper.pdf) and [PCGrad, Algorithm 1](https://papers.neurips.cc/paper_files/paper/2020/file/3fe78a8acf5fda99de95303940a2420c-Paper.pdf). Importing it later would need a scientific reason beyond renaming it.

### D4. Exact replay identifies a computational intervention, not anatomical causation

**Derivation.** Holding the recorded inputs and operator fixed makes `B^Q-B` an interpretable intervention effect within that computation. Downstream attention weights, activations and gates are allowed to change as consequences of `Q`; “only coordinates are intervened on” does not mean those mediators remain fixed.

C1 cannot show that the original support caused an anatomical mistake, that an alternative support is uniquely correct, or that the generator would naturally produce it. The distinction between observed counterfactual changes and feasible causal actions is central in [Karimi et al., §§3–4](https://arxiv.org/pdf/2002.06278); here the causal object is the known network computation, not an identified causal model of a patient.

Permutation of point indices leaves the real-arithmetic attention sum unchanged. This removes semantic uniqueness of point identities, not spatial-set geometry: density, spread, boundary distance and intervention effects remain meaningful measurements. Floating-point reduction order can produce small differences, and a displacement penalty tied to ordered factual points need not share the permutation symmetry.

### D5. Audit evidence can add useful computation without adding an external observation

**Derivation under an explicit premise.** If the complete recorded state is `Z` and deterministic audit evidence is `E=f(Z)`, then `I(Y;E|Z)=0`. Revisiting locations in a feature tensor already contained in `Z` also supplies no additional external measurement conditional on that full tensor.

This does not make evidence useless. A finite-capacity annotator may benefit from a supervised transformation it could not otherwise compute efficiently. It means the scientific benefit is better inference or use of computation, not automatic information gain or an independent second opinion. Different heads, detached tensors and separate losses do not establish statistically independent errors. Independence and calibration must be measured on the actual candidate distribution.

## 4. Two exact links between evidence quality and action utility

**D6 — hard rollback value.** For any selected mask `M`, replacing factual hard classes by pre-transition classes only on `M` gives

\[
\Delta N_{\rm rollback}=|M\cap R|-|M\cap F|.
\]

Persistent-correct and persistent-wrong pixels contribute zero, even when two incorrect labels differ. If the local evidence were calibrated conditional transition probabilities, the expected *unweighted per-pixel* benefit of full hard rollback would therefore be `p_R-p_F`. This requires neither independence of pixels nor a novel optimizer. A cost-weighted endpoint changes this decision rule, and a gated logit interpolation has additional margin-dependent outcomes.

This yields a precise skeptical baseline: evaluate same-evidence full/soft rollback and a prespecified `E_R-E_F` action score, alongside the shipped thresholded gated rollback. These are future experimental controls, not implemented policy changes. They must receive comparable predicted-FIX protection and final official auditing. The hypothesis is whether the coordinate intervention adds value beyond the local decision already encoded in signed evidence.

**D7 — true-FIX damage bound.** If the actual transferred candidate preserves every class on `F_hat`, then any destroyed true FIX lies in `F \\ F_hat`. For nonempty `F`,

\[
\frac{\#\text{destroyed true FIX}}{|F|}
\le 1-\frac{|F\cap\widehat F|}{|F|}.
\]

This is an upper bound from set inclusion, not an equality or a safety claim. Runtime violations add another term. It shows that a perfect check of predicted-FIX classes can coexist with substantial true-FIX damage if evidence coverage is poor. Measure coverage on accepted mixed transitions, not just pooled local classification accuracy.

## 5. Distinguish local controllability from interpretation

Linearize the transferred correction around `P`:

\[
\delta N\approx GJ\delta Q,\qquad
J=\frac{\partial B^Q}{\partial Q}\bigg|_P,
\]

where `G` applies the class-shared spatial gate. A desired restoration direction can be unreachable because of rank deficiency, vanishing local feature derivatives, downstream coupling, small margins on protected pixels, or the trust radius. More coordinate variables than output variables does not establish controllability.

Measure whether useful target directions lie in the locally reachable set while FIX margins remain feasible. Use Jacobian-vector products or restricted probes on a held-out diagnostic subset; no full dense Jacobian is required. Separate an **offline larger-budget oracle diagnostic** from the deployed one-step algorithm. A successful larger-budget diagnostic can demonstrate headroom missed by the deployed probe. Its failure supplies negative evidence within that search budget, not a certificate that no useful action exists. Neither can be inferred from coordinate heatmaps alone.

## 6. The temporal contract determines the estimand

Only the latest accepted **ordinary** transition is replayable. Accepted restitution clears that record. With three attempts, normal surviving paths are ordinary → restitution → ordinary; replay/state failures can instead route through an ordinary fallback. There is at most one usable restitution action on a three-attempt path. A no-op restitution can still be audited and rejected, ending the trajectory. Therefore Candidate C versus current Dynamic Window changes both actions and opportunities for later refinement.

Two studies are necessary:

* **Paired fixed-history intervention study:** all arms receive the identical frozen ordinary record, evidence, budget and downstream gate/auditor. It estimates the value of an intervention given that opportunity.
* **Full-policy patient study:** each arm runs its actual accept/reject/HALT process. It estimates deployment utility, including absent opportunities, fallback, termination and cost.

For a bounded patient utility in `[0,1]`, if two coupled policies are identical until the first C-eligible event `E`, then `|E[U_C-U_0]| <= Pr(E)`. This elementary dilution bound is not a segmentation-performance prediction. It explains why a large conditional gain on rare eligible cases can produce a negligible population effect.

## 7. The necessary hypothesis chain

| Link | Required evidence | What failure would mean |
|---|---|---|
| Opportunity | Accepted ordinary transitions contain meaningful true mixed FIX/REGRESS events | Too little deployment headroom; do not manufacture weak A0 |
| Evidence | Predicted REGRESS precision and FIX coverage on those events | Runtime target/constraint unreliable, regardless of replay correctness |
| Controllability | Useful corrections attainable inside the support budget | Coordinates cannot express the required action |
| Transfer | Counterfactual repairs survive C3 and final audit | Solver success does not translate into system utility |
| Necessity | Historical coordinate intervention beats fair rollback and current-state alternatives | Replay/geometry are optional parameterization or bookkeeping |
| Generalization | Patient-held-out and frozen external evaluation sustain the effect | Development-set or policy-selection explanation remains |

Each link has a distinct falsifier. A single Dice improvement cannot establish the chain. The [experimental protocol](04_falsification_protocol.md) defines the measurements; the [decision report](06_research_decision.md) specifies claim and stopping boundaries.

## 8. Executable checks of the deductions

[The small standalone probe](probes/restitution_counterexamples.py) evaluates five constructed examples: oracle masked rollback, the signed rollback identity, gate attenuation, λ-independent initial gradient, and a feasible repair missed by the checked ray. [Its JSON output](probes/restitution_counterexamples.json) records the numbers. These checks validate the arithmetic of counterexamples only. They do not test Self-Audit, trained checkpoints, GPU behavior, medical usefulness, or novelty.
