# Scientific decision: make historical geometry earn its necessity

Date: 2026-09-12. Source HEAD: `9726e6d519c33035489ca7a5e4681a046493d89e`. Decision concerns research and claims, not authorization for another implementation phase.

## 1. Decision now

**Continue investigating selected Candidate C, but do not yet make it the paper's established method contribution.** The current evidence supports a specific computational contract and a worthwhile falsifiable question. It does not establish that historical replay or coordinate optimization is necessary, novel enough, effective, or worth its runtime cost.

Current Dynamic Window should remain a baseline and implementation component. Learned offsets, structured anisotropic support, bilinear sampling, attention and recurrent conditioning do not independently supply a defensible headline novelty claim. Candidate C's possible contribution lies in a particular **transition-referenced intervention and preservation formulation**, not another name for deformable attention.

The competing mechanisms that must be named are G-BRS/f-BRS, spatial transformation optimization, minimal counterfactual recourse, counterfactual attention learning, FocalClick-style preservation, and the recent ActionSplice and LeCor formulations. Their concrete overlap and residual differences appear in [03](03_prior_art_challenges.md); [05](05_evidence_ledger.md) records inspection depth and gaps.

## 2. The strongest current equation, honestly stated

Let `Q1` be the single gradient proposal projected into the bounded support domain, and `Q2=P+(Q1-P)/2`. Define `select_first` to return the first proposal passing the actual predicted-FIX and objective checks, and `P` otherwise:

\[
\widehat Q=\operatorname{select\_first}_{Q\in(Q_1,Q_2)}
\left[ L_R(Q)+\lambda D(Q,P)<L_R(P),\quad
\operatorname{Preserve}_{\widehat F}(B^Q,B)\right],
\]

\[
\boxed{N=S+g\odot\left[U_{\rm frozen}(r;\widehat Q)-U_{\rm frozen}(r;P)\right]}
\quad\text{after verifying }U_{\rm frozen}(r;P)\simeq B.
\]

In the implementation, stored factual `B` is used in subtraction after the tolerance check, and failed proposals explicitly yield zero innovation. The final official Auditor still decides acceptance; failed identity/state replay routes to the ordinary path. This is a bounded intervention probe, not a certified minimizer. The first gradient is independent of `lambda`; that parameter screens proposal outcomes.

The boxed equation is **not independently novel**: residual transfer, counterfactual subtraction and frozen-model intervention each have prior art. What remains potentially distinctive is their joint restriction to an accepted ordinary segmentation transition, realized support intervention, predicted signed restoration/preservation evidence, and subsequent official re-audit. Narrowness alone does not make the combination scientifically important.

## 3. What deeper reasoning changed

| Earlier tempting interpretation | Deduction / competing explanation | Consequence for the paper |
|---|---|---|
| Geometry repairs regressions, so geometry is useful | Oracle true-R local rollback already repairs every R and leaves F unchanged | Need an advantage under predicted evidence, not oracle restoration alone |
| Predicted-FIX constraints guarantee safe correction | True-FIX damage is only bounded by coverage of true FIX plus runtime violations | Separate evidence quality, numerical constraint compliance and true harm |
| λ creates a minimum-change search | Its derivative at the only gradient point is zero | Report screening behavior; do not claim an optimized minimal action |
| No feasible proposal means geometry cannot repair | A feasible point can exist off the checked ray or at a smaller untested step | Separate representational headroom from bounded-solver failure |
| Exact replay explains why the first prediction failed | It establishes a computational intervention, not anatomical error attribution | Use causal language only with the computational object explicitly named |
| A successful counterfactual becomes a successful next state | C3 interpolation can retain the factual wrong class | Measure pre-transfer, post-transfer and final-retained outcomes |
| C is a multi-turn repair process | Three-attempt paths normally contain one restitution action; no-op can HALT | Match opportunities and report path lengths |
| Added audit exposure removes the mismatch | Early evidence is untrained; accepted history may be rare; solver has detached gradients | Exposure, informativeness and learnability are separate hypotheses |

The [formal report](01_problem_formulation.md) provides premises, equations and constructive examples. These observations narrow the claim without asserting that Candidate C will fail empirically.

## 4. Training exposure is a competing explanation, not a completed prerequisite

Current code preserves the primary zero-history bootstrap and adds optional predicted-history exposure. During annotation bootstrap, the auxiliary rollout uses an **untrained Auditor**, official acceptance and rejection-to-HALT, and loss on the retained state. The auditor-training phase can add predicted-history auditor examples while the annotator stays frozen. Joint training uses the deployed policy. See [unified_trainer.py](../../../src/self_audit/training/unified_trainer.py) and [finetune_joint.py](../../../src/self_audit/training/finetune_joint.py).

**Derivation for a simple conditioning layer:** if its preactivation is `W_h h + W_e e + b` and `e=0` on every training example, then the data-loss gradient for `W_e` is zero. Weight decay can still change parameters, and other layers can train. A nonzero derivative of output coordinates with respect to `e` can exist from initialization; that alone does not demonstrate learned, useful use of evidence.

When predicted history is enabled, three questions remain:

1. How many examples actually reach an ordinary annotation call with nonzero real evidence while that annotator is trainable?
2. Is that evidence informative and distributed like later inference evidence, rather than merely nonzero and shape-compatible?
3. Does the learned response improve correction utility under matched histories, rather than just moving coordinates?

The solver's innovation is detached. This removes gradients **through the coordinate optimization**, but its added value still changes the point at which a downstream supervised loss is evaluated; upstream parameters can receive different gradients. Therefore “no direct solver gradient” does not mean “no training effect.” Conversely, it is not an explicit objective for learning a representation that is easy to restitute.

The fair initial study is a small factorial comparison of **current vs Candidate C** crossed with **matched exposure off vs on**, holding A0, pretrained backbone, schedule, seeds and data constant. This distinguishes gains from the intervention from gains due to exposure. No such training was run here. Training a response specifically for useful one-step correction would need further research; [LeCor's meta-objective](https://arxiv.org/html/2609.09477v1) is a close competitor, so this is not an unclaimed novelty shortcut.

## 5. Minimal sequence of studies

### Study 0 — opportunity and measurement integrity

Use an existing frozen, eligible checkpoint if available; do not weaken A0 to manufacture regressions. Count all patient paths, accepted ordinary transitions, true mixed transitions, predicted qualifying transitions, replay failures, identity fallbacks, ordinary fallbacks and actual restitution actions. Report patient-level prevalence and distributions. Confirm actual tensor dtypes, C1 error and computational evaluations before interpreting any geometry.

Measure REGRESS precision and FIX coverage specifically on accepted transition histories; pooled pixel AUROC is insufficient. If an eligible checkpoint or actual run artifacts are unavailable, mark this study pending rather than creating synthetic performance claims.

### Study 1 — same-history, paired intervention

Use the same frozen record and evidence for identity, shipped direct rollback, a rollback control with identical predicted-FIX protection, current-state coordinate optimization and Candidate C. Add C without FIX only as a diagnostic. For current-state optimization, center its transferred innovation on its **own factual current-state forward**, while keeping previous restoration targets, gate, constraints and matched internal depth. Also hold turn/iteration conditioning identifiers fixed for the operator-controlled comparison; report the naturally deployed next-turn variant separately. Otherwise history, subtraction, conditioning and depth change together.

Full hard and soft rollback, including a prespecified signed score `E_R-E_F`, test whether a weak rollback gate created an artificial advantage. They are proposed controls, not deployed changes. A larger feature/bias optimization baseline becomes necessary if C clears these cheaper tests because of the direct G-BRS challenge.

Count factual replay, candidate forward checks, coordinate backward, final audit, memory, synchronization and wall time separately. One ordinary forward is a cheap baseline, not automatically a compute-matched one. A common budget ceiling and a measured latency frontier answer different questions.

### Study 2 — full-policy evaluation

Run the actual official acceptance/HALT process. All patients enter the primary population, including paths with no eligible C action. Report conditional eligible effects separately. A no-op action and an ordinary-next-step control expose the lost-opportunity/termination explanation without silently changing the main algorithm's fallback policy.

Use patient-paired outcomes and cluster uncertainty at patient level. Do not count slices or pixels as independent replicates. Prespecify a smallest useful improvement `delta_min`, tolerable harm `h_max` and compute ceiling **before** evaluating a locked test set. Their values require development evidence and a defined use case; this report does not invent clinical margins.

### Study 3 — external robustness

Keep these protocols separate: ACDC development and native evaluation; frozen ACDC-selected model tested externally on M&Ms; M&Ms native supervised training/evaluation. External M&Ms must not tune solver parameters, tau or checkpoint selection. Native M&Ms is a distinct experiment, not additional data for the ACDC model.

The existing development validation set is not a fresh confirmatory test merely because different hypotheses are evaluated on it. Reserve a locked independent evaluation or explicitly label further development analyses exploratory. Do not interpret a wide interval crossing zero as evidence of equivalence.

## 6. Risk calibration belongs to the complete policy

An Auditor is operationally separate from the coordinate optimizer, but both local evidence and final judgement depend on learned representations. Selection of proposals can change its error distribution. A threshold calibrated on ordinary candidates does not automatically calibrate candidates produced by the solver.

There is no need to claim that all sequential risk calibration is impossible. [Learn then Test](https://arxiv.org/pdf/2110.01052) offers a route using valid tests over a prespecified frozen policy family and held-out calibration cases. [Non-monotonic CRC](https://arxiv.org/html/2602.20151v1) addresses nonmonotone loss under explicit assumptions. Neither supplies a guarantee here without checking those assumptions and data separation; neither ordinary calibration nor a three-turn limit establishes external-domain safety.

An elementary bound `P(any harmful accepted action) <= sum_t P(reach t and harmful acceptance at t)` requires no independence. Useful numeric bounds would still need data for those actual reached-turn events. Per-turn AUROC and nominal tau do not provide them.

## 7. Predeclared interpretation of outcomes

* **Continue with a method claim:** historical coordinates offer a reproducible gain beyond strong same-evidence alternatives, with acceptable harm and cost, on locked patients and a relevant external protocol. Mechanistic diagnostics connect that gain to useful support interventions and preserved fixes.
* **Revise the mechanism claim:** benefit appears only with better predicted-history exposure; current-state latent refinement matches C; C3 erases otherwise useful repairs; or a larger-budget oracle has headroom that the deployed search misses. Identify which link failed before proposing another algorithm. This is not authorization to implement Candidate B or redesign Self-Audit.
* **Abandon Candidate C as the paper's headline mechanism:** precise comparisons show no useful advantage over cheaper controls, evidence cannot support reliable restoration, opportunity is too rare for the chosen use case, or harms/cost exceed prespecified limits. Keep any validated software as an optional component only if justified.

Failure on one model/dataset/budget does not prove that self-audit is impossible. A negative study becomes publishable through a generalizable explanation and rigorous controls, not by attaching a dramatic title. Likewise one positive Dice result does not establish geometry necessity or novelty.

## 8. Claim taxonomy and provisional ratings

| Item | Status now |
|---|---|
| Novel attention primitive | Not supported |
| Novel general optimizer / minimal-recourse objective | Not supported; established families and a restricted approximation |
| Novel use within Self-Audit | Distinct implementation/application context, insufficient by itself |
| Potentially distinct mechanism combination | Recorded accepted ordinary transition + realized-coordinate restitution + predicted FIX protection + factual subtraction + official re-audit |
| Novel training objective | Not established by current detached solver and exposure option |
| Engineering contribution | Replay checks, bounded compute, ephemeral history and diagnostics; meaningful engineering, not proof of scientific novelty |

Acceptable provisional wording:

> In the primary methods inspected through 12 September 2026, we did not identify an exact instance of the joint formulation above. Its closest competing mechanisms are frozen-network backpropagating refinement, constrained recourse, counterfactual attention and historical state transport. Whether the joint formulation is scientifically necessary remains an experimental question.

Ratings are **subjective research judgements, not calibrated probabilities or experimental scores**:

| | Current Dynamic Window | Candidate C under investigation |
|---|---:|---:|
| Novelty confidence | 2/10 | 4/10 |
| Technical soundness of the formulation | 7/10 | 6/10 |
| Relevance to the Self-Audit question | 6/10 | 9/10 |
| Implementation / experimental integration risk (higher = more risk) | — | 7/10 |

The C score remains moderate because close components are known, novelty closure has gaps, and geometry/history necessity is unmeasured. Technical soundness is limited by the search approximation, evidence quality, transfer attenuation and policy confounds; the rating is not a re-certification of GPU/runtime tests.

**Paper readiness:** not yet sufficient as a demonstrated paper-level mechanism. **Research readiness:** the question and falsification protocol are concrete enough for a bounded controlled pilot once an eligible checkpoint, independent evaluation plan and compute authorization are in place. This research task changes no production code and makes no novelty, superiority or clinical-performance claim from its arithmetic checks.
