# Theory and closest prior results for the executed stable-guidance run

Date: 2026-10-14. Research-only artifact, owned prefix `THEORY_PRIOR_ART`. Snapshot: `b937e3b746608725e43d3bc3cef8861666d152dc`.

The useful continuation is to distinguish a fidelity/selection bottleneck from a named-semantic bottleneck. Small weighted KL can limit how much a student becomes confident where its targets are uncertain; it cannot certify accuracy, unweighted coverage, rare-class performance, or new patients. A positive expansion theorem is possible with imperfect anchors and graphs, but only after explicitly paying for incorrect semantic support, graph leakage, student deviation, and the accepted-set denominator. The present evidence does not verify those semantic assumptions. Restart diffusion, clamping, consistency-based denoising, and confidence rejection already have strong prior art; none is a defensible standalone novelty claim here.

## Scope, provenance, and evidence ledger

Status vocabulary: **VERIFIED** means inspected source/artifact identity or checked arithmetic in this task; **REPORTED** means an archived run measurement, not re-executed here; **DERIVED** means a proof or consequence under stated assumptions; **BLOCKED** means the evidence required to apply a claim is absent. **NOT RUN** marks future diagnostics. No production changes, model/data downloads, training, GPU execution, new mask scoring, TEST analysis, commits, or pushes were performed.

All paths below are relative to `reports/stable_guidance_world_v2_20261014/` unless stated otherwise. `E/` abbreviates `executed_source/src/self_audit_guidance/`; it never denotes historical top-level `src/`.

| Claim | Status and exact evidence |
|---|---|
| Current checkout is the requested SHA | VERIFIED with `git rev-parse HEAD`; initial working tree clean. |
| Geometry execution base is `7d1c6a04064cda473d1eccfbed818392c37b1a37` plus launcher patch | REPORTED `summary.json.base_source_commit`; VERIFIED patch SHA-256 `ecbcd639fbaa010ac3f7fca9ac5ba3272278806ce92eecc8171c4833f70815ee`. All 11 `E/*.py` hashes match both `run_evidence/run/RECIPE.json.source_sha256` and `run_evidence/run/student/FROZEN.json.source_sha256`. Receipt: `THEORY_PRIOR_ART_CHECKS.json.source_hash_matches`. This verifies archive consistency, not remote execution independently. |
| Baseline training versus later forensic source | Baseline training at `d5a45f3` is the supplied task provenance. `baseline_forensics/RESULTS.json.source_commit` identifies the later forensic code as `7d1c6a0`; do not mislabel that as baseline training. `baseline_forensics/EXECUTION.json.original_main_commit=e4d3269...` is another recorded checkout identity, not proof of baseline training source; this worker did not reconstruct the baseline executable. |
| New run: 50 epochs, 127650 optimizer steps, 110461 semantic updates | REPORTED `summary.json.training`. Last-epoch CE `1.013808740407279`, entropy `0.981306225759442`, KL `0.032502514620783185`, active fraction `0.8648648648648649`; VERIFIED CE−H−KL differs from zero by `2.71e-11`. These are online equal-batch means. |
| Extremely sparse named output | REPORTED `summary.json.unknown_fraction=0.9963315431177358`, `metrics["val/coverage"]=0.0036684568822642327`, `rv_val_frames_with_predictions=0`, 36 VAL frames. Foreground Dice `0.05068521149021373`. UNKNOWN is 255, not BG: VERIFIED `E/student.py:148-157`, `E/core.py:301-303`. |
| Geometry mainly removed unsupported-header RV bank support | REPORTED `summary.json.bank.rv_geometry_comparison`: 4322 old RV pixels became zero for 28 `NO_CODED_WORLD_GEOMETRY` patients; 877 remained 877 for 52 coded patients. This identifies a support change, not whether removed seeds were correct or why all downstream RV predictions vanished. |
| No RV permission veto in the new run | DERIVED from reported `bank.class_patients=[62,14,55,47]` and VERIFIED permission rule `E/student.py:79-82`: all four pass the two-patient rule. The threshold is support permission, not correctness certification. |
| Training/naming/selection boundary | VERIFIED reviewed source uses detached soft targets, reliability weights, fixed final checkpoint, and exports without reference labels (`E/student.py:38-59,79-103,120-133,145-159`). The archived launcher patch forwards geometry and diagnostics and binds provenance; it adds an evaluation grid option, not semantic calibration. `summary.json.gt_used_for_training=false` and selection metadata are REPORTED execution claims. This is not a new comprehensive leakage audit. |

`summary.json.comparison` already warns that the precision macro changed its defined class set because RV precision is undefined. It must not be used as a like-for-like precision improvement. Existing VAL measurements are exposed reference-informed development evidence; they do not authorize fitting thresholds or naming rules against VAL. No TEST metric is used here.

## Result 1 — fidelity can bound high-confidence mass, on the exact same grid and measure

**DERIVED, complete proof.** Let `I` be a finite set of locations on one common grid, `q_i,p_i` probability vectors on the same K named coordinates, and logs be natural. Define an arbitrary probability measure `ν_i≥0, Σν_i=1`, weights `w_i≥0`, `W=Σν_i w_i>0`, and

\[
D_w=\sum_i\nu_iw_i\,\mathrm{KL}(q_i\Vert p_i),\qquad
\bar D_w=D_w/W,\qquad \mu_i=\nu_iw_i/W.
\]

Weights may depend on q; they must be the weights actually used in this calculation. Let `0<δ<τ<1` and

\[
A_\delta=\{i:\max_c q_{ic}\le\tau-\delta,\ \max_c p_{ic}\ge\tau\},\quad
\kappa(\tau,\delta)=d(\tau-\delta\Vert\tau),
\]

where `d(a||b)=a log(a/b)+(1−a)log((1−a)/(1−b))`. Then

\[
\|p_i-q_i\|_\infty\ge\delta\quad(i\in A_\delta),\qquad
\mu(A_\delta)\le\min\{1,\bar D_w/\kappa\}\le
\min\{1,\bar D_w/(2\delta^2)\}. \tag{1}
\]

The last inequality compares the two upper bounds; it does not say accepted mass equals either bound. If `τ−δ<1/K`, the set is empty. Any additional permission, margin, or status veto can only reduce the accepted subset, so (1) remains an upper bound.

**Proof.** Choose `c=argmax p_i`. Then `p_ic−q_ic≥δ`, which proves coordinate deviation. Group coordinate c against all others. The log-sum inequality gives `KL(q||p)≥d(q_c||p_c)`. For completeness, for each group G, Jensen applied to convex `−log` with weights `q_j/Σ_Gq` gives `Σ_Gq_j log(q_j/p_j)≥(Σ_Gq)log(Σ_Gq/Σ_Gp)`; sum the two groups. For `a<b`, `∂_a d(a||b)=log[a(1−b)/(b(1−a))]<0` and `∂_b d(a||b)=(b−a)/(b(1−b))>0`, so the minimum on `a≤τ−δ,b≥τ` is κ. At fixed b, `d(b||b)=0`, its a-derivative vanishes at b, and its second derivative `1/[a(1−a)]≥4`. Integrating twice gives `d(a||b)≥2(a−b)^2`, with boundary cases by continuity. Multiply by μ and sum over A. Infinite KL makes the bound vacuous rather than false. ∎

For the entire confidence acceptance set S, partition according to target confidence:

\[
\mu(S)\le\mu\{q_{\max}>\tau-\delta\}+\bar D_w/\kappa.
\tag{2}
\]

Thus low KL only restricts confidence acquired *outside* the target's high-confidence region. It supplies no lower bound on coverage and no truth information. At `τ=.7,δ=.1`, κ=`0.022582421084357415`, versus Pinsker's `0.02`; `THEORY_PRIOR_ART_CHECKS.json.checks.common_grid_binary_kl` verifies 1628 eligible random pairs among 20000 synthetic draws.

**Conversion and counterexamples.** On `w≥w0>0`,

\[
\nu(A_\delta\cap\{w\ge w_0\})\le D_w/(w_0\kappa).
\]

For all of A, add `ν{w<w0}`. This follows from `D_w≥w0 κ ν(A∩{w≥w0})`. A 99% zero-weight region can be arbitrarily wrong/confident while KL is zero on the remaining 1%; no unweighted guarantee follows. A rare class with weighted mass ε can absorb all divergence, giving a class-conditional bound proportional to `D_w/ε`, which may be ≥1. A mean across patients likewise does not give each-patient protection. High entropy alone does not determine the tail mass in (2), and small KL to a confidently wrong q preserves the wrong name.

### The actual denominator and timing

VERIFIED `E/student.py:53-58,98-103` uses `p_b=softmax(bilinear_resize(logits_b, bank_grid))`, q on the 48×48 bank grid, and

`w_bi = reliability_bi × allowed_classes[argmax(q_bi)]`.

The whole batch is inactive unless semantic permission exists and at least one positive-weight foreground argmax occurs. For `a_b∈{0,1}`, `W_b=Σ_i w_bi`, `d_b=max(W_b,1e−8)`, the reported epoch quantity is exactly

\[
D_{epoch}=B^{-1}\sum_b a_b\sum_i (w_{bi}/d_b)\,
 \mathrm{KL}(q_{bi}\Vert p_{bi}^{(step\ b)}).\tag{3}
\]

`E/experimental_diagnostics.py:18-38,46-56` explicitly includes inactive zeros. Let A be the positive number of active batches and `f=A/B`. Dividing (3) by f **always** gives the mean recorded active-batch scalar, including any denominator floor: `(D_epoch/f)=(1/A)Σ_{b:active}Σ_i(w_bi/d_b)KL_bi`. The last-epoch value is unconditionally `0.037581032530280556` as this recorded-scalar mean; no no-floor assumption is needed for that arithmetic identity.

The interpretation as a normalized KL measure is a separate issue. Set `λ_bi=a_b w_bi/(B d_b)` and `M=Σλ_bi≤f`. Then the **unnormalized online event mass** `Σλ 1_Aδ≤D_epoch/κ`; the normalized measure is λ/M and its divergence is `D_epoch/M`. Only when no active denominator is floored does M=f, so the active-batch scalar mean equals this normalized-measure KL. The published aggregate does not independently establish that floor condition. The amended script verifies a two-active-batch fixture with one floor: division by f still recovers the active recorded-scalar mean, while division by M yields a different normalized-measure KL (`checks.active_batch_mean_with_floor`). If A=0, both active normalization and selected active measure are undefined; the recorded epoch scalar is zero.

At τ=.7 and δ=.1, the raw bound is `1.4393`, exceeding even the total mass M≤.8649; the normalized bound is at least `1.6642` before capping at one, with equality to that value only without a floor. Thus the archived KL gives a **vacuous bound at this gap**, even for its proper online measure. Equal-batch averaging is not pooled-pixel averaging. Different p values in (3) come from different optimizer steps; it is not the final checkpoint's KL even on TRAIN. CE=entropy+KL here explains why CE near one is compatible with fitting uncertain targets; it does not prove global optimization or accuracy.

### The grid counterexample is decisive

The exporter evaluates native logits directly (`E/student.py:153-157`), while the loss downsamples logits before softmax. Let a 2×2 native block have four logit vectors

`z^(j)_c=log(q_c)+a(4·1[c=j]−1)`, with `q=(.4,.3,.2,.1)`.

Bilinear resize from 2×2 to 1×1 with `align_corners=False` averages these four vectors, giving log(q); the coarse student exactly equals q and coarse KL is zero. As a→∞ every native location becomes confident in a different class. With a=4, all four native maximum probabilities exceed `.9999989` (receipt `checks.grid_counterexample`). This is a proof by explicit construction that coarse fidelity cannot upper-bound native acceptance without a separate spatial/logit transport assumption. It is not a claim that the actual model realizes the construction. Even bilinear transport of coarse *probabilities* would define a different export operator (`E/experimental_diagnostics.py:115-123`).

## Result 2 — a conditional positive expansion guarantee with imperfect anchors and graphs

**DERIVED, complete proof; semantic premises BLOCKED on real data.** This result uses the actual restart family rather than pretending that the repository clamps seeds. For finite row-stochastic P and `0<α≤1`, define

\[
K_\alpha=\alpha\sum_{t=0}^{\infty}(1-\alpha)^tP^t,
\qquad Q=K_\alpha R,
\qquad R_j=(1-s_j)u+s_jV_j,
\]

where `u=(1/K,...,1/K)`, `0≤s_j≤1`, and V_j is a probability vector. The matrix Kα is stochastic, since every P^t is stochastic and the geometric coefficients sum to one. The convergent geometric series satisfies `Q=αR+(1−α)PQ`; uniqueness follows because the update contracts the maximum coordinate norm by `1−α`. For an approximate solution with residual r in that norm, `||Qhat−Q||≤r+(1−α)||Qhat−Q||`, hence error≤r/α. This matches `E/core.py:147-168`, including normalization of P and R before solving. Solver convergence alone is a statement about these inputs.

Fix a true class y_i for node i. Define `h_i=Σ_j Kα_ij s_j`. For `h_i>0`, call a route to j good if `V_j,y_i≥1−η`; let `b_i` be the fraction of weighted strength `Kα_ij s_j` carried by the remaining routes. A bad route includes a perfectly correct anchor from a *different* anatomical class. Thus b accounts for graph leakage as well as wrong anchor naming; anchor-count error rate is not a substitute. Then

\[
Q_{i,y_i}\ge L_i=(1-h_i)/K+h_i(1-b_i)(1-\eta).\tag{4}
\]

**Proof.** Expand Q using R. The uniform contributions sum to `(1−h_i)/K`; good routes contribute at least `h_i(1−b_i)(1−η)`; all bad-route contributions are nonnegative. Add the three terms. No independence of anchors, votes, or edges is required. If h_i=0, Q_i=u and b_i need not be defined. ∎

Consequently `L_i>1/2` gives the correct unique argmax and margin≥`2L_i−1`, because the total mass in all incorrect classes is at most `1−L_i`. The bound remains conservative under arbitrary correlation of the repeated multiscale votes; `E/core.py:280-287` explicitly does not assert vote independence.

**Robustness to graph and anchor misspecification.** For two restart systems `(P,R)` and `(P0,R0)` with the same α, suppose `max_j ||R_j−R0_j||∞≤η0` and `max_j TV(P_j,P0_j)≤ε`. Then

\[
\|Q-Q^0\|_\infty\le\eta_0+[(1-\alpha)/\alpha]\epsilon.\tag{5}
\]

**Proof.** Subtract fixed-point equations and write the difference as `α(R−R0)+(1−α)P(Q−Q0)+(1−α)(P−P0)Q0`. Every coordinate of Q0 is in [0,1], so a row's signed zero-sum difference against that coordinate is bounded by its total variation ε (split positive and negative parts). Take maximum norms, use stochasticity, and move `(1−α)||Q−Q0||` to the left. ∎ This is a worst-case bound, not an estimator of ε or η0. Lowering α weakens this perturbation control; it does not automatically repair semantics.

**From this teacher statement to selective risk and expansion.** Fix a probability measure ν on the same grid and domain as the intended claim, a candidate region C of mass c, and a bad semantic subset B of mass at most β inside C. Assume (4), or its conservative perturbation-adjusted version, holds outside B. Assume permitted class names and an active semantic checkpoint throughout C. Let an approximate teacher q differ from Q by at most e, and suppose outside B,

\[
L_i\ge\max\{\tau+e+d,\ (1+m)/2+e+d\},\quad d>0,\ m>0.\tag{6}
\]

Let a **fixed student** p have `D_w=Σνw KL(q||p)` and `w≥w0>0` throughout C. Restrict the usual probability/margin selector to C, calling its accepted set S. Define `ε_p=D_w/(2w0 d²)`. Then, whenever `c>β+ε_p`,

\[
\nu(S)\ge c-\beta-\epsilon_p,\qquad
\Pr_\nu(\hat y\ne y\mid S)\le
\min\{1,(\beta+\epsilon_p)/(c-\beta-\epsilon_p)\}.\tag{7}
\]

For an original anchor set Z, newly accepted mass outside Z is at least

`max(0, ν(C\Z)−β−ε_p)`.

**Proof.** For any coordinate difference larger than d, the two-group KL argument in Result 1 gives KL≥2d². Therefore the exceptional set `E={||p−q||∞>d}∩C` has ν-mass≤ε_p. At every point of `C\(B∪E)`, `p_y≥L−e−d≥τ`, and `p_y−max_{k≠y}p_k≥2(L−e−d)−1≥m>0`. Permission and status assumptions ensure acceptance with the correct class. At least `c−β−ε_p` mass is consequently accepted correctly; every accepted error lies in B∪E, of mass at most β+ε_p. Divide by ν(S), replacing it by its lower bound, and cap at one. Applying the same exclusion argument within C\Z proves expansion. ∎

For the repository's unrestricted exporter, (7) does **not** apply unless C covers its actual accepted set or additional accepted mass outside C is separately charged as potentially wrong. The theorem is an assumption contract, not a proposed new selector or architecture. If C or B can only be constructed by looking at target masks, it is not an admissible no-GT selection mechanism. If ν describes fixed TRAIN nodes, (7) says nothing about unseen patients; choosing population ν requires a separate generalization argument. If ν is patient-balanced or class-conditional, D_w must be recomputed under that measure. These restrictions prevent weighted-average fidelity from silently becoming a patient/RV risk certificate.

### Two distinct kinds of positive expansion

**Weak informative anchor crossing a hard threshold.** For P with all three rows `(1/3,1/3,1/3)`, α=.35, initial rows `(.625,.125,.125,.125)` and twice `(.985,.005,.005,.005)`, the initially below-threshold first node reaches `.781` in its correct coordinate. With strengths `(.5,1,1)` and η=.015, its h is `.7166667`, and (4) gives `.77675`; d=.02 leaves both `.7` probability and `.2` margin gates satisfied. This earlier fixture is expansion beyond an initially hard-accepted set, but its node already has informative source evidence. It is not neutral-node expansion, and neither informative source is wrong. The receipt preserves it under `checks.conditional_positive_expansion` with that explicit scope.

**Truly neutral node with one wrong source/edge.** Consider three abstract nodes with synthetic truth equal to the first abstract class y at all nodes. This is not an anatomical BG assignment. Let α=.35 and

\[
R_0=(1/4,1/4,1/4,1/4),\quad
R_1=(.995,.005/3,.005/3,.005/3),\quad
R_2=(.005/3,.995,.005/3,.005/3),
\qquad P_{i,:}=(.02,.97,.01)\quad(i=0,1,2).
\]

R0 contains no class contrast; R1 is an imperfect correct source and R2 is a confidently wrong source. Thus exactly one of the two informative sources is wrong, and its route from node 0 has nonzero weight .01. Every P row is nonnegative and sums to `.02+.97+.01=1`; each R row is nonnegative and sums to one. Since P has identical rows, `P²=P` and hence `P^t=P` for every integer t≥1. Therefore the geometric series gives `Kα=αI+(1−α)P`, so

\[
(K_\alpha)_{0,:}=(.363,.6305,.0065),\qquad
Q_{0,:}=(.7181083333333333,.0982683333333333,
 .0918116666666667,.0918116666666667).
\]

With strengths `(s0,s1,s2)=(0,1,1)`, h=.637 and `b=.0065/.637≈.01020408163`. Notice that half the informative source *count* is wrong, but only about 1.02% of propagated source influence is wrong. At η=.005, (4) gives

`L=.363/4+.6305·.995=.7180975`; exact `Q0,y=L+.0065·(.005/3)=.7181083333333333`.

The exact margin is `.61984`, entropy is `.9042739128732693` nats, and reliability `h[1−H(Q0)/log4]=.22148761052815352`; the entropy-based reliability lower bound is `.22141003391739125`. All three existing bank-style gates `.7/.2/.05` pass at the previously neutral node. This is proved by the closed form above and independently checked by converged iteration, kernel iteration, simplex checks, and separate strength propagation (`checks.neutral_expansion_with_one_wrong_source`). It does not rely on changing UNKNOWN to BG or lowering a gate.

For a same-grid perfect student p=Q, stipulated class permission and semantic status, nodes 0 and 1 pass the probability/margin gates and node 2 abstains. Under the uniform measure over these three nodes, accepted error mass is zero and newly accepted mass outside the informative-source set `{1,2}` is 1/3. Disjoint copies produce a positive-mass neutral region with the same calculation. The exact same-grid student need not be perfect: coordinate error at most .01 plus teacher numerical error at most 1e−6 leaves `L−.01−1e−6>.7` and conservative margin `2(L−.01−1e−6)−1>.2`. This verifies conditional feasibility of neutral-region expansion with wrong evidence; it does not demonstrate that the trained student meets that approximation budget or that actual class permissions are earned by a three-node fixture.

**Graph-family boundary.** Here `P00=.02`. The executed spatial-only graph has a unit self-loop and at most four cardinal-neighbor weights, each at most one (`E/core.py:121-143` with temporal edges disabled), so `Pii=1/(1+Σneighbor weights)≥1/5=.2`. Thus this fixture violates that spatial-only construction constraint. It establishes feasibility in the general row-stochastic restart family, **not a realizable workaround for the run graph**; no temporal-graph or anchor-generation realizability is asserted. Indeed the neutral-node ceiling below gives `.6982758621<.7` at self-transition .2, independently of how favorable neighboring targets are. The fixture's benefit comes from its strong correct-source influence and small self/wrong-source influence, and those conditions must not be silently assumed for real images.

Finally, a hypothetical region with `c=.3,β=.002,w0=.1,d=.02,D_w=2.4e−7` has ε_p=.003, coverage≥.295 and risk≤.01695 under (7). These positive examples are VERIFIED synthetic arithmetic under explicit assumptions, not measured anatomical guarantees.

**Limits that block blind application.** A neutral initial node has `R_i,c=1/K`, hence directly from the fixed point `Q_i,c≤α/K+(1−α)`. At K=4, α=.35 this is `.7375`, even before graph self-return and imperfect-anchor losses. With self-transition `P_ii=d0`, rearranging the i-th equation gives the stronger ceiling `[α/K+(1−α)(1−d0)]/[1−(1−α)d0]`. Neutral regions therefore cannot inherit arbitrary confidence under restart. Conversely, if every R_j=u, stochasticity gives Q=u on every graph; no graph improvement can invent a named signal. Hard clamping changes this operator and can also propagate a confidently wrong anchor perfectly. All these are algebraic consequences, not causal diagnoses of the run.

The bank additionally gates entropy-based reliability, whereas student export does not (`E/core.py:295-303` versus `E/student.py:154-157`). To apply a teacher acceptance claim, that gate must also be checked. For exact Q and h, if `Q_y≥L≥1/K`, concavity of entropy on the other K−1 coordinates gives `H(Q)≤−L log L−(1−L)log((1−L)/(K−1))`: at fixed Q_y the remaining mass has maximum entropy when uniform, and that maximum decreases for Q_y≥1/K. Thus h times one minus this quantity divided by log K is a conservative exact reliability lower bound. Approximate Q and h require separate numerical slack; (7) intentionally certifies student gates only.

## Result 3 — named risk needs semantic information; calibration is a separate assumption

**DERIVED, complete counterexample.** Suppose two admissible worlds have exactly the same observable images, features, geometry, graph, anchors, q, weights, and p, but exchange two ungrounded anatomical names on a region T. A deterministic pipeline using only those observations has the same outputs and accepted set in both worlds. At every accepted point of T the two truths differ, so the two error indicators sum to at least one. Integrating and dividing by positive accepted mass on T shows that at least one world's selective error is ≥1/2. Randomized pipelines obey the same inequality after averaging over their shared random seed. No statistic of fidelity, convergence, entropy, or selection can distinguish these worlds. If legitimate external anatomical knowledge excludes one world, that knowledge is precisely the semantic assumption that must be stated and tested; this is not an impossibility claim about all image-based anatomy inference.

For confidence calibration, write `s=max p`, `f=argmax p`. If the *true population* satisfies `E[1(f=Y)|s]=s`, then for the confidence-only set `Sτ={s≥τ}` with positive probability,

`P(f≠Y|Sτ)=E[1−s|Sτ]≤1−τ`.

**Proof:** apply conditional expectation given s, and use `1−s≤1−τ` on Sτ. This exact calibration premise is not implied by softmax or small KL. If selection also depends on class, patient, geometry, or other features, calibration conditional on s alone need not survive the filtering: a constant .8 confidence can be 80% correct overall yet entirely wrong in a retained subgroup containing the erroneous 20%. The proof then requires calibration conditional on the extra selector or direct selected-risk control. For this particular run, all permissions pass, and `pmax≥.7` implies second-largest≤.3 and margin≥.4, so the `.2` margin gate is redundant in exact arithmetic. Even here, the required true calibration has not been established. UNKNOWN is excluded from accepted risk but retained when reporting coverage and whole-image errors; it is never converted to BG.

## Closest primary results and the novelty they remove

Targeted search/read, not a systematic or exhaustive novelty survey. Primary full texts were retrieved on 2026-10-14; locators identify the actual mathematical statements read. Search-engine crawl dates were not treated as publication years. The six entries below contain theory/algorithm evidence, with Guo et al. explicitly empirical; none establishes cardiac correctness. No independent retraction/COI screening or external-model review is claimed.

| Primary source and inspected locator | Actual result/assumptions and applicable limit |
|---|---|
| [Zhu, Ghahramani & Lafferty, ICML 2003](https://pages.cs.wisc.edu/~jerryzhu/pub/zgl.pdf), §2 equation (5), §3.1, §5 equation (10), PDF pp.2–5 | Harmonic extension fixes boundary labels and solves `f_U=(I−P_UU)^{-1}P_US f_S`; the random-walk reading is first hit of labeled boundaries. Invertibility requires boundary reachability/no closed unlabeled component. The paper assumes noiseless labeled values for clamping and explicitly discusses attaching labeled dongles when labels are uncertain. Thus clamping, soft external evidence, and propagation beyond seeds are old. These boundary semantics differ from the executed all-node restart operator; neither supplies the correctness of anatomical anchor names. |
| [Zhou, Bousquet, Lal, Weston & Schölkopf, NIPS 2003 / proceedings 2004](https://proceedings.neurips.cc/paper/2506-learning-with-local-and-global-consistency.pdf), §2 equations (1)–(3), §3 equations (4)–(6), PDF pp.3–4 | `F(t+1)=a S F(t)+(1−a)Y` converges for `0<a<1`, with `S=D^{-1/2}WD^{-1/2}` and spectrum in [−1,1]. The limit is `(1−a)(I−aS)^{-1}Y`. The paper also explicitly substitutes row-stochastic P. This strongly anticipates restart propagation and the fitting/smoothness tradeoff. Repository α is the restart coefficient `1−a`, not their a. Their scalar prefactor can be dropped for argmax, but not for probability thresholds; the repository uses simplex targets on all nodes. |
| [Wei, Shen, Chen & Ma, ICLR 2021; arXiv 2010.03622](https://arxiv.org/pdf/2010.03622), Definition 3.1, Assumption 3.3, Assumption 4.1, Lemma 4.2, Theorem 4.3 and Appendix A.2 | Their named-label theorem assumes worst-class pseudo-label error `abar<1/3`, `(abar,cbar)` within-class expansion with `cbar>3`, and ground-truth transformation inconsistency≤μ. With `c=min(1/abar,cbar)`, a minimizer of their population disagreement/consistency objective has error≤`2 Err(Gpl)/(c−1)+2c μ/(c−1)`. It can beat the teacher when the RHS is smaller. The unsupervised result instead permits a class permutation. Therefore “consistency can denoise imperfect pseudo-labels” is not new. Weighted soft CE plus feature cosine does not establish their objective, classwise starting accuracy, transformation separation, or global minimization. These premises remain BLOCKED here. |
| [Geifman & El-Yaniv, NeurIPS 2017](https://proceedings.neurips.cc/paper_files/paper/2017/file/4a8423d5e91fda00bb7e46540e2b0cf1-Paper.pdf), §3 Lemma 3.1, Algorithm 1, Theorem 3.2, PDF pp.3–4 | The paper states simultaneous selected-risk bounds for its SGR iterations from a labeled iid sample, using binomial inversion and a `δ/ceil(log2 m)` allocation; it allows zero coverage when the request is infeasible. Theorem/proof was read, not re-proved here. Risk/coverage selection and confidence rejection are established. It requires labeled calibration evidence and its sampling/selection contract, not entropy confidence alone. Correlated pixels from 18 patients are not m independent trials, and the theorem is not permission to tune on exposed VAL or repeatedly inspect TEST. |
| [Guo, Pleiss, Sun & Weinberger, ICML 2017](https://proceedings.mlr.press/v70/guo17a/guo17a.pdf), §2 equation (1), §4.2 equation (9), PDF pp.2 and 6 | Defines correctness calibration conditional on confidence and empirically studies miscalibration. Temperature scaling fits T using validation NLL; positive scalar T preserves the predicted argmax while changing confidence and therefore selective coverage. This is not a distribution-free theorem that raw or temperature-scaled outputs have bounded anatomical error. “Change confidence without changing the winning class” is not a novel semantic repair. Fitting T with target masks would violate the current threshold/selection boundary; it is not proposed as an authorized change. |
| [Natarajan, Dhillon, Ravikumar & Tewari, NeurIPS 2013](https://dept.stat.lsa.umich.edu/~tewaria/research/natarajan13learning.pdf), §2, Lemma 1, Theorem 3, Appendix A | Binary class-conditional independent flips with known rates `ρ+ + ρ−<1` admit unbiased corrected loss `[(1−ρ_{−y})ℓ(t,y)−ρ_y ℓ(t,−y)]/(1−ρ+−ρ−)`. Lemma 1 proves expectation equals clean loss; Theorem 3 controls clean surrogate risk via complexity and the corrected loss's Lipschitz constant. Noisy supervision correction is old, but its identifiability/noise model matters. Anatomical heuristic errors can be instance-dependent, correlated, and class-asymmetric; entropy reliability is not a noise-transition estimate or an unbiased loss correction. No correction is implemented here. |

The synthesis is narrow: a credible contribution would have to establish an admissible named-semantic signal and measurable robustness/coverage beyond these existing results. The proofs above are an application-specific accounting of standard convexity, Markov propagation, and selection arguments; priority or theorem novelty is not claimed.

## Two central research questions and three decisive diagnostics

The canonical questions are those in [PROBLEM_FORMULATION.md](PROBLEM_FORMULATION.md); the theorem and fidelity analysis here serve them rather than creating another question set.

**RQ1. Where does named anatomical information first become inadequate: source/readout, propagation, or learned prediction?** Frozen-checkpoint fidelity and common/native-grid controls distinguish approximation/export failure from an already inadequate named teacher. They are subordinate diagnostics, not evidence that the teacher's names are correct.

**RQ2. Under explicitly bounded source error and leakage, can named support expand while its additional semantic error stays within a declared budget, and can learning/export preserve that improvement?** The new neutral-node fixture establishes general-family feasibility with one wrong source; (4)–(7) specify the semantic, graph-influence, approximation, and selection assumptions whose applicability must be checked separately.

These are proposed diagnostic decisions for the human; no method, architecture, acronym, threshold, or supervision-policy change has been chosen. At most three decisive experiments are proposed:

1. **Frozen final-checkpoint fidelity and grid audit — NOT RUN.** On a predeclared TRAIN enumeration and unchanged checkpoint, record q, raw reliability, effective w, activation, common-grid p, and native p. Report fixed δ bins for (1), weighted and unweighted event masses, low-weight exceptions, and per-class/per-patient denominators. Compare direct native export to the existing diagnostic projection without adopting projection as a new policy. Recompute fixed-checkpoint KL; do not substitute epoch (3). A matched q/p record can falsify “near-optimal fidelity explains all abstention,” while raw native deviations expose a grid explanation. The archive contains hashes and summaries, not the tensor/weight payload needed to execute this here; no download is requested or performed.
2. **Controlled anchor/graph stress experiment — only bounded arithmetic checks executed here.** Hold synthetic images' semantic truth, graph budget, and anchor support fixed; compare the existing restart operator and classical clamped extension, plus wrong-name and wrong-edge controls. Vary route-weighted contamination, not just seed counts; report true correctness/coverage and (4)–(7) margins. Keep one label-permutation control with identical observed confidence. The supplied checks establish existence/counterexamples and perturbation arithmetic only, not a full matched experimental screen. If a comparator improves coverage while inheriting wrong names, operator repair is insufficient. No target-mask anchor injection is allowed.
3. **Frozen stagewise semantic localization on exposed VAL — NOT RUN beyond reading published aggregates.** If authorized existing stage outputs become available, freeze image-derived anchors, propagated q, and student outputs before evaluation; compare each fixed stage with VAL references only in the evaluator. Use the same patients/classes and existing gates, report RV even when undefined, coverage, errors, and coded/uncoded geometry strata. No reference may enter naming, permission, thresholds, checkpoint choice, or training; no TEST evaluation. This would be declared reference-informed development, not an untouched validation confirmation. The current summary cannot determine which stage first loses anatomical meaning. No new stage masks or scores were created by this worker.

## Delivered checks and unresolved application assumptions

`THEORY_PRIOR_ART_CHECKS.py` is a standard-library-only CPU arithmetic script with fixed seed 20261014. It writes only `THEORY_PRIOR_ART_CHECKS.json`; it imports no repository code and accesses no dataset, model, or masks. It checks 20000 simplex pairs, 1000 restart perturbations, explicit zero-weight/grid counterexamples, separate weak-anchor and truly neutral-node expansion fixtures, one wrong source's influence and all three gates, the spatial-only graph boundary, the active-batch scalar identity with a denominator floor, reported CE/H/KL arithmetic, all 11 executable-source hashes against two archived manifests, and patch identity. Status: **PASS_SYNTHETIC_AND_ARTIFACT_ARITHMETIC_ONLY**. Proofs, rather than sampled checks, establish the mathematical statements.

During this follow-up, the available independent `ADVERSARIAL_REVIEW_CHECKS.json` receipt was read. Its `route_corruption_counterexample` has 99 correct anchors and one wrong anchor but routes the neutral node entirely to the wrong one; this supports retaining route-weighted b rather than substituting a source-count error rate. Its `epochs.patient024` records a historical baseline batch with weight sum about `9.76e−14` and denominator about `1e−8`, illustrating that the floor distinction is substantive; it does not establish floor incidence in the new run. Its `epochs.last_active_batch_mean_kl` agrees with the unconditional scalar mean above. These are the independent review's reported checks, not this worker's rerun or approval of the amended report; its reviewed-draft hashes predate these amendments.

Still BLOCKED: actual same-grid final-checkpoint divergence under patient/class measures; target confidence tails; real anchor correctness and route leakage; an admissible observable region C and β bound; unseen-patient generalization; calibrated correctness; and causal attribution for no RV. These are the deciding evidence gaps, not reasons to add a larger architecture. The final method remains a human choice under the stated no-GT boundary.
