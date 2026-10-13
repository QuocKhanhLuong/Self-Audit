# Self-Audit continuation: propositions, proofs, and evidence cards

Date: 2026-10-13. Scope: continuation of the 2026-10-12 theory report. No new architecture is selected. The three retained new findings are N1–N3; A–D are independently checked prerequisites, not new contributions. All logarithms are natural. A numerical example checks a construction; it does not prove a general result or establish cardiac accuracy.

## 0. Objects and notation

For a finite graph with n nodes and C=4 named classes, let u=(1/4,1/4,1/4,1/4). P is nonnegative and row-stochastic. Thus node i receives information from j when P_ij>0; a random walk starting at i follows i→j toward possible information sources. Put β=1−α, 0<α≤1. Matrix max norm means the largest absolute entry; the induced infinity norm means maximum absolute row sum.

Anchor construction is R_j=(1−s_j)u+s_j V_j with s_j∈[0,1] and V_j a probability vector. Define S_v={j:s_j>0}, the vote-support set, and S_c={j:R_j≠u}, the contrast-support set. They need not coincide: balanced conflicting votes have positive strength but uniform R. “Neutral” below means R_j=u; it does not mean that no evidence was observed.

At any node, a bank accepts when maximum probability ≥τ, winner/runner-up gap ≥m, and reliability ρ≥r. Defaults are (τ,m,r)=(.70,.20,.05). UNKNOWN is a rejection action.

## 1. Re-audit of A–C: restart confines hard acceptance

**Plain-language result.** Distance from informative evidence and repeated restart toward uniform probabilities limit confidence even after exact convergence. The spatial claim is valid, but temporal edges invalidate its degree-four assumption. A separate return-walk argument covers the idealized default temporal graph. These results constrain bank probabilities and bank hard acceptance.

**Status:** DERIVED_RESULT using REPO_FACTS; deterministic operator guarantee. **Nearest prior:** classical random-walk resolvents and maximum principles, including [Zhou et al., NeurIPS 2003](https://dennyzhou.github.io/papers/LLGC.pdf). The threshold consequences are applications, not new propagation theory.

### A. Fixed point, distance, and numerical residual

For every such P and R, the map T(Q)=αR+βPQ has the unique fixed point
\[
Q^*=KR,\qquad K=\alpha\sum_{t=0}^{\infty}\beta^tP^t
             =\alpha(I-\beta P)^{-1}.
\]
K is row-stochastic. Let S contain S_c. If d(i,S) is the shortest directed path length from i to S, then
\[
\max_cQ^*_{ic}\le\frac14+\frac34\beta^{d(i,S)}.
\tag{A1}
\]
For no reachable anchor, the right-hand contrast is zero. At α=1, interpret the contrast factor as 1 at distance zero and 0 at positive or infinite distance.

**Proof.** Since every row of P sums to one, \(\|PZ\|_{\max}\le\|Z\|_{\max}\). Thus T is a contraction of factor β<1, including β=0. The convergent Neumann series gives the fixed point; nonnegative K has row sums α∑β^t=1. Write R=1u+D. Every coordinate D_jc≤3/4, and D_j=0 outside S. For t<d, no positive path reaches S, so (P^tD)_ic=0. Hence
\[
Q^*_{ic}-1/4=\alpha\sum_{t\ge d}\beta^t(P^tD)_{ic}
 \le(3/4)\alpha\sum_{t\ge d}\beta^t=(3/4)\beta^d.
\]
If d=∞ all terms vanish. The α=1 case follows directly from Q*=R. For any approximate Q, subtract Q* from T(Q), use the contraction inequality, and rearrange to obtain
\[
\|Q-Q^*\|_{\max}\le\|T(Q)-Q\|_{\max}/\alpha .
\]
This also proves the code's stopping criterion in exact arithmetic. □

**Sharpness and limits.** A directed chain of d neutral nodes pointing to a pure, absorbing anchor attains (A1). This is an admissible row-stochastic graph, but is not the repository's symmetric-weight graph. A disconnected neutral component stays uniform. Uniform R everywhere stays uniform. The result does not depend on semantic correctness; an erroneous anchor obeys the same bound.

At α=.35, d=1 gives .7375 and d=2 gives .566875. Thus candidate B is valid. The returned iterate's tolerance of 10⁻⁶ cannot bridge the distance-two gap under the real-arithmetic certificate. The source then casts Q to float32; no complete IEEE rounding proof is claimed.

### C. Local self-loop bound and temporal restriction

If R_i=u and t=P_ii, then for every class
\[
Q^*_{ic}\le B(t):=\frac{\alpha/4+\beta(1-t)}{1-\beta t}
=1-\frac{3\alpha}{4(1-\beta t)}.
\tag{C1}
\]

**Proof.** In the fixed-point equation, every other Q_jc≤1. Therefore
\((1-\beta t)Q^*_{ic}\le\alpha/4+\beta(1-t)\).
The denominator is ≥α>0. B is nonincreasing in t. □

A unit self-loop and at most four other weights ≤1 imply t≥1/5. Thus C is valid for the spatial-only graph, with B(1/5)=81/116≈.698275862<.70. The margin is about .00172414, larger than the nominal solver tolerance.

The configured temporal graph has up to six other neighbors, not four. Substituting t=1/7 in C1 gives .710629921, which does **not** establish rejection. Applying the degree-four proof to the default temporal graph would be invalid.

For completeness, the previous report's stronger temporal result survives re-derivation. Suppose W=Wᵀ, W_ii=1, at most six other neighbors, and every off-diagonal weight ≤1. Write d_i=∑_{j≠i}W_ij≤6. Then
\[
K_{ii}\ge\alpha[1+\beta P_{ii}+\beta^2(P^2)_{ii}],
\quad
(P^2)_{ii}=\frac1{(1+d_i)^2}
+\sum_{j\ne i}\frac{W_{ij}^2}{(1+d_i)(1+d_j)}
\ge\frac1{(1+d_i)^2}+\frac{d_i^2}{42(1+d_i)} .
\]
The last inequality uses d_j≤6 and \(\sum W_{ij}^2\ge d_i^2/6\). At β=13/20, the derivative of
\[
f(d)=\frac{\beta}{1+d}+\frac{\beta^2}{(1+d)^2}
+\frac{\beta^2d^2}{42(1+d)}
\]
is
\[
f'(d)=\frac{-\beta-2\beta^2/(1+d)+\beta^2d(d+2)/42}{(1+d)^2}<0
\]
on [0,6], since β·48/42<1. Consequently
\[
K_{ii}\ge\alpha[1+\beta/7+\beta^2/7]=3229/8000,
\qquad Q^*_{ic}\le1-\tfrac34K_{ii}\le.69728125.
\tag{C2}
\]
This proves exclusion of all neutral restart nodes for that graph family.

**Code versus exact abstraction.** Spatial cosines are clipped; temporal cosines are not clipped above one after float32 feature normalization. Thus C2 is an exact-arithmetic theorem for the graph formula, not a universal bit-level guarantee for every machine input. A useful certificate for a particular stored normalized P avoids that assumption:
\[
Q^*_{ic}\le1-\tfrac34\alpha[1+\beta P_{ii}+\beta^2(P^2)_{ii}].
\tag{C3}
\]
C3 follows directly from nonnegative return terms and stochasticity of K. It is observable and certifies an upper bound on the numerical bank probability, not anatomical error.

**Acceptance gates.** From Q=(1−h)u+hV̄ with h=Ks, p=max Q≥.70 implies h≥.60 and winner gap ≥2p−1≥.40. Define
\[
H_{\max}(p)=-p\log p-(1-p)\log((1-p)/3).
\]
Concavity of entropy gives H(Q)≤H_max(p), so ρ=h[1−H(Q)/log4]≥.1929661. Thus the default .20 margin and .05 reliability tests add no exact-arithmetic rejection after .70 probability passes. Independently stopped Q/h iterates have separate residual bounds; the argument concerns their common fixed point.

**Errata to preserved history.** The prior PROPOSITIONS file omitted a plus sign in three displayed multiline formulas: the expression for (P²)_ii, the definition of f(d), and the graph perturbation bound. The correct last formula is
\[
\|Q-\widetilde Q\|_{\max}\le
\|R-\widetilde R\|_{\max}
+\frac{\beta}{\alpha}\|P-\widetilde P\|_{\infty\to\infty}.
\]
It follows by subtracting the fixed-point equations and applying the contraction bound. The derivations and conclusions used addition. The previous file is left unchanged; these are explicit continuation errata.

**What A–C do not prove.** They do not bound neural generalization, soft-target gradients, or accuracy; do not establish a task-level impossibility; and do not prevent support changes inside S_c. Qualitative prediction: additional iterations cannot remove the neutral-node hard-bank barrier, while a change of boundary condition can.

## 2. N1 — Useful expansion is possible with imperfect anchors and imperfect graphs

**Plain-language result.** Neutral restart consumes evidence before it travels far. A classical clamped-boundary formulation can carry evidence arbitrarily far without that dilution. Correct expansion remains possible with both corrupted anchors and cross-class edges, provided their *influence*, rather than their count, is sufficiently small. This separates attainability from semantic grounding.

**Status:** DERIVED_RESULT / constructive positive example. **Guarantee:** deterministic operator result plus a conditional semantic result. **Additional to the prior report:** a realizable graph with positive accepted expansion beyond initial evidence, allowing both kinds of error; the prior corruption inequality alone did not show those conditions were attainable at unanchored nodes.

### N1.1. Absorption kernel and a corruption-tolerant margin

Let S be nonempty boundary nodes and U its complement in a finite graph. Assume every i∈U has a positive-probability directed path to S. Fix Q_S=R_S and solve
\[
Q_U=P_{UU}Q_U+P_{US}R_S.
\]
Then \(H=(I-P_{UU})^{-1}P_{US}\) exists, is nonnegative and row-stochastic, and Q_U=HR_S.

For each i∈U with latent true class y_i, define
\[
h_i=\sum_{s\in S}H_{is}s_s>0.
\]
Suppose there is a good set G_i⊆S such that V_{s,y_i}≥1−η_i for every s∈G_i, where 0≤η_i≤1. Assume
\[
\sum_{s\notin G_i}H_{is}s_s\le b_i h_i,\qquad 0\le b_i\le1.
\]
Bad sources include corrupted names, legitimately different anatomical classes reached by wrong edges, and ambiguous anchors. Then
\[
Q_{i,y_i}\ge L_i:=(1-h_i)/4+h_i(1-b_i)(1-\eta_i).
\tag{N1}
\]
If L_i>1/2 and
\[
L_i\ge\tau,\quad 2L_i-1\ge m,\quad
h_i[1-H_{\max}(L_i)/\log4]\ge r,
\tag{N1-A}
\]
the bank rule accepts i with its correct name, **if** its reliability is defined by the same h/entropy formula. That harmonic reliability is a comparison definition, not a claim about an implemented alternative.

**Proof of kernel existence.** Choose, for every i∈U, a finite positive-probability path to S, of length at most n. Their finitely many positive path probabilities have minimum ε>0. Starting at any surviving U node, the probability of avoiding S for the next n steps is at most 1−ε. Iteration using the Markov property gives row sums of P_UU^(kn) at most (1−ε)^k→0. Hence its Neumann series converges. H is nonnegative, and
\[
\sum_{t=0}^{N-1}P_{UU}^{t}P_{US}\mathbf1
=(I-P_{UU}^N)\mathbf1\longrightarrow\mathbf1.
\]
Thus H has row sums one. Substitution and invertibility prove uniqueness. H_is is also the first-hit probability at s, by summing paths that remain in U before entering S.

**Proof of margin and acceptance.** Expand Q_i=∑H_is[(1−s_s)u+s_sV_s]. Its neutral contribution to class y_i is (1−h_i)/4. Good sources carry at least (1−b_i)h_i strength, each contributing ≥1−η_i; discard nonnegative bad contributions. This proves N1. A coordinate larger than one half is uniquely largest; all competitors have total mass ≤1−L_i, yielding margin ≥2L_i−1. Entropy is maximized at a fixed winning probability by equally distributing the remaining mass, and H_max decreases above 1/4. Therefore N1-A implies all gates. □

No independence of anchors, pixels, or frames is used. This is a worst-case influence argument. Counts alone cannot replace the b_i assumption.

### N1.2. A feasible family with wrong evidence, leakage, and arbitrary distance

Make an undirected graph with a hub, three boundary leaves connected to it with conductances .8,.1,.1, and a dangling path of any finite length d≥1 attached to the hub. Add a self-loop of weight one to every node. Assign true LV to all nonboundary nodes. Boundary leaf 1 is true LV with V_LV=.98 and the remaining .02 equally split. Leaf 2 is a corrupted true-LV anchor claiming pure RV. Leaf 3 is actual RV, correctly claiming pure RV, joined through a cross-class edge. All boundary strengths are .98.

Every node on the dangling path has the same harmonic value as the hub. To see this, the endpoint's equation forces equality to its only other neighbor; induction along the path forces equality throughout. In the hub equation all path and self-loop terms then cancel. The remaining harmonic combination has weights (.8,.1,.1). Therefore every U node has h=.98, b=.20, η=.02 and
\[
L=.02/4+.98(.80)(.98)=.77332.
\]
The actual vector in BG/RV/MYO/LV order is
\[
(767/75000,\;15467/75000,\;767/75000,\;19333/25000).
\]
Its winning margin is about .567093 and reliability about .543045. All default thresholds pass. Every U node was initially unanchored; the far endpoint is d+1 steps from the nearest boundary anchor. Thus accepted correct expansion occurs at arbitrary finite distance with a corrupted anchor, a cross-class edge, and an imperfect good anchor. The construction works for any named class; disjoint copies for RV, MYO and LV provide simultaneous nonzero expansion for all three. This establishes feasibility in a constructed finite model, not the validity of its anatomical assumptions for a real patient.

This is a constructed world, not a model of all cardiac anatomy. Swapping its latent LV/RV identities while preserving observations defeats semantic correctness unless admissible priors exclude that world. A one-boundary wrong-anchor graph can produce confidently wrong output everywhere. With no path to S, uniqueness and normalized H may fail. Requiring a path from only the query node is insufficient: it could jump with probabilities .5/.5 to an anchor or a closed unanchored trap. Our assumption requires a path from every U node. With h=0 the output is uniform. With sufficiently weak anchors or large b, the acceptance conditions fail. Removing neutral restart can thus solve reach, while also spreading wrong evidence farther. These are bank-rule statements; student export additionally needs open status and class permissions.

**Effects of self-loops.** At any U node with P_ii<1, the harmonic equation divides by 1−P_ii. Holding nonself conductances fixed, changing self-loop weight changes holding time but not H. This cancellation does not occur in the restart equation because a new uniform term is injected on every visit.

### N1.3. What can actually be certified

H, h, residuals, and gate inequalities using proposed bounds are computable from permitted inputs. The true good set, b_i, η_i, and the identification of y_i as LV/RV/MYO concern latent correctness. They are **not** estimated by this theorem.

If these semantic assumptions fail only on a set of patient-weighted measure ζ, and a selector accepts solely where N1-A is certified using valid bounds, every accepted error lies in that exceptional set. Hence selective error ≤min(1,ζ/coverage). This proof is set inclusion followed by division; it needs no pixel IID assumption.

For a new region B, mass b>0, and old accepted region A, mass c>0, with old predictions preserved, error accounting is
\[
R(A\cup B)=\frac{cR(A)+bR(B)}{c+b}.
\]
Thus nonzero expansion plus a bound on R(B), rather than increased confidence alone, is the useful contract. The harmonic construction supplies existence; a real-data semantic bound remains absent.

**Closest prior / novelty boundary.** [Zhu, Ghahramani and Lafferty, ICML 2003, Sections 2–3](https://mlg.eng.cam.ac.uk/pub/pdf/ZhuGhaLaf03a.pdf) already derive the clamped harmonic solution and random-walk interpretation. Those mechanisms are not proposed as new. N1's value here is to close a logical gap between a corruption inequality and feasible accepted expansion, while making unobservable assumptions explicit.

**Qualitative prediction.** A classical boundary comparator can eliminate the distance barrier without improving wrong-anchor identification. More distant support can include more confident errors.

### N1.4. Removing attenuation does not provide uniform graph stability

Let two row-stochastic matrices P and P̃ share the same S/U split and both satisfy the absorption assumption. Write N=(I−P_UU)⁻¹. Then
\[
\|Q_U-\widetilde Q_U\|_{\max}
\le \|R_S-\widetilde R_S\|_{\max}
+\|N\|_{\infty\to\infty}
\|P_{U,V}-\widetilde P_{U,V}\|_{\infty\to\infty}.
\]
**Proof.** Subtract the two harmonic equations and solve using N:
\[
Q_U-\widetilde Q_U
=H(R_S-\widetilde R_S)
+N[(P_{UU}-\widetilde P_{UU})\widetilde Q_U
+(P_{US}-\widetilde P_{US})\widetilde R_S].
\]
H is stochastic and all entries of the tilded distributions lie in [0,1]. Taking the stated norms proves the bound. N1's path-sum argument also gives \(\sum_jN_{ij}=\mathbb E_i[\tau_S]\), the expected number of visits to U before absorption. □

The absorption-time factor can be arbitrarily large. For one U node with unit self-loop and two clamped pure different-class anchors, give both edges weight ε. Its harmonic probability is 1/2. Change one edge to 2ε: the probability becomes 2/3, while the normalized P-row perturbation tends to zero with ε. Thus no graph-uniform Lipschitz constant independent of escape/connectivity assumptions exists for this family. This is sensitivity to graph evidence, not uncertainty from an unfinished numerical solve. A positive claim must control erroneous influence or relevant connectivity/perturbation budgets, not just prove eventual absorption.

## 3. N2 — Numerical confidence does not preserve evidence state

**Plain-language result.** The same normalized distribution can represent no evidence, balanced conflict, or strong systematic error. The current reliability formula loses some of this distinction. Keeping additional evidence provenance can distinguish observable evidence histories; it cannot, by itself, establish which anatomical name is correct.

**Status:** COUNTEREXAMPLES / DERIVED_RESULT, information and decision scope. **Assumptions:** explicit vote-pooling abstraction verified in core.py; semantic world ambiguity is a restricted model-class construction. Exact realizability of every constructed vote history by the upstream anatomical predicates has not been proved. **Additional to prior work here:** algebraic collisions of the stored bank quantities and a distinction between numerical certification, evidence accounting, and semantic certification.

### N2.1. Collisions of bank summaries

For a one-node P=[1], Q=R and h=s. Let total vote mass be M and V the normalized votes, with s=1−exp(−M).

* No evidence: M=0 gives s=0 and R=Q=u.
* Contradiction: M>0, equal votes for all four classes, gives V=u and R=Q=u.

Both produce the identical tuple
\[
(R,Q,H(Q),\rho,\text{hard label})=(u,u,\log4,0,\mathrm{UNKNOWN})
\]
for arbitrarily large M. Thus no deterministic function of these stored quantities can distinguish those histories: equal inputs must give equal outputs. An independently retained M or h could separate them; the current bank schema saves neither per-node raw votes nor h. It saves image/features as well, so this is **not** a claim that the complete input record is identical or cannot be reanalyzed.

For weak consistent versus strong mixed votes, pick 0<a<b<1. The histories
\[
(s,V)=(a,e_c),\qquad
(s,V)=(b,u+(a/b)(e_c-u))
\]
have the same Q=(1−a)u+ae_c and entropy. Their reliability differs when Q≠u, so it would be false to claim that reliability never distinguishes evidence strength. It distinguishes the implemented pooled strength in that case, without identifying independence or correctness.

For strong correct versus strong wrong evidence, hold all permitted observations and vote histories fixed and choose two admissible latent worlds in which the proposed class at a selected region is correct or incorrect. Q, entropy, h, reliability, numerical residuals, and any deterministic audit remain identical. If the region has positive accepted measure and the labels are different throughout it, the sum of error indicators across the worlds is at least one pointwise. Therefore at least one world has selective error ≥1/2 there for an arbitrary possibly randomized classifier, or error one if it deterministically emits the fixed proposed class and the second world contradicts it. This is a limitation of a model class that admits both worlds, not a theorem that actual cardiac images never contain discriminating information.

**Different decisions.** No evidence and unresolved conflict can both legitimately lead to UNKNOWN, but require different evidence-acquisition or audit actions: seek initial information versus resolve contradictory sources. Correct and systematically wrong high-confidence regions require different semantic actions. Compressing them to the same statistic cannot supply either distinction. A requirement for different actions must itself be specified; the collision does not prove that every selector needs different outputs in every case.

### N2.2. Replication increases implemented strength without increasing information

Repeat the same unanimous vote of mass a exactly r times. Then
\[
s_r=1-e^{-ra},\qquad Q_{c,r}=1/4+(3/4)s_r.
\]
This is immediate from the anchor formula. With a=.4, one vote gives p≈.49726; three identical votes give p≈.77410 and ρ≈.30439, passing all defaults on a one-node graph. Replication did not change which observation generated the vote.

For a statistical comparison, let Y be equiprobable between two named classes. A witness returns Y with probability .8. Repeating that one witness three times gives conditional error .2 after unanimous output. Three conditionally independent witnesses of the same accuracy give conditional error
\[
\frac{.2^3}{.8^3+.2^3}=1/65
\]
after unanimous output. Bayes' rule proves both claims. The observed three votes, total mass, Q, and reliability can be identical in the two models. The difference lies entirely in dependence. These are explicit auxiliary assumptions, not established witness statistics.

The default three clustering scales may repeat partitions; their loop adds votes without an independence correction. A toy replicated-vote calculation does not assert how often this happens in real cines. Requiring two patient IDs blocks a single patient's count from authorizing a class, but does not eliminate a systematic error shared by both patients.

**Minimum information requirement.** Any audit that claims to distinguish the four states must preserve or reconstruct (i) unsupported mass separately from opposing support, (ii) source/class-specific support before normalization, and (iii) source identities or justified dependence/corruption bounds. These are information requirements, not a proposed uncertainty module. To certify semantics, the model class additionally needs a valid observation-to-anatomy assumption that excludes the indistinguishable wrong world, or a justified error budget supplied without forbidden target masks.

**Numerical versus semantic certificates.** A residual bound δ_n gives coordinate intervals [Q̂_ic−δ_n,Q̂_ic+δ_n] for the exact solution. Probability and margin pass robustly if their lower bounds exceed thresholds (a gap loses at most 2δ_n). Entropy/reliability require their own interval bound. None bounds anatomical error. Deterministic perturbation robustness also need not assume IID; agreement under a corruption budget can be proved while the common answer remains wrong.

**Closest prior / scope.** Selective prediction and noisy-label identifiability already separate assumptions from attainable guarantees. No general claim to invent evidence decomposition is made. N2 does not establish empirical calibration failure or the superiority of a richer uncertainty representation.

**Qualitative prediction.** Duplicate-consensus controls can alter authorization without adding independent evidence; missing and conflicting evidence can have identical saved reliability.

## 4. N3 — Bank fidelity, training support, and exported support are different contracts

**Plain-language result.** The student can receive semantic gradients outside hard accepted bank pixels. This leaves a route to generalization, but also means that the bank's theorem does not certify student output. A useful positive transfer bound must cover teacher error, margins, weakly weighted locations, the readout connection, and the prediction grid.

**Status:** REPO_FACT / DERIVED_RESULT / COUNTEREXAMPLE. **Guarantees:** source transition invariant, deterministic fidelity bounds, conditional semantic risk. **Additional here:** an explicit weighted-soft-loss transfer bound and a loss-grid nullspace counterexample, rather than only a feature-margin statement.

### N3.1. Re-audit of D and the full loss path

Let a_c be fixed permissions computed from hard accepted pixels in at least two training patients. In an active batch,
\[
w_i=\rho_i a_{\arg\max Q_i},\qquad
\ell_{\rm sem}=\frac{\sum_i w_i[-\sum_c Q_{ic}\log p_{ic}]}{\sum_iw_i}.
\]
The implementation uses a denominator clamped below by 10⁻⁸; the displayed formula applies when the sum exceeds that value. The batch is active only when global foreground permissions exist and at least one non-BG target argmax has positive w. The hard-pixel acceptance indicator is absent from w.

For native logits first interpolated to the target grid, the gradient there is proportional to \(w_i(p_{ic}-Q_{ic})\). Thus a rejected pixel with positive w can train. Permission for a class controls whether its *argmax rows* are weighted, not whether that channel appears in soft CE. Even a disallowed class can receive nonzero decoder-channel gradients from another allowed class's target row.

If all foreground a_c are false in a fresh valid stable-guidance run, every batch is semantically inactive. Source explicitly sets decoder/head gradients to None before AdamW; those parameters are skipped rather than receiving zero tensors subject to momentum/decay. Encoder/projector parameters may continue learning. The immutable bank and permissions never change; semantic-update count remains zero; checkpoint status is REPRESENTATION_ONLY; export immediately returns all UNKNOWN before a forward pass. Induction over batches proves this invariant. It concerns the verified fresh-run transition, not arbitrary manually altered checkpoints.

Frozen semantic parameters do not imply frozen latent logits because the encoder can change. For partially allowed classes, the invariant is instead that the exporter vetoes every disallowed argmax. The mutable v3 trainer recomputes image/region evidence and records bootstrap history, so an empty support state there has additional outgoing paths and is not covered by D.

### N3.2. Support sandwich and teacher bias

Let p_i and q_i be student and reference probabilities on the **same domain and grid**, with \(\|p_i-q_i\|_\infty\le\delta\). Assume the export status gate is open; use τ>.5, m>0 and fixed class permissions. Write g(q)=q_(1)−q_(2).

1. Student acceptance implies max q_i≥τ−δ.
2. If q's winning class c is permitted, max q_i≥τ+δ, and g(q_i)≥m+2δ, then the student accepts c.
3. If g(q_i)>2δ, its argmax is preserved.

**Proof.** Every coordinate changes by at most δ, so the largest coordinate changes by at most δ. Each winner/competitor difference changes by at most 2δ. The second condition leaves a positive margin m and probability ≥τ; the third leaves a strictly positive winner gap. Permission then supplies the final export gate. □

The strict inequality in part 3 matters: q=(.6,.4), p=(.5,.5), δ=.1 permits a tie and a changed tie-broken label. A teacher whose confidently wrong margin exceeds 2δ is copied wrongly. If every neutral teacher maximum is ≤.69728125 and δ<.00271875, a same-grid faithful student cannot newly pass .70 there. This is conditional on probability fidelity; the implemented feature loss does not establish it.

### N3.3. A positive weighted-soft-target transfer bound

Let ν be any normalized patient-weighted measure on a common prediction grid; no IID pixels are assumed. Let q,p be probability vectors, w≥0, and
\[
D=\int w(x)\operatorname{KL}(q(x)\|p(x))\,d\nu(x)<\infty.
\]
Let A be any student export set with c=ν(A)>0, w₀>0 and δ>0. Let t=argmax q under a fixed tie convention and
\[
B_T(A)=\nu(A\cap\{t\ne Y\}),\quad
F_\delta=\{g(q)\le2\delta\},\quad U_{w_0}=\{w<w_0\}.
\]
Then
\[
R_S(A)\le\min\!\left(1,
\frac{B_T(A)+\nu(A\cap F_\delta)+\nu(A\cap U_{w_0})
+D/(2w_0\delta^2)}{c}\right).
\tag{N3}
\]
Consequently, positive export coverage with small teacher error, adequate reference margins, limited low-weight export mass, and sufficiently small D yields a nonvacuous semantic guarantee. Each of those conditions is necessary to assess whether this particular sufficient bound is useful.

**Complete proof.** For any class j, merge all other classes. The log-sum inequality gives KL(q‖p)≥kl(q_j‖p_j). To see log-sum directly, normalize non-j q entries to weights and apply convexity of −log to their likelihood ratios. For fixed b∈(0,1), the binary divergence f(a)=kl(a‖b) has f(b)=f′(b)=0 and f″(a)=1/[a(1−a)]≥4. Integrating the second derivative between a and b gives f(a)≥2(a−b)². Boundary cases follow by limits, with infinite divergences allowed. Thus
\[
\operatorname{KL}(q\|p)\ge2\|q-p\|_\infty^2.
\]
On the event E={w≥w₀, ||q−p||∞>δ}, w KL>2w₀δ², so ν(E)≤D/(2w₀δ²) by integration. Outside E∪U_w₀∪F_δ, part N3.2 preserves argmax. A student error there must be a teacher error. Union-bound the four sets within A and divide by c. □

For fixed q and weights, weighted CE minus weighted target entropy equals D (or D divided by total weight for normalized CE). A small raw CE number alone should not be substituted for D. Actual training normalizes per batch, and inactive batches contribute no semantic loss; a population D on future patients is not provided by the finite training objective.

**Counterexample showing the weight term is indispensable.** Put w=0 on a positive-measure region B. Choose p correct wherever w>0 and arbitrarily confidently wrong on B. Then D=0 yet export may accept B with risk one. If w is positive but arbitrarily small on B, D can be arbitrarily small with the same error. This is relevant because export has no bank reliability gate. It does not mean that useful generalization at low-weight locations is impossible.

### N3.4. Feature approximation requires a readout bridge

Suppose a reference logit readout f has each coordinate L-Lipschitz; teacher logits equal f(z_T); student logits differ from f(z_S) by at most κ; and \(\|z_S-z_T\|_2\le e\). Then logit discrepancy is ≤Le+κ. A softmax coordinate has gradient l1 norm \(2p_c(1-p_c)\le1/2\). Integrating its gradient along the segment between logits proves
\[
\|\operatorname{softmax}(z^{\rm logits}_S)
 -\operatorname{softmax}(z^{\rm logits}_T)\|_\infty
\le(Le+\kappa)/2.
\]
This supplies δ in N3.2 when the assumptions hold. Teacher bias is still B_T in N3. The actual projected features and semantic decoder are different branches; no common named readout or κ bound has been verified. Zero feature loss is compatible with an arbitrarily permuted semantic head. Cosine loss is also invariant to positive feature rescaling, so it does not by itself bound Euclidean e. For unit-normalized vectors, squared distance equals twice cosine loss pointwise; averaging still does not give a uniform bound.

### N3.5. The grid nullspace blocks an automatic loss-to-export theorem

Let I:R^N→R^m be the per-class linear interpolation from N native pixels to m bank pixels, with N>m. The source semantic loss uses softmax(Iz), whereas export uses softmax(z). By rank-nullity there is nonzero v with Iv=0. For any class c, change native logits by \(z'_{\cdot c}=z_{\cdot c}+tv\), leaving all other channels fixed. Then Iz′=Iz, so every semantic loss on the bank grid is unchanged. At a location with v_j≠0, choose the sign of t so that t v_j→+∞: exported probability of c tends to one. Set the latent true class there to another class and assume c is permitted; the native output becomes confidently wrong.

This proves non-identification at the loss interface. It does not prove that the specific CNN realizes every nullspace perturbation or that such errors occur in real data. Architecture restrictions or native-grid regularity could remove some ambiguity, but require a separate argument.

A finite example uses bilinear 2×2→1×1 interpolation. Starting from constant logits log(.1,.1,.1,.7), add 20 times the checkerboard [[1,−1],[−1,1]] to RV logits. The coarse loss is unchanged to floating tolerance; two of four native pixels become >.99 RV. The actual guidance_loss function verifies this example. No model fitting occurs.

### N3.6. Three transition families

| Family | What can change support? | What remains invariant or unproved? |
|---|---|---|
| Fixed bank/student | Neural predictions may change and generalize; bank, targets and permissions do not. | No-foreground permission state stays closed; disallowed names cannot export; bank guarantees need N3-type transfer conditions. |
| Recomputed readout/permissions | Changed representations or partitions can expose image evidence already present; rescored readout may cross gates and change permissions. | A legitimate transition must actually exist in code. Improved anonymous structure does not identify names without the semantic bridge. |
| Student predictions recycled as evidence | Computation and acceptance may change; previously extracted information can be used differently. | No independent evidence is created by copying a prediction. Shared systematic errors can authorize themselves unless provenance and admissible corruption assumptions constrain the update. |

For example, confidence sharpening \(q_c\mapsto q_c^r/\sum_dq_d^r\), r>1, drives any unique winning class toward probability one, including a wrong winner. A spatially constant class distribution is invariant under any row-stochastic averaging; constant wrong one-hot output is a fixed point of averaging and sharpening. These counterexamples refute unconditional improvement. They do not refute self-training under valid class-expansion/separation assumptions.

**Closest prior / limits.** KL-based transfer, margin stability, classical self-training assumptions and selective-risk accounting are established tools. N3's contribution to this continuation is the precise source-level composition and its missing terms, not a new general learning theorem. Qualitative prediction: reduced feature/soft-target loss can coexist with unchanged semantic bias, unsupported exported regions, or changed native boundaries.
