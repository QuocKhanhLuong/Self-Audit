# Theory propositions, proofs, and adversarial checks

Date: 2026-10-12. Source target: remote main at `d5a45f31481f0a702a1f9d5d418889fe704e9105`. All statements below distinguish an exact mathematical model from finite-precision software and from unknown anatomical truth. No empirical training outcome is a premise.

The six evidence cards and source register are in [THEORY_FINDINGS.md](THEORY_FINDINGS.md). The architecture-independent information contract is in [PROBLEM_FORMULATION.md](PROBLEM_FORMULATION.md). These proofs establish consequences of explicit assumptions; they do not establish that real cardiac MRI satisfies those assumptions.

## P1. Named semantics require an observable distinction; abstention gives partial identification

An anonymous partition, its name assignment, and its correctness are different objects. Rejecting ambiguous locations can make a restricted task identifiable, but does not create information about their names.

**Status:** DERIVED_RESULT and COUNTEREXAMPLE. **Guarantee type:** identification, with a conditional positive result.

Let the complete allowed observation be \(o=(D_{\rm train}^{X,M},X,M,\Pi,E,\mathcal L)\), where \(\mathcal L\) is the provenance ledger. Include algorithmic randomness separately as \(U\), independent of the latent world. Fix a finite spatial-temporal domain \(\Omega\) with positive pixel weights \(\mu\) summing to one. Let \(\mathcal Y(o)\) be all latent fields compatible with a specified model class and this same observation. The class set is \(\mathcal C=\{\mathrm{BG,RV,MYO,LV}\}\).

**P1a (exact partial identification).** For a deterministic observation-only predictor, zero error at every accepted location for every \(Y\in\mathcal Y(o)\) is possible exactly on subsets of
\[
I(o)=\{v:\ |\{Y(v):Y\in\mathcal Y(o)\}|=1\}.
\]
Consequently its maximum universally zero-error coverage is \(\mu(I(o))\).

**Proof.** On \(I(o)\), predict the unique common label. Outside \(I(o)\), at least two compatible fields disagree. Whichever name the predictor chooses is wrong in at least one of them. Since each location has positive mass, accepting such a location violates the stated guarantee. Rejecting it is necessary. Summing weights gives the coverage assertion. This is a logical version-space statement, not an algorithm for enumerating anatomically possible worlds. \(\square\)

**P1b (bounded two-world obstruction).** Suppose \(Y_0,Y_1\) have identical allowed observations and disagree on \(D\subseteq\Omega\). For a common deterministic acceptance set \(A\) of mass \(c>0\), let \(R_b\) be selective error in world \(b\). Then
\[
\max(R_0,R_1)\geq \frac{\mu(A\cap D)}{2c}.
\]
For randomized predictions/selection the same result holds for the ratio of expected error mass to expected accepted mass, replacing \(\mu(A\cap D)\) and \(c\) by their expectations.

**Proof.** At each accepted \(v\in D\), a single named prediction cannot equal both distinct labels. Thus the sum of the two error indicators is at least one. Integrate over \(A\), and, if needed, over the shared random seed \(U\). The maximum of two error masses is at least half their sum. Observational equality gives equal coverage in the two worlds. Divide by that coverage. \(\square\)

**Smallest counterexample.** One observed location, two possible labels, identical intensity/metadata/features. Full acceptance has worst-world error at least \(1/2\) for a randomized predictor, and one for any deterministic predictor. A label-free score can be identically one in both worlds; it cannot certify error below this bound.

**A non-heart-ring construction.** Put two separated copies of a closed annulus, enclosed bright cavity, and adjacent bright right-side component in the same finite cine. Give the copies the same feature descriptors and synchronized intensity/motion profiles; retain the same orientation metadata. In world 0 only the first copy is cardiac and the second is a noncardiac structure; in world 1 reverse them. Outside them keep a common BG region. All currently inspected local predicates—hole, adjacency, contrast, right-side position, repeated frames—are identical. The worlds obey that limited predicate model, including a unique actual heart in each world. They differ in spatial anatomy, not just a global renaming: common BG outside the copies fixes the BG name while the copies switch between BG and foreground. This is a countermodel to sufficiency of those predicates, not a claim that actual patients contain exact duplicate hearts. A stronger full-body/acquisition model could exclude it, but that model must be stated and justified.

Orientation breaks some reflection/RV-side ambiguities when trustworthy. An enclosure relation distinguishes wall from cavity within a chosen candidate. Intensity can distinguish different appearance classes only under a stated class-appearance model; sign-free contrast alone does not determine tissue identity. Shared motion constrains correspondence, not organ naming. A deterministic pretrained encoder cannot distinguish two identical observed inputs; its upstream semantic knowledge can narrow the admissible model class only if explicitly admitted.

**P1c (robust restricted positive identification).** Let \(\mathcal H\) be a finite set of candidate anatomical assignments. A declared generative/prior model associates each \(h\) with an observable signature \(s(h)\in\mathbb R^d\), for example specified acquisition coordinates and role relations. Assume the actual assignment \(h_*\in\mathcal H\) satisfies \(\|z-s(h_*)\|_\infty\leq e\), and
\[
\min_{h\ne h_*}\|s(h)-s(h_*)\|_\infty>2e.
\]
Then \(h_*\) is the unique assignment compatible with error \(e\). Without separation, the set \(\{h:\|z-s(h)\|_\infty\leq e\}\) still provides the consensus region of P1a.

**Proof.** If another \(h\) were compatible, the triangle inequality would give \(\|s(h)-s(h_*)\|_\infty\leq2e\), a contradiction. Consensus follows by applying P1a to the remaining assignments. \(\square\)

**Adversarial review.** This positive result does not smuggle target masks into the algorithm, but it does require a semantic observation model linking class roles to signatures. That bridge is external knowledge and is not established merely by assigning the names RV/MYO/LV to three templates. Duplicate rings violate separation; uncertain orientation enlarges \(e\) and may destroy it. If the true assignment is absent from \(\mathcal H\), uniqueness can confidently select the wrong object. No universal cardiac identifiability or operational zero-risk certificate follows.

**Nearest prior:** El-Yaniv and Wiener (2010), agreement-region selective prediction, and observational equivalence arguments such as Locatello et al. (2019). Their theorems are not new here and the disentanglement theorem is not being transferred to segmentation. The task-specific consequence is the required semantic bridge and the feasible-coverage restriction under the actual allowed information.

## P2. Neutral restart and graph returns restrict accepted bank support

The relevant question is where a fixed operator can produce an accepted distribution, not whether its iterations converge.

**Status:** DERIVED_RESULT supported by REPO_FACT. **Guarantee type:** deterministic operator and acceptance constraints.

Let \(P\in\mathbb R^{n\times n}\) be nonnegative and row-stochastic, \(R\in\Delta_C^n\), \(\alpha\in(0,1]\), and \(\beta=1-\alpha\). Row \(i\) receives information from \(j\) when \(P_{ij}>0\); a random walk goes from \(i\) toward its information sources \(j\). Use the entrywise norm \(\|A\|_{\max}=\max_{ic}|A_{ic}|\); for graph perturbations use the induced row-sum norm \(\|A\|_{\infty\to\infty}=\max_i\sum_j|A_{ij}|\).

**P2a (classical fixed point and residual).**
\[
Q^*=KR,\qquad K=\alpha(I-\beta P)^{-1}
       =\alpha\sum_{k=0}^\infty\beta^kP^k.
\]
\(K\) is nonnegative and row-stochastic. Iteration \(Q_{k+1}=\alpha R+\beta P Q_k\) converges from any finite \(Q_0\), and
\[
\|Q-Q^*\|_{\max}\leq
\frac{\|\alpha R+\beta PQ-Q\|_{\max}}{\alpha}.
\]
Each coordinate of \(Q_i^*\) lies between the minimum and maximum restart coordinate among nodes reachable from \(i\).

**Proof.** Row-stochasticity implies \(\|PA\|_{\max}\leq\|A\|_{\max}\). Hence the update contracts by \(\beta<1\). The Neumann series converges absolutely, solves the equation, and has row sums \(\alpha\sum\beta^k=1\). For \(r=T(Q)-Q\), \(\|Q-Q^*\|\leq\|r\|+\beta\|Q-Q^*\|\), giving the residual bound. The series puts nonnegative mass only on reachable nodes, giving the coordinate maximum principle. When \(\alpha=1\), \(K=I\). \(\square\)

**P2b (distance attenuation).** Let \(u=(1/C,\ldots,1/C)\), and \(S=\{j:R_j\ne u\}\). Define \(d(i,S)\) as shortest outgoing \(P\)-path length, infinity if none. If \(|R_{jc}-u_c|\leq\Delta_c\), then
\[
|Q^*_{ic}-u_c|\leq\Delta_c\beta^{d(i,S)}.
\]
In particular
\[
\max_c Q^*_{ic}\leq 1/C+(1-1/C)\beta^{d(i,S)}.
\]
For a pair of coordinates, replace \(\Delta_c\) by \(\Delta_{cr}=\max_j|R_{jc}-R_{jr}|\) to bound \(|Q^*_{ic}-Q^*_{ir}|\). If no informative anchor is reachable, \(Q_i^*=u\).

**Proof.** In \(\alpha\sum_k\beta^kP^k(R-\mathbf1u)\), terms \(k<d\) vanish. Every remaining coordinate has magnitude at most \(\Delta_c\), since rows of \(P^k\) are distributions. The geometric tail is \(\alpha\sum_{k=d}^\infty\beta^k=\beta^d\). A reachable deterministic chain ending at a self-looping pure anchor attains the bound. This proves sharpness only in the unrestricted directed family. \(\square\)

A first-hitting refinement is \(Q_i^*-u=\mathbb E_i[\beta^\tau(Q^*_{Z_\tau}-u);\tau<\infty]\) off \(S\), where \(\tau\) first hits \(S\). To prove it, iterate the neutral equation \(Q_i^*-u=\beta(P(Q^*-\mathbf1u))_i\) up to \(\min(\tau,N)\); the unhit remainder is bounded by \(\beta^N\), hence vanishes. No independence of cine pixels is used.

For \(C=4,\alpha=.35\), the probability bounds at distance one and two are .7375 and .566875. Therefore distance at least two precludes the .70 bank threshold. Distance one is necessary, not sufficient. A margin threshold \(m\) also requires \(\Delta_{cr}\beta^d\geq m\) for an accepted winner versus its runner-up. These are upper bounds, not promises of attainable confidence.

**P2c (stronger obstruction for the actual graph family).** In exact arithmetic suppose \(P=D^{-1}W\), \(W=W^\top\), \(W_{ii}=1\), each node has at most six other neighbors, and \(0\leq W_{ij}\leq1\). At \(\alpha=7/20\), every completely neutral restart node satisfies
\[
\max_c Q^*_{ic}\leq\frac{22313}{32000}=.69728125<.70.
\]
Thus **no node with zero initial anchor strength can become a hard accepted bank label**, even through temporal edges.

**Proof.** Write \(d_i=\sum_{j\ne i}W_{ij}\in[0,6]\). Since \(K_{ii}\) includes nonnegative walk-return terms,
\[
K_{ii}\geq\alpha\{1+\beta P_{ii}+\beta^2(P^2)_{ii}\}.
\]
Here \(P_{ii}=1/(1+d_i)\). Also \(d_j\leq6\), and Cauchy–Schwarz after padding to six weights gives \(\sum_{j\ne i}W_{ij}^2\geq d_i^2/6\). Thus
\[
(P^2)_{ii}
=\frac1{(1+d_i)^2}
 \sum_{j\ne i}\frac{W_{ij}^2}{(1+d_i)(1+d_j)}
\geq\frac1{(1+d_i)^2}+\frac{d_i^2}{42(1+d_i)}.
\]
For \(d\in[0,6]\), set
\[
f(d)=\frac{\beta}{1+d}+\frac{\beta^2}{(1+d)^2}
 \frac{\beta^2d^2}{42(1+d)}.
\]
Its derivative is
\[
f'(d)=
\frac{-\beta-2\beta^2/(1+d)+\beta^2d(d+2)/42}{(1+d)^2}<0,
\]
because \(d(d+2)\leq48\) and \(\beta(48/42)=26/35<1\). Therefore
\[
K_{ii}\geq\alpha[1+f(6)]
=\alpha[1+\beta/7+\beta^2/7]=3229/8000.
\]
At a neutral node \(R_{ic}=1/4\); all other restart coordinates are at most one. Stochasticity of \(K\) gives
\[
Q^*_{ic}\leq K_{ii}/4+(1-K_{ii})
\leq1-\tfrac34(3229/8000)=22313/32000.
\]
\(\square\)

The source graph has four in-plane neighbors, at most one mutual match to each adjacent time frame, no cross-slice edge, a unit self-loop, symmetric insertion, and normalized-feature weights at most one in real arithmetic. This matches the proposition. The two-step return mass is crucial: ignoring the fact that evidence can walk back to the neutral node gives a weaker bound that misleadingly leaves temporal expansion possible.

The source implementation is floating point: temporal cosines are not explicitly clipped above one. P2c is an exact-arithmetic result for the implemented graph formula, not a machine-checked IEEE theorem for every possible input. The analytical gap is .00271875; the solve's error is bounded by \(10^{-6}\), plus construction/casting roundoff. Numerical effects would need to bridge that gap to change the conclusion. We did not formally bound every floating-point operation. The source solver's certificate concerns its normalized stored \(P,R\), not exact construction of \(W\).

**P2d (the default extra bank gates are redundant).** The anchor construction is \(R_j=(1-s_j)u+s_jV_j\), \(s_j\in[0,1]\), \(V_j\in\Delta_4\). Let \(h=Ks\) and \(\rho_i=h_i[1-H(Q_i^*)/\log4]\). If \(p_i=\max_cQ^*_{ic}\geq.70\), then
\[
h_i\geq.60,\quad
p_i-\operatorname{second}(Q_i^*)\geq.40,\quad
\rho_i\geq .1929661051.
\]
Therefore the default margin .20 and reliability .05 reject no additional exact bank distributions once probability .70 passes.

**Proof.** By linearity, \(Q_i^*=(1-h_i)u+h_i\overline V_i\), with \(\overline V_i\) a distribution when \(h_i>0\). Thus \(p_i\leq1/4+3h_i/4\). The runner-up is at most \(1-p_i\). Entropy at fixed maximum \(p\geq1/4\) is at most
\[
H_{\max}(p)=-p\log p-(1-p)\log((1-p)/3),
\]
because the remaining mass has maximal entropy when equally divided. This function decreases for \(p\geq1/4\). Substitution yields \(.60[1-H_{\max}(.70)/\log4]=.1929661051\). \(\square\)

The solver runs separately for \(Q\) and \(h\), so equality of their finite iterates is not assumed. Their residual certificates bound deviation from the exact common-resolvent argument. Both margins above the extra thresholds are large; no borderline equality is being used.

**P2e (stability is not accuracy).** For fixed \(P\),
\[
\|K(R-\widetilde R)\|_{\max}\leq\|R-\widetilde R\|_{\max}.
\]
More locally, corruption restricted to nodes \(B\), of coordinate magnitude at most \(\delta\), changes node \(i\) by at most \(\delta\sum_{j\in B}K_{ij}\). For two stochastic graphs and restart matrices,
\[
\|Q-\widetilde Q\|_{\max}
\leq \|R-\widetilde R\|_{\max}
\frac{\beta}{\alpha}\|P-\widetilde P\|_{\infty\to\infty}.
\]
**Proof.** The first bounds use stochasticity of \(K\). Subtract the two fixed-point equations, putting \(\beta P(Q-\widetilde Q)\) on the left. The remaining graph term is \(\beta(P-\widetilde P)\widetilde Q\), bounded by the row-sum norm because simplex entries are at most one. Divide by \(1-\beta=\alpha\). \(\square\)

One wrong pure anchor on a one-node graph is an exactly converged, maximally confident wrong answer. Strong wrong cross-class edges can spread such error. Increasing duplicate votes also increases numerical confidence without independent evidence: \(r\) repetitions of the same vote of mass \(a\) yield \(s=1-e^{-ra}\), so a unanimous restart coordinate is \(1/4+(3/4)(1-e^{-ra})\). The source loops over clustering scales without a duplicate-evidence correction; this is not a statistical sample-size calculation.

**Scope and stronger-claim counterexamples.** The directed chain in P2b admits .7375 at a neutral immediate neighbor, so P2c is not universal for row-stochastic graphs. Hard accepted support can grow relative to thresholding the soft anchors alone *inside* the nonzero-anchor set. Also, the student uses reliability-weighted soft \(Q\) on permitted argmax classes, not only accepted hard bank pixels. Hence P2c does not prohibit soft supervision outside that set or student generalization. It concerns the hard bank, whose counts authorize class permissions. At \(\alpha=0\), uniqueness and the residual denominator fail; that endpoint is excluded by code. Uniform restart gives uniform output; disconnected components receive no outside evidence.

**Nearest prior:** Zhou et al. (2003), random-walk/resolvent diffusion and maximum principles. Their main operator is symmetric-normalized and their \(\alpha\) is the diffusion weight; ours is row-normalized and \(\alpha\) is restart. The graph-return/default-threshold consequence and gate redundancy are derived applications, not claims to have invented propagation.

## P3. Absorption belongs to the permission transition, not the whole network

Representation learning can continue while all named output is structurally forbidden.

**Status:** REPO_FACT plus DERIVED_RESULT. **Guarantee type:** state invariant; a conditional authorization result.

Represent state by \(s_t=(\theta_t,\phi_t,B_t,a_t,H_t,O_t)\): encoder, semantic parameters, bank, class permissions, readiness/history, and optimizer state.

**P3a (frozen-bank invariant).** In the inspected stable-guidance training procedure, suppose the frozen bank has no foreground class accepted in at least two TRAIN patients. For every completed optimizer step and every completed epoch: foreground permissions remain false, semantic updates remain zero, and the final checkpoint has \(\mathrm{REPRESENTATION\_ONLY}\) status. Its exporter returns UNKNOWN everywhere for every input cine.

**Proof.** Class counts and \(a\) are computed once before the epoch loop from the immutable bank. The Boolean semantic switch is the OR of foreground permissions and is false. Every semantic loss call is inactive. The counter increments only for active calls, so it remains zero by induction. No loop transition rewrites bank counts or permissions. Inactive semantic parameters have gradients set to None before AdamW; the optimizer skips such parameters, including their decay/momentum updates. The checkpoint status is therefore representation-only. The first exporter branch returns the prefilled UNKNOWN array before model inference. \(\square\)

This invariant does not assert \(\theta_{t+1}=\theta_t\). The cosine feature loss can update the encoder. Even with fixed \(\phi\), \(\phi(E_{\theta_t}(X))\) can change. Merely zeroing gradient values would not prove parameter invariance with momentum/weight decay; the None-gradient skip is relevant. The exported state is absorbing only within this training/export transition system; an externally rebuilt bank is a different transition.

**Why this does not transfer to v3/CQA.** Its anonymous assignments and components are recomputed from the trainable encoder, losses remain active without semantics, and \(\mathrm{observe}\) updates TRAIN-only history after each successful optimization step. Therefore absence of current support is not by itself an invariant of its full state. Inference does not call observe, so an unready frozen v3 checkpoint cannot become ready from new inference images. Gate activation is latched, while current local support is still required for an update.

**P3b (nontrivial conditional finite escape).** Consider the v3 readiness state with update floor \(U_0\), required distinct samples \(m\), patients \(p\), and observations per sample \(r\). Suppose:

1. There is a set \(J\) of at least \(m\) TRAIN sample identities spanning \(p\) patients and retained by the bounded history policy.
2. For \(rB\) updates after time \(t_0\), the encoder trajectory stays in a region where each sample in \(J\) has the same hard component masks, and all its local candidate inequalities pass. One sufficient condition is finite-grid assignment margins and gate slacks bounded away from zero, with total subsequent continuous score perturbation less than the smallest slack and no hard ownership changes.
3. In every block of \(B\) updates each identity in \(J\) is visited at least once. Updates continue until \(U_0\).

Then readiness activates no later than \(\max(U_0,t_0+rB)\), counting \(t_0\) in the same update clock.

**Proof.** No eligible observation of an identity is reset by failure, its masks have IoU one on each revisit, and retention prevents eviction. Induct on visits: its count reaches at least \(r\) by the end of \(r\) blocks. The number of qualified identities/patients and update floor then satisfy the exact readiness predicates. The observe transition sets ready true. \(\square\)

There is also a sampling version without independent frames: if at each step, conditional on all past draws, each fixed identity is sampled with probability at least \(q>0\), the probability that any of \(m\) identities is absent from any of \(r\) consecutive blocks of length \(B\) is at most \(mr(1-q)^B\leq mr e^{-qB}\). Indeed the chance of missing one identity for \(B\) steps is at most the product of \(B\) conditional upper bounds \(1-q\); union-bound across identities and blocks. Under assumptions 1–2, this bounds failure of the visit condition. It is a bound on the sampling schedule, not on independence or correctness of cine witnesses.

**Adversarial review.** This result supplies a robust finite region and recurrence requirement, rather than claiming more optimization will create support. Existence and reachability of such a region are unverified. All-void assignments, discontinuous mask changes, gate-slack loss, history eviction, insufficient patient diversity, or only one visit invalidate the premises. Authorization is not anatomical correctness, does not ensure RV support, and does not guarantee a subsequent nonzero semantic gradient; bank agreement and current local gates remain necessary. Minimal structural escape requires an allowed support-affecting transition not itself disabled by absence of semantic support, plus reachability of a persistent eligible state.

## P4. Quotient edits have an oracle error floor and cannot infer anatomy from image risk alone

Grouping can repair fragmentation. It cannot divide a mixed atom or restore a missing pixel.

**Status:** DERIVED_RESULT and COUNTEREXAMPLE supported by REPO_FACT. **Guarantee type:** representation/attainability, not optimization.

Let disjoint nonempty atoms \(A_1,\ldots,A_m\) cover \(S\subseteq\Omega\), with immutable void \(V=\Omega\setminus S\). A quotient partitions atom indices into blocks \(B_1,\ldots,B_l\) and replaces each block by \(\bigcup_{a\in B_j}A_a\). Allowed merges restrict this family further through adjacency, audit predicates, size, and merge budgets. Every output is constant on each original atom; void cannot become accepted through this operator.

**P4a (error floor).** For a latent field \(Y\), write \(w_a=\mu(A_a)\), \(w_{ac}=\mu(A_a\cap\{Y=c\})\). For any quotient-constant named predictor with UNKNOWN permitted, and rejection cost \(0<\lambda\leq1\),
\[
L_\lambda\geq
\lambda\mu(V)+\sum_a\min\{\lambda w_a,\ w_a-\max_c w_{ac}\}.
\]
Without rejection on atoms, replace each minimum by its second term. This lower bound is attained by an unconstrained atom-wise predictor; actual merge/readout restrictions can raise it.

**Proof.** Rejection of atom \(a\) costs \(\lambda w_a\). Assigning class \(c\) costs \(w_a-w_{ac}\), minimized by its majority class. A quotient cannot assign different labels within an atom and has no extra degree of freedom over the atom-wise predictor. Add the unavoidable void cost. \(\square\)

Exact full-domain naming without rejection requires no positive-mass void, pure atoms, and a permissible grouping/readout that assigns their correct names. Pure atoms alone are not sufficient when the readout requires enclosures that the allowed merges cannot form. A two-pixel atom containing RV and LV has unavoidable one-pixel error if accepted as one class, regardless of audit compute.

**P4b (topological reachability).** Let \(U\) be the union of all pixels an admissible wall proposal may include, and let \(C_0\) be a candidate cavity disjoint from \(U\). If every cavity pixel has a path to the image boundary in the complement of \(U\), then no wall \(W\subseteq U\) has a filled-hole component containing any cavity pixel. If the condition holds for only a subset of cavity pixels, those pixels cannot contribute to the hole overlap.

**Proof.** Such a path also avoids every \(W\subseteq U\). Flood filling from the boundary reaches the pixel, so binary hole filling does not mark it as a hole. \(\square\)

A void corridor from the cavity to the exterior is a concrete obstruction when that corridor remains outside every possible wall union. A single missing wall pixel is not by itself a universal theorem about all other possible wall proposals; the path condition is the precise statement. Conversely, a closed ring fragmented only by atom IDs, with no missing support, can become one wall by allowed merges.

The stable readout explicitly computes binary filled holes and rejects a wall when none exists before its .65 overlap score. Thus “partial enclosure” relaxes overlap with an already closed discrete hole; it does not remove the closed-hole existence requirement. A soft score does not imply a soft topological feasible set.

**P4c (appearance improvement can destroy a true boundary).** Consider two equally sized true regions with scalar observations \(\mu_a+\epsilon\) and \(\mu_b+\epsilon\), independent zero-mean noise of variance \(\sigma^2\), and \(n\) fit pixels per region independent of test pixels. Fit separate means or a joint mean and evaluate average squared test error. Then
\[
\mathbb E[R_{\rm merge}-R_{\rm split}]
=\tfrac14(\mu_a-\mu_b)^2-\frac{\sigma^2}{2n}.
\]
**Proof.** Each split mean has variance \(\sigma^2/n\); test error is \(\sigma^2+\sigma^2/n\). The pooled mean has variance \(\sigma^2/(2n)\), and each region's bias is \(\pm(\mu_a-\mu_b)/2\). Subtract. \(\square\)

Different anatomical classes with equal appearance means have strictly improved expected image risk after merging, yet a forced common anatomical label is wrong on half their union. This counterexample works even with independent fit/test noise; removing correlation alone would not establish anatomy.

A deterministic two-fold version assigns means \(+e,-e\) to the fit/test halves of atom A and \(-e,+e\) to atom B. In either fold direction, separate prediction risk is \(4e^2\), pooled risk is \(e^2\), so the audited excess is \(-3e^2\). Repeating the same temporal witness does not change this. This is a counterexample to the risk implication, not a claim that every additional CQA boundary veto passes.

**Restricted positive implication.** Suppose pure candidate regions of different classes have scalar mean separation at least \(\Delta>0\), each region has mixture weight in \([w_0,1-w_0]\), and a bias-corrected risk estimate differs from the population merge bias \(w(1-w)(\mu_a-\mu_b)^2\) by at most \(\eta\). If the merge threshold \(\tau+\eta<w_0(1-w_0)\Delta^2\), an accepted merge cannot cross a true class boundary.

**Proof.** A cross-class merge has population bias at least the right side, so its estimate exceeds \(\tau\). \(\square\)

The class-appearance separation and error bound are not verified for cine tissues; CQA does not implement that bias correction or such a confidence bound. Small regions, equal-intensity tissues, selection on the same image, and temporal dependence defeat easy transfer. This is a conditional opportunity, not calibration of the current score.

## P5. Relational objectives have exact feasibility constraints, but the current per-image graph is not capacity-starved

Permutation invariance makes a loss indifferent to prototype names. It does not supply those names or guarantee a useful optimization path.

**Status:** DERIVED_RESULT plus REPO_FACT; objective itself is LITERATURE_RESULT. **Guarantee type:** algebraic feasibility and stationary-point analysis.

For fixed atoms, let \(p_a\in\Delta_K\) be pooled assignments and \(z_{ab}=p_a^\top p_b\). Positive relations seek \(z=1\), negative relations seek \(z=0\). The actual smoothed probability is \(\epsilon+(1-2\epsilon)z\), \(\epsilon=10^{-6}\). For an active sample with both strata, every term has minimum \(-\log(1-\epsilon)\), and the balanced mean has the same lower bound.

**P5a (attainability criterion).** Contract connected components of the positive-relation graph. The per-term minimum is jointly attainable if and only if (i) no negative edge lies within a positive component, and (ii) the resulting negative graph is \(K\)-colorable.

**Proof.** For distributions \(p,q\), \(p^\top q\leq1\); equality requires both to be the same simplex vertex. A positive edge at its minimum therefore forces its endpoints to that same vertex, propagating through a positive component. A negative edge at its minimum requires disjoint supports. Thus a negative edge inside a positive component is impossible. For other components, choose any coordinate in their nonempty support; adjacent negative components have disjoint supports and hence distinct chosen coordinates. This is a \(K\)-coloring. Conversely, assign the one-hot color to every atom of each component in a proper coloring; all positive/negative terms attain their minima. Isolated components with no positive edges also admit this choice. \(\square\)

Examples: \(A\sim B,\ B\sim C,\ A\not\sim C\) is inconsistent for every \(K\). Three pairwise negative atoms require at least three prototypes. These are abstract relational counterexamples; they must not be reported as outputs of the current single-call auditor.

**P5b (source-specific positive feasibility).** For one CQA call on 2-D four-connected atoms, its emitted positive components contain no emitted negative edge. Its contracted negative graph is planar and admits a coloring with at most six colors. Thus \(K\geq6\), including the configured \(K=12\), suffices for a free per-atom assignment to attain all relation targets within that call.

**Proof.** Initially positive components are singletons. Before each union, the code checks every original interface that would disappear and forbids the union unless that interface was positive; a negative interface is never positive because the audit intervals are disjoint. Induction prevents a negative relation from entering a positive component. The atom adjacency graph is a planar minor of the finite pixel grid: contract each connected atom and delete void vertices/duplicate edges. Each accepted merge is a further edge contraction. The negative graph is a subgraph of the contracted planar graph, so is planar. For a finite simple planar graph on at least three vertices (connect components by noncrossing edges first if necessary), Euler's identity and the fact that each face has at least three boundary edges give \(m\leq3n-6\); hence some vertex has degree at most five. The same holds for every subgraph. Repeatedly remove such a vertex, color the remaining graph inductively with six colors, and restore it using an unused color among its at most five neighbors. Graphs with fewer than three vertices are immediate. Apply P5a. \(\square\)

This proves existence in free atom probabilities including the simplex boundary. Finite softmax logits can generally only approach one-hot assignments, so the corresponding claim is about an infimum. It does not prove representability by shared network parameters across samples, correctness of relations, or convergence of SGD. It also shows why a generic “too few prototypes” objection does not explain the default single-image relation system.

**P5c (uniform stationary point).** With fixed nonempty relations and softmax logits for each pixel, if every assignment is uniform \(p=(1/K,\ldots,1/K)\), the coassignment loss gradient with respect to all logits is zero, even when positive and negative strata are both present.

**Proof.** For one relation, \(\partial z_{ab}/\partial p_a=p_b=\mathbf1/K\). Multiplication by the softmax Jacobian \(J(p)=\operatorname{diag}(p)-pp^\top\) gives \(J(p)\mathbf1/K=0\). Pooling and summing finitely many relations preserve zero. Affine smoothing makes loss derivatives finite and changes no conclusion. \(\square\)

For \(K=4\) and one relation of each kind, the loss is approximately .8369875, strictly above its feasible minimum. A stationary point need not be a minimum. In the actual recomputed atom pipeline, fully uniform assignments fail the .70 atom-confidence gate, so relations disappear and this channel instead returns zero loss. Other image losses may break the state; P5c does not prove full-network collapse or unavoidable optimization failure.

A common prototype permutation leaves every dot product unchanged. Atom-ID renumbering with the same physical sets preserves the source's physical-order tie breaking. Refining granularity changes pooled probabilities, edges, and stratum weights: duplicating one relation among unequal losses changes their mean from \((l_1+l_2)/2\) to \((2l_1+l_2)/3\). Therefore neither invariance implies granularity invariance.

**Nearest prior:** Hsu, Lv, Schlosser, Odom and Kira, ICLR 2019, equations (6)–(7), already use categorical inner products and binary cross-entropy. The planarity/anti-chaining implication is a source-specific analysis using classical graph facts, not a new relation loss.

## P6. Positive semantic guarantees need a corruption budget and a readout bridge

Imperfect anchors can still suffice. Feature matching can preserve useful predictions. Neither statement can remove the need to connect observations to anatomy.

**Status:** DERIVED_RESULT and COUNTEREXAMPLE. **Guarantee type:** conditional semantic and selective-risk bounds, with deterministic stability.

**P6a (imperfect-anchor sufficient condition).** Use P2's \(K\), anchors \(s_j,V_j\), and \(h_i=\sum_jK_{ij}s_j>0\). Normalize informative influence as \(w_{ij}=K_{ij}s_j/h_i\). Suppose that outside a bad set of total \(w_i\)-mass at most \(b_i\), anchor distributions put at least \(1-\eta_i\) on the true label \(Y_i\). Wrong-edge influence and corrupted anchors may both be in the bad set. Then
\[
Q_{i,Y_i}^*\geq L_i
=\frac{1-h_i}{C}+h_i(1-b_i)(1-\eta_i).
\]
In particular \(L_i>1/2\) implies correct argmax. For \(C=4\), acceptance and correctness follow if
\[
L_i\geq\tau,\qquad 2L_i-1\geq m,\qquad
h_i[1-H_{\max}(L_i)/\log4]\geq r,
\]
where \(\tau,m,r\) are the probability, margin, and reliability thresholds and \(L_i\geq1/4\).

**Proof.** The neutral mixture contributes \((1-h_i)/C\). At least \(1-b_i\) of informative influence contributes at least \(1-\eta_i\) to the desired coordinate; discard the nonnegative remainder. If that coordinate exceeds half, it beats all competitors. Its margin is at least \(2L_i-1\), and the entropy bound follows as in P2d. \(\square\)

For example \(h=.9,b=.05,\eta=.05\) gives \(L=.83725\): nonzero corruption and imperfect anchors are allowed. Counting bad anchors without weighting by \(w_{ij}\) is insufficient; a single wrong high-influence anchor can dominate.

Neither \(b_i\) nor \(\eta_i\) is estimated from label-free confidence by this theorem. They concern true anatomy. If the assumptions hold outside a subset of patient-pixel measure at most \(\delta\), a selector accepting only locations meeting the sufficient inequalities has selective error at most \(\delta/c\), where \(c>0\) is its coverage. This follows because all its errors lie in that exceptional subset. It is conditional evidence, not a label-free operational certificate.

**P6b (representation-to-prediction bridge).** Let teacher and student normalized features be \(z_T,z_S\). A shared reference logit readout \(f:\mathbb R^d\to\mathbb R^C\) has each coordinate \(L\)-Lipschitz in Euclidean norm. Let the actual student logits differ from \(f(z_S)\) by at most \(\kappa\) in max norm. If \(\|z_S-z_T\|_2\leq e\), then a teacher logit margin \(\gamma_T>2(Le+\kappa)\) guarantees identical argmax.

**Proof.** Each logit changes by at most \(Le+\kappa\). The winner-versus-any-competitor difference decreases by at most twice that quantity and remains positive. \(\square\)

For a patient-weighted distribution \(\nu\), define feature distortion \(D=\mathbb E_\nu[1-z_T^\top z_S]\). For a common accepted set \(A\), \(c=\nu(A)>0\), let \(B_T(A)=\nu(A\cap\{\widehat Y_T\ne Y\})\). If the readout discrepancy bound holds on \(A\), then for every \(e>0\),
\[
R_S(A)\leq\frac{
B_T(A)+
\nu(A\cap\{\gamma_T\leq2(Le+\kappa)\})+
2D/e^2}{c},
\]
clipped at one.

**Proof.** A student error can occur only at a teacher error, insufficient teacher margin, or feature discrepancy greater than \(e\). The unit-vector identity \(\|z_S-z_T\|^2=2(1-z_S^\top z_T)\) and Markov's inequality bound the last event by \(2D/e^2\). Union-bound and divide by \(c\). No pixel independence is needed. \(\square\)

This bound exposes teacher bias, readout mismatch, low margins, feature approximation, and low coverage separately. Training distortion is not automatically population distortion. The repository's feature projection and semantic decoder are distinct; its cosine loss does not establish \(\kappa\) or a common Lipschitz semantic readout. Hard clustering/topology also requires a stable combinatorial region before a local regularity argument applies.

**Counterexample.** Two equally weighted sites have labels RV and LV. Features distinguish them perfectly and the student copies them exactly, so \(D=0\). A swapped readout makes every named prediction wrong. Even with a common readout, a biased teacher transfers the same mistake. Conversely a perfect teacher and a freely permuted student head show why feature matching alone cannot bound \(\kappa\).

**P6c (safe incremental coverage).** Let \(A\subset A'\), \(B=A'\setminus A\), with masses \(c>0,b>0\), and leave predictions on \(A\) unchanged. If the old and new-region selective errors are \(r_A,r_B\),
\[
R(A')=\frac{cr_A+br_B}{c+b},\qquad
L_\lambda(A')-L_\lambda(A)=b(r_B-\lambda).
\]
Thus adding coverage improves rejection-cost loss exactly when \(r_B<\lambda\). To retain a target selective error \(\varepsilon\), it is necessary and sufficient that
\[
br_B\leq\varepsilon(c+b)-cr_A.
\]

**Proof.** Error masses add on disjoint sets. Previously rejected \(B\) cost \(\lambda b\); afterward it costs \(br_B\). Rearrangement gives the last inequality. If old predictions also change, add their change in error mass to both formulas; mere containment of acceptance sets is then insufficient. \(\square\)

**Dependence and audit scope.** An interface score is evidence about an image discontinuity; a same-region claim adds a latent grouping assumption; an anatomical name adds a semantic assumption. Detaching targets removes a computational gradient path, not common causes. Freezing a teacher removes student-to-teacher feedback, not inherited bias. Checkerboard fitting withholds pixels from a scalar mean fit, while the proposal already depends on the whole image. Different cine frames remain correlated and may share artifacts. Whole-patient cross-fitting can remove direct fitting on that held-out patient under patient independence, but cannot ensure that the resulting score measures anatomy. Repeating one erroneous witness \(m\) times retains its error probability \(\eta\), not \(\eta^m\).

**Nearest prior:** corrupted-label learning and self-training already have positive conditional results (Natarajan et al.; Wei et al.). Selective classification and conformal risk control already study risk/coverage with appropriate labeled information and sampling assumptions. The bound here is a deliberately transparent conditional bridge; it is not a claim to supersede those theories.

## Mathematical checks and limits of review

[math_sanity_checks.py](math_sanity_checks.py) and [MATH_CHECKS.json](MATH_CHECKS.json) record exact rational threshold arithmetic; exhaustive checks of all 1,099 undirected unit-edge graphs on one through five vertices (5,405 possible neutral nodes); a sharp directed-chain example; uniform/disconnected/alpha=1 controls; an actual 27-node graph-constructor case with six neighbors and a seven-node clique; actual source residual checks; one autograd calculation at uniform logits; a two-fold risk counterexample; and finite relation-coloring examples. No parameters were optimized. These checks can catch arithmetic or source-matching errors; the general statements rely on the proofs above.

Separate skeptical passes narrowed several tempting claims: distance one is not sufficient; a loose six-neighbor bound missed return mass; the support obstruction concerns hard bank labels, not every soft training target; frozen semantic parameters do not freeze latent predictions; finite-gate activation is not accuracy; a void seam needs an actual forbidden path; generic relation inconsistency/capacity counterexamples are not automatically attainable outputs of this auditor; and a feature margin bound requires a readout coupling that the current loss does not provide.

An independent Orca reviewer was attempted but did not execute: dispatch `ctx_02202bed71cf` failed at `agent_readiness` with `timeout`; its unused terminal was released. The skeptical review above is by the same collaborator, not an independent review or formal proof assistant. External mathematical review remains desirable.
