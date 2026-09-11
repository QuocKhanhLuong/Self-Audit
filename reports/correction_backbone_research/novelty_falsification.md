# Second-round novelty falsification: correction backbone proposals

Date: 2026-09-11. Independent red-team review after reading candidate_A.md, candidate_B.md, candidate_C.md and state_formulation.md. Source HEAD independently verified as **35aaba9335e0d8a0aa344356e20e93ae902b950d**. The proposal documents are working-tree research artifacts, not implementations certified at that commit. This worker owns only this report; no training, production code, configuration, tests, commits or pushes were performed.

**Decision: retain A as a control; narrow B to an unproven class-bound accepted-history hypothesis; reject C as a present core architectural contribution and defer its bounded experiment.** No architecture currently has evidence establishing publishable novelty. The strongest new counterexample found in this round is *Subtract or Replay?* for C; quality-gated ConvLSTM and medical context-feedback refinement also materially strengthen the elementary-component objections.

## 1. Verdicts against the actual proposals

| Architecture | Verdict | What survives | What fails the novelty test |
|---|---|---|---|
| A: ConvGRU correction core | **RETAIN engineering/control; REJECT architectural novelty** | A capacity/compute-matched recurrent reference on cached ConvNeXt features; accepted-state persistence may be useful. | Recurrence, mask feedback, predicted-evidence inputs and a binary acceptance gate are established mechanisms. Rejected-state storage cannot improve returned masks under the stated HALT policy. |
| B: candidate-class FIX commitments and previous-class REGRESS obligations | **NARROW** | Test whether accepted action history, bound to the affected classes, improves future correction at identical retained logits and reduces destruction of earlier fixes. A positive result could support a representation/objective contribution. | “Dual memory,” “correction memory,” “verified writes,” and “transactional neural state” are insufficient. B has not separated its split latent features from the exact same P/R maps feeding one GRU or a feed-forward reader with identical losses. |
| C: immediately preceding accepted event replacement, joint logit/latent transfer | **REJECT as current core claim; NARROW/defer experiment** | A bounded, reproducible support intervention under current-state output constraints, with separately demonstrated value of the transferred latent state. | Additive subtraction and immediate snapshot replacement are bookkeeping identities. A new primary preprint explicitly studies decrement versus replay and corrected-record amendment. Existing CC-restitution already supplies the local output-space mechanism according to the proposal. |

These are adjudications of the written mechanisms, not performance predictions. No lack of a search hit is counted as positive novelty evidence. Exact B or C equivalence to a published complete segmentation system was **not established** by this bounded search; that does not establish originality.

## 2. Mechanism-level primary evidence

The following summaries are deliberately limited to what the independently accessed source supports. Peer-reviewed status, preprint status and access limitations are separated. Published experimental numbers are not imported as evidence for Self-Audit.

| Primary source and status | Actual mechanism and inspected location | Consequence for these proposals |
|---|---|---|
| [Skip RNN: Learning to Skip State Updates in Recurrent Neural Networks](https://arxiv.org/pdf/1708.06834), ICLR 2018; full paper inspected | Section 3, Eq. (3): \(s_t=u_t S(s_{t-1},x_t)+(1-u_t)s_{t-1}\), with binary update/copy decision. | Exact algebraic counterexample to treating state_formulation Eq. (1) itself as a new state-update primitive. Skip RNN learns its gate and can continue after a skip; Self-Audit audits a proposal and HALTs on rejection. These policy differences are real but do not invent the gated recurrence. |
| [Quality-Gated Convolutional LSTM for Enhancing Compressed Video](https://arxiv.org/pdf/1903.04596), 2019 arXiv version; full method inspected, venue not independently established here | Section 3.1, Eqs. (9)–(12): forget/input weights are obtained from quality-related features; Eq. (10) makes them complementary. In normalized notation \(C_n=(1-g_n)C_{n-1}+g_n\widetilde C_n\). | A stronger component match than a generic RNN: externally estimated quality regulates recurrent memory retention/write. Its bidirectional compressed-video setting differs from causal same-image correction; it does not implement accepted class obligations. |
| [Learning With Context Feedback Loop for Robust Medical Image Segmentation](https://arxiv.org/pdf/2103.02844), Girum et al., IEEE TMI 2021, DOI 10.1109/TMI.2021.3060497; author full paper inspected | Section II.C, Eqs. (3)–(5): encode the predicted mask into feedback features \(h_{f_i}=F_e(\hat y_i)\), then decode using image features and that feedback. Section II.D distinguishes its training procedure. | Mask-conditioned feature encoding and repeated medical correction are prior art. A/B cannot claim that feeding an annotation representation back into segmentation is new. This is a relevant matched adaptation, not proof that its training schedule equals the proposed one. |
| [Correction-aware interactive 3D tumor segmentation with sparse and revisable prompts](https://link.springer.com/article/10.1007/s00371-026-04560-5), Li and Li, *The Visual Computer*, 2026; publisher full text inspected | Section 3.2, Eqs. (2)–(3): choose a quality-scored candidate then apply a learned error-gated residual. Section 3.5, Eqs. (7)–(10): latest nonempty positive/negative instruction plus cumulative revision map, concatenated with image and selected mask. The paper explicitly says the residual gate is not a calibrated error probability. | Closest direct threat to B's correction-history maps, semantic revision and local residual update. Its memory binds human/simulated prompt instructions, not autonomously accepted transitions creating old-class restitution obligations. Do not describe its score as a post-refinement transition audit: candidate selection precedes residual refinement. |
| [SENTRY: SAM2-Enhanced Neighbor-Aware and Temporally Reasoned Memory for Visual Tracking](https://arxiv.org/html/2606.24449v1), June 2026 preprint; full method inspected | Section 3.2, Eq. (4), Algorithm 1: backward candidate tracklets, forward neighbor context and mean box-IoU trajectory similarity support matching before future memory use. Threshold rules can select a Kalman fallback. | Strong counterexample to broad verification-before-memory language. It selects/refines memory candidates; it is not an exact binary commit/rollback of learned same-image correction state. Algorithm 1's update/output ordering and the prose are not sufficient grounds to claim mask/state transactional equality. No B-style class obligation writer was demonstrated in the inspected method. |
| [MAIS: Memory-Attention for Interactive Segmentation](https://arxiv.org/html/2505.07511v1), May 2025 preprint; full method inspected | Sections 2.1.1–2.1.4: encode image once; FIFO stores sparse prompt and dense mask embeddings; memory attention conditions image features before decoding. These subsections give prose mechanisms, not numbered equations. | Stable visual features plus evolving medical correction memory are already explicit. The accepted autonomous FIX/REGRESS semantics would be the proposed extra ingredient, not caching or remembering masks. Human/simulated corrective input gives a different information budget. |
| [SAM2Long](https://arxiv.org/html/2410.16268v2), full v2 method inspected; [ICCV 2025 proceedings record](https://openaccess.thecvf.com/content/ICCV2025/html/Ding_SAM2Long_Enhancing_SAM_2_for_Long_Video_Segmentation_with_a_ICCV_2025_paper.html) | Section 3.2: \(S_{p,k}[t]=S_p[t-1]+\log(\mathrm{IoU}^{p,k}_t+\epsilon)\), expand and retain top pathways. Section 3.3 selects memories satisfying predicted IoU threshold and positive object-presence score, then modulates keys. Displayed equations in inspected HTML are unnumbered. | Both alternative memory trajectories and quality-filtered memory are established. Its final best-path selection can use later video evidence; it must not be advertised as the same causal HALT policy. Count its extra hypotheses when adapting it as a control. |
| [StreamDAM: Presence-Aware Memory for Real-Time Streaming Video Object Segmentation](https://arxiv.org/html/2608.03912v1), August 2026 preprint; full method inspected | Sections III.D–E, Eqs. (3)–(6): a GRU estimates presence; a threshold controls memory eligibility and hysteresis controls read recency. The signal also controls output suppression/redetection. | Another close recurrent-critic/memory-control precedent. Eq. (5) defines eligibility in the conditioning set, so “rejects every physical memory write” would overstate it. Presence is different from class-specific accepted-transition quality. |
| [DREAM: Dual-Memory Reasoning for Anticipatory 3D Perception in Autonomous Driving](https://openreview.net/pdf/a033cc4979b05be1f7f285f72a259e6378a133b1.pdf), anonymous CVPR 2026 submission copy; **venue/acceptance unverified** | Independently retrieved primary-host indexed passages: abstract, Fig. 1 and Section 3.4, Eq. (8), defining an error-memory bank for unreliable predictions. Direct full-PDF open was blocked by browser challenge; direct HTTP returned 403. | Broad latent/perceptual plus error-memory naming is threatened. Full update semantics, calibration and complete class/event provenance could not be verified here. Do not present DREAM as accepted CVPR work or assert it implements B1 from these excerpts. |
| [FocalClick: Towards Practical Interactive Image Segmentation](https://openaccess.thecvf.com/content/CVPR2022/papers/Chen_FocalClick_Towards_Practical_Interactive_Image_Segmentation_CVPR_2022_paper.pdf), CVPR 2022; primary-host indexed method inspected | Section 3.1, Progressive Merge: threshold the proposed mask, find the changed component containing the click, replace that region and retain the previous mask elsewhere. Section 3.2 evaluates correction of preexisting masks. Direct PDF opens failed, including an arXiv size limit; the primary indexed method passage was accessible. | Repair while preserving other mask regions is prior art. It does not make preservation depend on an accepted autonomous event's bound class. A predicted-region adaptation is an essential inexpensive control, with its changed prompt source disclosed. |
| [Subtract or Replay? Exact Deletion from Language-Model Memory](https://arxiv.org/html/2607.27539v1), Vishwajith Ramesh, July 30, 2026 preprint; full method inspected | Sections 3.1–3.2, Eqs. (1)–(3) define counterfactual output equality, record-addressable decrement, and \(\delta_x(S)=B(P,x,S)-B(P,S)\). Section 7 and Appendix G study suffix dependence, checkpoint replay and corrected-record amendment. | **Strongest new C counterexample.** Addressable subtraction, the failure of a fixed receipt after state-dependent descendants, and joint downstream/state consistency are already explicit. Different domain and intervention: record deletion/amendment, not bounded segmentation support optimization with local FIX protection. That distinction may motivate an experiment but cannot rescue a general replay/subtraction novelty claim. |

The C counterexample is a recent preprint and its numerical claims were not reproduced. Its disclosed mechanism is still relevant prior art. The same standard applies to SENTRY and StreamDAM. Older peer-reviewed gated recurrence and feedback/preservation methods already defeat the broad component claims without depending on recent preprints.

## 3. Causality, post-audit writes and HALT

The stated causal order is sound:

\[
(V,L_t,S_t,e_{t-1})\longrightarrow(\widetilde L,\widetilde C)
\longrightarrow(e_t,q_t)\longrightarrow b_t
\longrightarrow(S_{t+1},L_{t+1}).
\]

B's post-audit payload may depend on \(e_t\), because it first affects the **next** proposal. That is not circular inference. It would become circular if the current mask were decoded again from the post-audit state and returned using the old audit. Every changed candidate mask needs its own audit.

However, “audited state” is too strong. The Auditor sees the old/new annotations and visual evidence, not all information in \(W_R,W_P\). It verifies neither class-obligation truth nor future utility. A malicious or poorly trained writer could keep the currently accepted logits unchanged while writing arbitrary latent content; only a future proposal exposes that content to the annotation gate. Call these **writes conditioned on an accepted transition**. The acceptance gate alone does not certify useful hidden state.

The HALT equivalence lemma is correct as written. Inductively, two systems with identical accepted prefixes generate the same candidate and gate. At first rejection both return the same previous annotation, and no subsequent operation observes the different rejected latent tensor. Thus rejection rollback has zero mask-output benefit under its assumptions. A learned sigmoid GRU need not achieve an exact binary endpoint at finite parameters, but a gated recurrent update with an explicit branch does; the novelty objection does not depend on numerical sigmoid saturation.

Observable differences can arise if rejected tensors enter later training, saved state, another row, replay, random-number scheduling or mutable normalization. Those differences invalidate the comparison's assumptions; they do not prove the original lemma false. Final-turn post-audit writes are similarly unobservable when no later proposal reads them. Under a three-turn cap, report how many eligible writes actually have a subsequent decision, rather than assigning positive future-utility targets to terminal writes.

## 4. B: class semantics are meaningful, but neither calibrated nor architecture-identifying

For clarity, \(P^k,R^k\) below are B's class maps; C's coordinate support uses the same letter P for a different object in its document.

B1 has a meaningful hard-event interpretation:

| Accepted local event | Binding/action | Qualification |
|---|---|---|
| FIX with candidate class k | Set preservation to candidate class k; clear repair obligation for k. | A predicted FIX is evidence, not proof that k is true. |
| REGRESS with previous class j | Remove preservation at that location; add obligation to recover j. | Globally accepted improvement can contain local regressions. |
| No qualified FIX or REGRESS | Identity map write because the revised default sets f=r=0. | The unthresholded soft ablation lacks this identity; its counterexample is below. |
| Wrong class j changes to wrong class k | No guaranteed repair obligation under the GT transition taxonomy. | “UNCHANGED” does not mean unchanged class or safe region. A0 errors also lack event provenance. |

The class distinction is stronger than plain positive/negative prompts: an obligation says which previously held class should be restored. Nevertheless, sums clipped into confidence maps compress event history. A single age/identifier per location cannot describe several concurrent soft contributions or several earlier same-class losses. **Resolved during this review:** B now explicitly defines per-class/map/location last-touch metadata, not the origin of every soft contribution, and disclaims an event-resolved causal ledger. That is a coherent bounded summary; it must not later be promoted into exact action provenance.

### Concrete raw-soft counterexample; default writer now guarded

The initially inspected unthresholded writer, now retained only as an ablation, admits the following failure. At a pixel with unchanged predicted class k, let each accepted global turn have \(f=r=\epsilon>0\), representing small erroneous local event probabilities. Other pixels can make the global transition acceptable. Starting from empty maps, B1 becomes

\[
P^k_{n+1}=(1-\epsilon)^2P^k_n+\epsilon,\qquad
R^k_{n+1}=(1-\epsilon)R^k_n+\epsilon.
\]

Thus its fixed points are \(P^k=1/(2-\epsilon)\) and \(R^k=1\), despite no class change. This limiting example is analytical, not a proposal to extend the rollout. Already at \(\epsilon=.05\), three writes produce approximately \(P^k=.13585\), \(R^k=.14263\). Existing preservation also decays at accepted turns lacking a genuine event. For a categorical audit, \((1-r)(1-f)=u+fr\), not simply the UNCHANGED probability u; these strengths are heuristic weights, not automatically calibrated posteriors.

**Resolved at specification level in the final reread:** B now qualifies events by unique audit argmax, development-calibrated confidence thresholds and a changed hard class, with ties declared no-event. It sets f=r=0 when no event qualifies, makes Psi zero at zero ledger delta, and freezes both persistent C branches when no event qualifies anywhere in the row. Tentative scratch features may still decode an accepted confidence-only annotation change. This removes the unchanged-class counterexample from the default writer; it remains a falsifier of the raw-soft ablation. Event qualification must precede downsampling. False qualified events, wrong-to-wrong errors and calibration remain unresolved empirical risks. The row-level C freeze is an additional inductive choice, not a requirement derived from transactional correctness; compare it with retaining ordinary recurrence while suppressing only the semantic payload.

A hard-lock preservation gate is unsafe to the hypothesis: one false FIX can make the wrong class permanent. The current soft-penalty design avoids that specific lock but still needs contradictory-obligation, expiry and auditor-shift diagnostics. Finite lifetime limits damage duration, not the probability or severity of wrong obligations.

### Does B collapse to maps plus GRU?

This is unresolved and is the leading falsifier. If \(C=[C^R,C^P]\), the two updates can be represented as one block-structured recurrent map. Naming channel halves “repair” and “preserve” creates no identifiable semantics; learned transformations can rotate, mix or ignore them. Distinct P/R payloads constrain the interface, but the decoder can use either branch arbitrarily.

The decisive control is **exact same B1 maps + event metadata + one matched GRU + same post-audit input timing + same losses/gates/decoder budget**. A lacking P/R is an incomplete comparator. Also require maps with no learned persistent C, unstructured split branches, and the exact writer with only current evidence. Equalizing only parameter count misses the additional training probes and information budget.

Destroy event/class bindings while preserving location, confidence mass and history length; separately swap semantic roles and use supported donor/reset controls. If only the maps matter, remove the latent-architecture claim. If only the paired loss matters, claim an objective/protocol effect. If a larger generic core matches B, accept the simpler explanation.

## 5. C: revised consistency is sound, but the identity is weaker than it looks

The first inspected C combined a saved pre-ledger with an audit of current-to-restituted logits. That mixed two different transition histories. This was sent to the coordinator before report completion and **corrected in candidate_C.md**: current P/R is now updated from the actual current-to-restituted audit; the factual log keeps the ordinary event followed by a restitution event with contributions \(\beta\delta^{cf},\beta r^{cf}\). That correction resolves the identified P/R/event-log inconsistency at the specification level.

Consequently, \(\beta=1\) can recover the counterfactual additive accumulator and logit residual of the immediate action while the **complete** state differs from a never-factual history. Actual P/R, latest audit, accepted-event age/identity and turn count record an ordinary event plus restitution. A second historical audit would be needed to construct a separately defined alternative ledger; it is excluded. Do not call the present construction full counterfactual-state equality.

There is an even simpler counterexample to the need for an additive architecture. For any deterministic full post-action state map \(G(S_{\rm before},p)\), with current state equal to its factual result,

\[
C_{\rm factual}+
  [G(S_{\rm before},p^{cf})-G(S_{\rm before},p)]
=G(S_{\rm before},p^{cf}).
\]

This holds even when G is a nonlinear GRU. It needs a reproducible snapshot and no intervening state mutation, not additive latent evolution. The same argument applies to a deterministic full-logit map. It does not certify all metadata or make interpolation exact. **Immediate full-state snapshot replacement in A/B is therefore a mandatory C control**, using the same support solver, candidate logits, output constraints and audit count. C must justify why its event representation improves attribution, feasibility, efficiency or subsequent repair beyond this control.

The revised event log also supports a simple invariant: summing the factual ordinary delta and appended restitution delta must reproduce retained C; summing corresponding residuals must reproduce retained logits relative to their declared base. Clearing replay eligibility must not erase the contribution from that accounting. These are future implementation obligations, not tests run in this review.

Remaining requirements:

- Factual replay must reproduce the recorded values under the declared weights, dtype, batch composition, normalization and stochastic state. Equality across arbitrary AMP/batch changes cannot be assumed; define tolerated error and fallback.
- Current-state audit must evaluate the **realized** transferred logits after clamping, interpolation and output constraints. Small coordinate/latent norms do not guarantee a small affected mask region.
- \(\beta<1\) is an interpolated intervention. After nonlinear later actions, an old fixed delta is generally not a counterfactual without descendant replay; the present immediate-action restriction is appropriate.
- **Resolved in the final C reread:** section 6 now explicitly makes the two replay checks feasibility-only against recorded FIX classes and support/output bounds. They do not call the official Auditor; only the selected feasible candidate or ordinary fallback receives one final current-transition audit. This makes section 10's one-Auditor budget coherent. Additional audited alternatives would be a different protocol and must be charged separately.
- At a three-turn cap, a sequence ordinary → restitution → ordinary is the shortest natural way to observe a restitution latent benefit. Restitution on the last turn cannot demonstrate future usefulness. Report the eligible fraction and keep terminal cases in overall evaluation.
- Compare logit-only C1 and joint transfer at identical retained logits, P/R, latest audit and subsequent policy. Otherwise output improvement or extra search is mislabeled as latent memory benefit.

## 6. Future utility: useful estimand, insufficient training safeguard

Equation (2) is a legitimate paired intervention **within the specified computational system**. It is not a proof of clinical causality or of a new architecture. Its validity requires complete logits/probabilities fixed, not merely equal argmax masks: hidden confidence and entropy differences are visible inputs to both proposer and Auditor. The final state_formulation reread now explicitly fixes complete logits/remaining horizon, excludes terminal prefixes from the usefulness cohort, and adds null-input supervision and absolute-arm monitoring. Those are specification improvements; the following requirements describe how a later experiment must honor them.

Specify the intervention contract explicitly:

1. Fix V, complete retained L, newest audit, active row, turn index, remaining cap, weights and common random numbers. If recurrent C alone is tested, hold P/R and event metadata fixed; changing both answers a different question.
2. Use a fixed eligible checkpoint cohort and the deployment horizon. Include rejected, zero-innovation, failed-replay and fallback outcomes in end-to-end reporting. Conditional analyses of accepted writes must report their selection denominator.
3. Separately evaluate trained-null controls and supported donor-state interventions. A reset state paired with an advanced annotation can be outside training support; failure there is insufficient evidence that the semantic history was useful.
4. Freeze evaluation policy and scoring. The future GT outcome J must not be replaced by the same learned utility head being optimized.

Several gaming paths remain even with stop-gradient targets:

- **Weak initial mask:** weakening A0 creates artificial correction headroom. Keep A0 supervision, checkpoint selection and initial-quality distributions matched.
- **Damaged null comparator:** a shared trained reader can learn to fail on null state. Stopping gradients through teacher labels does not stop this distributional or shared-parameter effect. Give the null/control model its own adequate segmentation/future training and evaluate frozen matched policies; report both arms' absolute performance.
- **Fewer accepted turns:** a conservative writer/proposer can reduce regressions by causing early rejection. That is a policy effect, not proof of better repair memory. Report candidate quality, accepted fraction, repair coverage, final quality and destruction at matched operating points.
- **Utility-head success without useful state:** classification/ranking can be accurate while the persistent representation remains redundant. Direct future supervision and independent held-out paired outcomes are both necessary.
- **Gate exploitation:** detach blocks a direct gradient path; it does not eliminate adaptive overfitting to the Auditor through repeated proposals or shared training. Count all candidate checks, freeze calibration and evaluate against GT final outcomes plus gate-error diagnostics.
- **Self-defined preservation:** predicted P must not define the evaluation truth. Use separately maintained GT-scored accepted-FIX history; distinguish first destruction, repeated destruction and repair of an earlier regression.
- **No-headroom/terminal writes:** positive-margin demands where K=0 or no future repair is available reward arbitrary state distinctions. Such cases should have zero/undefined usefulness under a declared rule, not invented positive targets.

A useful result must survive patient-level out-of-sample evaluation, seed variation and the strongest simpler controls at comparable compute. No numerical gain, calibration evidence or real-data validation has been produced by this report.

## 7. Requested fixes and adjudication record

| Finding communicated through Orca | Current adjudication |
|---|---|
| C pre-ledger/current-transition mismatch; incomplete restitution-event accounting | **Resolved in the revised proposal, not implementation-verified.** Re-read current-ledger actual-history formulation and explicit appended restitution contributions. |
| Broad C decrement/replay novelty contradicted by arXiv:2607.27539 | **Incorporated in revised C closest-method table.** Full primary method independently inspected; preprint status retained. |
| B exact-map/single-GRU comparator, last-touch metadata and qualified no-event identity | **Added to revised B at specification level.** The counterexample now concerns the raw-soft ablation; comparative results and implementation checks remain pending. |
| Immediate nonlinear snapshot replacement versus C's additive event architecture | **Required narrowing/control.** Algebraic identity does not need a new neural primitive. |
| Same-logit utility, null-arm sabotage and terminal horizon | **Clarified in revised state_formulation.** Common frozen evaluation, null supervision and both absolute outcomes are required; no empirical closure. |
| C actual audit count versus its compute formula | **Resolved in final C reread.** At most two feasibility-only checks and one final official audit; extra audited alternatives are excluded. |

Early findings were sent as Orca messages msg_d10f7052abda, msg_d6466222b5b6 and msg_13059a60ce96; the final audit-count clarification request was msg_6948ed7f7acb. The coordinator's C clarification was msg_0ea3d4499ffe. Ownership of the proposal files remained with their authors/coordinator.

## 8. Search method, exact queries and access limits

This is a bounded adversarial second-round search, not a PRISMA systematic review. Searches ran live on 2026-09-11 using the web search tool, followed by primary-host full-text or indexed-method inspection. Search results from aggregators, blogs, forums and unrelated programming memory faults were discovery noise, not technical evidence. arXiv, Springer, CVF and OpenReview were the principal primary hosts; PubMed independently exposed the TMI paper's DOI/status. No production or training experiments were authorized or performed.

The tool batches four queries and returns a combined result stream without stable total-hit counts or reliable per-query counts. Therefore counts are not invented. Each group below records its returned relevant leads; an empty relevant return means only that this search did not identify a mechanism, never that none exists.

Direct retrievals of the supplied leads occurred before broad searches: SENTRY HTML v1, Li/Li publisher article, MAIS HTML v1 and StreamDAM HTML v1. Later targeted reads checked Skip RNN Section 3; QG-ConvLSTM Section 3.1; LFB-Net Section II.C–D; SAM2Long Sections 3.2–3.3; *Subtract or Replay?* Sections 3 and 7/Appendix G; and the FocalClick/DREAM primary-host indexed method passages. Equation/section anchors are in the evidence table.

Access/retrieval log:

- DREAM direct web PDF open returned an OpenReview browser challenge; a separate in-memory HTTP request returned 403. Its full writer was not verified. No credentials or access controls were bypassed.
- CVF SAM2Long/FocalClick PDF direct opens returned internal errors. SAM2Long arXiv v2 full method worked; CVF indexed primary text verified its proceedings identity. FocalClick primary indexed method worked; arXiv PDF retrieval separately failed with a content-size limit (11,684,663 bytes).
- An attempted SAM2Long arXiv identifier 2503.14053v1 was wrong: it returned *ON-Traffic*. It was immediately excluded; verified SAM2Long identifier is 2410.16268. The mistaken retrieval provides no evidence.
- RNN/context-free-grammar hidden-state rollback surfaced in an author-hosted paper; label-noise, class-incremental and counterfactual image-generation results also surfaced. They were not treated as exact B/C matches.
- Searches also returned the anonymous OpenReview dual-state correction lead dy6tnQMeyI and a positive/negative prototype-memory industrial segmentation article (ScienceDirect PII S0952197626018890). These remain screening leads, not established full-mechanism counterexamples in the verdict.
- No external-paper performance claim was independently reproduced. Recent manuscript dates/venues were not inferred from search-engine crawl dates.

### Exact query log

All query strings below were submitted verbatim, including quotation marks. No date or domain filter was used except the literal site restrictions inside particular query strings.


| ID | Exact submitted query |
|---|---|
| Q01 | "transactional neural state" |
| Q02 | "accepted state update" segmentation |
| Q03 | "rollback hidden state" neural network |
| Q04 | "error memory" segmentation |
| Q05 | "correction state network" segmentation |
| Q06 | feedback conditioned recurrent segmentation |
| Q07 | mask conditioned recurrent feature encoder |
| Q08 | "verified state update" neural memory |
| Q09 | "critic gated" memory update segmentation |
| Q10 | "quality gated" recurrent memory segmentation |
| Q11 | "segmentation correction memory" |
| Q12 | "dual repair" "preserve" memory segmentation |
| Q13 | segmentation memory class specific previous prediction correction preserve accepted action provenance restitution |
| Q14 | segmentation "counterfactual" "memory" correction |
| Q15 | neural network "rollback" "hidden state" |
| Q16 | segmentation "repair" "preservation" "memory" |
| Q17 | SAM2Long Enhancing SAM 2 Long Video Segmentation constrained tree search arxiv |
| Q18 | DREAM dual memory continual LiDAR segmentation perceptual error memory |
| Q19 | segmentation "class-specific" "memory" "correction" |
| Q20 | segmentation "previous class" "regression" memory |
| Q21 | site:openreview.net/pdf/a033cc4979b05be1f7f285f72a259e6378a133b1.pdf "3.3" |
| Q22 | site:openreview.net/pdf/a033cc4979b05be1f7f285f72a259e6378a133b1.pdf "error" "update" |
| Q23 | FocalClick progressive merge difference mask largest connected component |
| Q24 | segmentation correction "class" "provenance" memory |
| Q25 | segmentation "corrected pixels" "memory" |
| Q26 | segmentation "previously corrected" "class" |
| Q27 | segmentation "false positive" "false negative" "dual memory" |
| Q28 | "accepted" "correction" "memory" segmentation neural |
| Q29 | "DREAM: Dual-Memory Reasoning" "error" "memory" update |
| Q30 | "DREAM" "perceptual memory" "3.4" |
| Q31 | transactional neural state accepted state update segmentation verified state update |
| Q32 | "correction state" segmentation recurrent memory |
| Q33 | site:openaccess.thecvf.com/content/CVPR2022/papers/Chen_FocalClick_Towards_Practical_Interactive_Image_Segmentation_CVPR_2022_paper.pdf "Progressive Merge" "untouched" |
| Q34 | site:openreview.net "DREAM: Dual-Memory Reasoning" "uncertain" |
| Q35 | site:openreview.net "DREAM" "Error Memory Representation" "entropy" |
| Q36 | "class-bound" segmentation memory correction |

Returned leads by submitted batch:

- Q01–Q04: DREAM and author-hosted RNN/context-free-grammar rollback surfaced; many programming-memory false positives; no exact accepted-state or transactional segmentation mechanism identified.
- Q05–Q08: LFB-Net (TMI), Recurrent Mask Refinement, Deep Recurrence and adjacent feedback-model leads; no exact verified-state writer identified.
- Q09–Q12: QG-ConvLSTM and CVPR 2020 quality-weighted recurrent enhancement; cross-domain critic/memory papers; no exact dual repair/preserve writer identified.
- Q13–Q16: Subtract or Replay?; recurrent-state rollback; counterfactual image-generation and unrelated memory research. The first was followed to its primary arXiv full method.
- Q17–Q20: SAM2Long verified identifier/proceedings, DREAM, Li/Li and adjacent class-specific memory/class-incremental papers; no exact previous-class obligation writer identified.
- Q21–Q24: FocalClick primary paper and correction-memory leads; exact-hash DREAM queries did not resolve full-PDF access.
- Q25–Q28: Li/Li, positive/negative prototype memory, interactive propagation and label-correction leads; many sensor-memory false positives.
- Q29–Q32: DREAM Section 3.4 Eq. (8) primary-host indexed passage; anonymous dual-state correction OpenReview lead; recurrent entity networks and transactional-software noise.
- Q33–Q36: FocalClick Progressive Merge primary method and DREAM Section 3.4 retrieved; class-bound query mainly returned unrelated class-boundary/programming results.


## 9. Final inspected proposal snapshots

The final reread on 2026-09-11 included the coordinator's C history and audit-count corrections, B qualified-event/last-touch revision, exact B1-plus-GRU comparator, and state utility safeguards. SHA-256 digests identify these working-tree inputs independently of source HEAD; later document revisions require a new consistency check.

| Input | SHA-256 at final inspected revision |
|---|---|
| candidate_A.md | f109d32b5be2b5fc2304941c9da5927a925c1911c72b0d5133be7bef15b64ff3 |
| candidate_B.md | 1ba2e10f6867a2c2f29c8659bb4c38eee29b15bfe0494415510a462ca975ce1f |
| candidate_C.md | 943fe3bfae419e8b3c3a0781ec17ae1ba421b86a1d2f69b9348feb9ce1d9d29f |
| state_formulation.md | c255f41006abd9817c726017ed4d323f30a414db35970e76ef830e858d216337 |
