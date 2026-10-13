# Self-Audit: from propagation limits to a defensible support-expansion problem

Technical continuation, 2026-10-13. Read this report with [the complete propositions and proofs](PROPOSITIONS.md) and [the human decision brief](HUMAN_DECISION_BRIEF.md). Previous research is preserved; this is not a replacement survey.

## 1. Executive finding

The hard-bank acceptance obstruction is real within its stated graph assumptions, but it is neither a theorem against soft supervision nor a task-level impossibility. Classical harmonic propagation already supplies a counterexample to the broad negative story: correct accepted support can expand to arbitrary graph distance despite imperfect anchors, corrupted labels, and cross-class edges. Therefore “make diffusion cross the threshold” is primarily a local repair or baseline comparison, not the leading paper contribution.

The substantive problem is now more precise: **under an explicit image/metadata/prior information contract, determine when additional named anatomical predictions are justified by bounded erroneous evidence influence, and preserve that justification through soft-target learning and native-grid export.**

Three additions change the earlier assessment:

1. A constructive positive result closes the gap between a conditional corruption bound and actually attainable expansion. It also shows why repairing propagation can increase the reach of mistakes.
2. Exact evidence-state collisions show what the current normalized bank forgets: no evidence and balanced conflict can have the same R, Q, entropy, reliability, and UNKNOWN output. Repetition can cross authorization thresholds without adding independent information.
3. A weighted soft-target risk bound and a native-grid nullspace example identify what is missing between bank analysis and student output. Teacher bias, low-weight exported mass, margins, and grid transfer remain explicit terms; decreasing feature or semantic loss does not remove them.

These are a stronger problem characterization and source-specific consequences of established mathematics. They are not a validated new method, a label-free anatomical certificate, or a publication-readiness claim.

## 2. Snapshot, allowed information, and actual operators

The live remote HEAD and refs/heads/main both resolved to **d5a45f31481f0a702a1f9d5d418889fe704e9105**, committed **2026-10-11 16:48:33 +0700**, “Merge PR #34: official unlabeled ACDC test root.” It has not changed since the previous report.

The working checkout is a different branch, **QuocKhanhLuong/astra-pseudolabel-v3-audit**, at **5cb8414278361dd18cba77b3f70cf0d9f3721a42**, committed 2026-10-12 10:12:53 +0700. All source conclusions below concern the pinned remote snapshot, not an assumed local-main equivalence.

The immutable source copy is /tmp/self-audit-scientific-20261012-d5a45f3. Critical files were checked against Git blobs at the remote SHA. No .codegraph directory exists. Directly specified source paths were read; no new index was created.

Permitted inputs are full patient cine images, image-header geometry/orientation, patient/split identities, declared anatomical priors and a frozen pretrained representation with recorded lineage. Manual target masks, target-derived crops, reference-based class mapping, readiness calibration, threshold selection, and checkpoint selection are excluded. References may enter only independent evaluation after predictions and the recipe are frozen. The upstream representation is declared SSL; its full preprocessing/supervision ancestry remains unaudited. No stronger upstream annotation-free claim is made.

The source supplies named semantics through explicit wall/cavity, adjacency, intensity, border and patient-right predicates. Anonymous features do not introduce these names on their own. A non-heart ring remains compatible with those local predicates. This continuation does not repeat the earlier ring proof or reinterpret a negative run as a general theorem.

### Complete stable-guidance information path

| Stage | Verified operation | Scientific implication |
|---|---|---|
| Image-derived anchors | Multiscale component votes are summed; R=(1−s)u+sV, s=1−exp(−mass). Wall hypotheses require an actual enclosed hole before soft enclosure scoring. | Names enter through anatomy/orientation priors. Vote mass is not an independent sample count. |
| Soft bank | Row-normalized P=D⁻¹W; Q_next=αR+(1−α)PQ; separately h_next=αs+(1−α)Ph. Q and h start from their restart targets. | Fixed numerical diffusion; no student feedback. |
| Reliability | ρ=h[1−H(Q)/log4]. | Pooled support times concentration; not calibrated correctness. |
| Hard bank | Probability ≥.70, margin ≥.20, reliability ≥.05; otherwise 255. | These labels produce per-patient class counts. |
| Class permissions | A class is allowed if at least two TRAIN patients have any accepted pixel of that class. Permissions are calculated once. | A class-level authorization gate; not proof of anatomical correctness or pixel-level trust. |
| Training weights | w=ρ·allowed[argmax Q]. An active batch requires some positive-weight foreground argmax and a global allowed foreground class. | A hard-rejected pixel can still contribute soft CE. BG rows can contribute within an active batch. |
| Student losses | Feature cosine loss; weighted four-channel soft CE after logits are interpolated to the bank grid. Targets are detached. | Disallowed class channels can still receive CE gradients from allowed-argmax rows. Hard labels are not the pixel training mask. |
| Student state | If no semantic updates occur, status is REPRESENTATION_ONLY. On inactive steps decoder/head gradients are set to None; encoder feature learning continues. | Fixed-bank no-foreground permissions form an invariant, not a claim that all network parameters stop learning. |
| Export | REPRESENTATION_ONLY returns UNKNOWN before forward. Otherwise native logits pass probability/margin and fixed class-permission gates. | No bank-reliability gate is applied at export; native predictions are not the same object as bank-grid Q. |

Source anchors: [core.py, graph/solver](https://github.com/QuocKhanhLuong/Self-Audit/blob/d5a45f31481f0a702a1f9d5d418889fe704e9105/src/self_audit_guidance/core.py#L115), [core.py, votes and bank](https://github.com/QuocKhanhLuong/Self-Audit/blob/d5a45f31481f0a702a1f9d5d418889fe704e9105/src/self_audit_guidance/core.py#L193), [student.py, losses](https://github.com/QuocKhanhLuong/Self-Audit/blob/d5a45f31481f0a702a1f9d5d418889fe704e9105/src/self_audit_guidance/student.py#L38), [permissions/training](https://github.com/QuocKhanhLuong/Self-Audit/blob/d5a45f31481f0a702a1f9d5d418889fe704e9105/src/self_audit_guidance/student.py#L66), [export](https://github.com/QuocKhanhLuong/Self-Audit/blob/d5a45f31481f0a702a1f9d5d418889fe704e9105/src/self_audit_guidance/student.py#L139).

The bank persists image, features, Q, R, reliability and hard labels; not per-node raw vote histories, strength, or propagated h. Image/features permit potential reanalysis, so the lost-summary result is not a claim that all original observations are destroyed. See [bank schema](https://github.com/QuocKhanhLuong/Self-Audit/blob/d5a45f31481f0a702a1f9d5d418889fe704e9105/src/self_audit_guidance/io.py#L113) and [preparation](https://github.com/QuocKhanhLuong/Self-Audit/blob/d5a45f31481f0a702a1f9d5d418889fe704e9105/src/self_audit_guidance/pipeline.py#L25).

### Verdicts on the provisional findings

| Candidate | Verdict | Exact object and correction |
|---|---|---|
| A: distance-dependent cap | **Valid**, with explicit direction and endpoint conventions | Fixed-point Q for fixed nonnegative row-stochastic P; directed distance i→source. Uniform means neutral restart, which may also encode conflict. |
| B: α=.35, d≥2 cannot reach .70 | **Valid** for hard bank acceptance | d=2 cap=.566875. Solver tolerance 10⁻⁶ does not bridge it under the stated residual model. It says nothing directly about student outputs. |
| C: spatial self-loop obstruction | **Valid but restricted** | P_ii≥1/5 gives .698275862. Default temporal=true allows six neighbors; self-loop-only reasoning then gives .710629921 and is insufficient. A separate symmetric two-step-return proof yields .69728125 under real-arithmetic weights≤1. |
| D: no-foreground absorption | **Valid for fixed-bank stable guidance** | No allowed foreground ⇒ no semantic updates ⇒ representation-only checkpoint ⇒ UNKNOWN export. Encoder learning and changing latent predictions are possible. It is not a universal statement about mutable v3/CQA. |

The solver uses float64 and renormalizes stored P/R once, stops when residual≤α·10⁻⁶, and raises rather than freezing an unconverged bank. Hard acceptance casts Q to float32. Temporal match weights are not clipped above one, so the idealized graph-family result is not advertised as a complete floating-point proof. The proofs provide a computable return-mass bound for any particular stored P.

The previous file had three missing plus signs in displayed formulas; explicit errata and corrected derivations appear in [PROPOSITIONS.md](PROPOSITIONS.md). Previous documents remain unchanged.

## 3. New findings and their boundaries

### N1: possible expansion is classical; justified expansion is conditional

For a finite graph with a path from every unanchored node to a boundary anchor, clamped harmonic propagation has first-hit kernel H=(I−P_UU)⁻¹P_US. If h is incoming anchor strength, b is the fraction of that strength arriving from bad or cross-class sources, and good sources put at least 1−η on the true class, then
\[
Q_{i,Y_i}\ge L_i=(1-h_i)/4+h_i(1-b_i)(1-\eta_i).
\]
L>.5 and explicit probability/margin/entropy-reliability inequalities imply correct acceptance. The complete proof and graph construction are in N1 of the proof document.

The construction includes a corrupted LV anchor, a cross-class RV edge, imperfect good evidence, and self-loops. It gives h=.98, b=.20, η=.02, correct probability .77332 and accepted expansion at arbitrary finite distance. Tiny exact calculations checked distances 2, 4 and 8; the general statement is proved analytically.

The positive result establishes mathematical possibility. The latent bounds b and η are not available from a low residual or high confidence. A classical boundary correction can remove neutral-restart dilution while extending the influence of a wrong anchor. Its graph-perturbation bound also scales with expected absorption time; weak escape can make this factor arbitrarily large. This weakens any proposed contribution consisting solely of a new propagation rule.

### N2: preserve evidence provenance before claiming semantic confidence

No votes and balanced opposing votes can both save uniform R/Q and zero reliability. Conversely, repeated unanimous votes increase s=1−exp(−mass), so three identical votes can cross the current numerical gates even though they derive from one observation. Strong correct and strong systematically wrong evidence are indistinguishable if the permitted observation model admits both latent worlds.

A useful uncertainty contract must distinguish absent support, opposing support, source duplication/dependence, and remaining semantic ambiguity. Retaining those distinctions makes audit decisions better specified; it does not certify truth. Cross-fitting can remove direct fitting on a held-out patient under patient-level sampling assumptions. Detachment, a frozen teacher, adjacent cine frames, or copied predictions do not establish independent semantic witnesses.

### N3: transfer needs teacher error, weighting, margins and native-grid control

Let D=∫w KL(Q‖p)dν on a common grid. For a student export set A of mass c>0, the proof gives
\[
R_S(A)\le\frac{
B_T(A)+\nu(A\cap\{\operatorname{gap}(Q)\le2\delta\})
+\nu(A\cap\{w<w_0\})+D/(2w_0\delta^2)}{c},
\]
clipped at one. B_T is the teacher's error mass, not a measured label-free quantity. This positive bound permits imperfect targets; it also exposes why export at almost unweighted locations is not covered by low training loss.

An additional source-level break occurs at resolution transfer: training interpolates native logits down to the bank grid; export uses native logits. A nonzero perturbation in the interpolation nullspace leaves the semantic loss unchanged while changing native predictions. A 2×2→1×1 example keeps coarse loss fixed and changes two native pixels to confidently wrong RV. This is a function-space counterexample, not a claim that the actual CNN will realize it.

Feature approximation can supply a probability-stability bound only with a common regular named readout and bounded head mismatch. Current feature loss does not establish that bridge. Fidelity may preserve teacher mistakes as well as correct margins.

## 4. The strongest skeptical alternative and primary literature

**“Is this just a bad operator and an incompatible threshold?”** For the hard neutral-node bank barrier: substantially yes. That specific obstruction can be removed by an established boundary formulation. The remaining questions concern uncertain semantic grounding and composing an evidence guarantee through learning and selection. A method paper must outperform or explain more than that local repair.

This is a focused counterargument search. The prior broad survey is not repeated.

| Existing result / exact setting | What it already addresses | What remains here |
|---|---|---|
| [Zhu et al., ICML 2003](https://mlg.eng.cam.ac.uk/pub/pdf/ZhuGhaLaf03a.pdf), Sections 2–3: fixed labeled boundaries on a weighted graph; harmonic solution and first-hit interpretation. | Expansion without uniform restart at every unlabeled node. Matrix conventions were checked: row-normalized P and clamped values. | Boundary labels are supplied. It does not justify image-derived RV/MYO/LV identities. Our construction is a consequence, not a replacement theory. |
| [Scott, Blanchard & Handy, COLT 2013](https://proceedings.mlr.press/v30/Scott13.pdf), Proposition 3 and Appendix C.2: two contaminated class distributions, total contamination<1 and mutual irreducibility. | Unknown asymmetric noise rates need not make recovery impossible. Their proof identifies a unique maximal-denoising solution via mixture proportions. | Instance-dependent, spatially structured anatomical errors need not obey that two-mixture model. Mutual irreducibility is an identifying assumption, not something established by vote entropy. This directly weakens any claim that labels or known noise rates are always necessary. |
| [Ju et al., ICML 2026](https://proceedings.mlr.press/v306/ju26c.html), Theorem 3.1 and Appendix A: normalized graph/attribute influence and a lower bound on true-class influence. | Contradiction-based auditing and label correction on noisy graphs already exist. | The lower bound uses latent true labels and is unknown in practice. A normalization restriction also matters: average true-class influence is ≤1/|C_true|, so their disjoint-interval δ>.5 regime requires singleton contributing class groups. The displayed mixture weights differ between Eq. 5 and the theorem; α=.5, used in their experiments, removes that discrepancy. This is not transferred as a no-GT semantic certificate. |
| [Gosch et al., ICML 2026](https://proceedings.mlr.press/v306/gosch26a.html), Theorem 3.1 and Appendix C.1: deterministic partition/aggregation, bounded numbers of affected partitions under label/edge poisoning. | Prediction robustness can be certified for non-IID structured data; self-training and sparse graph supervision are already considered together. Their proof controls vote changes through a perturbation budget. | The certified object is invariance relative to an unperturbed prediction. It does not establish that the unperturbed anatomical name is correct or estimate our latent corruption budget. Correlation alone does not invalidate such deterministic guarantees. |
| [El-Yaniv & Wiener, JMLR 2010](https://www.jmlr.org/papers/v11/el-yaniv10a.html), retained from prior proof review: labeled realizable selective classification. [Conformal Risk Control](https://arxiv.org/abs/2208.02814), ICLR 2024: exchangeable calibration losses under stated monotonicity/boundedness conditions. | Established risk/coverage and risk-calibration frameworks. | Semantic calibration losses need references; target-mask calibration would violate this project's training/selection contract. Neither permits replacing anatomical loss by entropy. |
| [Wei et al., ICLR 2021](https://arxiv.org/abs/2010.03622), retained from prior proof review: class expansion/separation and sufficiently accurate pseudo-labels. | Self-training can improve noisy initial supervision under nontrivial assumptions. | The needed class geometry and initial error assumptions are not implied by image consistency or current gates. Our wrong fixed points refute unconditional improvement, not their conditional results. |

Publication metadata were verified on primary proceedings pages: Scott 2013 is COLT, and the two 2026 papers are ICML proceedings, not merely preprints. The new theorem/proof reading covered Zhu's harmonic derivation and random-walk section, Scott Proposition 3/Appendix C.2, Ju Theorem 3.1/Appendix A, and Gosch Theorem 3.1/Appendix C.1. Zhu's equation page was rendered because PDF text extraction corrupted mathematical symbols.

Grady's 2006 random-walker segmentation paper was located, but the full author PDF could not be read in this continuation: web open returned “Internal Error,” and the direct fetch failed certificate verification. No new theorem transfer relies on it. Wu et al. ICML 2024 topological sample selection was screened and its PDF retrieved; its theorem was not relied upon or claimed fully reviewed.

### Gap classification after the counterargument

| Candidate gap | Classification | Strength / remaining requirement |
|---|---|---|
| Neutral restart cannot meet hard-bank threshold | Local operator/contract mismatch | Fully supported analytical diagnosis; weak novelty; classical comparator already resolves attainability. |
| Confidence cannot distinguish absence, conflict and duplicated evidence | Limitation of the present summary and related pooling families | Exact counterexamples; general uncertainty decomposition is established territory. Need task-specific benefit, not a renamed score. |
| Image-derived named sources have bounded bad influence on new locations | Task-level problem under a declared observation model | Highest relevance; currently conditional rather than operational. Must state a plausible semantic bridge and retain lookalike counterexamples. |
| A bank guarantee survives soft training and native export | Missing composed understanding for this pipeline and similar dense-prediction systems | Explicit weighting, margin and resolution counterexamples; potentially useful beyond this repo, but not yet a novel general theorem. |

## 5. One complete architecture-independent problem

**Plain-language statement.** Given cine images and explicitly admissible prior information, produce fixed-name BG/RV/MYO/LV predictions while expanding the portion of anatomy that can be named with controlled error. The system must be allowed to reject genuinely unresolved locations, but rejection cannot satisfy the task by itself.

### Inputs, latent object, and information contract

For each patient p, observe \(O_p=(X_p,M_p)\), where X_p:Ω_p×T_p→R is a cine intensity field and M_p contains permitted geometry, timing/index and patient identity metadata. Y_p:Ω_p×T_p→{BG,RV,MYO,LV} is latent. RV/LV denote ventricular cavity regions and MYO ventricular myocardium according to the fixed evaluation ontology; BG denotes the complement, not “unrecognized.”

An admissible prior ledger Π states orientation accuracy, acquisition field-of-view assumptions, anatomical relations, failure cases and provenance. E is any allowed pretrained representation with documented upstream supervision. Naming information can enter through Π, allowed upstream supervision, or an explicit image-to-role rule; its origin must be declared. No mechanism name is part of the problem definition.

Train-time representation, proposals, audit, pseudo-labels and student fitting can access only the designated training O, Π and E. A frozen recipe may process a held-out patient's observed cine at inference if declared in advance. Fitting or selecting the recipe using the held-out cohort is a different transductive protocol and must be reported as such. The current student export is frozen inductive inference. Evaluation alone may access Y after predictions and selection choices are frozen.

Outputs are \(a_p(x)\in\{0,1\}\) and \(\hat y_p(x)\) in the named ontology where a=1; otherwise output UNKNOWN. Anonymous partitions are equivalent up to label permutations. Named outputs are not: a permutation exchanging RV/LV changes the task loss. An anonymous partition result can be useful, but is not successful named recovery.

### Objective, coverage, and dependence

Let ν first select a patient with equal weight and then a location under a predeclared normalized within-patient measure μ_p. Physical-volume/time weighting or a specified frame weighting must be fixed; frames and pixels are not independent patients.
\[
C=\mathbb E_p\int a\,d\mu_p,\quad
N=\mathbb E_p\int a\,1\{\hat y\ne Y\}\,d\mu_p,\quad
R=N/C\quad(C>0).
\]
At C=0, selective risk is undefined, not reported as zero. Report patient distributions and missing classes in addition to pooled ratios.

Use the full-domain rejection-cost loss
\[
L_\lambda=\mathbb E_p\int
[a\,1\{\hat y\ne Y\}+\lambda(1-a)]d\mu_p,\qquad0<\lambda<1,
\]
and coverage/risk constraints for named foreground classes. For c∈{RV,MYO,LV}, define true-class coverage \(C_c=\nu(a=1,Y=c)/\nu(Y=c)\), and prediction-class selective error \(R_c=\nu(a=1,\hat y=c,Y\ne c)/\nu(a=1,\hat y=c)\), with absent denominators explicitly marked. Require nonzero per-class coverage and bounded error at a declared patient-coverage level; BG dominance cannot satisfy this contract.

Positive coverage is feasible only if the admissible model class contains sufficiently large identifiable regions for each required class, the source/error assumptions hold there, and the output family can represent them. If indistinguishable admissible worlds disagree everywhere relevant, no method can fulfill a low-risk positive-coverage contract. Narrowing Π must be justified by acquisition/anatomical knowledge, not by silently encoding desired masks.

For an old accepted set A and added set B, report added-region risk separately. With predictions on A unchanged,
\[
L_\lambda(A\cup B)-L_\lambda(A)=\nu(B)[R(B)-\lambda].
\]
If old predictions change, add the old-set change in error mass. Increased coverage is beneficial only under this error accounting.

### Demonstrated obstruction and possible progress

The demonstrated obstructions are neutral-restart attenuation, non-identifying evidence summaries, fixed permission invariants, and incomplete bank-to-export transfer. The general task obstruction is the possibility of observation-equivalent anatomical worlds under insufficient priors.

Progress is mathematically possible when admissible evidence bounds the influence of semantic corruption/cross-class leakage, preserves enough reliable mass to clear gates, and the learning/export stage satisfies a nonvacuous fidelity/coverage condition on the actual output domain. N1 establishes one such conditional existence result. N3 supplies a conditional imperfect-target transfer bound. Neither verifies the assumptions on real cardiac MRI.

### Two central research questions and falsifiable hypotheses

**RQ1 — Grounding and uncertainty:** On a declared cardiac acquisition/prior model class, can permitted evidence distinguish unsupported, conflicting and systematically wrong anatomical claims well enough to support a nonvacuous bound on named-source error or bad influence?

**H1, empirical claim to test:** A frozen image-derived source/audit rule attains a predeclared error bound ε for each predicted foreground class, at nonzero true-class coverage κ, including a challenge stratum with lookalikes and uncertain orientation. This is an empirical hypothesis, not already proved by N1. A suggested decision contract is ε=.10 and κ=.05; these are proposed research utility targets, not measured results or clinically validated limits. The human must lock or replace them before opening references. Failure of this particular rule does not refute all no-GT rules.

**RQ2 — Expansion and transfer:** Under bounded imperfect evidence, which conditions allow a positive increase in correct named coverage while controlling new-region error through pseudo-labeling, student learning and native export?

**H2, empirical claim to test:** With sources and evaluation recipe frozen, support outside initially informative regions increases true-class coverage by at least a predeclared Δ for each required foreground class while new-region selective error stays ≤ε, and old accepted error mass does not increase. A suggested utility target is Δ=.02 with the same ε=.10. The comparison must include classical harmonic propagation with the same sources and graph, and distinguish bank-grid from student/native outcomes. If only the classical baseline meets the contract, the local obstruction is repaired but a proposed self-audit contribution is unsupported.

These hypotheses describe the claim to be supported, not a selected architecture. Choice of audit mechanism, evidence model and final idea remains human-owned.

## 6. Decisive evidence and human decision

Only two subsequent evaluations are proposed; neither was run.

1. **Frozen grounding/false-evidence challenge.** Lock the image-only source rule, uncertainty ledger, assumptions and all selection thresholds. Include missing evidence, explicit conflict, duplicate witnesses, orientation perturbations and appearance-matched non-heart structures as separate strata. Freeze named predictions before independent reference scoring. H1 is falsified on the declared scope if established error exceeds ε or coverage is below κ; a wide uncertainty interval is inconclusive, not evidence of success. An admissible indistinguishable-world counterexample falsifies any universal certificate immediately. Show results for each foreground class and patient, not an aggregate dominated by BG.
2. **Matched attainability and transfer challenge.** Keep identical initial sources, graph and information access. Compare static sources, current restart, a classical clamped-boundary comparator, and the human-selected audit/refinement mechanism if one is later chosen. Track correct/corrupt source influence, hard-bank support, soft-target weights, old/new-region error, and bank/native resolution. H2 is falsified when the coverage increase is below Δ, new-region error exceeds ε, or old error mass grows. A gain explained entirely by the classical comparator does not support new refinement novelty. Additional training is not authorized by this report.

Reference-based influence diagnostics are evaluation-only latent diagnostics; they cannot become thresholds or training oracles. Quantify uncertainty at the patient level. IID-patient guarantees require an explicit sampling assumption; correlated cine pixels do not supply independent observations. A finite-cohort analysis without that assumption supports only a finite-cohort claim.

**Recommended problem, not method:** trustworthy expansion of named anatomical supervision under provenance-aware imperfect evidence, with an explicit contract from evidence to final predictions. Do not frame “convergent diffusion,” “more confident pseudo-labels,” “frozen teacher,” or “detached audit” as the contribution. Those properties can coexist with wrong names.

The human's open choices are (i) the strongest defensible acquisition/anatomical assumptions and acceptable upstream supervision, (ii) the intended risk/coverage utility contract, (iii) whether the main contribution should be source grounding, evidence-preserving auditing, or a bank-to-export guarantee, and (iv) the final mechanism after classical controls. A broader architecture is not justified by this analysis.

## 7. Evidence status and exact delivery

**Established:** proofs A–D with stated restrictions; N1's constructive imperfect-evidence expansion; N2's summary collisions and duplication counterexamples; N3's support/margin, weighted-KL and resolution results. Source checks establish information flow. Tiny checks verify finite constructions and actual small loss/export operators.

**Unverified assumptions:** real anatomical source accuracy, graph bad-influence bounds, independence of patient sampling, readout regularity/head mismatch, native-grid approximation, full upstream SSL preprocessing lineage and the practical prevalence of duplicated hypotheses.

**Still empirical:** attainable class/patient coverage, native anatomical error, calibration of any proposed audit, incremental error under coverage growth, and benefit beyond classical/no-feedback controls. No patient images or masks were read, and no model was trained.

The primary technical artifacts are:

- CONTINUATION_REPORT.md — this report and complete problem formulation.
- PROPOSITIONS.md — complete proofs, counterexamples and evidence cards.
- HUMAN_DECISION_BRIEF.md — English decision brief.
- math_checks.py and MATH_CHECKS.json — exact/tiny mathematical and source checks.
- PROVENANCE.json, PREVIOUS_REPORT_HASHES.json and SHA256.json — snapshot, review and preservation receipts.
- INDEPENDENT_REVIEW.md, REVIEW_RECEIPT.json and REVIEW_RELEASE.json — the actual Orca proof-review payload and settled-worker receipts.

All are under **/Users/alvinluong/orca/workspaces/Self-Audit/astra-pseudolabel-v3-audit/reports/theory_continuation_20261013/**. They are local and uncommitted. No commit or push was performed; these files are not claimed available on GitHub. The previous reports/theory_first_20261012 directory is preserved byte-for-byte.

The analyzed source is limited to core.py and student.py in full; the bank schema and preparation/export path in io.py/pipeline.py; the relevant mutable-v3 evidence/training transitions; the stable config and relevant stable guidance documentation excerpts. The earlier broad CQA/readout survey and real-data experiment history were not restarted. Exact files, hashes, reading scope, mathematical-check environment and review receipt are recorded in PROVENANCE.json.

The independent Orca reviewer completed task task_9a0ff574e706 / dispatch ctx_12adf3cffd77 and checked the harmonic construction, probability/logit and weighted-KL proofs, and the core/student source flow. The review required positive-margin/tie handling, teacher error over the entire student export set, batch gating, and valid-checkpoint restrictions; those appear explicitly in the retained results. The reviewer allocated the good anchor's residual .02 probability to RV, whereas the main construction splits it equally; both give the same .77332 lower bound and pass the gates. A–C, the added harmonic perturbation bound, evidence-compression examples and native-grid nullspace proof were checked separately by the coordinator and are not claimed independently reviewed. The worker was released; no reclaimable worker remains.

**What changed in our understanding:** the task is not blocked by a universal inability to propagate sparse evidence. We now have a feasible positive construction and explicit reasons why its guarantees do not automatically become no-GT certificates or native student guarantees. This changes the next research decision from “which larger model or stronger diffusion?” to “which semantic error assumptions can be defended, which evidence must survive compression, and how will the final output preserve the resulting contract?”
