# Problem formulation: named anatomical supervision from image-only cine

Date: 2026-10-12. PRE-IDEA research; no architecture selected.

## 1. Allowed information and semantic entry points

For patient \(i\), observe \(X_i:\Omega_i\to\mathbb R\), where \(\Omega_i=\{0,\ldots,T_i-1\}\times\Lambda_i\), and \(\Lambda_i\) is a finite native three-dimensional voxel lattice. Metadata \(M_i\) may include image affine, spatial units/spacing, acquisition orientation when reliable, frame ordering, and patient identity for grouping/split enforcement. No fixed motion correspondence is assumed between frames. The latent anatomical field is \(Y_i:\Omega_i\to\mathcal C\).

The ontology is \(\mathcal C=\{\mathrm{BG}=0,\mathrm{RV}=1,\mathrm{MYO}=2,\mathrm{LV}=3\}\): background includes everything outside the target right ventricular cavity, left ventricular myocardium, and left ventricular cavity. Papillary-muscle, basal-plane, and partial-volume conventions must be fixed in the scientific ontology before evaluation; the checked operators do not derive those annotation conventions from images. UNKNOWN \(=255\) is an action to abstain, never a fifth anatomical tissue or a synonym for BG.

Inputs also include a declared prior set \(\Pi\), an optional pretrained representation \(E\), and a provenance ledger \(\mathcal L\). Anatomical meaning enters through the ontology and the mapping from observable properties to named roles specified by \(\Pi\); it can also enter through upstream supervised knowledge in \(E\), if such knowledge is present. Naming channels 1–3 does not itself ground those channels.

The primary track prohibits target manual masks, scribbles, manual ROI/ED–ES selections, target-mask-trained teachers, and reference-based class matching from representation adaptation, proposals, audit, pseudo-labels, readiness, threshold selection, and checkpoint selection. Independent reference scoring occurs after the recipe, model-selection rule, checkpoint, and predictions are frozen. The ledger must distinguish:

| Information source | Admissible role | What it does not establish |
|---|---|---|
| Raw cine and acquisition metadata | Representation, appearance/motion evidence, physical coordinates | Anatomical labels or independent repeated trials |
| General anatomical/orientation priors | Explicit semantic bridge with stated domain of validity | Unique identification of a heart from a ring predicate |
| Frozen self-supervised pretraining | Image representation with source/weight identity and upstream disclosures | Complete absence of upstream annotations |
| Random initialization/projection | Declared algorithmic randomness | Class meaning |
| TRAIN patient identities | Locked sampling, support diversity, no cross-split adaptation | Anatomical correctness of accepted support |
| Independent reference masks | Final named evaluation only | Permission to tune the next threshold on an already exposed test cohort |

The inspected CineMA adapter pins the MAE hash and records `SSL_DECLARED_PREPROCESSING_NOT_FULLY_AUDITED`. This session verifies that local contract, not the complete ancestry of all pretraining data. A different upstream-supervision allowance changes the ledger and the scope of the scientific claim.

## 2. Information available by stage

Every learning decision is measurable with respect to allowed images/metadata, \(\Pi,E,\mathcal L\), previous allowed states, and declared randomness—not target reference masks.

| Stage | Permitted information | Inspected implementation versus general formulation |
|---|---|---|
| Representation learning | TRAIN images/metadata, admitted pretrained weights, image objectives | v3 updates its encoder; stable guidance freezes CineMA and trains a separate student |
| Proposal construction | Observed image, representation, geometry, anonymous probabilities | Confident components/quotients or multiscale frozen-feature clusters |
| Auditing | Proposal and declared image/metadata witnesses | CQA compares scalar image risks and interfaces; witnesses are dependent |
| Semantic readout/pseudo-labeling | Proposals plus explicit priors; optional justified historical evidence | Enclosure/orientation/intensity rules provide names; stable graph redistributes fixed restarts |
| Student learning | TRAIN images and allowed frozen/detached features, soft distributions, weights and permissions | Stable student has a fixed bank and fixed class permissions |
| Inference | Frozen recipe/weights plus the new patient's observed cine and permitted metadata | Current stable export is student inference; it does not rebuild a held-out teacher bank |
| Evaluation | Frozen outputs and independent references | Stable full runner locks both VAL and TEST exports before reference evaluation; official TEST is unscored when references are unavailable |

Patient-specific optimization or graph construction using one observed test cine is **within-patient transductive inference** if predeclared. Fitting shared thresholds, representations, priors, or a recipe on the held-out cohort is **cohort-level transduction/adaptation** and must be declared separately. Fixed per-image normalization at inference is not automatically cohort fitting. The main scientific target is a frozen recipe evaluated on new patients; training-time patient-specific graphs do not turn the current student exporter into a transductive method.

## 3. Three different recovery problems

An anonymous partition is an equivalence relation on supported locations. Two indexed assignments \(z,z'\) are equivalent when \(z'=\pi\circ z\) for a single permutation of anonymous IDs. Partition recovery is evaluated modulo that permutation and does not assign anatomical names.

Named recovery requires a field in the fixed ontology. A global RV/LV swap is an error even if its anonymous partition is perfect. A spatially wrong partition cannot be repaired merely by choosing a permutation. Hungarian remapping against reference masks changes the evaluated task and is excluded from primary named evaluation.

Selective prediction outputs \((q,a)\), \(q:\Omega\to\Delta_4\), \(a:\Omega\to\{0,1\}\), with \(\widehat Y(v)=\arg\max q(v)\) when \(a=1\) and UNKNOWN otherwise. Neither \(q\) nor an audit score is assumed calibrated to anatomy. A method may instead output a set of compatible names internally; emitting a singleton then requires a declared decision contract.

These definitions do not mention CQA, CineMA, graph propagation, or Dynamic Window. Those mechanisms may be replaced without changing the task.

## 4. Quality objective and rejection contract

Give every patient equal outer weight. Within patient \(i\), use a declared normalized measure \(\mu_i\) over native locations and frames; uniform voxels/frames is one choice. For \(A_i=\{a_i=1\}\), define
\[
c_i=\mu_i(A_i),\quad n_i=\int_{A_i}\mathbf1\{\widehat Y_i\ne Y_i\}\,d\mu_i,\quad
r_i=n_i/c_i\quad(c_i>0).
\]
When \(c_i=0\), selective error is **undefined**, with zero coverage explicitly reported. It is not a successful zero-error segmentation.

Population patient-weighted coverage and error mass are \(C=\mathbb E_i c_i\) and \(N=\mathbb E_i n_i\). \(R=N/C\), when \(C>0\), corresponds to first drawing a patient uniformly and then a location by \(\mu_i\), conditional on acceptance. It differs from the macro-average of \(r_i\) over covered patients; report both, along with zero-coverage patients and lower-tail behavior. Do not replace the patient count by the number of correlated frames or pixels when forming uncertainty intervals.

Use a full-domain rejection-cost loss
\[
L_\lambda=N+\lambda(1-C),\qquad 0<\lambda\leq1,
\]
and parameters \(c_{\min}\in(0,1]\), \(\varepsilon\in[0,1)\), with a contract such as
\[
\min N\quad\text{subject to } C\geq c_{\min}>0,\quad R\leq\varepsilon.
\]
The optimization notation defines the **latent scientific objective**. Neither \(N\) nor \(R\) is available to a no-GT trainer. A method can only optimize permitted surrogates and justify a bridge to this objective. All-UNKNOWN incurs \(L_\lambda=\lambda\) and violates positive coverage; it may still be the honest output when the assumed feasible contract is unsupported.

Overall coverage can be satisfied by easy background. Therefore the cardiac claim must additionally specify foreground/class coverage:
\[
C_c=\mathbb E_i\!\left[
\frac{\mu_i(A_i\cap\{Y_i=c\})}{\mu_i(Y_i=c)}
\ \middle|\ \mu_i(Y_i=c)>0\right],\quad c\in\{\mathrm{RV,MYO,LV}\},
\]
with predeclared requirements \(C_c\geq\gamma_c>0\) for each present foreground class, plus class-specific errors and patient availability. These quantities require references for measurement; predicted class counts are runtime support diagnostics, not substitutes. An accepted wrong class counts as accepted-but-wrong in \(C_c\)'s companion error metric. Class-balanced full-domain error and patient-macro foreground Dice/IoU, with rejected true foreground retained as false negatives, prevent BG prevalence from concealing failure.

Reference availability may be sparse in time. Metrics on annotated ED/ES frames cannot establish dense cine correctness. No manual phase information may be fed back into training under this contract.

**Feasibility assumptions.** P1 shows that zero-risk positive coverage is possible only where admissible worlds agree. A bounded-risk contract needs a semantic observation model, enough class-distinguishing evidence, and controlled violations—e.g. an anchor/witness corruption budget. Coverage on each class also needs that class to be present and sufficiently observable. Demanding fixed positive coverage from every pathological, truncated, motion-corrupted or unorientable cine without such assumptions can be impossible. Report infeasibility rather than treating abstention itself as quality.

Observable objectives—reconstruction, contrastive/relational loss, feature cosine error, temporal consistency, entropy, support count, solver residual—are separately named surrogates. None equals anatomical correctness by definition.

## 5. Demonstrated obstructions and conditions for progress

The strongest combined obstruction is an **information-and-attainability gap**: anonymous image evidence may not determine a name (P1); a transition family may be unable to accept new support even if its numerical solve converges (P2–P3); a quotient may not represent the missing boundary (P4); and satisfying relation targets or matching features need not close the semantic gap (P5–P6).

For the default stable bank, the exact graph formula cannot hard-accept a pixel with completely neutral restart: its maximum is bounded by .69728125, below .70. The probability gate also implies the default margin/reliability gates. This is a particular operator/threshold limitation, not an impossibility theorem for image-only segmentation. Student soft targets are a distinct path and can extend beyond hard accepted bank pixels once class permissions exist.

Progress is possible in a restricted class where observable role signatures distinguish the anatomy robustly, accepted semantic evidence has a controlled influence-weighted corruption rate, the representation can express needed boundaries, and the state transitions can reach and retain support without circular semantic permission. P1c, P3b, P4's restricted separation implication, and P6 supply explicit positive sufficient conditions. Their anatomical premises remain unverified for real cine.

## 6. Leading research problem and two central hypotheses

**Leading problem.** Determine the attainable trade-off between patient-level named anatomical coverage and semantic error from image-only cine, under explicitly declared priors and upstream supervision, when the available proposal/audit evidence is dependent and imperfect. Characterize when accepted support can be created or expanded, and when an image-based audit adds semantic information beyond anonymous consistency.

**RQ1 — Semantic grounding and feasible coverage.** Under which independently defensible acquisition/anatomical assumptions do allowed observations reduce the ambiguity set enough to support useful, named RV/MYO/LV predictions with controlled error and nonzero class coverage?

**H1 — Falsifiable task hypothesis, currently UNVERIFIED.** A predeclared, restricted model class admits a nontrivial region of named identifiability for all three foreground classes, and the stated observable prior signatures distinguish that region from admissible noncardiac lookalikes under specified perturbation/corruption bounds.

Falsification has two levels: (i) a pair of admissible observationally equivalent worlds with different names on the proposed identifiable region refutes the claimed guarantee; (ii) after fixing the region-selection rule and risk/coverage targets, independent frozen patient evaluation failing those targets refutes its applicability to that cohort. The theorem can survive while the anatomical model fails. Risk and minimum per-class coverage values must be scientifically selected in advance by the human; this session does not invent clinical tolerances.

**RQ2 — Valid support expansion from imperfect evidence.** When can a permitted representation/partition update increase accepted named support while bounding the semantic error added by that update?

**H2 — Falsifiable mechanism-class hypothesis, currently UNVERIFIED.** Within H1's declared domain, image-derived audit evidence contains incremental information about the correctness of newly accepted anatomical labels beyond existing proposal confidence/geometry; an admissible update can use that information to improve the fixed full-domain risk/coverage objective.

The current stable hard-bank graph cannot instantiate expansion into neutral locations at its defaults (P2c). CQA grouping cannot instantiate recovery requiring atom splits or added void pixels (P4). These refute particular instantiations of H2, not the whole hypothesis. A meaningful learning contribution must beat assembly-only and classical region evidence under matched information and budget, and must evaluate error on newly accepted locations, not just total coverage.

**One alternative problem.** If the human does not endorse a defensible semantic bridge for H1, study anonymous partition repair and selective correspondence consistency as the primary task, with named anatomy only as an externally evaluated downstream outcome. This is a narrower problem and would change the central scientific claim; it is an option for the human, not an adopted pivot.

## 7. Minimum properties of any future solution

1. A provenance-complete information ledger and an explicit source of class meaning; distinguish no target masks from no upstream knowledge.
2. A support/abstention contract with patient and class coverage, full-domain error, and explicit infeasibility.
3. An attainable update family: identify whether missing pixels, mixed atoms, closed-hole requirements, or fixed permissions make the requested transition unreachable.
4. A support-affecting path that can operate before semantic permission, if escape from zero support is claimed; persistent eligibility and recurrence still need justification.
5. A bridge from witnesses to anatomical error with bounded corruption or stated exceptions. Computational detachment, repeated scales, and temporal repetition are not independent verification.
6. For representation claims, a connection between feature approximation, semantic readout, teacher bias, and acceptance; lower cosine loss alone is insufficient.
7. Frozen model/selection rules before reference access, with patient-level evaluation and clear inductive/transductive boundaries.

Possible mechanism families offer different properties: pure coarsening can join fragments but cannot split/add support; fixed restarting diffusion can redistribute existing distributions but inherits its attenuation/return bounds; representation adaptation can change partitions but needs a semantic bridge; abstention can restrict risk claims but must account for lost coverage. No family or architecture is selected here.

## 8. At most three decisive later evaluations — NOT RUN

These are falsification tests for H1/H2, not an execution plan or an authorization to train.

1. **Grounding and lookalike challenge (H1).** Freeze the declared prior model and exact named readout. Evaluate admissible duplicate-ring/nonheart, orientation-error, absent-class, and cardiac cases. Report anonymous partition quality separately from fixed-name selective error and per-patient/per-class coverage. An admissible indistinguishable pair immediately falsifies universal naming; on real held-out patients, failure of the predeclared joint risk/coverage contract falsifies applicability. No GT remapping or threshold adjustment afterward.
2. **Attainability and incremental-risk accounting (H2).** On fixed proposals and stored evidence, partition candidate additions by zero versus nonzero anchor strength, graph reachability, mixed-atom/void obstruction, and readout feasibility. The mathematical/source portion is feasible without training; this session already checked its miniature cases. Any exact default hard-bank acceptance at a genuinely neutral node falsifies P2c's source match or one of its assumptions. In later frozen reference scoring, a coverage increase whose added-region risk violates P6c's budget falsifies a safe-expansion claim.
3. **Incremental value of audit feedback (H2).** Only after grounding/attainability are established, compare no feedback, assembly-only, the same image witnesses with classical merging, and learned relational feedback using matched proposals, priors, patient split, representation, selection rule and compute budget. The stronger contribution is falsified if feedback has no prespecified clinically/scientifically meaningful improvement in patient-level full-domain objective over the strongest matched control, or if its gain is entirely support inflation with worse added-region risk. These effect-size and risk margins must be chosen before reference exposure. No such comparison was launched.

## 9. Human decision boundary

The human must choose the admissible prior/pretraining contract, target patient/acquisition domain, acceptable risk and per-class coverage, and whether the primary claim remains named anatomy or becomes anonymous partition repair. The evidence currently recommends studying the named selective-supervision problem above before selecting a mechanism. It supports a problem-characterization contribution, not a validated new model, clinical safety certificate, or publication-readiness claim.
