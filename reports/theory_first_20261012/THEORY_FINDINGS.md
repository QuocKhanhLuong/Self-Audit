# Theory-first findings: Self-Audit cardiac segmentation

Date: 2026-10-12. Read with [complete proofs](THEORY_PROPOSITIONS.md), [problem formulation](PROBLEM_FORMULATION.md), [primary literature map](THEORY_PRIOR_ART.md), and [human decision brief](HUMAN_DECISION_BRIEF.md).

## Executive finding

The earliest scientific bottleneck is the connection between allowed image evidence and **correct named anatomy at useful coverage**. The theoretical analysis sharpens this into three separate obligations: semantic information must distinguish the names; the admissible transition family must be able to reach the proposed support; and newly accepted predictions need an error budget. Better reconstruction, more feature learning, or a converged graph solve does not establish all three.

The strongest new source-specific result is a support obstruction. For the default stable-guidance graph, in exact arithmetic, a pixel restarted to the uniform distribution has class probability at most **.69728125**, below the configured **.70** acceptance threshold—even with temporal edges and unlimited iterations. The proof uses unavoidable short return walks caused by symmetric edges and self-loops. This is stronger than the distance-only bound. It concerns **hard bank labels**, not all soft student targets or generalization.

A second consequential result is that the default extra bank gates add no exact rejection: probability at least .70 already implies margin at least .40 and the defined reliability at least **.1929661**, comfortably above .20 and .05. The reliability score is not independent semantic corroboration.

The project remains scientifically viable as a constrained research problem; neither this graph obstruction nor the non-heart counterexample proves universal impossibility. We derive positive sufficient conditions allowing imperfect anchors, a robust identification model, and a reachable readiness state. Their anatomical assumptions are not yet verified. The strongest present contribution is **problem characterization and falsifiable requirements**, not a new architecture or validated self-auditing model.

## 1. Repository identity and scope

Live `git ls-remote origin HEAD refs/heads/main` resolved both references to:

- **Remote HEAD/main:** `d5a45f31481f0a702a1f9d5d418889fe704e9105`.
- **Commit date:** 2026-10-11 16:48:33 +0700; “Merge PR #34: official unlabeled ACDC test root.”
- **Verification session:** 2026-10-12; initial audit timestamp 04:14:38 UTC.
- **Local worktree:** `/Users/alvinluong/orca/workspaces/Self-Audit/astra-pseudolabel-v3-audit`.
- **Local branch/HEAD:** `QuocKhanhLuong/astra-pseudolabel-v3-audit` at `5cb8414278361dd18cba77b3f70cf0d9f3721a42`, 2026-10-12 10:12:53 +0700, “research log.”
- The starting worktree was clean. Its history is distinct from current remote main. Source propositions use the remote snapshot, not an assumed branch equivalence.

The existing isolated source snapshot `/tmp/self-audit-scientific-20261012-d5a45f3` was used for remote-source reading and tiny function calls. Relevant files are byte-compared with Git blobs at the remote SHA in `PROVENANCE.json`. No checkout switch, production code/configuration/split edit, training run, dataset download, commit, or push was performed.

No `.codegraph/` directory exists in the worktree. Semble was used for code/document discovery; explicitly supplied paths were read directly. A Semble result for the historical source-of-truth report belongs to the local audit branch, not the remote snapshot; that distinction was checked.

## 2. Allowed information and actual operators

The mathematical input is raw patient cine \(X\), permitted acquisition metadata \(M\), declared anatomical priors \(\Pi\), optional pretrained representation \(E\), and its supervision-provenance ledger. Target reference masks are available only to independent evaluation after freezing. UNKNOWN is an abstention action separate from BG. Full definitions, patient weighting and feasible risk/coverage constraints appear in [PROBLEM_FORMULATION.md](PROBLEM_FORMULATION.md).

Source facts:

| Route | What changes | Where class meaning enters | What can authorize semantics |
|---|---|---|---|
| v3 bootstrap | Anonymous representation/assignments, proposals, TRAIN history | Handwritten connectedness/enclosure/orientation and image evidence | Repeated eligible physical wall/cavity observations plus local gates |
| CQA extension | Quotient grouping and optional relation gradients before semantics | The unchanged anatomical readout after grouping | Grouping itself has no named-class authority; readiness still applies |
| Stable guidance bank | Fixed pretrained features, fixed anchors and fixed graph during solve | Explicit wall/cavity/RV readout over multiscale regions | Hard accepted bank support in at least two TRAIN patients per class |
| Stable student | Encoder and, when permitted, semantic decoder | Frozen soft bank targets and class permissions | Permissions are fixed before training; feature loss alone cannot alter them |

The actual stable operator is row-normalized, nonnegative \(P=D^{-1}W\), with a unit self-loop, four spatial neighbors and at most two mutual temporal matches. It has no cross-patient or cross-slice edges. It solves \(Q_{k+1}=\alpha R_0+(1-\alpha)PQ_k\), starting at \(R_0\), after one normalization of rounding error. At defaults \(\alpha=.35\), residual stopping guarantees entrywise error at most \(10^{-6}\) for the stored operator.

The anchor formula is \(R_{0j}=(1-s_j)u+s_jV_j\), \(u=(.25,.25,.25,.25)\), \(s_j=1-\exp(-\text{vote mass}_j)\). Support \(h=Ks\) is propagated separately and reliability is \(h(1-H(Q)/\log4)\). These are deterministic scores; no correctness calibration is present.

The student uses detached soft \(Q\) weighted by reliability wherever the target argmax class is permitted. It does **not** restrict semantic loss to the hard accepted pixels. This matters: the hard-support obstruction is not a theorem that semantic gradients cannot reach other locations once a class has been authorized.

## 3. Theory evidence cards

Each card's proof reference points to a complete statement, definitions, derivation and adversarial checks in [THEORY_PROPOSITIONS.md](THEORY_PROPOSITIONS.md). “Derived” means established here under stated assumptions; it does not mean novel in the literature.

### F1 — Only distinctions present in the allowed information can ground names

- **Status/type:** DERIVED_RESULT; COUNTEREXAMPLE; identification.
- **Scope/formal result:** For every finite observation-compatible set of possible fields (or arbitrary such set on a finite domain), universally correct deterministic named acceptance is confined to pixels where all fields agree. For two indistinguishable worlds differing on \(D\), any shared accepted set \(A\) has worst-world selective risk at least \(\mu(A\cap D)/(2\mu(A))\).
- **Complete proof:** P1a–P1b; pointwise disagreement and integration. P1c proves a robust positive identification condition for finite role assignments with signature separation exceeding twice observation error.
- **Counterexample:** One ambiguous site is sufficient for the basic obstruction. Duplicate non-heart ring/cavity/right-neighbor structures show that ambiguity can survive current orientation/topology/appearance predicates and concern the spatial partition, not merely a class permutation.
- **Assumption provenance:** Current predicates are source facts. Admissible-world ambiguity and a separated observation model are explicit abstractions. Actual anatomical prevalence and model completeness are unverified.
- **Closest prior/additional content:** Agreement-region selective classification and general observational-equivalence arguments. The cardiac predicate countermodel and explicit feasible-coverage consequence are the useful specialization.
- **Research implication:** The semantic bridge must be part of the problem statement and tested separately from partition quality.
- **Does not prove:** Unsupervised cardiac segmentation is impossible, or that real patients contain exact duplicate structures.
- **Prediction:** Stable anonymous partitions and repeated image evidence can coexist with wrong names on cases outside the prior model.

### F2 — The default stable bank cannot hard-accept previously neutral pixels

- **Status/type:** DERIVED_RESULT with REPO_FACT; operator/selection.
- **Scope/formal result:** For the exact row-stochastic solve, contrast at distance \(d\) from informative restart is bounded by \((1-\alpha)^d\) times maximum restart contrast. With the source graph's symmetry, unit self-loop, six-neighbor weight bound and \(\alpha=.35\), any uniform-restart node has \(Q_{\max}\leq.69728125<.70\). Moreover \(Q_{\max}\geq.70\) implies margin \(\geq.40\) and reliability \(\geq.1929661\).
- **Complete proof:** P2a–P2e, including residual, return-walk bound, entropy/mixture bound, and corruption perturbations.
- **Counterexample to a stronger claim:** A directed chain with no symmetric return permits .7375 at its immediate neutral neighbor. Thus the no-expansion result needs the source graph assumptions.
- **Assumption provenance:** Defaults, normalization, topology and formulas verified in source. Bounds refer to ideal arithmetic; temporal cosine roundoff is not formally certified. The solver certificate alone does not certify graph construction or anatomy.
- **Closest prior/additional content:** Classical resolvent diffusion. Additional analysis is the precise attainable-support/default-threshold consequence and gate redundancy.
- **Research implication:** More iterations cannot solve this hard-support limitation. A support-expansion claim must specify the initial support definition and operator family.
- **Does not prove:** No graph influence, no improvement inside existing soft-anchor support, no soft student supervision outside hard labels, or no eventual student generalization.
- **Prediction:** At the same frozen bank/default formula, increasing solver iterations will not produce accepted hard labels on truly neutral restart nodes.

### F3 — Frozen semantic permissions are absorbing even while features learn

- **Status/type:** REPO_FACT and DERIVED_RESULT; transition invariant.
- **Scope/formal result:** If no foreground class has accepted stable-bank support in two TRAIN patients, every completed training epoch retains zero semantic updates and produces a representation-only checkpoint; export returns UNKNOWN for all inputs.
- **Complete proof:** P3a induction over the fixed permissions and counter. P3b gives a positive finite activation result for mutable v3 history under robust eligibility, retention and recurrent visits, including a conditional sampling bound \(mr e^{-qB}\).
- **Counterexample to a stronger claim:** A changing encoder can change logits through a frozen semantic head. Mutable v3 components/history can open a gate; zero semantic CE alone is not a global absorbing-state theorem.
- **Assumption provenance:** Fixed bank/permissions and exporter branches are code facts. Existence of a stable eligible representation neighborhood is a restricted, unverified assumption.
- **Closest prior/additional content:** Elementary state invariance and recurrence, not a new optimization theorem. The distinction between stable guidance and v3 prevents a false universal stagnation claim.
- **Research implication:** Any claimed escape needs a support-affecting path available before semantic authorization and a reachable persistent eligible state.
- **Does not prove:** The encoder cannot learn, a gate activation is anatomically correct, or added epochs will create such a state.
- **Prediction:** Feature loss may decrease while the frozen stable route exports only UNKNOWN; v3 can instead exhibit changes in eligibility without guaranteed naming accuracy.

### F4 — Merge-only editing has a nonzero representational floor

- **Status/type:** DERIVED_RESULT and COUNTEREXAMPLE; attainable partition family.
- **Scope/formal result:** With immutable void and indivisible atoms, rejection-cost error is at least \(\lambda\mu(V)+\sum_a\min\{\lambda w_a,w_a-\max_cw_{ac}\}\). A cavity-to-boundary path outside every admissible wall prevents enclosure of that cavity pixel.
- **Complete proof:** P4a–P4c. P4c derives image merge-risk improvement despite a true anatomical boundary, and a restricted positive implication under class-appearance separation and bounded risk-estimation error.
- **Counterexample:** A two-class mixed atom cannot be repaired by merging. Two different tissues with identical scalar means can benefit from pooling in image risk while losing their anatomical boundary.
- **Assumption provenance:** Quotient-only operations, immutable void, and closed-hole readout requirements are source facts. Latent labels appear only in the analysis/evaluation characterization, never in training. Appearance separation is unverified.
- **Closest prior/additional content:** Classical coarsening and statistical region merging. The useful contribution is the error floor and topology-specific applicability limit, not “merging” itself.
- **Research implication:** Determine whether the needed correction is representable before evaluating the auditor or investing in optimization.
- **Does not prove:** CQA cannot help fragmentation, or every small gap makes every possible wall unrecoverable.
- **Prediction:** Grouping can help an already supported ring split by atom IDs; mixed atoms and forbidden missing seams remain unresolved by the quotient step.

### F5 — Relational targets can be feasible without being semantically useful

- **Status/type:** DERIVED_RESULT with REPO_FACT; algebraic/optimization distinction.
- **Scope/formal result:** Joint per-term minima of coassignment BCE require no negative relation inside a positive component and \(K\)-colorability of the quotient negative graph. The actual single-call CQA graph avoids the first conflict and is planar; six colors suffice, so default \(K=12\) has enough free assignment capacity. Uniform softmax assignments are stationary for fixed relation targets.
- **Complete proof:** P5a–P5c, including a self-contained six-color argument and softmax-Jacobian calculation.
- **Counterexample:** A positive-positive-negative triangle is inconsistent in general; a negative triangle exceeds \(K=2\). These do not establish that current CQA emits such a conflicting graph. Uniform fixed-target predictions have zero gradient at positive loss.
- **Assumption provenance:** Atom connectivity, adjacency and anti-chaining checks are code facts. Free per-atom variables are an abstraction; shared neural parameter feasibility is not established.
- **Closest prior/additional content:** Hsu et al. (ICLR 2019) already use the same dot-product BCE. Planarity/capacity and the fixed-target stationary case clarify this implementation.
- **Research implication:** Nominal prototype capacity is not the leading single-image bottleneck. Relation validity and optimization under confidence-gated target availability are more important.
- **Does not prove:** SGD converges, relation satisfaction gives names, or the entire teacher must collapse. Uniform assignments in the actual adapter lose their atoms, disabling the relation channel.
- **Prediction:** Low relation loss may reflect absent targets or successful anonymous consistency without anatomical correctness.

### F6 — Safe added coverage requires a semantic corruption budget

- **Status/type:** DERIVED_RESULT and COUNTEREXAMPLE; conditional semantic/selective risk.
- **Scope/formal result:** If informative anchor influence has bad mass at most \(b\), good anchor true-class probability at least \(1-\eta\), and support \(h\), then \(Q_Y\geq(1-h)/4+h(1-b)(1-\eta)\). A Lipschitz readout plus bounded feature/readout discrepancy preserves teacher argmax when its margin is large enough. Added support changes rejection-cost loss by \(b_{\rm new}(r_{\rm new}-\lambda)\) when old predictions remain fixed.
- **Complete proof:** P6a–P6c; deterministic mixtures, margin perturbation, Markov/union bounds and disjoint error accounting.
- **Counterexample:** Perfect feature copying with a swapped or biased readout can produce completely wrong named predictions. Repeating a wrong witness does not reduce its error probability.
- **Assumption provenance:** Linearity and scores are source matched. True corruption, teacher error, semantic readout regularity and population feature error are not label-free estimates supplied by this pipeline.
- **Closest prior/additional content:** Existing noisy-label, self-training and selective-risk theories already give conditional positive guarantees. Here the explicit dependency chain exposes what the no-GT route has yet to justify.
- **Research implication:** An audit must relate to added-region semantic error; more coverage or lower entropy alone is insufficient.
- **Does not prove:** An operational label-free safety certificate, student improvement over teacher, or generalization from training distortion.
- **Prediction:** Feature agreement and confidence can improve while anatomical error remains unchanged or increases, especially on newly accepted regions.

## 4. Claim, evidence and assumption boundaries

| Claim | Current support | Boundary |
|---|---|---|
| Numerical fixed-point and support bounds | Complete deterministic proofs; source-matched defaults; tiny checks | Exact model/stored-operator numerical scope |
| Stable representation-only absorption | Direct control-flow induction | Fixed bank/permissions and current exporter only |
| CQA attainability limits and per-call relation feasibility | Complete finite-set/graph arguments and actual operators | No neural convergence or anatomy guarantee |
| Useful named selective supervision is possible | Conditional positive results | Anatomical model, corruption and coverage feasibility not validated |
| CQA feedback improves named segmentation | Hypothesis only | Prior objective overlap and matched controls unresolved |
| Stable feature guidance improves named anatomy | Hypothesis only | Teacher bias/readout coupling/population generalization unresolved |
| Current route is fully annotation-free upstream | Not established | Official MAE identity is pinned; complete upstream lineage pending |
| Existing CI proves anatomical recovery | Unsupported | The read receipts concern manufactured inputs and software execution |

Read experiment evidence included the CQA verification receipt (44 focused CPU mechanism tests), stable-guidance receipt (47 focused CPU contract tests), and stable-full CI/local preflight receipts. These are **repository-reported evidence**, not tests rerun in this theory session. The stable-full receipt reports genuine MAE inference on manufactured images and explicitly excludes real-data accuracy. Historical local research reports were used for provenance/claim boundaries only. No negative or positive experiment result was used to prove the theory.

## 5. Skeptical review and actual mathematical checks

Every major proposition includes a separate adversarial paragraph or counterexample in the proof document. Degenerate cases considered include uniform anchors, a wrong pure anchor, disconnected graphs, strong wrong edges, \(\alpha=1\), excluded \(\alpha=0\), missing orientation/classes, all UNKNOWN, mixed atoms, missing wall support, conflicting relations, inadequate prototype capacity, duplicate/non-heart rings and perfectly correlated witnesses.

Executed this session: exact rational default-bound calculation; all 1,099 undirected unit-edge graphs with one to five nodes (5,405 neutral-node cases); source solver calls on a directed chain and uniform/alpha=1 cases, a 27-node actual-constructor graph with six neighbors at its center, and a seven-node clique; one source coassignment autograd calculation with zero uniform-logit gradient; two-fold appearance-risk counterexample; finite relation-coloring checks. See [MATH_CHECKS.json](MATH_CHECKS.json). There was no optimization, neural training, data loading, parameter search or GPU experiment.

The independent Orca review attempt failed before task execution at `agent_readiness: timeout`. Run `run_5c067d337187`, dispatch `ctx_02202bed71cf`; the created unused terminal was released. Local PyTorch 2.14.0 optimizer source was also inspected to verify that None-gradient parameters are excluded from AdamW updates; this does not constitute a target-environment training test. No independent reviewer result exists. The completed skeptical pass is an internal second pass, not independent validation.

## 6. Exact source/document read register

All following remote-source links are pinned to the audited SHA. Ranges describe inspected scope, not newly modified code.

| File / inspected scope | What was checked |
|---|---|
| [docs/pseudolabel_v3_bootstrap_experimental.md](https://github.com/QuocKhanhLuong/Self-Audit/blob/d5a45f31481f0a702a1f9d5d418889fe704e9105/docs/pseudolabel_v3_bootstrap_experimental.md), full document | Anonymous warmup, repeated readiness, declared evidence limits |
| [docs/pseudolabel_v3_cqa.md](https://github.com/QuocKhanhLuong/Self-Audit/blob/d5a45f31481f0a702a1f9d5d418889fe704e9105/docs/pseudolabel_v3_cqa.md), full document | Quotient/audit/relation proposal and its qualifications |
| [docs/stable_guidance.md](https://github.com/QuocKhanhLuong/Self-Audit/blob/d5a45f31481f0a702a1f9d5d418889fe704e9105/docs/stable_guidance.md), full document | Frozen representation/bank/student and lineage boundary |
| [docs/stable_guidance_full.md](https://github.com/QuocKhanhLuong/Self-Audit/blob/d5a45f31481f0a702a1f9d5d418889fe704e9105/docs/stable_guidance_full.md), full document | Native export, full workflow, freeze-before-evaluation, CI scope |
| [core.py](https://github.com/QuocKhanhLuong/Self-Audit/blob/d5a45f31481f0a702a1f9d5d418889fe704e9105/src/self_audit_guidance/core.py), lines 1–290 | Defaults, graph degree/normalization, solver, anchor mixture, selection |
| [student.py](https://github.com/QuocKhanhLuong/Self-Audit/blob/d5a45f31481f0a702a1f9d5d418889fe704e9105/src/self_audit_guidance/student.py), lines 1–152 | Feature/semantic losses, permissions, optimizer guard, exporter |
| [pipeline.py](https://github.com/QuocKhanhLuong/Self-Audit/blob/d5a45f31481f0a702a1f9d5d418889fe704e9105/src/self_audit_guidance/pipeline.py), lines 1–120 | TRAIN bank construction and frozen held-out student export |
| [teacher.py](https://github.com/QuocKhanhLuong/Self-Audit/blob/d5a45f31481f0a702a1f9d5d418889fe704e9105/src/self_audit_guidance/teacher.py), lines 1–116 | MAE hash/freeze, declared upstream lineage, image tiling/projection |
| [io.py](https://github.com/QuocKhanhLuong/Self-Audit/blob/d5a45f31481f0a702a1f9d5d418889fe704e9105/src/self_audit_guidance/io.py), lines 70–140 | Native image/affine/orientation, normalization and frozen bank fields |
| [full.py](https://github.com/QuocKhanhLuong/Self-Audit/blob/d5a45f31481f0a702a1f9d5d418889fe704e9105/src/self_audit_guidance/full.py), lines 172–233 | Both prediction cohorts locked before reference evaluation |
| [cqa_v3.py](https://github.com/QuocKhanhLuong/Self-Audit/blob/d5a45f31481f0a702a1f9d5d418889fe704e9105/src/self_audit_pseudolabel/cqa_v3.py), lines 1–403 | Witnesses, risk operator, merge invariants, relation loss, adapter |
| [trainer_v3.py](https://github.com/QuocKhanhLuong/Self-Audit/blob/d5a45f31481f0a702a1f9d5d418889fe704e9105/src/self_audit_pseudolabel/trainer_v3.py), lines 88–225 | Current local evidence, bank agreement, losses, training/inference transitions |
| [bootstrap_v3.py](https://github.com/QuocKhanhLuong/Self-Audit/blob/d5a45f31481f0a702a1f9d5d418889fe704e9105/src/self_audit_pseudolabel/bootstrap_v3.py), candidate assessment and lines 274–420 | Confidence/topology/local evidence and history/activation rules |
| [losses_v3.py](https://github.com/QuocKhanhLuong/Self-Audit/blob/d5a45f31481f0a702a1f9d5d418889fe704e9105/src/self_audit_pseudolabel/losses_v3.py), lines 1–114 | Anonymous gradients versus gated raw-logit semantic CE |
| [evidence.py](https://github.com/QuocKhanhLuong/Self-Audit/blob/d5a45f31481f0a702a1f9d5d418889fe704e9105/src/self_audit_pseudolabel/evidence.py), lines 55–158 | Whole-hole/enclosure and orientation naming |
| [stable configuration](https://github.com/QuocKhanhLuong/Self-Audit/blob/d5a45f31481f0a702a1f9d5d418889fe704e9105/configs/stable_guidance_experimental.json), full | Actual defaults used in P2 |
| [CQA configuration](https://github.com/QuocKhanhLuong/Self-Audit/blob/d5a45f31481f0a702a1f9d5d418889fe704e9105/configs/pseudolabel_v3_cqa_experimental.json), full | Twelve prototypes, gates/history and relation settings |
| `reports/cqa_20261008/VERIFICATION.json`; `reports/stable_guidance_20261011/VERIFICATION.json` | Reported software/mechanism evidence and exclusions |
| `reports/stable_full_20261011/CI_VERIFICATION.json`; `PREFLIGHT_LOCAL.json` | Reported manufactured-input MAE smoke and local dependency blocker |

On the **local audit branch**, `reports/astra_v3/12_SOURCE_OF_TRUTH.md` was read for historical evidence authority and exposed-evaluation boundaries; it is absent from the remote snapshot. Relevant excerpts of the newly committed historical `reports/main_baseline_20261002/03_RESEARCH.md` were consulted, not used as new empirical evidence. No presumed `reports/stable_full_20261011/VERIFICATION.json` exists: the actual filename is `CI_VERIFICATION.json`.

Primary papers and exact theorem/proof scopes read are recorded in [THEORY_PRIOR_ART.md](THEORY_PRIOR_ART.md). Remaining unknowns are real anatomical validity/coverage, valid corruption bounds, generalization, useful feedback beyond matched baselines, full upstream supervision lineage, and independent mathematical review.
