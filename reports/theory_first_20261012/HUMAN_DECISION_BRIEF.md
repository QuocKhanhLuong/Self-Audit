# Human decision brief

Date: 2026-10-12. This brief recommends a research problem, not an architecture.

## Decision supported by the analysis

Keep the central question focused on **whether image-only evidence can support trustworthy named RV/MYO/LV supervision at useful patient-level coverage**. Treat self-audit/refinement as a hypothesis about the validity and incremental value of evidence. Do not yet position a larger model, feature distillation, graph convergence, or the coassignment loss itself as the contribution.

The current evidence supports a rigorous problem-characterization package. It does not establish publication readiness, real cardiac accuracy, universal identifiability, or anatomical safety.

## What materially changed

1. **A specific support-expansion claim is analytically ruled out.** In the ideal default stable-bank graph, a completely neutral restart node has maximum class probability no greater than .69728125, below the .70 threshold. Unavoidable return walks strengthen the simple distance argument. More solver iterations cannot turn this into hard accepted support. This says nothing against soft targets or a student generalizing beyond the hard bank.
2. **The extra default bank gates are mathematically redundant.** Probability .70 already entails margin at least .40 and the implemented reliability at least .1929661. Those latter scores do not constitute additional independent corroboration at their current thresholds.
3. **Zero semantic support has different meanings in different routes.** Stable guidance freezes the bank and permissions; feature training can proceed indefinitely while its representation-only exporter remains all UNKNOWN. Mutable v3/CQA evidence can change eligibility, so a global no-learning theorem would be false.
4. **CQA has both a capability and a boundary.** It can join existing fragments. It cannot split a mixed atom or add missing support. Its per-call relation graph is consistent with enough free prototype capacity; twelve prototypes are not an inherent bottleneck. That feasibility does not guarantee useful optimization or correct anatomy.
5. **The loss novelty claim needs narrowing.** Hsu et al. (ICLR 2019) already derive categorical inner-product BCE. Any contribution must come from a justified image-derived relation source, its effect on semantic support, and gains beyond assembly/classical merging.
6. **A positive theory route remains.** Explicit semantic signatures, bounded influence-weighted anchor corruption, a reachable persistent gate state, and a stable readout can suffice. These are conditional results; the current code does not label-free certify their anatomical premises.

## Two central research questions

**RQ1:** Which admissible acquisition/anatomical assumptions make a useful region of fixed-name RV/MYO/LV supervision identifiable despite noncardiac lookalikes and imperfect orientation?

**RQ2:** Under those assumptions, when does image-based audit evidence justify increasing accepted named support while controlling the error added by that increase?

The corresponding hypotheses and explicit falsifiers are in [PROBLEM_FORMULATION.md](PROBLEM_FORMULATION.md). The leading question is RQ1; RQ2 depends on a meaningful semantic bridge and an attainable update family.

## Choices that remain with the human

| Choice | Why it is consequential |
|---|---|
| Allowed anatomical/acquisition priors and upstream supervision | Determines which worlds are distinguishable and what “no-GT” actually means |
| Intended patient/acquisition domain | A restriction excluding lookalikes or certain truncated/basal cases must be scientifically justified and explicit |
| Acceptable semantic risk, per-class coverage, and rejection cost | Defines feasibility and success; no clinical tolerances were invented in this session |
| Inductive deployment versus predeclared per-patient adaptation | Changes the permitted inference information without allowing hidden cohort tuning |
| Named anatomy as primary target versus anonymous partition repair | The latter is a meaningful narrower alternative but changes the scientific claim; it has not been adopted here |

The work does not require a choice of architecture now. Any future candidate must show where names come from, what corrections it can represent, how it can obtain support before semantic permission, and why its accepted additions have tolerable error.

## Evidence completed now

- Resolved current GitHub HEAD/main to `d5a45f31481f0a702a1f9d5d418889fe704e9105`, dated 2026-10-11 16:48:33 +0700, and distinguished the local research-history branch.
- Read the four requested current research documents, relevant operators, readiness/training/export information flow, actual configurations and verification receipts.
- Read primary theory/proof material for propagation, selective classification, disentanglement, self-training, noisy labels and risk control; inspected exact unsupervised/pairwise segmentation objectives and protocols.
- Wrote six theory evidence cards with complete inspectable propositions, proofs, counterexamples, positive conditions, source applicability and nonclaims.
- Executed only tiny mathematical/source sanity checks: exact constants, miniature graphs, a stationary-gradient calculation, a two-fold risk example and relation-feasibility examples.
- Performed a separate internal skeptical pass. The attempted independent Orca worker failed before task execution (`agent_readiness: timeout`), and its terminal was released; no independent review is claimed.

## Evidence not supplied by this session

No real-data segmentation result, new checkpoint, GPU benchmark, training run, dataset download, hyperparameter sweep, complete upstream pretraining-lineage audit, or independent proof review. Existing verification receipts were read as reported software evidence, not reclassified as anatomical evidence.

No production source, existing research history, split or configuration was changed. Only this new report bundle and its mathematical checks were written. No commit or push was made.

At most three later decisive evaluations are specified, with falsification criteria, in the problem formulation. They remain NOT RUN. Their purpose is to test semantic grounding, actual support attainability and the incremental value of audit feedback—not to tune an architecture until the hypothesis looks favorable.
