# Evidence ledger and search boundary

Cutoff: **2026-09-12**. Repository evidence refers to **`9726e6d519c33035489ca7a5e4681a046493d89e`**, verified against fetched `origin/main` at the beginning of this task. This is a targeted scientific audit broadened across conceptual families, not a claim to have exhausted every venue, language, unpublished submission or September 2026 preprint.

## Evidence labels

* **CODE:** inspected current implementation; says what the code computes, not whether a trained model behaves usefully.
* **METHOD:** primary method/equations or algorithm read live. A paper's empirical claims remain its authors' claims unless independently reproduced.
* **ABSTRACT:** metadata/abstract verified; insufficient to rule out mechanism overlap.
* **DERIVATION:** our mathematical deduction with explicit premises; not a literature discovery or performance result.
* **UNMEASURED:** requires a checkpoint, data or controlled experiment not run here.

## Root-reviewed primary literature beyond a conventional attention survey

The competing-method detail and additional primary links are in [03](03_prior_art_challenges.md). This table records the wider conceptual tests independently used in the synthesis.

| Source / status | Primary material inspected | What it challenges | Boundary of the comparison |
|---|---|---|---|
| G-BRS, CVPR 2022 — METHOD | [Paper, §2](https://openaccess.thecvf.com/content/CVPR2022/papers/Lin_Generalizing_Interactive_Backpropagating_Refinement_for_Dense_Prediction_Networks_CVPR_2022_paper.pdf): auxiliary refinement variables, localized changes, consistency | Frozen dense predictor + feedback-driven latent optimization is established | Human click labels, different variables and temporal reference; not evidence of exact Candidate C equivalence |
| Counterfactual Attention Learning, ICCV 2021 — METHOD | [Paper, §3, Eqs. 4–6](https://openaccess.thecvf.com/content/ICCV2021/papers/Rao_Counterfactual_Attention_Learning_for_Fine-Grained_Visual_Categorization_and_Re-Identification_ICCV_2021_paper.pdf): factual versus intervened attention effect | Attention intervention and factual subtraction are not new in isolation | Training objective over attention maps; not recorded accepted-transition support restitution |
| Spatially Transformed Adversarial Examples, ICLR 2018 — METHOD | [Paper, §3.3](https://arxiv.org/pdf/1801.02612): bilinear sampling via optimized flow and deformation cost | Optimizing spatial coordinates under target/proximity objectives | Input deformation for adversarial classification differs from internal support repair and FIX protection |
| Algorithmic Recourse: from Counterfactual Explanations to Interventions, FAccT 2021 — METHOD | [Paper, §§3–4](https://arxiv.org/pdf/2002.06278): actions as structural interventions, feasibility, descendant effects | Minimal counterfactual change does not automatically identify an actionable causal explanation | The known network is a computational causal object; the patient's anatomy is not thereby causally identified |
| GEM, NeurIPS 2017 — METHOD | [Paper, Eqs. 6–8 and Algorithm 1](https://papers.neurips.cc/paper/2017/file/f87522788a2be2d171666752f97ddebb-Paper.pdf): limit loss increases on past memories through gradient constraints | “Correct something while preserving previous achievements” and projected conflict constraints | Parameter updates across tasks, not sampling-coordinate interventions across segmentation turns |
| PCGrad, NeurIPS 2020 — METHOD | [Paper, §2.3, Algorithm 1](https://papers.neurips.cc/paper_files/paper/2020/file/3fe78a8acf5fda99de95303940a2420c-Paper.pdf): remove conflicting gradient components | A future FIX/REGRESS gradient decomposition is not automatically a novel optimizer | Multi-task optimization; does not supply Candidate C's history/evidence semantics |
| DAgger, AISTATS 2011 — METHOD | [Paper, Algorithm 3.1 and §4](https://proceedings.mlr.press/v15/ross11a/ross11a.pdf): collect learner-visited states, aggregate expert labels | Training on one state distribution and deploying on another is a longstanding sequential-learning problem | Self-Audit has predicted feedback, not expert action labels; DAgger guarantees do not transfer |
| Safe Policy Improvement with Baseline Bootstrapping, ICML 2019 — METHOD | [Paper, §2 and Theorem 2](https://proceedings.mlr.press/v97/laroche19a/laroche19a.pdf): baseline restrictions under statistical uncertainty | Fallback is a design pattern, not by itself a safe-improvement guarantee | Finite-MDP/data assumptions are not established for Self-Audit |
| Learn then Test, 2021/2022 preprint — METHOD | [Paper, Definition 1, §2.1, Theorem 1](https://arxiv.org/pdf/2110.01052): valid tests and family-wise error control over a fixed parameter family | Held-out risk selection can be formulated for a whole frozen prediction pipeline | Requires valid calibration sampling and tests; not provided by `tau=0` or an Auditor AUROC |
| Conformal Risk Control, 2022 preprint / ICLR 2024 — ABSTRACT + primary implementation | [Paper metadata](https://arxiv.org/abs/2208.02814), [author implementation](https://github.com/aangelopoulos/conformal-risk) | Expected-risk calibration for monotone loss families | Do not impose monotonicity requirements on all later risk-control approaches |
| Conformal Risk Control for Non-Monotonic Losses, 2026 preprint — METHOD | [Paper, Theorem 1, §2.2.3](https://arxiv.org/html/2602.20151v1): stability-based bounds, nonmonotone selective losses | Nonmonotonicity alone does not make calibration impossible | Exchangeability/stability or suitable finite-grid assumptions still need justification; not an OOD guarantee. Metadata says February 2026, manuscript body has an August date; cited as 2026 without inferring a venue |
| ActionSplice, September 2026 preprint — METHOD | [Paper, §§3.1–3.3, Eqs. 1–2](https://arxiv.org/html/2609.08230v1): matched rollback target at a fixed solver step, learned residual state transport, prefix mask | Broad novelty claims about revising a previous computation while preserving committed content face close adjacent work | At deployment it avoids replaying completed sampler evaluations; C explicitly replays a previous ordinary correction and uses predicted transition evidence |
| LeCor, September 2026 preprint — METHOD | [Paper, §§3.2–3.3, Eqs. 1–3 and Algorithm 1](https://arxiv.org/html/2609.09477v1): one gradient on case adapters; outer objective on unclicked slices | Training a model so corrections generalize outside the supervised region is active prior art | Human clicks, case-specific weight adaptation and meta-learning; C keeps parameters frozen and optimizes coordinates. No universal improvement theorem is inferred |
| From Few-Shot Segmentation to Clinician-in-the-Loop Medical Image Analysis, September 2026 — METHOD/FORMULATION, **Perspective** | [Article, §5, Eqs. 1–5](https://arxiv.org/html/2609.10001v1): bounded updates, protected reference retention, value of feedback, accept/query/defer | Broad decision-theoretic and “governed correction” framing is already articulated | A normative perspective, not evidence that such a system or Candidate C equivalent has been empirically established; its clinician supplies external information |
| FACTOR, May 2026 preprint — ABSTRACT | [Metadata and abstract](https://arxiv.org/abs/2605.03294) | Test-time counterfactual comparison without weight changes is not an empty research area | Full-method collision not independently closed by root; do not use this entry to prove either equivalence or uniqueness |

Recent material remains preprint evidence unless an official venue is verified. We do not inherit a paper's “first” claim. Source inspection differs across workers and mirrors; where an official PDF was unavailable to a worker, the report must identify the alternate rendering or access gap.

## Search strategy: conceptual equivalents rather than only names

Queries covered combinations of `error conditioned deformable attention`, `feedback conditioned sampling`, `critic guided correction`, `quality guided segmentation refinement`, `counterfactual attention`, `residual guided sampling`, `uncertainty deformable attention`, `backpropagating refinement`, `interactive correction consistency preservation`, `algorithmic recourse intervention`, `spatial transformed adversarial examples`, `action editing rollback replay`, `meta learned correction medical segmentation 2026`, `safe policy improvement`, `nonmonotonic conformal risk`, and `learn then test`.

For close hits the search followed method equations and references into other families. The point was to test reductions: replace coordinates by other latent variables; replace historic replay by current-state computation; replace optimization by masked rollback; replace hard FIX constraints by consistency or merge. These are scientific counter-hypotheses, not claims that every substitution yields an identical algorithm.

The older Dynamic Window literature folder supplies historical leads but does not override current code or the user's selection of Candidate C. This pass does not recertify every entry of that older matrix or claim exhaustive CVPR/ICCV/ECCV/MICCAI/NeurIPS/ICLR/AAAI/TMI/MedIA/CMIG/TPAMI coverage. Remaining search gaps should reduce novelty confidence, not be hidden behind a source count.

## Code evidence used in the synthesis

| Claim | Current source |
|---|---|
| Realized coordinate override bypasses geometry generation; same attention sampling path | [dynamic_window.py](../../../src/self_audit/models/dynamic_window.py) |
| Depth depends on turn; ordinary candidate has an inner class-shared gate; replay uses detached functional state | [annotation_expert.py](../../../src/self_audit/models/annotation_expert.py) |
| C1 check; restoration objective; one gradient and two checks; C3; latest ordinary record; no-op versus ordinary fallback | [self_audit_net.py](../../../src/self_audit/models/self_audit_net.py) |
| FIX/REGRESS are changes in correctness, not merely changes in label | [targets.py](../../../src/self_audit/audit/targets.py) |
| Optional predicted-history exposure; primary zero-history bootstrap unchanged; auxiliary auditor-only phase | [unified_trainer.py](../../../src/self_audit/training/unified_trainer.py) |
| Bootstrap evidence explicitly untrained; retained-state auxiliary loss; detached predicted-history collection | [finetune_joint.py](../../../src/self_audit/training/finetune_joint.py) |

## What this task did and did not establish

**Established by inspection/derivation:** λ has zero initial displacement-gradient contribution; predicted-FIX preservation is different from true-FIX preservation; same-endpoint convex mixing condition; oracle true-REGRESS rollback ceiling; sparse temporal opportunity; solver rejection is not an infeasibility certificate.

**Executed:** five standalone arithmetic checks in [the reproducible probe](probes/restitution_counterexamples.py), with [saved JSON](probes/restitution_counterexamples.json).

**Not measured:** actual replay pass rate on trained AMP inference, real-evidence exposure frequency, gradient sensitivity, eligible patient prevalence, correction gain, true harm, latency, external generalization or clinical utility. No new production behavior, training, checkpoint evaluation, full test suite or GPU run occurred in this research task.
