# Primary literature and the remaining gap

Search/read date: 2026-10-12. This is a targeted, adversarial literature review, not an exhaustive novelty search. Publication status is distinguished from the date of the accessible manuscript. External results are not transferred merely because their keywords match the project.

## Material sources: setting, assumption, result, and transfer boundary

### 1. Local/global consistency already supplies the basic propagation theory

[Zhou et al., Learning with Local and Global Consistency](https://papers.neurips.cc/paper_files/paper/2003/hash/87682805257e619d49b8e0dfdc14affa-Abstract.html), NIPS 2003; [full author PDF](https://dennyzhou.github.io/papers/LLGC.pdf), section 2 fixed-point derivation and section 3 regularization read.

Setting: labeled plus unlabeled examples in a fixed affinity graph. The principal operator is \(S=D^{-1/2}WD^{-1/2}\), with zero diagonal affinity; its diffusion coefficient multiplies \(S\). The paper derives a resolvent fixed point and explicitly discusses the row-normalized \(P=D^{-1}W\) variant. Local/global class consistency is the semantic premise, not a conclusion of convergence.

Self-Audit instead has unit self-loops, simplex restarts at every node, and restart coefficient \(\alpha\), so the paper's diffusion coefficient corresponds to \(1-\alpha\). The row-stochastic proof in P2 is self-contained. Resolvent convergence and smoothing are not novelty. The graph-return/default-threshold obstruction is a specific consequence of this implementation's normalization, neutral restarts and acceptance contract.

### 2. Selective prediction already formalizes agreement and coverage

[El-Yaniv and Wiener, On the Foundations of Noise-free Selective Classification](https://www.jmlr.org/papers/v11/el-yaniv10a.html), JMLR 11, 2010, 1605–1641; [PDF](https://www.jmlr.org/papers/volume11/el-yaniv10a/el-yaniv10a.pdf), definitions 4–6 and Theorems 7–8 with proofs read.

Setting: realizable classification with a labeled sample and a hypothesis/version space containing the target. The consistent selective strategy predicts only where all remaining hypotheses agree. Theorem 7 establishes coverage optimality among strategies guaranteeing zero risk with certainty; Theorem 8 gives a finite-class coverage bound using IID sampling.

P1's consensus construction is closely related, not a new selective-learning principle. Here there is no target-labeled version space, and the anatomy-compatible set must come from an explicit observation/prior model. Neither labeled-sample coverage rates nor IID pixel guarantees follow for cine. This prior result substantially weakens a novelty claim based merely on abstention or agreement.

### 3. Disentanglement nonidentifiability is a warning about assumptions, not our theorem

[Locatello et al., Challenging Common Assumptions in the Unsupervised Learning of Disentangled Representations](https://proceedings.mlr.press/v97/locatello19a.html), ICML 2019, PMLR 97; main Theorem 1 and [supplementary Appendix A proof](https://proceedings.mlr.press/v97/locatello19a/locatello19a-supp.pdf) read.

Setting: latent dimension greater than one and a factorized continuous latent density with only observations available. The construction maps marginal CDFs to independent uniforms, then Gaussians, applies a mixing orthogonal transformation, and maps back. The transformed latent distribution is observationally equivalent while its coordinates are entangled.

This does not prove impossibility of named cardiac segmentation with anatomical priors. We construct the finite anatomical-predicate counterexample separately in P1. The transferable lesson is to identify admissible symmetries and the assumptions that break them. A claim of impossibility for the entire image-only cardiac task would be stronger than either this result or our countermodel.

### 4. Feature-correspondence distillation is established unsupervised segmentation

[Hamilton et al., Unsupervised Semantic Segmentation by Distilling Feature Correspondences](https://arxiv.org/abs/2203.08414), ICLR 2022; arXiv v1 dated 16 March 2022. Full manuscript sections 3–4, equations (3)–(4), and evaluation protocol read.

STEGO freezes a pretrained representation and trains a segmentation representation to preserve feature correspondences, including attractive and repulsive relationships. Its unsupervised clustering evaluation uses Hungarian alignment against reference labels; a linear-probe evaluation is a separate protocol.

This directly weakens novelty claims for frozen-feature guidance, correspondence distillation or anonymous semantic clustering. It is not evidence that image-only cardiac clusters carry preassigned RV/MYO/LV names without a semantic bridge. No theorem of anatomical correctness is imported. Its cluster scores are not interchangeable with this project's fixed-name, no-reference-mapping evaluation.

### 5. The coassignment BCE has an almost exact prior objective

[Hsu, Lv, Schlosser, Odom and Kira, Multi-class Classification without Multi-class Labels](https://arxiv.org/abs/1901.00544), ICLR 2019; arXiv v1 dated 2 January 2019. Full derivation in section 3, equations (4)–(7), and supervision variants in section 4 read.

The predicted similarity is \(f(x_i)^\top f(x_j)\) for categorical distributions, and the objective is binary cross-entropy against pairwise same/different targets. This matches the central algebra of CQA's pooled coassignment loss before its smoothing and balancing details.

The pairwise relation source determines supervision: the paper considers supplied pairwise labels, transfer of a similarity network from an auxiliary labeled domain, and semi-supervised variants. CQA's possible distinction lies in how image-based relations are obtained and whether their feedback helps a constrained anatomical bootstrap. The loss formula or prototype-permutation invariance alone is not a defensible novelty claim.

### 6. Statistical region merging already connects image statistics to grouping

[Nock and Nielsen, Statistical Region Merging](https://www.lix.polytechnique.fr/Labo/Frank.Nielsen/pdf/2004-StatisticalRegionMerging-PAMI.pdf), IEEE TPAMI 26(11), 2004. Statistical-image setup, merging predicate, bounded-difference theorem, Corollary 1 and its proof read.

The statistical model decomposes image values into independent bounded random variables; concentration controls differences between region means for the stated statistical regions. It motivates an efficient image-region merge predicate.

This establishes strong prior overlap for intensity-based region evidence and merging. It does not identify anatomical class names. Its independence/model assumptions do not become true for checkerboard pixels or adjacent cine frames by detaching a target. CQA's deterministic thresholds are not the paper's calibrated concentration bounds. A matched classical-merging control is necessary before attributing gains to audit-driven learning.

### 7. Self-training can improve imperfect labels under explicit population structure

[Wei et al., Theoretical Analysis of Self-Training with Deep Networks on Unlabeled Data](https://arxiv.org/abs/2010.03622), ICLR 2021; inspected arXiv v5, revised 20 April 2022 (initial preprint 7 October 2020). Definitions/assumptions 3.1–3.3 and 4.1, Theorems 3.6/4.3, and Appendices A/B proofs read.

Theorem 4.3 assumes classwise pseudolabel error below \(1/3\), sufficient class-conditional expansion (\(c>3\)), and small true-class neighborhood inconsistency \(\mu\). A population minimizer of the specified consistency/pseudolabel objective has error at most \(2\mathrm{Err}(G_{\rm pl})/(c-1)+2c\mu/(c-1)\). The unsupervised result recovers classes up to permutation under its own expansion, separation, and mass assumptions.

This is a serious existing positive answer to generic “can self-training denoise?” questions. Self-Audit has not established the population assumptions, nonempty accurate initial names, objective equivalence, or optimization to such a minimizer. Adjacent graph nodes/frames are not IID examples. Our gap must concern grounding and admissible support, not the blanket absence of self-training theory.

### 8. Noisy-label correction needs an identifiable noise model

[Natarajan et al., Learning with Noisy Labels](https://proceedings.neurips.cc/paper_files/paper/2013/hash/3871bd64012152bfb53fdf04b401193f-Abstract.html), NIPS 2013; [primary PDF](https://papers.nips.cc/paper_files/paper/2013/file/3871bd64012152bfb53fdf04b401193f-Paper.pdf), class-conditional noise setup and Lemma 1 read; its two-by-two unbiasedness calculation checked algebraically.

Setting: binary labels independently flipped with class-dependent rates. Known rates whose sum is below one permit a corrected loss whose conditional expectation equals the clean loss. Finite-sample guarantees additionally use IID data and function-class assumptions; none are transferred here.

Cine pseudo-label mistakes can be spatially correlated, image-dependent, and systematically misnamed. The noise rates are not supplied by entropy. This result narrows the gap: imperfect labels are not inherently unusable, but a valid noise/semantic model must be established. Our influence-weighted corruption result is a simpler deterministic sufficient condition, not a replacement for this literature.

### 9. Risk control is available when calibration losses are observable

[Angelopoulos et al., Conformal Risk Control](https://arxiv.org/abs/2208.02814), ICLR 2024; inspected arXiv v4 revised 13 June 2025 (initial preprint 4 August 2022). Theorem 1 and its full proof read. The OpenReview PDF endpoint returned HTTP 403; the authors' arXiv manuscript was accessible.

For exchangeable bounded, non-increasing, right-continuous loss curves and a feasible endpoint, the corrected calibration-risk threshold controls expected next-example loss. The proof compares a symmetric threshold on \(n+1\) loss curves to the implementable conservative threshold and uses exchangeability.

Anatomical loss curves require labels or an independently justified observable equivalent. The theorem does not certify hidden anatomical error from image-only entropy. With patient-level calibration, dependence within a patient can be included inside a single loss; treating frames as independent calibration patients is unjustified. This result controls expected risk, not automatically every patient's selective conditional risk.

### 10. Recent adaptive risk control also does not remove the semantic-label requirement

[Blot et al., Automatically Adaptive Conformal Risk Control](https://proceedings.mlr.press/v258/blot25a.html), AISTATS 2025, PMLR 258, 19–27. Primary PDF section 2/Theorem 1 and its proof, and section 3.2 segmentation application read.

The method studies richer weighted/conditional risk control via function classes, symmetry/exchangeability and an optimization condition. Its segmentation loss explicitly depends on the reference mask; calibration is a distinct data stage.

It is a close existing answer if the project permits labeled calibration and seeks conditional risk control. Under the present prohibition on target masks for threshold selection, that operational ingredient is unavailable. No adaptive-conformal theorem is claimed for CQA witnesses. The 2026 preprint “Anytime-Valid Conformal Risk Control” also surfaced in the current search, but only its metadata/abstract were screened; its theorems were not used or claimed applicable.

## Literature-to-gap map

| Candidate gap | Nearest existing result that may already answer it | Remaining issue here | Classification |
|---|---|---|---|
| “Can diffusion converge?” | Zhou resolvent derivation | Nothing novel at that level; P2 reveals a default support obstruction | Classical operator fact + method-family limitation |
| “Can unlabeled representations yield useful partitions?” | STEGO; Wei's unsupervised population result | Grounding fixed anatomical names; admissible acquisition priors | Task-level problem under restricted information |
| “Can pairwise targets train classes?” | Hsu et al. inner-product BCE | Validity and incremental value of this image-derived target source | Conditional research hypothesis; loss itself established |
| “Can an image interface be removed statistically?” | Statistical Region Merging | Anatomy-specific meaning, proposal selection and cine dependence | Missing bridge, not absence of merging theory |
| “Can pseudo-labels improve?” | Wei et al.; Natarajan et al. | Initial semantic accuracy, corruption/expansion assumptions, reachable support | Task-specific applicability |
| “Can abstention control risk?” | El-Yaniv–Wiener; conformal and adaptive risk control | No target-labeled calibration; justified ambiguity/violation model | Open under the chosen information constraint |
| “Can better feature matching improve named output?” | Representation distillation + elementary margin stability | Bias of named readout and a coupling to the trained student head | Missing theoretical/empirical bridge |

## Ranked gap assessment

Scores below are qualitative judgments, not bibliometric or experimental measurements.

1. **Named selective supervision under explicit priors and imperfect dependent evidence.** Strong theoretical support for the obstruction; highest task relevance; substantial prior overlap for component lemmas; nontrivial because semantic grounding, feasible support and risk must hold together. This is the leading problem.
2. **Reachability of accepted support under gated learning.** Strong source and mathematical support; high practical relevance. As a standalone claim it risks being a state-machine/operator characterization. Useful as a necessary part of the leading problem, not an automatic new learning method.
3. **Anonymous audited partition feedback.** Feasible algebraically and potentially useful, but exact prior-loss overlap is high. Its meaningful claim requires a valid relation source and gains beyond assembly/classical merging. It is the alternative if the human narrows the target task.
4. **Numerical convergence, ordinary distillation, or architecture scale.** Mature prior art; weak novelty; does not resolve the earliest semantic bottleneck. Not recommended as the central research gap.

An implementation mismatch with an intended behavior is an engineering defect; a mathematically correct but support-limited operator is a method-family limitation; unknown links between witness scores and latent anatomy are missing understanding; and simultaneous named recovery/risk/coverage under the information contract is the task-level research problem. These categories must not be conflated.

No absence-of-prior-art, first-ever, SOTA, or publication-ready claim is established by this search.
