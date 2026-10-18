# R6 — Independent prior-art, falsification, and generalization dossier

## Immutable independent draft — 2026-10-18

**Status: Phase I, LOCAL_ONLY. Every candidate method is PROPOSED_NOT_APPROVED.** Author: isolated R6 worker, requested gpt-6-astra / ultra; dispatch parameters are not independent proof of backend identity. This first draft was written before reading any other worker dossier or interrupted draft. AI-assisted searching, source inspection and drafting are disclosed; no human-read claim is made. Production was read-only. Pinned project SHA: `f00e4faa52204f8e6863f76a493684b68ed3a132` (verified by `git rev-parse HEAD`).

Read: session STATE.md, AGENTS.md, ACTIVE_PIPELINE.json, CURRENT.md, RUNBOOK.md, ASTRA_NEXT.md, and RTK guidance. A quick memory pass supplied historical provenance cautions; current conclusions below use the pinned current documents and freshly inspected primary sources. Current source execution/gradient verification is the coordinator's separate responsibility. I do not independently elevate documented current behavior into executed evidence.

### Independent conclusion

The broad mechanism space is crowded. Exact nullspace corrections, multiview feature upsampling, coarse-label distribution matching, direct image-energy regularization, dependent weak-source modeling, structured distillation, masked consistency, and self-training denoising all have close prior work. The defensible unresolved question is narrower: **when image-derived, potentially dependent and semantically wrong cardiac evidence can improve named per-class dense segmentation, under explicit information and inference contracts.** No reviewed paper establishes that guarantee for this contract.

Do not make a universal “without labels, cannot improve” claim: noisy teacher supervision can be denoised with extra assumptions/inductive information. Conversely, image-only consistency, agreement or smoothness alone cannot identify fixed names or validate an unreliable prior. The first useful experiment is a matched simpler replacement, not a new collection of modules.

## Search scope and source verification

Search date: 2026-10-18. This is a scoped adversarial mechanism search, not a systematic review, exhaustive novelty certificate, or publication-readiness assessment. Primary hosts checked: arXiv, CVF proceedings, PMLR, JMLR, ICLR/OpenReview, author/institution pages and official GitHub repositories. Search strings included exact titles plus: null-space data consistency; label super-resolution; regularized weak segmentation losses; source dependencies; structured knowledge distillation; masked image consistency; self-training expansion; feature upsampling; test-time adaptation; domain generalization. Secondary search hits were leads only.

The literature-review skill informed source extraction. Its comprehensive-review PDF/PRISMA/AI-figure workflow is outside this explicitly scoped dossier task; no figure-generated evidence or full-systematic-review status is asserted. OpenReview PDF endpoints repeatedly returned browser verification; several CVF direct opens returned 403/internal errors. Those cases were replaced by accessible author arXiv versions where available, and otherwise marked limited-access. No CAPTCHA circumvention, dependency installation, model weight download or dataset download occurred.

### Primary source register

- **P1.** Johannes Schwab, Stephan Antholzer, Markus Haltmeier, *Deep Null Space Learning for Inverse Problems: Convergence Analysis and Rates*. arXiv:1806.06137, 2018 preprint; journal bibliographic record is Inverse Problems 35, 025008 (2019), DOI 10.1088/1361-6420/aaf14a. Inspected [author preprint PDF](https://arxiv.org/pdf/1806.06137): Definition 3.2; training Eq. (3.2); Theorems 2.4, 2.8, 3.3–3.4. Publisher full text was not independently opened; theorem numbering here refers to the inspected preprint.
- **P2.** Stephanie Fu, Mark Hamilton, Laura E. Brandt, Axel Feldmann, Zhoutong Zhang, William T. Freeman, *FeatUp: A Model-Agnostic Framework for Features at Any Resolution*, ICLR 2024. Inspected [v2, 1 April 2024](https://arxiv.org/html/2403.10516v2), §3 Eq. (1), §3.1–3.3, supplemental §6.4 and §6.13; [official code](https://github.com/mhamilton723/FeatUp) at `6b5a6c0e91f75e69194807128dcbc39c3084a30d`. Multiview low-resolution reconstruction; trained JBU and per-image implicit variants. Label-free feature training does not establish named cardiac classes.
- **P3.** Thomas Wimmer, Prune Truong, Marie-Julie Rakotosaona, Michael Oechsle, Federico Tombari, Bernt Schiele, Jan Eric Lenssen, *AnyUp: Universal Feature Upsampling*. [v2, 16 February 2026](https://arxiv.org/html/2510.12764v2); arXiv metadata and [official repository](https://github.com/wimmerth/anyup) explicitly state ICLR 2026 Oral. §4.1–4.3, Eq. (2), Appendix C.1; code `351807a9c4287368732cc247f26c7c81c9139af4`. Encoder-agnostic inference is already prior art. Current code's cosine-plus-MSE normalizes MSE using target channel min/max (loss.py:12–24).
- **P4.** Kolya Malkin, Caleb Robinson, Le Hou, Rachel Soobitsky, Jacob Czawlytko, Dimitris Samaras, Joel Saltz, Lucas Joppa, Nebojsa Jojic, *Label Super-Resolution Networks*, ICLR 2019. [Microsoft author record](https://www.microsoft.com/en-us/research/publication/label-super-resolution-networks/) verified authors/status and explicitly required coarse/fine joint label distribution. [Conference PDF](https://openreview.net/pdf?id=rkxwShA9Ym) blocked by browser verification. Therefore abstract-level mechanism/assumption verified, exact equation and experiment audit NOT VERIFIED; no claim that this paper's specific joint distribution was label-free.
- **P5.** Novi Quadrianto, Alex J. Smola, Tibério S. Caetano, Quoc V. Le, *Estimating Labels from Label Proportions*, JMLR 10:2349–2374, 2009. [Journal PDF](https://jmlr.org/papers/volume10/quadrianto09a/quadrianto09a.pdf), §2: known bag proportions with full-column-rank mixing matrix. Known proportions are supervision; teacher probabilities are not automatically calibrated proportions.
- **P6.** Meng Tang, Federico Perazzi, Abdelaziz Djelouah, Ismail Ben Ayed, Christopher Schroers, Yuri Boykov, *On Regularized Losses for Weakly-supervised CNN Segmentation*, ECCV 2018, pp. 507–522. [CVF record](https://openaccess.thecvf.com/content_ECCV_2018/html/Meng_Tang_On_Regularized_Losses_ECCV_2018_paper.html); inspected [arXiv v2, 10 April 2018](https://arxiv.org/pdf/1803.09569), §2 Eqs. (1)–(6), §3. Actual supervision is partial user scribbles. Official rloss code `1caa759e568db2c7209ab73e73ac039ea3d7101c`, DenseCRFLoss.py:21–47 implements an image-affinity quadratic and gradient; it is not a new semantic source.
- **P7.** Hoel Kervadec, Jose Dolz, Meng Tang, Eric Granger, Yuri Boykov, Ismail Ben Ayed, *Constrained-CNN losses for weakly supervised segmentation*, Medical Image Analysis 54:88–99, 2019, DOI 10.1016/j.media.2019.02.009. [Author-hosted published PDF](https://cs.uwaterloo.ca/~m62tang/ConstrainedCNNLosses.pdf), [institution record](https://uwspace.uwaterloo.ca/items/4d5c7f7c-be38-4d01-a28b-1e2130931dd7), [official code](https://github.com/LIVIAETS/SizeLoss_WSS). Partial labels plus size/tag constraints; penalty enforcement does not itself guarantee exact satisfaction. Source inspected at abstract/front-matter scope; do not infer unlabeled size priors from its results.
- **P8.** Stephen H. Bach, Bryan He, Alexander Ratner, Christopher Ré, *Learning the Structure of Generative Models without Labeled Data*, ICML 2017, PMLR 70:273–282. [Paper](https://proceedings.mlr.press/v70/bach17a/bach17a.pdf), §3, §4 Eqs. (6)–(8), Theorem 1 and Corollary 2; [supplement](https://proceedings.mlr.press/v70/bach17a/bach17a-supp.pdf), Appendix A.2. Dependency recovery assumes model specification, information/curvature, and nonvanishing dependency strengths. It is not an arbitrary-source reliability oracle.
- **P9.** Yifan Liu, Ke Chen, Chris Liu, Zengchang Qin, Zhenbo Luo, Jingdong Wang, *Structured Knowledge Distillation for Semantic Segmentation*, CVPR 2019, pp. 2604–2613. [CVF record](https://openaccess.thecvf.com/content_CVPR_2019/html/Liu_Structured_Knowledge_Distillation_for_Semantic_Segmentation_CVPR_2019_paper.html); inspected [arXiv v1](https://arxiv.org/pdf/1903.04197v1), §3.1. Pixel, pairwise and holistic distillation of a semantic teacher. **Version warning:** the latest arXiv 1903.04197 is retitled *Structured Knowledge Distillation for Dense Prediction*; do not attribute its extension to the original CVPR paper silently. [Official repository](https://github.com/irfanicmll/structure_knowledge_distillation) also contains later extensions.
- **P10.** Lukas Hoyer, Dengxin Dai, Haoran Wang, Luc Van Gool, *MIC: Masked Image Consistency for Context-Enhanced Domain Adaptation*, CVPR 2023. [v2, 24 March 2023](https://arxiv.org/html/2212.01322v2), §3.1–3.2, Eq. (12), supplemental D.1/G. Masked target training plus EMA full-image teacher is already explicit. The host objective includes labeled source data. D.1 tests random segment-label noise, which differs from whole-class systematic semantic swaps. Official code `2f932a98b5dd9f598aaeb32411863ceea0809314`, masking_consistency_module.py:54–143.
- **P11.** Colin Wei, Kendrick Shen, Yining Chen, Tengyu Ma, *Theoretical Analysis of Self-Training with Deep Networks on Unlabeled Data*, ICLR 2021. [Author preprint](https://arxiv.org/pdf/2010.03622), Assumptions 3.3 and 4.1, Theorems 3.6/4.3, Eq. (4.1), Appendix A.2/C.2. The named-label theorem is conditional on true-class accuracy and neighborhood expansion/separation; unsupervised recovery uses label-permutation error. These are different guarantees.
- **P12.** Mark Hamilton, Zhoutong Zhang, Bharath Hariharan, Noah Snavely, William T. Freeman, *Unsupervised Semantic Segmentation by Distilling Feature Correspondences* (STEGO), ICLR 2022; [arXiv:2203.08414](https://arxiv.org/abs/2203.08414), [official code](https://github.com/mhamilton723/STEGO) at `eb4d6b521740bd4265681b353547f2ffca65d673`. Source utils.py:219–247 accumulates prediction/target contingency and obtains Hungarian matching. Cluster quality evaluated after reference matching is not fixed-name deployment accuracy.
- **P13.** Dequan Wang, Evan Shelhamer, Shaoteng Liu, Bruno Olshausen, Trevor Darrell, *Tent: Fully Test-Time Adaptation by Entropy Minimization*, ICLR 2021. [Paper](https://arxiv.org/pdf/2006.10726), [official code](https://github.com/DequanWang/tent) at `e9e926a668d85244c66a6d5c006efbd2b82e83e8`; tent.py:49–61,96–110: entropy gradient updates and batch-statistic normalization. This changes the inference protocol.
- **P14.** Ishaan Gulrajani, David Lopez-Paz, *In Search of Lost Domain Generalization*, ICLR 2021; [author preprint](https://arxiv.org/pdf/2007.01434), model selection and matched evaluation. Its benchmark conclusion does not prove ERM best on every task; it supports a carefully controlled ERM baseline and explicit selection rule.
- **P15, scoped lead only.** *LoftUp: Learning a Coordinate-Based Feature Upsampler for Vision Foundation Models*, [ICCV 2025 proceedings PDF](https://openaccess.thecvf.com/content/ICCV2025/papers/Huang_LoftUp_Learning_a_Coordinate-Based_Feature_Upsampler_for_Vision_Foundation_Models_ICCV_2025_paper.pdf), not CVPR 2025; *RaysUp: Ultra-light Universal Feature Upsampling via Geometry-Aware Ray Representation*, [arXiv:2606.22749v1](https://arxiv.org/abs/2606.22749), author metadata says ECCV 2026. These reinforce search freshness; detailed equations/results and final proceedings status of RaysUp were NOT audited. Neither is used to establish a theorem or quantitative superiority.

Official source copies and SHA256 values are saved under `probes/R6/official_code_sources.json`. No repository code was executed. Copies are research evidence, not proposed dependencies.

## Collision matrix

| Potential contribution | Nearest prior / shared mechanism | Material difference in assumptions | Remaining gap | Simplest plausible baseline |
|---|---|---|---|---|
| Exact coarse-preserving native correction | P1 nullspace network | Known linear forward operator and reconstruction examples versus uncertain semantic observations | Correct observation space; trustworthy direction inside kernel | Fixed image-guided correction plus projection; conventional head |
| New multiview native feature learning | P2/P3; jitter/crop feature reconstruction and guided upsampling | Natural-image feature behavior; no fixed RV/MYO/LV truth | Cardiac semantic sufficiency and teacher validity | Bilinear, nonlearned JBU/guided filter, then one frozen pretrained upsampler only after lineage approval |
| Coarse-to-dense label learning | P4/P5; aggregate output matching | Externally known label relationship/proportions | Whether Q actually represents coarse measurement of native class mass | Support-aware coarse CE versus aggregate probability matching, separate versions |
| Native image-energy proposal and distillation | P6; CRF/graph-cut proposals or direct regularized loss | Scribbles or correct semantic unary evidence | Energy/anatomy alignment under missing and wrong anchors | Same fixed energy as direct loss versus detached proposal; no learned refiner |
| Shape/volume constraint learns anatomy | P7 | Partial labels and domain constraints | Valid broad anatomical prior under pathology/missing anatomy | Soft size interval, audited provenance; no per-patient GT-derived bounds |
| Multi-source agreement gives reliability | P8; weak-source generative modeling | Identifiability and model specification | Shared failure/semantic polarity cannot be estimated from duplicates alone | Group equal-lineage sources, equal weight by source group, leave-one-group-out |
| Pairwise or holistic feature teacher | P9/P12 | Semantic teacher or clusters evaluated by matching | Fixed-name identity and native evidence | Plain feature cosine + coarse CE; no new structure module |
| Full-image teacher / masked student | P10 | Source labeled semantics and target-domain fitting | Robustness to coherent wrong classes in current bank | Fixed-bank masked coarse CE; no EMA initially |
| Self-training surpasses teacher | P11 | True-class expansion/separation and teacher error conditions | Unverified conditions and dense pixel dependence | Fixed teacher, one consistency loss; separately test proposal/transfer |
| Generalizes without target supervision | P13/P14 distinguish adaptation and selection | Fit target normalization or online optimizer changes task | Independent-cine deployment, missing metadata, cohort shift | Frozen training normalization versus per-cine normalization, explicit protocols |

The shared generic mechanism is not the novelty. None of these collisions proves the complete proposed combination already exists under this exact information contract. Likewise, composing known mechanisms is not by itself evidence of a research contribution.

## Evidence cards and falsification

### R6-C1 — Exact nullspace algebra is not a truth certificate

- **Status:** LITERATURE_RESULT + DERIVED_RESULT + COUNTEREXAMPLE.
- **Scope/assumptions:** finite output space; linear D; right inverse U with D U = I; any detail r. The correction is on the variable to which D actually applies.
- **Formal statement:** for every c,r, `D[U c + (I - U D) r] = c`. This is a data-consistency identity. It neither selects the correct element of `ker(D)` nor prevents semantic errors. If D averages logits, it does not imply `D softmax(z) = softmax(D z)`.
- **Proof/missing step:** expand the product and use D U = I. P1 Definition 3.2 is the closest mechanism; its regularization theorems require an appropriate reconstruction manifold and Lipschitz map. No generic Dice theorem follows.
- **Counterexample:** two binary pixels with average probability 0.5 admit native vectors (1,0) and (0,1), both observation-consistent, with MSE 0 and 1 against truth (1,0). Binary logits (4,0) average to 2, while mean sigmoid is 0.741007 and sigmoid(2) is 0.880797.
- **Evidence:** `bounded_checks.py/json`; P1 Definition 3.2/Thm.3.3. Manufactured outputs, no patient evidence.
- **Implication/limitation:** retain decomposition as a control when useful; do not protect it as the scientific hypothesis. Declare logits/probabilities/support/mass before claiming conservation. This finding is not a defect claim against current source, which documents a logit decomposition.
- **Falsifiable prediction:** same coarse discrepancy can coexist with very different classwise native accuracy; a cell-permuted detail control can preserve D yet destroy anatomy.

### R6-C2 — Named semantics require symmetry-breaking information

- **Status:** COUNTEREXAMPLE + DERIVED_RESULT.
- **Scope/assumptions:** any learner receives only observed tuple O = (X,M,F,A,Q,provenance); consider an admissible pair of worlds with identical law of O and true RV/LV names exchanged. No assumption in this counterexample asserts trustworthy class polarity of A/Q/M.
- **Formal statement:** for any deterministic prediction function g(O), on binary balanced names with `Y_B = 1 - Y_A`, `R_A(g) + R_B(g) = 1` under 0–1 error, hence max risk ≥ 1/2. Randomized g obeys the same expectation statement. This does not apply once admissible assumptions exclude one world by valid named evidence.
- **Proof:** pointwise a binary output is correct in exactly one of the two worlds. This is a classical identifiability argument, not novel theory.
- **Counterexample to an overstrong negative claim:** verified noisy named anchors plus a valid class-preserving smoothness prior can break the symmetry and permit denoising; P11 is a positive conditional precedent.
- **Prior/evidence:** P12 source shows reference-based Hungarian matching; P11 separates permutation-invariant from named-label error. A fixed-label cardiac deployment cannot quietly use reference matching.
- **Implication/limitation:** unverified cardinal coordinates, clustering, high confidence or source agreement do not prove a named class. Anatomical priors may help, but their meaning/provenance must be declared and their failures tested.
- **Falsifiable prediction:** swapping all RV/LV channels consistently leaves class-symmetric regularizers and feature objectives unchanged. A valid named semantic signal must respond differently for a declared reason.

### R6-C3 — Refinement is viable conditionally; compare direct energy against proposal learning

- **Status:** PROPOSED_NOT_APPROVED + DERIVED_RESULT.
- **Scope/assumptions:** baseline b, image-derived proposal t, student s and truth y live in the same finite probability-vector space/distribution; square loss has finite moments. One fixed energy uses only authorized training information. Coordinate/support transport is exact.
- **Formal statement:** by the L2 triangle inequality, `sqrt(E ||s-y||²) ≤ sqrt(E ||t-y||²) + sqrt(E ||s-t||²)`. Therefore proposal RMSE plus transfer RMSE below baseline RMSE is a sufficient improvement condition for this loss, not Dice. No independence assumption is needed. The truth-related terms are latent here.
- **Proof/missing step:** standard Minkowski inequality; missing operational step is identifying proposal truth improvement without reference masks. Low transfer error alone only says the student copied its proposal. Energy descent does not supply that missing step.
- **Counterexample:** a visually sharp but wrong septal boundary or a correct boundary assigned to the wrong class lowers a chosen image cost while worsening named masks. A strongly wrong anchor can propagate confidently.
- **Closest prior/evidence:** P6 Eqs.(1)–(6) and §3 already compare regularized losses against proposal generation. This proposal must beat that simpler baseline, as well as no-detail training.
- **Implications:** compare the same native energy applied (a) directly as a loss and (b) through a detached proposal, matched in gradient/compute budget. Freeze proposal parameters on TRAIN. No generic “refinement beats teacher” claim.
- **Limitations/falsifiable prediction:** synthetic distractor edges, intensity inversions, absent RV seeds and wrong anchors should produce bounded/flagged behavior; failure to beat direct energy or interpolation eliminates added method complexity. Patient benefit remains NOT RUN.

### R6-C4 — Duplicated sources do not create evidence

- **Status:** LITERATURE_RESULT + DERIVED_RESULT + COUNTEREXAMPLE.
- **Scope/assumptions:** source errors e_j have common zero mean, variance σ² and pairwise correlation ρ; this toy model is not a claim that cardiac errors are unbiased/exchangeable.
- **Formal statement:** `Var(m⁻¹ Σ_j e_j) = σ²[ρ + (1-ρ)/m]`. For exact duplicates ρ = 1, variance does not decrease. More generally, observing k deterministic copies of A adds no information conditional on A.
- **Proof:** expand covariance of the mean; deterministic duplication leaves the sigma-algebra unchanged. If all sources share a bias, averaging also leaves that bias. P8 is the closest structured-source precedent, with explicit recoverability assumptions.
- **Counterexample:** ten variants of one wrong RV seed generator can unanimously mislabel RV; naive conditionally independent product aggregation grows confidence without new independent evidence.
- **Evidence:** bounded check gives variance 0.1 for ρ=0, 0.91 for ρ=0.9, 1.0 for ρ=1 at m=10, σ²=1. P8 §4 conditions (6)–(8), Theorem1; AppendixA.2 is accessible.
- **Implication/limitations:** first deduplicate by computational lineage and inspect conflict/abstention separately. A learned label model is not automatically identifiable from this bank. Patient pixels are not independent replicates.
- **Falsifiable prediction:** cloning a source must not improve the claimed confidence or effective sample size; removing all descendants of one source should expose dependence hidden by per-map leave-one-out.

### R6-C5 — Self-training improvement has a real theorem, with unavailable premises here

- **Status:** LITERATURE_RESULT; current-setting applicability NEEDS_EVIDENCE.
- **Scope/assumptions:** P11 population classifier, true-class neighborhood separation `R_B(G*) ≤ μ`; worst per-true-class pseudolabel error `ā < 1/3`; `(ā,c̄)` expansion with c̄ > 3; c = min(1/ā,c̄); optimizer is a global minimizer of paper Eq.(4.1).
- **Formal statement:** P11 Theorem4.3 gives `Err(Ghat) ≤ 2 Err(Gpl)/(c-1) + 2 c μ/(c-1)`. Its `Err`, μ, ā, true-class expansion and population minimization assumptions are not measured certificates in this project. A reduction in CE under AdamW is not the theorem's conclusion.
- **Proof/missing step:** literature proof in Appendix A.2; missing transfer from independent examples/classifier neighborhoods to dense cardiac pixels and concrete augmentations, plus true-class teacher adequacy. Paper's pseudo-label setup uses labeled-source or semi-supervised teachers; an image-derived teacher could enter the mathematical abstraction only if it independently satisfies the assumptions.
- **Counterexample:** a coherent entire RV class mapped to BG violates the per-class error premise even when aggregate teacher accuracy is high. Consistency reinforces that error; augmentations that remove anatomy can violate separation.
- **Prior/implication:** the idea of surpassing a noisy teacher with consistency is prior art, and it is possible. A proposed theorem must explicitly relax or address this setting's actual violated assumptions.
- **Limitations/falsifiable prediction:** stress tests should use classwise coherent noise and missing classes, not only iid flips. Stop before training escalation when no justified named RV support exists. This is a premise gate, not an anatomical validation.

### R6-C6 — Masking and structure transfer have close baselines and distinct supervision

- **Status:** LITERATURE_RESULT + PROPOSED_NOT_APPROVED.
- **Scope/assumptions:** frozen full-image teacher distributions; masked student inputs; explicit support and transformed target coordinates. Current feature-only full-view loss is not masked semantic supervision.
- **Formal candidate:** `L_mask = E[w_i CE(q_i, p_theta(mask(X))_i)]` with stop-gradient q, mask policy and valid weights frozen on TRAIN. This is a known consistency pattern, not a novelty claim. No inequality to true anatomical risk is asserted.
- **Proof/missing step:** no proof of improvement in this setting; masking removes evidence and can make a target unknowable. Class identity inherited from q is not corrected merely by occlusion.
- **Counterexample:** systematic teacher RV→BG mapping remains a minimizer under masked and unmasked consistency. Masking every informative RV region teaches the conditional average, possibly BG.
- **Closest prior/evidence:** P10 §3.2/Eq.(12), supplementary D.1; official code lines70–83 and132–143. P9 §3.1 distills pixel/pairwise/holistic semantic structure. P10's source-mask data and P9's trained semantic teacher are additional supervision relative to this contract.
- **Implication/limitation:** if masking is later considered, first fixed-bank masked CE versus unmasked CE, then separately any EMA or feature structure. Random pseudo-label-noise robustness from MIC does not validate whole-class cardiac prior failures.
- **Falsifiable prediction:** any claimed robust method must retain dense classwise performance under one-source coherent corruption and fail visibly when all named evidence for one class is removed.

### R6-C7 — Frozen inference, per-image fitting and target-cohort adaptation are different claims

- **Status:** CODE_FACT (external official code) + PROPOSED_NOT_APPROVED protocol.
- **Scope/assumptions:** define inference as `yhat_i = f_theta(T(X_i))`, where theta and all cohort-derived parameters were fixed using allowed TRAIN information. T may compute statistics from the current cine alone if specified before evaluation.
- **Formal distinction:** if preprocessing uses `a = Fit({X_j: j in target cohort})`, then `yhat_i = f_theta(T(X_i;a))` depends on other target patients. Even with frozen theta, this is a cohort-adaptive/transductive protocol, not independent-cine inference. Per-cine test-time optimization also differs from one frozen forward pass, although it need not access other patients.
- **Proof:** compare functional dependence; replace unrelated X_j while holding X_i fixed. No statistical performance theorem is claimed.
- **Counterexample:** target normalization fitted using the entire hospital cohort changes one patient's predictions when another patient arrives. A cached feature PCA basis or updated BN statistics can create the same dependence.
- **Prior/evidence:** P2 implicit training creates an optimizer for each image (official code:167–237). P13 explicitly updates parameters and batch statistics. P14 shows why selection protocol belongs to the evaluated algorithm.
- **Implications:** report three separate regimes: frozen one-cine; per-cine fitting with reset; cohort/online adaptation. Preprocessing fit partitions and hidden teacher calls are part of provenance.
- **Limitations/falsifiable prediction:** duplicate/reorder/add unrelated target cines; frozen independent-cine predictions must remain unchanged up to declared deterministic tolerance. Failures invalidate the inference claim, not necessarily the method in a separately declared adaptive setting.

### R6-C8 — Evaluation must make RV failure, BG domination and abstention visible

- **Status:** COUNTEREXAMPLE + PROPOSED_NOT_APPROVED evaluation contract.
- **Scope/assumptions:** predictions frozen before authorized scoring; no TEST/VAL selection; fixed named mapping; patients are evaluation units; true absent-class cases have an explicit policy. Historical exposed validation is development only.
- **Formal statement:** a predictor correct on 90 BG pixels and missing all 10 RV pixels has accuracy 0.9 but RV Dice 0. Aggregating pixels cannot establish each foreground class improved. Selective risk at acceptance coverage c is conditional on acceptance, so it cannot replace full-domain error.
- **Proof/counterexample:** arithmetic; a reject-all predictor has zero coverage and undefined selective risk, not perfect segmentation. Filtering all difficult RV pixels can improve accepted-set metrics while providing no RV output.
- **Closest prior/evidence:** P12 matching implementation illustrates a different clustering evaluation target; current STATE.md explicitly preserves UNKNOWN=255. Bounded checks reproduce BG example. No patient score is computed here.
- **Implications:** primary future report should show dense fixed-name Dice per RV/MYO/LV, per-patient macro and worst-tail summaries, patient-level uncertainty, absent-class policy, coverage and rejection by class/support. Gated and raw outputs stay separate. For full-domain scoring, rejected true-class pixels cannot disappear from denominators; they are missing predictions, not relabeled BG. Report BG separately.
- **Limitations/falsifiable prediction:** add rejected RV pixels or duplicate BG slices; the principal claim must not improve solely because denominators shrink or background grows. Classwise teacher/proposal/student comparisons must reveal whether any gain is actual transfer versus increased coverage.

## Candidate order and kill criteria

All below are **PROPOSED_NOT_APPROVED**, not a Phase-II authorization.

1. Preserve current coarse-only student and conventional-head matched control. Check exact variable/observation semantics before treating Q as a native class-mass observation.
2. Compare interpolation and one fixed image-guided refinement using the same coarse evidence and valid support. Stress geometry, false edges, topology variations, missing/contradictory anchors, and coherent class swaps. Do not tune on reference masks.
3. Compare direct image-energy training against detached-proposal transfer, sharing the energy and information. A proposal failure is not repaired by adding a student; transfer and proposal evidence must be separated.
4. Only if the information premise remains plausible, compare fixed masked consistency; separate EMA and extra feature/structure mechanisms. Existing upsamplers are collision controls, not approved pretrained dependencies.

**No-GT kill criteria:** source cloning changes reliability; semantic output survives arbitrary source-class permutation without a declared symmetry breaker; cohort changes alter allegedly independent inference; zero/unsupported evidence becomes BG supervision; a metric hides UNKNOWN or class failures; claimed native gradient exists only from resized targets; increased energy quality is called anatomy improvement; masked/temporal views violate the target correspondence. Passing these gates establishes software/protocol validity only. Without new authorized evidence, true anatomical benefit is still unresolved.

**Scientific kill criteria for a future authorized evaluation:** no patient-level foreground benefit over the simplest matched baseline; RV collapse despite aggregate improvement; gain vanishes under equal coverage; proposal improves but student cannot transfer; improvement depends on target-cohort fitting while advertised frozen; or result depends on inspected TEST/VAL tuning. Prespecify meaningful effect and failure bounds with the human before data access; no post-hoc threshold proposed here.

## Executed evidence and boundaries

Executed: primary web searches and exact source reads; project SHA/current contract reads; six tiny official-code snapshots with SHA256; pure-Python manufactured scalar check (`5 checks`, exit 0). The probe does not establish a general theorem beyond the explicitly derived identities, patient performance, training feasibility, or robustness. No production edit, test suite execution, patient/GPU training, weights/datasets, dependency changes, commit, push or merge.

Memory used only as navigation/provenance caution: MEMORY.md:17–24, historical integration thread `01a13f0c-a763-7043-8bbb-e1fc27b00972`; current documentation verified separately. Initial draft will be preserved verbatim at `probes/R6/independent_initial.md`, with a SHA256 receipt. Revisions and real review responses must be appended below this draft, never silently substituted.

