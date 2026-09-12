# Candidate C: competing explanations and prior-art challenges

Date: 2026-09-12. Source HEAD: `9726e6d519c33035489ca7a5e4681a046493d89e`.

This final synthesis replaces an AGY draft rejected for overclaims and incorrect metadata. It uses only material already collected before the user stopped searching. The [evidence ledger](05_evidence_ledger.md) distinguishes method reads, abstracts and gaps. No performance study is reported here.

## 1. Start with an equivalence challenge

Candidate C can be represented as a frozen differentiable function `B(z)` plus a test-time search for an auxiliary variable `z`, feedback-derived output targets, preservation constraints and a proximity cost. Its chosen variable is the sequence of realized sampling coordinates of a recorded accepted ordinary correction. This places it in an established class of auxiliary-variable refinement and constrained counterfactual search.

That abstraction does **not** make every member equivalent. Coordinates, feature biases and network weights have different reachable output changes. Historical and current-state interventions have different conditioning states and reference points. The research task is to determine whether those distinctions explain a useful outcome, rather than assuming they do.

The exact implemented approximation and transfer are stated in [01](01_problem_formulation.md) and [02](02_mechanism_analysis.md). It selects among factual support and at most two checked proposals along one initial gradient direction. It does not certify an optimum or absence of feasible alternatives.

## 2. Mechanism-level map

| Competing mechanism | Intervention variable and signal | Temporal / preservation relationship | What Candidate C would have to add |
|---|---|---|---|
| [BRS / f-BRS, CVPR 2019/2020](https://arxiv.org/abs/2001.10331), §3 | Test-time input refinement, then intermediate feature scale/bias, driven by corrective clicks | Frozen prediction network; refine the current interaction | Evidence from an accepted transition, historical replay and a useful coordinate-specific action space |
| [G-BRS, CVPR 2022](https://openaccess.thecvf.com/content/CVPR2022/papers/Lin_Generalizing_Interactive_Backpropagating_Refinement_for_Dense_Prediction_Networks_CVPR_2022_paper.pdf), §2 | Auxiliary refinement variables with localized effects and consistency | Refine dense outputs while limiting unrelated changes | A benefit from predicted signed evidence and replayed supports beyond generic latent refinement |
| [Counterfactual Attention Learning, ICCV 2021](https://openaccess.thecvf.com/content/ICCV2021/papers/Rao_Counterfactual_Attention_Learning_for_Fine-Grained_Visual_Categorization_and_Re-Identification_ICCV_2021_paper.pdf), §3, Eqs. 4–6 | Replace attention maps and compare factual/intervened outputs in training | Attention effect contributes to learning; not previous accepted correction restitution | Specific coordinate intervention and temporal contract; subtraction itself is already known |
| [Spatially Transformed Adversarial Examples, ICLR 2018](https://arxiv.org/pdf/1801.02612), §3.3 | Optimize continuous bilinear input-sampling flow under target and deformation costs | Frozen classifier; perturb the current input | Internal support rather than image deformation, and measured correction/preservation utility |
| [Wachter et al., 2017 preprint / 2018 publication](https://arxiv.org/abs/1711.00399), counterfactual objective | Target outcome plus proximity to factual input | Minimal counterfactual search | A meaningful action restriction and temporal use; target-plus-distance objective is not new |
| [Karimi et al., FAccT 2021](https://arxiv.org/pdf/2002.06278), §§3–4 | Feasible structural interventions with action cost | Distinguishes counterfactual explanations from causal recourse | Clearly identify the computational causal object; do not imply patient-level causal explanation |
| [ActionSplice, September 2026 preprint](https://arxiv.org/html/2609.08230v1), §§3.1–3.3, Eqs. 1–2 | Learned residual transport toward matched rollback states under a changed action | Preserve committed positions; resume a frozen sampler without replaying completed evaluations at deployment | Actual replay and online support search using predicted FIX/REGRESS; no claim to first historical correction |
| [LeCor, September 2026 preprint](https://arxiv.org/html/2609.09477v1), §3.3, Eqs. 1–3 | One click-driven gradient on case adapters; meta-learn initialization and step sizes | Outer objective rewards effects on unclicked slices; adapters accumulate within a case | Frozen parameters, machine evidence and support-space intervention; no claim to first learning-for-correction |
| [GEM, NeurIPS 2017](https://papers.neurips.cc/paper/2017/file/f87522788a2be2d171666752f97ddebb-Paper.pdf), Eqs. 6–8 | Constrain updates against increased losses on past memories | Preserve prior learned performance while updating parameters | If gradient protection is added later, its optimization principle needs no new name |
| [PCGrad, NeurIPS 2020](https://papers.neurips.cc/paper_files/paper/2020/file/3fe78a8acf5fda99de95303940a2420c-Paper.pdf), Algorithm 1 | Modify conflicting task gradients | Trade-offs among simultaneous objectives | FIX/REGRESS semantics could define a use case, not a new general projection algorithm |
| [Deformable DETR, ICLR 2021](https://arxiv.org/abs/2010.04159), §4.1 | Learned sparse continuous support with sampled features | Feedforward perception | The current window is a sampling substrate; Candidate C's possible contribution lies elsewhere |

These are deliberately not all equally close. G-BRS/f-BRS are direct algorithm-family competitors; recourse and spatial perturbation challenge the objective/intervention framing; ActionSplice challenges broad temporal novelty; LeCor challenges a potential future training claim. GEM and PCGrad prevent an optimizer extension from being mistaken for a novel principle.

## 3. Six reductions an expert reviewer can demand

### R1. Replace historical coordinates by other auxiliary variables

Starting from the same frozen record, optimize feature scale/bias or a spatial bias under the same predicted restoration targets and FIX output checks. This is the natural G-BRS/f-BRS challenge. Equal raw parameter norms across coordinates and feature units are not a fair constraint; use a prespecified resource budget and report action magnitudes and outcomes.

If a cheaper auxiliary-variable action is equivalent within a precise, useful margin, coordinates have not earned a special role. A failure to reject a difference with little statistical power is not equivalence. Conversely, different coordinate heatmaps alone do not establish an advantage.

### R2. Replace historical replay by a current-state intervention

Build a current-state factual forward `C(P_cur)` and optimize its support using the same previous restoration targets. Transfer the centered innovation `C(Q_cur)-C(P_cur)`, not a difference against the historical factual candidate. Match internal depth and the outer gate; otherwise temporal reference, subtraction and compute change together.

A precise comparison on correction, harm and cost tests historical necessity. It does not test whether C1 was correctly implemented. Exact replay remains necessary for the **claim that a particular old computation was intervened on**, even if that claim ultimately proves unnecessary for useful segmentation.

### R3. Replace the solver by output rollback

With full hard rollback on mask `M`, net correct-pixel gain is exactly `|M∩R_true|-|M∩F_true|`. With oracle `M=R_true`, repair is complete and prior true FIX remains unchanged. Therefore Candidate C cannot distinguish itself merely by restoring old labels on correctly identified regressions.

Practical controls use the same predicted evidence: shipped gated rollback, protected soft rollback, and separately labelled hard/signed-evidence variants. C must demonstrate robustness to evidence error or another useful effect. Do not weaken rollback with a small gate and attribute the resulting gap to geometry without checking that explanation.

### R4. Replace test-time search by learned proposal generation

ActionSplice supplies a related idea: use matched alternative trajectories to train a residual transport mechanism, then avoid replaying completed computation at deployment. A future C study could compare a learned proposal or distillation against online search, under an explicit training-data and compute budget.

This is a conceptual control, not an implementation recommendation now. Distillation matching C would show that its online solver is dispensable at deployment; it would not erase the role of counterfactual replay in producing training targets. This distinction prevents an overly broad rejection claim.

### R5. Ask whether training exposure explains the benefit

Iterative mask-guided training in [RITM](https://arxiv.org/abs/2102.06583) provides a relevant counter-thesis to backpropagating refinement: a stronger training protocol can reduce the need for inference-time optimization. This is not a theorem that feedforward refinement always dominates.

Candidate C currently uses a detached solver, while optional predicted-history exposure changes the training examples and downstream loss evaluation points. Compare current/C crossed with exposure off/on before crediting geometry. LeCor's method further makes the response to a correction part of a training objective, so “learning to be correctable” is not an untouched novelty category.

### R6. Ask what the preservation constraint really preserves

Predicted-FIX class preservation is a finite output property. True-FIX protection depends on predicted coverage; causal safety does not follow. A spatial output-merge control tests whether the same benefit could be obtained by preserving those predictions directly.

FocalClick and error/confidence-guided medical refinement are relevant leads for this comparison. Their details must be kept distinct from Candidate C's numerical constraints and official re-audit. A gradient feasible-direction method would also need comparison with existing constrained-gradient principles such as GEM; renaming preservation as restitution does not create novelty.

## 4. Search leads retained without overstating closure

The workers collected the following additional primary leads. Root did not independently close every equation-level comparison before searching was stopped, so these do **not** justify a “no prior work” claim:

| Lead | Primary source | Remaining boundary |
|---|---|---|
| CGAM, ISBI 2023 | [Author-hosted paper](https://hvcl.korea.ac.kr/wp-content/papercite-data/pdf/ISBI23_paper_msh.pdf?_t=1680668180) | Worker-reported method read; root comparison of optimized attention variables and preservation remains incomplete |
| FocalClick, CVPR 2022 | [CVF paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Chen_FocalClick_Towards_Practical_Interactive_Image_Segmentation_CVPR_2022_paper.pdf) | Local refinement/progressive merge lead; exact equivalence to a C preservation control is not asserted |
| MECCA | [arXiv paper](https://arxiv.org/abs/2111.07716) | Action-confidence medical refinement lead; no unverified runtime rejection rule or venue asserted |
| GU-Net, 2024 | [Full text](https://pmc.ncbi.nlm.nih.gov/articles/PMC11415573/) | Counterfactual/attention medical segmentation lead; not identified as C-equivalent |
| SCORE, 2025 | [arXiv record](https://arxiv.org/abs/2511.02576) | Worker abstract-level lead; full-method comparison not closed |
| DAT, CVPR 2022 | [CVF paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Xia_Vision_Transformer_With_Deformable_Attention_CVPR_2022_paper.pdf) | Known sampling-substrate lead; this report does not redo the entire Dynamic Window matrix |

No new search is implied by retaining these gaps. They lower novelty confidence and identify the limits of the completed synthesis.

## 5. The strongest skeptical review

> The method belongs to the established family of frozen-network auxiliary-variable refinement under a target/proximity objective. Its distinctive historical support intervention is precisely defined, but necessity is unproven. Please isolate the historical reference, intervention variable, preservation rule, training exposure and rollout opportunity; compare against protected rollback and current-state refinement using the same evidence and measured resource budget. Explain which gains survive C3 and final audit. Numerical replay correctness and a different parameterization do not establish a paper-level contribution.

This is our constructed reviewer argument, not a quotation from a published review.

## 6. Claim boundary

The defensible object is a **bounded, replay-checked intervention on the realized support of a previously accepted ordinary transition, guided by predicted signed evidence and protected predicted FIX classes, followed by factual subtraction and official re-audit**. No source inspected here was identified as an exact instance of that full contract. This limited observation is not an exhaustive novelty certificate.

Reject claims of first dynamic/deformable/audit-conditioned attention, anatomical causal explanation, exact constrained optimum, guaranteed true-FIX preservation, or superiority from software checks. The decision to retain C as a headline contribution should depend on prespecified utility, harm and cost comparisons with uncertainty, as defined in [04](04_falsification_protocol.md) and [06](06_research_decision.md). Low feasibility alone is not mathematical impossibility, and overlapping confidence intervals alone are not evidence of equivalence.
