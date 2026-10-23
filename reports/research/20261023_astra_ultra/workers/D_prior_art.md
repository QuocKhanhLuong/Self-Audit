# Worker D — primary-source novelty audit

**Status: RESEARCH CANDIDATE; NOVELTY AND PATIENT EFFICACY NOT ESTABLISHED**
Access date: **2026-10-23**
Inspected source: **637f38ced217fe1db2347a87924695533056d8f0**
Worker D / astra_prior_art. Report-only scope; no production changes, patient/GPU jobs, weight/data downloads, or Git writes.

## Decision

The current method combines established ideas. Its strongest missing precedents are GEM, Orthogonal Gradient Descent (OGD), CPO/TRPO, and safe deep SSL (DS3L). Existing novelty documentation understates the curvature/last-layer overlap with Not All Unlabeled Data are Equal. The **HCO full PDF is now accessible**, resolving the old abstract-only limitation.

No reviewed source was found to contain the exact complete Self-Audit transaction. That bounded search result does not establish novelty. Additional pessimistic/projection and 2025–2026 calibration precedents are preserved in [D_safe_ssl.md](../evidence/D_safe_ssl.md). Showing precedent for the ingredients also does not prove that the combination is useless.

**One narrow candidate contribution, contingent on evidence:** a falsifiable characterization of when joint pseudo-class function-response allocation retains useful unlabeled learning at fixed dense-label and compute budgets beyond common guarded step shrinkage, extra labeled optimization, and established projection/reweighting methods, including numerical stalls and negative outcomes. This is a proposed characterization, not a demonstrated new optimizer or medical safety claim.

## Scope and source grounding

This is a targeted collision audit, not a systematic review or exhaustive novelty clearance. Primary publisher PDFs/pages, author project pages and author-submitted arXiv papers support the mechanism comparisons. Full text checked means the relevant equations/algorithm were inspected from accessible extracted text, not that the entire implementation was reproduced. Literature-reported outcomes are not local evidence. The bounded literature-review workflow has no PRISMA or exhaustive-screening claim.

Source versions, URLs, access levels, and search strings are recorded in [D_source_register.json](../evidence/D_source_register.json). Historical Astra no-GT work at f170dc1 is context only. Current v2 is explicitly semi-supervised and charges every audit mask to the dense TRAIN budget.

| Current contract | Source at inspected SHA |
|---|---|
| Supervised AdamW-body/SGD-head step creates theta_s; audit and supervised patients differ within the transaction. | [engine.py](../../../../self_audit/engine.py#L126), lines 126–154 |
| Fixed EMA pseudo targets; at most four class directions; shared accepted-pixel denominator. Summed directions equal a same-block plain pseudo SGD step. | [control.py](../../../../self_audit/control.py#L132), lines 132–159; [decoder_control.py](../../../../self_audit/decoder_control.py#L24), lines 24–46 |
| Three foreground one-v-rest half-Brier risks, including negatives and absent-class false positives. | [control.py](../../../../self_audit/control.py#L83), lines 83–129 |
| Joint box-QCQP with per-group quadratic budgets and coefficient ridge; numerical failure returns zero. | [control.py](../../../../self_audit/control.py#L22), lines 22–61 |
| Affine-logit/softmax probability derivative or decoder JVP; actual functional trials check every audit-group risk and mean risk. | [control.py](../../../../self_audit/control.py#L191), lines 191–228; [decoder_control.py](../../../../self_audit/decoder_control.py#L48), lines 48–82 |
| Accepted displacement commits directly, then EMA updates once. | [engine.py](../../../../self_audit/engine.py#L191), lines 191–192 |

This supports only a current-TRAIN, extra-pseudo-step acceptance contract. It does not protect the supervised step, cumulative trajectory, next audit patient, unseen-patient risk, EMA predictions or Dice.

## Claim-by-claim collision matrix

| Claim under investigation | Closest primary precedent and version | Match and remaining difference | Assessment |
|---|---|---|---|
| Use trusted labeled data to prevent harmful updates. | [GEM, NeurIPS 2017, full PDF §3](https://papers.neurips.cc/paper/2017/file/f87522788a2be2d171666752f97ddebb-Paper.pdf); [HCO, MICCAI 2026](https://papers.miccai.org/miccai-2026/paper/0945_paper.pdf) | GEM protects memory losses using gradient constraints; HCO privileges labeled directions. Self-Audit instead constrains foreground Brier after a separate supervised step within bounded pseudo coefficients. | **Broad claim occupied.** The selected risk/subspace must earn its value. |
| Small joint optimization handles competing updates. | [GEM](https://papers.neurips.cc/paper/2017/file/f87522788a2be2d171666752f97ddebb-Paper.pdf); [CAGrad, NeurIPS 2021](https://proceedings.neurips.cc/paper_files/paper/2021/hash/9d27fdf2477ffbff837d73ef7ae23db9-Abstract.html) | GEM has a small dual QP; CAGrad balances mean loss against worst local task improvement. Current coefficients scale existing pseudo directions rather than projecting freely in parameter space. | **Not a new joint-optimization principle.** |
| Function response offers more than loss-gradient alignment. | [OGD, AISTATS 2020, full PDF §§2–3](https://proceedings.mlr.press/v108/farajtabar20a/farajtabar20a.pdf) | OGD protects prior outputs through output derivatives, explicitly distinguishing these from loss derivatives. Current probability JVPs permit beneficial current-audit risk changes instead of preserving historical logits. | **Output-Jacobian protection is established.** A different measure is not automatically a contribution. |
| Labeled feedback determines unlabeled contribution. | [Ren et al., ICML 2018](https://proceedings.mlr.press/v80/ren18a.html); [Ren/Yeh/Schwing, NeurIPS 2020](https://proceedings.neurips.cc/paper_files/paper/2020/hash/f7ac67a9aa8d255282de7d11391e1b69-Abstract.html) | Clean-set meta gradients and influence-based unlabeled-example weighting already do this. Current weights operate on pseudo-class directions with group constraints and finite acceptance. Audit is TRAIN optimization data. | **Broad reweighting claim occupied.** Alpha is still a vector of scalar source weights. |
| Curvature and a last-layer path are novel distinctions. | [Not All Unlabeled Data are Equal, full text Eq. 6/§3.1](https://arxiv.org/html/2007.01293) | Its influence calculation uses inverse Hessian information; its practical approximation exactly inverts the last-layer Hessian. Current curvature is a response Gram matrix in candidate coordinates, not an influence inverse Hessian. | **Neither curvature nor last-layer tractability alone is new.** |
| Cross-source compensation is a new mathematical effect. | [GRAD-MATCH, ICML 2021, §3/Algorithm 2](https://proceedings.mlr.press/v139/killamsetty21a/killamsetty21a.pdf); [GLISTER, AAAI 2021; arXiv v4 2021-06-11](https://arxiv.org/html/2012.10630) | Weighted gradient residual fitting and subset evaluation through validation objectives already allow joint interactions. Current residuals are weighted probability responses with separate foreground budgets. | **Not a new algebraic phenomenon.** A toy example proves possibility, not real prevalence or utility. |
| A local constrained solve plus nonlinear checking creates a new safety architecture. | [CPO, ICML 2017, §6/Algorithm 1](https://proceedings.mlr.press/v70/achiam17a/achiam17a.pdf); [TRPO, ICML 2015](https://proceedings.mlr.press/v37/schulman15.pdf) | Both use approximate constrained proposals with line search; CPO explicitly backtracks to satisfy sampled constraints. Current checks use actual fixed-slice Brier rather than policy costs/KL. | **Acceptance architecture established.** Current source has no KL trust region, so do not call it an exact TRPO/CPO specialization. |
| Safe SSL compares favorably against a supervised reference. | [DS3L, ICML 2020, Eq. 3, Algorithm 1, Theorem 3](https://proceedings.mlr.press/v119/guo20i/guo20i.pdf) | Bilevel unlabeled weighting is optimized against labeled empirical risk, with a supervised-reference safeness theorem. Current source instead enforces a finite measured per-step predicate, without globally solving the bilevel problem. | **“First safe deep SSL” is untenable.** Neither result is automatically a test-Dice guarantee. |
| Labeled outcomes improve the teacher. | [Meta Pseudo Labels, arXiv v4 2021-03-01 / CVPR 2021](https://arxiv.org/abs/2003.10580) | MPL adapts its teacher using labeled student feedback. Current teacher is plain EMA; acceptance only changes which final student state reaches it. | **No teacher novelty.** Once-only EMA is state correctness. |
| Pseudo-label error control is itself new. | [ELN, CVPR 2022](https://cvlab.postech.ac.kr/research/ELN/); [BCP, CVPR 2023](https://openaccess.thecvf.com/content/CVPR2023/html/Bai_Bidirectional_Copy-Paste_for_Semi-Supervised_Medical_Image_Segmentation_CVPR_2023_paper.html) | ELN learns pseudo-pixel error masks; BCP mixes labeled/unlabeled regions and supervision under Mean Teacher. Self-Audit tests realized updates instead of these interventions. | **A useful distinction to test, not established superiority.** |
| Accepted updates cannot regress because of the guard. | Current source above | This follows from the acceptance predicate if measured evaluation and state are faithful. Budgets/tolerances allow changes; zero updates can make it vacuous. | **Finite observed-sample contract, not a new generalization theorem.** |

### HCO full-algorithm distinction

The [MICCAI publisher page](https://papers.miccai.org/miccai-2026/0458-Paper0945.html) identifies the September 2026 accepted version and lists code/supplement as unavailable. Algorithm 1 on PDF pages 3–4 first obtains a labeled bias-corrected AdamW direction; it masks unlabeled gradient coordinates by sign agreement, adds those masked gradients to both moment buffers, normalizes the mask by its mean, and applies half labeled and half gated-refined directions plus decay. Both gradients are evaluated at the pre-update parameters.

The repository's per-source mean-Brier sign test is therefore **not HCO**. A faithful comparator requires the published moment updates, normalization, split coefficients, decay and evaluation points. Numerical reproduction is **NOT RUN**. ([Algorithm 1](https://papers.miccai.org/miccai-2026/paper/0945_paper.pdf))

### DS3L theorem boundary

DS3L Theorem 3 compares labeled empirical risk to an argmin-defined supervised estimator. The supplementary proof relies on a feasible all-zero unlabeled-weight fallback, with exact inner/outer optimization. This is a theoretical optimization comparison; it cannot certify that arbitrary finite practical training iterates satisfy the inequality. In fact, if the supervised estimator globally minimizes exactly the same labeled objective over the same parameter class, no candidate can strictly improve that objective below its global minimum. Current source instead compares actual finite predictions to its actual finite supervised-step reference. That is a meaningful contract distinction, but not a first supervised-reference safety claim. The theorem's optimization assumptions and implementation gap must remain explicit. ([Theorem 3](https://proceedings.mlr.press/v119/guo20i/guo20i.pdf))

## Reduction of the current mathematics

This is a derivation from inspected code. Stack candidate displacements into D. With probability Jacobian J_c and audit weights W_c:

- V_c = sqrt(W_c) J_c D.
- A_c = D^T J_c^T W_c J_c D.
- b_c = D^T J_c^T W_c (y_c - p_c).
- E_c(alpha) = e_c - b_c^T alpha + alpha^T A_c alpha / 2.

A_c is the Gauss–Newton term restricted to the candidate subspace. The exact nonlinear Brier Hessian also contains residual-weighted second derivatives of probabilities. Thus response curvature is not an exact Hessian even for affine logits, because softmax is nonlinear.

Off-diagonal terms are ordinary joint least-squares interactions. Dropping them changes the problem. Their existence explains the compensation example without proving a new theorem. First-order audit effect, curvature, chosen update norm and nonlinear acceptance must be isolated experimentally.

Coefficient ridge and box bounds depend on column scaling and duplication. Canonical class IDs prevent one duplication failure; they do not establish general reparameterization invariance. Pseudo learning rate, common denominator and source definition are experimental factors.

Mechanism map:

    supervised reference -> fixed pseudo-class directions
                         -> joint response model -> actual TRAIN risk check
                         -> commit/reject -> EMA once
    GEM / reweighting / OGD inform the response-selection collision.
    CPO / TRPO inform the local-model-plus-finite-check collision.
    DS3L informs the supervised-reference safe-SSL collision.

This map does not assert algorithmic equivalence.

## Evidence required for the one narrow candidate contribution

1. **Liveness before efficacy.** Test moment-scale robustness against reference optima and objective gaps. Use KKT residuals only where suitable multipliers/constraint qualification exist; zero-budget degeneracies can lack them, as [B shows](B_theory.md#2-the-zero-budget-boundary-is-a-real-constraint-not-automatically-a-numerical-bug). Separate true zero optimum, no candidates, numerical failure, and finite-guard rejection. Coordinator-reported SLSQP liveness concerns require a separate verified receipt; worker D did not reproduce them.
2. **Common scalar plus same guard.** Add the direct “all current directions, one common coefficient, identical finite classwise backtracking” control. Current nine arms omit it. For the exact restriction alpha=a*1, the scalar ridge must be m*lambda because the joint penalty is lambda*m*a^2/2. Prespecify displacement-norm/trial-budget matching only as secondary same-state mechanism analysis; do not post-hoc constrain the primary policy effect.
3. **First-order joint plus same guard.** Use identical audit data, coefficient domain and state timeline without response curvature. Current unguarded sign is insufficient to isolate curvature or finite checking.
4. **Direct labeled optimization.** A same-audit-gradient update with the same block and actual guard tests whether candidate allocation is merely an expensive route to more labeled descent. Account for audit reads and match compute, not just optimizer steps.
5. **Faithful prior-art controls.** Run faithful HCO as a total-recipe comparator: preserve its pre-update labeled/unlabeled gradients, moment carryover, split coefficients and decay, which cannot simultaneously equal the Self-Audit supervised-reference timeline. Name any timeline-matched variant HCO-inspired, not HCO. Separately require a clearly labeled GEM-style projection/reweighting adaptation with matched protected labels, targets, block and compute. Do not compare incompatible published numbers.
6. **Real compensation prevalence.** On TRAIN-only development traces, quantify jointly useful directions that independent allocation rejects, off-diagonal contributions, local/actual mismatch, norms and accepted learning. A single manufactured example is insufficient.
7. **Frozen unseen-patient endpoints.** Use dense raw class Dice, precision/recall, false positives and patient-level uncertainty over label-subset/training seeds. Current audit risk is not confirmation. Completed baseline arms do not establish the complete guard.
8. **Practical opportunity cost.** Retain same-block/full EMA, extra supervised compute and a suitable strong SSL/BCP comparator. Count JVPs, trial forwards, teacher calls, memory and time. Report rejection, absent-positive groups, teacher drift and label-representativeness failures.

A useful result may be negative: compensation exists algebraically but is rare, fails nonlinear transfer, is matched by scalar shrinkage or labeled replay, or costs more than its unseen-patient benefit. Such a characterization requires matched evidence too.

## Claim limits and kill conditions

Retire the candidate if its effect disappears after scalar/norm/compute matching, only reused-TRAIN risk improves, unlabeled progress is negligible, or faithful prior methods match it at the same budget. Do not rescue it by adding networks after inspecting confirmation outcomes.

Do not claim first safe SSL, first output/gradient protection, novel least-squares interaction, a new teacher, clinical safety, patient-Dice protection, or no-GT learning. The current conclusion is **CONTINUE ONLY AS A BOUNDED FALSIFICATION STUDY**.

## Access limits and execution status

Relevant full-text mechanisms were inspected for GEM, OGD, Ren 2018, Ren/Yeh/Schwing 2020, GRAD-MATCH, GLISTER, HCO, CPO, TRPO and DS3L. A-GEM OpenReview hit a browser challenge; author arXiv HTML was accessible and its publication version verified. MPL, ELN, BCP and CAGrad are used for high-level distinctions only; they were not implementation-reviewed. Two CVF fetches returned 403 and were replaced by author sources. HCO screenshot rendering failed; evidence is extracted PDF text, not visual inspection.

No external implementation, numerical parity, patient training or GPU execution was performed by worker D. The report does not inherit old no-GT or synthetic readiness claims.

## Independent review response

E reviewed the stable draft and requested three corrections, all accepted: the aggregate scalar restriction uses m*lambda; KKT diagnostics require suitable multipliers/qualification; faithful HCO is a total-recipe comparator, while a matched-timeline adaptation must be named HCO-inspired. Norm matching is prespecified secondary same-state mechanism analysis, not post-hoc alteration of the policy effect. These corrections are incorporated in the evidence requirements above.
