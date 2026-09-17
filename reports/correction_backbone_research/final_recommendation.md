# Final recommendation: keep ConvNeXt; test accepted-history usefulness first

Research date: 2026-09-11. Inspected source: `main` / refreshed `origin/main` at `35aaba9335e0d8a0aa344356e20e93ae902b950d`. This is a literature/source-based decision and proposed experimental protocol. There are no new trained results, clinical findings, measured GPU costs or established novelty claims in this study.

**Do not replace ConvNeXt now.** The strongest next experiment is Candidate B's narrowly defined accepted-transition history on the current pretrained visual encoder, with Candidate A as the recurrent control. Defer architecture C. The visual backbone question and the correction-state mechanism question must be tested independently.

## 1. Is replacing ConvNeXt scientifically justified?

No present evidence justifies it. Current Self-Audit already caches visual features and conditions its expert on annotation/audit state. Replacing the CNN for the same heads would be low-novelty engineering. A smaller model could be valuable if it improves the accuracy/cost frontier, but that is a result to measure, not a consequence of the architecture sketch. Keep pretrained ConvNeXt + current Self-Audit, + early auditing, and + current CC-restitution as mandatory controls.

## 2. Is training from scratch realistic for ACDC/M&Ms?

A compact roughly 5M system is a reasonable **pilot**, roughly 10M is plausible with regularization and sufficient patient-disjoint evidence, and roughly 15M is a higher-risk capacity control. None is known to work better. ACDC provides few independent patients relative to correlated slices; do not justify a random ~28M encoder by slice count. The [training strategy](training_strategy.md#2-data-scale-and-initialization-controls) documents primary dataset sources, initialization controls and patient splits.

Use the ordinary small residual visual CNN from scratch first, then train-only medical SSL with identical encoder architecture. Keep a strong ImageNet-pretrained ConvNeXt baseline and do not reduce its A0 supervision. For ACDC-to-M&Ms, all M&Ms exposure remains external evaluation only. A separately labeled native M&Ms experiment can train on its own declared training partition; it answers a different question.

## 3. Should visual evidence be stable while correction state evolves?

Yes as a sensible decomposition to test, and stable V is already part of the baseline. Keep image evidence numerically unchanged during a rollout, with fine spatial skips available to corrections. This need not freeze encoder weights during training. Compare permanent encoder freezing, normal annotation-gradient training, and repeated full mask-conditioned encoding as distinct controls. Cached V plus a small state core shortens recurrent compute; it does not by itself prove better representations or a saving over current Self-Audit.

## 4. Is transactional state publishably distinct from recurrent refinement?

Not by itself. `(1-b)S+bW` is gated recurrence. Under first-rejection HALT, discarding a rejected latent proposal has no observable future output effect. Moreover, [SENTRY](https://arxiv.org/html/2606.24449v1) verifies candidate predictions before their use in future memory; [SAM2Long](https://openaccess.thecvf.com/content/ICCV2025/html/Ding_SAM2Long_Enhancing_SAM_2_for_Long_Video_Segmentation_with_a_ICCV_2025_paper.html) maintains scored memory pathways. We must reject “verification before memory write” as a general novelty claim.

The possible remaining contribution is a **task-specific state representation plus objective** binding accepted correction consequences to future repair and preservation. It remains unproven and could reduce to an objective or explicit-map control rather than a new neural architecture.

## 5. Which of A/B/C is strongest?

| Architecture | Decision | Reason |
|---|---|---|
| [A: recurrent correction core](candidate_A.md) | Retain as mandatory control/engineering alternative | Low complexity; no architectural novelty claim. |
| [B: accepted class-bound repair/preservation](candidate_B.md) | Strongest research hypothesis; start with ConvNeXt V | Explicit transition consequences support falsifiable semantics and future-usefulness tests. |
| [C: replayable accepted events](candidate_C.md) | Reject standalone novelty; defer implementation | Immediate snapshot replacement in an ordinary RNN gives the same state intervention; coordinate-only restitution may suffice. |

B's advantage is experimental clarity, not demonstrated accuracy. The cheapest informative control is actually P/R maps with the **current expert and no extra latent recurrence**. If that matches B, simplify immediately. This is an ablation of B, not a fourth proposed architecture.

## 6. What exact equation is the proposed contribution?

Let f/r be detached confidence-weighted FIX/REGRESS events for the transition just audited: evidence must pass its development-calibrated threshold, win the local three-class argmax, and accompany a changed predicted label; otherwise its event weight is zero. With old/candidate predicted one-hot classes Yold/Ynew, B proposes the class-bound write

\[
\tilde P^k=(1-r)(1-f)P^k+fYnew^k,\qquad
\tilde R^k=\operatorname{clip}_{[0,1]}((1-fYnew^k)R^k+rYold^k),
\]
\[
(S_{t+1},L_{t+1})=
\begin{cases}
(W_\theta(S_t,\tilde C,\tilde P,\tilde R),\tilde L),&q_t>\tau,\\
(S_t,L_t),&q_t\le\tau\quad\text{and HALT}.
\end{cases}
\]

The hypothesis needs a second, essential condition:

\[
\boxed{J_Y(\mathcal T_K(V,L_{t+1},S_{t+1}))
>J_Y(\mathcal T_K(V,L_{t+1},\operatorname{NullWrite}(S_t)))}
\]

with the same retained logits, same current audit and policy, lower previous-FIX destruction, and superiority over parameter-matched generic memory/direct correction-memory adaptations. The full causal timing, losses, no-op lemma and limitations are in [state_formulation.md](state_formulation.md). These elementary write equations are not asserted to be mathematically novel. Their potential paper contribution is the tested link between **accepted class-bound consequences, persistent state and safer future correction**. If that link fails, reject the claim.

## 7. What closest prior art threatens it?

The strongest combination is [Li and Li's correction-aware medical revision memory](https://link.springer.com/article/10.1007/s00371-026-04560-5), SENTRY's verified memory writes, and FocalClick's selective preservation. MAIS threatens medical correction-history memory; SAM2Long/SAMURAI threaten quality-gated memory; RMR/ICBNet/Mask2Former/RITM threaten mask-driven recurrent features. StreamDAM further weakens broad state-update claims; DREAM is a provisional error-memory lead with access/venue uncertainty. [Subtract or Replay?](https://arxiv.org/html/2607.27539v1) directly threatens C's record-local subtraction and replay algebra. The [prior-art matrix](prior_art_matrix.md) gives mechanism-level distinctions and source confidence; the [post-proposal red team](novelty_falsification.md) tests actual candidate mechanisms and records which claims were rejected or narrowed.

Positive/negative human prompts are not identical to predicted FIX/REGRESS transition events, and video-time is not same-image refinement-time. Those distinctions motivate an experiment; they do not establish novelty by changing task names.

## 8. How should early auditing be trained?

Start the Auditor in shadow mode while preserving A0 training. Use measured training-transition GT only as supervision, including real model proposals and bounded synthetic mixed fix/regression examples. Next let **detached predicted** evidence condition correction without hard rejection. Train zero-history and predicted-history cases so the core does not depend on perfect evidence. Use hard gating/HALT only after held-out transition-quality checks and calibration; retain an ungated training-only pair source to avoid starving the Auditor.

Skip soft commits by default. If studied, jointly blend annotation/state, re-audit the actual blend before writing its local evidence, count the extra call, and label train/deployment mismatch. Do not write an audit of one candidate into a different interpolated transition. Compare early auditing on the existing architecture first; curriculum gains cannot be attributed to a new backbone.

## 9. How should CC-restitution interact with the architecture?

Remain a coordinate-only mechanism on the current encoder/expert first. With B, use predicted P/R as explicit history context and preserve the existing action-replay identity checks and output constraints. Architecture C may transfer both a bounded latent event difference and its corresponding logit-residual difference **from the same historical action**, but only if this improves future correction at identical current logits over mask-only CC.

No claim of complete rollback/counterfactual policy reconstruction is allowed for event subtraction. Limit deployment to the most recent ordinary accepted action; clear replay eligibility after restitution. Editing older actions would require descendant recomputation, outside this study's proposed first experiment.

## 10. What results would make us abandon the new backbone?

Any of the following suffices to abandon the corresponding architectural claim:

- ConvNeXt + early auditing or current CC matches the new system at a matched operating point.
- A no-state expert or explicit P/R maps without latent C matches B under the same objective, A0, gate and compute.
- Same-mask state deletion/shuffle has no reproducible future cost, or FIX/REGRESS semantic permutations are indistinguishable from correct evidence.
- Apparent gain comes from lower A0, extra proposal evaluations, accepting fewer useful turns, GT input, target-domain tuning, or parameter budget differences.
- C fails factual identity/output constraints or adds latency/VRAM without a benefit over coordinate-only restitution.
- A compact replacement fails preregistered final-quality noninferiority or increases harmful accepted transitions across patients; reject it even if it has attractive parameter count.

An imprecise confidence interval means inconclusive, not success. The [experiment plan](experiment_plan.md) defines denominators, uncertainty, matched controls and proposed numerical stop gates before measurement.

## 11. Core contribution, secondary contribution, or engineering alternative?

**Now: B is a secondary research hypothesis; A is an engineering alternative; C is deferred. Keep the current ConvNeXt system.** Promote B to a core contribution only after independent same-annotation state-usefulness evidence, semantic perturbation sensitivity, reduced accepted-FIX destruction, and direct prior-method controls survive on ACDC and frozen external M&Ms. A generic encoder replacement should remain an engineering result even if faster.

## Scores

Subjective research judgments on a 1–10 scale, not empirical measurements or probabilities. Higher is better for novelty, soundness, efficiency and fit. **Higher means worse for implementation/compute risk.** “Current system” means the established Self-Audit design, not a claim that the ConvNeXt backbone itself is novel; existing code verification does not establish efficacy.

| CURRENT CONVNEXT SYSTEM | Score | Basis |
|---|---:|---|
| Novelty | 5/10 | Transition audit/restitution may support a paper; generic encoder adds none; literature differentiation still needed. |
| Technical soundness | 8/10 | Coherent retained-annotation policy and mature source contracts; learned Auditor reliability and real-data effects unresolved. |
| Data efficiency | 7/10 | Pretrained visual evidence is a credible small-data control; not measured against new models. |
| Self-Audit fit | 8/10 | Cached evidence, predicted audit feedback, explicit retention and bounded replay already align well. |

| RECOMMENDED NEW ARCHITECTURE: B on stable ConvNeXt V first | Score | Basis |
|---|---:|---|
| Novelty confidence | 3/10 | Strong nearby revision/error/verified-memory methods; only narrow accepted-class future-utility relation remains open. |
| Technical soundness | 6/10 | Causal ordering is specified; predicted obligations can be wrong or persist incorrectly. |
| Self-Audit fit | 9/10 | Meaning depends on transition FIX/REGRESS and accepted history. |
| Data efficiency | 6/10 | Modest core, but sparse useful histories and extra supervision may hurt small-data learning. |
| Implementation risk | 6/10 | Row-atomic semantic writes, paired probes and history provenance require care. |
| Compute risk | 4/10 | Small ordinary core; training forks cost extra. Architecture C would score higher risk. |

## Deliverables and review provenance

Read [current architecture](current_architecture.md), [prior art](prior_art_matrix.md), [novelty falsification](novelty_falsification.md), [state formulation](state_formulation.md), [A](candidate_A.md), [B](candidate_B.md), [C](candidate_C.md), [training strategy](training_strategy.md), and [experiment plan](experiment_plan.md) alongside this decision. Estimates and proposed experimental thresholds are labeled throughout.

Orca run: `run_ce5fe3fa3a80`. Independent literature task `task_d9691916c8f8` / dispatch `ctx_6cabb3ce54d4`; training/evaluation task `task_ad3415678e95` / `ctx_5101240d26e2`; post-proposal red-team task `task_1dd2d3db0f50` / `ctx_9e5a44afbe5e`. The coordinator inspected current source, formulated candidates, reconciled research findings and checked the combined report set. Worker model launch arguments inherited configured defaults; runtime fleet observations identified `gpt-6-astra`. No non-Orca subagents were used.

| Research task | Accepted outcome and evidence | Remaining limitation |
|---|---|---|
| Literature audit | Succeeded: 39 comparison rows, 40 primary-source entries, access grades and recorded searches in prior_art_matrix.md | ICBNet full methods and DREAM full writer/venue remain unverified; bounded search cannot prove absence. |
| Training/evaluation design | Succeeded: both owned reports reconciled to B1/C1, required diagnostics and stronger matched controls; parameter/MAC arithmetic independently reproduced | Every training, profiling and empirical decision gate remains NOT RUN. |
| Post-proposal red team | Succeeded: 36 mechanism-specific queries, primary counterexamples, conservative verdicts and corrected specification issues | No publishable-novelty or usefulness claim is established. Reviewed A/B/C/state SHA-256 values match the delivered files. |

All expected Dispatches settled. The literature and red-team terminals were released; Orca retained the protocol terminal under its recorded `user_takeover` ownership. No reclaimable worker remained. Final documentation checks verified the exact ten-file set, local links and anchors, source-reference definitions, balanced code/math delimiters, and the reviewed candidate hashes. The coordinator additionally inserted direct Mask2Former, MAIS, context-feedback, quality-gated recurrence and RPCM adapters into the final experiment matrix. Source HEAD was refreshed again and remained unchanged; production/configuration/test files were untouched. No production test suite was run for this documentation-only study.
