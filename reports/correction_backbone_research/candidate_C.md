# Candidate C — high-risk replayable accepted correction events

Decision: reject its standalone novelty rationale and defer implementation. This architecture integrates **CC-restitution**, the existing Candidate C coordinate-replay mechanism, with explicit latent events. It is not justified until simpler coordinate-only restitution, snapshot replacement and Candidate B succeed.

## 1. Exact state variables

Retain stable V, L_t, B's class-bound P/R, and a bounded additive correction accumulator `C_t = C_0 + sum_j delta_j` over accepted events in the current slice. Store only the most recent accepted **ordinary** event as replay-eligible: `(S_before, L_before, P_factual, delta_factual, residual_factual, gate_factual, e_before, e_after, weights/version, dtype, normalization mode, turn/depth, factual batch identity)`. Earlier events may remain accumulated but cannot be individually edited. No recursive replay and no cross-patient memory.

## 2. Equations: difference of the same action

Let the ordinary event writer and decoder be a frozen functional map

\[
(\delta(P),r(P))=\mathcal F_\theta(V,S_{t-1},L_{t-1},e_{t-2},P),\quad
C_t=C_{t-1}+\delta(P_t),\quad L_t=L_{t-1}+r(P_t)
\]

only after the ordinary transition has been accepted. The initial replay requirement is `(delta(P_t), r(P_t))` matching recorded factual values. Changing only bounded support P gives

\[
\delta^{cf}=\delta(P_t^{cf})-\delta(P_t),\quad r^{cf}=r(P_t^{cf})-r(P_t),
\]
\[
\boxed{\tilde C=C_t+\beta\delta^{cf},\qquad \tilde L=L_t+\beta r^{cf}.}\tag{C1}
\]

This is **replacement of an attributable event contribution**, not adding another full historical correction. `beta∈[0,1]` is fixed/calibrated before external testing. The support optimization targets predicted REGRESS while protecting predicted prior FIX in **output space**, and bounds realized support displacement, state delta norm, and logit delta norm. A small latent difference need not produce a small mask change; output feasibility checks remain mandatory. The Auditor judges `(L_t,Ltilde)` after the transfer; only acceptance commits both C/L and new event evidence.

For beta=1, C1 recovers the counterfactual **additive accumulator** of that immediate event, provided there were no intervening accepted actions and no post-write nonlinear transformation of C. Beta<1 is an interpolated intervention, not exact historical replay. This does not reconstruct the complete alternative history. Keep P/R metadata outside the additive accumulator and apply B1 to the **actual** current-to-restituted transition using current P/R and its new audit. The real accepted history contains the ordinary event followed by a restitution event; the latter contributes `beta*delta_cf` and `beta*r_cf`. Its class ledger can therefore differ from the hypothetical history in which the counterfactual support was used originally. Never write a historical pre-to-counterfactual ledger using current-to-restituted evidence: they are different transitions. A full alternative-history ledger would require a second historical audit and separate experiment, excluded here.

The candidate decoder must remain event-based: `r(P)` is a residual decoded from the same recorded pre-state, not `D(C_t + delta_cf)-D(C_t)` asserted equal to historical replay. If a nonlinear persistent decoder is used instead, candidate logits must be explicitly decoded and audited; no algebraic equivalence is promised. This restriction is the price of making the event attributable.

## 3. Visual encoder

Pretrained ConvNeXt with cached V first. Small scratch CNN/SSL controls follow only after utility evidence. Support changes sample fixed V; no weight updates, image edits or encoder reruns occur inside restitution. Coordinates have physical/grid-unit definitions and the same interpolation/clamping semantics as factual replay.

## 4. Correction updater

A small sparse event writer samples fixed V and the recorded pre-state at P, combines these with annotation/audit features and scatters an additive event delta. Keep per-event state increments bounded and avoid in-place normalization of accumulated C. Spatial support controls where evidence is sampled, not necessarily the final affected pixels; convolution/attention receptive fields can spread its effect. Log actual output influence.

## 5. Decoder

Decode each ordinary action's bounded logit residual with stable fine visual skips; keep its factual residual and optional gate in the replay record. The A0 head is independently strong. For restitution, transfer the counterfactual-minus-factual residual in C1 and audit the resulting current annotation. Future ordinary decoding reads the updated C plus actual retained logits. Evaluate whether this extra persistent state helps beyond logit-only C1.

## 6. Auditor interaction

Official global gate and local predicted transition evidence are detached. The inner solver may use predicted regression/FIX evidence from the recorded action but cannot optimize through a changing Auditor. It is not allowed to repeatedly query and maximize the official quality score under unreported search. Use the same bound of one coordinate-gradient evaluation and at most two **feasibility-only replay checks** as the existing CC control for the first comparison. These two checks evaluate replay logits against already recorded FIX classes and support/output bounds; they do not call the official Auditor. After selecting the feasible candidate, exactly one official current-transition audit is used. An ordinary fallback also receives its one final official audit. Any extension that audits additional alternatives must add those calls explicitly to the budget and is not this first comparison.

## 7. Commit/reject behavior

Atomic acceptance writes C/L/P/R and clears ordinary replay eligibility for the next action, matching current CC's nonrecursive policy. Rejection preserves the full retained tuple and HALTs. Numeric/stale replay failure goes to a declared ordinary current-state proposal plus official audit; do not disguise failure as successful restitution. A feasible identity proposal is reported as zero innovation, not a repair. Under the current three-turn cap only a small number of such state effects can ever be observed.

## 8. CC-restitution compatibility and rollback distinction

This is its strongest integration. Direct rollback proposes a step toward the earlier L/C. C1 instead reruns the **same accepted action with changed support**, subtracts its factual contribution, and replaces only that contribution. The distinction is technically meaningful when action identity is reproducible, output constraints hold and future correction improves. It is not automatically novel: hidden-state editing, gated memory and selective mask updates already exist, and current Self-Audit already does output-space CC transfer.

Editing an older event after later nonlinear actions is invalid without recomputing all descendants; this candidate expressly excludes that extension. If attribution requires expensive descendant replay or a long event log, abandon it under the current small-data/three-turn setting. Coordinate-only restitution should remain separate unless latent C1 beats it.

**Red-team reduction:** for any deterministic recurrent map G and the immediately preceding action, if `C_t=G(S_before,P)`, then `C_t+[G(S_before,Pcf)-G(S_before,P)]=G(S_before,Pcf)`. Thus an ordinary saved pre-state plus replacement can achieve the same immediate accumulator intervention; additive event architecture is not necessary. The full audit/ledger tuple still follows the actual-history distinction above. A/B snapshot replacement under identical constraints, weights and evaluation cost is mandatory. This reduction rejects C's standalone architectural argument, even if support-constrained restitution itself proves useful.

## 9. Parameter estimate

Planning totals including the [training-strategy reserves](training_strategy.md#7-analytical-total-system-51015m-envelopes) are approximately 4.65–5.05M / 10.17–10.57M / 15.38–15.78M for the compact reference family, depending on event heads; trim shared width to fit a strict cap. Solver/replay has no learned parameters. ConvNeXt-hosted C retains the large baseline encoder. Runtime state snapshots, factual support and autograd storage are additional **memory**, not parameters; cite neither as zero-cost.

## 10. Compute per refinement turn

Ordinary turns are near B's 1.3/2.9/4.0 GMAC plus sparse sampling/scatter cost. A restitution attempt substitutes up to three historical writer/decoder forwards and one coordinate backward for one ordinary writer/decoder forward, plus one official audit. State exact work as `3*(writer+decoder) + backward_coordinates + Auditor`, not three full image encodings. A stale/factual failure followed by ordinary fallback costs both attempted replay and fallback. Factual batch replay can process more rows than survive. No latency, peak VRAM or fixed backward/forward ratio has been measured; CC training with paired utility adds further graphs.

## 11. Training schedule

Train ordinary corrections and early audit first. Freeze proposal weights for each factual/counterfactual pair; never reuse a replay record after an optimizer step. Introduce bounded CC examples using predicted evidence, not GT-selected inference supports. Train state usefulness with the same candidate annotation in latent-transfer and no-latent-transfer arms. Move to deployment-matched gates; do not enable free-running bad early Auditor rejection. Keep solver work, A0 losses and gate calibration matched to ConvNeXt + CC.

## 12. Novelty risk

Very high implementation and compute risk; uncertain novelty. The potentially distinct relationship is an attributable accepted event, output-constrained counterfactual replacement, and improved future repair after that replacement. Additive storage and state subtraction alone are bookkeeping. If event linearity limits segmentation quality, or future gains occur only from the extra decoder searches, the architecture should be rejected. Evidence does not currently justify making C a core paper contribution.

## 13. Five closest methods

| Method | Mechanism threatening C | Required separation |
|---|---|---|
| [Current Self-Audit CC-restitution](../candidate_c/final_integration_review.md#5-implemented-c1c2c3) | Exact accepted-action support replay and output innovation already implemented. | Additional latent transfer must improve future corrections at the same current mask. |
| [SENTRY, 2026 preprint](https://arxiv.org/html/2606.24449v1) | Backtracking hypotheses and validating future memory writes. | Counterfactual support replacement of one accepted same-image action. |
| [SAM2Long, ICCV 2025](https://openaccess.thecvf.com/content/ICCV2025/html/Ding_SAM2Long_Enhancing_SAM_2_for_Long_Video_Segmentation_with_a_ICCV_2025_paper.html) | Alternative memory trajectories and quality-based branch retention. | Fixed bounded one-event intervention, not search-tree capacity. |
| [Correction-aware interactive 3D tumor segmentation, 2026](https://link.springer.com/article/10.1007/s00371-026-04560-5) | Revision-aware correction memory. | Learned autonomous accepted-action restitution, without fresh human prompts. |
| [Subtract or Replay? Exact Deletion from Language-Model Memory, 2026 preprint](https://arxiv.org/html/2607.27539v1) | Record-local decrement, suffix-dependent state contributions and corrected-record replay. | Segmentation-specific output constraints and accepted-FIX preservation with same-mask future utility; subtraction/replay algebra is not the contribution. |

These are mechanism comparisons, not claims that the papers implement C1. Cross-domain state rollback precedents are in [novelty_falsification.md](novelty_falsification.md).

## 14. Experiments that falsify C

Require all of: factual identity under eval/AMP/batch changes; bounded support and output constraints; lower destruction than direct rollback; benefit over random support search with equal evaluations; benefit over coordinate-only CC on ConvNeXt; positive same-mask future utility of the latent transfer; no dependence on GT at inference; and acceptable measured latency/VRAM. Track replay failure, fallback and identity-proposal rates with explicit denominators. If C's extra state has no future benefit at T=3, do not extend the horizon post hoc to rescue the claim. A longer horizon is a separately preregistered experiment with matched controls.
