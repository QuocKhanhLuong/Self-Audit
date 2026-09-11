# Candidate B — accepted class-bound repair and preservation state

Decision: strongest **bounded research hypothesis**, initially on the existing ConvNeXt encoder. Broad dual-memory or transactional novelty is rejected. No claim of publishable distinctness is established yet.

## 1. Exact state variables

Retain `L_t`, cached `V`, previous accepted `e_{t-1}` and active bit. Split d correction channels into `C_t^repair` and `C_t^preserve`. Also retain stride-4 class-indexed confidence maps `R_t^k` and `P_t^k`, k∈{BG,RV,MYO,LV}, plus event age/identifier. P stores the **candidate class established by an accepted predicted FIX**; R stores the **previous class lost in an accepted predicted REGRESS**. They are not generic foreground/background maps. All begin zero. Store full-resolution class/event metadata for evaluation/replay; downsampling uses class-wise area weights, not interpolation of integer class IDs.

The memory is deliberately bounded to the current slice and the rollout cap. R is not a full error map: preexisting A0 errors and wrong-to-wrong changes remain visible through V/mask/entropy but may not appear as obligations. A global acceptance can contain local REGRESS, so accepted R is not contradictory. **Metadata contract:** for each class/map/location, retain the identifier of the most recent accepted event that changed that map value and its age; this is last-touch metadata, not the origin of every component of a soft accumulated value. Older contributions are compressed by B1 and cannot be individually reconstructed or replayed. B claims class-bound accepted-history summaries, not an event-resolved causal ledger; C requires its separate factual action record.

## 2. Equations and semantic write

First create a tentative correction using the previous ledger:

\[
\tilde C^{R}_{t+1}=C^R_t+F_R(V,a_t,C^R_t,\operatorname{Read}_R(R_t),e_{t-1}),\quad
\tilde C^{P}_{t+1}=C^P_t+F_P(V,a_t,C^P_t,\operatorname{Read}_P(P_t),e_{t-1}).
\]

Decode the candidate using both branches; then predict the transition audit. Let `chi_F = 1[argmax(e_t)=FIX and e_t^FIX >= kappa_F and yold != ynew]` and define `chi_R` analogously for REGRESS. Resolve local audit ties as no-event; resolve equal class logits using the fixed class-index argmax convention shared with the baseline. Use confidence-weighted events `f_t=e_t^FIX*chi_F`, `r_t=e_t^REGRESS*chi_R`; thresholds are calibrated on development data only. Let `Yold/Ynew` be one-hot old/candidate predicted classes. One explicit, deliberately simple research writer is

\[
\tilde P^{k}_{t+1}=(1-r_t)(1-f_t)P_t^k+f_tYnew^k,
\]
\[
\tilde R^{k}_{t+1}=\operatorname{clip}_{[0,1]}
\big((1-f_tYnew^k)R_t^k+r_tYold^k\big).\tag{B1}
\]

Evidence, class masks and event identities are detached; values are class-wise downsampled before state reads. The first equation weakens an old preservation claim when later accepted evidence reports regression and replaces it when a new FIX establishes another class. The second clears a repair obligation only when predicted FIX restores its bound class; it accumulates a newly lost pre-class. If no qualified event occurs, f=r=0 gives P'=P and R'=R exactly. Unthresholded soft evidence is an ablation because it erodes memory on predicted no-change steps. No learned writer may silently swap FIX/REGRESS roles. False class obligations can persist; use finite rollout lifetime and test calibrated confidence/expiry, with those knobs selected only on development data.

After audit, persistent branch features are written using **different evidence-dependent payloads**:

\[
W_R=C^R+n_t[\tilde C^R-C^R+\Psi_R(V,a_t,\tilde a,\tilde R-R)],\quad
W_P=C^P+n_t[\tilde C^P-C^P+\Psi_P(V,a_t,\tilde a,\tilde P-P)].
\]

Here `n_t=1[any qualified FIX or REGRESS event in the row]`. Each Psi is multiplied by its class-ledger delta so `Psi(...,0)=0`; when n=0, both persistent C branches and P/R are unchanged even if tentative decoding used nonzero scratch features. Commit `(W_R,W_P,Ptilde,Rtilde,Ltilde,e_t)` only if `q_t>tau`; otherwise keep S/L and HALT. The annotation/audit may still change on an accepted confidence-only step; the no-event identity applies to correction memory, not the entire annotation tuple. W is not used to change the mask already audited. A fresh next-turn proposal consumes W and undergoes its own audit. The proposed contribution is **B1 constrained by same-mask future utility equation (2) in state_formulation**, not B1's elementary arithmetic alone.

**Illustrative arithmetic, not data:** after an accepted predicted RV FIX with f=0.8, zero-initialized `P^RV=0.8`. A later accepted RV-to-LV transition with r=0.7 changes it to 0.24 and records `R^RV=0.7`. A later predicted FIX back to RV with f=0.9 reduces that obligation to 0.07 and sets preservation to 0.924. An equally confident FIX to another class does not clear the RV obligation under B1. That last case can be correct behavior or a false stale obligation, depending on whether the earlier Auditor was right; it is a required failure analysis, not a truth guarantee. A no-event accepted step keeps these memory values unchanged.

## 3. Visual encoder

Use pretrained ConvNeXt and current cached FPN for the mechanism-first experiment. Stable V is unchanged within each rollout; encoder weights may train with annotation supervision. Only after B has independent evidence, replace V with the 5/10/15M regime CNNs. Random initialization, train-only medical SSL and pretrained controls must not be conflated.

## 4. Correction updater

Two small residual convolutional readers at stride 8, d/2 channels each, with a shared visual projection. Repair reads class-bound R and can suggest changes; preservation reads class-bound P and predicts a soft protection penalty/gate. Cross-talk is permitted through the decoder but the write interfaces remain explicit and testable. Equal-size unstructured two-branch and single-GRU controls isolate whether semantic structure, rather than added capacity, matters.

## 5. Decoder

Stride-4 V skip, projected logits, C^R and C^P produce a bounded residual. A soft preservation cost discourages departing from committed classes on high-confidence P. Do **not** absolutely lock predicted FIX: an erroneous Auditor could otherwise make mistakes irreversible. A0 has independent full supervision and does not read invented history. Log any gain from stronger segmentation loss separately from state gains.

## 6. Auditor interaction

Detached transition-level local evidence drives B1 after proposal. The official global gate initially stays unchanged for fair comparison; a calibrated alternative gate is its own ablation. GT supplies training targets/diagnostics, never the deployed state or decisions. Train an auxiliary same-annotation future-utility objective and past-FIX destruction penalty; generic Dice/CE and a utility classifier alone do not establish useful semantic state.

## 7. Commit/reject behavior

One acceptance event writes all tensors, class bindings, age and annotation atomically. Rejection preserves them and HALTs. The broad transactional rule is inherited from equation (1) and receives no independent novelty credit. Test row isolation and accepted-state persistence; the observationally inert rejection storage control should tie.

## 8. Compatibility with CC-restitution

Keep CC-restitution's coordinate-only intervention as the first branch. P supplies a predicted preservation region, R a repair region, but the replay must still use the original action's recorded pre-state and official current-state audit. Do not replace strict existing FIX constraints with an opaque gate and call it equivalent. After an accepted restitution, re-audit/update current P/R according to the actual current-to-restituted transition; clear the ordinary replay record as in main. Compare with current ConvNeXt + CC before considering latent transfer. B does not require CC to exist.

## 9. Parameter estimate

Illustrative compact planning totals about 4.45M / 9.97M / 15.18M, including A0, updater, class readers, decoder, Auditor and the omitted-term reserves in [training_strategy.md](training_strategy.md#7-analytical-total-system-51015m-envelopes). The widest reference exceeds a strict 15M cap and requires shared width reduction. Additional class-map storage is small (eight stride-4 FP16 maps ≈0.0625 MiB), excluding IDs and full-resolution diagnostics. B on pretrained ConvNeXt is larger and must be labeled by its actual eventual count. Capacity numbers are analytical allowances, not implemented counts.

## 10. Compute per refinement turn

Reference A budget plus approximately 0.2 / 0.5 / 0.7 GMAC for dense small semantic readers/writers: about 1.3 / 2.9 / 4.0 GMAC including the Auditor, with one visual encoding per rollout. Writes occur after audit. Extra future-utility training forks are additional training cost, not hidden inside these inference estimates. Use measured peak VRAM and synchronized patient latency before making efficiency claims.

## 11. Training schedule

Shadow audit from early epochs on real evolving proposals; confidence-scaled **predicted** evidence enters feedback-only trajectories once the Auditor has learned meaningful transition distinctions. Introduce class-bound writes and balanced past-FIX/regression losses without early random hard rejection. Move to deployment-matched hard commit after held-out calibration; optional soft-commit phase is explicitly an ablation. Use the [training strategy](training_strategy.md) for concrete epochs/readiness gates. No GT event maps enter the deployed reader even during an evaluation run.

## 12. Novelty risk and narrower claim

High. Fixed image features, positive/negative revision memory, error-gated residuals and validated memory writes all exist. B is defensible only if **accepted autonomous transitions create class-specific obligations whose persistence causally improves future repair without destroying accepted fixes**, beyond those direct adaptations. Even then the contribution could be an objective/protocol rather than architecture. If plain logits plus event maps match the split C, delete the latent branches and report the simpler finding.

## 13. Five closest methods after the first falsification search

| Method | Threat | Remaining distinction to test |
|---|---|---|
| [Correction-aware interactive 3D tumor segmentation, Li and Li, 2026](https://link.springer.com/article/10.1007/s00371-026-04560-5) | Revisable positive/negative/revision memories with error-gated correction. | Autonomous predicted transition roles and class obligations, not human prompt revision. |
| [SENTRY, 2026 preprint](https://arxiv.org/html/2606.24449v1) | Verification before future-memory write; negative neighbor context. | Same-image accepted local FIX/REGRESS consequences and measured future utility at fixed annotation. |
| [FocalClick, CVPR 2022](https://openaccess.thecvf.com/content/CVPR2022/html/Chen_FocalClick_Towards_Practical_Interactive_Image_Segmentation_CVPR_2022_paper.html) | Selectively preserves existing mask regions during correction. | Preservation of specifically accepted earlier fixes, with class-bound history. |
| [MAIS, 2025 preprint](https://arxiv.org/html/2505.07511v1) | Medical interactive segmentation with persistent interaction/mask memory. | Supervised transition-specific obligation writer instead of generic interaction history. |
| [SAM2Long, ICCV 2025](https://openaccess.thecvf.com/content/ICCV2025/html/Ding_SAM2Long_Enhancing_SAM_2_for_Long_Video_Segmentation_with_a_ICCV_2025_paper.html) | Quality-scored memory pathways suppress error propagation. | Same-image correction utility, no tree search or extra hypotheses hidden in compute. |

DREAM's error/perceptual dual memory and StreamDAM are further threats recorded in the [matrix](prior_art_matrix.md), not excluded merely because this table is limited to five.

## 14. Experiments that falsify B

1. At identical L/V/latest audit, null-write or shuffle class/event history; require future utility gain across patients and seeds. Off-manifold resets need support-matched donor and trained-null controls.
2. Swap FIX↔REGRESS, zero audit, shuffle spatial evidence, and preserve masses while destroying event/class binding. If effects are indistinguishable from correct evidence, reject the audit-semantic representation claim.
3. Match A, two unstructured branches, **the exact B1 P/R writer plus one equal-parameter ConvGRU and identical losses**, P/R maps without latent C, FocalClick-style preservation, and Li/Li-style revision memory under identical gates/budgets. If any simpler control ties, narrow/remove B.
4. Remove the paired utility loss, then remove class binding, then replace accepted history with current mask only. Identify exactly what contributes.
5. Require reduced past-FIX destruction at matched A0 and final quality/headroom. Gains obtained only by accepting fewer turns or by weakening A0 do not qualify.
6. If repair obligations latch onto incorrect labels, privacy-independent per-slice state does not make the method sound; reject or simplify the writer. No performance evidence currently resolves this risk.
