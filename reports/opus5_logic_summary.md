# Self-Audit — Logic Summary (short form)

**HEAD:** `a901eeae56cf6835bb9a1d9c79a9d5d3e99374c6` (`main`) · **Date:** 2026-09-05
Full report: `reports/opus5_model_logic_audit.md` · Raw worker evidence: `reports/workers/`

---

## THE MODEL THAT THE CODE CURRENTLY IMPLEMENTS IS

Derived from executable code only (`src/self_audit/models/self_audit_net.py`, `annotation_expert.py`, `auditor.py`), not from docs.

1. A 2.5-D input `[B,3,H,W]` (three neighbouring MRI slices packed as RGB) goes into an ImageNet-pretrained ConvNeXt-Tiny, producing four pyramid levels.
2. A top-down FPN fuses them and returns **only the finest level**, `shared ∈ [B,96,H/4,W/4]`. The other three refine branches are computed and discarded (12 dead parameters).
3. `InitialAnnotationHead` maps `shared` → 4-class logits at H/4 and bilinearly upsamples ×4 to full resolution. That is **A0**. It never sees a target.
4. `state ← A0`; `previous_audit_evidence ← None`.
5. For each turn `t`, over the rows still active, **one shared** `AnnotationExpert` runs. Its input is 104 channels: `shared` (96) ‖ upsampled `A_t` (4) ‖ per-pixel entropy of `A_t` (1) ‖ previous audit evidence (3).
6. Inside the expert, a **single** `DynamicWindowAttention` block is applied `depth = min(t+1, 3)` times (hard-coded cap 3). Weights are genuinely shared across turns and across depth iterations — verified by Python object identity — but the block is *conditioned* by per-turn and per-iteration `nn.Embedding` tables, so the conditioning is stage-indexed.
7. The candidate is a gated residual edit: `A_cand = A_t + σ(gate) ⊙ Δ`, where `Δ` is 4-channel and `gate` is a single channel broadcast over classes. `gate` initializes near `σ ≈ 0.38`, so the expert cannot emit an exact null update.
8. `CounterfactualAuditor` receives **detached** `shared`, `softmax(A_t)`, `softmax(A_cand)`, their difference, and both entropies — 110 channels, all bilinearly downsampled to stride 4. It never sees the raw image and never sees a target.
9. It emits `local_logits ∈ [B,3,H,W]` (FIX/UNCHANGED/REGRESS, computed at stride 4 then upsampled) and a scalar `delta_q ∈ [B,1]` = predicted signed ΔDice.
10. The gate is `accepted = delta_q.detach() > tau_accept`, with `tau_accept = 0.0` in every config. **`local_logits` never enters this decision.**
11. Accept → `A_{t+1} = A_cand` and `previous_audit_evidence = softmax(local_logits).detach()`. Reject → `A_{t+1} = A_t`, evidence zeroed, and **the row halts permanently** (`active = accepted`).
12. Output is the retained final state, plus per-turn candidates, audits, accept masks and halt turns.
13. **Phase A** supervises `A0, A1, A2, A3` all directly against GT with effective weights `[0.167, 0.233, 0.267, 0.333]` (`softDice` with background + CE). No auditor, no gate; the expert's 3 audit channels are hard zeros.
14. Because `state` is passed un-detached between turns, `L(A3) → expert → A2 → A1 → A0 → head → FPN → encoder` is a **single active autograd path**. Measured: A0's head receives **7.8%** of its gradient from its own term and **92.2%** from A1–A3.
15. **Phase B** freezes everything except the auditor and, per batch, builds **3 on-policy adjacent pairs + 4 GT-synthesized counterfactuals** (one around every state including A0), then takes a uniform mean over the 7 groups. Synthetic = **57.1%** of the loss.
16. Phase-B **validation uses the identical generator and the identical code path**, and `primary_metric = AUROC` on that mixture selects the exported checkpoint.
17. **Phase C** runs the real gated `infer(mode="self_audit")`, supervises **only the final retained state** against GT, and adds a detached audit term. It is 100% on-policy. Gradient firewall verified in both directions (120/120 and 28/28 params).
18. Threshold calibration takes an argmax over 81 τ on the **validation** cache and reports the maximized value — then every reported self-audit Dice uses the **hard-coded τ = 0.0** instead, so the calibrated τ is never actually consumed.
19. Reported Dice is 2-D per-slice on a resized 256×256 grid, foreground classes only, with **empty-vs-empty scoring 1.0**. Phase A additionally batch-pools while Phase C is per-slice.
20. No held-out test split is ever evaluated. `test_split: test` is declared in all three configs and used nowhere.

## THE MODEL WE INTENDED TO IMPLEMENT IS

Image → shared encoder → initial head → **A0** → **one shared recurrent Annotation Expert** → `A_candidate^(t)` → **Counterfactual Audit Expert** → local transition evidence {FIX, NEUTRAL, REGRESS} **and** global predicted ΔQ → **ACCEPT/REJECT** → accept: `A_{t+1} = A_candidate`; reject: `A_{t+1} = A_t`, optionally halt → repeat up to T. GT is a train/validation oracle only and never touches a deployable decision. Counterfactual generation is a training mechanism. The central hypothesis is that the expert creates useful correction **headroom** and the auditor captures it while blocking harmful edits, so `A0 / candidate / always-accept / self-audit / oracle` must be separable, with the auditor's headline metric measured on **on-policy** transitions and one consistent neutral margin ε throughout.

## THE IMPORTANT DIFFERENCES ARE

| # | Intended | Actual |
|---|---|---|
| 1 | Local evidence **+** global ΔQ drive accept/reject | **ΔQ alone**; the local head has no path to the decision |
| 2 | Headline auditor metric is on-policy | **No on-policy-only metric exists anywhere**; every reported number is a 57.1%-synthetic mixture that also selects the checkpoint |
| 3 | Expert creates headroom | Every stage is pulled to the same GT fixed point; **92.2%** of A0's gradient arrives from the later stages, so the objective actively removes headroom |
| 4 | One ε governs IMPROVE/NEUTRAL/REGRESS | **Three** ε (0.0 / 0.005 / 0.02); 8 sites use zero; `ΔDice == 0` is counted as a *harmful acceptance*; AUROC's positive class is `> 0.0` |
| 5 | "Optionally halt" | Rejection **always** halts the row permanently |
| 6 | Local FIX/NEUTRAL/REGRESS evidence | UNCHANGED conflates "both correct" with "both wrong"; the head runs at stride 4 against a per-pixel target — oracle-through-stride-4 **FIX recall = 0.0000** |
| 7 | One shared expert | Weights genuinely shared, but conditioned by per-turn / per-iteration embedding tables |
| 8 | Audit evidence conditions the next candidate | Hard-zero throughout Phase A and Phase B; only non-zero at inference — a train/deploy distribution shift on 3 of 104 channels |
| 9 | Metric supports comparison to literature | 2-D, resized, empty-class→1.0, no volume-level Dice, no test split, τ selected and reported on the same data |
| 10 | GT is a train/val oracle only | **Correct — verified.** No leakage. `oracle_target` in the `infer` signature is an API-clarity risk, not a leak |

---

## P0 findings (4)

| # | Finding | Evidence |
|---|---|---|
| **P0-1** | Reported Dice is **not the ACDC metric**: empty-vs-empty scores **1.0** in all six Dice implementations, at 2-D slice granularity on a resized grid; `foreground_only` never enabled; no volume-level Dice is ever computed (`evaluate_comparison_modes` has **zero callers**). A slice where the model is totally wrong still scores **0.667**. | `metrics.py:58`, `targets.py:81-84`, `self_audit_net.py:355`, `_utils.py:285` |
| **P0-2** | Phase-A headline Dice is **batch-pooled**, Phase-C is **per-slice** — executed: **0.6825 vs 0.8333** on identical inputs, and Phase A is `batch_size`-dependent. The two headline numbers are not comparable. | `train_annotation.py:242` vs `finetune_joint.py:286` |
| **P0-3** | `modes/headroom_capture_ratio` is a **mean-of-ratios** with a `1e-8` guard. Executed demo returns **625000.4** where the correct ratio-of-means is **0.667**. This is the paper's central statistic, and near-zero headroom rows are exactly what §7 predicts. | `audit_decomposition.py:305-308` |
| **P0-4** | **No held-out test evaluation anywhere.** τ is chosen by argmax over 81 values on `val` and the maximized value is reported on that same `val`; separately, the calibrated τ is **never consumed** — reported self-audit Dice uses hard-coded `τ = 0.0`. | `train_self_audit.py:471-473,570-598`; `calibrate_threshold.py:66` |

## P1 findings (7)

| # | Finding | Evidence |
|---|---|---|
| **P1-1** | Phase-B **validation** uses the same GT-driven `CounterfactualGenerator` as training — synthetic carries **57.1%** of the loss weight behind `val/auroc`, which is also the checkpoint-selection metric. **AUROC 0.82 is reachable from `ΔP`/entropy statistics alone**, with no image and no GT. No on-policy-only auditor metric exists in the repo. | `train_auditor.py:97-127,217,392-402,428-429,522,530` |
| **P1-2** | **Objective-induced A0 saturation.** A0's head gets **7.8%** of its gradient from its own supervision and **92.2%** from A1–A3 through the shared expert; nothing selects on A0 quality; no single-stage A0 baseline exists. `refinement_gain > 0` is therefore not evidence for the headroom hypothesis. | `train_annotation.py:93,98`; `self_audit_net.py:76`; measured |
| **P1-3** | The **positive** counterfactual branch is degenerate at realistic confidence: at `p_max ≥ 0.95`, **58–68%** of positive CFs change **zero** pixels (still returned `valid=True`) and **91–95%** land inside the loss's neutral band. Class prior `P(ΔDice>0) = 0.230` ⇒ a **~77% "do not accept"** curriculum, training genuine repairs to `Δq ≈ 0` against a strict `> 0` gate. | `counterfactual.py:316-320,332`; `losses/audit.py:20,36` |
| **P1-4** | Neutral-margin semantics violated at **8 sites** across 3 mutually inconsistent ε. `ΔDice == 0` is counted as a *harmful acceptance* (executed: `harmful_acceptance_rate = 0.8` where the true value is 0.2), which steers τ selection more conservative. AUROC's positive class at `> 0.0` means **the reported AUROC scores a task the loss never trained**. | `metrics.py:286,287,316,317`; `threshold.py:85,86`; `self_audit_net.py:346`; `audit_decomposition.py:306` |
| **P1-5** | The local audit head **cannot represent its own target**: trunk at stride 4 vs per-pixel target. Oracle-through-stride-4 **FIX recall = 0.0000**. Target also conflates "both correct" with "both wrong" as UNCHANGED — a MISMATCH vs the intended 3-way semantics. | `auditor.py:97-114`; `targets.py:55-70` |
| **P1-6** | **No volume-level 3-D Dice is ever reported**; ED/ES are pooled; the only native-geometry path has zero callers. The reported number is not comparable to published ACDC results. | `volume_inference.py:294-300` |
| **P1-7** | Train/inference mismatch: `previous_audit_evidence` is **hard-wired `None`** in Phase A and Phase B but fed at inference. Plus per-turn / per-iteration `nn.Embedding` tables in `DynamicWindowGenerator` make the "one shared expert" claim true of weights but not of conditioning. | `self_audit_net.py:64` vs `:217-221`; `dynamic_window.py:64-65` |

## What is verifiably correct (could not be falsified)

- **No GT leakage in deployable inference.** Bitwise-zero GT-permutation invariance across A0, candidates, `Δq`, accept masks, final logits and halt turns — with a **powered positive control** (`oracle_accept` on the same pair flips every decision and moves logits by 0.5098).
- **Gradient firewall holds both ways**: audit-only backward leaves all 120 annotation params `grad=None`; annotation-only backward leaves all 28 auditor params `grad=None`. Enforced redundantly at three points.
- **The auditor genuinely audits a transition**, not candidate quality: a candidate-only shortcut scores **AUROC 0.525** — chance.
- **Weight sharing is real** (one module object, 6 refinement invocations over 3 turns).
- **Stage-wise attribution algebra is correct**: reject-harmful `= +0.1667`, reject-beneficial `= −0.1667`, residual `< 1e-7`. All four modes and all four headroom quantities exist and are logged.
- **Patient-level split is clean**: 80/20 patients, zero overlap, ED/ES never straddle, no GT-guided cropping.
- Checkpoint loading is `strict=True` with architecture validation.

## Recommended next experiment

Do **not** change architecture yet. Run `scripts/audit_checkpoint.py` on the existing trained checkpoint — it already computes exactly the right quantities and appears never to have been run — after three minimal fixes: P0-1 (empty-class policy + volume-level Dice), P0-3 (ratio-of-means headroom capture), and P1-4 (one ε).

Then fill this 2×2 and report all four modes plus `CandidateGain`, `PositiveHeadroom`, `RealizedGain`, `AuditGateValue`, and **on-policy-only AUROC** in each cell:

| | ACDC val (in-domain) | M&Ms (out-of-domain, `configs/self_audit_acdc_to_mnms.yaml`) |
|---|---|---|
| **Volume-level, native geometry, empty-class excluded, per-class RV/MYO/LV, ED/ES separate** | `initial_only` · `always_accept` · `self_audit` · `oracle_accept` | same four |

The diagnosis reads straight off the table:

- `oracle ≈ initial` in both cells ⇒ **no headroom exists**. Disambiguate with a `stage_weights: [1.0]`, `max_turns: 0` single-stage baseline: if it matches A0, A0 is genuinely strong; if it is much worse, deep supervision saturated A0.
- `oracle ≫ initial` but `self_audit ≈ initial` ⇒ **headroom exists, the auditor cannot find it** — a real, localizable, publishable negative.
- `always_accept ≫ self_audit` ⇒ the gate is over-conservative; τ=0 against a 77%-reject curriculum is the prime suspect (P1-3).
- Flat in-domain but positive out-of-domain ⇒ the contribution is real and the paper should be framed as domain-shift robustness, not in-domain ACDC.

This costs one evaluation pass, requires no retraining, and settles questions 3, 4 and 5 of the audit brief in a single run.


---

## Post-remediation status (added 2026-09-06)

The original findings above are **preserved as historical evidence** and describe
`a901eea`. A first remediation pass has since landed. It changed measurement and
reporting only — no architecture, no training objective, no retraining, no new split.

**Fixed:** all four P0s (empty-class policy and volume-level reporting; Phase-A/Phase-C
proxy alignment; `headroom_capture_ratio` as a ratio of sums; calibrated τ persisted and
actually consumed), plus P1-1 (on-policy vs synthetic Phase-B namespaces, `primary_metric`
now on-policy AUROC), P1-4 (one canonical decision margin, neutral never counted harmful)
and P1-6 (per-class RV/MYO/LV, ED/ES separation, truthful `metric_space` labels).

**Still open by design:** A0 saturation (P1-2), positive-counterfactual degeneracy (P1-3),
the stride-4 local head (P1-5), local evidence not reaching the gate, and the absence of a
held-out test split. Those need training changes or an architecture pass.

**Test suite:** 73 passed at `a901eea` → 103 passed after remediation.

**Next action:** run `scripts/audit_checkpoint.py` on the existing checkpoint. Do not
retrain. Details, exact commands and the pre-committed interpretation table are in
`reports/remediation_metrics_evaluation_results.md`.
