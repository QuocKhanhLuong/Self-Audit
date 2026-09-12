# Falsification protocol for Candidate C

Date: 2026-09-12. Source HEAD: `9726e6d519c33035489ca7a5e4681a046493d89e`. Authority for definitions and deductions: [01](01_problem_formulation.md) and [06](06_research_decision.md); mechanism details at code level: [02](02_mechanism_analysis.md).

**This document proposes measurements. It reports none.** No study below was executed, no checkpoint was evaluated, no training was run, and no production, configuration or test file was modified. It contains no performance numbers, no effect-size targets and no clinical margins; required thresholds are named symbolically with their values deferred to development evidence and a defined use case. Three phrases are excluded from every claim it could support: *first*, *optimal*, *clinically safe*.

Two cautions carry through from 01: preservation of **predicted** FIX classes is not preservation of **true** FIX pixels (D7); and failure of any link below is evidence about this mechanism, model and budget, not proof that self-audit or geometric restitution is impossible.

## 1. Objects, and the three states that must never be merged

For one accepted ordinary transition, 01 §2 fixes `r = (H, A, E_in, t, J, θ, P, B, g)` with `B = U_θ(r;P)` the factual candidate and `B^Q = U_θ(r;Q)` the frozen replay at alternative realized supports. Three distinct states must be reported separately, because each can succeed while the next fails:

1. **`B^Q` — the counterfactual.** What the solver achieved inside the replay.
2. **`N = S + g ⊙ (B^Q − B)` — the transferred candidate.** The class-shared gate `g` is the recorded one; by D2 a repair present in `B^Q` can vanish here.
3. **`S'` — the finally retained state.** The official Auditor accepts (`S' = N`) or rejects, and rejection HALTs the trajectory. A no-op action (`B^Q = B`, zero innovation) is still audited.

Collapsing these into one "did it work" number is the single most common way this mechanism can be over-credited or under-credited.

Pixels partition into four **disjoint** sets against ground truth, following [`targets.py`](../../../src/self_audit/audit/targets.py) with `a = argmax A`, `b = argmax B`:

* **true FIX** `F = {a ≠ Y, b = Y}`
* **true REGRESS** `R = {a = Y, b ≠ Y}`
* **persistent-correct** `{a = Y, b = Y}`
* **persistent-wrong** `{a ≠ Y, b ≠ Y}` — including wrong-to-different-wrong changes.

The union is every pixel; ground truth appears only in evaluation, never at runtime. Runtime uses the thresholded predicted sets `F̂, R̂`.

**Net-correct decomposition.** For any two states `X → Z`, define `ΔN(X→Z) = #{x : X wrong, Z right} − #{x : X right, Z wrong}` on valid labelled pixels. Report `B → B^Q`, `B → N`, and `B → S'` to isolate restitution, and `A → S'` separately for the combined original correction plus restitution. Keep both unsigned components: equal repair and damage can cancel in a net figure.

**D6 as a cheap control, not a claim.** For exact hard rollback on any mask `M`, 01 §4 gives `ΔN_rollback = |M ∩ R| − |M ∩ F|` exactly; persistent-correct and persistent-wrong pixels contribute nothing. This makes a prespecified signed evidence score `E_R − E_F` the natural cheap competitor to any coordinate action, and it requires neither pixel independence nor a new optimizer. It must be measured as a control arm, not assumed to be weaker.

**D7 as a bound, not a safety guarantee.** If every predicted-FIX class is held in the transferred candidate, destroyed true FIX lies in `F \ F̂`, so for non-empty `F` the destroyed fraction is at most `1 − |F ∩ F̂| / |F|`, plus any runtime violation term. Perfect constraint compliance can therefore coexist with substantial true-FIX damage when coverage is poor. Coverage must be measured **on accepted mixed transitions**, not inferred from pooled pixel classification quality.

## 2. The six-link hypothesis ladder

Each link has its own falsifier. The ladder is evaluated in order; a link that fails stops the program at that link and identifies what to explain, not what to rebuild.

| # | Link | Hypothesis | Primary measurement | Falsifier | Prespecified limit |
|---|---|---|---|---|---|
| L1 | Opportunity | Accepted ordinary transitions contain true mixed FIX/REGRESS events often enough for the intended use | Patient-level prevalence of eligible events; path lengths; replay/identity/ordinary fallback counts | Eligible events are too rare, or exist only when A0 is deliberately weakened | `π_min`: minimum eligible prevalence worth pursuing |
| L2 | Evidence | Predicted REGRESS and FIX carry usable signal *on those events* | REGRESS precision and FIX coverage conditional on accepted mixed transitions | Evidence on eligible events is uninformative for the signed decision in D6 | `ε_min`: minimum conditional evidence quality |
| L3 | Controllability | Useful restoration directions are attainable inside the support budget | Jacobian-vector probes and a larger-budget offline oracle search, both with FIX margins tracked | Even the larger-budget oracle cannot reach useful directions | — (diagnostic gate, no outcome threshold) |
| L4 | Transfer preservation | Repairs in `B^Q` survive `N` and the official audit | `ΔN` at all three states; D2 gate-threshold survival per pixel | Counterfactual repairs are systematically erased at transfer or rejected | — (diagnostic gate) |
| L5 | Necessity | Historical coordinate intervention beats fair same-evidence alternatives at matched constraints and accounted cost | Paired fixed-history study, §3 arms | C does not exceed protected rollback or current-state optimization by `δ_min` within `κ_max`, or exceeds `h_max` | `δ_min`, `h_max`, `κ_max` |
| L6 | Generalization | The effect survives patient-held-out and frozen external protocols | Full-policy study; frozen-model external protocol | Effect is confined to the development split or to policy selection | `δ_min`, `h_max` re-applied |

`δ_min` (smallest improvement worth claiming), `h_max` (tolerable harm) and `κ_max` (compute ceiling) are **symbols, not values**. Their numbers require development evidence and a stated clinical use case; inventing them here would be a fabricated specification. They must be fixed and recorded before any locked evaluation is opened. A link is cleared when its measurement clears its limit on the prespecified primary endpoint and denominator — not when any endpoint in the table happens to move.

## 3. Design A — paired fixed-history intervention

**Estimand.** Value of an intervention *given* an opportunity. Main controlled arms receive identical frozen record `r`, predicted evidence `E`, restoration targets, recorded gate `g`, predicted-FIX set/margin fraction `γ` and final Auditor. The shipped rollback arm intentionally documents existing behavior without claiming constraint parity, the hard-rollback diagnostic changes transfer strength, and the no-FIX ablation intentionally removes protection. Label these exceptions; do not describe all arms as differing in only one variable.

| Arm | Action | Why it is in the design |
|---|---|---|
| A0 | Identity / no-op | Floor; also exposes the audit behaviour of a zero-change candidate |
| A1 | Shipped thresholded gated logit rollback | The deployed cheap action as it actually exists |
| A2 | Rollback with **identical predicted-FIX protection**, in **hard** and **soft** variants, plus a prespecified `E_R − E_F` score variant | Tests whether a weak rollback gate, not geometry, created any advantage (D6) |
| A3 | Current-state coordinate optimization | Must subtract **its own** factual current-state forward, at **matched internal depth**, keeping the previous restoration targets, gate and constraints — otherwise history, subtraction reference and depth change together |
| A4 | Candidate C as shipped | The subject |
| A5 | Candidate C without the FIX constraint | Diagnostic only; never a deployment candidate |

An ordinary Dynamic Window forward is a **cheap reference, not a compute-matched control**; naming it "the baseline" without cost accounting misstates the comparison. A same-budget frozen-network latent/feature-bias arm in the G-BRS family becomes necessary only if C clears the cheap controls, because that family is the direct challenger identified in 01 §1.

**Cost accounting, itemised.** Count and report separately: the factual replay forward; candidate check forwards (zero, one or two); the single coordinate backward; the Auditor forward; and peak memory, synchronisation and wall time. A common budget ceiling and a measured latency frontier answer different questions and must not be conflated. Per 01 §D3 the initial gradient is taken at `P`, where the displacement gradient vanishes, so `λ` can only change which proposal passes screening; the protocol measures exactly that and claims no minimal-action optimisation.

## 4. Design B — full-policy patient study

**Estimand.** Deployment utility of the whole policy, including opportunities that never arise. Each arm runs its actual accept/reject/HALT process, with the no-op fallback and HALT behaviour intact — the fallback is part of the policy under test, not an implementation detail to be silently replaced.

**Denominators, stated in advance.**

* **All-patient ITT:** every patient enters, including those whose path contains no eligible C event. This is the primary denominator.
* **Conditional-eligible:** restricted to paths that reached an eligible restitution attempt. Secondary, and reported with its own count.
* **Zero denominator is `NA`, never zero.** A patient with no eligible event contributes no conditional rate; substituting zero manufactures an effect.

**Aggregation.** Class-wise metrics first, then patient-macro averaging, as the primary. Pooled pixel aggregation is secondary and reported alongside, never instead. Uncertainty is clustered at patient level; slices and pixels are not independent replicates.

**Dilution.** From 01 §6, for bounded patient utility and two policies coupled until the first eligible event `E`, `|E[U_C − U_0]| ≤ Pr(E)`. This is an arithmetic ceiling on the ITT effect, not a performance prediction: a large conditional gain on rare events can be population-negligible, and that arithmetic must be shown rather than discovered late.

Report path-length distributions, halt turns, and the frequency of each fallback reason per arm, because Candidate C changes both the action taken and the opportunities available later (01 §6).

## 5. Offline diagnostics that must never become inference baselines

**Factorial predicted × oracle masks.** The informative offline contrast is the 2×2 of *mask source* (predicted `R̂` vs oracle `R`) crossed with *action* (rollback vs coordinate intervention). It isolates evidence quality from action quality. It is offline only: the oracle mask uses ground truth and cannot be an available inference baseline (01 §D1).

A related contrast must be dropped, not measured: comparing a "ground-truth target" against a "pre-transition-class target" **on true REGRESS pixels** is vacuous, because `a(x) = Y(x)` there by the definition of `R` in `targets.py`. The two targets coincide on exactly that set; only the *mask* factor is informative.

**Headroom versus search.** A larger-budget offline search with oracle masks can exhibit useful feasible corrections that the deployed one-step probe misses. Finding them establishes headroom on those records. Failing to find them is stronger negative evidence but still not a certificate of absent headroom in a nonconvex problem. Keep its steps, checks and feasibility strategy separate from the deployed budget. The same distinction applies to the L3 falsifier: abandon based on bounded practical headroom and prespecified utility, not an unsupported impossibility conclusion.

**Mechanistic probes.** Jacobian-vector products on a held-out diagnostic subset (no dense Jacobian needed), gate-survival checks against the D2 threshold, and the arithmetic checks already executed in [the standalone probe](probes/restitution_counterexamples.py) with [saved output](probes/restitution_counterexamples.json). Those five checks validate constructed arithmetic only; they say nothing about trained models, GPU behaviour, or usefulness.

## 6. Data protocols and generalization

Three protocols are kept separate and are never pooled:

1. **ACDC development.** The patient-level split manifest [`splits/acdc_patient_split_seed42.json`](../../../splits/acdc_patient_split_seed42.json) is present and verified as a patient-level split with seed 42 and an 80/20 train/validation partition over its 100 listed patients, i.e. 20 validation patients. This is a **development** set. Evaluating new hypotheses on it does not convert it into a confirmatory test; either a locked independent evaluation is reserved, or all further analyses on it are labelled exploratory.
2. **Frozen external.** A model selected on ACDC, frozen, evaluated on M&Ms with **no** tuning of solver parameters, `tau`, or checkpoint selection on the external data.
3. **Native M&Ms.** Supervised training and evaluation on M&Ms as a separate experiment — not additional data for the ACDC model.

**Availability is not assumed.** `checkpoints/` currently contains only `teachers/`, so an eligible frozen Candidate-C-capable checkpoint is **not** established as available at this HEAD. L1 is therefore gated on obtaining or training one; if it is unavailable, L1 is marked *pending* rather than estimated. Data availability beyond the verified split manifest is likewise not asserted here. A0 must not be weakened to manufacture regressions.

## 7. Risk calibration, stated without overreach

Threshold calibration performed on ordinary candidates does not automatically transfer to solver-produced candidates, whose error distribution can differ. Three things follow, and no more.

* **A calibration route exists in principle.** [Learn then Test](https://arxiv.org/pdf/2110.01052) (Angelopoulos, Bates, Candès, Jordan and Lei; Definition 1, §2.1, Theorem 1) controls **family-wise error** — not FDR — over a prespecified finite family of frozen configurations using valid tests on held-out calibration data. Treating the whole frozen patient-level policy as one configuration in such a family is a formulable approach. Its assumptions — a valid calibration sample, valid per-hypothesis tests, and no unaccounted adaptive training or expansion of the tested family using that sample (selection within the valid testing procedure is allowed) — are **not established here**.
* **Non-monotone losses are not a blocker by themselves.** [Conformal risk control for non-monotonic losses](https://arxiv.org/html/2602.20151v1) (Theorem 1, §2.2.3) addresses non-monotone selective losses under explicit stability assumptions. It is therefore incorrect to claim that turn-wise non-exchangeability prohibits all risk calibration; the correct statement is that the required assumptions have not been checked for this setting, and that none of this supplies an out-of-distribution guarantee.
* **The elementary union bound needs no independence.** `P(any harmful accepted action) ≤ Σ_t P(reach turn t and harmful acceptance at t)`. This holds without independence assumptions; what it lacks is data for the reached-turn events. Per-turn AUROC and a nominal `tau` do not supply them.

## 8. Minimal sequence, and what would end it

The program is deliberately short: **L1 → L2 → Design A (L3–L5) → Design B (L6)**. Anything beyond it — larger latent-optimization baselines, training-exposure factorials, calibration studies — is contingent on the preceding link clearing its limit. A giant pre-planned suite would spend credibility this mechanism does not yet have.

Interpretation boundaries, restated so they cannot be quietly dropped: a single Dice improvement does not establish the ladder; a wide interval crossing zero is not evidence of equivalence; absence of benefit in this program does not demonstrate that the mechanism is impossible; predicted-FIX constraint compliance is not true-FIX safety; and none of these studies has been run.

## 9. Source verification

Primary method material read at the equation level, reused from the root ledger ([05](05_evidence_ledger.md)) rather than re-searched:

| Source | Material | Role here |
|---|---|---|
| [G-BRS, CVPR 2022, §2](https://openaccess.thecvf.com/content/CVPR2022/papers/Lin_Generalizing_Interactive_Backpropagating_Refinement_for_Dense_Prediction_Networks_CVPR_2022_paper.pdf) | Auxiliary refinement variables on a frozen predictor | The same-budget latent-optimization challenger in §3 |
| [f-BRS, §3](https://arxiv.org/abs/2001.10331) | Auxiliary scale/bias optimization instead of inputs | Defines the competing intervention-variable family (01 §1) |
| [Counterfactual Attention Learning, ICCV 2021, §3 Eqs. 4–6](https://openaccess.thecvf.com/content/ICCV2021/papers/Rao_Counterfactual_Attention_Learning_for_Fine-Grained_Visual_Categorization_and_Re-Identification_ICCV_2021_paper.pdf) | Factual-minus-intervened attention effect | Why factual subtraction alone is not a distinguishing contribution |
| [Algorithmic recourse, FAccT 2021, §§3–4](https://arxiv.org/pdf/2002.06278) | Interventions versus counterfactual explanations | Why §5 probes a computational object, not patient anatomy (01 §D4) |
| [Learn then Test, Def. 1, §2.1, Thm. 1](https://arxiv.org/pdf/2110.01052) | FWER control over a fixed family via valid tests | The calibration route named in §7 |
| [Non-monotonic CRC, Thm. 1, §2.2.3](https://arxiv.org/html/2602.20151v1) | Stability-based bounds for non-monotone losses | Why non-monotonicity alone is not a blocker |

Preprint entries remain preprint evidence. No claim of venue, priority or exhaustive coverage is made, and none of these sources is cited as evidence that Candidate C works.
