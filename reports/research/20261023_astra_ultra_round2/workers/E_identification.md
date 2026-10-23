# E — What a frozen transaction comparison identifies

Status: **DERIVED + EXECUTED MANUFACTURED CPU PROBE; NO PATIENT EVIDENCE**. Research-only round 2 at parent `ccfde9bb99b6050e00ad94fbd6d57fcf9afc8b3f`; canonical source `637f38ced217fe1db2347a87924695533056d8f0`, measured experiment `8960b6baac4c6600553d900c303f192c2fcfa4f0`. No prior report, production source, config, baseline, checkpoint or dataset was changed. No GPU, patient execution, weight download, commit or push.

## Finding

A positive same-state current-audit J−S contrast does **not identify** either a positive contrast on a different support or the sign of the complete training-policy effect. This report supplies an exact numerical construction, not just a generalization warning. Both J and S pass their actual current-audit guard. J is better on that audit but worse on a second support; subsequent EMA argmax feedback amplifies the reversal. No numerical solver failure or nonlinear approximation is needed.

The fixed-P replay remains useful: it identifies what each transaction rule would do at the same P-visited state. It is a cheap conditional mechanism probe. It is not a surrogate for independently trained J/S trajectories.

## Two questions and estimands

**Q1: At a fixed supervised reference visited by P, what changes when only the allocation/guard rule is replaced by J versus S?** Let H_t contain student theta_s, frozen teacher phi_t, optimizer/buffer states and the prior history. Let Z_t contain the fixed supervised/audit/U examples, coordinates, masks, augmentations and exogenous schedule. D_t=G(H_t,Z_t) is the fixed candidate set. For policy m, delta_m(H_t,Z_t,D_t) is its finite accepted displacement. Define, with positive values favoring J,

`tau_A^P = average_t [ R_A_t(theta_s^P + delta_S) − R_A_t(theta_s^P + delta_J) ]`.

The empirical average is over the predeclared P-reference transaction sequence, with the **same H_t, Z_t and D_t within each paired comparison**. A shadow-support counterpart `tau_B^P` changes only the measurement support; **B reference labels and measured scores** cannot enter allocation, guard, teacher, live state, schedule or checkpoint selection. The image/feature support on which B is measured may separately be an allowed unlabeled proposal input; that is transductive support, not an untouched held-out patient. The allowed role must be explicit. These are controlled transaction effects under P's visited-state distribution. They can be computed without observing a live J trajectory.

Report the all-scheduled-transactions estimand first, including zero-candidate/zero-update cases. A secondary multi-source estimand may condition on `m(H_t,Z_t)>=2`, a common **pre-intervention** event. Do not condition the primary comparison on J acceptance, S acceptance, or a favorable post-update risk: those differ by policy. Define denominators and missing measurements explicitly.

**Q2: What is the difference in final outcomes when the entire sequence of commits and EMA updates follows J rather than S?** For a fixed recipe and matched initialization/exogenous schedule seed,

`tau_T = M(theta_T^J) − M(theta_T^S)`

for a higher-is-better patient metric M, or reverse the subtraction for risk. Average paired differences over prespecified subset/init blocks. Each policy follows its own recurrence: supervised reference, teacher q, confidence support, directions, acceptance, direct commit and once-only EMA. Subsequent differences in q/D are **mediated effects of the policy**, not failures to hold the input fixed. The comparison identifies the total policy effect under that recipe; it cannot attribute the final difference solely to an off-diagonal Gram term.

Sharing exogenous patient/crop schedules removes avoidable nuisance differences but does not force equal model states, pseudo labels or accepted support. A fixed P trajectory samples `d^P`, while full J and S runs visit `d^J` and `d^S`. There is no support or weighting argument in the current repository that lets a positive `tau_A^P` determine `tau_T`. Deterministic state trajectories can have no overlap at all. Candidate opportunity counts on P must not be presented as J's on-policy opportunity counts.

## Exact counterexample with valid probabilities

Two manufactured supports A and B each have an RV-positive observation. Let theta=(theta_A,theta_B), with four-class probabilities

`p_x(theta) = (0.98−theta_x, theta_x, .01, .01)`

for BG/RV/MYO/LV. Both supports have the same class meaning. The protected audit is A. The three foreground half-Brier risks are

`R_A(theta) = ( (1−theta_A)^2/2, .00005, .00005 )`.

B reference truth is used only for synthetic shadow/endpoint scoring. Its image/feature support is also used by the explicitly declared second-step teacher proposal map below. Therefore B is a **label-hidden, transductive toy outcome support**, not an untouched test patient. This does not authorize reading U masks in the native pipeline; the toy manufactures its labels and contains no patient data. Start theta=phi=(.5,.5). The supervised reference is the identity to isolate the extra transaction. Initial candidate columns are

`d1=(.35,−.35)`, `d2=(−.15,.35)`.

Use the mean of the three group risks plus ridge `lambda ||alpha||²/2`, lambda=.001, zero budgets, alpha in [0,1]², followed by the exact finite group guard. The local probabilities are affine here, so the model and actual risks coincide. These are explicitly handcrafted directions: **this construction does not claim parity with the v2 CE candidate generator or softmax decoder**. B/F supply complementary source-faithful probes; this one isolates sequential identification.

J's exact box optimum is alpha=(1,0). Its objective derivative at that corner has first component negative and second positive, the required upper/lower-bound signs; strict convexity gives uniqueness. S restricts alpha=a(1,1), with the correct scalar ridge **2lambda**. Its derivative at a=1 is negative, so the exact scalar optimum is a=1. Both updates are finite and pass the current-A guard.

| Quantity | Before | J | S (= P at this state) |
|---|---:|---:|---:|
| theta_A | .5 | .85 | .70 |
| theta_B | .5 | .15 | .50 |
| Current-A RV half-Brier | .125 | .01125 | .045 |
| Shadow-B RV half-Brier | .125 | .36125 | .125 |

Thus the same-state current-A **RV half-Brier** contrast is **+0.03375** in J's favor, but the changed-support RV contrast is **−0.23625**. All scalar contrasts reported in this example use RV half-Brier; the corresponding three-foreground-group mean contrasts are one-third as large because the other two groups remain constant. This is not an audit-guard violation: MYO/LV risks remain constant and RV risk on A falls. No change in solver precision, ridge matching, or finite-backtracking accuracy removes this logical reversal.

### One EMA step makes the proposal distributions different

Update **parameter EMA** once with decay .5. The teacher coordinates become phi_J=(.675,.325) and phi_S=(.60,.50). Because this toy probability map is affine, averaging parameters also averages its probabilities exactly. This equality does not hold generally for a softmax network or justify replacing native parameter EMA by probability EMA. On B, J's teacher predicts BG while S's teacher predicts RV. There is no tie on either trajectory: the BG/RV boundary is theta_B=.49, while these teacher coordinates are .325 and .50 (margins .33 and .02 respectively). Define a transparent second-step proposal map:

`d_next = (.05, +.05)` if the teacher's B argmax is RV, otherwise `(.05, −.05)`.

This is a declared abstract EMA-coupled response map, **not a claim that a fixed binary CE feature vector has these gradients**. The prototype uses no hidden B reference label in that map: it reads teacher probabilities only. The B ground truth is used only to score the synthetic endpoint. The single source's optimum is alpha=1 for both policies, and both exact A guards accept again.

| Final policy | theta | A RV half-Brier | B RV half-Brier | Equal A/B mean risk |
|---|---|---:|---:|---:|
| J | (.90,.10) | .005 | .405 | .205 |
| S | (.75,.55) | .03125 | .10125 | .06625 |

J still wins on its reused A audit by .02625 but loses on B by **.30375**, and loses on the fixed equal-support outcome by **.13875**. Under a diagnostic intervention that holds the second-step teacher at the initial (.5,.5) for both policies, J's B risk is .32 instead of .405. EMA feedback contributes another **.085** B risk in this constructed comparison. That intervention changes the algorithm and is not a general natural-direct/indirect-effect identification claim.

On a P-reference replay, the second state is instead (.70,.50), its teacher predicts RV, and both shadow J/S receive the positive-B direction. The second replay contrast is zero. The actual J trajectory's second BG candidate is **never seen by that P replay**. This concretely demonstrates the missing visited-state support rather than assuming it away.

## Executed harness and purity checks

Run:

```sh
rtk proxy python reports/research/20261023_astra_ultra_round2/probes/E_identification.py
```

Receipt: [E_identification.json](../evidence/E_identification.json). Result: **PASS**. Python standard library only; no production imports, model/data arrays or package installation. Joint coefficients use an exact face-enumeration oracle for the manufactured rank-one ridge objective, not the known scale-sensitive production SLSQP routine. The receipt records probabilities, directions, coefficients, risks, state/candidate hashes, per-policy trajectories and the teacher-frozen intervention.

The live P run draws one scheduled RNG value per step. Every shadow gets cloned state and an isolated RNG restored from the frozen state; it deliberately consumes that isolated RNG to exercise separation. All six P/S/J evaluation orders, each with one or three shadow repetitions, produce the same live endpoint as the no-shadow run: **12/12 invariance cases**. Per-shadow live-state/RNG hashes are unchanged; candidate/reference hashes match across paired policies; optimizer/buffer sentinels are unchanged; there are exactly two live EMA updates for two live P commits. Shadow telemetry is never read by the live update.

These checks establish purity of this tiny harness only. They do not certify native PyTorch/CUDA replay, arbitrary stochastic modules, production optimizer snapshots or real teacher histories.

## Minimal telemetry contract for A/B/F probes and a future native replay

A single versioned receipt should make the following six records joinable by immutable `transaction_id`, `snapshot_id`, `policy_id`, and `support_id`. A receipt with absent fields must say `UNAVAILABLE`; it must not invent reconstructable traces from aggregate counts.

| Record | Minimum fields and purpose |
|---|---|
| Frozen reference | Canonical/experiment/research code hashes; config/weight/data identities when applicable; iteration; student **after supervised reference** hash; fixed teacher/optimizer/buffer hashes; supervised optimizer count; EMA count; patient/slice/crop IDs; all live RNG states or restorable hashes plus device/determinism metadata. Separately identify the full runtime, because source_digest alone does not cover it. |
| Fixed candidate inputs | Teacher q or reproducible payload hash; U input/strong-view/valid-mask/accepted-mask hashes; threshold; accepted denominator Z; per-class counts; unique source IDs; candidate parameter hashes/norms; exact block; pseudo LR. A paired frozen comparison must agree here before its results are comparable. No U-reference labels. |
| Response and solve | A_c,b_c,e_c in original units; ridge (m*lambda for restricted S); budgets/tolerances; raw/scaled solve settings; oracle/source/numerical status; feasible-control objective comparison; alpha; conditioning/active-set diagnostics where meaningful. Keep exact-zero, numerical false-stall, no-candidate and finite rejection separate. A supplies scale evidence; B supplies geometric feasibility evidence. |
| Actual trial/commit | Every trial scale; local predicted and actual class risks; finite/feasible decision; effective parameter and probability delta; changed-element count; exact committed-trial hash or explicit no-commit; JVP/gradient/trial counts and time. `scale>0` is not sufficient evidence of a real update. |
| Shadow endpoint | Support identity and whether used in any previous optimization; explicit label access charged to L; pre/post class risks, positive/negative support, confusion/Dice/FP/FN with absent conventions, prediction/hash or small sufficient fixture. This lets F's Brier-to-Dice failure be audited. No shadow score may determine a live decision. |
| Purity and accounting | Live student/teacher/optimizer/buffer/RNG hash before and after each shadow; method order; clone/local RNG provenance; only the preselected live policy's commit count and once-only EMA delta; shadow costs separated from live costs. Fail closed on any unexplained state/RNG difference. |

For PyTorch, snapshot Python/NumPy/Torch CPU and every used CUDA RNG stream, optimizer state, mutable buffers and teacher—not just trainable parameters. Restore neither an already-mutated live state nor only the RNG after accidentally running a shadow optimizer. Prefer functional calls on detached snapshot state and cloned buffers; any local randomized diagnostic uses an isolated generator. Freeze model mode and verify outputs; `eval()` alone is not a universal purity proof. The current CLI lacks this native replay contract, scalar arm and separated schedule seeds.

## Minimum honest extension beyond P replay

First finish numerical and mechanism qualification on manufactured/source cases. A frozen P trace can then screen whether J has any effective advantage over S at meaningful multi-source opportunities, with Brier and same-support Dice/FP/FN reported together. No support-shift or teacher-rate tuning may be chosen using confirmation outcomes.

If that screen is informative, the minimal policy study trains **J and S independently** from identical initial states with identical pre-generated exogenous schedules; each maintains its own teacher, optimizer timeline and acceptance support. Retain P only if the claim includes improvement over plain SSL. Evaluate a fixed final checkpoint and a prospectively specified patient outcome, with repeated subset/init blocks as already described in round 1. Do not expand this into the 81-arm matrix. Greater practical superiority claims still need their label/compute/strong-SSL controls, but are outside this two-question identification probe.

For an intermediate diagnostic, replay all rules on snapshots from **each** reference policy, producing tau^P, tau^S and tau^J with the reference distribution in the name. This tests reference-distribution sensitivity; it still does not replace final policy outcomes. Report all scheduled support/candidate histograms per reference. Do not use P-only logs to estimate how frequently J stalls or invokes a particular pseudo class on-policy.

Audit reuse remains optimization. A second L patient unseen in the current step can reveal support sensitivity but is not a new independent generalization cohort if its labels influenced past states. Changing a shadow score into a gate, schedule, tuning or stopping rule creates another algorithm and requires a new declared estimand. The present counterexample requires only known algebra and executable state recurrences; no external empirical claim or literature-derived theorem is needed.

## Independent B review disposition

B independently checked 12 policy/teacher-intervention/step branches, including every coefficient-box corner's simplex validity, affine parameter-EMA/probability-mixture equality, tie behavior and final risks; receipt `../evidence/B_review_E.json`. Accepted clarifications: B labels/scores are excluded from live decisions while its image/feature support enters the declared second proposal map; reported scalar gains are RV-channel half-Brier, not three-group mean; affine EMA equivalence is not a softmax-network claim. The example is transductive and manufactured, with no patient-mask access or native candidate-realizability claim.
