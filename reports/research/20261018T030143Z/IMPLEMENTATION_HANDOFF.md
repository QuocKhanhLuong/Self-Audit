# Proposed implementation handoff — NOT APPROVED

Phase I ends with this document. No implementation contract is activated. Nothing here changes canonical `train.py`, ACTIVE_PIPELINE, source pins, bank policy, reference, baseline/ablation or historical runs. All designs are **PROPOSED_NOT_APPROVED**; no job is queued. Local base: `f00e4faa52204f8e6863f76a493684b68ed3a132`. The human must select H06/H11 or the semantic-first H07 emphasis in DECISION_BRIEF.md before Phase II.

## Entry conditions for a later approved phase

1. Re-resolve remote main and pin the implementation base in a separate clean worktree. Compare its changes with this research base; do not silently transplant source facts from this snapshot. Coordinate with the concurrent implementation agent, preserve its checkout and processes, and identify one owner of shared production files.
2. Record the approved named hypothesis, exact permitted inputs/semantic priors, inference setting, native observation, risk measures and excluded claims. A change to Q/Z/support, decoder, source pin, data sampling or optimizer is a separate factor; do not combine them.
3. Create a versioned opt-in policy only after approval. Required fields are specified below; no ACTIVE_PIPELINE/config mutation is authorized now. Baseline/reference execution and historical runs remain immutable.
4. Establish disabled-path output/loss/gradient parity against the new pinned base using identical tensors/RNG/state. Run the original 47 reference tests plus that base's current suite and canonical routing checks. This session ran 153 current tests, not differential parity of a future implementation. Existing reference hashes must be verified from its recorded manifest; no new pin is selected by this document.
5. Confirm official-pretrained source/weight provenance and loading receipts separately from random/manufactured software tests. This session downloaded neither weights nor data and did not verify the chat-reported mirror receipt.
6. Approve a separate compute/data/evaluation budget before any patient work. Implementation approval does not automatically authorize 4070 training, patient data access, reference-based tuning, full50, merge or publication.

## Proposed policy fields

| Field | Requirement |
|---|---|
| research_basis | This session path, base SHA and selected hypothesis ID |
| information_contract | Exact TRAIN/inference inputs; metadata values/validity/missingness; ontology; source lineage; upstream labels and priors |
| target_policy | Historical full Q unchanged unless separately approved; any h/Z construction must be explicitly justified and versioned |
| observation_space | Logits versus probabilities; original teacher grid/transport versus student D/U; coordinate transforms and normalization |
| native_evidence | Exactly one frozen evidence mechanism; source of anchor estimates; absence/conflict policy; no current-student reconstruction bypass |
| support | Full valid domain, evidence support, pixel weights and denominator; partial/empty cells and all-zero support handled explicitly |
| edit_family | Fixed scalar direction versus unrestricted detail head, bound, coarse freeze policy, shared encoder paths and output collateral |
| proposal_contract | Origin, coordinate system, metric, step count, step size, feasibility rule, detach point, refresh schedule and content hash |
| optimization | Parameters changed, initialization, LR groups, clipping, optimizer/scaler/scheduler state, freeze schedule and computation accounting |
| evaluation | Fixed class/channel map, no GT naming/permutation, patient denominators, absent-class policy, dense/abstention metrics, no tuning on held-out truth |
| inference | Frozen forward/per-cine preprocessing/per-image optimization/cohort adaptation as separate values, never inferred from a filename |
| authorization | Explicit approval reference and separate execution budget; default remains NOT_APPROVED |

Truth-dependent quantities such as anchor error, actual mixed-cell occupancy, native margin, proposal truth gain, graph anatomical consistency and true-RV density overlap cannot appear as if operationally measured by no-GT confidence. Manufactured truth may sweep those variables; real code must not read hidden masks to set them.

## Minimal proposed H06 surface

Use the existing model and historical bank as the baseline. A first research adapter works at outputs, not inside a larger backbone:

```text
input: native X, allowed frozen anchor estimates, current lifted/coarse logits
if required evidence missing/conflicting or support empty:
    emit NO_ADDITIONAL_EVIDENCE; do not invent target, loss denominator or class
else:
    freeze c_B=(X-mu_B)^2 and c_M=(X-mu_M)^2
    declare supported complete cells, constant-per-cell g, output coordinates
    compare no edit against exact k_j * (I-U D)X_j
    compare same frozen energy directly against a detached controlled proposal
    record actual probability/argmax changes for BG, RV, MYO, LV separately
```

The proposed insertion point is after `LayerGuidedStudent.forward` (`layer_upgrade/model.py:102–114`) and at native loss assembly beside `training_v1/runner.py:433–470`. The currently rejected native phase is not to be enabled as a quick flag flip. A new versioned route requires the new contract. `training_v1/data.py:500–558` remains historical mass/support transport; it is not repurposed as an unverified anatomical forward model.

A fixed scalar v=(0,1,−1,1) edit provides the theorem-aligned output control. It preserves RV:LV odds, not BG probability. An unrestricted head follows a different gradient and needs its own tests. If a validity mask changes the projector, derive its metric/adjoint rather than applying a loss mask and calling it the same H. Do not silently add graph, motion, shape, reconstruction and EMA losses together.

## Minimal proposed H11 surface

Only after the proposal is qualified under a selected protocol, freeze it and compare transfer in the same model/initialization. Record proposal quality separately from student approximation on the same support/measure. Use the same frozen native energy as the direct-loss comparator. For same-origin one-step squared matching, metric-correct gradient equality is expected. Multi-step or remote-origin targets must state why they add more than scaling, sharpening or stale-target regularization.

Require a precise actual input audit: a proposal using full temporal context may not be pointwise transferable to a single-frame student. Do not present omitted context as insufficient network capacity. Stop-gradient prevents target movement; it does not certify target correctness. With no extra proposal evidence, preserve the declared baseline loss; no fake fifth class or forced-label fallback is introduced.

## Positive and negative test contracts

| Contract | Positive check | Required counterexample / negative check |
|---|---|---|
| No intervention | Disabled outputs, losses, gradients, parameter groups and serialization match pinned baseline | Hidden default enables evidence or changes sampling/normalization |
| Information lineage | Every anchor/target has source and support; repeated source gets no extra independent evidence | Duplicate aliases inflate confidence; missing geometry replaced by default anatomy |
| Named semantics | A permitted prior changes the expected name mapping in a manufactured oriented case | Indistinguishable RV/LV swap remains unresolved; bright BG cannot become RV/LV by appearance alone |
| Q/Z | Existing Q route stays exact; separately approved mixture inverse satisfies stated algebra | h=0, invalid inverse, small-h perturbation, unsupported inferred h |
| Native gradient | Finite complete-cell example has nonzero P∇E, nonzero final-head gradient and actual realized update | Constant image/equal means; cell-constant head features give zero weight gradient despite output force |
| Initialization | First-step zero head can update while upstream detail-stem gradient is zero | Treat zero upstream gradient as proof of permanently frozen network |
| Attainability | Joint correction/descent/feasibility interval nonempty in a known toy | Confident wrong logits/low occupancy make it empty; no success claimed |
| Output scope | Controlled edit preserves RV:LV odds; explicitly measure BG change | Unrestricted gradient changes odds; pure-cell nuisance creates false structure |
| Teacher observation | Supported weighted observation matrix used; rank/sensitivity measured in fixed metric | Missing support loses rank; false weak bridge changes hard admissible set discontinuously |
| Projection/arithmetic | Real identities within dtype/scale tolerances on complete cells | Padding and partial validity alter conservation; finite precision is not exact zero |
| Direct/proposal identity | Same point, metric, tangent and scaling produce equal full parameter gradients | Wrong weights, CE matching, stale base, clipping or changed support break premises |
| Optimizer attribution | Equal complete gradients and equal deterministic state give equal one-step update | Raw gradient projection alone fails under AdamW; rollback must restore state and next-step parity |
| Rare evidence | Separate global and fixed source-stratum KL, support masses, NA inactive groups | Total KL falls while rare proxy KL rises; protected proxy group omits all true RV in toy |
| Feedback | Frozen A repeated relaxation matches closed-form one-shot blend | Repeated copies of a wrong source create fictitious evidence or confidence |
| Inference protocol | Holding one cine fixed while changing other cases leaves frozen prediction invariant | Target-cohort statistics or online updates alter it; declare adaptation |
| Evaluation | Dense raw and selective metrics both use explicit full valid denominators | All-BG and reject-all fail; absent-class false positives and zero coverage remain visible |

Source clone/removal/permutation tests run at the consuming anchor/proposal stage, or in a separately authorized matched retraining comparison. Mutating a TRAIN bank after fitting a frozen bank-free inference model should leave predictions unchanged and is not a failed source-dependence test. Distinguish joint encoding permutation from corrupting one semantic source while holding independent priors fixed. Per-cine optimization must reset model/kernel parameters, optimizer, statistics, prototypes and learned caches; test A alone, B then A, repeated A and process reset under coupled RNG.

Counterexamples already exist in probes/R1–R6 and probes/coordinator. They are mathematical/software examples, not another trainer or a representative cardiac benchmark. Do not tune a new candidate against the few known examples and then claim general validity; predeclare broader manufactured families and hold some configurations for falsification.

## Matched controls and accounting

Keep architecture, initialization and frozen guidance identical for the first method comparison. Change only the evidence/update under study. Compare full tuning, permanently frozen encoder, decoder warm-up and smaller encoder LR as standard controls after evidence exists. For a guarded update compare the exact same AdamW-proposed ray with backtracking before introducing gradient surgery. Distinguish same data exposure, equal total compute and matched-displacement diagnostics; rejection overhead prevents matching all simultaneously. Raw parameter norms also depend on parameterization, so accompany them with fixed-input output drift and do not call a norm ratio the cause of accuracy.

A fixed proxy acceptance panel is TRAIN optimization data. A diagnostic panel ceases to be independent if used to choose thresholds/checkpoints or revise the design. No true-RV guarantee follows from proxy strata without coverage/density assumptions. Missing strata are NA, not zero loss. A fixed protection panel must still be evaluated after feature-only/semantically inactive candidate updates; an inactive current batch does not disable a panel-defined guarantee. If no such check is performed, record the group as untested for that step. A consumed-but-rejected batch and a rewind policy are different exposure schedules; compare the chosen complete transaction against the matching skip/rewind control and verify the next accepted update. An anchor-relative endpoint cap and cumulative positive debt need different fields and tests if ever selected.

## Stop and hand back

Stop for human direction if naming requires an undeclared source, the evidence needs oracle mixed-cell selection, a simpler established baseline matches it, all apparent gain is confidence/coverage, or novel machinery is justified only by an unverified bound. Do not launch more modules to rescue a negative finding. A narrower useful regime is acceptable if its observable prerequisites and unobservable assumptions are reported honestly.

The human decides the named design and information setting next. Then Phase II produces a reviewed versioned contract and evaluation protocol on a freshly pinned base. This document grants no implementation, production merge or training permission. The later explicit human request authorizes publishing this report on the separate research branch only.
