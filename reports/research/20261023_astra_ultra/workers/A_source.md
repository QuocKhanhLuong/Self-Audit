# Worker A — source, state timeline, and numerical mechanism audit

Draft v1, 2026-10-23. Authoring task: `/root/astra_source`; delegated read-only data-contract check: `/root/astra_source/data_contract`. This is a report-only audit. No production/config/baseline edits, patient training, GPU jobs, weight downloads, commits, or pushes were performed by this worker.

## Decision

**Continue only as a controlled study of useful, audit-constrained pseudo-updates. Do not enlarge the architecture first.** The active method is a new semi-supervised pipeline with a conventional native decoder. The historical no-GT/semantic-identifiability objection and exact D/U proposal-gradient equivalence argument do not directly adjudicate this pipeline. Dense labels now supply anatomy semantics. Its remaining scientific question is whether a joint allocation of class-partitioned EMA pseudo-gradients yields useful patient improvement, after paying for audit labels and computation, beyond simpler scalar guarded pseudo-updates and extra supervised optimization.

The strongest concrete blocker found here is a **reproduced numerical liveness defect**: the unscaled SLSQP objective can terminate near zero while a materially better feasible solution exists. This does not violate the actual-risk guard; it can silently suppress the intervention whose value is being tested. The same solver code appears at both pinned commits.

## Evidence and pins

- Canonical source: `637f38ced217fe1db2347a87924695533056d8f0` (`main` at audit start), research worktree `/Users/alvinluong/Self-Audit-research-20261023`.
- Experimental source compared read-only: `8960b6baac4c6600553d900c303f192c2fcfa4f0`, `origin/experiments/acdc-v2-vram24` as supplied by coordinator.
- Historical archive is explicitly `924265f7df604d3dca84dda49578b1a0cf28d821`; `AGENTS.md:1-20`, `ACTIVE_PIPELINE.json:1-20`, and `docs/METHOD.md` define the changed setting. Prior council memory was used only to avoid carrying obsolete no-GT conclusions into v2; current findings come from these Git sources.
- `VERIFIED_SOURCE`: direct source/Git inspection. `EXECUTED_SYNTHETIC_CPU`: bounded local tensors/vectors, never patient or GPU evidence. `DERIVED`: consequences of those contracts. `REPORTED`: retained documents/receipts not independently rerun here. `NOT_RUN`: missing scientific/target-hardware verification.
- Reproducible probe: [`../probes/A_source_probes.py`](../probes/A_source_probes.py); receipt: [`../evidence/A_source_probes.json`](../evidence/A_source_probes.json); terminal log including finite deprecation warning: [`../evidence/A_source_probes.log`](../evidence/A_source_probes.log). Python 3.11.16, Torch 2.14.0, SciPy 1.17.1, NumPy 2.4.6. Script hash is stored in its JSON receipt.

## 1. Actual information and state timeline

| Stage | Current implementation | What it establishes |
|---|---|---|
| Initialization | `self_audit/model.py:111-131`: local user-supplied pretrained file hash, local backbone Git commit, tracked-clean source check; no random patient fallback. | `VERIFIED_SOURCE`: reproducible local identity. It is not an independent upstream provenance certificate. |
| Supervised reference | `engine.py:62-73,127-129,154-155`: native CE+present-class Dice, global norm clipping, AdamW on all non-head parameters, momentum/decay-free SGD on affine head. | Names and boundaries come from budgeted TRAIN GT, including audit patients. The supervised step itself is outside the pseudo-step guard. |
| Teacher target | `engine.py:125,163-165`: frozen EMA teacher, clean U input, detached probabilities, evaluated after the supervised reference but before EMA update. | The teacher state is the previous finalized iteration. Its target is endogenous, not independent correctness evidence. |
| Fixed directions | `control.py:132-159`; `decoder_control.py:22-46`: one source per accepted pseudo-argmax class, all using the same accepted-pixel denominator. | At most four canonical directions. Summing them gives ordinary SGD on the same hard-pseudo CE and parameter block. |
| Local allocation | `control.py:22-61,120-129`: classwise probability-response moments, ridge-regularized box-QCQP, numerical feasibility checks. | Local Brier proxy; convex mathematical problem, numerical solution quality not certified. |
| Actual trials | `control.py:205-224`; `decoder_control.py:67-79`: direct functional parameter trials, finite predictions, per-class and mean-risk checks, backtracking. | Bounded extra pseudo-displacement on current TRAIN audit measure only. No subsequent AdamW transformation of that displacement. |
| EMA | `engine.py:191-192`, implementation `54-59`: update once after final student state; buffers copied. | Rejected candidate steps do not update the teacher, but the supervised reference still updates EMA normally. |

The one shared timeline is genuinely implemented. No source path in this audit revived frozen-Q/CineMA-bank training. `model.py:48-81` implements ordinary multiscale decoder refinements plus native image detail, not the old exact D/U architecture.

`DERIVED`: For head scope, conditional on frozen features, the logits are affine in the candidate parameters, so the analytic logit response is exact and the softmax response is first-order (`control.py:179-193`). Decoder scope updates lateral/up/refine/detail/fuse/head parameters but excludes encoder and input adapter (`decoder_control.py:22`); its probability response is a true parameter JVP and its finite update remains nonlinear. These scopes are different interventions. A head-only success cannot establish that U learned useful native image features.

`VERIFIED_SOURCE`: supervised learning updates encoder and detail features (`engine.py:127-129`; `model.py:54-59`). `tests/test_pipeline.py:21-26` checks gradients for a manufactured model. The nearby-slice input is spatial 2.5D: `data.py:75-77` selects `z-1,z,z+1` at fixed time `t`; it is not temporal context.

## 2. Objective and denominator alignment

The pseudo directions use one global accepted-pixel denominator, with no inverse-frequency amplification (`control.py:140-158`; decoder equivalent `decoder_control.py:24-43`). `tests/test_control.py:73-86` checks the head sum against ordinary CE gradient and analytic JVP against autograd. My tiny decoder probe independently found maximum parameter error `7.45e-9` between the production `ema_block` transaction and one direct selected-block SGD update. This is `EXECUTED_SYNTHETIC_CPU`, not evidence about clinical data or DINO execution.

Audit risks include negative pixels and explicitly log absent positives (`control.py:83-117`). Present classes give positives and negatives half the mass each, then each batch element equal mass. The non-padded native main path has one audit slice, so “patient-normalized” means one slice from one patient per transaction. Experimental B52 draws many slices from the same audit patient, so equal batch-element mass means equal sampled-slice mass, not 52 independent patients. It is not a volume-level prevalence measure.

`VERIFIED_SOURCE`: `data.py:82-90` uses GT to foreground-enrich labeled/audit slice sampling with probability .5. This stays within the labeled budget but further differentiates the audit measure from a uniform full-cine metric. Supervised CE is pixel-normalized, present-class Dice is aggregated across its batch, audit Brier is per-slice balanced one-vs-rest, and pseudo CE is accepted-pixel-normalized. Those are intentionally different objectives, not interchangeable denominators.

At experimental `8960b6b`, `crop_native` pads labels with 255 and creates image-geometry-only U validity (`self_audit/data.py:21-41`), and that validity reaches all pseudo arms (`engine.py:174-195`; `control.py:143-146`; decoder counterpart `26-30`). No U GT is required to exclude padding. This fixes support handling but creates a declared crop/batch experimental factor. Native single-slice main results cannot simply be pooled with C256/B52 results.

## 3. Priority findings

### P1 — Small-response SLSQP stagnation is reproduced

Location: `self_audit/control.py:44-60`. The solver begins at zero, uses an absolute `ftol=1e-12`, and checks feasibility plus improvement against zero after optimization. Those are safety checks; they do not certify near-optimality or a useful nonzero step. A `NUMERICAL_FEASIBLE` status is consistent with avoidable zero movement.

For consistent scalar moments `r=1, v=1e-6`, hence `A=1e-12, b=1e-6, e=.5`, with `ridge=1e-6`, the exact scalar specialization gives `alpha=0.999999000001`. SLSQP returns `alpha=1e-6`, status `NUMERICAL_FEASIBLE`, leaving an objective gap about `4.999985e-7`. For `v=1e-7`, the exact alpha is about `.1`; SLSQP returns zero. These are feasible squared-response moments, not arbitrary inconsistent matrices.

The receipt also constructs an actual finite affine-head example, avoiding reliance on boundary residual `r=1`: zero affine logits give `p=.25`, all pseudo and audit labels are class 1, constant finite features, and `pseudo_lr=1e-6`. Real CE directions and analytic softmax responses give exact scalar alpha `.11458332677`, while the joint solver returns zero and performs no trial. At larger tested rates `1e-5` through `1e-3`, this simple fixture succeeds. Thus the defect is scale-dependent, not a claim that all ordinary runs stall. A probe-only equivalent reformulation subtracts the constant baseline and divides the whole objective, its gradient, constraints and constraint Jacobian by the same positive scale, preserving ridge and zero budgets. It returns `.11458333333` for that finite-head case and near-exact scalar solutions throughout the sweep. This separates numerical scale sensitivity from an actual guard conflict; it is not a production fix or general solver validation.

**Consequence:** `DERIVED`, a large-scale joint-versus-independent comparison can confound allocation geometry with unequal numerical solver behavior. `independent_guard` uses analytic scalar coefficients; joint uses SLSQP. Small direction size can arise from a chosen small step, small accepted class fraction, saturation, or weak audit response. Whether this actually occurred in retained patient runs is `NOT_RUN`; do not attribute reported patient outcomes to it without saved moments.

**Cheapest falsification/fix gate:** add a response-scale sweep against analytic scalar answers and 2–4-source independently solved tiny cases; record objective gaps against exact or independently verified feasible comparators. KKT residuals are only informative under verified constraint regularity: zero-budget quadratic constraints can force a singleton feasible set with zero constraint gradients, so a true optimum need not admit ordinary KKT multipliers. A box-only projected-gradient residual is not a stationarity certificate for this constrained QCQP. Consider subtracting the constant baseline and scaling objective/constraints by an explicitly recorded positive scale. Keep the mathematical ridge penalty and budgets unchanged. A successful numerical change must preserve the existing problem and actual guard; it is not evidence for new modeling efficacy.

### P1 — Plain decoder baseline pays for unnecessary joint machinery

Location: `decoder_control.py:36-57`. Even `ema_block`, whose alpha is unconditionally all ones, constructs separate per-class gradients, computes every JVP, and forms all moments before selecting that constant alpha. `predict` calls the complete model (`50-54`), so each JVP and each actual trial reruns the unchanged encoder. GroupNorm has no train/eval running statistics, and the transaction uses eval mode, so caching fixed encoder/input-adapter outputs is a promising exact refactor, subject to parity tests.

`EXECUTED_SYNTHETIC_CPU`: with four classes, the tiny-model `ema_block` call performed seven encoder forwards: audit reference, U forward, four JVP calls, one final trial. Its committed parameters matched direct one-backward selected-block SGD within `7.45e-9`. In the current receipt the transaction took `.1312 s`, direct SGD `.00529 s`; the direct comparator omits audit diagnostics, and this ratio is **not** a GPU throughput estimate or a production benchmark.

`DERIVED`: an efficiency claim comparing full-EMA runtime with this implementation of plain block-EMA is contaminated by removable diagnostic work. For guarded decoder methods, caching unchanged encoder outputs could also save repeated work, while streaming moment accumulation could reduce transient output-memory usage. Preserve both fixed-step and compute-accounted comparisons, and distinguish optimized baseline cost from a deliberately diagnostic path.

### P1 — `ema_full` does not isolate the effect of parameter scope

Location: `engine.py:166-175` versus `177-189`. Full EMA calls the shared body AdamW and head SGD optimizers, uses `body_lr/head_lr`, adds an optimizer step, and clips the pseudo gradient. Block EMA uses direct raw SGD directions at `pseudo_lr`, without clipping, on head/decoder only. Defaults are body LR `3e-5`, head LR `3e-3`, pseudo LR `1e-3` (`engine.py:19-23`). Thus optimizer family, step size, clipping, optimizer-state evolution, and parameter scope all change.

This is a valid stronger package comparator when declared as such; it cannot isolate “U-to-encoder benefit” or prove that allocation alone causes a gap. Add an equal-update-rule full-block comparator only if that causal question matters. Do not weaken full EMA merely to favor the proposed method.

### P2 — `joint_no_class` removes local constraints only

Location: `control.py:202-217`; `decoder_control.py:64-74`. Its local SLSQP constraints are disabled, but it remains in the `guarded` set and every real trial still enforces per-class budgets. The method name is ambiguous if a results table calls it “without per-class protection.” It is a clean ablation of **local QCQP constraints while retaining actual per-class protection**, which should be its explicit description.

`joint_no_guard` is the complementary ablation: it retains local class constraints but disables the actual finite-step guard. `independent_guard` uses independently computed scalar coefficients, then validates their combined update; it may reject interactions even when each source separately passes. These distinctions must be visible in the paper's arm table.

### P2 — Experimental matrix regeneration misses C256/B52

At experimental `8960b6b`, `configs/pilot.json:27-28`, `configs/full.json:27-28`, and `docs/RUNBOOK.md:136-138` declare `batch_size=52,crop_size=256`, while `experiments/plan.py:17-22` copies engine defaults and overrides only method/scope/steps/warmup/seed. New defaults are B1/crop0. Regenerating the planner therefore yields native single-slice jobs. Retained `results/configs/*` use C256/B52, so this is a reproducibility/planner defect, not evidence that already-retained runs were mismatched. It should be repaired before any newly generated experiment matrix is executed.

### P2 — Manifest identity is stronger than arbitrary path/hash alias handling

Canonical preparation excludes U masks and charges all audit labels to L (`prepare_acdc.py:64-93`, `data.py:37-44,94-98`). However, `TrainingData` trusts the supplied manifest's cohort/subset declarations instead of recomputing membership from a locked split and declared subset seed (`data.py:31-38`). Its array cache keys by path and skips hashing on a cache hit (`data.py:65-72`), so a second record with the same path and different expected hash can bypass that record's hash check. The global duplicate-declared-hash test (`46-48`) does not eliminate inconsistent aliases.

These are fail-closed provenance-hardening opportunities, not a demonstrated canonical U-mask leak or evidence that retained datasets were cherry-picked. Cheapest tests use tiny temporary arrays/manifests, not patient data: inconsistent path/hash reuse, different paths to identical arrays, nested subset reconstruction, and split identity mismatch.

## 4. Rejection atomicity and remaining limitations

`VERIFIED_SOURCE`: the head trials operate on cloned weights with `F.conv2d`; decoder trials use detached cloned selected parameters and cloned buffers via `functional_call`. Actual writes occur only after acceptance (`control.py:222-224`; `decoder_control.py:78-79`). No optimizer or teacher object enters either control function. `autograd.grad` does not step the live optimizers. Current eval-mode native modules do not require stochastic trial RNG.

`EXECUTED_SYNTHETIC_CPU`: the head harmful-direction zero-allocation path preserved parameters, populated AdamW state, and Torch RNG exactly. A separate decoder fault injection forced alpha=1 for a harmful background pseudo-direction and tested three real nonzero proposals; all were rejected, parameters/buffers and RNG stayed exact, and an external empty optimizer state remained untouched. The latter validates rejection mechanics under a deliberately faulty allocator, not that the natural optimizer emits that direction or that populated decoder optimizer state has been empirically exhaustively covered.

`DERIVED`: future stochastic/stateful modules need fresh checks; “functional” calls alone do not imply universal atomicity for arbitrary module side effects. The initial reference and candidate forwards call the live model. No arbitrary-module guarantee is claimed here.

`VERIFIED_SOURCE`: EMA is initialized from the initial model before warmup, then updated once each step (`engine.py:125,191-192`). The default eight-step pilot warmup and decay .99 leave substantial initial-state contribution; full recipes use a longer warmup. This is ordinary EMA lag, not a coding error. Target entropy, accepted-pixel coverage, source-class occupancy, coefficient magnitude, backtracks, and final parameter-displacement norm should be logged to distinguish bad-teacher/no-candidate/numerical-zero/geometric-rejection causes. `pseudo_accepted` currently counts `scale>0`, so it need not imply a measurable nonzero committed displacement.

The actual-risk contract protects only current TRAIN audit Brier relative to the supervised-only reference. It does not certify the preceding supervised step, later iterations, the EMA predictor, unseen patients, per-patient Dice, or anatomical plausibility. Audit labels are optimization labels; reusing them across steps cannot yield a held-out generalization certificate. Full native evaluation differs from both foreground-enriched audit sampling and experimental training crops.

## 5. Defensible next direction and cheapest sequence

1. **Numerical and arm-definition gate first.** Resolve the small-response solver defect under exact scalar/tiny-vector tests; make local-vs-actual constraint ablation names explicit; fix planner protocol reproduction. Retain zero/rejection outcomes and add displacement norms, source moments, and solver residuals to future authorized telemetry. No patient rerun is justified solely by passing CPU tests.
2. **Mandatory global-ray comparator.** Use `d=sum_j d_j` with one scalar beta in `[0,1]`, the same audit groups/budgets and actual guard. Its exact local moments are `a_c=1' A_c 1` and `b_c=1' b_c`. To be the literal restriction of the current vector objective to `alpha=beta*1`, use scalar ridge `M*lambda`, not silently `lambda`. This comparator needs one decoder JVP instead of M and tests whether joint class allocation buys more than safe global step sizing. Treat a differently penalized scalar baseline as a declared variant.
3. **Replay before long training.** On saved frozen TRAIN transactions only, compare actual accepted displacement, mean/per-class Brier, prediction disagreement, joint-vs-global-ray gain, numerical failure rates, and time. Independent TRAIN audit patients not used for that transaction's allocation can diagnose transfer; they still consume the same declared labeled budget and do not become an independent final test. Abort a joint-allocation efficacy story if coefficients mostly vanish, useful interactions are rare, or the scalar comparator matches the gain at materially lower cost.
4. **Then a small matched patient experiment, only under separate authorization.** Match label budget, patient subset, source snapshot, pretrained weights, schedule, warmup, input support, pseudo denominator, parameter block, and update rule. Include supervised, extra-supervised, plain block EMA, guarded global ray, independent sources, joint sources, and the separately declared strong full-EMA package. Report volume/patient/class metrics and actual compute. Preserve exposed VAL as development and keep any genuinely unexposed evaluation frozen.

The possible contribution is a demonstrably useful **finite-step, class-conditioned allocation policy** under a stated TRAIN audit measure. Classical EMA, convex QCQP, ridge regression, 1x1-head JVP, native decoder, or a safety check in isolation is not enough. The burden is incremental useful update and patient improvement relative to simple matched controls, not a more elaborate decoder diagram.

## Revision record

- v1: initial source dossier, numerical counterexample, bounded atomicity/cost/equivalence probes, and delegated data-contract findings. Independent cross-review pending. No production correction is claimed.
- v2: added probe-only equivalent numerical normalization and reran the bounded probes; refreshed timing values to the retained receipt. Independent review of F requested terminology precision about volumetric frame metrics and training versus subset seeds; F accepted both amendments.
- v3: accepted B's constraint-qualification challenge and replaced the unqualified stationarity suggestion with exact/independent objective comparisons; B independently reproduced the solver failure on finite head probabilities. Numerical gap remains a liveness issue, not an actual-risk guard violation.
