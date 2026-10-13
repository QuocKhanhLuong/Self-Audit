# Repair decision and bounded next action

**Full 50-epoch run: NO-GO. Actual checkpoint diagnosis and real TRAIN pilot: BLOCKED on 4070 authentication/artifact access.** Passing software tests, more support and falling loss are insufficient grounds to change this decision.

## Executed repair scope

1. `world_left_v2`, opt-in on `run_stable_guidance.py prepare --orientation-policy world_left_v2`. It uses coded native world geometry and the exact bank/native pixel-center transform, records header provenance and abstains for invalid/uninformative geometry. `cardinal_v1` remains the default. A new bank is mandatory.
2. `run_stable_guidance.py train --diagnostics` records CE, target entropy, KL, active-batch fraction, per-class loss and target mass with the unchanged objective and epoch reduction. The default behavior is preserved. There is no extra reference input.
3. `scripts/diagnose_stable_guidance.py` has independent `probes`, `audit` and `metrics` modes. It refuses an existing output directory or output inside a frozen/input run. Its checkpoint mode checks expected hash, frozen bank/student bindings, effective split, preflight TRAIN identities, permissions and prediction-lock manifest identities, then performs bounded CPU forward/autograd checks. It never steps an optimizer. Default budget: one TRAIN patient and one batch, choosing the first eligible block in fixed lexicographic T/Z order without consulting loss or references. This selection is not a population estimate of active-batch frequency. Official TEST arrays are not analyzed; prediction lock checks read manifest identities only.

The native projection is a diagnostic control only. No resolution repair has yet been selected. No source/graph semantic repair is justified from available patient evidence. New architecture or grounding priors remain a human research decision.

## Use after artifact access is restored (NOT RUN on real artifacts)

These are invocation templates, not claims of execution. Run on the machine where the original paths are valid, using the isolated changed source and a diagnostic directory outside the original job. Keep the original job/source/frozen folders read-only. The default checkpoint digest is pinned to this run.

```bash
python scripts/diagnose_stable_guidance.py audit \
  --run /path/to/original/job \
  --out /path/to/separate/forensics_train_one_batch \
  --max-patients 1 --max-batches 1

python scripts/diagnose_stable_guidance.py metrics \
  --evaluation /path/to/original/job/run/evaluation_val_verified_grid \
  --out /path/to/separate/forensics_val_counts
```

`audit` can accept the run directory or launcher job containing `run/`. A complete TRAIN support/schedule review needs `--max-patients 80` (after verifying the actual effective split); one patient is intentionally not presented as the whole cohort. The tool's metrics mode consumes frozen confusion counts and splits aligned/recovery strata without reading masks. It verifies internal bindings but does **not** reexecute voxel-equality proofs, check every TEST prediction payload, reconstruct original raw votes or generate VAL teacher outputs. Those remain separately required real-artifact checks, not hidden capabilities.

Before changing a bank, use the original run-local config and frozen SSL teacher. The geometry option is exposed in the prepare stage, not silently threaded into the full-pipeline/preflight recipe. Do not call the legacy full-run wrapper and assume it has become geometry-v2. Prepare to a new directory with `--orientation-policy world_left_v2`; record the CLI, source hashes and new bank manifest. Do not manufacture missing configuration from current defaults.

## Decision gates in order

**First: restore evidence identity.** Verify RECIPE, original/effective split, exclusion receipts, preflight, checkpoint, frozen banks, exports, locks and frame records. Recompute original metrics from TP/FP/FN and preserve null denominators. Resolve every mismatch before causal claims. The current aggregate checks alone do not meet this gate.

**Second: localize the first failing stage without optimizing against VAL.** On TRAIN, inspect named R0/Q/accepted support, permissions, weighted soft support, candidate caps/conflicts and the exact active schedule. On one unchanged fixed TRAIN batch, inspect CE/H/KL and both encoder gradients. On exposed VAL, use a fixed TRAIN recipe and frozen teacher; any newly generated teacher prediction must be isolated and frozen before scoring. Compare R0→Q→student common-grid→native gated and ungated outputs. Keep all patients and report aligned/recovery strata. No reference-derived ROI, class remapping, threshold search or feedback into TRAIN permissions/weights.

**Third: permit at most one matched small real TRAIN pilot only if indicated by those diagnostics.** Predeclare, for example, two identity-selected TRAIN patients, seed 42 (or the locked baseline seed), **32 optimizer steps per arm**, identical initialization/order/batching/compute, baseline versus geometry only. These are proposed budgets, not an executed pilot or a tuned prescription. A geometry arm needs a new bank built with the same teacher/config/input patients. Reuse the identical bank for any later student-only comparison. Record actual steps, finite gradients, target entropy/KL, support, fidelity, native/coarse behavior, runtime and memory. Do not substitute epoch count for a fixed step budget; the current full trainer is not a new 32-step pilot runner.

Geometry is falsified as a useful explanation if the original eligible hypotheses have valid cardinal orientation already, or the repair leaves relevant RV candidates/support unchanged. A transfer explanation is weakened if native and the predetermined projection control closely agree. A fitting explanation is weakened if common-grid weighted KL is already small; an anchor explanation remains if matched student fidelity is adequate but frozen anatomical supervision is wrong. No threshold or pass criterion should be chosen retrospectively to rescue VAL Dice.

Even a technically successful pilot does not establish trustworthy named anatomical supervision. A future full-run decision needs a coherent first-stage attribution, no-GT provenance, retained baseline, demonstrated finite training signal and an explicitly disclosed development-evaluation justification. New semantic grounding beyond the current priors requires the human to choose the research direction.

## What changed in our understanding

The source now shows exactly why three superficially similar observations differ: hard bank abstention can coexist with soft training mass; CE near one can coexist with near-perfect fitting of uncertain targets; and identical common-grid semantic loss can coexist with different native predictions. Geometry has a concrete, local mathematical defect with a tested opt-in repair. These findings change which measurements must precede another run. They do **not** yet identify the dominant failure in the completed ACDC experiment.

The immediate human action is to restore key-based read access to 4070 or provide the original exported artifact location. At the original analysis handoff these files were local/uncommitted. The user subsequently authorized pushing the experimental branch; this does not authorize a new full run or a main merge. See the [runbook](../../docs/guidance_forensics_runbook.md) for executable diagnostic commands.
