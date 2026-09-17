# Luna auditor/trainer bulk-audit integration

Status: **implemented and focused-equivalence tested on the released local
baseline**.  This is CPU software evidence only: no throughput, CUDA, VRAM,
real-ACDC/M&Ms, segmentation, clinical, or scientific-improvement claim is
made.  The root `BASELINE_CAPTURED` gate authorized the production changes;
the profiler's paired timing window remains a separate gate.

## Integration contract

`audit_bank` remains the independent scalar reference, including its original
fit/score order, two-round slot schedule, threshold (`gain >= threshold` and
`gain > best_gain`), regional guards, semantic handling, labels, validity and
trace fields.  The additive `audit_banks` entry point validates every unit,
flattens the unchanged per-unit banks in BxK order, invokes the canonical W1
`fit_many` and `score_many` APIs once each, reconstructs each unit's logical
round/evaluation order, and reuses one shared decision path.  Canonical bulk
dispatch is opt-in only: exact `ObservationModel`/callback identities are
required; subclasses, instance overrides, custom callbacks and legacy
`Components` stay scalar.

For canonical regional scoring, each frozen fit is prepared once and each
selected-region reduction uses `score_region`.  The scalar path never uses the
prepared cache.  Per-unit traces retain the original logical
`score_calls`/regional counts; physical work is explicit and separate.  Shared
`fit_many`/`score_many` call/item counters are recorded once by the trainer,
while regional prepared/reduction counters are scoped to each unit.  Producer
features are detached and transferred to CPU once for the complete physical
batch; bank generation and all candidate/round budgets are unchanged.

The preflight receipt now persists both `audit_counters` and a filtered
`physical_audit` mapping so a file-backed gate can distinguish logical audit
budgets from physical API work.  Checkpoint identity, RNG state, sampler
permutation, loss paths, accumulation, component stepping and exact resume
source-hash policy are unchanged.

## Focused evidence

Commands run after the root opened the focused test window:

```text
PYTHONPATH=.:src pytest -q tests/test_maskfree_auditor_fast.py tests/test_maskfree_trainer_performance.py
13 passed in 7.60s

PYTHONPATH=.:src pytest -q tests/test_maskfree_hypotheses.py tests/test_maskfree_trainer.py tests/test_maskfree_observation_fast.py
18 passed in 2.62s
```

The new checks cover: scalar-vs-bulk candidate IDs, labels, probabilities,
validity, selected/accepted/rejected/semantic fields, score-search traces and regional
values; exact binary threshold/tie and below-threshold behavior; mid-budget
connected-region exhaustion; a real producer/student neural batch comparing
detached targets, producer/student losses and metrics, pre-optimizer gradients,
and post-step model state; canonical interrupted/resumed continuation; and a
durable physical-batch-8/image-128 canonical preflight.  The latter records 8
audited units, 32 unchanged candidates, one shared `fit_many` call with 32
items, one shared `score_many` call with 32 items, 32 prepared scores, and
nonzero regional reductions in both the returned and persisted receipts.

Observation reductions are float64 and may differ only at reduction-rounding
scale: the focused accepted-edit fixture's largest fit/score scalar difference
was `1.14e-13` (the regional margin/validity tensors were equal), attributable
to batched reduction order and below the `3e-10` test bound.  Producer/student
outputs, losses, detached targets, pre-step gradients and post-step model
tensors were bitwise equal on the synthetic 32x32 fixture; the canonical bulk
resume test also found exact equality for serialized models, optimizers, RNG,
sampler cursor and scientific history fields (the expected `resumed_from_batch`
operational marker is the only intentional history difference).  These are
bounded local equivalence results, not a performance result; the paired
before/after profile and any real-data execution remain pending or unavailable.

## Owned files

* `src/self_audit_maskfree/auditor.py`
* `src/self_audit_maskfree/trainer.py`
* `tests/test_maskfree_auditor_fast.py`
* `tests/test_maskfree_trainer_performance.py`
* this report
