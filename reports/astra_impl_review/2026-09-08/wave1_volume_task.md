# W1.3 — volume identity / metric-space safety (TASK SPEC, not implemented)

Scope: Proposal 1 correctness only. Implementer AGY via Orca, reviewer Astra.
Run only after revision 2 is reviewed. No architectural/objective changes, training,
commits/push, docs.md edits, or historical-report overwrites. Preserve others' edits.

## Objective and evidence

`threshold.py` currently emits pooled `volume_macro_dice` when cache has statistics
but lacks case IDs. Independent fixture: two rows with TP/FP/FN for class 1
(1,0,0) and (0,9,9), Q=(1,0), unchanged proposals, delta_q=1, tau=0.
No case IDs yields volume=.1/count=1; IDs case1/case2 yield volume=.5/count=2.
Missing identity is not evidence of a single volume.

Collector calculates Q for each 2-D slice even when given a volume contract.
Do not label those Q values volume scores. Slice cache calibration remains diagnostic.

## Expected files and minimal implementation

- `evaluation/threshold.py`, `training/finetune_joint.py`, `evaluation/contracts.py`
  only where needed; cache/calibration CLI wiring and focused tests.
- No implicit cohort pooling. Without valid case identity, either omit volume outputs
  with explicit unavailable reason, or fail if caller explicitly requests volume scoring.
  Do not fabricate identifiers. Explicit single-volume identity is acceptable.
- Emit distinct, correct contract metadata for slice replay and any derived resized-volume
  metrics. Reject volume_native claims from this network-grid slice cache.
- Reject a volume contract as the contract of per-slice Q; do not silently reinterpret it.
- Statistics used for volume aggregation must be a complete aligned TP/FP/FN bundle:
  exact N,C and N,T,C shapes, supported class indices, finite nonnegative integer counts.
  Reject partial bundles and inconsistent statistics vs Q under the declared slice contract.
  Do not truncate fractional counts to int to make scores appear valid.
- Keep existing legitimate per-case aggregation. No new volume-calibration algorithm
  or geometry refactor here; unsupported use must fail clearly.

## Invariants

Reject -> HALT, strict global gate, stop-gradient, training targets/losses, proposer,
feedback, data/splits, weights, and interpolation remain unchanged.

## Mandatory tests / pass criteria

1. Above two-case fixture: IDs -> .5/count2, no IDs never -> asserted volume .1/count1.
2. Same two cases direct grouped statistics == replay selected grouped statistics.
3. Collector rejects volume_native and volume_resized per-slice target contracts.
4. Partial stats, NaN/inf/negative/fractional counts, wrong class dimension and Q/stat
   inconsistency rejected. Valid t_max=0 and blank->hallucination remain supported.
5. Distinct slice and derived-volume metric metadata asserted.
6. Focused tests then full tests/ and source compile; exact commands/counts/output.

Return files/diff rationale, test output, unresolved concerns in a new worker report.
Do not claim patient protocol/real-checkpoint validation from these synthetic tests.
