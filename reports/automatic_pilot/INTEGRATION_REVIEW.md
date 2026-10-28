# Integration review

Coordinator review of agy/Antigravity worker implementations, starting from
`bdb1c399d29c63ae49865af8db78954f79f9ad9d`. This file records software review;
it is not evidence of patient Dice, GPU performance, or novelty.

## Findings requiring correction before delivery

1. **Cached/actual argmax disagreement.** A deterministic FP32 example has
   cached/actual logit error below `1e-5` but different winning classes.
   Allclose parity alone cannot certify hard Dice. Finite actual functional
   logits must pass Brier and Dice/absent-FP predicates before direct commit.
2. **Rejection mutation.** Copying unchanged baseline parameters on rejection
   increments tensor version counters. Rejected trials must perform no writes
   to live parameters and preserve gradients, optimizer state and RNG.
3. **Aggregation.** Slice balancing and pooling all patient voxels change the
   objective. Brier weights and Dice reductions must explicitly use complete
   frame volumes, then frames within patients, then patient macro. Overall
   foreground evaluation averages available classes within each patient.
4. **Incomplete cache identity.** Weight-only fingerprints omit executed input,
   label, geometry, validity, view and metadata changes. Aliased metadata copies
   also defeat invalidation. Bounds must account for padded cache/workspace.
5. **Incomplete initialization/resume contract.** Every arm needs the same
   externally hash-pinned support-only warm-start phase. Resume must restore
   both optimizers, the actual local generator, counters and fixed support,
   and reject changed arm, data or algorithm configuration.
6. **Teacher feature drift.** Head adaptation freezes student and teacher
   features; head-only EMA occurs once after the final student state. A
   separate full-EMA control explicitly updates the feature path.
7. **Unverified reporting.** Inferred latency percentiles, invented checkpoint
   payloads and accepted-count-as-displacement are invalid receipts. Missing
   values remain unmeasured. Equal-time requires an enforced measured budget.
8. **False research GO.** Four Dice numbers without verified independent query,
   coverage, initialization, label lineage and compute matching cannot yield GO.
9. **Evaluation isolation.** Export and hash all raw-student native predictions
   before reference I/O; require complete TRAIN-L/U lineage, canonical bounded
   frame IDs and new safe output locations. Undefined scores remain undefined.

Final disposition and executed checks are recorded in the delivery receipt.
Worker-reported test counts are provisional until coordinator verification on
the final integrated source.
