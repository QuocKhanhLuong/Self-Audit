# Maskfree150 performance refactor — independent red-team review

Status: **final v2 paired receipts accepted for synthetic CPU equivalence; no GPU/clinical conclusion** (2026-09-15).

Reviewer scope is read-only scientific/performance-contract review. The only
owned path is this report. No production code, configuration, test, checkpoint,
run artifact, or benchmark output is modified by this review. The coordinator
owns integration and the final decision; this report is not an authorization to
run the 150-epoch experiment.

## Evidence boundary

The checkout is `/Users/alvinluong/Self-Audit` on `main`, with unrelated dirty
work preserved. `reports/maskfree150/performance_baseline_5070ti.md` records that
the requested RTX 5070 Ti/GPU baseline was blocked before any batch was measured.
Therefore no GPU speedup, throughput, memory, or determinism claim can be made
from this checkout. Synthetic CPU checks can establish software equivalence or
firewall behavior only; they cannot stand in for real-GPU evidence. I will not
run CPU timing/profiler tests while the coordinator's profiler baseline is
active.

## Invariants to preserve

The performance refactor is admissible only if it leaves these scientific and
operational invariants unchanged:

| Area | Required invariant |
|---|---|
| Protocol and shape | physical batch 8, accumulation 1, image size 128, deterministic FP32 mode, seed 42, full 150 epochs; any fallback is explicit and retains effective-batch provenance |
| Candidate budget | exactly four candidate fits per unit (including incumbent), two rounds partitioning challenger slots, and at most five nuisance-fit iterations inside each fit; on a full batch this is up to 32 candidate fits and 160 fit iterations, not eight candidates or an undisclosed `4 x 2` multiplication |
| Score budget | full regional budget and fixed candidate capacities/background modes; retain logical score-call counts even where physical work is cached |
| Data firewall | producer/proposals/nuisance fit use only `O_fit`; selection is adaptive evidence only; `O_verify` and reference masks remain sealed until every compared prediction/checkpoint is frozen |
| Snapshot identity | fit snapshots deep-copy candidate tensors and bind hypothesis digest, source/unit/partition, model recipe, fit support and source version; mutation or stale cache reuse must fail closed |
| Students | two separately stored, byte-identical initial students; same bank, initialization, augmentations, and budgets; no student-to-producer or cross-arm leakage; losses and skipped-invalid accounting preserved |
| Numerics | float32 inputs/outputs, fixed variance floor and empty-region fallback, optional background-2 capacity matched across candidates, deterministic tie/argmax ordering, exact padding/odd/singleton fallbacks |
| Resume/provenance | batch-boundary resume preserves partial accumulation, RNG, sampler cursor/permutation, model/optimizer/scaler, source/config/cohort/partition identity, counters and committed log offsets; mismatch fails closed |
| Validation/freeze | immutable epoch snapshots and freeze manifests are rehashed and identity-checked; validation uses frozen snapshots, never `last.pt` or reference feedback; selection and verify remain separate |
| Measurement honesty | benchmark includes all declared warmup/measured batches, short/invalid batches and both student arms; no cherry-picking, skipped work, hidden cardinality or reduced scientific budget; stage timing is not summed when nested/inclusive |

The current contract additionally requires preserving the exact auditor decision
boundary: `gain >= improvement_threshold AND gain > best_gain`, including first
candidate tie behavior. Regional margins compare normalized NLL only; complexity
and prior remain in the global score. If a regional budget ends mid-region, any
already-scored best gap is retained. Logical calls must remain accounted for even
when a cache avoids duplicate physical score work.

## Adversarial matrix prepared before implementation

The following checks are planned for the integrated final diff. A check is not
marked passed until it is independently reproduced against the final tree and
the exact command/result is recorded below.

1. **Vectorized equivalence.** Compare old/reference and optimized observation,
   audit, or model paths on identical detached FP32 fixtures, including outputs,
   gradients where applicable, selected candidate ID, masks, regional margins,
   counts, and trace ordering. Cover the production 128-even ladder plus odd,
   non-square, 2x2, 1xN/Nx1 and singleton fallback shapes. Check deterministic
   repeated runs and CPU/GPU-independent ordering without timing them.
2. **Frozen snapshot and cache identity.** Mutate caller hypotheses, fitting
   tensors, source-version markers, unit IDs, epoch IDs, partition IDs and model
   recipes after caching. A stale or cross-unit/epoch/source cache must be
   rejected or recomputed; a valid cache must not alter logical-call counters,
   candidate order, fit count, or output hashes. Exercise cache reuse across
   both students and repeated epochs.
3. **Observation numerics.** Exercise `background_modes=1` and `2`, empty and
   sub-threshold regions, variance exactly at/below/above the floor, constant
   images, all-background and all-invalid supports, padding boundaries, and
   finite/non-finite inputs. Compare fallback counts, variance, density, NLL
   denominators and unavailable reasons to the reference path; ensure no NaN or
   silent zero metric appears.
4. **Discrete selection and budget.** Test threshold equality, just-below and
   just-above gains, exact ties, first-candidate ordering, four-candidate/two-
   round slot assignment, full regional score budget, mid-region exhaustion,
   unchallenged/unobserved regions, and preservation of normalized-NLL-only
   regional gaps. Confirm no extra round, refit, or challenger starvation.
5. **Resume/checkpoint provenance.** Interrupt at an accumulation boundary and
   compare resumed versus uninterrupted state/history/selection for exact parity.
   Alter source/config/cohort/partition/model recipe, cache version, checkpoint
   digest, or committed-log offset and require fail-closed behavior. Verify a
   stale rolling checkpoint cannot substitute for an immutable validation
   snapshot or final freeze identity.
6. **Validation and GT firewall.** Mutate selection and verify/reference inputs
   independently. Selection changes may affect selection scores only; verify or
   reference mutations must not affect training, proposals, selection,
   checkpoints, or frozen identities. Validation must rehash the same frozen
   snapshot and preserve select/verify separation. No reference read may happen
   before complete freeze.
7. **Benchmark integrity.** Audit the final benchmark manifest for exact batch
   and unit IDs, five warmups plus all declared measured batches, short/invalid
   rows, skipped student updates, fit/iteration/whole/regional score counts,
   both student arms, and declared batch/image/candidate/round/fit budgets. Any
   cherry-picking, unreported skip, changed cardinality, reduced regional
   budget, profiler-overhead conflation, or synthetic-CPU-to-real-GPU claim is a
   review failure.

## Final review evidence (current live tree)

The coordinator reports the final combined maskfree suite passed (`164 passed,
2 CUDA skipped`) with `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:src python -m
pytest -q tests/test_maskfree*.py`; this is software evidence, not GPU or
clinical evidence. My focused red-team runs against the current absolute tree
were:

```text
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:src rtk python -m pytest -q \
  tests/test_maskfree_ontology_fast.py tests/test_maskfree_observation_fast.py \
  tests/test_maskfree_auditor_fast.py \
  --basetemp=/private/tmp/selfaudit-luna-redteam-1510
38 passed, 1 skipped in 1.29s

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:src rtk python -m pytest -q \
  tests/test_maskfree_performance_profile.py \
  --basetemp=/private/tmp/selfaudit-luna-profile-1510
7 passed in 11.57s
```

After the density gather change, my independent focused run was:

```text
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:src rtk python -m pytest -q \
  tests/test_maskfree_observation_fast.py \
  --basetemp=/private/tmp/selfaudit-luna-final-density
23 passed in 0.69s
```

The coordinator additionally reports 84 independent direct helper edge cases
passing (including empty/singleton, fractional/NaN/Inf/negative/out-of-range
labels) and the combined final suite passing (`164 passed, 2 CUDA skipped`).

The connected-components shortcut has an exact proof under its stated guards.
For each 4-connected component, the maximum raster ID is the unique source;
after `t` synchronous self-plus-four-neighbour maxima, exactly the pixels at
graph distance at most `t` from that source carry the maximum. Thus coverage
after `cap-1` masked cross dilations implies the frozen recurrence reaches its
stable state by the `cap`-th stability check (`converged=True`), while any
failed certificate takes the unchanged NumPy recurrence, preserving the
cap-boundary full-label/False and partial-label/False outcomes. The live code
uses an explicit cross, masked dilation, original raster IDs and the empty-mask
zero/True branch; the coordinator additionally reports 553 bitwise label/flag
cases against the original implementation, including non-contiguous, sparse,
dense and long non-converged masks.

The accepted ordinary paired artifacts also expose an attribution limit: the
baseline reports cache unavailable, while the optimized report has a cache
with 160 measured hits (and zero loads/misses). The ordinary matcher marks all
checks true but does not match cache availability, hit/load counters, or cache
key mapping. That pair therefore supports only end-to-end synthetic-CPU timing
for the source revisions; it cannot isolate the observation bulk path or claim
cache-neutral performance. This is a limitation on attribution, not evidence
of a numerical mismatch.

### Final v2 paired receipts and gate disposition

The final source snapshot is the optimized package digest
`dc19f207f4b5f3ac61a6913bb5b1e6e0fca3a81cb65ffaa4fb795d019f831dfe`.
`after_density_ordinary/matched_report.json` and
`after_density_instrumented/matched_report.json` both report `matched=true`
with all 44 checks true against the retained original baseline receipts. The
source snapshot manifest/observed aggregate digests are non-empty and equal;
the scientific/config hash, fixture content, sample order, image inventory,
initial/final model and RNG hashes, candidate/evaluation/tensor/validity/margin
hashes, effective determinism, AMP state, per-batch student accounting, and
per-unit/aggregate budgets all match the receipt contract.

Checkpoint provenance is also boundedly reproduced: the coordinator loaded
trusted baseline/optimized `last.pt` files at step 25 and found exact equality
for models, optimizers, scaler, RNG (including NumPy), sampling generator and
epoch permutation, epoch/batch cursors, completed-epoch/step counters,
component steps, micro-batches, accumulation-boundary state and pending
microbatches. Expected source/config/run identity and diagnostic floats were
excluded; this is not evidence of cross-source resume migration.

The final ordinary measured-batch p50 is `6.777027750 s` (retained baseline)
versus `2.298810354 s` (final optimized), a measured `2.948058650x` CPU
wall-time ratio; p95 is `7.472338993 s` versus `2.413252386 s`. The retained
v1 intermediate pair was `2.977856021x`, so the density gather refinement does
not demonstrate an additional full-loop speedup and must not be advertised as
one. Final instrumented p50 is `6.893134333 s` versus `2.210947104 s`
(`3.117729194x` diagnostic ratio), but instrumentation is not ordinary
throughput evidence.

The gathered hard-label implementation was reviewed directly: each row's
`[B,R,M]` means/variances/log-weights/normalizers is indexed by `[B,N]` labels,
transposed to `[B,M,N]`, evaluated once by the existing density equation, and
reduced over the unchanged mode axis. Invalid labels are converted to a safe
index, restored to NaN, and rejected; padded modes remain `-inf`; no row or
unit shares component parameters. The independent helper tests above are
bitwise (`torch.equal`) across batch sizes 1/8/32, Gaussian/Student-t,
background modes 1/2, missing labels and strided inputs.

The coordinator supplied all four v1 artifacts under
`reports/maskfree150/performance_cpu/`: baseline/optimized ordinary and
instrumented profiles plus both matched reports. Both matched reports are
`matched=true` with all 44 checks true. The ordinary measured-batch p50 is
`6.777027750 s` (baseline) versus `2.275807729 s` (optimized), a measured
`2.978x` CPU wall-time ratio; it must not be rounded to `>=3x`. Instrumented
p50 is `6.893134333 s` versus `2.227080166 s`; this is a diagnostic timing
mode, not an ordinary throughput claim.

The v1 receipts close the six earlier matcher concerns: canonical bulk state is
false for the baseline and true before/during/after hooks for the optimized
run, with 25 `fit_many`/`score_many` calls and 800 items; every snapshot has a
non-empty manifest/observed aggregate digest with equal values; all available
fit/select/accepted/rejected score records are finite; every measured batch has
one producer loss, two student losses, one optimizer step and zero skips;
per-unit budgets and decision hashes are recorded alongside aggregate totals;
and the effective post-seed deterministic flag is true while AMP is false.
The matcher also compares candidate/evaluation/tensor/validity/margin hashes,
final model/RNG state, source identity, sample order, image inventory and
fixture content. The ontology shortcut proof and 553 bitwise label/flag cases
remain independently recorded above.

### Remaining limits

* The ordinary pair intentionally has different cache capability: baseline
  cache is unavailable, while optimized measured runs report 160 warm-cache
  hits (zero loads/misses). This is expected scope for the combined refactor,
  but the pair cannot isolate observation-vectorization speed or claim
  cache-neutral performance. Root's final prose should say “combined
  end-to-end synthetic CPU refactor” and not an observation-only causal gain.
* The fixture still reports 159 semantic-unresolved units out of 160. The
  result is software/provenance evidence only; unresolved semantics, synthetic
  images and the corrected fixture do not establish clinical validity.
* The final pair is a bounded 25-step run (five warmups plus 20 measured
  batches), not a completed 150-epoch training or validation/freeze run. No
  real RTX 5070 Ti was available, so no GPU speedup, utilization, memory, or
  production throughput claim is admissible. The `2.948x` number is a
  synthetic-CPU, combined end-to-end refactor result only; density-only
  causality and cache-neutral attribution remain unproven.
