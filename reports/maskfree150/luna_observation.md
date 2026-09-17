# Luna observation fast-path implementation and equivalence

Status: **implementation complete on the released local baseline; integration
pending root review**.  This note records the API and equivalence contract for
the observation worker; it does not claim a speedup, GPU evidence, or
real-ACDC/M&Ms evidence.
Only `src/self_audit_maskfree/observation.py`,
`tests/test_maskfree_observation_fast.py`, and this report are owned by this
worker.  Existing W1 equations and all scalar score guards remain the source of
truth; the bulk/prepared paths were added only after the root profiler released
`BASELINE_CAPTURED`.

## Existing equations and guards

For a frozen candidate label map `y(p)`, the fit-normalized intensity is modeled
as a region-conditioned Gaussian (or fixed-`dof=4` Student-t) mixture with a
shared affine bias plane:

```
I(p) ~ sum_m pi[y(p),m] D(mu[y(p),m] + b(p), sigma2[y(p),m])
b(p) = a0 + a1*x_norm + a2*y_norm,   x_norm,y_norm in [-1,1]
sigma2 = max(empirical variance, variance_floor)
```

The fitting loop starts with `b=0` and repeats exactly `max_iterations` (the
maskfree150 default is 5): bias-corrected region statistics, optional
deterministic background two-mode assignment (25/75 percentiles plus three
hard-assignment updates), and one weighted ridge solve for the three plane
coefficients.  Regions with fewer than `MIN_REGION_PIXELS=8` fit pixels use the
fit-global mean/variance and are explicitly reported.  All observation
arithmetic is detached `float64`; neural producer/student paths remain FP32.

The score is retained exactly as:

```
normalized_nll = nll_sum / count
complexity = neighbour_disagreement_fraction + 0.01*(capacity - 4)
total = normalized_nll + beta*complexity + prior
```

`fit` accepts only a validated `FittingView` and never receives scored data.
`score` accepts only a validated `ScoringView`; study, unit, and partition IDs,
fit/scoring support disjointness, full model recipe, frozen support encoding,
and the SHA-256 hypothesis digest are mandatory.  Empty scoring support is
unavailable (`count=0`, `normalized_nll=None`, `total=None`, and the existing
reason), never a numerical zero.  Fitted hypotheses are deep-copied and no
cross-candidate state or stale cache is permitted.

## Exact public bulk API

The scalar methods stay public and authoritative for compatibility.  The bulk
methods are ordered, flat APIs: result `i` always corresponds to input `i`.
Repeated view objects are valid (the normal bank case); a sequence must have
the same length as its candidate sequence.  Inputs are validated with the same
guards as the scalar methods before any batched operation.

```python
def fit_many(
    self,
    hypotheses: Sequence[Hypothesis],
    fitting_views: FittingView | Sequence[FittingView],
) -> list[FittedHypothesis]: ...

def score_many(
    self,
    fitted: Sequence[FittedHypothesis],
    scoring_views: ScoringView | Sequence[ScoringView],
) -> list[EvidenceScore]: ...

def prepare_score(
    self,
    fitted: FittedHypothesis,
    scoring_view: ScoringView,
) -> PreparedScore: ...

def score_region(
    self,
    prepared: PreparedScore,
    support: torch.Tensor,
) -> EvidenceScore: ...
```

`PreparedScore` is an immutable-by-contract snapshot (a frozen dataclass is not
enough to freeze a PyTorch tensor).  It stores private detached clones of the
full-support per-pixel NLL and every scalar/recipe/provenance field needed for a
regional reduction; it never aliases `ScoringView.image` or
`FittedHypothesis.parameters`.  A cryptographic digest (or an equivalent
constant-time version/identity guard plus digest fallback) covers the cached
tensor, support, and recipe snapshot.  `score_region` validates that digest and
raises on tampering, while accessors may return clones.  Mutating the original
fitted hypothesis or scoring view after preparation therefore cannot silently
alter a prepared result.  The public `score` remains the independent existing
scalar implementation/reference, not a wrapper around `prepare_score`; tests
compare the two implementations on deterministic fixtures so equivalence is
not vacuous.  The full-support prepared result must agree within the documented
float64 rounding tolerance (and exactly agree on availability, count, reason,
and discrete ordering).

Recipe values are recursively converted to immutable tuples/frozensets before
being stored, so a future nested recipe field cannot alias a live model object.

`score_region` accepts a boolean `[H,W]` subset mask.  It intersects that mask
with the immutable prepared full support, sums only the cached NLL at the
intersection, and computes the same complexity/prior fields as the scalar
score.  It never reads new image values, refits, or changes the prepared
snapshot.  A zero intersection returns the same unavailable reason/count
contract as scalar `score`.  A region mask outside the prepared support is
therefore harmless and cannot manufacture evidence.  The role remains the
original scoring role (`select` or `verify`), and provenance IDs cannot be
changed by a regional call.

`score_many` may group only candidates with compatible geometry, device, role,
and dtype; each entry's protocol/recipe and study/unit/partition provenance is
validated independently before grouping.  A group fallback to scalar `score` is
valid.  This permits corresponding views from different units without sharing
state or silently relaxing any ID guard.
There is no model-global cache.  Any prepared snapshots are scoped to the call
or to one auditor-owned regional pass, and changing a fitted hypothesis,
scoring partition, or support invalidates rather than reuses a snapshot.

## Vectorization without semantic drift

For compatible entries in a corresponding view sequence, the supported batched
shape is `[B,N]` over a common `H*W=N` geometry, with each entry's logical `K`
fit pixels selected by its own boolean support (normally a repeated view for a
bank, but different units can be grouped whenever geometry/device/recipe and
protocol match).  When supports are identical this is the usual logical
`[B,K]` path; masked full-grid storage is used to retain exact per-unit support
counts without padding or sharing observations.  Shared image and affine bases are built once per compatible
geometry group in float64.  Label IDs become `[B,N]`; the scalar
`Tensor.mean`/`Tensor.var(unbiased=False)` calls remain per-candidate for global
and region statistics so the estimator itself is unchanged, while the three
bias solves use a batched `torch.linalg.solve` over `[B,3,3]`.  Every candidate
still receives its own component dictionary, fit score, fitting-step count,
capacity, and digest.  No candidate's parameters, labels, or prior are shared.
The scalar fallback is retained for heterogeneous views and any background-mode
path that cannot preserve the fixed ordering exactly; unsupported shapes never
silently change the protocol.

For scoring, fitted component parameters and plane coefficients are stacked
per candidate.  The full `[B,K]` residual/log-density calculation is detached
float64.  Component normalizers use the same Python `math.log` calls as the
scalar reference (materialized from frozen Python component values, without a
GPU-to-host hot-path copy); only batched tensor reductions/matrix solves are
allowed to reorder arithmetic.  A full-support NLL vector is retained for each prepared candidate;
regional reductions then cost only masked sums and counts rather than repeating
the density calculation.  A background two-mode mixture uses the same fixed
component order and `logsumexp` expression as scalar scoring.

Bulk methods must not migrate devices.  Device is derived from live view
tensors; CPU/GPU copies are only the detached copies already required by the
scalar identity/support guards.  Deterministic operations are used, and no
autograd graph is created.

## Numerical and selection contract

Vectorized reductions may reorder floating additions relative to scalar code.
The implementation and tests report the maximum absolute/relative
difference for fit parameters, per-pixel NLL, `nll_sum`, normalized NLL, and
total on deterministic synthetic fixtures, with tolerances explicitly tied to
float64 machine-scale accumulation.  The acceptance gate is stricter for
discrete decisions: candidate ordering and selected index must be identical to
the scalar reference for every fixture, including ties, all-background and
permuted semantic labels.  A numerical perturbation that changes a discrete
selection is a fast-path failure, even if its absolute error is small.  No
epsilon pruning, approximate envelope, or altered threshold is allowed.

## Regional auditor integration

The integrated regional audit prepares each frozen bank fit exactly once against
the full `O_select` support.  For each selected connected region it then calls
`score_region` on the selected fit and every challenger that disagrees on an
observed selection pixel.  The existing bounded region and score-call budgets,
zero margin for unobserved/unchallenged/over-budget regions, and zero regional
refits are unchanged.  The per-region denominator is the exact count of
`prepared.full_support & region`; raw NLL sum, count, normalized NLL, role,
availability, and reason remain in the same report fields.  The auditor may
fall back to scalar `score` when a custom component or heterogeneous geometry
cannot use a prepared snapshot, but it must not reconstruct a score from a
different support or fit again.

## Strict test matrix (owned fast-test file)

The focused fast suite will compare scalar and bulk paths and will assert:

* exact result order and candidate IDs for `fit_many`/`score_many`, including a
  repeated view and a per-candidate view sequence;
* float64 parameter/NLL error bounds plus identical `argmin`/strict-improvement
  selection for Gaussian and Student-t, `background_modes=1` and `2`, and
  `max_iterations=5`;
* immutable fitted/view/recipe snapshots: mutation of caller hypothesis,
  fitted tensors/parameters, scoring image/support, or model recipe is rejected
  or cannot alter a prepared result;
* changed unit/study/partition IDs, support overlap, protocol, capacity, and
  recipe mismatches fail closed;
* empty full support and empty regional subsets preserve the scalar
  `available=False`, `count=0`, `normalized_nll=None`, `total=None`, and reason;
* tiny images, one-pixel/degenerate shapes, absent regions, all-background
  labels, and two-mode background fallback stay finite and report counts;
* candidate independence: mutating one bulk output's parameters or digest does
  not alter any other output; no cross-unit parameter sharing or stale cache;
* regional subset sums equal exact scalar score reductions for arbitrary masks,
  including masks outside the original support, and full prepared score equals
  public `score`;
* fit-only firewall: changing withheld/select/verify intensities cannot alter
  any fit output, while changing scored intensities changes only the score;
* deterministic repeated runs and a final `git diff --check`.

## Focused implementation evidence

After the root profiler sent `BASELINE_CAPTURED` for the verified bounded
source artifact `reports/maskfree150/local_baseline/profile_report.json` and
later opened `TEST_WINDOW_OPEN`, the owned correctness checks were run on CPU:

```
PYTHONPATH=src python -m pytest tests/test_maskfree_observation_fast.py -q
10 passed in 0.51s
PYTHONPATH=src python -m pytest tests/test_maskfree_observation.py \
    tests/test_maskfree_hypotheses.py -q
9 passed in 0.58s
PYTHONPATH=src python -m pytest tests/test_maskfree_auditor_fast.py -q
5 passed in 0.97s
PYTHONPATH=src python -m pytest tests/test_maskfree_trainer.py -q
3 passed in 2.77s
PYTHONPATH=src python -m pytest tests/test_maskfree_evaluation.py -q
3 passed in 0.76s
```

The latest focused runs therefore passed **30 tests** in total.  The fast suite
checks model recipe mutation after preparation, study/unit/partition mismatch,
one-pixel fallbacks, all density/background combinations, full regional fields,
and bulk short-result cardinality guards.  No full repository run was attempted.

Across Gaussian/Student-t and `background_modes=1/2` on the deterministic
16x16 phantom, the maximum observed fit-parameter difference between scalar
and batched fits was `3.33e-16`, fit-NLL difference `5.68e-14`, and full score
NLL difference `1.42e-14`.  Candidate ordering was identical in every case.
These are local CPU equivalence measurements, not throughput or GPU evidence;
the profiler's paired timing window remains separate.

The per-configuration maxima (absolute differences) were:

| distribution | background modes | parameter | fit NLL | score NLL |
| --- | ---: | ---: | ---: | ---: |
| gaussian | 1 | `3.33e-16` | `1.42e-14` | `0` |
| gaussian | 2 | `2.22e-16` | `5.68e-14` | `1.42e-14` |
| student_t | 1 | `3.33e-16` | `0` | `1.42e-14` |
| student_t | 2 | `2.22e-16` | `1.42e-14` | `1.42e-14` |

The 128x128 three-unit fixture exercises corresponding view sequences and all
fit/score fields; its focused test is correctness-only and intentionally does
not report a timing claim.

The source artifact for these numbers is the deterministic test/implementation
pair `tests/test_maskfree_observation_fast.py` and
`src/self_audit_maskfree/observation.py` in this working tree; no historical
artifact or unverified profiler output was substituted.

The suite was intentionally not run before the root's test-window release.
