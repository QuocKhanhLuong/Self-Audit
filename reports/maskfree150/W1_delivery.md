# W1 delivery — predictive observation model

Contract: `reports/maskfree150/architecture_contract.md`, section "W1 observation API".
Authority document: `self_audit_maskfree_150_orca_prompt.md` (read in full).
Contract version consumed and re-emitted: `maskfree150.v1`.

**Revision 2**, after root inspection of revision 1. Changes in this revision:
frozen fitted identity (deep copy plus SHA-256 digest re-checked at scoring),
full recipe matching at scoring instead of capacity alone, stricter
probability/validity validation including sum-to-one, canonical
`self_audit_maskfree` imports in the test under `PYTHONPATH=src`, and one added
regression test. `background_modes=1` stays the primary setting and
`prior_penalty` keeps its default of zero, both as instructed.

**Verdict: PASS** for the spatial appearance profile, with the explicit limitations
and one flagged interface question listed in sections 6 and 7.

## 1. Owned files

| File | Status |
| --- | --- |
| `src/self_audit_maskfree/observation.py` | new, 750 lines |
| `tests/test_maskfree_observation.py` | new, 305 lines, 4 focused tests |
| `reports/maskfree150/W1_delivery.md` | this report |

No other file was created, edited, staged, committed or pushed. `contracts.py`
is coordinator-owned and was read only. No supervised namespace is imported: the
module's entire import list is `copy`, `hashlib`, `math`, `dataclasses`,
`typing`, `torch` and the relative `.contracts`. There is no mask path, no teacher, no pretrained weight,
no download, and no reference to `self_audit` or `self_audit_candidate_c`.

## 2. Resolved API, exactly as implemented

```python
ObservationModel(
    max_iterations: int = 5,
    variance_floor: float = 0.05,
    beta: float = 0.01,
    *,
    background_modes: int = 1,      # 1 or 2; a property of the MODEL, not of a candidate
    bias_ridge: float = 0.1,        # contract minimum, enforced
    distribution: str = "gaussian", # or "student_t"
)
ObservationModel.capacity -> int                                  # 3 + background_modes
ObservationModel.recipe() -> dict                                 # full comparison recipe
ObservationModel.supports_temporal -> False                       # class attribute
ObservationModel.fit(hypothesis: Hypothesis, fitting_view: FittingView) -> FittedHypothesis
ObservationModel.score(fitted: FittedHypothesis, scoring_view: ScoringView) -> EvidenceScore
```

`ObservationContractError(ValueError)` is raised for every contract violation, so
callers can distinguish a contract breach from an arbitrary numerical failure.

`background_modes` lives on the model rather than on a candidate precisely so that
capacity cannot vary within a bank. W2 constructs one `ObservationModel` and uses
it for every member of a bank; `score` re-checks `fitted.capacity == self.capacity`
and refuses a cross-capacity comparison.

## 3. Resolved exact equations, capacity and fitting budget

### Generative form

For a candidate partition `Y` with labels `y(p) in {0=BG, 1=RV, 2=MYO, 3=LV}`, the
observed fit-normalized intensity at pixel `p = (x, y)` is modeled as

```
I(p) ~ sum_m  pi_{y(p),m} * D( mu_{y(p),m} + b(p),  sigma^2_{y(p),m} )
b(p) = a0 + a1 * x_norm + a2 * y_norm          # x_norm, y_norm in [-1, 1]
sigma^2_{k,m} = max( empirical_var, variance_floor = 0.05 )
```

`D` is `Normal` by default, or a Student-t with **fixed** `dof = 4` when
`distribution='student_t'`. The bias plane `b` is shared by all regions, has
exactly three degrees of freedom for the whole image, and is the only spatial
nuisance freedom in the model. There is no per-pixel free parameter, no decoder,
and no path by which a scored intensity can reach any parameter.

### Capacity

Foreground regions RV/MYO/LV get exactly one component each. Background gets
`background_modes` components (1 by default, 2 when the nuisance option is on).
`capacity` is therefore 4 or 5 and is identical for every member of a bank; it is
recorded on every `FittedHypothesis` and re-checked at scoring time.

### Fitting budget (deterministic, no RNG)

`max_iterations = 5` outer alternating passes, the same for every candidate,
from the same initialization `b = 0`:

1. **Region statistics.** On the bias-corrected residual, closed-form mean and
   variance per region, variance floored at `variance_floor`. A region with fewer
   than `MIN_REGION_PIXELS = 8` fit pixels falls back to the fit-global mean and
   variance; these regions are counted in `parameters['fallback_regions']` and
   `parameters['fallback_region_count']` rather than silently given a sharp
   component that a degenerate partition could exploit.
2. **Background nuisance modes** (only when `background_modes == 2`): hard
   assignment seeded at the 25th/75th residual percentiles, refined for a fixed
   three inner passes, mixture weights from the resulting counts. Fully
   deterministic; a degenerate split falls back to the fit-global variance.
3. **Bias plane.** Weighted ridge solve in float64,
   `(D^T W D + lambda I) c = D^T W (I - mu_{y(p)})` with `W = diag(1/sigma^2_{y(p)})`
   and `lambda = bias_ridge = 0.1`. The ridge penalizes the intercept as well,
   which keeps the plane a shading correction rather than a global offset.

Executed passes are reported as `FittedHypothesis.fitting_steps`. All inputs are
`.detach()`-ed; the nuisance fit is a closed-form estimator, never a gradient
path back into the producer, and no state is shared between candidates (each
`FittedHypothesis.parameters` is a fresh dict).

### Score

```
normalized_nll  = nll_sum / count                # nats per valid observed pixel
complexity      = neighbor_disagreement_fraction + component_penalty
component_penalty = 0.01 * (capacity - 4)        # fixed; 0 at background_modes=1
total           = normalized_nll + beta * complexity + prior
```

`neighbor_disagreement_fraction` is the fraction of 4-neighbor pixel pairs whose
labels differ — the boundary-length cost that stops a candidate from buying
likelihood with arbitrarily many small regions. `prior` is read from
`hypothesis.metadata['prior_penalty']` (non-negative nats, default 0.0): **W1
never invents an anatomical prior**, it consumes the penalty the ontology layer
declares. `nll_sum`, `count`, `normalized_nll`, `complexity` and `prior` are all
reported separately on `EvidenceScore`, as required.

Residual units: nats per pixel, where the pixel intensity is in W3's
fit-normalized scale. `variance_floor = 0.05` is in those units squared. Because
the scale is fit-normalized and not raw HU/greyscale, an NLL value is only
comparable within a unit; do not average NLL across units without the stated
denominators.

## 4. Firewall properties actually enforced in code

* `fit` accepts only `FittingView`; the scored observations are not an argument
  and are unreachable from the fitting code path.
* **Frozen fitted identity.** `fit` deep-copies the candidate before using it, so
  a `FittedHypothesis` shares no tensor, list or dict with the caller's
  `Hypothesis`. It records `metadata['hypothesis_digest']`, a SHA-256 over
  candidate id, source, `semantic_unresolved`, the three tensors and the prior
  penalty. `score` recomputes that digest and raises if the snapshot was mutated
  in place. Without this, a challenger round rewriting labels in place would
  retroactively change which partition every stored fit appears to describe,
  while its appearance parameters still belonged to the old partition.
* **Full recipe matching, not capacity alone.** `score` compares
  `fitted.metadata['recipe']` against `self.recipe()` and names every differing
  key. The recipe covers `capacity`, `background_modes`, `max_iterations`,
  `variance_floor`, `bias_ridge`, `beta`, `distribution`, `student_t_dof`,
  `min_region_pixels`, `component_penalty` and `contract_version`. Two fits made
  under different variance floors or density families are the same capacity but
  are not comparable numbers, so this is refused.
* `score` accepts only `FittedHypothesis` and `ScoringView`, performs **no**
  refit, and mutates nothing. Repeat scoring is bit-identical.
* `study_id`, `unit_id` and `partition_id` must all match between the fit and the
  scoring view, otherwise `ObservationContractError`. A fit cannot be scored
  against another unit or against a different frozen observation partition.
* Both views are `validate()`-ed on entry, so a `FittingView` that still contains
  withheld intensities in image or context is rejected before any fitting.
* The **full draft partition** enters the likelihood. `hypothesis.validity` is
  deliberately never used as a scoring weight, so a candidate cannot hide a badly
  explained region by declaring it invalid.
* Strict type and value checks: labels long `[H,W]` within `0..3`;
  probabilities float32 `[4,H,W]`, finite, non-negative and summing to one over
  the four semantic classes at every pixel within
  `PROBABILITY_SUM_TOLERANCE = 1e-4`; validity float32 `[H,W]`, finite, within
  `[0,1]`; prior penalty finite and non-negative.
* `protocol` must be `spatial_predictive`. `cine_predictive` raises with an
  explicit message rather than silently running a spatial fit that a report could
  then mislabel as a temporal experiment. `ObservationModel.supports_temporal`
  is `False`.

## 5. Tests — exact commands and output

The test imports the package under its canonical name `self_audit_maskfree`, so
the exact command needs `PYTHONPATH=src`. This avoids creating a second copy of
the module under a different top-level name; see the integration note in
section 6, item 4.

```
$ cd /Users/alvinluong/Self-Audit && PYTHONPATH=src python3 -m pytest tests/test_maskfree_observation.py -v
============================= test session starts ==============================
platform darwin -- Python 3.11.16, pytest-9.1.1, pluggy-1.6.0 -- /Users/alvinluong/miniforge3/bin/python3
cachedir: .pytest_cache
rootdir: /Users/alvinluong/Self-Audit
plugins: anyio-4.14.2
collecting ... collected 4 items

tests/test_maskfree_observation.py::test_target_blind_fit_and_positive_score_sensitivity PASSED [ 25%]
tests/test_maskfree_observation.py::test_partition_sensitivity_against_degenerate_candidates PASSED [ 50%]
tests/test_maskfree_observation.py::test_matched_budgets_and_no_refit_during_scoring PASSED [ 75%]
tests/test_maskfree_observation.py::test_frozen_fit_identity_and_recipe_mismatch PASSED [100%]

============================== 4 passed in 0.46s ===============================
```

Run together with W8's firewall file, which owns its own import convention:

```
$ cd /Users/alvinluong/Self-Audit && PYTHONPATH=src python3 -m pytest tests/test_maskfree_observation.py tests/test_maskfree_firewall.py -q
....xxx                                                                  [100%]
4 passed, 3 xfailed in 0.46s
```

What each test asserts:

1. **`test_target_blind_fit_and_positive_score_sensitivity`** — negative control:
   the withheld intensities are inverted and tripled, and every fitted parameter
   plus the fit NLL stays bit-identical. Positive control: the same frozen fit
   scores the clean withheld image strictly better than the corrupted one, so the
   score is not inert. Also asserts that empty scoring support returns
   `available=False`, `normalized_nll=None`, `total=None` with a reason (never a
   zero score), and that a foreign `unit_id` raises.
2. **`test_partition_sensitivity_against_degenerate_candidates`** — the two
   standard counterexamples. A mask-shuffled partition and an all-background
   partition both score strictly worse than the true partition, on
   `normalized_nll` alone as well as on `total`, so the win is not an artifact of
   the complexity penalty. Also asserts that all-background reports its three
   empty regions explicitly (`fallback_regions == [1, 2, 3]`).
3. **`test_matched_budgets_and_no_refit_during_scoring`** — a four-candidate bank
   at `background_modes=2`: identical `fitting_steps`, identical `capacity`,
   identical component count, no shared mutable parameter dict, distinct fitted
   values. Scoring is pure (parameters unchanged, results repeatable). A fit made
   at capacity 5 is refused by a capacity-4 model. `cine_predictive` raises.
4. **`test_frozen_fit_identity_and_recipe_mismatch`** (added in revision 2) — the
   caller zeroes its own `Hypothesis` labels, probabilities, validity and sets
   `prior_penalty = 9.0` *after* fitting. The stored fit is unaffected:
   `fitted.hypothesis.labels` still equals the fitted partition, the stored
   metadata never acquires the injected prior, and the rescored `total` and
   `prior` are identical to the pre-mutation values. Mutating the frozen snapshot
   itself then raises `"mutated after fitting"`. Five models that differ only in
   `variance_floor`, `beta`, `distribution`, `bias_ridge` or `max_iterations` —
   all at the *same* capacity — are each refused with `"recipe mismatch"`.
   Finally, probabilities perturbed off the simplex are refused at `fit` with
   `"sum to one"`.

### Measured behavior recorded for the coordinator

Synthetic concentric phantom, 32x32, four well-separated region means, 768 fit
pixels / 256 selection pixels, `beta=0.01`, `distribution='gaussian'`. Selection
-role numbers, nats/pixel. Revision 2 changed no default and reproduces revision
1's numbers exactly:

| background_modes | candidate | fit NLL | select NLL | complexity | total |
| --- | --- | --- | --- | --- | --- |
| 1 (primary) | truth | -0.1717 | -0.1219 | 0.0968 | -0.1209 |
| 1 (primary) | all_bg | 1.3466 | 1.4138 | 0.0000 | 1.4138 |
| 1 (primary) | permuted | -0.1717 | -0.1219 | 0.0968 | -0.1209 |
| 1 (primary) | shifted | 0.7027 | 0.7576 | 0.0968 | 0.7586 |
| 2 (optional) | truth | -0.1265 | -0.0955 | 0.1068 | -0.0945 |
| 2 (optional) | all_bg | 1.1625 | 1.2731 | 0.0100 | 1.2732 |
| 2 (optional) | permuted | -0.1673 | -0.1183 | 0.1068 | -0.1173 |
| 2 (optional) | shifted | 0.6271 | 0.6556 | 0.1068 | 0.6566 |

Student-t at `dof=4` reproduces the same ordering with slightly higher NLL
(truth -0.0690, all_bg 1.4625, shifted 0.8051 at `background_modes=1`).

Digest behavior observed in the same probe: `truth` and `permuted` have the same
select NLL at `background_modes=1` but different `hypothesis_digest` values
(`2e5bd0ee…` vs `ace6ac39…`), so a likelihood tie between semantic alternatives
is still two distinguishable frozen identities for W2 and W7 to record.

## 6. Findings the coordinator and W2 need

1. **Semantic permutation is exactly likelihood-neutral at `background_modes=1`,
   and is NOT neutral at `background_modes=2`.** The table above shows `permuted`
   tying `truth` to the last digit with one background component, which is the
   contract-required behavior: a semantic relabeling that leaves appearance
   unchanged must not acquire an invented likelihood distinction, and the
   ontology resolver must expose the ambiguity instead. With two nuisance
   background modes, background is the only privileged region, so a permutation
   that moves a different anatomical region into slot 0 changes its capacity and
   breaks the tie (-0.1183 vs -0.1173). **This is an asymmetry in the nuisance
   option, not evidence about anatomy.** Recommendation: keep
   `background_modes=1` for any comparison that includes semantic alternatives,
   or have W2 treat a tie threshold rather than exact equality. **Resolved by the
   coordinator: `background_modes=1` is the primary setting.** It remains the
   constructor default; the two-mode option stays available and is now covered by
   the recipe check, so a one-mode fit can never be scored by a two-mode model.
2. **`prior_penalty` is a new metadata key on the `Hypothesis` contract.** W1
   reads `hypothesis.metadata['prior_penalty']` as non-negative nats and defaults
   it to 0.0 when absent, so nothing breaks if W2 never sets it. It is the only
   channel by which an anatomical-prior violation can reach the score, and W1
   does not define its magnitude. **Accepted by the coordinator with the default
   of zero.** Note that the key is read at fit time and baked into the frozen
   snapshot and its digest, so setting it after `fit` has no effect and is
   detected if it mutates the snapshot; W2 must set it before fitting.
3. **W2 must supply labels over the whole `[H,W]` grid, including withheld
   locations**, since the likelihood is evaluated at scoring pixels using
   `hypothesis.labels` there. Those label values must come from fitting-side
   inference or nearest-fit extension, never from the scored intensities. W1
   cannot verify that provenance and does not attempt to; it is W2's obligation
   under the contract and a natural target for W8's red team.
4. **Import convention is now split across two test files, and root should pick
   one.** `tests/test_maskfree_observation.py` imports the canonical
   `self_audit_maskfree` and therefore needs `PYTHONPATH=src`, as instructed.
   W8's `tests/test_maskfree_firewall.py` imports `src.self_audit_maskfree`,
   the older repository convention used by `tests/test_self_audit_*`. Running
   both together under `PYTHONPATH=src` passes (4 passed, 3 xfailed), because
   neither file passes objects to the other. But the two names do load two
   separate module objects with two distinct `Hypothesis` classes, so a future
   test that builds a candidate under one name and fits it under the other would
   fail an `isinstance` check for a reason that looks like a bug in W1. The
   durable fix is a root-owned `conftest.py` putting `src` on `sys.path`, after
   which every file can use the canonical name. I did not create it: `conftest.py`
   is not an owned file of mine, and W8's test file is not mine to edit.
5. **Bank-level contract W2 now inherits.** Because `score` matches the whole
   recipe, W2 must construct exactly one `ObservationModel` per bank comparison
   and reuse it for every `fit` and every `score`. Constructing a fresh model with
   different arguments mid-bank now raises instead of silently producing an
   unmatched comparison.

## 7. Limitations, stated plainly

* **Spatial profile only.** `cine_predictive` is not implemented. No temporal
  deformation model, no motion parameters, no transport to withheld times. The
  module raises instead of degrading silently, so no report can accidentally
  present a spatial-appearance result as a temporal or biomechanical one.
* **The Student-t option is a moment-matched robustification, not an EM t-fit.**
  `variance` is used as the squared *scale* parameter, so it is not the
  distribution variance (`variance * dof/(dof-2)`). This is documented in the
  module docstring for `_log_density` and must not be described as a maximum
  likelihood Student-t estimate.
* **All evidence in this report is CPU synthetic-phantom evidence about the
  implemented equations.** This checkout contains no real ACDC or M&Ms data, and
  no GPU was contacted. Nothing here measures segmentation quality, and
  `EvidenceScore.total` is a predictive NLL, not accuracy and not a calibrated
  probability of correctness.
* **Not run:** the broad repository suite, any real-data path, any commit, stage
  or push. Per the dispatch, only the four focused checks above were executed.
* **The identity digest guards against mutation, not against a wrong partition.**
  It proves that the partition being scored is the partition that was fitted. It
  says nothing about whether W2 derived the labels at withheld locations from
  fitting-side inference rather than from the scored intensities; that provenance
  remains W2's obligation and W8's target.
* The bias plane is affine in XY only. If real cardiac fields show curved
  shading, that model mismatch will show up as residual structure; the contract
  fixes this restriction on purpose and any extension must be renegotiated
  through the coordinator rather than added locally.

## 8. Worker identity

The local harness self-reports the model as **Opus 5, model ID `claude-opus-5`**.
No environment variable exposes a resolved provider model in this session
(`CLAUDE_MODEL` is unset), so this is a harness self-report, **not** a verified
provider launch receipt. The coordinator should take model provenance from the
actual Orca launch transcript, not from this line.
