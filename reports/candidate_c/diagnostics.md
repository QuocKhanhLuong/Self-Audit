# Candidate C diagnostics — delivery report

**Worker:** Claude diagnostics (Worker 4)
**Base:** HEAD `9a612485b7ead1273082d7dd52879f8e9e7a3e83`
**Contract:** `reports/candidate_c/architecture_contract.md`
**Date:** 2026-09-11 (revision 4 — invocation accounting correction)

## Scope and ownership

Files written by this worker, and no others:

| File | Status |
|---|---|
| `src/self_audit/evaluation/candidate_c.py` | new |
| `scripts/audit_checkpoint.py` | modified (additive: probe 4) |
| `tests/test_candidate_c_diagnostics.py` | new |
| `reports/candidate_c/diagnostics.md` | this file |

Revision 4 (this pass) is a narrow accounting correction on top of revision 3:
a keyed group whose four counters are all explicitly zero executed **no** solver
invocation, so ordinary annotation turns, direct rollback and stale-record
preflights no longer inflate the invocation count, and any row that might be
hiding solver work now makes the exact totals `null` even when other groups are
known. No other change.

Revision 3 fixed one material compute bug: the solver's evaluation
counters were summed across rows although one batched solver invocation copies
the same counters into every surviving row, so reported work was multiplied by
the number of survivors. Actual invocation totals, per-sample attribution and
per-row opportunity counts are now three separately named quantities. See
**Solver work** below.

Revision 2 implemented the coordinator's five diff-review
amendments: the PROPOSAL/RETAINED ground-truth split with unbiased row
admission, the pre-`infer` window-mode gate, the inherited batch cap and honest
cost wording, per-batch geometry measurement with raw-tensor release, and input
validation that rejects malformed geometry instead of repairing it. No core,
training, config or provenance file was touched.

`src/self_audit/evaluation/__init__.py` was deliberately **not** touched — it is
outside the granted ownership, and `audit_checkpoint.py` already imports every
evaluation module by its full path, so no re-export is required. Nothing was
committed, pushed, reverted or staged; no training, core model or config file
was edited.

## Coupling agreed with the core worker

Agreed through the coordinator before any consumer code was written, and frozen:

* `infer()` emits `candidate_c_diagnostics`: strictly JSON-compatible per-attempt
  **sample** rows, carrying the contract's field set plus `record_turn`.
* `infer(..., capture_geometry=True)` additionally emits **`candidate_c_geometry`**:
  a list of the same length and in the same order as `candidate_c_diagnostics`,
  element *i* describing row *i*. An available element carries `turn`,
  `record_turn`, `sample_index` (an index into the **full** batch, not into
  `active_indices`), `feature_hw`, `k`, and two equal-length depth-ordered tuples
  `factual` and `chosen`. Each depth entry carries `coordinates` `[1,H,W,K,2]`,
  optional `coordinates_preclamp`, optional `attention`, `depth_index`,
  `turn_index`, `iteration_index` and `state_identity`. All tensors are detached
  ordinary (non-inference) tensors with a leading batch dimension of exactly 1.
  An element that instead carries `unavailable_reason` is reported unavailable.
* Attention arrives in either the native `[1, heads, H, W, K]` layout or the
  flattened `[1, heads, H*W, K]` layout; the last axis is always `K`. The
  consumer accepts both. A missing attention map is `None`, never zeros.

### Temporal alignment (coordinator correction, mandatory)

For a restitution row judged at `turn` whose replayed record came from
`record_turn`:

| Role | Source |
|---|---|
| pretransition `P` | `transition_previous[record_turn][sample_index]` |
| factual `F` (also the currently retained state) | `transition_candidates[record_turn][sample_index]` |
| final judged candidate `Q` | `transition_candidates[turn][sample_index]` |

`record_turn` is the previous accepted **ordinary** turn, not the current turn.
No `transition_factual_candidates` key was requested and no extra ordinary
forward is run to synthesise one. A fourth role, ``R`` (retained), is derived
without any extra tensor: it is ``Q`` when the row was accepted and ``F`` when
it was rejected, because a reject HALTs the row. Only a row with **no**
`record_turn` — an ordinary turn that replayed nothing — is reported
**unavailable**, with a reason; it is never scored as zero. Acceptance is a
stratum, not an admission filter (see the PROPOSAL/RETAINED section).

## Designed vs. measured

Everything in the "Measured" column is read out of the rows the runtime solver
emitted. This module never re-executes the C1 replay, never re-solves for
coordinates and never re-decides acceptance, so a claim made from these numbers
is a claim about the runtime — not about a reimplementation. That limitation is
stated inline in the emitted JSON (`candidate_c.measurement_scope`, and
`rows.c1.note`).

### Per-attempt row quantities (mandated list, all measured)

| Mandated quantity | Emitted key | Denominator / note |
|---|---|---|
| Replay error | `rows.c1.max_abs_err`, `rows["numeric/c1_max_abs_err"]` | pooled only over rows that attempted a replay (`rows.c1.attempted_count`) |
| Replay outcome | `rows.c1.pass_rate` | `attempted_count`; `None` when zero |
| Coordinate displacement | `rows["numeric/coordinate_displacement"]` | solver-reported scalar per row |
| Objective before / after | `rows["numeric/objective_before"] / ["numeric/objective_after"]` | — |
| Objective movement | `rows["numeric/objective_delta"]`, `objective_strictly_decreased_rate` | denominator = rows with **both** endpoints; rows missing one are counted in `pairs_missing` |
| Predicted R / F mass | `rows["numeric/regress_mass"] / ["numeric/fix_mass"]` | — |
| Protected FIX count | `rows["count/num_protected"]` | sum + per-row stats |
| Ties excluded | `rows["count/num_ties_excluded"]` | sum + per-row stats |
| Constraint violation | `rows["numeric/constraint_violation"]` | — |
| Fallback reason | `rows["histogram/fallback_reason"]` | a null reason shows as `__null__`, not dropped |
| Actual solver work | `rows.evals.{factual_replay, coordinate_backward, candidate_checks, total_forward}` | `rows.evals.rows_without_evals_block` counts rows that reported none |
| Innovation magnitude | `rows["numeric/innovation_magnitude"]` | — |
| Final Auditor `delta_q` | `rows["numeric/delta_q"]` | — |
| Final decision | `rows["flag/accepted"]` | `{true_count, false_count, null_count, denominator, rate}` |
| Record kind / accepted path / eligibility | `rows["histogram/record_kind"]`, `["histogram/accepted_path"]`, `["flag/eligible"]` | — |

### Per-point geometry quantities (mandated list, all measured)

| Mandated quantity | Emitted key | Definition |
|---|---|---|
| Displacement at feature-pixel scale | `geometry.displacement_mean_pixels`, `..._max_pixels`, `..._moved_point_rate` | `(chosen − factual) / step`, `step = 2/(size−1)`; a size-one axis has step `0`, contributes exactly `0`, and is named in `degenerate_axes` |
| Coordinate saturation | `geometry.saturation_rate`, `geometry.clamped_rate` | saturated = `|coord| ≥ 1 − eps` (`eps = 1e-6`); clamped = `|preclamp| > 1`; denominator = component count |
| Duplicate supports | `geometry.duplicate_point_rate`, `geometry.queries_with_duplicate_rate` | two points of one query agreeing on **both** axes to within `duplicate_tolerance_pixels` (default `0.5` feature pixels); tolerance is republished next to every count |
| Attention entropy | `geometry.attention_mean_entropy_nats`, `geometry.attention_mean_normalized_entropy` | `−Σ p log p` in nats; normalized form divides by `log K` and is `None` when `K == 1` |
| Support spread | `geometry.spread_mean_radius_pixels`, `geometry.spread_mean_axis_extent_pixels` | mean radial distance to the per-query support centroid, in feature pixels |

### Evaluation-only ground-truth metrics: PROPOSAL vs RETAINED

**Revised in this pass.** The first version admitted a row only when it was
`accepted` *and* its `accepted_path` was a settled `counterfactual`/`rollback`.
That conditioned every correction rate on success: identity fallbacks and
rejected proposals vanished from the denominator, so the block reported the
repair rate of the attempts that happened to work and called it the repair
rate. On the live payload it discarded **every** row and reported
`available: false`; with the fix the same payload scores 2 rows.

Admission is now decided by **history, not outcome**. Any row naming a valid
earlier `record_turn` has a real triplet and is scored — `factual_support`
identity fallbacks, rejected proposals and eligible rows that fell back to
ordinary annotation included. Only a row with no history at all (an ordinary
turn that replayed nothing) stays unavailable, because for it no pretransition
state exists to attribute against.

Two families are reported from identical masks, differing only in which tensor
plays the "after" role:

| Family | After | Meaning |
|---|---|---|
| `proposal` | `Q = transition_candidates[turn]` | what the solver put forward, scored whether or not it was accepted |
| `retained` | `Q` if accepted else `F` | what the row actually kept — reject HALTs, so the factual state is unchanged |

A rejected proposed repair therefore appears under `proposal` and **never**
under `retained`. Selecting the tensor is not re-deciding the audit: the
decision is read from the row's `accepted` field. A row whose `accepted` is
unmeasured is unavailable, since its retained state is undetermined.

With `p = argmax P`, `f = argmax F`, `a` the family's after-argmax and `g` the
target, over pixels valid in the target:

| Metric | Mask | Denominator |
|---|---|---|
| prior true FIX retained | `(p≠g) ∧ (f=g) ∧ (a=g)` | `\|(p≠g) ∧ (f=g)\|` |
| prior true FIX destroyed | `(p≠g) ∧ (f=g) ∧ (a≠g)` | `\|(p≠g) ∧ (f=g)\|` |
| prior true REGRESS repaired | `(p=g) ∧ (f≠g) ∧ (a=g)` | `\|(p=g) ∧ (f≠g)\|` |
| new error introduced | `(f=g) ∧ (a≠g)` | `\|f=g\|` |
| new error outside any prior fix | `(f=g) ∧ ¬priorFix ∧ (a≠g)` | `\|(f=g) ∧ ¬priorFix\|` |
| net correct-pixel delta | `\|a=g\| − \|f=g\|` | `valid_pixels` for the rate form |

Both families publish the **same** denominators — the masks depend only on
`P` and `F` — so the two rate sets are directly comparable. The
`_outside_prior_fix` variant removes the overlap between "new error" and
"destroyed prior fix" from both numerator and denominator.

A `strata` block breaks the scored rows down `by_accepted`, `by_accepted_path`,
`by_fallback_reason` and `by_eligible`, so a conditional read is still
available without the pooled numbers being conditioned. `merge_ground_truth_blocks`
pools counts (never averages rates) for both families and all four strata.

Flat keys are named for the family — `candidate_c/ground_truth_proposal_*` and
`candidate_c/ground_truth_retained_*` — so the old ambiguous names cannot
silently acquire a new meaning; the unprefixed keys are gone.

**GT leakage discipline** (unchanged). `ground_truth_restitution_metrics`
accepts only `rows`, the transition sequences, `target`, `ignore_index`,
`geometry` and `keep_per_row` — no model parameter — and the module contains no
`.infer(`, no `annotation_expert`, no `auditor(` and no `nn.Module` reference at
all; a test asserts those strings are absent. In `audit_checkpoint.py` probe 4
keeps the same three-step shape as probe 2, and a test inspects the recorded
`infer` kwargs to confirm no `mask`/`target`/`oracle_target`/`labels` argument
was ever passed.

### Worked fixture (hand-computable)

Four samples share one 1×4 prior transition, so every row has the same
denominators: `g = [0,0,0,0]`, `p = [1,1,0,0]`, `f = [0,0,1,0]` ⇒ per sample
prior_true_fix = 2 px, prior_true_regress = 1 px, factual_correct = 3 px,
outside = 1 px, valid = 4 px. They differ only in `Q` and the decision:

| # | Case | `Q` | Decision |
|---|---|---|---|
| 0 | success | `[0,0,0,0]` | accepted, `counterfactual` |
| 1 | identity fallback | `= f` | accepted, `factual_support` |
| 2 | rejected harmful | `[1,1,1,1]` | rejected |
| 3 | rejected beneficial | `[0,0,0,0]` | rejected |

Pooled denominators: fix 8, regress 4, factual_correct 12, outside 4, valid 16.

| Rate | `proposal` | `retained` |
|---|---|---|
| prior true FIX retained | 6/8 = 0.75 | 8/8 = 1.0 |
| prior true FIX destroyed | 2/8 = 0.25 | 0/8 = 0.0 |
| prior true REGRESS repaired | 2/4 = 0.5 | 1/4 = 0.25 |
| new error | 3/12 = 0.25 | 0/12 = 0.0 |
| new error outside prior fix | 1/4 = 0.25 | 0/4 = 0.0 |
| net correct-pixel delta | −1 | +1 |

The old admission rule kept sample 0 alone and would have reported a repair
rate of **1.0** — four times the realized 0.25 and twice the proposed 0.5.
`test_the_two_families_actually_differ_and_the_old_selection_was_biased`
asserts exactly that comparison.

## `audit_checkpoint.py` integration

Additive only. Probe 4, `probe_candidate_c_path`, joins the three existing
probes; the fixed-state real-vs-zero-evidence probe (probe 2), the synthetic
positive probe, the GT firewall, the calibration-lineage verification, the
checkpoint binding and `verify_bound_state` are all untouched and still run in
the same order. `DIAGNOSTIC_SCHEMA_VERSION` stays at `1` because the change is
purely additive; the new block carries its own
`CANDIDATE_C_DIAGNOSTIC_SCHEMA_VERSION = 1`.

### Cost, stated plainly

Probe 4 runs **one additional ordinary inference pass** over the capped
batches, on top of probes 1–3. It does not separately re-run the solver — the
solver runs inside that pass exactly as it would in deployment, and the probe
only reads the rows it emitted — but the rollout itself is real extra compute.
The earlier wording could be read as implying no added cost; it now says so in
the module docstring, the probe docstring (`COST.`), the `--candidate_c_batches`
help text and the emitted `measurement_scope` string.

**Mode gate, before any forward.** `current`, `feature_only` and `free_offsets`
never build an accepted-transition record, so probing them buys nothing and
costs a full extra rollout. The probe now reads `model.window_mode` and returns
immediately for those: `infer_calls == 0`, `batches_measured == 0`,
`mode_active: false`, with the reason naming the mode. `candidate_c`,
`candidate_c_no_fix` and `direct_rollback` are all supported comparison paths,
measured and labelled by name in `candidate_c/window_mode`. The active-mode set
is imported from `self_audit.models.self_audit_net.RECORD_CONSUMING_MODES` (with
a documented literal fallback) so it cannot drift from core. A model with no
`window_mode` attribute is measured with `mode_active: null` rather than being
assumed either way.

**Empty is not available.** An empty `candidate_c_diagnostics` list used to set
`available: true` with a row count of zero. It now reports `available: false`
with `mode_active: true` and `diagnostics_key_present: true` — "the machinery
runs but nothing became replay-eligible" — which is a different finding from
"this checkpoint does not run the machinery at all" (`mode_active: false`) and
from "the key was never emitted" (`diagnostics_key_present: false`).

### Solver work: actual totals vs per-row attribution

**Fixed in this pass.** `_diagnostic_row` copies `dict(result["evals"])` into
**every** surviving row of one batched solver invocation. Summing those rows
multiplied real work by the number of survivors: the brief's B=4 group with 2
survivors and `total_forward = 3` was reported as **6**, and the CLI called it
actual work. Dividing by `solver_group_size` would have been wrong in the other
direction, because a halted row emits no diagnostics row at all, so the group
size is not the number of rows that were summed.

Counters are now **deduplicated per invocation**. The invocation key is
`solver_invocation_id` whenever core supplies it (the consumer prefers it
automatically and needs no further coordination if core adds it); otherwise the
composite `(batch_scope, turn, record_turn)`. Core confirmed through the
coordinator that at most one `_AcceptedOrdinaryRecord` and one restitution
solver invocation run per turn per `infer` call, so the composite key is exact
for this architecture. `probe_candidate_c_path` supplies `batch_scope` as the
loader batch index, so a repeated `(sample_index, turn)` pair in a later batch
cannot be folded into an earlier batch's invocation and have its work vanish.

Three quantities are published under `rows.evals`, none of them a rebranding of
another:

| Key | What it is |
|---|---|
| `actual_invocation_totals.totals` | real work, one count per solver invocation. This is what the `*_sum` flat keys carry. |
| `sample_associated_mean` | per-**row** statistics of the shared counters — an attribution of one invocation's cost to each sample that shared it. Its `sum` entries are the raw over-count and say so. |
| `row_sums_overcounted` | the raw per-row sums on their own, kept so the old number is visible rather than hidden. |
| `per_row_opportunity` | `rows_per_invocation`, `solver_group_size`, `rows_with_evals`. A diagnostics row is an opportunity to restitute, not a unit of work. |

**Scope.** The totals are **solver-only**. They exclude the official Auditor's
forward, the ordinary Annotation Expert's forward, and the separate
ordinary-fallback evaluation that core's new C1-failure path runs after a failed
replay. That exclusion is stated in `actual_invocation_totals.note`, in the
probe docstring and on the CLI line itself.

**A zero-work group is not an invocation.** Revision 3 counted every keyed
group as an invocation, so the live three-turn ordinary → C → ordinary
trajectory reported **3** invocations for a single solve: the two ordinary turns
key as `(scope, turn, None)` and were counted as phantom invocations. A group
whose four counters are all explicitly zero is now classified as **executed
nothing**, and three cases land there:

* an **ordinary annotation turn** — no record, no solver;
* a **direct rollback** — it *consumes* an accepted record but never calls the
  solver, so record consumption and solver execution are reported separately
  (`record_consumption.rows_consuming_a_record`, broken down
  `by_accepted_path`);
* a **stale-record preflight** that aborts before any replay — zero
  `factual_replay`, no backward, no checks.

`invocations` is now executed invocations only. The attempt stays visible,
named: `invocation_groups_attempted` and `groups_without_executed_work`. Row
attempt statistics are untouched — every row still counts in `rows_total`,
`rows_with_evals` and the row summary — and `per_row_opportunity` reports
`rows_per_invocation` over executed groups beside `rows_per_group` over all
keyed groups. Classification is by the counters themselves, never by path name,
so a rollback that ever did report work would be counted rather than silently
dropped.

**Unknown is not zero — and a partial sum is not a full one.**

* An invocation whose rows disagree on a counter is listed in
  `inconsistent_invocations` with the offending field names; that field's total
  becomes `None`, never an average. Fields the group agreed on still report.
  `totals_over_known_invocations` gives the part that is known, so one bad group
  does not erase the rest.
* A counter that is missing, negative, non-integer or non-finite makes its
  invocation `unmeasured` and its total `None`.
* A row with **no counters at all** could be hiding solver work: the exact
  totals go `None` even when every other group is perfectly known, and
  `totals_over_known_invocations` carries the known subtotal. A partial sum is
  never presented as the full figure. `rows_that_could_hide_work` says how many
  such rows there were.
* An **unkeyable** row is split by what it reported:
  `rows_without_invocation_key_uncertain` (might carry work → totals `None`)
  versus `rows_without_invocation_key_zero_work` (explicitly zero → still an
  exact total). An explicit ordinary all-zero row is *not* unknown.
* **Zero-forward baseline guard retained:** with genuinely no attempt rows and
  nothing unkeyed, the totals are `0`, because nothing ran.

**Flat keys.** `candidate_c/total_forward_sum`, `factual_replay_sum`,
`coordinate_backward_sum` and `candidate_checks_sum` now carry the **actual
invocation totals**. The raw over-counting sums remain published under explicit
`candidate_c/*_row_sum` names, alongside `candidate_c/solver_invocations` (executed),
`candidate_c/solver_invocation_groups_attempted`,
`candidate_c/solver_groups_without_executed_work`,
`candidate_c/solver_rows_that_could_hide_work`,
`candidate_c/solver_invocations_inconsistent`,
`candidate_c/solver_invocation_key_source` and
`candidate_c/mean_rows_per_solver_invocation`. The CLI prints the two on
separate lines, the first labelled
`candidate_c_work[solver-only, per-invocation; excludes Auditor, ordinary annotator and C1-failure fallback]`
and the second `candidate_c_rowsum[OVER-COUNTS shared counters, not work]`.

**Measured on the real core.** The same live payload as elsewhere in this
report (6 rows, 3 turns):

```
paths: ['ordinary', 'ordinary', 'factual_support', 'factual_support', 'ordinary', 'ordinary']
turns: [(0, None), (0, None), (1, 0), (1, 0), (2, None), (2, None)]
invocations(executed) 1 | groups attempted 3 | zero-work groups 2
ACTUAL totals {'factual_replay': 1, 'coordinate_backward': 1, 'candidate_checks': 2, 'total_forward': 3}
RAW row sums  {'factual_replay': 2.0, 'coordinate_backward': 2.0, 'candidate_checks': 4.0, 'total_forward': 6.0}
could_hide_work 0 | record_consumption {'factual_support': 2}
```

Three turns, one solve, B=2: `invocations == 1`, actual `total_forward == 3`,
raw duplicated sum `6`.

`total_forward` was being reported as **6** before revision 3; the actual
figure is **3**. The invocation count was being reported as **3** before
revision 4; the actual number of solves is **1**.

### Batch cap

`resolve_candidate_c_batches(candidate_c_batches, probe_batches, max_val_batches)`
returns the first non-`None` of the three. `build_report` calls it, so a
programmatic caller that leaves `candidate_c_batches` at its default now
inherits the bound the other probes already run under instead of quietly
traversing the entire validation loader one more time. `None` is returned only
when the caller supplied no bound anywhere; an explicit `0` is honoured.

### Memory

Geometry is measured **per batch**: each element goes through the existing pure
`geometry_row_metrics`, only the resulting numeric dictionary is kept, and the
raw `candidate_c_geometry` payload — full coordinate and attention tensors,
possibly on the GPU — is dereferenced before the loader produces the next
batch, along with the inference output and the batch itself. The new
`summarize_geometry_rows` pools those numeric rows; `summarize_geometry` is now
a thin wrapper for callers that already hold the whole payload. Nothing
image-sized reaches the report: `test_no_tensor_survives_in_the_probe_result`
walks the result for any `torch.Tensor`, and
`test_raw_geometry_tensors_are_released_between_batches` holds weak references
to every tensor the fixture creates per batch and asserts all of them are dead
after the loop.

### CLI flags

```
--skip_candidate_c                            skip probe 4 entirely
--candidate_c_batches N                       batch cap (default: --probe_batches, then --max_val_batches)
--candidate_c_geometry                        request infer(capture_geometry=True)
--candidate_c_duplicate_tolerance_pixels F    default 0.5 FEATURE pixels
--candidate_c_saturation_eps F                default 1e-6
```

### Report shape

A nested `payload["candidate_c"]` block, plus **47** flat keys under the
`candidate_c/` prefix merged into `payload["flat"]`. Every one is always
present, with a `None` value when unmeasured, so a reader can tell "not
measured" from "measured as zero" — guaranteed by a single
`_finalize_candidate_c_detail` used by every exit path (measured, inactive mode,
no eligible history, `--skip_candidate_c`). New keys this pass:
`candidate_c/window_mode`, `candidate_c/mode_active`, `candidate_c/infer_calls`,
`candidate_c/ground_truth_rows_scored`, and the six
`ground_truth_{proposal,retained}_*` rates. `check_required_keys` now also
requires `candidate_c/infer_calls` and `candidate_c/mode_active`.

Honest unavailability, never a fabricated run:

* baseline window mode → `available: false`, `mode_active: false`,
  `infer_calls: 0`, reason names the mode;
* active mode, no attempt → `available: false`, `mode_active: true`,
  `diagnostics_key_present: true`;
* key never emitted → `diagnostics_key_present: false`;
* `--candidate_c_geometry` requested but `model.infer` has no
  `capture_geometry` parameter → `geometry.available: false` with an explicit
  reason (detected by signature inspection; the kwarg is not passed);
* `--skip_candidate_c` → `available: false` naming the flag.

The inference it does run is ordinary deployable `self_audit` inference, so a
rejected candidate HALTs its row exactly as in deployment. The probe never
forces acceptance, never extends a halted row and never supplies an oracle
target.

### Input validation

* `duplicate_tolerance_pixels` and `saturation_eps` are the denominators of the
  duplicate and saturation definitions. A `nan` tolerance makes every
  comparison false and a negative one makes every count zero — both publish a
  confident zero that means nothing — so a non-finite or negative limit now
  raises `ValueError` at every entry point rather than being absorbed.
* `feature_hw` must be a pair of positive integers; otherwise the element is
  unavailable with the reason, not scored.
* A non-finite `coordinates_preclamp` no longer feeds a bogus overshoot; clamp
  activity is reported unavailable.
* **Negative attention weights are rejected, not renormalised.** A negative
  weight is not a probability: rescaling a row containing one still yields
  negative "probabilities", and `−p log p` over those is not an entropy at all.
  Such a map is reported unavailable with that reason. Non-finite maps and rows
  of zero total mass are likewise rejected with their own reasons, and the
  specific reason survives into the row-level summary instead of collapsing
  into one opaque "unavailable". Renormalisation still applies to non-negative
  rows whose mass is merely off by scale, and is flagged (`renormalized: true`).

## Sample output

Real output from `evaluate_candidate_c` on a synthetic 3×3 / K=2 fixture
(abridged; `numeric/*`, `count/*` and geometry pools all share the shape
`{count, null_count, nonfinite_count, rows_considered, mean, min, max, sum}`):

```json
{
  "rows": {
    "available": true, "rows_total": 1, "schema_version": 1,
    "numeric/c1_max_abs_err": {"count": 1, "null_count": 0, "nonfinite_count": 0, "mean": 3.6e-07, "max": 3.6e-07},
    "numeric/regress_mass":   {"count": 1, "mean": 11.25},
    "numeric/fix_mass":       {"count": 1, "mean": 3.5},
    "numeric/objective_delta":{"count": 1, "mean": -0.1542, "pairs_missing": 0},
    "objective_strictly_decreased_count": 1,
    "objective_strictly_decreased_denominator": 1,
    "objective_strictly_decreased_rate": 1.0,
    "count/num_protected":    {"count": 1, "sum": 3.0},
    "count/num_ties_excluded":{"count": 1, "sum": 1.0},
    "numeric/constraint_violation": {"count": 1, "max": 0.0},
    "numeric/coordinate_displacement": {"count": 1, "mean": 0.4142},
    "numeric/innovation_magnitude":    {"count": 1, "mean": 0.7731},
    "numeric/delta_q":        {"count": 1, "mean": 0.2814},
    "flag/accepted": {"true_count": 1, "false_count": 0, "null_count": 0, "denominator": 1, "rate": 1.0},
    "histogram/fallback_reason": {"__null__": 1},
    "histogram/accepted_path":   {"counterfactual": 1},
    "evals": {
      "factual_replay":      {"count": 1, "sum": 1.0},
      "coordinate_backward": {"count": 1, "sum": 1.0},
      "candidate_checks":    {"count": 1, "sum": 2.0},
      "total_forward":       {"count": 1, "sum": 4.0},
      "rows_without_evals_block": 0
    },
    "c1": {"attempted_count": 1, "passed_count": 1, "failed_count": 0, "pass_rate": 1.0,
           "rows_without_replay_attempt": 0,
           "note": "Reported from the runtime solver's own replay result. This module does not re-execute the replay and cannot independently confirm it."}
  },
  "geometry": {
    "available": true, "rows_total": 1, "rows_usable": 1, "rows_unavailable": 0,
    "unavailable_reasons": {},
    "duplicate_tolerance_pixels": 0.5, "saturation_eps": 1e-06,
    "displacement_mean_pixels":          {"count": 1, "mean": 0.08333333333333333},
    "saturation_rate":                   {"count": 1, "mean": 0.027777777777777776},
    "clamped_rate":                      {"count": 1, "mean": 0.027777777777777776},
    "duplicate_point_rate":              {"count": 1, "mean": 0.4444444444444444},
    "attention_mean_entropy_nats":       {"count": 1, "mean": 0.6931471805599453},
    "attention_mean_normalized_entropy": {"count": 1, "mean": 1.0},
    "spread_mean_radius_pixels":         {"count": 1, "mean": 0.08333333333333333}
  },
  "ground_truth": {
    "available": true, "evaluation_only": true,
    "rows_total": 4, "rows_scored": 4, "rows_unavailable": 0, "unavailable_reasons": {},
    "strata": {
      "by_accepted":        {"False": 2, "True": 2},
      "by_accepted_path":   {"counterfactual": 3, "factual_support": 1},
      "by_fallback_reason": {"__null__": 3, "no_strict_improvement": 1},
      "by_eligible":        {"True": 4}
    },
    "proposal": {
      "family": "proposal", "rows_scored": 4,
      "counts": {"prior_true_fix": 8, "prior_true_fix_retained": 6, "prior_true_fix_destroyed": 2,
                 "prior_true_regress": 4, "prior_true_regress_repaired": 2,
                 "factual_correct": 12, "new_error": 3,
                 "factual_correct_outside_prior_fix": 4, "new_error_outside_prior_fix": 1,
                 "after_correct": 11, "valid_pixels": 16},
      "denominators": {"prior_true_fix": 8, "prior_true_regress": 4, "factual_correct": 12,
                       "factual_correct_outside_prior_fix": 4, "valid_pixels": 16},
      "prior_true_fix_retained_rate": 0.75,
      "prior_true_fix_destroyed_rate": 0.25,
      "prior_true_regress_repaired_rate": 0.5,
      "new_error_rate": 0.25,
      "new_error_outside_prior_fix_rate": 0.25,
      "net_correct_pixel_delta": -1, "net_correct_rate_delta": -0.0625
    },
    "retained": {
      "family": "retained", "rows_scored": 4,
      "counts": {"prior_true_fix": 8, "prior_true_fix_retained": 8, "prior_true_fix_destroyed": 0,
                 "prior_true_regress": 4, "prior_true_regress_repaired": 1,
                 "factual_correct": 12, "new_error": 0,
                 "factual_correct_outside_prior_fix": 4, "new_error_outside_prior_fix": 0,
                 "after_correct": 13, "valid_pixels": 16},
      "denominators": {"prior_true_fix": 8, "prior_true_regress": 4, "factual_correct": 12,
                       "factual_correct_outside_prior_fix": 4, "valid_pixels": 16},
      "prior_true_fix_retained_rate": 1.0,
      "prior_true_fix_destroyed_rate": 0.0,
      "prior_true_regress_repaired_rate": 0.25,
      "new_error_rate": 0.0,
      "new_error_outside_prior_fix_rate": 0.0,
      "net_correct_pixel_delta": 1, "net_correct_rate_delta": 0.0625
    }
  }
}
```

(The `ground_truth` block above is the worked four-row fixture, so every number
in it can be checked by hand against the table earlier in this report; the rows
and geometry blocks come from the 3x3 / K=2 synthetic element.)

Inactive-mode shape — a baseline checkpoint, measured at zero cost:

```json
{"available": false, "mode_active": false, "window_mode": "current",
 "infer_calls": 0, "batches_measured": 0, "diagnostics_key_present": false,
 "reason": "window_mode='current' never builds an accepted-transition record, so there is no restitution attempt to measure; the probe was skipped and ran no inference",
 "flat": {"candidate_c/available": false, "candidate_c/mode_active": false,
          "candidate_c/infer_calls": 0, "candidate_c/rows_total": 0,
          "candidate_c/c1_pass_rate": null, "candidate_c/accepted_rate": null,
          "candidate_c/ground_truth_proposal_new_error_rate": null,
          "candidate_c/ground_truth_retained_new_error_rate": null,
          "...": "all 47 keys present"}}
```

Active mode that recorded no attempt — a different finding, reported
differently:

```json
{"available": false, "mode_active": true, "window_mode": "candidate_c",
 "infer_calls": 1, "batches_measured": 1, "diagnostics_key_present": true,
 "reason": "no Candidate C attempt was recorded over 1 measured batch(es): the model emitted the diagnostics key but no sample became replay-eligible"}
```

## Live coupling check against the landed core

Re-run after this revision against the **real** `SelfAuditNet` candidate_c path
— random weights, no checkpoint, no dataset, 2x3x32x32 random images,
`mode="always_accept_refinement"`, `tau_accept=-1e9`, `t_max=3`,
`capture_geometry=True`, CPU. Measured:

| Observation | Value |
|---|---|
| rows emitted / geometry elements | 6 / 6, index-aligned as agreed |
| accepted paths seen | `ordinary` x4, `factual_support` x2 |
| rows that attempted a C1 replay | 2 |
| C1 pass rate / max abs replay error | `1.0` / `0.0` (exact) |
| geometry elements usable | 2 of 6; the other 4 carry `"ordinary annotation path: no accepted ordinary record was replayed"` |
| coordinate tensor | `[1, 8, 8, 4, 2]`, `float32`, `is_inference() == False`, `requires_grad == False` |
| attention layout | native `[1, 4, 8, 8, 4]` (5-D), accepted as agreed |
| mean displacement | `0.0` feature pixels — both replays fell back to the factual support |
| mean coordinate saturation | `0.1436` of components at the domain edge |
| mean attention entropy | `1.3831` nats, against `log 4 = 1.3863` — near-uniform support weighting |
| **ground-truth block** | **`available: true`, 2 of 6 rows scored** |
| GT strata | `by_accepted {True: 2}`, `by_accepted_path {factual_support: 2}`, `by_fallback_reason {infeasible: 1, no_improvement: 1}` |
| GT denominators (both families) | prior_true_fix 44 px, prior_true_regress 45 px, factual_correct 497 px, outside 453 px, valid 2048 px |
| GT rates (both families) | fix_destroyed `0.0`, regress_repaired `0.0`, new_error `0.0`, net delta `0` |
| full result | `json.dumps` succeeds, no `NaN`, no `Infinity`, **no tensor retained** |

This is the clearest evidence that the admission fix mattered. On the identical
payload the previous version discarded all six rows and reported
`ground_truth.available: false`; it now scores the two `factual_support`
identity fallbacks against **real, non-zero denominators** (44 genuine prior
fixes and 45 genuine prior regressions) and reports exactly zero effect —
which is the truth about those attempts, and precisely what the old
accepted-only rule hid.

Four live tests pin this down: `test_consumer_handles_the_real_core_payload`
(real payload end to end, including the non-inference-tensor and
native-5-D-attention assertions),
`test_core_rows_now_carry_record_turn_so_geometry_is_not_required`,
`test_core_row_schema_matches_the_consumer_field_names` (a renamed core row
field fails here rather than silently degrading to nulls), and the parametrised
mode-gate tests.

These numbers describe an **untrained random network**. They demonstrate that
the coupling is real and the measurement is honest; they say nothing about
whether Candidate C works.

### Core `record_turn` gap: CLOSED

The previous revision reported that `_diagnostic_row` omitted `record_turn`,
which forced the ground-truth block to depend on `--candidate_c_geometry`. Core
has since landed it (`src/self_audit/models/self_audit_net.py:1283`,
`"record_turn": None if record_turn is None else int(record_turn)`), and the
live rows now carry it. That limitation is withdrawn: the ground-truth block
works from the rows alone, with or without geometry.

The index-aligned geometry fallback is retained as a belt-and-braces path — the
row field always wins, and the hint is only consulted when the row omits the
field *and* the element agrees with the row on both `turn` and `sample_index`.
`test_core_rows_now_carry_record_turn_so_geometry_is_not_required` asserts the
two paths produce identical counts and strata on the live payload;
`test_the_row_field_wins_over_a_conflicting_geometry_hint` asserts the
precedence.

Two further core observations, reported and not acted on:

* rows use `accepted_path="factual_support"` when the solver falls back. These
  are now **scored**, not excluded: an identity fallback genuinely repaired
  nothing while a real prior fix and a real prior regression were on the table,
  and dropping it is exactly the bias this revision removes.
* `_diagnostic_row` defaults the `evals` counters to `0` rather than `None` for
  rows that never invoked the solver. That is factual zero work rather than a
  fabricated metric, so it is pooled as-is; `solver_group_size` is reported
  separately because the counters are per solver invocation and shared by every
  row of a batched group.

## Tests actually run

Focused diagnostics file plus compile/import only, per the revision brief.
Environment: the pre-existing PyTorch 2.4.1 CPU venv
`/private/tmp/self-audit-torch241/bin/python`.

```
python -m pytest tests/test_candidate_c_diagnostics.py -q
105 passed in 2.17s
```

Compile/import, both owned entry points:

```
python -c "import ast; ast.parse(open('src/self_audit/evaluation/candidate_c.py').read())"
python -c "import ast; ast.parse(open('scripts/audit_checkpoint.py').read())"
# both modules then exec'd: "cli import ok 47 flat keys" / "module import ok"
```

The long `tests/test_calibration_lineage.py` suite was **not** re-run in this
pass, as instructed; it passed (`185 passed in 283.90s`) against the previous
revision, and this revision does not touch the tau/lineage path. Complete
combined test runs are coordinator work after integration.

### What the 105 tests cover

*Row read-out (7).* The full mandated quantity list; unmeasured fields staying
`null`; non-finite values counted separately from missing ones; the `c1` pass
rate using only rows that attempted a replay; the absent-key case.

*Geometry measurement (16).* Displacement in feature pixels (exactly `1.0` for a
one-pixel move); the degenerate size-one axis contributing exactly zero pixels
while still reporting the normalized move honestly; saturation and clamp counts;
missing and non-finite `coordinates_preclamp` leaving clamp activity `None`;
duplicate counting at the default and a widened tolerance; attention entropy at
the uniform (`log K`, normalized `1.0`) and one-hot (`0.0`) extremes; both
accepted attention layouts; missing attention producing `available: false`
rather than zero entropy; support spread; declared-unavailable and
structurally-broken elements; agreement with the core `normalized_pixel_step`.

*Input validation, new this pass (8).* Negative attention weights rejected with
that specific reason rather than renormalised into an invalid entropy;
non-finite attention; zero-mass rows; merely-unnormalised attention still
measured and flagged; non-finite / negative `duplicate_tolerance_pixels` and
`saturation_eps` raising `ValueError` at all three entry points; non-positive
and non-integer `feature_hw` reported unavailable; `summarize_geometry_rows`
pooling pure numbers only.

*Ground truth, rewritten this pass (17).* The four-row hand-computable fixture
(success / identity fallback / rejected harmful / rejected beneficial) with every
pooled count and rate asserted against the table above, for both families;
identical denominators across families; the identity fallback counting in the
denominator with zero repairs; the rejected beneficial proposal counted under
`proposal` and **not** under `retained`; an explicit comparison showing the old
accepted-only rule would have reported `1.0` where the realized rate is `0.25`;
per-row acceptance strata; `keep_per_row=False`; zero-denominator rates `None`;
all four row-rejection reasons including the new "acceptance is unmeasured";
`ignore_index`; merge pooling counts rather than averaging rates and matching a
single whole-batch computation exactly; merge of nothing; the absence of any
model-touching symbol in the module; the `record_turn` geometry fallback, its
refusal on disagreement, and row-field precedence.

*Probe 4 (16).* Parametrised mode gate — `current`/`feature_only`/`free_offsets`
skipped with `model.calls == []` and `infer_calls == 0`; parametrised support for
`candidate_c`/`candidate_c_no_fix`/`direct_rollback` with correct labels; an
active mode with an empty diagnostics list reported unavailable with
`mode_active: true`; a missing key distinguished from an empty one; an unknown
`window_mode` attribute not blocking measurement; `resolve_candidate_c_batches`
inheritance including the explicit-zero case; the batch cap with training-mode
restoration; no tensor surviving in the result; weak references to every raw
per-batch tensor all dead after the loop; geometry pooled across four batches
without the payload; the complete 47-key flat set on three different paths; the
runtime read-out plus after-the-fact GT scoring with no leaking `infer` kwarg; a
rejected row reaching the report under both families; the
unsupported-`capture_geometry` path; the completeness checker.

*Solver work, new this pass (19).* The brief's exact case — a B=4 group with 2
survivors and `total_forward = 3` reporting **3**, not 6, with the raw 6 still
published under `row_sums_overcounted` and `solver_group_size` visible but
unused as a divisor; the sample-associated mean identified as attribution, not a
total; two turns counted as two invocations; repeated ids across batches
collapsing without the scope namespace and separating with it; a
scopes/rows length mismatch raising; `solver_invocation_id` preferred over the
composite key in both directions (two ids in one turn ⇒ two invocations, one
shared id across three rows ⇒ one); a self-contradicting group reporting
`None` for the disputed field while the agreed fields survive and the
disagreement is listed; one bad group not erasing
`totals_over_known_invocations`; parametrised malformed counters
(`None`, negative, non-numeric, `nan`) reported unmeasured rather than zero;
unkeyable rows counted and totals `None`; the zero-forward baseline guard
returning an honest `0` when nothing ran; the exclusion note naming the
Auditor, the ordinary annotator and the C1-failure fallback; and four probe-level
tests covering cross-batch namespacing through the CLI flat keys, an unknown
total reaching the report, the inactive-mode guard with the full 47-key set, and
`*_sum` differing from `*_row_sum` whenever survivors > 1.

*Invocation accounting, new this pass (12).* The real ordinary → C → ordinary
trajectory at B=2 reporting `invocations == 1`, `invocation_groups_attempted == 3`,
`groups_without_executed_work == 2`, actual `total_forward == 3` and raw sum
`6`, with all six rows retained in the row statistics; an all-ordinary payload
reporting a hard zero rather than unknown; direct rollback counted as zero
invocations while its record consumption is reported separately by path; a
stale-record preflight counted as an attempt but not an execution; a single
non-zero counter still counting as executed; a row with no counters making the
exact totals `null` while `totals_over_known_invocations` keeps the known part;
an unkeyable row that might carry work forcing `null`; an unkeyable row that
explicitly reported zero work leaving the total exact; the same mixed trajectory
across two namespaced batches counting two solves out of six groups; the
zero-work and unknown notes naming the three cases; the probe-level flat keys on
a mixed batch with the full 47-key set; and a live test asserting one solve on
the real three-turn output.

*JSON and live coupling (10).* Serialisability with no `NaN`/`Infinity`; the
real-core payload end to end; `record_turn` now on the row so geometry is not
required; the core row schema guard.

### Mutation checks

Each mutation was applied, the suite run, and the file restored immediately.

| Mutation | Result |
|---|---|
| shared rate helper `… if denominator else None` → `else 0.0` | 2 failed (`test_unmeasured_fields_are_null_not_zero`, `test_ground_truth_rate_with_zero_denominator_is_none`) |
| re-introduce the accepted-only + restitution-path-only GT admission | **13 failed**, including both family tests, the bias-comparison test and the rejected-row tests |
| retain the raw geometry payload across batches instead of releasing it | 2 failed (`test_no_tensor_survives_in_the_probe_result`, `test_raw_geometry_tensors_are_released_between_batches`) |
| remove the window-mode gate (`if False:`) | 3 failed (all `test_baseline_window_modes_are_skipped_before_any_forward` cases) |
| totals taken from the raw per-row sum (the compute bug itself) | **14 failed** across the solver-work section and the probe flat-key tests |
| drop the probe's per-batch scope namespace | 1 failed (`test_probe_namespaces_repeated_ids_across_batches`) |
| average an inconsistent group instead of reporting unknown | 3 failed (both inconsistency tests and the probe-level one) |
| count every keyed group as an invocation (the phantom-invocation bug) | **7 failed** across the zero-work section, the probe and the live trajectory |
| let a work-hiding row leave the totals "known" | 4 failed (the unknown-totals tests) |

## Limitations

1. **The only live data is from an untrained random network.** The live
   coupling check uses random weights on random images; no checkpoint and no
   dataset were involved. It proves the schema coupling and the consumer's
   arithmetic, not anything about Candidate C's behaviour on real data. That
   run produced **no settled restitution**: both replayed rows fell back to the
   factual support, so neither the counterfactual nor the rollback acceptance
   path has been exercised end to end here. The `proposal`/`retained` split is
   therefore verified only on the synthetic four-row fixture, where all four
   outcome shapes are present by construction.
2. **Replay correctness is reported, not verified.** `c1_passed` and
   `c1_max_abs_err` are the solver's own numbers. This module cannot
   independently confirm a claimed replay pass, and says so in the JSON.
3. **No GPU, no checkpoint, no dataset** was involved. The end-to-end
   `audit_checkpoint.py` CLI path was exercised only through
   `probe_candidate_c_path` with a fake model and through the live
   `SelfAuditNet`; a full CLI run needs a real checkpoint and is a coordinator
   gate, not something this worker could run. `build_report` itself was not
   executed — the batch-cap fix is covered by the pure
   `resolve_candidate_c_batches` helper it now calls, not by running the whole
   report.
4. **`tests/test_calibration_lineage.py` was not re-run in this pass**, as the
   revision brief instructed. It passed against the previous revision
   (`185 passed in 283.90s`, exit 0) and this revision does not touch the
   tau/lineage path, but that is inference from scope, not a fresh measurement.
   Complete combined runs are coordinator work after integration.
5. **Memory is bounded for geometry, not for rows.** The raw geometry payload is
   released per batch and no tensor reaches the report, but the JSON diagnostic
   rows and the per-batch ground-truth blocks are still accumulated across the
   capped batches. They are small scalar dictionaries, so the growth is linear
   in attempts rather than in pixels; on an uncapped cohort with many attempts
   it is still growth. `--candidate_c_batches` bounds it.
6. **The two GT families share a denominator but not an interpretation.**
   `proposal` measures a counterfactual that mostly did not happen; only
   `retained` describes deployed behaviour. Quoting a `proposal` rate as an
   outcome would be exactly the error this revision removed, in the opposite
   direction. The flat keys are prefixed to make that hard to do by accident.
7. **No effectiveness claim.** Nothing here is evidence that Candidate C helps.
   The ground-truth block is in-sample diagnostic scoring under the report's
   existing `evidence_class="diagnostic_only"` stamp.
8. **Core is mid-change; this is not a stability claim.** Core is currently
   correcting the C1-failure path to fall back to NORMAL annotation rather than
   identity or a crash, hard-binding the runtime backtrack budget to `<= 2`,
   making the half step a genuine projected-support backtrack, and separating
   `feasible` from `improved` with measured failed-check diagnostics. The
   consumer here is additive-tolerant — unknown row keys are ignored, and the
   history-based GT admission already scores the normal-fallback rows because
   they carry `record_turn` — but nothing in this report should be read as a
   claim that the combined system is settled. In particular, the composite
   invocation key is exact only while at most one solver invocation runs per
   turn; core confirmed that holds today and has been asked to add an optional
   `solver_invocation_id`, which the consumer already prefers when present.
   Re-verification against settled core is coordinator work.
9. **The solver-only totals are not a compute budget.** They cover the
   restitution solver alone. The official Auditor's forward, the ordinary
   annotator's forward and the post-C1-failure ordinary fallback evaluation are
   real costs that this block does not measure and does not pretend to.
   Probe 4's own extra rollout is reported separately as
   `candidate_c/infer_calls`.
10. **Geometry cost.** Duplicate detection is `O(H·W·K²)` per depth. At
   `H=W=64, K=8` that is ~262k comparisons per depth — fine — but
   `--candidate_c_geometry` on a large batch cap is not free. The default keeps
   per-row geometry detail out of the report (`keep_rows=False`) so the JSON
   stays bounded.
