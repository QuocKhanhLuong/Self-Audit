# W2 delivery — hypotheses, ontology, evidence auditor

Verdict: **PASS with one contract question** (see "Open question for the coordinator").

Scope executed: code only, on the existing `main` checkout at `/Users/alvinluong/Self-Audit`.
No worktree, no remote, no GPU, no real data, no commit, no push, no staging.
Only the three owned modules, the one owned test file and this report were written.

## Files written

| Path | Lines | Status |
| --- | --- | --- |
| `src/self_audit_maskfree/ontology.py` | 661 | new |
| `src/self_audit_maskfree/hypotheses.py` | 489 | new |
| `src/self_audit_maskfree/auditor.py` | 407 | new |
| `tests/test_maskfree_hypotheses.py` | 316 | new |
| `reports/maskfree150/W2_delivery.md` | this file | new |

Nothing else was touched. `git status --porcelain src/self_audit_maskfree` shows the
package as a single untracked directory; no other worker's file was edited or reverted.

## Implemented interfaces (frozen API section W2)

### `ontology.resolve_roles(partition, fitting_view, *, candidate_id, source, continuity=None) -> Hypothesis`

Takes a `[H,W]` long tensor of **anonymous** group ids and names them with nine
enumerated, machine-readable rules. Group ids are never used as class indices.
Every rule that fired is recorded in `metadata['ontology_trace']['rules_fired']`.

| Rule | Content | Constants |
| --- | --- | --- |
| `R1_background` | union of groups covering >= `BG_RING_SHARE` of the image border ring; fallback `largest_area`; the "background is dark" check is a recorded diagnostic that never overrides geometry | `BG_RING_SHARE=0.10`, `BORDER_RING_WIDTH=1` |
| `R2_merge_excess` | repeatedly merge the smallest excess foreground group into its most-adjacent neighbour | `MAX_FOREGROUND_GROUPS=3` |
| `R3_enclosure` | `A` encloses `B` when >= `ENCLOSURE_FRACTION` of `B` lies inside a hole of `A` (complement component not touching the border); unique pair -> `MYO=A`, `LV=B`; exact ties left unresolved | `ENCLOSURE_FRACTION=0.90`, `ENCLOSURE_TIE=1e-6` |
| `R4_remainder` | non-BG group 4-adjacent to MYO -> `RV`; a non-adjacent group stays unresolved rather than being renamed or deleted | — |
| `R5_intensity_prior` | only when R3 cannot resolve: enumerated prior that blood pool is brighter than myocardium orders the groups; always sets `semantic_unresolved` and records every pairwise role swap as an alternative | — |
| `R6_continuity` | optional `{role: mean_fit_intensity}` from already-resolved units of the same study, tie-break only | — |
| `R7_absent` | an absent class is legal, recorded in `absent_roles`, and carries **zero** prior penalty | — |
| `R8_contour_convention` | LV = cavity incl. papillary pixels, MYO = wall excl. cavity, RV = right cavity; recorded on every hypothesis | — |
| `R9_unsupported` | no foreground, non-converged component pass, or missing required geometry -> `semantic_unresolved`, affected pixels `validity=0` | `UNRESOLVED_VALIDITY=0.0` |

Left/right disambiguation fires only when all of `ORIENTATION_KEYS = ("view",
"inplane_left_axis", "geometry_valid")` are present in the whitelisted metadata and
`view == "short_axis"`. This checkout has no real data, so in every check those keys
are absent and the resolver reports
`orientation_metadata_absent_left_right_unresolved` and the missing key list. Cine
geometry remains UNKNOWN here; nothing was assumed.

Connected components, holes, dilation and adjacency are implemented in pure torch
(bounded 4-connected label propagation with an explicit `converged` flag). No scipy,
no sklearn, no numpy dependency added.

`ontology.apply_alternative(hypothesis, partition, alternative, candidate_id)`
materialises a recorded naming as its own hypothesis. Geometry is unchanged and the
prior penalty is **copied**, not recomputed, so a permutation the rules could not
separate cannot gain or lose likelihood by being written down differently.

### `hypotheses.generate_bank(fitting_view, features=None, *, seed=42) -> list[Hypothesis]`

Exactly `BANK_SIZE=4` candidates, index 0 the incumbent, built from fitting
observations only:

0. `grouping` — bounded deterministic k-means over fit intensity, optional producer
   features (first `MAX_FEATURE_CHANNELS=8` channels, detached) and coordinates at
   `SPATIAL_WEIGHT=0.25`. `GROUPING_K=6`, `KMEANS_ITERATIONS=10`. **K is deliberately
   larger than the four anatomical roles**: background routinely splits into several
   appearance groups, and R1/R2 then fold them. Nothing here calls a cluster anatomy.
1. `anatomical_initializer` — rule-based, image-only: brightest connected blob inside
   the central box, its `INITIALIZER_WALL_PIXELS=3` annulus, and the nearest bright
   blob within `INITIALIZER_NEIGHBOUR_SHARE=0.35` of the FOV diagonal.
   `INITIALIZER_CAVITY_QUANTILE=0.90`, `INITIALIZER_CENTER_SHARE=0.25`.
2. `boundary_edit` — incumbent partition, largest foreground group grown by
   `BOUNDARY_EDIT_PIXELS=1`.
3. `split_merge` — largest foreground group split at its fit-intensity median, **or**
   `semantic_alternative` when the incumbent came back `semantic_unresolved` with
   recorded alternatives, so a tie in appearance likelihood still leaves a competing
   hypothesis in the bank.

Hidden locations are filled by `nearest_fit_extension`, a bounded 4-connected
dilation of **fit-side labels**. No selection or verification intensity is reachable:
`generate_bank` accepts no `ScoringView` at all. Every candidate carries
`metadata['generation']` with algorithm, budget, seed, `hamming_vs_incumbent`,
`identical_to_incumbent` and wall seconds, plus `bank_id` / `bank_index`.

### `auditor.audit_bank(bank, fitting_view, selection_view, model, *, rounds=2, improvement_threshold=0.001) -> AuditResult`

- Refuses `role != "select"`, a unit-id mismatch and a `partition_id` mismatch.
- `_round_slots` partitions challenger slots over the rounds: 4 candidates, 2 rounds
  -> `[[1,2],[3]]`. The incumbent is fitted once before round 1 and cached, so the
  bank costs **4 fits, not 8**. `trace['fits_performed']` records the actual number.
- Streamed: each candidate is fitted, scored and released before the next is touched;
  no bank-wide graph and no shared nuisance state. Fit iterations and capacity are the
  W1 model's, identical for every candidate by construction.
- Acceptance: strict `incumbent_total - candidate_total >= improvement_threshold`
  (0.001 nats/pixel), best gain wins, incumbent retained on every tie or shortfall.
  An unavailable score can never be selected; its reason is recorded.
- **Anti-starvation**: a rejection is edit-scoped. Every preregistered slot is spent
  regardless of earlier rejections; `rejection_is_edit_scoped` and
  `learning_sample_retained` are recorded.
- Regional margin `[H,W]`: for a contested pixel, the smallest score gap among
  *disagreeing* candidates; for an uncontested pixel, the largest available gap.
  Both are score differences in nats/pixel, explicitly not probabilities.
- Validity: `selected.validity * (1 - (1 - reproducibility) * (1 - margin_weight))`,
  with `margin_weight = 1 - exp(-max(margin,0)/MARGIN_SCALE)` and
  `MARGIN_SCALE=0.01` nats/pixel. Semantic validity gates first, so an unresolved
  role contributes nothing and keeps its **draft label**; no pixel is rewritten to
  background. Verified by an assertion in check 3.
- Empty/degenerate challenge: `search_inconclusive=True`, margin identically zero,
  and no margin weight is granted. An unchallenged incumbent is not a confident one.
- Trace also carries class pixel counts, valid class weight, `class_absent`, coverage
  mean and above-zero fraction, per-candidate distinctness, prior violations,
  generation costs, the full evaluation order with both scores per candidate, the
  budget block, `selection_observations_consumed` and an explicit
  `adaptive_information_use` note.

## Checks run (3 focused checks, no broad suite)

```
cd /Users/alvinluong/Self-Audit
PYTHONPATH=src /Users/alvinluong/miniforge3/bin/python -m pytest tests/test_maskfree_hypotheses.py -q
...                                                                      [100%]
3 passed in 0.55s
```

1. `test_role_rules_resolve_enclosure_report_ambiguity_and_never_write_background`
   — concentric phantom resolves `MYO`/`LV` by R3 with `absent_roles == ["RV"]` and
   zero prior penalty; two congruent non-enclosing blobs give `semantic_unresolved`
   with recorded alternatives, `validity == 0` on the foreground, and labels that are
   **not** background; an all-background partition yields zero validity everywhere.
2. `test_tied_semantics_incumbent_retention_and_no_starvation_after_rejection`
   — a semantic permutation of the same geometry scores identically
   (`pytest.approx(abs=1e-9)`) and carries an identical prior penalty; a bank of three
   degenerate edits is fully rejected with the incumbent retained; a genuine
   improvement banked in the **last** round is still reached after two earlier
   rejections (`fits_performed == 4`, `accepted is True`, round 2 reached).
3. `test_bank_audit_path_and_sealed_verification_firewall`
   — non-trivial bank: 4 candidates with the expected sources, at least one genuine
   challenger, byte-identical rebuild for the same seed; audit completes with matched
   capacity and fitting steps across all fits, validity in `[0,1]`, unresolved pixels
   still zero; a one-candidate bank reports `search_inconclusive` with a zero margin;
   `role="verify"` and a foreign `partition_id` both raise `AuditError`.

### Observed end-to-end behaviour on the synthetic phantom

Software evidence only — this checkout contains no real cardiac data.

```
grouping               total=0.46212 unres=True  rules=[R1,R5,R7,R8]  absent=['RV'] prior=0.050
anatomical_initializer total=0.72850 unres=True  rules=[R1,R3,R8]     absent=[]     prior=0.000
boundary_edit          total=0.10131 unres=False rules=[R1,R3,R7,R8]  absent=['RV'] prior=0.000
semantic_alternative   total=0.46212 unres=True  rules=[R1,R5,R7,R8,semantic_alternative] prior=0.050
selected = c2_boundary   accepted=True   gain=0.36081 nats/pixel
class_pixel_counts = {'BG': 683, 'RV': 0, 'MYO': 284, 'LV': 57}   coverage_mean_validity = 1.0
```

The semantic alternative ties the incumbent to five decimal places, which is the
permutation-invariance property under test. On this phantom the auditor loses RV
(R1/R2 absorb the small separate blob), which is recorded as `class_absent`, not
hidden — a real limitation of the current bounded rule set, not a passing result.

## Firewall statement

- No import of `self_audit`, `self_audit_candidate_c`, numpy, scipy or sklearn in the
  three owned modules (`grep` over the three files returns nothing outside
  `self_audit_maskfree`).
- No mask path, no pretrained weight, no GT-derived crop/split/threshold/checkpoint.
- `generate_bank` and `resolve_roles` accept `FittingView` only; `audit_bank` accepts
  a `ScoringView` with `role="select"` and rejects `"verify"` with an explicit error.
- Contracts were used exactly as frozen; no dataclass was duplicated or redesigned.
  The only signature additions are keyword arguments with defaults on
  `resolve_roles` (`candidate_id`, `source`, `continuity`), which leave the frozen
  positional contract `resolve_roles(partition, fitting_view)` intact.

## Open question for the coordinator

The integration note said to avoid adding an anatomy score penalty "without contract
constants". W1's `observation._prior_penalty` reads `metadata['prior_penalty']` and
documents it as produced by the ontology layer, so the term is contractually mine to
emit, but the **values** are not in `architecture_contract.md`. I therefore enumerated
them as named module constants:

```python
PRIOR_PENALTIES = {"lv_not_enclosed_by_myo": 0.02, "myo_fragment": 0.01,
                   "rv_enclosed_by_myo": 0.02}
PRIOR_PENALTY_CAP = 0.10   # nats, so no geometric oddity can delete a pathology
```

They are purely geometric, never fire for an absent class (R7), are capped, and tie
across recorded alternatives. If the coordinator wants zero anatomical prior in v1,
setting all three to `0.0` is a one-line change that leaves every other behaviour and
all three checks intact. Please confirm the values or freeze them into the contract.

## Limitations and what is NOT claimed

- Synthetic CPU phantoms only. No ACDC, no M&Ms, no cine, no GPU, no 150-epoch run.
  Nothing here is evidence about real cardiac data or about segmentation accuracy.
- `EvidenceScore.total` is predictive NLL in nats per valid observed pixel plus the
  declared complexity and prior terms. Regional margins are score differences. Neither
  is a calibrated correctness probability, and no ECE/Brier claim is made.
- `GROUPING_K=6`, the bank composition, the two-round budget, `MARGIN_SCALE` and the
  initializer constants are **provisional fixed budgets**, not empirically optimal
  settings. They are logged, not tuned.
- The cine protocol is untouched: `ObservationModel` supports `spatial_predictive`
  only, and the bank inherits that restriction rather than approximating a temporal
  experiment.
- R6 continuity is implemented but unexercised: no multi-unit study fixture exists in
  this checkout, so study-level continuity is code, not evidence.
- The `anatomical_initializer` is a bounded rule-based proposal, not a validated
  cardiac detector. On the phantom it resolved enclosure but scored worst.

## Unrelated observation (not my file, not touched)

`tests/test_maskfree_observation.py::test_target_blind_fit_and_positive_score_sensitivity`
currently fails in this checkout:

```
E   RuntimeError: Boolean value of Tensor with more than one value is ambiguous
tests/test_maskfree_observation.py:122: assert fitted_clean.parameters == fitted_corrupt.parameters
```

That is W1's module and W1's test (a dict equality over parameters that now contain a
tensor). Reported to the coordinator for W1; not modified here.
