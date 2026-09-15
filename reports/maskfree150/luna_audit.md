# W2 bounded audit — Luna review

Status: **PASS with bounded limitations**.

This review was performed in `/Users/alvinluong/Self-Audit` on the existing
checkout. It used no worktree, GPU, SSH, remote command, commit, push, staging,
or trainer integration. The three required source modules and focused test file
were the only implementation/test paths edited; the root agent retains final
integration and publication authority.

## Changes completed

### Regional predictive margin and audit budget

`auditor.py` now passes the aligned frozen `FittedHypothesis` list, the W1
`ObservationModel`, and the `role='select'` `ScoringView` into regional scoring.
For each selected connected region, it creates one restricted view from the
same O_select support and calls `model.score` on the selected frozen fit and
each candidate that disagrees on an observed pixel in that region. It never
refits during this pass. Differences confined to fitting pixels or regions with
no observed selection support receive a zero margin; no distinct challenger or
no available regional score makes the search inconclusive.

The audit trace records primary score calls, extra regional score calls, every
regional score count, total score calls, and regional refits (which must remain
zero). Candidate, fit-iteration, and round ceilings are enforced at four, five,
and two respectively. Rejection remains edit-scoped and does not starve later
candidate slots.

The validity diagnostic is named `same_bank_agreement`. It is explicitly
agreement within the one bounded candidate bank, capped at `0.5` when used as a
validity weight; it is not called reproducibility or repeated-run stability.

### Bank identity and CPU feature boundary

`hypotheses.py` makes consumed producer features detached, contiguous CPU data
before hashing or clustering. The bank identity is composed from explicit
partition, fit-input, feature, and candidate-content hashes. Candidate content
includes draft tensors, semantic status/alternatives, source/id, and primary
prior; generation latency, wall clock, and device metadata are excluded. Stable
aliases used by existing consumers remain present.

### Ontology and prior semantics

`ontology.py` now records `orientation_available`,
`orientation_rule_executed`, and `orientation_resolution` separately. The
orientation rule is marked executed only after usable affine-derived metadata
and actual RV/LV centroid projections are evaluated. Valid metadata with no
applicable blood-pool roles is reported as inapplicable; missing or unusable
metadata remains explicitly unresolved. No metadata-presence flag is treated as
proof that a spatial rule ran.

The v1 scored anatomical prior remains exactly `0.0`. Geometric prior values are
diagnostics only. `study_continuity` remains an image-only helper but is not
threaded through the trainer here; evidence is therefore limited to the
single-slice/unit path until the root-owned trainer integration supplies a
study-ordered continuity mapping.

## Focused evidence

Command:

```text
PYTHONPATH=.:src rtk proxy /Users/alvinluong/miniforge3/bin/python -m pytest tests/test_maskfree_hypotheses.py -q
```

Result: **5 passed in 0.92s** on CPU synthetic phantoms.

The checks cover enclosure and absent anatomy, unresolved geometry without
background substitution, actual orientation execution versus metadata-only
availability, tied semantic likelihood and incumbent retention, anti-starvation,
sealed verification rejection, same-support regional score accounting with zero
regional refits, and latency-independent content identity with detached CPU
features.

## Limits

The evidence is software evidence only. This checkout has no real ACDC/M&Ms
execution, no GPU run, no cine temporal transport, and no claim of anatomical
accuracy, calibration, novelty, or clinical benefit. W1 immutable fitted-state
and support guards are consumed as existing interfaces; W5 trainer wiring and
the final combined integration gate remain root-owned.
