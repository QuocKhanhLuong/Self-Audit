# Maskfree150 — independent leakage and scientific red team (W8)

> Historical worker snapshot. Subsequent integration fixes and independently
> reproduced checks are recorded in [the root final review](final_review.md).
> The open findings below describe the earlier tree, not the published verdict.

Contract version: `maskfree150.v1`.
Reviewer role: W8, independent red team. Read-only on all production code.
Owned files: `tests/test_maskfree_firewall.py`, this report, `reports/maskfree150/W8_delivery.md`.
Coordinator owns the final verdict; nothing here is an integration decision.

Environment: `/Users/alvinluong/Self-Audit`, branch `main`, uncommitted worker output.
Interpreter `/Users/alvinluong/miniforge3/bin/python` (Python 3.11, CUDA false).
Canonical imports `self_audit_maskfree.*` under `PYTHONPATH=.:src`.
CODE ONLY: no GPU, no remote host, no real data, no commit, no push.

All evidence below is CPU synthetic phantoms and synthetic `.npy` volumes. It is
**software evidence about the implemented data flow only** and says nothing about
real ACDC or M&Ms content, which this checkout does not contain.

This report supersedes the first W8 pass. That pass ran when only `contracts.py`
and `observation.py` existed and used `xfail(strict=True)` markers to record gaps
against APIs that did not exist yet. The whole package now exists, so the marked
tests have been **replaced by four passing integration tests**. There are no
`xfail` markers left: a firewall regression must turn the suite red.

## 1. Scope of this pass

Integration review across the full package (~11 200 lines), driving real public
APIs end to end rather than inspecting modules in isolation.

| Area | Modules | Exercised |
| --- | --- | --- |
| data | `data/{discovery,dataset,partition,geometry,firewall}.py` | yes, end to end |
| hypotheses / audit | `hypotheses.py`, `auditor.py`, `ontology.py` | yes, real bank + audit |
| observation | `observation.py`, `contracts.py` | yes |
| verification / freeze | `evaluation/{verification,freeze}.py` | yes, real gates |
| trainer | `trainer.py`, `config.py`, `runtime.py` | yes, bounded and full-timeline runs |
| export | `export.py` | reached, and it is where finalization fails |
| experiments / metrics | `experiments.py`, `evaluation/metrics.py` | reached through the trainer only |

The package was edited by its owners while this review ran. Every finding below
was re-verified against the version present at the time of writing, and the two
places where the code moved under the review are stated explicitly.

## 2. The directed critical check: study-wide verify mutation — **PASSES**

The coordinator's primary concern was a global leak: if the XY role map were
keyed per unit rather than per study, a voxel sealed for a patient's ED unit
would be an ordinary fitting pixel for that same patient's ES unit, and the
sealed intensity would resurface as fitting context one unit over.

It is not. `discovery.py:378-384` builds `partition_id` from `study_id`, native
height/width and the seed — explicitly study-level, with a comment saying why —
and `dataset.py:257` and `dataset.py:432` both rebuild it the same way and refuse
on drift.

Driven end to end: an M&Ms-shaped tree of 4 cases × (ED, ES) = 8 units across
train/dev/test; every verify-role pixel of one study rewritten (`x * -7 + 999`)
in **every slice of every acquisition of that study**; all 8 units rebuilt
through `discover_dataset` + `build_training_unit` and compared.

```
units: 8
partition counts: {'fit': 1868, 'select': 992, 'verify': 960, 'guard_dropped': 1076}
VERIFY-MUTATION changed units: NONE
normalization statistics identical: True
SELECT-MUTATION fit unchanged: True      (positive control)
SELECT-MUTATION selection changed: True  (positive control)
```

Nothing moved: not the fitting image, not the three context channels, not the
fitting support, not the selection view, not the fit-derived normalization
statistics. The selection control moved the selection view and nothing else, so
the comparison is not vacuous. Guard band and normalization provenance both hold
at study scope.

Locked in by `test_study_wide_verify_mutation_changes_no_training_unit`.

## 3. Findings

### F1 — HIGH — the study-scoped role map silently degrades to per-unit when a study's acquisitions disagree on native grid

**Where:** `discovery.py:378-384` (`partition_identity(study_id, height, width, seed)`),
`partition.py` (`partition_identity`), `dataset.py:257`.
**Owner:** W3.
**Class:** software gap; reintroduces exactly the leak section 2 proves is otherwise closed.

`partition_identity` hashes the native height and width alongside the study id.
Two acquisitions of the same patient with different native grids therefore get
**different role maps**, and nothing detects it:

```
mnms:case001_ED (72, 68) 4f3a910aea train
mnms:case001_ES (64, 60) 8a9d30756d train      <- same study, different role map
mnms:case002_ED (72, 68) 1ee81b62f3 train
mnms:case002_ES (72, 68) 1ee81b62f3 train
same study: True   partition ids equal: False
limitation about mixed study grids present: False
```

Both units then resize to the same `image_size`, so the maps overlap on the
output grid without agreeing:

```
unit ES fit pixels that are unit ED SELECT pixels: 217 / 813  (27%)
unit ES fit pixels that are unit ED VERIFY pixels: 415 / 777  (53%)
```

Over half of one unit's sealed verification observations are ordinary fitting
pixels for another unit of the same patient, in the same training split. Both
units train in the same run, so a sealed intensity for unit ED is fitted on
directly as unit ES. `build_training_unit` checks each unit against its *own*
manifest `partition_id`, so both units are individually consistent and no drift
error fires. No entry appears in `manifest['limitations']`.

This is not currently active on ACDC (collapses to one unit per patient) or on a
well-formed M&Ms preprocessed tree (ED and ES share a grid). It is a latent
precondition that is never checked, on a multi-vendor multi-centre dataset where
per-acquisition grids are exactly the thing that varies.

**Suggested fix (W3):** in `discover_dataset`, group records by `study_id` and
require one native grid per study. Either refuse the study with a named error, or
keep one canonical study grid, build the role map there once, and resample it per
acquisition — never silently issue two maps for one study. At minimum emit a
hard limitation code so the condition cannot pass unnoticed.

**Repro:** synthetic mnms tree with `case001_ED` at `(72,68,5)` and `case001_ES`
at `(64,60,5)`; full script in section 6.

### F2 — HIGH — finalization is blocked by a W3/W7 record-key mismatch, so no completed run is possible

**Where:** `data/dataset.py:183-207` (`_unit_record` emits `depth`),
`export.py:121` and `export.py:280` (require `num_slices`).
**Owners:** W3 + W7, to be settled by root.
**Class:** software defect; blocks the entire post-training chain.

A full-timeline run on the synthetic tree reaches finalization and fails:

```
status failed   stop_reason finalization_failed
finalization available: False
reason: ExportError: record needs integral slice_index and num_slices: 'num_slices'
```

W3's whitelisted unit record carries `depth`, `depth_axis` and `slice_index`.
W7's exporter requires `num_slices`. The only other `num_slices` in the
repository is `src/self_audit/evaluation/volume_inference.py` — the **forbidden
supervised package** — so the exporter appears to have taken the key from legacy
vocabulary rather than from W3's contract.

Consequence: `assemble_volume`, `freeze_predictions`, `validate_freeze`,
`load_verification_unit` and `verify_frozen_bank` are all unreachable. No
completed 150-epoch run, no freeze, no verification, no reference evaluation.

Mitigating and worth crediting: the trainer **fails closed**. It reports
`status=failed`, `stop_reason=finalization_failed`, `finalization.available=False`
with the exception text, and does not mark the run completed.

**Suggested fix:** agree one key name through root. Do not resolve it by
importing anything from `self_audit`.

### F3 — observed and apparently fixed mid-review — `_BankIndex` construction arity

An earlier full-timeline run in this same session failed with:

```
TypeError: _BankIndex.__init__() missing 2 required positional arguments:
'challenge_source' and 'pool_source'
```

On re-run minutes later the error had changed to F2, and `experiments.py:379` now
passes all seven fields. Recorded for the coordinator's timeline, not as an open
defect. It is evidence that finalization had not been executed end to end by its
owners before this review reached it.

### F4 — LOW (residual) — `FittingView.metadata` still has no enforced whitelist

**Where:** `contracts.py:26`, `contracts.py:29`. **Owner:** coordinator.

Carried over from the first pass and still open in the schema: `metadata` is
`dict[str, Any]` and `validate()` does not inspect it, although
`architecture_contract.md` line 41 says metadata is whitelisted.

Downgraded from MEDIUM because the only writer is now W3's `_unit_record`
(`dataset.py:183`), which *is* a whitelist and carries no intensities, and
because its `normalization` block was verified byte-identical under study-wide
selection and verification mutation (section 2). The residual risk is a future
writer, not current behaviour.

### F5 — carried forward from the first pass, now closed

- **R3 (fit/scoring support overlap) — FIXED by root.** `observation.py:439-440`
  records `fit_support_shape` and packed `fit_support_bits`; `score` at
  `observation.py:684-685` rejects any overlap with the scoring support. A real
  regression test is now in `test_candidate_generation_and_scoring_are_structurally_fit_only`.
- **R1 (target-informed candidate) — resolved structurally, as root argued.**
  `generate_bank(fitting_view, features=None, *, seed=42)` has no scoring
  parameter; the firewall is in the signature, not a skippable runtime check.
  Verified behaviourally: rewriting every selection *and* verification intensity
  of a study leaves all four candidates byte-identical. The first pass's exploit
  required a hostile in-process caller hand-writing labels, which root correctly
  ruled out as proof of pipeline leakage.
- **R2 (verify reuse) — gated at the right APIs, as root argued.**
  `ObservationModel.score` is generic mathematics and will score any view; the
  gates are `load_verification_unit` and `verify_frozen_bank`, and both hold
  (section 4). `verification.py:182-189` additionally refuses a second
  verification of the same freeze id.
- **R5 (stale-bias fit NLL)** — not re-examined this pass; still open as a
  low-severity reporting nit for W1.

## 4. Gates verified against the real APIs

| Gate | Behaviour | Where |
| --- | --- | --- |
| `ImageOnlyDataset.load_verification_unit` | raises `VerificationAccessError` | `dataset.py:348` |
| `load_verification_unit` with no receipt | `FreezeReceiptError` | `dataset.py:441` |
| … receipt for a different manifest | `FreezeReceiptError` | `dataset.py:392` |
| … receipt not covering this partition | `FreezeReceiptError` | `dataset.py:397` |
| … valid receipt | one `role='verify'` view, disjoint from fit and select supports | `dataset.py:443` |
| … frozen file edited after the freeze | `FreezeReceiptError` on sha256 mismatch | `dataset.py:404` |
| `audit_bank` given a verify view | `AuditError`, "verification observations are sealed until every prediction and checkpoint is frozen" | `auditor.py` |
| `verify_frozen_bank` given a select view | `ValueError` | `verification.py:173` |
| `verify_frozen_bank` second use of one freeze | refused unless explicitly opted in | `verification.py:182-189` |
| Bounded run | `status != completed`, `finalization.attempted is False`, no verify role anywhere in the report | `trainer.py:497-509` |
| Finalization ordering | freeze → `validate_freeze` → **only then** `load_verification_unit` → `verify_frozen_bank` | `trainer.py:1168-1208` |

The ordering comment at `trainer.py:1192` ("ONLY NOW may O_verify be opened") is
matched by the code above it.

## 5. What the owners got right

- Study-level role map with an explicit rationale comment, correct at study
  scope under direct end-to-end mutation (section 2).
- Normalization statistics derived from fitting pixels only, across all three
  context slices, verified immune to selection and verification mutation.
- Independent nearest-resize of three role masks can round two roles on to one
  output pixel; `dataset.py` resolves that with a fixed priority so fitting can
  never claim a pixel a withheld role also claims (`select &= ~verify`,
  `fit &= ~(verify | select)`).
- Guard band is real: 1076 of 4896 pixels dropped from fit in the test partition.
- `load_full_input` exists but is documented and separated as deployment-only.
- The trainer fails closed on both a bounded run and a broken finalization rather
  than reporting completion.
- No import of `self_audit` or `self_audit_candidate_c` anywhere in the package;
  no `pretrained` flag, no `torch.hub`, no download. `config.py:59-66` denylists
  `pretrained_encoder`, `pretrained`, `mask_root` and `gt_root` by construction.

## 6. Commands and exact output

```
$ cd /Users/alvinluong/Self-Audit
$ PYTHONPATH=.:src /Users/alvinluong/miniforge3/bin/python -m pytest tests/test_maskfree_firewall.py -v
tests/test_maskfree_firewall.py::test_study_wide_verify_mutation_changes_no_training_unit PASSED [ 25%]
tests/test_maskfree_firewall.py::test_sealed_verification_is_reachable_only_behind_a_valid_freeze PASSED [ 50%]
tests/test_maskfree_firewall.py::test_candidate_generation_and_scoring_are_structurally_fit_only PASSED [ 75%]
tests/test_maskfree_firewall.py::test_bounded_run_is_partial_and_never_opens_verification PASSED [100%]
============================== 4 passed in 1.64s ===============================
```

F1 repro (throwaway script, session scratchpad, not committed):

```python
root = tmp; base = root/'preprocessed_data'/'mnm'/'train'/'volumes'
np.save(base/'case001_ED.npy', rng.normal(120,25,(72,68,5)).astype('float32'))
np.save(base/'case001_ES.npy', rng.normal(120,25,(64,60,5)).astype('float32'))
m = discover_dataset(root, 'mnms', seed=42)
# -> same study_id, two different partition_ids, no limitation emitted
```

F2 repro:

```python
cfg = MaskfreeConfig(dataset='mnms', data_root=..., output_dir=..., total_epochs=2,
                     warmup_epochs=1, image_size=32, batch_size=2, device='cpu',
                     allow_cpu=True, amp=False, wandb_mode='disabled')
MaskfreeTrainer(cfg).run()
# -> status failed / finalization_failed / ExportError: ... 'num_slices'
```

## 7. Verdict

**REVISE.**

The directed critical check passes and the two findings root pushed back on in
the first pass (R1, R2) are correctly resolved where root said the gates belong.
R3 is fixed and now has a real regression test. What remains is two HIGH
findings, both reproduced:

- **F1**, a latent reintroduction of the cross-unit leak the study-scoped role
  map exists to prevent, triggered by an unchecked precondition;
- **F2**, a W3/W7 key mismatch that makes a completed run impossible today.

Neither is a scientific objection to the method. Both are integration defects
with named owners and small local fixes.

This verdict covers the firewall. It is not a PASS on the pipeline, and the
coordinator owns the final verdict.

### Limitations of this review

**Software:**
- Four tests. Absence of a fifth finding is not evidence of absence.
- `experiments.py` and `evaluation/metrics.py` were reached only through the
  trainer, and F2 blocks the path before either produces results. E0–E5, the
  negative controls, matched-coverage analysis, coverage-versus-error curves and
  the reference evaluator are therefore **untested by this pass**, not clean.
- `export.py` was reached only as far as its first validation error.
- W4's input-mask defense and strict pseudo-target validation were assigned after
  this review and are **not covered**; they need a check once they land.
- The package was edited during the review; findings hold against the version
  present at the time of writing and no later.

**Scientific:**
- All evidence is synthetic Gaussian noise volumes and synthetic phantoms. A
  loader that keeps roles disjoint on noise will keep them disjoint on cardiac
  MRI, but nothing here says whether predictive NLL tracks anatomical
  correctness, whether semantic roles resolve, or whether coverage survives.
- No real data, no GPU, no 150-epoch run, no reference masks. Nothing here bears
  on Dice, on the audited-versus-unaudited student comparison, or on the
  hypothesis under test.
- The F1 overlap percentages (27% and 53%) are specific to the two synthetic
  grids chosen. They establish that the condition is unguarded and that the
  exposure is large when it occurs; they do not predict its frequency on real
  M&Ms.
- This review tests whether the implemented firewall matches the written
  contract. It does not validate the contract, and it is not evidence for or
  against the underlying method.
