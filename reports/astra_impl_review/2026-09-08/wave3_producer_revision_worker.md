# W3.1 producer revision + bounded W3.2 runner integration (worker report)

Date: 2026-09-08. Implementer: Claude Code via Orca (`task_65e1fdd77649` / `ctx_1729fcee7642`).
Scope: the producer REVISE items, plus the runner/cache integration W3.2 (`ctx_b0cee3308413`)
requested and the coordinator approved. Ownership unchanged: `src/self_audit/provenance.py`,
`training/_utils.py` checkpoint helpers, `scripts/train_self_audit.py`,
`scripts/cache_validation_transitions.py`, `tests/test_checkpoint_binding.py`. No other
worker's file was edited or reverted.

## 1. Metric semantics now come from the measurement

**Was wrong.** `scripts/cache_validation_transitions.py` passed neither `metric_space` nor
`neutral_margin` to `build_lineage`, so `lineage["semantics"]` carried `None` for both. The
unified runner passed both, but as free-standing caller values that could disagree with the
contract the cache was actually collected under — the lineage would then describe semantics
the measurement was never taken with.

**Now.** New `provenance.resolve_lineage_semantics(...)`, used by `build_lineage(cache=...)`:

- The transition cache is authoritative. Its recorded `metric_contract` is resolved to the
  full versioned definition, so `metric_space`, `neutral_margin`, `empty_class_policy`,
  `metric_contract_version` and `cache_schema_version` are all derived and are never `None`
  once a contract is known. `semantics_source` states where they came from.
- The cache is checked against the contract it names: a cache whose own `metric_space`,
  `neutral_margin`, `metric_contract_version` or `empty_class_policy` contradicts that
  definition raises `ContractMismatchError`.
- A caller argument that contradicts the derived value is rejected, not silently overridden
  (`_agree_or_fail`, exact for strings, 1e-7 tolerance for the margin).
- Both call sites pass `cache=cache`. The runner keeps passing `metric_space` and
  `neutral_margin` — they are now cross-checked assertions rather than recorded claims. The
  CLI passes its `--metric_contract` the same way, so a flag that disagrees with the
  collected cache fails instead of being stamped over it.

This is stricter than W3.2's REQUEST 3 asked for (it only wanted the two values passed).

## 2. Bound state is verified before anything is written

Both the cache CLI and the runner now call `verify_bound_state(..., boundary=
"pre_write_validation_transition_cache")` immediately before `torch.save`. A model mutated
after collection stops the write; no cache file is produced.

## 3. Cohort coverage cannot outrun what was observed

- `observed_samples` smaller than the number of available/iterated samples now forces
  `covers_full_split=False`, with the shortfall stated in `subset_note`. Previously a
  shortfall was recorded but did not affect the claim.
- A **truncated prefix of a shuffled loader** is refused outright (`ValueError`). Its
  membership is decided by an unrecorded random permutation, so there is no honest identity
  to write down; the caller is told to iterate the full split, disable shuffling, or record
  the indices. Fail-closed, per the task's allowance. Unshuffled truncation stays
  describable and is simply marked not-full.
- The `Subset` direct/nested index fixes from the previous round are unchanged.

## 4. Bounded runner integration for W3.2

**REQUEST 1 (done).** `save_calibration(..., lineage=lineage, extra={...unchanged})`. The
artifact now carries lineage as a first-class field; `extra["lineage"]` is left in place so
nothing that read it breaks.

**Post-save roundtrip (done, per the coordinator's clarification).** After the artifact is
written and read back, the runner rebuilds the expected lineage from the live runtime
objects (`build_expected_lineage`) and runs `verify_calibration_lineage(..., cohort_policy=
CohortPolicy(role=COHORT_ROLE_CALIBRATION))` **before** the calibrated diagnostic uses the
tau. The verification record lands in `report["calibration"]["lineage_verification"]` with
`boundary="post_save_roundtrip"`.

**REQUEST 2 (done, without changing training).** `_resolve_tau_accept` now verifies the
artifact whenever `--use_calibrated_tau` is set and one exists — including when
`--tau_accept` overrides the value, so an explicit CLI tau cannot smuggle an unverified
artifact into the run. A legacy artifact without lineage hard-fails; it remains readable
through `inspect_legacy_calibration`, but never usable. The record goes to
`report["calibration_artifact_verification"]` with `boundary="phase_c_starting_state"`.
`t_max`, `neutral_margin_c` and the resolved metric contract are computed just above the
tau resolution (a pure reordering) so the expected lineage can be built.

**The one design decision worth flagging.** W3.2's request said to *bind* the starting
weights with `bind_evaluation_checkpoint`. That loads the checkpoint into the live model —
and this runner keeps one model alive across A → B → C, so loading `phase_c_best.pt` at
Phase C start would silently discard the Phase A/B weights. That is a training-behaviour
change and is out of scope. Instead I added
`training/_utils.bind_existing_evaluation_state(...)`: it selects the same candidates, reads
the payload, and **describes** the binding the live model already satisfies, loading nothing
(`restored == ()`). If the live state does not already equal the checkpoint, it raises and
tells the operator to resume from that checkpoint or drop `--use_calibrated_tau` — it will
not overwrite live weights to force a match. Weight identity is therefore verified in full
(`checkpoint_sha256`, both state digests, producer SHA, model identity) with zero effect on
what the run trains. `bind_evaluation_checkpoint` and the new helper share
`_select_checkpoint_candidate` / `_checkpoint_binding_from_payload`; no logic is duplicated
and no W3.2 comparison logic is reimplemented.

## 5. Tests — `tests/test_checkpoint_binding.py` (43, up from 30)

New this round:

- **Semantics from the measurement**: lineage built from a *real* collected cache has
  complete, contract-consistent semantics; contradicting `neutral_margin`, `metric_space` or
  `metric_contract` arguments are rejected; a cache whose own embedded `metric_space`,
  `neutral_margin`, `metric_contract_version` or `empty_class_policy` contradicts its named
  contract is rejected; agreeing arguments are accepted as assertions.
- **The actual cache CLI**, driven through `cli.main()` with only the config/dataset/loader
  construction stubbed: the saved cache's lineage has no `None` semantics, matches the
  cache's own recorded fields, names the bound checkpoint, and records `observed_samples`.
  A second test mutates the model inside the collector and asserts the write is refused and
  no file exists.
- **Cohort honesty**: observed shortfall cannot claim full split; shuffled+truncated is
  refused; unshuffled truncation stays describable.
- **Describe-only binding**: leaves the live weights bit-identical, reports `restored == ()`,
  and refuses (without mutating) when the live state differs.
- **Artifact consumption**: a verifiable artifact passes and records the starting-state
  binding; `--tau_accept` does not skip verification and does not rescue a mismatched
  artifact; a lineage-less artifact is refused; an uncalibrated run consults nothing.
- The end-to-end runner test now also asserts the artifact's **top-level** lineage and the
  post-save roundtrip verification record.

One pre-existing test was corrected, not weakened: it passed `neutral_margin=0.0` against a
contract whose margin is 0.005 — exactly the contradiction now rejected — and now passes the
contract's own margin.

## 6. Exact outputs

```
$ python -m pytest tests/test_checkpoint_binding.py -q
...........................................                              [100%]
43 passed in 12.94s

$ python -m pytest tests -q
1 failed, 337 passed, 1 warning in 55.25s
FAILED tests/test_metric_contract_and_replay.py::test_cli_calibrate_threshold_roundtrip

$ python -m compileall -q src scripts tests; echo compileall_rc=$?
compileall_rc=0

$ git diff --check; echo diff_check_rc=$?
diff_check_rc=0
```

Exit codes above are each command's own, not a pipeline tail's. The single warning is
pre-existing (`tests/test_self_audit_core.py:32`).

### The one failure is not mine

`test_cli_calibrate_threshold_roundtrip` asserts `artifact["schema_version"] == 1` and now
gets `2`, because W3.2 bumped `CALIBRATION_SCHEMA_VERSION` to 2 in
`src/self_audit/evaluation/threshold.py:75`. That file, `scripts/calibrate_threshold.py` and
`tests/test_metric_contract_and_replay.py` are all outside my ownership set and I did not
touch them; the assertion is on the artifact W3.2's CLI writes. Reported to the coordinator
(`msg_042855adb497`) for assignment. Earlier full-suite runs during this task also showed
transient failures in `tests/test_calibration_lineage.py` and `tests/test_transition_bank.py`
that cleared on re-run while those workers were mid-edit; both suites pass now.

## 7. Limits

- Producer side plus the two integration points W3.2 asked for. W3 is still not complete
  from this alone; the calibration and bank gates are other workers' to close.
- All fixtures are tiny and synthetic on CPU. No real checkpoint, no real bank, no medical
  validation, no Dice claim, no training run.
- Membership signatures prove membership, not content.
- Legacy checkpoints and legacy calibration artifacts stay unverified and unusable for a
  calibrated evaluation; nothing is upgraded.
- No commit, no push, no architecture/objective/config/split/checkpoint change. The only
  behavioural changes to the runner are the verification gates described above and the pure
  reordering in §4.
- Doc fix noted by W3.2: the `state_digest`/`CheckpointBinding` docstrings referred to
  `verify_bound_state` as if it lived in `provenance.py`; it lives in `training/_utils.py`.
  Left as-is this round to keep the diff bounded — flagging it rather than silently
  bundling an unrelated edit.

## 8. Tooling note

All source edits in this round were applied with the coordinator-supplied
`apply_patch` executable (`/Users/alvinluong/.codex/tmp/arg0/codex-arg0bSOkbZ/apply_patch`),
fed patches on stdin; no Python or shell string rewrites were used. Shell commands are
`rtk`-prefixed. The report file itself was written with the file tool, which is not a source
edit.
