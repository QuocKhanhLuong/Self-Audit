# Q3D U-cycle / resume.v3 review — 2026-11-12 (READ-ONLY)

Reviewer: dispatched worker, task_af8767ea44b9 / ctx_2ee3dccbfc96
Scope: uncommitted diff in `self_audit/candidate_q3d/data.py`, `engine.py`, `baselines.py`,
schema consumers (`__init__.py`, `export.py`, `stages.py`, `diagnostics.py`,
`scripts/smoke_q3d.py`, `scripts/q3d_physio_cache.py`, `scripts/check_repo.py`,
`ACTIVE_PIPELINE.json`, spec §8/§9 FIX notes), and the tests
`test_q3d_repository_clean.py`, `test_q3d_u_cycle_logging.py`,
`test_q3d_sampler_cycles.py`, `test_q3d_baselines.py`, `test_q3d_engine_weighting.py`.

No repository edits. Full suite NOT run, as instructed. Targeted run only:
`pytest -q -p no:cacheprovider tests/test_q3d_sampler_cycles.py tests/test_q3d_u_cycle_logging.py`
with basetemp in scratch: **14 passed in 8.99s**. SOFTWARE_ONLY evidence.

## Verdict

**No blocking findings.** Every requested contract is implemented and tested:

| Requirement | Status | Where |
|---|---|---|
| Baseline receipt only in tmp_path | OK | `tests/test_q3d_baselines.py:217` (now `tmp_path / ...`) |
| Clean-copy full suite, exact empty porcelain | OK, with gaps M1/M2 below | `tests/test_q3d_repository_clean.py` |
| U: shuffled patient cycle from `seed+1`, separate frame RNG `seed+2` | OK | `data.py:256-257, 283-286, 297-312` |
| L / C0 stream unchanged | OK | `data.py:253, 288-289`; tests `test_l_stream_matches_legacy...`, `test_c0_and_paired...` |
| Sampler `joint_sampler.v3`, v2 fails clearly | OK | `data.py:233-234, 340-345` |
| Resume `resume.v3`, v2 fails clearly | OK | `engine.py:194-198`; `export.py:316-319`; all consumers use `CHECKPOINT_SCHEMA` |
| `realized_mean_omega` from consumed omegas per full permutation | OK | `engine.py:1960-2020, 2587-2599` |
| Persistence across resume | OK | observations live in `state["j_u_patient_cycle"]`, saved with `sampler_state`; tested via real `_load_checkpoint` round-trip |
| O1 refresh does not destroy cycle | OK | partial-cycle reset block removed (old `engine.py:2728-2766`); per-observation snapshot SHA kept |

Sampler load is validated fully before commit (`data.py:356-396`), cursor/cycle/updates
consistency is enforced, and the engine cross-checks tracker position against the sampler
record each update (`engine.py:1991-1995`). No checkpoint save lies between the tracker
mutation (`engine.py:2589`) and `state["sampler_state"] = sampler.state_dict()`
(`engine.py:2816`); `_refresh_o1_physiology_and_churn` does not save. So tracker and sampler
cannot persist out of step.

## Non-blocking findings

### M1 (medium, test gap) — clean-suite test ignores `.gitignore`d writes
`tests/test_q3d_repository_clean.py:115-117, 164-174` asserts `git status --porcelain == ""`
but the copied tree includes the tracked `.gitignore`, which ignores `runs/`, `cache/`,
`data/`, `*.pt`, `*.npy`, `*.json` named `local_*`, etc. A test that drops `runs/x/latest.pt`
or `cache/...` into the repo root would leave porcelain empty and pass.
Repro (mental): add `Path("runs/leak.pt").write_bytes(b"x")` to any test; the child suite
still reports empty porcelain.
Suggestion: also assert `git status --porcelain --ignored` is empty (feasible here because
`PYTHONDONTWRITEBYTECODE=1` and `-p no:cacheprovider` already suppress `__pycache__` and
`.pytest_cache`), or diff `git ls-files -o` before and after.

### M2 (medium, test gap until commit) — new untracked tests not in the clean copy
`_copy_tracked_working_source` (`test_q3d_repository_clean.py:47-70`) copies `git ls-files`
plus itself only. Right now `tests/test_q3d_sampler_cycles.py` and
`tests/test_q3d_u_cycle_logging.py` are untracked (`git status`: `??`), so the isolated child
suite does not run them. Running the suite today does not show those two files leave the
tree clean. This fixes itself once they are committed or `git add`ed. Run the clean-suite
check again after staging.

### L1 (low, contract gap) — O1 single-snapshot cycles never checked for mean = 1
`engine.py:2596` skips the realized-mean assertion for **all** O1 cycles (`not is_o1`). The
spec (§8 FIX) only exempts cycles that **mix snapshots**. An O1 cycle wholly inside one
snapshot, with a passing gate, should still give 1 ± 1e-12, but that is never enforced.
Suggestion: condition on `len(u_cycle_record["snapshots"]) == 1` instead of `not is_o1`.
Also no engine-level test drives O1 through a mid-cycle refresh plus resume. Only the
helper-level `test_realized_mean_uses_observations_across_online_cache_refresh` covers
mixed snapshots.

### L2 (low, logging gap) — full-cycle event lacks gate/override flags
The `physiology_weighting_full_u_cycle` event (`engine.py:2799-2815`) logs realized and cache
stats, but not `gate_passed`/`override_used`/`override_policy`, although they are in
`u_cycle_weighting`. Spec §8: "Synthetic override gate phải được ghi rõ". Under a synthetic
override the event can show realized mean ≠ 1 with no in-row marker. The marker is only in
the separate `physiology_support_gate` event. Suggestion: copy those three fields into the
event.

### L3 (low, robustness) — `u_cycle_weighting["gate_passed"]` can raise TypeError
`engine.py:2595-2596`: if `state["physiology_cohort_weighting"]` is absent (the fallback path
in `_new_patient_omega`, `engine.py:1841-1859`, explicitly allows that for direct probes),
`copy.deepcopy(None)["gate_passed"]` raises `TypeError` instead of a clear error. The real
stage path runs the gate first (`engine.py:2320-2326`), so production is unaffected. Use
`.get` with an explicit check.

### I1 (info) — committed receipt is now stale but untouched
`reports/q3d-implementation-20261112/baseline_test_adapter.json` is no longer rewritten by
the test (intended). Its `source_sha256` values now refer to pre-change sources. This is
historical evidence only. Do not cite it as current-source evidence.

## Checked, no issue
- `Q3DData.unlabeled_patients` order comes from the manifest and is pinned by
  `data_identity`, so `seed+1` permutations are reproducible across resume.
- RNG seeds `seed`, `seed+1`, `seed+2` do not collide with engine RNGs (`seed+11`, `+29`, `+47`).
- `test_q3d_engine_weighting.py` monkeypatched sampler now emits cycle fields consistent
  with the tracker validation.
- `baselines.py`, `export.py`, `stages.py`, `diagnostics.py`, smoke/physio scripts,
  `check_repo.py` and `ACTIVE_PIPELINE.json` agree on `resume.v3`. No v2 literal is
  accepted anywhere. `export.py` rejects v1/v2 with an explicit message.
- No remaining consumer of the removed tracker fields (`seen`, `last_cycle_update`,
  `partial_u_cycle_closed`).

Evidence class: SOFTWARE_ONLY (CPU, synthetic). Real data / GPU / campaign: NOT_RUN.
