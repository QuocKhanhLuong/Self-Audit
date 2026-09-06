# AGY-3 — Phase-B on-policy vs synthetic evaluation

**Owner:** AGY-3
**File edited (only):** `src/self_audit/training/train_auditor.py`
**START_HEAD:** `a901eeae56cf6835bb9a1d9c79a9d5d3e99374c6`
**Scope:** measurement and checkpoint selection only. Training behaviour is unchanged and proven bit-identical below.

---

## 1. What was wrong

`build_auditor_transitions` emits 3 on-policy adjacent pairs plus 4 GT-synthesized
counterfactuals per batch. `_auditor_batch` takes a uniform mean over those 7 groups, so the
synthetic half carries 4/7 = 57.1% of the weight. `validate_auditor_epoch` called the *same*
`_auditor_batch` with the *same* generator instance, so `val/auroc` was computed on that
57.1%-synthetic mixture — and `primary_metric` was that contaminated AUROC, which selected
`best.pt` and the exported `phase_b_auditor.pt`.

The transition dicts already carried a `provenance` string (`"on_policy"` /
`"synthetic:{kind}:{operation}"`). It was simply never used for reporting.

## 2. What changed

Measurement and selection only. The loss, the 3+4 group composition, the uniform mean, the
generator, and `epsilon_neutral` are all untouched.

1. **`build_auditor_transitions`** gained keyword-only `include_on_policy` / `include_synthetic`
   (both default `True`). Each transition dict keeps its existing `provenance` string and gains
   a coarse `provenance_kind` in `{"on_policy", "synthetic"}`.
2. **`_auditor_batch`** now buckets collected predictions/targets by `provenance_kind` **inside
   the existing loop**, reusing the single existing `model.encode` / `forward_annotation` pass.
   No second forward — verified by call counting (Check 3). `transition_data` became a mapping
   `{"on_policy": tup, "synthetic": tup, "combined": tup}`; `"combined"` is appended in loop
   order so it is bit-identical to the flat tuple this function used to return. A new
   `transitions_by_provenance` count is also returned.
3. **`validate_auditor_epoch`** emits three namespaces — `audit/on_policy/*`,
   `audit/synthetic/*`, `audit/combined/*` — each with `auroc`, `auprc`,
   `improve_regress_accuracy`, `correlation_delta_q_delta_dice`, `local_fix_f1`,
   `local_regress_f1`, `transition_count` (plus `transition_group_count`, and any extra keys
   AGY-1's `transition_audit_metrics` returns, which pass straight through). The flat legacy
   keys still emit the **combined** value.
4. **`primary_metric` is now on-policy AUROC**, with the fallback chain recorded verbatim in a
   new `primary_metric_source` field: `"on_policy_auroc"` →
   `"on_policy_improve_regress_accuracy"` → `nan` with source `"undefined"`. It never falls back
   to the combined or synthetic value — asserted in Check 5, including the case where the
   on-policy bucket is entirely empty.
5. **`main()`** prints three extra namespace lines and logs every `audit/*` key plus
   `val/primary_metric` and `val/primary_metric_source` to W&B. All pre-existing keys and the
   original print line are preserved unchanged.
6. **`check_generation_tolerance(epsilon_neutral, neutral_margin, context="Phase-B CounterfactualGenerator")`**
   is called once in `main()` before the generator is constructed. Neither number is changed;
   the 0.02 vs 0.005 conflict is now surfaced as a `RuntimeWarning` at runtime.
7. `neutral_margin` is threaded through to `transition_audit_metrics`. AGY-1 landed its keyword
   while this work was in flight, so the keyword path is the live one — see §5.

### Behavioural change to call out loudly

**`primary_metric` changes value.** It used to be combined AUROC (57.1% synthetic); it is now
on-policy AUROC. This changes which epoch wins `best.pt`, and the reported number will almost
certainly drop. That drop is the finding, not a regression. `best_metric` values persisted in
existing checkpoints are on the old scale and are not comparable to new ones.

`primary_metric` can now be `nan` where it previously returned a combined value — specifically
when the on-policy bucket is empty or degenerate. `main()` already maps non-finite metrics to
`-inf` before comparing against `best_metric`, and `scripts/train_self_audit.py:437` already
guards with `np.isfinite(metric)`, so neither selection path breaks.

## 3. Public signatures — new and changed

```python
# CHANGED (keyword-only additions; positional signature untouched)
def build_auditor_transitions(
    output: Any,
    ground_truth: torch.Tensor,
    generator: CounterfactualGenerator,
    *,
    include_on_policy: bool = True,
    include_synthetic: bool = True,
) -> list[dict[str, Any]]

# NEW
ON_POLICY = "on_policy"
SYNTHETIC = "synthetic"
PROVENANCE_KINDS = ("on_policy", "synthetic")
TRANSITION_METRIC_KEYS = (
    "improve_regress_accuracy", "auroc", "auprc",
    "correlation_delta_q_delta_dice", "local_fix_f1", "local_regress_f1",
)

def provenance_kind(transition: Mapping[str, Any]) -> str
def resolve_primary_metric(on_policy_metrics: Mapping[str, Any]) -> tuple[float, str]

# CHANGED return shape (private helper)
_auditor_batch(...) -> tuple[torch.Tensor | None, dict[str, Any]]
#   data["transition_data"]           : dict[str, tuple] with keys
#                                       "on_policy" / "synthetic" / "combined"
#                                       (was: a bare 4-tuple; now available under "combined")
#   data["transitions_by_provenance"] : dict[str, int]   (new)

# validate_auditor_epoch: signature UNCHANGED. Return dict gains
#   "primary_metric_source": str
#   "audit/{on_policy|synthetic|combined}/{metric}": float
# and keeps every key it emitted before.
```

`provenance_kind` falls back to parsing the `provenance` string when `provenance_kind` is
absent, so transitions built by older code or by hand in a test still bucket correctly.

## 4. Verification

Script: `/private/tmp/claude-501/.../scratchpad/agy3_verify.py` (outside the repo, per the brief).
Synthetic tensors only — there is no `preprocessed_data/` and no `weights/` in this checkout.
Check 1 compares against a verbatim transcription of the pre-change `build_auditor_transitions`
from `START_HEAD`, run under the same seed.

```
========================================================================
CHECK 1 -- default flags reproduce the pre-change transition list
========================================================================
len(old)=7  len(new)=7
  [0] provenance='on_policy'                                  kind=on_policy  bitwise_identical=True
  [1] provenance='on_policy'                                  kind=on_policy  bitwise_identical=True
  [2] provenance='on_policy'                                  kind=on_policy  bitwise_identical=True
  [3] provenance='synthetic:positive:local_connected_error_repair' kind=synthetic  bitwise_identical=True
  [4] provenance='synthetic:positive:local_connected_error_repair' kind=synthetic  bitwise_identical=True
  [5] provenance='synthetic:positive:local_connected_error_repair' kind=synthetic  bitwise_identical=True
  [6] provenance='synthetic:positive:local_connected_error_repair' kind=synthetic  bitwise_identical=True
counts: on_policy=3 synthetic=4
PASS: default list bit-identical, 3 on_policy + 4 synthetic

========================================================================
CHECK 2 -- include_synthetic=False / include_on_policy=False
========================================================================
include_synthetic=False -> n=3 kinds=['on_policy']
include_on_policy=False  -> n=4 kinds=['synthetic']
PASS

========================================================================
CHECK 3 -- _auditor_batch partitions by provenance, ONE encoder pass
========================================================================
model.encode calls=1  forward_annotation calls=1
transition_data keys=['combined', 'on_policy', 'synthetic']
groups by provenance={'on_policy': 3, 'synthetic': 4}  total groups=7
rows per namespace={'on_policy': 6, 'synthetic': 8, 'combined': 14}
combined == cat(on_policy, synthetic) (build order) -> True
PASS

========================================================================
CHECK 4 -- CONTRADICTORY on-policy vs synthetic: primary follows on-policy
========================================================================
  audit/on_policy  auroc=1.0000 auprc=1.0000 acc=1.0000 corr=1.0000 fix_f1=1.0000 regress_f1=1.0000 n=40
  audit/synthetic  auroc=0.0000 auprc=0.3192 acc=0.0000 corr=-1.0000 fix_f1=1.0000 regress_f1=1.0000 n=40
  audit/combined   auroc=0.5000 auprc=0.5310 acc=0.5000 corr=0.0000 fix_f1=1.0000 regress_f1=1.0000 n=80
  legacy flat auroc=0.5000 (== combined: True)
  primary_metric=1.0000 primary_metric_source='on_policy_auroc'
PASS: primary follows the ON-POLICY half; synthetic shows the bad value

========================================================================
CHECK 5 -- primary_metric fallback chain never leaks combined/synthetic
========================================================================
  auroc finite            -> (0.7, 'on_policy_auroc')
  auroc nan, acc finite   -> (0.9, 'on_policy_improve_regress_accuracy')
  both nan                -> (nan, 'undefined')
  empty mapping           -> (nan, 'undefined')
  on-policy bucket empty: primary_metric=nan source='undefined' synthetic_auroc=0.0000 combined_auroc=0.0000
PASS: no silent fallback to combined/synthetic

========================================================================
CHECK 6 -- end-to-end validate on a real toy batch (no monkeypatch)
========================================================================
  audit/on_policy  auroc=nan n=6 groups=6.0
  audit/synthetic  auroc=nan n=8 groups=8.0
  audit/combined   auroc=nan n=14 groups=14.0
  primary_metric=nan source='undefined' legacy_auroc=nan
PASS: all three namespaces present with the required keys

========================================================================
CHECK 7 -- training loss is untouched (uniform mean over 7 groups)
========================================================================
loss=1.45994484 groups=7 by_provenance={'on_policy': 3, 'synthetic': 4}
PASS: 3 on-policy + 4 synthetic still both in the loss, uniform mean

========================================================================
CHECK 8 -- check_generation_tolerance surfaces 0.02 vs 0.005
========================================================================
returned=0.02 (numbers unchanged)  warnings=1
  Phase-B CounterfactualGenerator: hard-neutral search tolerance epsilon_neutral=0.02 exceeds the canonical decision margin neutral_margin=0.005. Generated 'hard neutral' transitions with 0.005 < |delta
PASS

ALL CHECKS PASSED
```

Check 6's `auroc=nan` is correct, not a failure: the toy annotation model's states differ only by
a constant logit offset, so every `delta_dice` lands in one class and the ranking population is
degenerate. It exercises the `nan` → `"undefined"` path end to end.

Every key `main()` reads off the validation dict was checked to exist against a real
`validate_auditor_epoch` return (28 referenced keys, 0 missing), and the three new print lines
were formatted against it. 42 `audit/*` keys are emitted — AGY-1's extra
`neutral_count` / `beneficial_count` / `harmful_count` / `ranking_count` / `tau` /
`neutral_margin` flow through each namespace automatically.

### `pytest -q` on the full suite

```
$ python3 -m pytest tests/ -q
ERROR collecting tests/test_audit_decomposition.py
  src/self_audit/evaluation/audit_decomposition.py:27: in <module>
      from self_audit.audit.semantics import (
  E   ModuleNotFoundError: No module named 'self_audit'
1 error in 0.92s
```

That collection error aborts the whole run and is **not mine** — it is in AGY-2's file (see §6).
Excluding that one module:

```
$ python3 -m pytest tests/ -q --ignore=tests/test_audit_decomposition.py
1 failed, 69 passed, 1 warning in 3.20s

FAILED tests/test_self_audit_hardening.py::test_metrics_use_consistent_both_empty_dice_convention
    assert {1: nan, 2: nan, 3: nan} == {1: 1.0, 2: 1.0, 3: 1.0}
```

That failure is AGY-1's deliberate empty-class policy flip (`"exclude"` → `nan`) landing against
a test written for the old `empty_score=1.0` convention. AGY-8 owns the test update.

Phase-B specific tests, which are the ones my change can break:

```
$ python3 -m pytest tests/test_self_audit_regressions.py -q
21 passed in 1.68s
```

`tests/test_self_audit_regressions.py` asserts `stats["transitions"] == 7.0` and that the auditor
sees every adjacent on-policy pair in order; both still hold.

**Honest statement:** I did not establish a pre-change baseline for the two failures by checking
out `START_HEAD`, because other workers were mid-edit in the same tree and a checkout is
forbidden. Both failures are confined to files I do not own and neither exercises
`train_auditor.py` — `git diff --stat` confirms my only modified file is
`src/self_audit/training/train_auditor.py`.

## 5. Things I could not do / deviations

* **The defensive `transition_audit_metrics` fallback is now dead code.** The brief asked for a
  `TypeError` fallback because AGY-1 was landing in parallel. AGY-1 has since landed:
  `transition_audit_metrics(..., *, neutral_margin=None, tau=0.0)` accepts the keyword, so the
  keyword branch is the live one and the `except TypeError` branch is unreachable. It is
  guarded on `"neutral_margin" in str(error)` so it cannot swallow an unrelated `TypeError`.
  **Opus should delete `_transition_metrics_for`'s `try/except` at integration.**
* **No `include_synthetic=False` evaluation pass was wired into `main()`.** The plan's item (3)
  asks for provenance-partitioned reporting from the *existing* single pass, and item (2)
  explicitly forbids a second forward. The flags exist and are tested, but nothing in
  `train_auditor.py` calls them with non-default values — they are there for AGY-6's diagnostics
  CLI and for tests. Deliberate.
* **`transition_count` is a row count, not a group count.** `audit/*/transition_count` counts
  scored transition *rows* (one per sample per group), which is the population the AUROC is
  computed over. The group count (7 per batch) is available separately as
  `audit/*/transition_group_count`. The plan did not specify which; I emitted both so the
  ambiguity cannot cause a silent misread.
* **`primary_metric_source` is a string in the metrics dict.** `scripts/train_self_audit.py:424`
  splats the whole validation dict into its W&B payload and into `report["phase_b"]`.
  `WandbLogger.log` (`training/_utils.py:1008`) explicitly passes `str` through, and the report
  is JSON, so both are safe. Verified by reading, not by running W&B.

## 6. Observations in files I do NOT own (described, not edited)

1. **`src/self_audit/evaluation/audit_decomposition.py` (AGY-2) breaks the whole pytest run.**
   Line 27 does `from self_audit.audit.semantics import ...`, but that module has no
   `sys.path` bootstrap and `tests/test_audit_decomposition.py` imports it via the
   `src.self_audit.` package path, under which the bare `self_audit` name is not importable.
   Sibling modules such as `evaluation/metrics.py` avoid this by not importing across the
   package root, and `training/train_auditor.py` avoids it by inserting `ROOT`/`SRC` into
   `sys.path` at the top of the module. The fix is either a relative import
   (`from ..audit.semantics import ...`) or the same `sys.path` bootstrap. **AGY-2 or Opus.**

2. **`tests/test_self_audit_hardening.py:93`
   (`test_metrics_use_consistent_both_empty_dice_convention`)** hard-codes the old
   `empty_score=1.0` convention and now fails against AGY-1's `"exclude"` default. It needs to
   assert `nan` under the default policy and `1.0` only under an explicit
   `empty_policy="legacy_one"`. **AGY-8 / Opus.**

3. **`scripts/train_self_audit.py:425` (AGY-7)** reads `val_stats["primary_metric"]` for Phase-B
   checkpoint selection and so inherits the new on-policy semantics automatically — correct, and
   no edit is needed. Two things AGY-7 may want: it does not construct
   `check_generation_tolerance` for its own generator at line 383 (same 0.02 vs 0.005 conflict,
   unsurfaced there), and its `_log` at line 424 will now carry ~42 extra `phase_b/audit/*`
   keys plus the string `phase_b/primary_metric_source`.

4. **`src/self_audit/training/train_auditor.py:588` visualize block** rebuilds transitions and
   re-runs the auditor outside `_auditor_batch`, duplicating the forward. It is inside my file
   and I left it alone — it is `--visualize`-only, does not feed any metric, and touching it is
   outside this pass's scope. Worth folding into the shared path in a later hygiene pass.

## 7. Guardrails — confirmed not violated

No change to `AnnotationExpert` or `Auditor` architecture; no auditor decoder; stage embeddings
untouched; no local evidence into the gate; Phase-A objective and `stage_weights` untouched; A0
untouched; `CounterfactualGenerator` and its `epsilon_neutral` default untouched; nothing
retrained; the 80/20 ACDC split untouched and never called "test"; no SOTA claims; legacy
checkpoint loading untouched. No `git add` / `commit` / `checkout` / `stash` / `push` was run.
