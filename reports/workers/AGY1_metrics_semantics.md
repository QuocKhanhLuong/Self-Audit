# AGY-1 — Metric Semantics, Neutral Margin, Empty-Class Policy

**File owned and edited:** `src/self_audit/evaluation/metrics.py` (only file touched).
**START_HEAD:** `a901eeae56cf6835bb9a1d9c79a9d5d3e99374c6`
**Scope:** measurement only. No training objective, no generator, no architecture, no checkpoint semantics touched.

---

## 1. What changed and why

### 1.1 One decision rule for a signed delta

Every signed-delta classification in this module now routes through
`self_audit.audit.semantics.classify_delta` / `beneficial_mask` / `neutral_mask` /
`harmful_mask`. There is no second copy of the rule left in the file.

The old code used `delta > 0.0` as the improve/regress boundary while the audit loss
used `0.005`, and it used two *different* zero-boundaries in the same function
(`delta <= 0.0` for harmful, `delta > 0.0` for beneficial), which made `ΔDice == 0`
count as harmful but never as beneficial.

Import is **relative** (`from ..audit.semantics import ...`) on purpose — see §6.1.

### 1.2 `transition_audit_metrics` — neutral transitions leave the ranking population

The ranking task is *beneficial vs harmful*. A NEUTRAL transition is neither class, so
folding it into the negative class (which `actual_sign = delta_dice > 0.0` did) scored a
task the loss never trained. Neutrals are now dropped from AUROC / AUPRC / accuracy, and
the number dropped is reported so the exclusion is never invisible.

`correlation_delta_q_delta_dice` stays on the **full** population (a correlation of
continuous values needs no margin). `local_fix_f1` / `local_regress_f1` are pixel-label
F1 and are untouched by the margin, as specified.

AUROC/AUPRC keep the **existing** `binary_auroc` / `binary_auprc` behaviour on a
degenerate population: AUROC is `nan` when the non-neutral population is empty or
single-class; AUPRC is `nan` when it contains no positives and stays finite (`1.0`) when
it is all-positive. No value is invented, and the existing hardening test that asserts
`np.isfinite(auprc)` on a single beneficial transition still passes.

### 1.3 `acceptance_metrics` — neutral excluded from the denominators

`harmful = accepted & harmful_mask(delta)`, `beneficial_rejection = rejected &
beneficial_mask(delta)`, with denominators `max((accepted & ~neutral).sum(), 1)` and
`max((rejected & ~neutral).sum(), 1)` exactly as specified.
`net_dice_gain_after_auditing` is a raw sum over accepted transitions and is unchanged.

**See §5 — the two acceptance criteria for this function contradict each other.** The
implemented formula is the one written in the plan and in the task's "EXACTLY WHAT TO
IMPLEMENT"; the raw counts needed to recover the other reading are returned as well.

### 1.4 Empty-class policy

`empty_policy: str | None = None` (resolving to `"exclude"`) added to `per_class_dice`,
`dice_score`, `per_class_precision_recall`, `annotation_metrics`, `slice_proxy_dice`.

* A class empty in **both** prediction and target takes the policy score
  (`"exclude"` → `nan`, `"legacy_one"` → `1.0`, `"zero"` → `0.0`).
* A class present in exactly one of them scores **0.0** and is **never excluded**, under
  every policy. Verified in both directions (§4.4).
* Macros go through nan-aware `macro_mean`. All classes excluded ⇒ macro is `nan`, not
  `1.0` and not `0.0`.
* `per_class_dice` returns the per-class floats *including* `nan`, so a caller can
  aggregate across cases without double counting.

`surface_metrics(empty_score=0.0)` was **not** touched: that parameter is a distance, not
a Dice value.

### 1.5 `slice_proxy_dice` — the single 2-D training proxy

New. Per-**sample** foreground macro Dice, shape `[B]`. Accepts `[B,C,H,W]`
logits/probabilities (argmax over the class axis) or `[B,H,W]` labels; a bare `[H,W]`
input is promoted to a batch of 1. Metric space is `slice_proxy`
(`metrics.SLICE_PROXY_METRIC_SPACE`), documented in the module docstring as a
training/monitoring proxy, **not** a paper metric.

Because it is computed per sample, it is exactly batch-partition invariant, unlike
pooling confusion counts over a whole `[B,H,W]` block (§4.7, §4.8). AGY-7 wires Phase A
and Phase C to it; this worker only provides it.

---

## 2. Public signatures (new / changed)

```python
# changed — keyword-only additions, all positional signatures preserved
per_class_dice(prediction, target, num_classes=4, include_background=False,
               empty_score=_UNSET, *, empty_policy=None) -> dict[int, float]

dice_score(prediction, target, num_classes=4, include_background=False,
           empty_score=_UNSET, *, empty_policy=None) -> float

per_class_precision_recall(prediction, target, num_classes=4, include_background=False,
                           empty_score=_UNSET, *, empty_policy=None)
    -> tuple[dict[int, float], dict[int, float]]

annotation_metrics(prediction, target, num_classes=4, spacing=None, spacing_known=None,
                   *, empty_policy=None, empty_score=_UNSET) -> dict[str, Any]

transition_audit_metrics(predicted_local, target_local, predicted_delta_q,
                         actual_delta_dice, *, neutral_margin=None, tau=0.0)
    -> dict[str, Any]

acceptance_metrics(accepted, actual_delta_dice, turns=None, *, neutral_margin=None)
    -> dict[str, Any]

# new
slice_proxy_dice(prediction, target, *, num_classes=4, empty_policy=None) -> np.ndarray  # [B]
SLICE_PROXY_METRIC_SPACE = "slice_proxy"
```

`empty_score` default is a private `_UNSET` sentinel, so *passing* it (positionally or by
keyword) still works and still wins over `empty_policy`, and it now emits a
`DeprecationWarning` naming `empty_policy`. When passed, the pre-remediation behaviour is
reproduced **exactly**: any zero denominator — both-empty *and* one-sided-empty in
precision/recall — takes that value.

`transition_audit_metrics` / `acceptance_metrics` return annotations widened from
`dict[str, float]` to `dict[str, Any]` because the new count fields are `int`.

**New keys returned**

* `transition_audit_metrics`: `neutral_count`, `beneficial_count`, `harmful_count`,
  `ranking_count`, `transition_count`, `neutral_margin`, `tau`.
* `acceptance_metrics`: `neutral_count`, `beneficial_count`, `harmful_count`,
  `accepted_count`, `rejected_count`, `accepted_signed_count`, `rejected_signed_count`,
  `harmful_accepted_count`, `beneficial_accepted_count`, `neutral_accepted_count`,
  `beneficial_rejected_count`, `transition_count`, `neutral_margin`,
  `neutral_acceptance_rate`, `beneficial_acceptance_rate`.
* `annotation_metrics`: `empty_policy`.

Rate definitions, stated so no one has to reverse-engineer the denominator:

| key | numerator | denominator |
|---|---|---|
| `harmful_acceptance_rate` | accepted ∧ harmful | accepted ∧ ¬neutral |
| `beneficial_acceptance_rate` | accepted ∧ beneficial | accepted ∧ ¬neutral |
| `beneficial_rejection_rate` | rejected ∧ beneficial | rejected ∧ ¬neutral |
| `neutral_acceptance_rate` | accepted ∧ neutral | **all** accepted |

`harmful_acceptance_rate + beneficial_acceptance_rate == 1` whenever any materially
signed transition was accepted.

---

## 3. LOUD: the default reported number changes

**Headline Dice goes DOWN.** Previously a foreground class absent from both prediction
and ground truth scored `1.0`. On ACDC, apical and basal slices routinely contain no RV
and no MYO, so a 2-D per-slice macro was being handed free `1.0`s. Under the new default
those classes are excluded, and every macro is a nan-aware mean over the classes that
actually existed. Every Dice, precision and recall number produced by this module with
default arguments will be **lower than the previously reported one**, and in the
all-empty case it is `nan` rather than `1.0`.

**Direction of the two audit metrics.** `harmful_acceptance_rate` stops counting
`ΔDice == 0` as harmful, which removes an upward bias; it simultaneously stops diluting
the denominator with no-ops, which is an upward pressure. The net direction is
data-dependent, not predictable a priori — see §5.

**AUROC changes meaning.** It now scores beneficial-vs-harmful, the task the loss trains,
instead of a `delta > 0` split that put every no-op edit in the negative class.

**`nan` is now reachable in returned dicts** (per-class values, macros, AUROC/AUPRC).
Anything downstream that JSON-serialises these dicts must handle `nan`
(`json.dumps(float("nan"))` emits bare `NaN`, which is not valid JSON for strict
parsers). Flagged for AGY-6.

---

## 4. Verification — real pasted output

All checks ran on synthetic tensors (no `preprocessed_data/`, no `weights/` in this
checkout). Script:
`/private/tmp/claude-501/-Users-alvinluong-Self-Audit/0af8f10f-cd0c-4ab9-ad2b-6f26bddfd2be/scratchpad/verify_agy1.py`

### 4.1 MANDATORY neutral acceptance example

```
=== 1. MANDATORY neutral acceptance example ===
  beneficial_count = 0
  harmful_count = 0
  neutral_count = 3
  harmful_acceptance_rate = 0.0
  beneficial_rejection_rate = 0.0
  neutral_acceptance_rate = 1.0
  accepted_count = 3
  net_dice_gain_after_auditing = 0.0
```

`accepted=[True,True,True]`, `delta=[0.0,+0.001,-0.001]`, `eps=0.005` — required values
met exactly.

### 4.2 The old harmful example

```
=== 2. Old harmful example ===
  harmful_acceptance_rate (spec formula, denom = accepted & ~neutral) = 1.0
  harmful_accepted_count = 1  accepted_count = 5  accepted_signed_count = 1
  derived all-accepted-denominator rate = harmful_accepted_count/accepted_count = 0.2
  beneficial_rejection_rate = 1.0 (rejected={-1: idx5 delta=+0.20} -> 1/1)
  neutral_count = 4  beneficial_count = 1  harmful_count = 1
  OLD CODE would have given: harmful=accepted&(delta<=0) = 4/5 = 0.8
```

The `0.8` bug is gone: exactly one of the five accepted transitions is harmful, not four.
See §5 for the `1.0` vs `0.2` question.

### 4.3 / 4.4 / 4.5 Empty-class policy

```
=== 3. per_class_dice empty-class policy ===
  exclude   : {1: nan, 2: nan, 3: nan}
  legacy_one: {1: 1.0, 2: 1.0, 3: 1.0}
  zero      : {1: 0.0, 2: 0.0, 3: 0.0}
  dice_score(exclude)    = nan
  dice_score(legacy_one) = 1.0

=== 4. class present in GT, predicted empty -> 0.0 under BOTH policies (never excluded) ===
  exclude   : {1: 0.0, 2: nan, 3: nan}
  legacy_one: {1: 0.0, 2: 1.0, 3: 1.0}
  reverse (predicted but absent in GT): {1: 0.0, 2: nan, 3: nan}

=== 5. precision/recall empty handling ===
  exclude    precision: {1: 0.0, 2: nan, 3: nan}  recall: {1: 0.0, 2: nan, 3: nan}
  legacy_one precision: {1: 0.0, 2: 1.0, 3: 1.0}  recall: {1: 0.0, 2: 1.0, 3: 1.0}
```

A total miss scores `0.0` and survives into the macro under every policy — in both
directions (class in GT but not predicted, and class predicted but not in GT).

Note the precision/recall consequence, which the spec did not spell out and which I chose
deliberately: an empty prediction for a class that exists in GT used to give
**precision 1.0**; it now gives 0.0. Same for recall on a hallucinated class. That matches
Dice and stops a class the model never predicts from inflating precision. Legacy is one
keyword away.

### 4.6 Deprecated override

```
=== 6. deprecated empty_score override reproduces legacy exactly ===
  value: {1: 1.0, 2: 1.0, 3: 1.0}
  warning: DeprecationWarning -> per_class_dice(empty_score=...) is deprecated and overrides empty_policy; pass e ...
```

### 4.7 `slice_proxy_dice` batch-partition invariance

```
=== 7. slice_proxy_dice batch-partition invariance ===
  full  : [0.204582 0.234141 0.316648 0.351408 0.294624 0.287824 0.215403]
  1+2   : [0.204582 0.234141 0.316648 0.351408 0.294624 0.287824 0.215403]
  identical (bitwise): True
  per-sample concat identical: True
  metric space: slice_proxy
  labels-in accepted: [0.204582 0.234141 0.316648 0.351408 0.294624 0.287824 0.215403]
  all-background sample -> [nan nan]
```

`slice_proxy_dice(x) == concat(slice_proxy_dice(x[:3]), slice_proxy_dice(x[3:]))`
bitwise, and equal to the concatenation of seven single-sample calls. `[B,C,H,W]` logits
and `[B,H,W]` argmax labels give the identical array.

### 4.8 Why the proxy exists — pooling really does diverge

```
pooled dice_score over [B,H,W] = 0.673469387755102
slice_proxy_dice per sample    = [1.         0.66666667]  mean = 0.8333333333333333
```

Two samples, same data: pooling confusion counts over the batch weights by pixel count
(`0.6735`), per-sample macro weights samples equally (`0.8333`). Phase A currently uses
the pooled form, so its headline is batch-size dependent; Phase C uses the per-sample
form. Same shape of mismatch the audit reported.

### 4.9 `transition_audit_metrics`

```
=== 9. transition_audit_metrics: neutrals dropped from ranking ===
  improve_regress_accuracy = 0.6666666666666666
  auroc = 0.5
  auprc = 0.5
  correlation_delta_q_delta_dice = -0.07454312343673752
  neutral_count = 3
  beneficial_count = 1
  harmful_count = 2
  ranking_count = 3
  transition_count = 6
  neutral_margin = 0.005
  tau = 0.0
  local_fix_f1 = 1.0
  local_regress_f1 = 1.0
  non-neutral rows: idx0(+0.20,B) idx1(-0.30,H) idx5(-0.40,H); scores 0.5,-0.4,0.9
  hand AUROC = P(score_B > score_H) = 0.5 vs {-0.4,0.9} -> (1+0)/2 = 0.5
  hand accuracy at tau=0: pred [T,F,T] vs actual [T,F,F] -> 2/3 = 0.6666666666666666
```

`delta_q = [0.5,-0.4,0.1,-0.2,0.3,0.9]`, `delta_dice = [0.20,-0.30,0.001,-0.002,0.0,-0.40]`.
Both AUROC and accuracy match the hand computation on the 3-row non-neutral population.

Degenerate populations and `tau`:

```
=== 10. single-class / empty ranking population ===
  all-neutral -> ranking_count = 0  auroc = nan  auprc = nan  accuracy = nan
  single beneficial -> auroc = nan  auprc = 1.0  (auprc stays finite: existing binary_auprc behaviour, preserved)

=== 11. tau parameter ===
  tau=0.25 accuracy = 0.6666666666666666 (pred [F,F,T] vs actual [T,F,T] -> 2/3)
```

### 4.10 `annotation_metrics` and the margin boundary

```
=== 12. annotation_metrics macro is nan-aware ===
  per_class dice: {1: 1.0, 2: nan, 3: nan}
  dice = 1.0  precision = 1.0  recall = 1.0  empty_policy = exclude
  all-empty case -> dice = nan (was 1.0 before this change)
  all-empty legacy_one   -> dice = 1.0

=== 13. margin boundary is closed on the neutral side ===
  neutral_count = 2  beneficial_count = 1  harmful_count = 1
```

`delta = [0.005, -0.005, 0.0051, -0.0051]` — `abs(delta) == eps` is NEUTRAL, closed on the
neutral side, matching the shared contract.

### 4.11 Full test suite

```
$ python3 -m pytest tests/ -q
tests/test_audit_decomposition.py:8: in <module>
    from src.self_audit.evaluation.audit_decomposition import (
src/self_audit/evaluation/audit_decomposition.py:27: in <module>
    from self_audit.audit.semantics import (
E   ModuleNotFoundError: No module named 'self_audit'
ERROR tests/test_audit_decomposition.py
!!!!!!!!!!!!!!!!!!!! Interrupted: 1 error during collection !!!!!!!!!!!!!!!!!!!!
```

That collection error is **not mine** — it is an in-flight edit in AGY-2's file
(`evaluation/audit_decomposition.py:27`), see §6.1. Excluding that one module:

```
$ python3 -m pytest tests/ -q --ignore=tests/test_audit_decomposition.py
........................F.............................................   [100%]
FAILED tests/test_self_audit_hardening.py::test_metrics_use_consistent_both_empty_dice_convention
E       assert {1: nan, 2: nan, 3: nan} == {1: 1.0, 2: 1.0, 3: 1.0}
1 failed, 69 passed, 1 warning in 3.29s
```

Exactly one failure, and it is the **intended** default change: that test pins the old
`empty_score=1.0` convention this task was sent to remove. I do not own `tests/`, so I
left it. AGY-8 or the integrator should update it to
`annotation_metrics(..., empty_policy="legacy_one")` for the legacy assertion, plus a new
assertion that the default is `nan`.

`test_self_audit_hardening.py::test_transition_metrics_accept_cpu_bfloat16_outputs`
(the bfloat16 / finite-AUPRC test) **passes** unchanged.

---

## 5. Contradiction in my own spec — resolved, and flagged

The task gave two mutually exclusive requirements for `acceptance_metrics`:

* **"EXACTLY WHAT TO IMPLEMENT" (2)**, matching the plan verbatim:
  `harmful_acceptance_rate = harmful.sum() / max((accepted & ~neutral).sum(), 1)`.
* **ACCEPTANCE**, last bullet: *"old harmful example now returns 0.2 not 0.8"*.

On `accepted=[T,T,T,T,T,F]`, `delta=[0,0,0,+1e-9,-0.20,+0.20]`, `eps=0.005`, four of the
five accepted transitions are neutral. The formula gives `1/1 = 1.0`. The `0.2` figure
requires the denominator to be **all** accepted transitions (`1/5`) — i.e. neutrals kept
in the denominator, which is what the prose forbids two lines earlier. `0.2` is the
number quoted in the audit's damage estimate, written before the exclude-neutral
denominator was settled; the formula is stated twice (task and plan) and the plan is
declared authoritative.

**I implemented the formula (`1.0`).** To make the disagreement recoverable rather than
baked in, `acceptance_metrics` also returns `harmful_accepted_count` and `accepted_count`,
so `0.2` is one division away for any caller (`1/5`, printed in §4.2) and no information
is lost either way. If the integrator wants the all-accepted denominator as the headline,
it is a one-line change in one place — say so and I will make it.

---

## 6. Things I noticed in files I do NOT own (described, not edited)

### 6.1 `src/self_audit/evaluation/audit_decomposition.py:27` — import style breaks the tests

It uses the **absolute** form `from self_audit.audit.semantics import ...`. The test suite
imports the package as `src.self_audit.*` (`tests/test_audit_decomposition.py:8`) with the
repo root on `sys.path` and `src/` *not* on it, so `self_audit` is not importable and the
whole module fails to collect. This currently aborts `pytest tests/` at collection.

`metrics.py` uses the **relative** form `from ..audit.semantics import ...`, which works
under both `self_audit.*` and `src.self_audit.*` and also avoids instantiating the module
twice. AGY-2 should switch to the relative import. (Same risk applies to any other worker
adding an absolute `self_audit.` import — `threshold.py` and `volume_inference.py` are
also modified in the working tree; I did not inspect them.)

Note that `..audit.semantics` executes `self_audit/audit/__init__.py`, which imports
`counterfactual`/`gate`/`targets` and therefore torch. `metrics.py` was previously
torch-free at import time. This is unavoidable for any import of the shared contract
short of bypassing the package, and torch is already a hard dependency of the training
and volume-inference paths; noting it because the module docstring advertises numpy-only
operation.

### 6.2 `training/finetune_joint.py:437-460` — hand-built fallback dicts will go stale

When there are no transitions, that file constructs literal fallback dicts for
`acceptance` and `audit_metrics` containing only the old key set. Any downstream consumer
that starts reading the new keys (`neutral_count`, `ranking_count`, …) will hit a
`KeyError` on the empty path. AGY-7 should replace those literals with calls to the real
functions on empty tensors, which now return a complete, correctly-shaped zero dict.

### 6.3 `training/train_annotation.py:242` — the Phase-A batch-size dependence

`per_class_dice(final_logits.argmax(dim=1), batch["mask"], num_classes=...)` pools over
the whole `[B,H,W]` batch, which is exactly the divergence measured in §4.8. It is now a
one-line swap to `slice_proxy_dice(...)` plus `np.nanmean`. That is AGY-7's wiring, not
mine.

**This one needs attention before the next Phase-A run, and it is the highest-priority
item in this section.** `train_annotation.py:246-251` accumulates
`dice_sums[cls] += float(value) * batch_size` and then reports
`val_dice_class_{cls}`. With the new default policy `value` is `nan` whenever a class is
absent from both prediction and mask in that batch — on ACDC that is common — and a single
`nan` poisons the running sum for the whole epoch, so `val_dice_class_*` and the derived
macro become `nan` permanently. Anything selecting a best checkpoint on that macro would
break. The accumulation must become a nan-aware sum with a per-class present-count
denominator (or go through `slice_proxy_dice` + `np.nanmean`). Verified by reading the
code, not executed — no data in this checkout to run Phase-A validation against. Confirmed
by inspection that this path is validation logging only: `phase_a_loss` is computed
separately at line 236 and no gradient flows through `per_class_dice`, so training
behaviour stays bit-identical.

### 6.4 `tests/test_self_audit_hardening.py:89-94` — see §4.11.

---

## 7. What I could NOT do

* **Could not fix the failing test.** `tests/` is not mine. Documented above with the
  exact edit needed.
* **Could not fix the `pytest` collection abort.** `audit_decomposition.py` is AGY-2's.
  Documented above with the exact edit needed.
* **Could not run anything on real data.** No `preprocessed_data/` and no `weights/` in
  this checkout, so every number in §4 is from synthetic tensors, as instructed. No real
  Dice or AUROC figure is claimed anywhere in this report.
* **Did not resolve §5 unilaterally beyond implementing the authoritative formula** — the
  integrator's call.

## 8. Guardrails

No architecture change, no auditor decoder, no stage-embedding change, no gate change, no
Phase-A objective or `stage_weights` change, no generator or `epsilon_neutral` change, no
retraining, no split change, nothing renamed "test", no SOTA claim, no checkpoint-loading
change. Training behaviour is bit-identical: nothing in this module is called inside a
training loss path — `train_annotation.py` calls `per_class_dice` for *logging* only, and
even there the reported value changes, not the gradient.
