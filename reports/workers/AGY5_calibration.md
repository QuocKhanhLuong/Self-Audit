# AGY-5 — Threshold calibration semantics + persisted artifact

**Files changed (only these):**

- `src/self_audit/evaluation/threshold.py`
- `scripts/calibrate_threshold.py`

**New file created:** none in `src/`/`scripts/` beyond the two above. This report is the only new file.

Scope was measurement only. The gate simulation (per-sample halting, `final += actual` on accepted
turns) is byte-for-byte the pre-existing logic and remains faithful to `SelfAuditNet.infer`. No
training path, objective, generator parameter, split, or checkpoint-loading behaviour was touched.

---

## 1. What changed and why

### 1.1 Neutral-aware decision counting (`evaluate_threshold`)

Before, at `threshold.py:85-86`:

```python
harmful_total += int((accepted & (actual[:, turn] <= 0.0)).sum())
beneficial_rejection_total += int((rejected & (actual[:, turn] > 0.0)).sum())
```

Two defects. `ΔDice == 0` — the modal outcome, since a large share of candidate edits change no
pixel — was counted as a *harmful acceptance*; and the two counters used opposite boundary
conventions (`<= 0` vs `> 0`), so an exactly-zero transition was harmful when accepted but not
beneficial when rejected. The denominators were the full accepted/rejected totals, so the neutral
mass inflated the numerator *and* the denominator of `harmful_acceptance_rate` asymmetrically.

`harmful_acceptance_rate` is the second element of `select_threshold`'s lexicographic key, so this
bias fed straight into threshold selection: whenever two grid points tied on `final_macro_dice`
(common, because neutral transitions contribute exactly `0.0` to the final Dice either way), the old
rate preferred the τ that *rejected* the neutral edits — a systematically more conservative τ than
the project's own neutral semantics warrant. §3.2 below demonstrates that shift on a worked case.

Now the actual deltas are classified once with `classify_delta` from the shared contract, and:

- `harmful_total` counts accepted transitions that are strictly `HARMFUL` (`delta < -margin`);
- `beneficial_rejection_total` counts rejected transitions that are strictly `BENEFICIAL`
  (`delta > +margin`) — symmetric with the above;
- **both rates exclude neutral transitions from their denominators**:
  `harmful / max(accepted_non_neutral, 1)` and `beneficial_rejection / max(rejected_non_neutral, 1)`;
- `neutral_acceptance_rate = neutral_accepted / max(accepted_total, 1)` is emitted so the excluded
  population is visible rather than merely absent;
- raw counts are emitted so any caller can re-aggregate without re-deriving denominators.

The boundary follows the shared contract exactly: `abs(delta) == margin` is NEUTRAL (closed on the
neutral side). Verified in §3.5.

`final_macro_dice`, `net_dice_gain`, `mean_attempted_turns`, `mean_accepted_turns` and
`acceptance_rate` are computed by the unchanged code and are numerically identical (§3.1).

### 1.2 Self-describing selection (`select_threshold`)

The lexicographic key `(final_macro_dice, -harmful_acceptance_rate, -abs(tau_accept))` is unchanged.
The returned dict now additionally carries `objective` (`SELECTION_OBJECTIVE`, a human-readable
statement of what was maximised, including the fact that selection and reporting share one split),
`n_rows` (how many grid points the argmax ranged over — the size of the selection bias),
`neutral_margin`, and `grid` (`min`/`max`/`steps` recovered from the rows). The plan text asks for
`grid` as well as the three the task spec names, so all four are present.

### 1.3 Calibration artifact, schema v1 (new)

Previously `scripts/calibrate_threshold.py` wrote an ad-hoc `calibration.json` that **nothing read
back**. Every reported self-audit Dice therefore ran at the hard-coded `tau_accept = 0.0` from config
while the pipeline separately reported an optimistically-selected `final_macro_dice` that no deployed
configuration ever used. `threshold.py` now owns both halves of that contract:

```
CALIBRATION_SCHEMA_VERSION = 1
VALIDITY_DIAGNOSTIC_ONLY   = "diagnostic_only"
VALIDITY_REASON            = <verbatim paragraph, see §2>
SELECTION_OBJECTIVE        = <verbatim string, see §2>
```

`save_calibration` writes exactly these top-level keys: `schema_version`, `tau_accept`,
`neutral_margin`, `source_split`, `checkpoint_path`, `checkpoint_sha256`, `t_max`, `threshold_grid`
(`{min, max, steps}`), `selected_row`, `created_at` (UTC ISO-8601), `metric_space`, `validity`,
`validity_reason`, `extra`.

- `validity` is always the literal `"diagnostic_only"`.
- `validity_reason` states in plain words that τ was selected by argmax on the split it was measured
  on, that the result is therefore an optimistically-biased diagnostic and not held-out evidence, and
  that this checkpoint has no independent test set. It travels **inside the artifact**, so the caveat
  cannot be separated from the number.
- `checkpoint_sha256` is computed when the file exists and is readable; otherwise `null`. A missing
  or unreadable checkpoint never fails the write.
- `metric_space` is validated against `METRIC_SPACES` from the shared contract, so a typo cannot
  produce an unlabelled metric space.

`load_calibration` validates strictly and **raises** on: a `schema_version` that is not exactly `1`,
any unknown top-level key, any missing required key, or a `validity` other than `"diagnostic_only"`.
Nothing is silently ignored — an artifact this loader does not fully understand must not be allowed
to hand a threshold to inference.

### 1.4 `scripts/calibrate_threshold.py`

Writes the artifact via `save_calibration`. New optional flags: `--neutral_margin` (forwarded to
`sweep_thresholds` and recorded), `--source_split` (default `"val"`), `--checkpoint` (SHA-256 recorded
when readable), `--metric_space` (default `slice_proxy`, the space `_foreground_dice_per_sample`
actually produces). Every pre-existing flag keeps its name, type and default; **the default grid is
unchanged** (`-0.20 … 0.20`, 41 points). `t_max` is derived from `delta_q.shape[1]` of the cache.
The old artifact's `source` / `constraint` / `rows` payload is preserved under `extra`.

No new split was created and the validation split was not renamed.

---

## 2. Exact new / changed public signatures

```python
# src/self_audit/evaluation/threshold.py

CALIBRATION_SCHEMA_VERSION: int = 1
VALIDITY_DIAGNOSTIC_ONLY: str = "diagnostic_only"
VALIDITY_REASON: str          # verbatim diagnostic-only statement
SELECTION_OBJECTIVE: str      # verbatim statement of the selection key

def evaluate_threshold(
    tau_accept: float,
    initial_dice: Any,
    delta_q: Any,
    actual_delta_dice: Any,
    active_mask: Any | None = None,
    *,
    neutral_margin: float | None = None,      # NEW, keyword-only
) -> dict[str, float]: ...

def sweep_thresholds(
    transitions: Mapping[str, Any],
    thresholds: Iterable[float],
    *,
    max_harmful_acceptance_rate: float | None = None,
    neutral_margin: float | None = None,      # NEW, keyword-only
) -> list[dict[str, float]]: ...

def select_threshold(rows: Iterable[Mapping[str, float]]) -> dict[str, float]: ...
    # signature unchanged; result gains objective, n_rows, neutral_margin, grid

def save_calibration(                          # NEW
    path: Any,
    *,
    tau_accept: float,
    neutral_margin: float | None,
    source_split: str,
    checkpoint_path: Any = None,
    t_max: int,
    threshold_grid: Any,                       # Mapping{min,max,steps} or an iterable of taus
    selected_row: Mapping[str, Any],
    metric_space: str,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]: ...

def load_calibration(path: Any) -> dict[str, Any]: ...   # NEW
```

`evaluate_threshold` return keys — unchanged: `tau_accept`, `final_macro_dice`, `net_dice_gain`,
`harmful_acceptance_rate`, `beneficial_rejection_rate`, `mean_attempted_turns`,
`mean_accepted_turns`, `acceptance_rate`. Added: `neutral_margin`, `neutral_acceptance_rate`,
`accepted_total`, `rejected_total`, `accepted_non_neutral_total`, `rejected_non_neutral_total`,
`harmful_total`, `beneficial_rejection_total`, `neutral_accepted_total`.

**Backwards compatibility.** All four pre-existing public names still exist with their positional
signatures intact; every new parameter is keyword-only with a default. `src/self_audit/evaluation/__init__.py`
(not mine) re-exports the three original names and is unaffected. A legacy 5-positional-argument call
is exercised in §3.5.

### Behaviour change to state loudly

**`harmful_acceptance_rate` and `beneficial_rejection_rate` are not comparable to any number
recorded before this change.** They now use different numerators *and* different denominators. Any
previously logged `harmful_acceptance_rate` was inflated by the neutral mass; on the synthetic cache
in §3.4 the same τ moves from 0.329 (zero-boundary, neutral-in-denominator) to 0.318. On a
degenerate all-neutral cohort it moves from **1.0 to 0.0**. Historical W&B series for these two keys
should be treated as a different metric, not as a continuation.

Defaults: `neutral_margin=None` resolves to `DEFAULT_NEUTRAL_MARGIN = 0.005`. There was no previous
margin parameter here — the old code hard-coded a `0.0` boundary — so the *default behaviour of these
two rates changes*, deliberately, and that is the point of the task. Passing `neutral_margin=0.0`
does **not** restore the old numbers exactly: the old code counted `delta == 0` as harmful and
neutral-with-margin-0 still excludes exactly-zero deltas from both denominators. That asymmetry was
the bug, so it is not reproduced under any parameter setting.

---

## 3. Verification (real executed output)

Reproduction script: `scratchpad/verify_agy5.py`, which imports the pre-change module from
`git show a901eeae:src/self_audit/evaluation/threshold.py` as `threshold_old` so BEFORE/AFTER are
both executed, not recalled. All inputs are synthetic tensors — there is no `preprocessed_data/` and
no `weights/` in this checkout.

### 3.1 Three accepted transitions, `delta == 0.0` exactly

```
BEFORE harmful_acceptance_rate = 1.0
AFTER  harmful_acceptance_rate = 0.0
AFTER  neutral_acceptance_rate = 1.0
AFTER  counts: accepted_total=3 harmful_total=0 neutral_accepted_total=3 accepted_non_neutral_total=0
final_macro_dice unchanged: old=0.5 new=0.5 equal=True
```

### 3.2 Mixed case — the calibrated τ shifts less conservative

Two samples, one turn. Sample A: `delta_q=0.05`, actual `Δ=0.00` (a NEUTRAL no-op edit).
Sample B: `delta_q=0.20`, actual `Δ=+0.10` (BENEFICIAL). Grid `{0.0, 0.1}`.

```
OLD tau=+0.00 final_macro_dice=0.6500 harmful_acceptance_rate=0.5000
OLD tau=+0.10 final_macro_dice=0.6500 harmful_acceptance_rate=0.0000
NEW tau=+0.00 final_macro_dice=0.6500 harmful_acceptance_rate=0.0000
NEW tau=+0.10 final_macro_dice=0.6500 harmful_acceptance_rate=0.0000
OLD selected tau = +0.10  (harmful_rate 0.0000)
NEW selected tau = +0.00  (harmful_rate 0.0000)
final_macro_dice identical across both taus: [0.65]
NEW extras: objective_len=268 n_rows=2 neutral_margin=0.005 grid={'min': 0.0, 'max': 0.1, 'steps': 2}
```

Both τ deliver the same Dice — the neutral edit contributes exactly `0.0` whether accepted or
rejected. The old counting invented a 0.5 harmful rate at `τ=0.0` out of that no-op and used the
tie-break to pick the stricter `τ=+0.10`. The neutral-aware rate ties at 0.0, so the third key
`-|τ|` decides and the calibrated τ lands at `0.0`. This is the selection-bias mechanism described in
§1.1, executed.

### 3.3 Artifact round trip and strict loading

```
keys: ['checkpoint_path', 'checkpoint_sha256', 'created_at', 'extra', 'metric_space',
       'neutral_margin', 'schema_version', 'selected_row', 'source_split', 't_max',
       'tau_accept', 'threshold_grid', 'validity', 'validity_reason']
tau_accept bitwise equal: True (0x0.0p+0 vs 0x0.0p+0)
neutral_margin bitwise equal: True (0x1.d2d7c5f17a9a0p-8 vs 0x1.d2d7c5f17a9a0p-8)
validity='diagnostic_only'  metric_space='slice_proxy'  source_split='val'  t_max=1
created_at=2026-09-06T09:09:36.224015+00:00  threshold_grid={'max': 0.1, 'min': 0.0, 'steps': 2}
checkpoint_sha256=175519951d1f19d85423d40afa6e4505011081c74e166f8071d1494db7c22932

  checkpoint_path that does not exist -> sha256 null, no failure:
  checkpoint_path='/nope/missing.pt' checkpoint_sha256=None
```

The `neutral_margin` used for the round trip was the deliberately awkward `0.0071234567890123`;
both it and `tau_accept` compare bitwise-equal after the JSON round trip (hex float shown).

```
schema bump  -> ValueError: Unsupported calibration schema_version 2 in .../cal_bump.json;
                this build understands version 1 only
unknown key  -> ValueError: Calibration artifact .../cal_extra.json has unknown top-level keys:
                totally_new_field
missing key  -> ValueError: Calibration artifact .../cal_drop.json is missing: metric_space
```

`validity_reason` as written into every artifact:

> tau_accept was selected by argmax over the threshold grid on the same validation split on which its
> final_macro_dice is reported. Selecting and reporting on one split makes every number in
> selected_row an optimistically-biased diagnostic, not held-out evidence: the reported gain includes
> the selection bias of the grid search. The current checkpoint has no independent test set -- the
> project's 80/20 ACDC split has a train half and a validation half and nothing else -- so no
> unbiased estimate of this threshold's benefit exists yet. Treat this artifact as a reproducible
> record of how tau_accept was chosen and of which tau a run actually deployed, never as evidence of
> generalisation.

### 3.4 End-to-end CLI on a synthetic 64×3 transition cache

`python3 scripts/calibrate_threshold.py --transitions <synthetic>.pt --output <out>.json --num_thresholds 41`

```
{"acceptance_rate": 0.7687074829931972, "accepted_non_neutral_total": 66, "accepted_total": 113,
 "beneficial_rejection_rate": 0.0, "beneficial_rejection_total": 0,
 "final_macro_dice": 0.7190007575190975, "grid": {"max": 0.2, "min": -0.2, "steps": 41},
 "harmful_acceptance_rate": 0.3181818181818182, "harmful_total": 21,
 "mean_accepted_turns": 1.765625, "mean_attempted_turns": 2.296875, "n_rows": 41,
 "net_dice_gain": 0.01918638575443765, "neutral_acceptance_rate": 0.415929203539823,
 "neutral_accepted_total": 47, "neutral_margin": 0.005, "objective": "argmax over the threshold
 grid of the lexicographic key (final_macro_dice, -harmful_acceptance_rate, -abs(tau_accept));
 harmful_acceptance_rate excludes neutral transitions from its denominator; selection and reporting
 share one split, so the result is diagnostic only", "rejected_non_neutral_total": 24,
 "rejected_total": 34, "tau_accept": -0.03}
validity=diagnostic_only neutral_margin=0.005
saved=.../cli_calibration.json
```

Same cache with `--neutral_margin 0.0 --checkpoint README.md --source_split val --metric_space slice_proxy`:

```
"accepted_non_neutral_total": 73, "harmful_acceptance_rate": 0.3287671232876712,
"harmful_total": 24, "neutral_accepted_total": 40, "neutral_margin": 0.0,
"beneficial_rejection_rate": 0.038461538461538464, "tau_accept": -0.03
validity=diagnostic_only neutral_margin=0.0
```

41.6% of accepted transitions on this synthetic cache are neutral — that is the mass the old
denominator was silently absorbing. `final_macro_dice`, `net_dice_gain`, `acceptance_rate`,
`mean_attempted_turns` and `mean_accepted_turns` are invariant to `--neutral_margin`, as required.

Artifact top level (rows elided):

```json
{
  "checkpoint_path": null,
  "checkpoint_sha256": null,
  "created_at": "2026-09-06T09:09:46.531159+00:00",
  "extra": { "max_harmful_acceptance_rate": null, "rows": "<41 rows elided>",
             "source_transitions": ".../fake_transitions.pt" },
  "metric_space": "slice_proxy",
  "neutral_margin": 0.005,
  "schema_version": 1,
  "selected_row": { ... "tau_accept": -0.03 ... },
  "source_split": "val",
  "t_max": 3,
  "tau_accept": -0.03,
  "threshold_grid": { "max": 0.2, "min": -0.2, "steps": 41 },
  "validity": "diagnostic_only",
  "validity_reason": "<as quoted in 3.3>"
}
```

### 3.5 Margin plumbing, closed neutral boundary, legacy call shape

```
delta=-0.004 margin=0.005 -> harmful_rate=0.0 neutral_accepted=1
delta=-0.004 margin=0.001 -> harmful_rate=1.0 harmful_total=1
delta=-0.005 margin=0.005 -> harmful_rate=0.0 (boundary closed on neutral side)
legacy positional 5-arg call OK: final_macro_dice=0.5750
ALL ASSERTIONS PASSED
```

### 3.6 Test suite

```
$ python3 -m pytest tests/ -q
E   ModuleNotFoundError: No module named 'self_audit'
ERROR tests/test_audit_decomposition.py
!!!!!!!!!!!!!!!!!!!! Interrupted: 1 error during collection !!!!!!!!!!!!!!!!!!!!
```

Collection is currently interrupted by an in-flight edit in a file I do not own (see §5.1). Excluding
that one module:

```
$ python3 -m pytest tests/ -q --ignore=tests/test_audit_decomposition.py
FAILED tests/test_self_audit_hardening.py::test_metrics_use_consistent_both_empty_dice_convention
1 failed, 69 passed, 1 warning in 4.16s
```

```
$ python3 -m pytest tests/test_self_audit_hardening.py -q -k "threshold"
3 passed, 7 deselected in 0.61s
```

Both surviving problems belong to other workers' files, not to mine (§5). The three threshold tests —
including the pre-existing `test_threshold_calibration_simulates_per_sample_halting`, which asserts
`final_macro_dice == 0.575`, `mean_attempted_turns == 2.0`, `harmful_acceptance_rate == 0.0` and
`tau_accept == 0.0` — pass unmodified.

---

## 4. What I could not do

- **Nothing was consumed.** I produced a readable artifact and a strict loader, but nothing in the
  repo *reads* a calibration file yet, so the headline defect ("every reported self-audit Dice runs
  at the hard-coded `tau_accept = 0.0`") is only half closed by this worker. Wiring belongs to AGY-6
  (`--calibration` / `--tau_accept` precedence in `scripts/audit_checkpoint.py`) and AGY-7
  (`scripts/train_self_audit.py`). Both are outside my ownership. See §5.2 for the exact call shape
  they need.
- **No real-data numbers.** There is no `preprocessed_data/` and no `weights/` in this checkout, so
  every number above is from synthetic tensors. I did not estimate what τ or what harmful rate the
  real checkpoint would produce, and this report contains no such figure.
- **The selection bias is recorded, not removed.** τ is still chosen by argmax on the split it is
  reported on. Removing that requires a held-out split, which the plan defers (§5 of the plan:
  "must be created *before* a clean retraining run, not retrofitted onto this checkpoint"). The
  artifact therefore documents the bias instead of pretending it is absent.

---

## 5. Observed in files I do not own — described, not edited

### 5.1 `src/self_audit/evaluation/audit_decomposition.py` (AGY-2) — breaks test collection

At `audit_decomposition.py:27` the new import is absolute:

```python
from self_audit.audit.semantics import (...)
```

`tests/test_audit_decomposition.py:8` imports the module through the `src.` package path
(`from src.self_audit.evaluation.audit_decomposition import ...`), under which the top-level name
`self_audit` is not importable, so the whole suite fails at collection with
`ModuleNotFoundError: No module named 'self_audit'`. This is not present at `START_HEAD`
(`git show a901eeae:src/self_audit/evaluation/audit_decomposition.py` has no such import).

**Suggested fix for the owner:** use a relative import, `from ..audit.semantics import (...)`, which
works under both entry points. That is what I used in `threshold.py:34`, and `threshold.py` imports
cleanly via both `src.self_audit.evaluation.threshold` and `self_audit.evaluation.threshold`.
The same absolute-import pattern is worth checking in every wave-1 file that newly imports the shared
contract. I did not edit anything to work around it; I ran the suite with `--ignore` and reported
both results above.

### 5.2 Call shape for the τ-consumer workers (AGY-6, AGY-7)

For the record, so neither worker has to re-derive it:

```python
from self_audit.evaluation.threshold import load_calibration, save_calibration

cal = load_calibration(args.calibration)        # raises on any schema/key surprise
tau = cal["tau_accept"]                          # bitwise the calibrated value
margin = cal["neutral_margin"]
assert cal["validity"] == "diagnostic_only"      # already enforced by the loader
```

Everything needed to stamp `"evidence_class": "diagnostic_only"` with a reason is on the loaded dict
(`validity`, `validity_reason`, `source_split`, `checkpoint_sha256`, `metric_space`, `t_max`,
`threshold_grid`, `selected_row["n_rows"]`, `selected_row["objective"]`). Please echo the resolved τ
*and its source* into the report JSON, per the plan's precedence rule
(explicit `--tau_accept` > `--calibration` > config > `0.0`).

### 5.3 `scripts/train_self_audit.py:586-593` (AGY-7)

Still writes the pre-schema `calibration.json` (`{"best", "threshold_min", "threshold_max",
"threshold_steps"}`) directly with `_save_report`, and still passes no `neutral_margin` to
`sweep_thresholds`. It should call `save_calibration` instead — it already has every argument in
scope: `t_max`, the grid bounds from `args.threshold_*`, `select_threshold`'s row, and the checkpoint
path. Note that the file it writes and the file `scripts/calibrate_threshold.py` writes have the same
name, so leaving one on the old shape means two mutually incompatible `calibration.json` layouts in
the tree. I did not touch it.

### 5.4 `tests/test_self_audit_hardening.py::test_metrics_use_consistent_both_empty_dice_convention`

Fails with `{1: nan, 2: nan, 3: nan} == {1: 1.0, 2: 1.0, 3: 1.0}` — the intended consequence of
AGY-1's `empty_policy="exclude"` default in `evaluation/metrics.py`. The existing test encodes the
old `empty_score=1.0` convention. Someone (AGY-1 or AGY-8) needs to decide whether this test is
updated or superseded by the new test file; I did not edit it, per the ownership rule.

### 5.5 `src/self_audit/evaluation/__init__.py`

Re-exports only `evaluate_threshold`, `select_threshold`, `sweep_thresholds`. If the integrator wants
`save_calibration` / `load_calibration` / `CALIBRATION_SCHEMA_VERSION` reachable as
`self_audit.evaluation.*`, that file needs three names added. It is not mine to edit; importing from
`self_audit.evaluation.threshold` works today without any change.

---

## 6. Guardrail compliance

| Guardrail | Status |
|---|---|
| No auditor/expert architecture change | No model file touched |
| No full-resolution decoder, no stage-embedding removal | Not touched |
| No local evidence into the gate | Not touched |
| Phase-A objective / `stage_weights` unchanged | Not touched |
| A0 not weakened | Not touched |
| Generator / `epsilon_neutral` unchanged | Not touched |
| No retraining | None run |
| 80/20 ACDC split unchanged, no subset called "test" | No split created or renamed; artifact records `source_split="val"` |
| No SOTA claims | None made; every number here is synthetic and labelled as such |
| Legacy checkpoint loading semantics | Not touched |
| Training behaviour bit-identical | Yes — the gate simulation loop is unchanged; only classification, reporting keys and artifact I/O changed |
