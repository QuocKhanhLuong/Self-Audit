# Remediation Plan — Metrics & Evaluation Hardening (Pass 1)

**Repository:** `QuocKhanhLuong/Self-Audit`
**START_HEAD (verified):** `a901eeae56cf6835bb9a1d9c79a9d5d3e99374c6` (`main`, working tree clean apart from untracked audit reports)
**Planner / integrator:** Claude Opus 5
**Source documents:** `reports/opus5_model_logic_audit.md`, `reports/opus5_logic_summary.md`
**Scope:** measurement correctness only. **No architecture redesign, no retraining, no new data split.**

**Environment verified:** torch 2.13.0 · numpy 2.4.6 · pytest 9.1.1 · matplotlib 3.11.1 · PyYAML 6.0.3 (installed during Phase 0 — a pre-existing config test failed on its absence, not on logic). No `preprocessed_data/` and no `weights/` in this checkout, so every acceptance test must run on synthetic tensors.

---

## 0. Two decisions Opus had to make before workers could start

### Decision 1 — the `epsilon_neutral = 0.02` question: **A, generation-specific**

`epsilon_neutral` is consumed only inside `CounterfactualGenerator._hard_neutral_single` (`audit/counterfactual.py:452-493`) as the **retry-loop search tolerance**: keep resampling a combined repair+regression edit until the measured `|ΔDice|` falls below it, otherwise return the closest attempt. That is a *generation-time* parameter. `neutral_margin` is a *decision/reporting* margin applied to an already-existing transition. Different roles; **do not unify them into one number.**

They are not independent, however. At `epsilon_neutral = 0.02` vs `neutral_margin = 0.005` the generator emits samples it labels "hard neutral" whose delta the loss ranks as strictly beneficial or harmful (`losses/audit.py:20,23`). That is a genuine semantic conflict.

**Resolution in this pass (guardrail: do not redesign the generator):** document the distinction, surface the conflict loudly via `check_generation_tolerance()`, and **do not change the numeric default**. Changing `epsilon_neutral` alters generated training data, which is a training-behaviour change and belongs to the generator redesign pass. Recorded as a deferred research decision (§5).

### Decision 2 — worker decomposition is by **file ownership**, not by concern

The brief's seven concern-oriented workers collide badly: `metrics.py` is touched by both the epsilon and the empty-class concern; `audit_decomposition.py` by both epsilon and headroom; `audit_checkpoint.py` by three concerns at once. Concurrent edits to one file by independent workers would produce conflicts Opus would then have to reconcile blind.

The concerns are preserved verbatim; the **ownership** is remapped so every file has exactly one owner, and cross-file dependencies are resolved by landing a shared contract module **before** any worker starts.

---

## 1. Wave 0 — shared contract (Opus, already landed)

**New file: `src/self_audit/audit/semantics.py`** (+ re-exports in `src/self_audit/audit/__init__.py`).

This is the single source of truth every worker imports, so no worker depends on another worker's diff.

| Symbol | Contract |
|---|---|
| `DEFAULT_NEUTRAL_MARGIN = 0.005` | the one decision margin |
| `classify_delta(delta, eps=None) -> ndarray` | `+1` beneficial, `0` neutral, `-1` harmful; boundary **closed on the neutral side** (`abs(delta) == eps` is NEUTRAL) |
| `beneficial_mask` / `neutral_mask` / `harmful_mask` | elementwise convenience wrappers |
| `resolve_neutral_margin(value)` | validates finite, non-negative; `None` yields the canonical value |
| `check_generation_tolerance(eps_neutral, neutral_margin)` | `RuntimeWarning` when the generator's search tolerance exceeds the decision margin |
| `DEFAULT_EMPTY_CLASS_POLICY = "exclude"` | headline empty-class policy |
| `empty_class_score(policy)` | `"exclude"` → `nan`, `"legacy_one"` → `1.0`, `"zero"` → `0.0` |
| `macro_mean(values)` | nan-aware macro; `nan` when every class was excluded |
| `METRIC_SPACE_SLICE_PROXY` / `_VOLUME_RESIZED` / `_VOLUME_NATIVE` | truthful metric-space labels |

Verified on landing:

```
classify_delta([0, +0.001, -0.001, 0.005, 0.006, -0.006], eps=0.005) -> [0, 0, 0, 0, 1, -1]
empty_class_score()            -> nan       empty_class_score("legacy_one") -> 1.0
macro_mean([nan, 0.8, 0.6])    -> 0.7       macro_mean([nan, nan])          -> nan
```

The brief's worked example is satisfied exactly: `accepted=[T,T,T]`, `delta=[0.0, +0.001, -0.001]`, `eps=0.005` gives `beneficial=0`, `harmful=0`, `neutral=3`, `harmful_acceptance_rate = 0`.

**Empty-class semantics, stated precisely** (this is the part that is easy to get wrong): only a class empty in **both** prediction and target is excluded. A class present in the target but missing from the prediction (or vice versa) scores `0.0` by the ordinary Dice formula and is **never** excluded — otherwise a total miss would silently vanish from the macro.

---

## 2. Wave 1 — five workers, disjoint files, run in parallel

| Worker | Owns exclusively | Concern (brief) |
|---|---|---|
| **AGY-1** | `src/self_audit/evaluation/metrics.py` | A (neutral semantics) + D (empty-class policy, proxy Dice) |
| **AGY-2** | `src/self_audit/evaluation/audit_decomposition.py` | B (headroom attribution + aggregation) |
| **AGY-3** | `src/self_audit/training/train_auditor.py` | C (on-policy vs synthetic) |
| **AGY-4** | `src/self_audit/evaluation/volume_inference.py` | D (volume / geometry) |
| **AGY-5** | `src/self_audit/evaluation/threshold.py`, `scripts/calibrate_threshold.py` | E (calibration artifact) |

### AGY-1 — `evaluation/metrics.py`

**Issue.** Eight sites use `0.0` as the improve/regress boundary while training uses `0.005`; `ΔDice == 0` is counted a *harmful acceptance* (measured `harmful_acceptance_rate = 0.8` where the true value is 0.2); `empty_score=1.0` default inflates every macro; `per_class_dice` pools over whatever array shape it receives, making the Phase-A headline batch-size dependent.

**Exact behavioral change.**
1. `transition_audit_metrics(..., neutral_margin=None)` — AUROC/AUPRC positive class becomes `classify_delta(...) == BENEFICIAL`; **neutral transitions are dropped from the ranking population** (they are neither class) and reported as `neutral_count`. `improve_regress_accuracy` is computed on the non-neutral population with `predicted_sign = delta_q > tau` (`tau` defaults to 0.0, explicit parameter).
2. `acceptance_metrics(..., neutral_margin=None)` — `harmful = accepted & harmful_mask(delta)`, `beneficial_rejection = rejected & beneficial_mask(delta)`. **Denominators exclude neutral transitions**: `harmful_acceptance_rate = harmful / max(accepted & ~neutral, 1)`. Also emit `neutral_acceptance_rate`, `beneficial_acceptance_rate`, `neutral_count`, `beneficial_count`, `harmful_count`, `accepted_count`, `rejected_count` so any caller can re-aggregate correctly.
3. `per_class_dice` / `dice_score` / `per_class_precision_recall` / `annotation_metrics` gain `empty_policy: str | None = None` (default `"exclude"`); `empty_score` is retained as a deprecated explicit override. Macros use `macro_mean`. Return **per-class values including `nan`** so callers can aggregate across cases without double-counting.
4. New `slice_proxy_dice(logits_or_labels, target, *, num_classes, empty_policy) -> np.ndarray [B]` — per-sample foreground macro, the **single** proxy both Phase A and Phase C will call (AGY-7 wires it).

**Acceptance tests.** Brief's neutral example → all-neutral, `harmful_acceptance_rate == 0`. `per_class_dice` on an all-empty class returns `nan` under `"exclude"` and `1.0` under `"legacy_one"`. `slice_proxy_dice` over a batch equals the concatenation of per-sample calls (batch-size invariance).

**Dependencies.** Wave 0 only. **Scientific consequence.** The reported AUROC currently scores a task the loss never trained; after this it scores the trained task. `harmful_acceptance_rate` stops being biased upward by the modal `Δ == 0` case, which in turn stops pushing calibrated τ conservative.

### AGY-2 — `evaluation/audit_decomposition.py`

**Issue.** `headroom_capture_ratio` is a mean-of-ratios with a `1e-8` guard (executed demo: returns **625000.4** where the correct ratio-of-means is **0.667**); `evaluate_audit_decomposition` averages per-batch rates unweighted and drops NaN batches (executed: **0.5625 vs 0.2222**); `stage_t/oracle_gain` is a mislabeled alias of `stage_t/headroom`; `phase_a/headroom_collapse_ratio` divides by a hardcoded `0.005`.

**Exact behavioral change.**
1. **Attribution algebra is unchanged** — `delta`, `candidate_gain = delta`, `realized = accepted * delta`, `gate_value = realized - delta`, identity `candidate_gain + gate_value == realized`. Do not touch it.
2. `evaluate_audit_modes_batch(..., return_per_sample=False)` gains a per-sample return path exposing raw tensors (`initial_dice`, `always_dice`, `self_dice`, `oracle_dice`, per-stage `delta`/`accepted`/`active`) so the loader-level function can concatenate instead of averaging scalars.
3. `evaluate_audit_decomposition` accumulates per-sample/per-transition tensors across the whole loader and computes **every rate and ratio exactly once at the end**. No per-batch means. Must be invariant to batch partition.
4. `headroom_capture_ratio` becomes a ratio of sums over material headroom: `eligible = oracle_headroom > neutral_margin`; `capture = sum(self_gain[eligible]) / sum(oracle_headroom[eligible])`; **`nan` when no eligible rows** — never `0.0`. Additionally report `headroom_available_rate`, `headroom_eligible_count`, `oracle_headroom_sum`, `self_gain_on_headroom_sum`.
5. Rename `stage_t/oracle_gain` → `stage_t/positive_headroom` (it is `clamp(delta, min=0)` on the *self-audit* trajectory, not the oracle trajectory). Keep `stage_t/headroom` as the same value for continuity; do **not** keep a key named `oracle_gain` on a non-oracle quantity.
6. `phase_a/headroom_collapse_ratio` must use the passed `neutral_margin`, not a literal.
7. Classification uses `classify_delta` from wave 0 throughout; the local `> margin` / `< -margin` logic (already correct) is replaced by the shared helper so there is one implementation.

**Acceptance tests.** The 625000 case now returns a finite ratio equal to the hand-computed ratio-of-sums; splitting one loader into batches of 1/2/4 gives bitwise-equal aggregates; `attribution_residual < 1e-6`; a zero-headroom cohort returns `nan`, not `0.0`.

**Scientific consequence.** This is the paper's central statistic. Given the audit predicts near-zero headroom on most rows, the old estimator was guaranteed to be dominated by exactly the rows it should exclude.

### AGY-3 — `training/train_auditor.py`

**Issue.** Phase-B validation calls the same GT-driven generator as training — synthetic carries **57.1%** (4 of 7 groups, verified at runtime) of the loss weight behind `val/auroc`, which is also the checkpoint-selection metric. AUROC 0.82 is reachable from `ΔP`/entropy alone, so a large share of that number is generator-fingerprint detection. No on-policy-only metric exists anywhere in the repo.

**Exact behavioral change.**
1. `build_auditor_transitions(output, ground_truth, generator, *, include_on_policy=True, include_synthetic=True)`. Defaults preserve today's training behaviour exactly. **Training is unchanged** (3 on-policy + 4 synthetic; guardrail says the generator stays as-is this pass).
2. `_auditor_batch` collects predictions/targets **partitioned by provenance** (`"on_policy"` vs `"synthetic"`), reusing the single existing encoder pass — no second forward.
3. `validate_auditor_epoch` reports three namespaces: `audit/on_policy/*`, `audit/synthetic/*`, `audit/combined/*`, each carrying `auroc`, `auprc`, `improve_regress_accuracy`, `correlation_delta_q_delta_dice`, `local_fix_f1`, `local_regress_f1`, `transition_count`.
4. **`primary_metric` = on-policy AUROC.** Documented fallback chain, applied in order and recorded in a `primary_metric_source` field: on-policy AUROC → on-policy `improve_regress_accuracy` → `nan` (never silently falls back to the combined or synthetic value).
5. Legacy flat keys (`auroc`, `auprc`, …) continue to be emitted, aliased to **combined**, so existing callers and W&B history do not break — but they are no longer used for selection.

**Acceptance test.** Assert the value driving `best.pt` is computable from on-policy transitions alone: build a validation batch where synthetic and on-policy predictions are deliberately contradictory and confirm `primary_metric` follows the on-policy half.

**Scientific consequence.** Reported AUROC will drop, probably a lot. **That drop is the finding** — it is the first honest estimate of the auditor on the task it actually faces at inference.

### AGY-4 — `evaluation/volume_inference.py`

**Issue.** `evaluate_comparison_modes` reconstructs a volume but scores it on the **resized 256×256 network grid** while nothing labels it as such; it has zero callers; ED/ES are pooled; per-class RV/MYO/LV is not broken out.

**Key Phase-0 discovery that changes this workstream.** `scripts/preprocess_acdc.py:67-88,121-148` **already writes** `orig_shape`, `orig_spacing`, `effective_spacing` and `num_slices` per case into `metadata.json`, and `data/acdc.py:84-95` already plumbs effective spacing into `ACDCRecord.spacing`. Native geometry metadata therefore **exists**.

**But native Dice is still not computable from `preprocessed_data/` alone,** and the plan must say so: the stored mask was destructively resized to 256×256 with `order=0` nearest (`preprocess_acdc.py:80`). Inverse-resizing a prediction back to `orig_shape` and comparing it against an inverse-resized copy of that same downsampled mask measures the resampler, not the model.

**Exact behavioral change.**
1. Every returned metric block carries an explicit `metric_space` field from wave 0: `"slice_proxy"`, `"volume_resized"`, or `"volume_native"`. Never conflated, never inferred.
2. `evaluate_comparison_modes` gains `empty_policy`, `neutral_margin`, per-class breakout (`RV=1`, `MYO=2`, `LV=3`) and returns `metric_space="volume_resized"` plus the recorded `orig_shape` / `orig_spacing` / `effective_spacing` **as declared metadata only**.
3. New `to_native_geometry(prediction_zhw, orig_shape, *, order=0)` — nearest-neighbour inverse resize of a label map to the original in-plane grid, preserving slice order and label values (no interpolation across labels).
4. New `evaluate_volume_native(..., native_ground_truth=...)` which computes `metric_space="volume_native"` **only when a caller supplies true native GT** (i.e. from raw ACDC NIfTI via an explicit `--raw_acdc_root`). Absent that, it must **raise or return `None`** — it must never fabricate a native number from the resized mask.
5. HD95 only when real physical spacing is present; otherwise omit the key entirely. Do not invent spacing.

**Acceptance tests.** Hand-built 3-slice volume with known per-class overlap gives the expected per-class Dice; slice order preserved through reconstruct→inverse-resize; ED/ES kept separate; an empty-empty class is excluded from the headline macro; asking for `volume_native` without native GT fails loudly rather than returning a resized number under a native label.

**Scientific consequence.** This is what decides whether any Dice in the paper is quotable. The honest current answer is `volume_resized`; `volume_native` becomes reachable only once raw ACDC is wired in, and the code now makes that distinction impossible to blur.

### AGY-5 — `evaluation/threshold.py` + `scripts/calibrate_threshold.py`

**Issue.** τ is chosen by argmax over 81 values on `val` and the maximized value is reported on that same `val`; separately the calibrated τ is **never consumed** — every reported self-audit Dice runs at the hard-coded `0.0`. `evaluate_threshold` also counts `ΔDice == 0` as harmful, which steers the selection.

**Exact behavioral change.**
1. `evaluate_threshold(..., neutral_margin=None)` uses `classify_delta`; neutral transitions excluded from both harmful and beneficial denominators. Emits `neutral_acceptance_rate` and the raw counts.
2. `select_threshold` keeps its lexicographic key `(final_macro_dice, −harmful_acceptance_rate, −|τ|)` but must now record `objective`, `neutral_margin`, `grid` and `n_rows` in its result.
3. **New calibration artifact schema v1** — `save_calibration(path, ...)` / `load_calibration(path)` writing:
   `schema_version`, `tau_accept`, `neutral_margin`, `source_split`, `checkpoint_path`, `checkpoint_sha256`, `t_max`, `threshold_grid` (`min`/`max`/`steps`), `selected_row`, `created_at` (UTC ISO-8601), `metric_space`, and an explicit `"validity": "diagnostic_only"` marker.
4. `load_calibration` validates `schema_version` and raises on unknown fields rather than silently ignoring them.
5. The artifact must carry, verbatim, the sentence that the τ was selected on the split it was measured on and is therefore an optimistically-biased diagnostic, not held-out evidence.

**Acceptance test.** Round-trip: `save_calibration` → `load_calibration` → the τ handed to inference is bitwise the saved value; `neutral_margin` survives the round trip; a `schema_version` bump is rejected.

**Scientific consequence.** Closes the gap where the pipeline reported an optimistically-selected number that no configuration ever used, while separately reporting self-audit Dice at an uncalibrated τ.

---

## 3. Wave 2 — three workers, start after Wave 1 lands and is reviewed

| Worker | Owns exclusively | Depends on |
|---|---|---|
| **AGY-6** | `scripts/audit_checkpoint.py` | AGY-1, AGY-2, AGY-4, AGY-5 |
| **AGY-7** | `training/train_annotation.py`, `training/finetune_joint.py`, `scripts/train_self_audit.py` | AGY-1, AGY-3, AGY-5 |
| **AGY-8** | `tests/test_remediation_metrics.py` (new file only) | all of Wave 1 |

### AGY-6 — checkpoint diagnostics CLI

Becomes the primary research instrument, emitting **one machine-readable JSON** with a stable, versioned schema.

Required: four modes and their Dice; `candidate_path_gain`, `self_audit_gain`, `oracle_headroom`, `audit_rescue_vs_always`, `headroom_capture_ratio`, `headroom_available_rate`; per-stage `prev_dice`, `candidate_dice`, `candidate_gain`, `positive_headroom`, `realized_gain`, `audit_gate_value`, `attempt_rate`, `accept_rate`, `reject_rate`, `beneficial/neutral/harmful_candidate_rate`, `beneficial_capture_rate`, `harmful_block_rate`, `attribution_residual`.

Plus the three new probes:

* **Probe 1 — synthetic positive quality.** On real checkpoint trajectories, report `positive_generated_count`, `positive_valid_count`, `positive_argmax_change_rate`, `positive_beneficial_rate`, `positive_neutral_rate`, `positive_harmful_rate`, `positive_mean_delta_dice`. Generator behaviour is **not** modified — this only measures whether the audit's synthetic-degeneracy finding (58–68% zero-pixel edits at `p_max ≥ 0.95`) reproduces on the trained checkpoint.
* **Probe 2 — audit-evidence conditioning value.** Same active state, expert run twice: once with the real `previous_audit_evidence`, once with zeros. Report `audit_evidence/candidate_logit_l1`, `argmax_change_rate`, `dice_delta_real_vs_zero`. **Both candidates are generated GT-free**; GT enters only afterwards to score them. If the branch is inert, say so — that is the finding.
* **Probe 3 — GT firewall.** Extend beyond batch 0. Report `gt_firewall/num_batches_checked`, `max_abs_logit_diff`, `decision_mismatch_rate`, `passed`.

Also: `--calibration <path>` / `--tau_accept <float>` with documented precedence (explicit `--tau_accept` > `--calibration` > config > `0.0`), the resolved τ and its source echoed into the JSON; `--empty_policy`; `--metric_space`; and every JSON report stamped `"evidence_class": "diagnostic_only"` with the reason (the checkpoint was trained on the existing 80/20 split; **no held-out test set exists**).

### AGY-7 — proxy-Dice alignment and τ wiring

* Phase A must call the **same** `slice_proxy_dice` as Phase C so the two headline numbers are comparable and Phase A stops being batch-size dependent (executed mismatch today: **0.6825 vs 0.8333** on identical inputs).
* `finetune_joint._foreground_dice_per_sample` routes through the shared proxy with the canonical empty policy.
* `scripts/train_self_audit.py` writes the new calibration artifact and, when present, uses it for the final diagnostic evaluation instead of the hard-coded `0.0`; the τ actually used is logged.
* Explicitly **out of scope**: the Phase-A objective, `stage_weights`, and anything that changes what is optimized. This worker changes *reporting*, not training loss.

### AGY-8 — tests

New file only, no edits to existing tests, covering all fifteen items in the brief.

---

## 4. Integration gates (Opus)

Every worker diff is reviewed against this checklist before merge:

- no duplicated metric implementation (one `classify_delta`, one empty-class policy, one proxy Dice)
- no accidental change to a training objective, `stage_weights`, or generator output
- no GT crossing a deployable boundary; probes 1–3 verified GT-free in the candidate-generation half
- denominators exclude neutral where required; no mean-of-batch survivors
- no empty-class inflation reachable by default
- no silent fallback (primary metric never quietly becomes the synthetic or combined one)
- exactly one τ source with documented precedence
- legacy checkpoints still load (`strict=True` path untouched)
- no `metric_space="volume_native"` without a true native GT
- the current validation split is never renamed "test"

Then `pytest -q` over the full suite, with any skipped/unrunnable subset documented explicitly.

---

## 5. Deliberately NOT fixed in this pass

Recorded so nothing is lost, each with the reason it is deferred:

| Item | Why deferred |
|---|---|
| `epsilon_neutral = 0.02` numeric value | changing it changes generated training data → generator redesign pass |
| Positive-counterfactual degeneracy (58–68% zero-pixel edits) | generator redesign; Probe 1 measures it first |
| Local audit head at stride 4 vs per-pixel target (FIX recall 0.0000) | architecture; guardrail forbids a full-resolution decoder |
| Local evidence not entering the gate | guardrail explicitly forbids wiring it this pass |
| A0 saturation (7.8% / 92.2% gradient split) | requires a Phase-A objective change and retraining |
| Held-out test split | must be created **before** a clean retraining run, not retrofitted onto this checkpoint |
| Halt-on-first-reject | behavioural change to `infer`; needs its own ablation |
| Dead FPN `refine[1..3]`, Phase-C auditor LR group, phase-tag checkpoint validation | P2 hygiene, not blocking measurement |

---

## 6. What happens after this pass

**Run the diagnostics. Do not retrain.**

```
python scripts/audit_checkpoint.py \
    --checkpoint weights/self_audit/phase_c_joint.pt \
    --config configs/self_audit_joint.yaml \
    --calibration reports/<run>/calibration.json \
    --output reports/<run>/diagnostics.json
```

Interpretation is fixed in advance so the result cannot be rationalized after the fact:

| Observation | Conclusion |
|---|---|
| `oracle ≈ initial` | no correction headroom |
| `oracle ≫ initial` and `self ≈ initial` | headroom exists, Auditor/gate fails to capture it |
| `always ≫ self` | over-conservative Auditor |
| in-domain flat, OOD positive | the contribution is domain-shift robustness |
