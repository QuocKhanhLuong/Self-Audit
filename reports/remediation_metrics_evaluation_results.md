# Remediation Pass 1 — Metrics & Evaluation Hardening: Results

**Repository:** `QuocKhanhLuong/Self-Audit`
**Integrator:** Claude Opus 5 (planner, conflict resolver, final code + scientific reviewer)
**Plan:** `reports/remediation_plan_metrics_and_evaluation.md`
**Audit this remediates:** `reports/opus5_model_logic_audit.md`, `reports/opus5_logic_summary.md`

---

## 1. Heads

| | SHA |
|---|---|
| **START_HEAD** (verified before any edit) | `a901eeae56cf6835bb9a1d9c79a9d5d3e99374c6` |
| **FINAL_HEAD** | see the commit created by this pass (`fix: harden self-audit metrics and evaluation protocol`) |

Working tree was clean at start apart from untracked audit reports. `self_audit_architecture.html` was already untracked before this session and was deliberately **not** committed — it is unrelated to this work.

---

## 2. Changed files

29 files, +10,694 / −444.

**New**
- `src/self_audit/audit/semantics.py` — the canonical decision-margin and empty-class contract (Wave 0, written by Opus so no worker depended on another worker's diff)
- `tests/test_remediation_metrics.py` — mandate items 1–7
- `tests/test_remediation_evaluation.py` — mandate items 8–15
- `reports/remediation_plan_metrics_and_evaluation.md`, this file, and the worker reports under `reports/workers/`

**Modified**
- `src/self_audit/evaluation/`: `metrics.py`, `audit_decomposition.py`, `threshold.py`, `volume_inference.py`, `__init__.py`
- `src/self_audit/training/`: `train_auditor.py`, `train_annotation.py`, `finetune_joint.py`
- `src/self_audit/audit/__init__.py`
- `scripts/`: `audit_checkpoint.py`, `calibrate_threshold.py`, `calibrate`/`train_self_audit.py`
- `tests/test_self_audit_hardening.py` — one test updated by Opus because it pinned the pre-remediation empty-class convention (see §5)

---

## 3. How this was run, and what went wrong with it

Two waves of Orca workers on strictly disjoint file ownership (the brief's seven concern-oriented workers would have collided on `metrics.py`, `audit_decomposition.py` and `audit_checkpoint.py`, so ownership was remapped by file and the concerns preserved).

| Worker | Files | Outcome |
|---|---|---|
| AGY-1 | `evaluation/metrics.py` | completed, reported |
| AGY-2 | `evaluation/audit_decomposition.py` | code complete, **never sent `worker_done`, no report** |
| AGY-3 | `training/train_auditor.py` | completed, reported |
| AGY-4 | `evaluation/volume_inference.py` | completed, reported |
| AGY-5 | `evaluation/threshold.py`, `scripts/calibrate_threshold.py` | completed, reported |
| AGY-6 | `scripts/audit_checkpoint.py` | code complete, **never sent `worker_done`, no report** |
| AGY-7 | `training/train_annotation.py`, `training/finetune_joint.py`, `scripts/train_self_audit.py` | code complete, **never sent `worker_done`, no report** |
| AGY-8 | `tests/` | **~half complete** — wrote items 1–7, never created the second file for 8–15, no report |

**Four of eight workers ended their turn silently.** Their code landed and is sound, but the verification they were asked to paste never arrived. Rather than trust unreported diffs, Opus reviewed each one and **re-ran every acceptance criterion independently**; those numbers are in §6 and are the evidence of record for AGY-2/6/7. The gap AGY-8 left — mandate items 8–15 — was written by Opus as `tests/test_remediation_evaluation.py`.

Orca task rows for the four silent workers still read `dispatched`; `task-update` did not take while their dispatch was live. That is bookkeeping only and does not affect the code.

**One coordination error of mine, recorded for honesty.** Three workers independently reported that AGY-2 had broken pytest collection with an absolute `from self_audit...` import, and I relayed that to AGY-2 as a required fix before checking it. It was wrong: that import style is pre-existing at `HEAD:17`, and the real cause was running `pytest` without `PYTHONPATH=src`. I had captured the baseline (73 passed at START_HEAD *with* `PYTHONPATH=src`), so I could and should have checked before relaying. I sent a correction.

---

## 4. Findings fixed

### P0-1 — reported Dice was not the ACDC metric
Empty-vs-empty foreground classes scored `1.0` in every Dice in the repo. Now governed by an explicit policy: `DEFAULT_EMPTY_CLASS_POLICY = "exclude"` (nan, dropped from a nan-aware macro). **Only a class empty in *both* prediction and target is excluded**; a class present in exactly one side scores `0.0` and is never excluded, so a total miss can never be hidden. `empty_policy="legacy_one"` reproduces the old behaviour for explicit comparison.
**This lowers every reported Dice.** That is the point.

### P0-2 — Phase-A and Phase-C headline Dice were computed differently
Phase A pooled over the whole batch (batch-size dependent); Phase C averaged per slice. Both now call one helper, `slice_proxy_dice`. Verified agreement `1.46e-08`, and batch-8 vs batch-4+4 differ by `0.00e+00`.

### P0-3 — `headroom_capture_ratio` was a mean of per-sample ratios
Replaced by a ratio of sums over *material* headroom (`oracle_headroom > neutral_margin`), `nan` when nothing is eligible — never `0.0`, which would falsely claim the audit captured nothing when nothing was available. On the pathological pair, mean-of-ratios gives **312,500.3**; the new estimator gives **0.666667**, matching the hand-computed `0.20/0.30`.

### P0-4 — calibrated τ was never consumed; no held-out evaluation
`save_calibration` / `load_calibration` (schema v1) persist τ with its `neutral_margin`, source split, checkpoint SHA-256, grid, timestamp, and a mandatory `validity: "diagnostic_only"` plus a plain-words `validity_reason`. `audit_checkpoint.py` gained `--calibration` with documented precedence (`--tau_accept` > `--calibration` > config > `0.0`), asserts the τ actually used equals the saved one, and **hard-errors** if the artifact's margin disagrees with the evaluation margin. **No test split was created** — that must precede a clean retraining run, not be retrofitted onto this checkpoint.

### P1-1 — Phase-B validation was 57.1% synthetic and selected the checkpoint
Validation now reports `audit/on_policy/*`, `audit/synthetic/*`, `audit/combined/*`. **`primary_metric` is on-policy AUROC**, with a recorded `primary_metric_source` and the documented fallback chain `on_policy_auroc → on_policy_improve_regress_accuracy → nan/undefined`. It never borrows the combined or synthetic value. Legacy flat keys still emit *combined* for W&B continuity but no longer drive selection.
**Training is unchanged** — proven by replaying `build_auditor_transitions` against a START_HEAD copy under the same seed: 7 groups, 3 on-policy + 4 synthetic, identical tensors.

### P1-4 — three epsilons, eight zero-boundary sites
One canonical margin (`DEFAULT_NEUTRAL_MARGIN = 0.005`) and one `classify_delta`, boundary closed on the neutral side. The generator's `epsilon_neutral = 0.02` is **kept** and documented as a *generation-time search tolerance* (Decision A in the plan) — changing its value would change generated training data, which belongs to the deferred generator pass. `check_generation_tolerance` now raises a `RuntimeWarning` at Phase-B start so the 0.02-vs-0.005 conflict is visible rather than silent.

### P1-6 — no volume-level, native-geometry reporting
`evaluate_comparison_modes` now breaks out RV/MYO/LV, applies the empty-class policy, and stamps `metric_space="volume_resized"`. New `to_native_geometry` (nearest-neighbour, refuses to guess axis order) and `evaluate_volume_native`, which **raises** if genuine native ground truth is absent. `split_cases_by_phase` separates ED/ES and never guesses.

---

## 5. Two integrator decisions that overrode the brief

**(a) Headline `harmful_acceptance_rate` uses the FULL accepted denominator.** AGY-1 caught that my own spec was self-contradictory. Restricting the denominator to materially signed transitions gives `1.0` on the audit's worked example — *more* conservative than the original `0.8` bug, and since `select_threshold` tie-breaks on this rate it would have pushed τ further conservative, the opposite of the fix. Neutral is excluded from the **numerator** only; the conditional forms are emitted as `*_signed` for diagnosis. Verified: the audit's example now yields **0.2** (was 0.8 at HEAD), and the all-neutral case yields **0.0**.

**(b) `metric_space` is reserved for the grid axis.** `annotation_metrics` was already using that key for `physical`/`pixel` surface-distance units, and `evaluate_comparison_modes` nests that dict inside a block stamping its own `metric_space` — two meanings in one result. The distance-unit key is now `distance_space`. These are genuinely independent axes.

**(c) One existing test was updated, not deleted.** `test_metrics_use_consistent_both_empty_dice_convention` pinned `empty_score=1.0`. It now asserts the new default *and* that the legacy opt-in still works, preserving its original intent; a companion test asserts a total miss still scores `0.0` under both policies.

---

## 6. Verification (all executed by Opus, on synthetic tensors)

```
Test suite            START_HEAD baseline : 73 passed, 0 failed   (PYTHONPATH=src)
                      after remediation   : 103 passed, 0 failed
```

**Neutral semantics**
```
acceptance_metrics(accepted=[T,T,T,T,T,F], delta=[0,0,0,+1e-9,-0.20,+0.20])
   harmful_acceptance_rate = 0.2      (HEAD gave 0.8)
   harmful_acceptance_rate_signed = 1.0   <- why this must not be the headline
brief example accepted=[T,T,T] delta=[0.0,+0.001,-0.001] eps=0.005
   beneficial=0  harmful=0  neutral=3  harmful_acceptance_rate=0.0
evaluate_threshold, 3 accepted transitions all delta==0.0
   harmful_acceptance_rate = 0.0      (HEAD gave 1.0)
```

**Headroom attribution (AGY-2, unreported — verified by Opus)**
```
mean-of-ratios (1e-8 guard) = 312,500.3   ->   ratio-of-sums = 0.666667   (hand-computed 0.20/0.30)
zero-material-headroom cohort: capture = nan, available_rate = 0.0, eligible = 0
batch-partition invariance at sizes 1 / 2 / 3: max |diff| vs whole loader = 0.00e+00
attribution residual = 0.00e+00 ; stage key 'oracle_gain' removed, 'positive_headroom' present
```

**Proxy alignment and nan safety (AGY-7, unreported — verified by Opus)**
```
Phase-A proxy vs Phase-C proxy      max|diff| = 1.46e-08
batch 8 = 0.2530965064  vs  4+4 = 0.2530965064   diff = 0.00e+00
all-empty class through real Phase-A validation: class1=1.0000 class2=1.0000 class3=nan macro=1.0000
   (a legitimately-nan class no longer poisons the epoch)
```

**Phase-B training unchanged**
```
HEAD groups=7  CURRENT groups=7
provenance both: ['on_policy','on_policy','on_policy','synthetic','synthetic','synthetic','synthetic']
IDENTICAL transition set: True
```

**GT firewall — with a powered control**
```
deployable self_audit under two different GTs:
   a0_logits / logits / candidates / delta_q / accept masks / halt_turn   max|diff| = 0.0e+00
POSITIVE CONTROL (oracle_accept): accept-mask diff = 1.0, logits max|diff| = 0.5317
   accepted with GT=candidate: [True, True, True]
   accepted with GT=previous : [False, False, False]
```
My **first** attempt at this control was unpowered: with two *random* GTs on an untrained net the oracle rejects everything under either, so `oracle_accept` output is identical and the test would have passed even if GT were leaking. The controls are now derived from the model's own trajectory so the oracle decision provably flips. The same construction is used in `tests/test_remediation_evaluation.py` and was broadcast to the workers.

**Diagnostic CLI, end to end on a synthetic checkpoint**
```
schema=1  evidence_class=diagnostic_only  tau=0.0 source=config_audit_tau_accept
completeness ok=True  missing=[]
modes/initial_dice 0.1547  always 0.1743  self_audit 0.1547  oracle 0.1651
modes/oracle_headroom 0.0103  headroom_capture_ratio 0.0  headroom_available_rate 0.333
probe1  positive_generated_count 16  argmax_change_rate 1.0  neutral_rate 0.9375  mean_delta_dice 0.0389
probe2  candidate_logit_l1 nan  argmax_change_rate nan  dice_delta_real_vs_zero nan
probe3  batches_checked 3.0  max_abs_logit_diff 0.0  decision_mismatch_rate 0.0  passed True
JSON serialized OK (13,361 bytes)
```
These numbers are from an **untrained** network and carry no scientific meaning; they prove the instrument runs, serializes, and propagates `nan` without coercing it to `0.0`.

---

## 7. Semantics after remediation

**Decision margin.** `delta > +eps` BENEFICIAL, `delta < -eps` HARMFUL, `|delta| <= eps` NEUTRAL, `eps = 0.005`, boundary closed on the neutral side. Neutral is excluded from every numerator; headline denominators are the full accepted/rejected populations.

**Empty class.** Excluded (nan, nan-aware macro) only when empty in both prediction and target. Present-on-one-side always scores `0.0`.

**Phase-B validation.** Three namespaces; `primary_metric` = on-policy AUROC; synthetic is auxiliary. Training composition untouched.

**Calibration.** Schema v1 artifact, `validity: "diagnostic_only"`, τ round-trips bitwise, margin mismatch is a hard error.

**Metric spaces.** `slice_proxy` (training proxy) · `volume_resized` (3-D on the network grid) · `volume_native` (3-D after inverse mapping, **requires real native GT**). Never conflated. Surface-distance units are the separate `distance_space` axis.

**GT firewall.** `gt_firewall/{num_batches_checked,max_abs_logit_diff,decision_mismatch_rate,passed}`; `passed` requires exact `0.0` on both, no tolerance.

---

## 8. Deliberately NOT fixed

| Item | Why |
|---|---|
| `epsilon_neutral = 0.02` numeric value | changing it changes generated training data → generator pass |
| Positive-counterfactual degeneracy (58–68% zero-pixel edits) | generator redesign; probe 1 measures it first |
| Local audit head stride 4 vs per-pixel target (FIX recall 0.0000) | architecture; guardrail forbids a decoder |
| Local evidence not entering the gate | guardrail forbids wiring it this pass |
| A0 saturation (7.8% / 92.2% gradient split) | needs a Phase-A objective change + retraining |
| Held-out test split | must precede a clean retraining run |
| Halt-on-first-reject | behavioural change to `infer`; needs its own ablation |
| Dead FPN `refine[1..3]`, Phase-C auditor LR group, phase-tag checkpoint validation | P2 hygiene, not blocking measurement |

---

## 9. Remaining research risks

1. **Nothing here makes the science work** — it makes it *measurable*. Every P1 that threatens the paper's claim is still open by design.
2. **Probe 2 needs an accepted transition.** With a gate that rejects at turn 0 there is no turn ≥ 1 carrying real audit evidence, so the audit-evidence probe returns `nan`. On the real checkpoint, `nan` there is itself a finding: it means the gate never accepted anything.
3. **`volume_native` remains unreachable** from `preprocessed_data/`: the stored mask was destructively resized to 256×256 with `order=0`. AGY-4 found a 1-voxel-thick native structure is annihilated by that downsample. A truthful native number needs the raw ACDC NIfTI.
4. **Reported Dice will drop** once the empty-class fix takes effect. Expect it, and do not read the drop as a regression.
5. **The current checkpoint is diagnostic-only** and every artifact now says so in-band. It was trained on the existing 80/20 split; τ is selected on the split it is reported on.

---

## 10. Run the diagnostics next — do not retrain

```bash
PYTHONPATH=src python3 scripts/audit_checkpoint.py \
    --checkpoint weights/self_audit/phase_c_joint.pt \
    --config configs/self_audit_joint.yaml \
    --output reports/diagnostics/acdc_val.json
```

With a calibration artifact, and with the volume pass:

```bash
PYTHONPATH=src python3 scripts/calibrate_threshold.py \
    --config configs/self_audit_joint.yaml \
    --checkpoint weights/self_audit/phase_c_joint.pt \
    --neutral_margin 0.005 \
    --output reports/diagnostics/calibration.json

PYTHONPATH=src python3 scripts/audit_checkpoint.py \
    --checkpoint weights/self_audit/phase_c_joint.pt \
    --config configs/self_audit_joint.yaml \
    --calibration reports/diagnostics/calibration.json \
    --output reports/diagnostics/acdc_val_calibrated.json
```

Add `--skip_volume` for the fast slice-only pass; `--firewall_batches N` widens the GT-firewall probe.

Interpretation, fixed in advance so the result cannot be rationalized afterwards:

| Observation | Conclusion |
|---|---|
| `oracle ≈ initial` | no correction headroom exists |
| `oracle ≫ initial` and `self ≈ initial` | headroom exists; the Auditor/gate fails to capture it |
| `always ≫ self` | over-conservative Auditor |
| in-domain flat, OOD positive | the contribution is domain-shift robustness |

The disambiguating control for the first row — a `stage_weights: [1.0]`, `max_turns: 0` single-stage A0 baseline — does not exist yet and requires training, so it belongs to the next pass.
