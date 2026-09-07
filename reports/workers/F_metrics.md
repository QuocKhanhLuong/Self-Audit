# WORKER F — Metrics, Diagnostics and Scientific-Protocol Audit

Repo: `/Users/alvinluong/Self-Audit` @ `a901eea` (branch `main`). Read-only audit.

**Execution environment.** `torch 2.13.0` / `numpy 2.4.6` are importable and every claim marked
"executed" below was produced by running the repository's own functions. `pytest` is **not**
installed in the active interpreter (`/Users/alvinluong/miniforge3/bin/python`), so
`tests/test_audit_decomposition.py` and `tests/test_self_audit_volume.py` could not be run as a
suite; their assertions were reproduced by direct calls instead. `matplotlib` is also missing from
that interpreter, and `src/self_audit/evaluation/__init__.py:12` imports `.visualizer`
unconditionally, so **`import self_audit.evaluation.*` hard-fails without matplotlib** — including
`scripts/audit_checkpoint.py:22-24`. All experiments below therefore stub `matplotlib` first. No
preprocessed data is present (`preprocessed_data/` does not exist), so §2 is quantified
analytically and by synthetic reproduction, not on real ACDC volumes. This is stated as a limit,
not hidden.

---

## Severity summary

| # | Finding | Severity |
|---|---|---|
| F1 | Phase-A headline Dice is **batch-pooled**, Phase-C headline Dice is **per-slice**; the two headline numbers are not comparable and Phase A depends on `batch_size` | **P0** |
| F2 | `empty-empty -> 1.0` convention is used by *every* Dice in the repo and `foreground_only` is never enabled, so fully-empty ACDC slices score exactly 1.0 and inflate every reported Dice | **P0** |
| F3 | `modes/headroom_capture_ratio` is a **mean-of-ratios** with a `1e-8` guard; a single near-zero-headroom row can drive it to ~1e6 | **P0** |
| F4 | Zero is used as the improve/regress boundary in 8 sites while training uses `neutral_margin=0.005` (and the counterfactual generator uses `epsilon_neutral=0.02`) — three mutually inconsistent epsilons | **P1** |
| F5 | No volume-level (3D) Dice is ever reported; `evaluate_comparison_modes` (the only 3D path) has **zero callers** | **P1** |
| F6 | `evaluate_audit_decomposition` aggregates as an **unweighted mean over batches** of per-batch rates and silently drops NaN batches | **P1** |
| F7 | W&B has no `audit/on_policy/*` vs `audit/synthetic/*` split; namespacing is inconsistent across phases and non-finite metrics are silently dropped | **P1** |
| F8 | `stage_t/oracle_gain` is an alias of `stage_t/headroom` measured on the *self-audit* trajectory — it is not the oracle path | **P2** |
| F9 | `phase_a/headroom_collapse_ratio` divides by a hardcoded `0.005`, ignoring the `neutral_margin` argument | **P2** |
| F10 | `gt_firewall/*` probe runs on **batch 0 only** | **P2** |
| F11 | `self_audit_net._dice` hardcodes classes `(1,2,3)`; latent breakage for any non-4-class model | **P2** |

**What the audit could NOT falsify:** the stage-wise attribution algebra in
`decompose_self_audit_output` is **correct**. Sign conventions verified by execution (§4). The
attribution identity `candidate_gain + audit_gate_value == realized_gain` holds to `<1e-7`. The four
comparison modes and all four headroom metrics named in the brief **do exist and are logged** (§5).

---

## 1. Every Dice implementation in the repo

Six distinct implementations. Table is the result of reading each one, not of naming.

| # | Symbol | Location | Hard/Soft | Background | Empty-empty | Epsilon | Aggregation unit |
|---|---|---|---|---|---|---|---|
| A | `per_class_dice` / `dice_score` | `src/self_audit/evaluation/metrics.py:40-59`, `62-77` | hard (argmax at `:29-33`) | **excluded** (`include_background=False`) | **1.0** (`empty_score`, `:58`) | none | **whatever array shape is passed** — pooled over the entire input |
| B | `multiclass_dice` | `src/self_audit/audit/targets.py:73-87` | hard | excluded (`range(1,num_classes)`, `:78`) | **1.0** (`:81-84`) | `clamp_min(1)` on denom only | **per row** (`flatten(1)`), returns `[B]` |
| C | `SelfAuditNet._dice` | `src/self_audit/models/self_audit_net.py:348-357` | hard | excluded, **classes hardcoded `(1,2,3)`** (`:350`) | **1.0** (`:355`) | `clamp_min(1)` | per row, returns `[B]` |
| D | `_compute_dice_fast` | `src/self_audit/evaluation/visualizer.py:175-187` | hard | excluded | **1.0** (`:183`) | none | whole 2-D array |
| E | `soft_dice_loss` | `src/self_audit/losses/annotation.py:10-25` | **soft** (softmax, `:16`) | **INCLUDED** (`include_background=True`, `:10`) | 1.0 via `eps/eps` | `eps=1e-6` in num **and** denom | mean over `[B,C]` jointly |
| F | `_foreground_dice_per_sample` | `src/self_audit/training/finetune_joint.py:286-287` | thin wrapper over B with `num_classes=logits.shape[1]` | — | — | — | per row |

### Inconsistencies (each is a finding)

1. **Training optimizes a different Dice than reporting measures.** `phase_a_loss`
   (`train_annotation.py:98`) calls `annotation_loss(value, target)` with no `include_background`
   argument, so it defaults to `True` (`losses/annotation.py:33`, `:10`). The **background class is
   included in the training soft-Dice**, while every evaluation Dice (A/B/C/D) excludes it. On
   cardiac MRI, background is ~95% of pixels and its Dice is ~0.99, so the loss signal on the three
   foreground classes is diluted by 4x relative to what is reported.

2. **B vs C differ only in class parameterization.** `multiclass_dice` takes `num_classes`;
   `_dice` hardcodes `(1,2,3)`. Identical for `num_classes=4` (all shipped configs — verified:
   `configs/*.yaml` all set `num_classes: 4`), but `_dice` silently returns
   `(real_dice + 1.0 + 1.0)/3` for any 2-class model. `scripts/prepare_mnm_binary.py:106` exists and
   produces binary masks, so this is a live trap, not a hypothetical. **F11 (P2).**

3. **`decompose_self_audit_output` calls `multiclass_dice(previous_active, target_active)` with no
   `num_classes`** (`audit_decomposition.py:178-179`), taking the default `4`, whereas
   `_foreground_dice` (`:253-254`) passes `num_classes=logits.shape[1]`. Two different class counts
   inside the same module.

4. **A is the only shape-polymorphic one.** Passed a `[Z,H,W]` volume it computes a genuine 3-D
   volume Dice; passed a `[B,H,W]` batch it computes a *batch-pooled* Dice. This ambiguity is the
   direct cause of F1.

---

## 2. F2 (P0) — the `empty -> 1.0` convention and how much it inflates reported Dice

**Every** Dice in the repo returns `1.0` when `|pred_c| + |gt_c| == 0`:
`metrics.py:58`, `targets.py:81-84`, `self_audit_net.py:355`, `visualizer.py:183`, and `eps/eps`
in `annotation.py:22-23`. `tests/test_self_audit_hardening.py:89-93` *asserts* this behavior as
intended (`{1:1.0, 2:1.0, 3:1.0}` for an all-zero prediction vs an all-zero target).

**The headline metric uses this convention.** Both paths do:
- Phase A: `train_annotation.py:242` -> `per_class_dice` (impl A, `empty_score=1.0`).
- Phase C: `finetune_joint.py:286` -> `multiclass_dice` (impl B, `1.0` at `targets.py:81-84`).
- Decomposition/modes: `audit_decomposition.py:253-254` -> impl B.

There is no `empty_score=0.0` or "exclude absent classes" path anywhere in the reporting chain.

**No filtering removes empty slices.** `VolumeSliceDataset.foreground_only` defaults to `False`
(`data/common.py:304`, `:317`; `acdc.py:215`; `mnms.py:161`) and `build_patient_dataset`
(`training/_utils.py:270-288`) **never passes it** — the kwarg whitelist at `_utils.py:285` is
`("split_manifest","seed","depth_axis","expected_slices","max_cache")`. So every slice of every
ACDC volume, including the fully-background slices above the base and below the apex, is in both
train and val.

**Executed evidence** (`per_class_dice`, `dice_score`, `multiclass_dice` called directly):

```
empty-vs-empty per_class_dice: {1: 1.0, 2: 1.0, 3: 1.0}
empty-vs-empty dice_score   : 1.0

10-slice synthetic volume, 2 empty apical slices, 8 imperfect slices:
  slice-level mean over ALL 10 slices  = 0.9758   <- what Phase C reports
  slice-level mean over non-empty only = 0.9697
  volume-level (3D) Dice               = 0.9697   <- what a paper should report
  inflation from 2 empty slices        = +0.0061

single slice where RV(1) is absent in GT and correctly predicted absent:
  per-class {1: 1.0, 2: 0.667, 3: 1.0} -> slice Dice 0.8889
  same slice excluding the absent RV   -> 0.8333   (+0.0556 inflation)
```

**Risk quantification.** Let `f` be the fraction of (slice, class) pairs where the class is absent
in GT *and* correctly predicted absent, and `D` the true Dice over present classes. Reported
macro-Dice `= (1-f)D + f`. Executed sensitivity table:

| f | D=0.80 | D=0.85 | D=0.90 |
|---|---|---|---|
| 0.05 | 0.810 (+0.010) | 0.858 (+0.008) | 0.905 (+0.005) |
| 0.10 | 0.820 (+0.020) | 0.865 (+0.015) | 0.910 (+0.010) |
| 0.20 | 0.840 (+0.040) | 0.880 (+0.030) | 0.920 (+0.020) |
| 0.30 | 0.860 (+0.060) | 0.895 (+0.045) | 0.930 (+0.030) |

ACDC's RV in particular vanishes on basal/apical slices, and MYO/LV vanish on the outermost slices;
`f` in the 0.10-0.25 range is plausible for a per-slice ACDC protocol. **I did not measure `f` —
no data is checked in.** The point stands regardless of `f`'s exact value: the convention moves the
headline number in one direction (up), by an amount nobody in this repo measures.

**Second-order harm, specific to this project's hypothesis.** On a fully-empty slice, `_dice`
returns `1.0` for both previous and candidate, so:
- `delta = 0` exactly -> classified NEUTRAL (`audit_decomposition.py:187`), fine;
- but `_candidate_improves` (`self_audit_net.py:340-346`) uses strict `>`, so **`oracle_accept`
  halts at t=0 on every empty slice** and `oracle_headroom = 0` there;
- and `acceptance_metrics` counts an *accepted* zero-delta transition as **harmful**
  (`metrics.py:316`, `delta <= 0.0`). Empty slices therefore push `harmful_acceptance_rate` up
  while pushing every Dice up. See F4.

---

## 3. F1 (P0) + F5 (P1) — slice-level vs volume-level; ED/ES

### What number would go in the paper

Two different, mutually incomparable numbers are produced:

- **`val/macro_foreground_dice`** (Phase A, `train_annotation.py:253`, logged at `:392`).
  `train_annotation.py:242` calls `per_class_dice(final_logits.argmax(dim=1), batch["mask"], ...)`
  on the **entire `[B,H,W]` batch at once**. Impl A pools the whole array into one
  numerator/denominator per class, so this is a **batch-pooled** Dice, then weighted by
  `batch_size` (`:247`) and divided by the slice count (`:252-253`). Net: a mean of per-batch
  pooled Dice over arbitrary random batches of slices from different patients.

- **`final_foreground_macro_dice`** (Phase C, `finetune_joint.py:465`, logged as `val/final_dice`
  at `:858`). `_foreground_dice_per_sample` -> `multiclass_dice` -> **per-slice**, then
  `torch.cat(...).mean()` (`:424-425`). A genuine per-slice mean.

**Executed proof that these diverge** (same 2 predictions, same GT):

```
slice0: tiny structure, perfect.  slice1: large structure, zero overlap.
batch-POOLED per_class_dice (train_annotation val): {1: 0.0476, 2: 1.0, 3: 1.0} -> macro 0.6825
PER-SLICE multiclass_dice   (validate_phase_c)    : [1.0, 0.6667]              -> mean 0.8333
```

0.6825 vs 0.8333 for identical inputs. The Phase-A number is also **`batch_size`-dependent**:
`configs/self_audit_joint.yaml:12` sets `batch_size: 2`, so changing it changes the reported
Phase-A Dice with no change to the model. Reporting "A0 Dice" from Phase A and "final Dice" from
Phase C in the same table is invalid.

### Volume-level reassembly

`volume_inference.py` **does** implement correct 3-D reassembly:
`reconstruct_volume` (`:83-119`) with a strict slice-index permutation check (`:112-113`),
`build_25d_batch` (`:65-81`) with replicated boundary neighbors, and
`evaluate_comparison_modes` (`:236-330`) which runs all four modes on a patient volume and calls
`annotation_metrics` on the reassembled `[Z,H,W]` prediction (`:326-329`) — a true volume Dice,
plus HD95/ASSD.

**But `evaluate_comparison_modes` has no callers.** Grep over `src/`, `scripts/`, `tests/` finds
only its definition and its re-export in `evaluation/__init__.py:7,29`. `infer_patient_volume` is
called only by `scripts/visualize_predictions.py:96` (figures, not metrics). **No training script,
no evaluation script, and no test ever computes a volume-level Dice.** Every number that reaches
W&B or `reports/*.json` is slice-level. **F5 (P1).**

`scripts/audit_checkpoint.py` — the only dedicated diagnostic CLI — is slice-level throughout
(`:114-131` -> `evaluate_annotation_headroom` / `evaluate_audit_decomposition`, both DataLoader-based).

### ED/ES

**Pooled, never separated.** `scripts/preprocess_acdc.py:48` iterates
`[(ed_frame,'ED'), (es_frame,'ES')]` and emits each as an independent case
`f"{patient_folder}_{frame_name}"` (`:88`). `patient_id_from_case_id`
(`data/common.py:47-54`) strips the frame/phase suffix so that the *patient split* is leak-free —
correct — but the consequence is that ED and ES become two anonymous volume records. No metric
anywhere carries a phase key, and there is no `phase`/`frame` field in `VolumeRecord`. **Per-phase
ED/ES Dice (standard for ACDC reporting) cannot be produced from the current code.** ES is the
harder phase; pooling hides that.

---

## 4. Stage-wise attribution — VERIFIED CORRECT

Definitions in `decompose_self_audit_output` (`audit_decomposition.py:81-247`):

| Brief's definition | Code | Line | Match |
|---|---|---|---|
| `Delta_t = Dice(cand)-Dice(prev)` | `delta = candidate_dice - previous_dice` | `:180` | yes |
| `CandidateGain_t = mean Delta_t` | `stage_t/candidate_gain = _mean_or_nan(delta)` | `:196` | yes |
| `PositiveHeadroom_t = mean max(Delta_t,0)` | `headroom = torch.clamp(delta, min=0.0)`; `stage_t/headroom` | `:191`, `:197` | yes |
| `RealizedGain_t = accepted_t * Delta_t` | `realized = delta * accepted_float` | `:189`, `:199` | yes |
| `AuditGateValue_t = RealizedGain_t - CandidateGain_t` | `gate_value = realized - delta` | `:190`, `:200` | yes |

`previous_dice`/`candidate_dice` are computed **only on active rows** (`_select_rows`, `:175-177`),
so halted rows are excluded — correct.

### Hand-computed 2-row example, run through the actual function

Setup: GT = class 1 on the left half of a 4x4 grid, 2 rows.
Row 0: previous predicts all background (Dice 2/3 because classes 2,3 are empty-empty -> 1.0),
candidate is perfect (1.0) -> `delta = +1/3`.
Row 1: mirror image -> `delta = -1/3`.
Executed against `decompose_self_audit_output(..., neutral_margin=0.005)`:

```
row0: prev=0.666667 cand=1.000000 delta=+0.333333
row1: prev=1.000000 cand=0.666667 delta=-0.333333

[ideal: accept-good, reject-bad]        [worst: reject-good, accept-bad]
  candidate_gain       = +0.000000        candidate_gain       = +0.000000
  realized_gain        = +0.166667        realized_gain        = -0.166667
  audit_gate_value     = +0.166667        audit_gate_value     = -0.166667
  headroom             = +0.166667        headroom             = +0.166667
  beneficial_capture   = +1.000000        beneficial_capture   = +0.000000
  harmful_block_rate   = +1.000000        harmful_block_rate   = +0.000000
  attribution_residual = +0.000000        attribution_residual = +0.000000
```

**Sign conventions are correct**: rejecting a harmful candidate yields **positive**
`audit_gate_value`; rejecting a beneficial candidate yields **negative**. The attribution identity
`candidate_gain + audit_gate_value == realized_gain` holds exactly. `_safe_rate` (`:69-73`)
correctly returns NaN rather than 0 when a conditioning population is empty, and the neutral band
is respected (`tests/test_audit_decomposition.py:63-80` reproduces this; verified by direct call).

### F8 (P2) — `stage_t/oracle_gain` is mislabeled

`audit_decomposition.py:197-198` assigns **the same value** to `stage_t/headroom` and
`stage_t/oracle_gain`. This is `mean(clamp(delta,0))` measured on the **self-audit trajectory**
(`self_out`, passed at `:322-327`), not on the `oracle_accept` trajectory. Once the self-audit gate
rejects at stage 0, stages 1..T-1 have no active rows on that row, so `stage_t/oracle_gain` is
structurally incapable of representing the oracle path's stage-t gain. Two keys for one quantity,
one of them named after something it is not.

### Degenerate aggregate

The same run shows `candidate_gain = 0.000000` for **all four** gate configurations, because +1/3
and -1/3 cancel in the mean. `audit_gate_value` is likewise 0 for both "accept both" and "reject
both". This is arithmetically correct but means the headline `audit/gate_value` can read 0.000 both
when the auditor is doing nothing and when it is doing equal amounts of good and harm. The
per-population `harmful_block_rate` / `beneficial_capture_rate` do separate these (1.0/0.0 vs
0.0/1.0 in the run above) — so the diagnostic set is adequate, but `audit/gate_value` alone is not
a sufficient headline.

---

## 5. Four modes and the headroom metrics — PRESENT

`evaluate_audit_modes_batch` (`audit_decomposition.py:257-330`) runs all four:

| Mode | Line |
|---|---|
| `initial_only` (`t_max=0`) | `:267` |
| `always_accept_refinement` (`tau_accept=-inf`) | `:268-273` |
| `self_audit` | `:274-279` |
| `oracle_accept` (with `oracle_target=ground_truth`) | `:280-285` |

Required quantities, all present and all logged:

| Brief | Code key | Line | Definition in code |
|---|---|---|---|
| `OracleHeadroom = Dice(oracle)-Dice(A0)` | `modes/oracle_headroom` | `:302`, `:318` | `oracle_dice - initial_dice` — matches |
| `SelfAuditGain = Dice(self)-Dice(A0)` | `modes/self_audit_gain` | `:303`, `:317` | `self_dice - initial_dice` — matches |
| `AuditRescue = Dice(self)-Dice(always)` | `modes/audit_rescue_vs_always` | `:305`, `:316` | `self_dice - always_dice` — matches |
| `HeadroomCapture = SelfAuditGain/OracleHeadroom` | `modes/headroom_capture_ratio` | `:306-308`, `:319` | **mean-of-ratios, see F3** |

Logged from `scripts/train_self_audit.py:531` (`_log(logger,"C",epoch+1,decomposition)`) and
written to `reports/audit_decomposition.json` by `scripts/audit_checkpoint.py:132-142`. Printed at
`train_self_audit.py:559-566` and `audit_checkpoint.py:151-161`. **This is not a measurability gap.**

### F3 (P0) — the division guard is 7 orders of magnitude too small, and the estimator is wrong

```python
# audit_decomposition.py:305-308, 319
positive_headroom = oracle_headroom > 1e-8
capture = torch.full_like(oracle_headroom, float("nan"))
capture[positive_headroom] = self_gain[positive_headroom] / oracle_headroom[positive_headroom]
...
"modes/headroom_capture_ratio": _mean_or_nan(capture[torch.isfinite(capture)]),
```

Two independent defects:

1. **The guard is `1e-8` while the project's own neutral margin is `5e-3`** — 500,000x larger.
   A row whose oracle headroom is `2e-8` (indistinguishable from noise, and *below* the
   float32 resolution of a Dice difference near 1.0) passes the filter and contributes a ratio of
   millions. `torch.isfinite` does not catch it: the value is finite, just enormous.
2. **Mean-of-ratios, not ratio-of-means.** `E[X/Y] != E[X]/E[Y]`. The brief defines
   `HeadroomCapture = SelfAuditGain/OracleHeadroom` — an aggregate ratio.

Executed, using the repo's exact expression:

```
oracle_headroom = [2e-8, 0.10, 0.10, 0.10]   self_gain = [0.05, 0.05, 0.05, 0.05]
per-row ratios          : [2500000.0, 0.5, 0.5, 0.5]
reported mean-of-ratios : 625000.375
correct ratio-of-means  : 0.667
```

**625000 vs 0.667.** The headline "how much of the available headroom does the auditor capture"
number can be off by six orders of magnitude from one degenerate row. It is also unbounded above
and can be negative (a row where `self_gain < 0` while `oracle_headroom > 0`). This metric as
implemented is not reportable.

It is then averaged a *second* time across batches (`:383-387`), producing a mean of means of
ratios.

---

## 6. NEUTRAL MARGIN CONSISTENCY — exhaustive site list (section 8 of the audit)

### The three epsilons in play

| Constant | Value | Declared at | Consumed by |
|---|---|---|---|
| `neutral_margin` | `0.005` | `configs/self_audit_joint.yaml:32`, `configs/self_audit_auditor.yaml:29` | `losses/audit.py:20`, `audit_decomposition.py:185-186,481-482` |
| `epsilon_neutral` | **`0.02`** | `configs/self_audit_auditor.yaml:27` | `audit/counterfactual.py:282-289,477,485` |
| (implicit) | **`0.0`** | — | every site in the table below |

`configs/self_audit_auditor.yaml` declares **both** `epsilon_neutral: 0.02` (line 27) and
`neutral_margin: 0.005` (line 29) in the same block. The counterfactual generator accepts a
synthetic transition as "NEUTRAL" when `|delta| < 0.02`
(`counterfactual.py:485`, metadata flag at `:477`), but the ranking loss that consumes those same
transitions only zeroes out `|delta| < 0.005` (`losses/audit.py:20`). **A synthetic transition with
`delta = 0.01` is generated and labeled NEUTRAL, then trained by `signed_ranking_loss` as a
strictly beneficial ordering example** (`audit.py:23`, `ranking_actual > 0`). The auditor is taught
to rank a transition it was told is neutral. 4x mismatch between two constants in the same config
block.

### Every site where ZERO is the boundary while training uses a nonzero margin

| # | Site | Code | Used for | Impact |
|---|---|---|---|---|
| 1 | `metrics.py:286` | `predicted_sign = delta_q > 0.0` | improve/regress accuracy | prediction thresholded at 0, not `tau_accept`, not `neutral_margin` |
| 2 | `metrics.py:287` | `actual_sign = delta_dice > 0.0` | **AUROC / AUPRC / accuracy positive-class label** | a `+1e-9` Dice change is a positive; a `0.0` change is a negative |
| 3 | `metrics.py:316` | `harmful = accepted & (delta <= 0.0)` | `harmful_acceptance_rate` | **`delta == 0` counted as harmful** |
| 4 | `metrics.py:317` | `beneficial_rejection = ~accepted & (delta > 0.0)` | `beneficial_rejection_rate` | asymmetric with #3: zero belongs to "harmful", not to "beneficial" |
| 5 | `threshold.py:85` | `accepted & (actual[:,turn] <= 0.0)` | threshold calibration, harmful count | same zero-is-harmful bias, now steering `tau_accept` selection |
| 6 | `threshold.py:86` | `rejected & (actual[:,turn] > 0.0)` | threshold calibration, beneficial count | same asymmetry |
| 7 | `self_audit_net.py:346` | `return candidate_score > previous_score` | **the `oracle_accept` gate itself** | oracle accepts a `+1e-9` improvement; defines `modes/oracle_dice` and hence `OracleHeadroom` |
| 8 | `audit_decomposition.py:306` | `oracle_headroom > 1e-8` | `headroom_capture_ratio` denominator filter | see F3 |

Near-misses that are **not** violations, for completeness:
- `losses/audit.py:23-24` (`ranking_actual > 0` / `< 0`) sits *after* the
  `actual.abs() < neutral_margin` filter at `:20`, so it is margin-aware **provided the caller
  passes a margin**. The signature default is `neutral_margin: float = 0.0`
  (`audit.py:16`, `:74`) — a caller that forgets collapses to a zero boundary. All current callers
  do pass it (`train_auditor.py:204`, `finetune_joint.py:250`).
- `audit/targets.py:66-70` (`local_audit_targets`) defines FIX/REGRESS by per-pixel correctness
  flips. Exact integer comparison, no epsilon needed. Correct.
- `audit_decomposition.py:185-186,481-482` correctly use `> +margin` / `< -margin`. These are the
  only margin-correct classification sites in the repo.

### Concrete damage, executed

```
acceptance_metrics(accepted=[T,T,T,T,T,F], delta=[0,0,0,+1e-9,-0.20,+0.20])
  -> {'harmful_acceptance_rate': 0.8, 'beneficial_rejection_rate': 1.0, ...}
     (4 of 5 accepted flagged harmful; only ONE was actually harmful)

evaluate_threshold(tau=0.0, 3 accepted transitions all with delta == 0.0 exactly)
  -> harmful_acceptance_rate = 1.0
```

Exact-zero deltas are **not rare** here: any transition where the candidate argmax equals the
previous argmax gives `delta == 0.0` bit-exactly, and on a fully-empty slice (§2) that is the
common case. `harmful_acceptance_rate` — a headline safety metric, logged as
`val/harmful_acceptance_rate` (`finetune_joint.py:858`) — is therefore biased upward by an amount
equal to the accepted-neutral rate, and `select_threshold`
(`threshold.py:135-149`) uses `harmful_acceptance_rate` as its tie-break, so **the calibrated
`tau_accept` is systematically pushed higher (more conservative) than the neutral-margin semantics
warrant.**

The `> 0.0` at `metrics.py:287` also defines the AUROC/AUPRC positive class. Under the project's
own semantics, a `+0.001` transition is NEUTRAL; under the metric, it is a POSITIVE the auditor is
scored on ranking above every negative. **The reported AUROC is measuring a task the loss was never
trained on.**

---

## 7. Auditor AUROC / AUPRC

`binary_auroc` (`metrics.py:222-234`, rank-sum with 0.5 tie credit) and `binary_auprc`
(`:237-249`, average precision). Both consumed by `transition_audit_metrics` (`:268-296`).

- **Score**: `predicted_delta_q`, the global scalar head, flattened.
- **Positive class**: `actual_sign = delta_dice > 0.0` (`:287`) — "the candidate strictly increased
  Dice". Zero-boundary, see F4 #2.
- **Population — two different ones, same metric name:**
  - **Phase B (`val/auroc`, `train_auditor.py:415-419,553`): SYNTHETIC.** Transitions come from
    `CounterfactualGenerator` inside `_auditor_batch`; there is no `model.infer` gate involved.
    These are constructed edits, not transitions the annotation expert actually proposed.
  - **Phase C (`val/audit_auroc`, `finetune_joint.py:446-452,864`): ON-POLICY, active rows only.**
    `validate_phase_c` runs `model.infer(mode="self_audit")` (`:334-339`) and appends only rows
    selected by `_transition_active_mask` (`:361-367`, guarded by `if not bool(active_mask.any()): continue`
    at `:369`) — so halted rows are correctly excluded, and only *attempted* transitions are scored.
- **Class-imbalance handling**: **none for AUROC/AUPRC.** No prevalence is logged, so the AUPRC
  baseline (= positive rate) is unknown and an AUPRC of e.g. 0.7 cannot be interpreted. The
  *local* per-pixel head does get balancing (`train_auditor.local_class_weights`,
  `train_auditor.py:131`, `max_weight=5.0`, applied at `:187,205`; `finetune_joint.py:61,246`), but
  the **global `delta_q` head has no reweighting and no imbalance-aware metric.**
  `signed_ranking_loss` (`audit.py:26-31`) forms all positive-negative pairs, which is
  imbalance-robust in expectation, but nothing reports the balance.
- `binary_auroc` returns `nan` when either class is absent (`:230-231`) — correct — but
  `validate_auditor_epoch` then silently falls back to `improve_regress_accuracy` as
  `primary_metric` (`train_auditor.py:428-429`), and that is the metric checkpoint selection uses
  (`:544-547`). A run where all synthetic transitions land on one side would select checkpoints on
  a different metric with no visible signal in W&B.

---

## 8. `harmful_acceptance` / `beneficial_rejection` — exact definitions

Two independent implementations, both zero-boundary, with **different denominators**:

**`acceptance_metrics`** (`metrics.py:298-329`), used by `validate_phase_c` (`:430-433`):
```python
accepted_count  = max(int(accepted_array.sum()), 1)        # :314
rejected_count  = max(int((~accepted_array).sum()), 1)     # :315
harmful               = accepted_array & (delta <= 0.0)    # :316
beneficial_rejection  = (~accepted_array) & (delta > 0.0)  # :317
harmful_acceptance_rate    = harmful.sum() / accepted_count       # :319
beneficial_rejection_rate  = beneficial_rejection.sum() / rejected_count  # :320
```
Denominators are **accepted** and **rejected** counts respectively — i.e. these are conditional
rates P(harmful | accepted) and P(beneficial | rejected). Reasonable. The `max(...,1)` guards
silently turn "0 accepted" into a rate of 0/1 = 0 rather than NaN — a run that accepts nothing
reports a *perfect* `harmful_acceptance_rate` of 0.0.

**`evaluate_threshold`** (`threshold.py:78-104`): same predicates (`:85-86`), but denominators are
accumulated across turns (`:92-93`) with the same `max(...,1)` guard. Consistent with
`acceptance_metrics` in definition, inconsistent with `audit_decomposition`'s
`harmful_block_rate` / `beneficial_capture_rate` (`audit_decomposition.py:206-207`), which:
- use the **neutral margin** (`:185-186`) rather than zero, and
- condition on the **candidate class** (`P(rejected | harmful)`, `P(accepted | beneficial)`) rather
  than on the gate decision.

So the repo reports two families of gate-quality rates that use different epsilons *and* different
conditioning directions, under names similar enough to be confused in a paper table.
**Neither `harmful_acceptance_rate` nor `beneficial_rejection_rate` uses the same epsilon as the
training loss.**

---

## 9. F7 (P1) — W&B logging

`WandbLogger.log` at `training/_utils.py:995-1016`. Four `wandb.log` call sites:
`_utils.py:1012,1014` (metrics), `:1031,1033` (images), `visualizer.py:509` (figures).

### Keys actually logged, per phase

**Phase A** — `train_annotation.py:387-397`, `wandb_logger.log(payload, step=epoch+1)`:
`epoch`, `train/loss`, `train/lr`, `val/loss`, `val/macro_foreground_dice`, `val/dice_RV`,
`val/dice_MYO`, `val/dice_LV`, `best_macro_dice`.

**Phase B** — `train_auditor.py:550-563`, `step=epoch+1`:
`epoch`, `train/loss`, `train/lr`, `train/transitions`, `val/audit_loss`, `val/auroc`,
`val/auprc`, `val/local_fix_f1`, `val/local_regress_f1`, `val/correlation_delta_q`,
`val/improve_regress_accuracy`, `best_primary_metric`.

**Phase C** — `finetune_joint.py:851-869`, `step=epoch+1`:
`epoch`, `train/loss`, `train/annotation_loss`, `train/audit_loss`, `train/lr`,
`val/initial_dice`, `val/final_dice`, `val/net_gain`, `val/harmful_acceptance_rate`,
`val/beneficial_rejection_rate`, `val/mean_attempted_turns`, `val/mean_accepted_turns`,
`val/audit_auroc`, `val/audit_fix_f1`, `val/audit_regress_f1`, `best_final_macro_dice`.

**Single-process pipeline** — `scripts/train_self_audit.py:177-180` (`_log`), which prepends
`pipeline/phase` and `pipeline/epoch` and calls `logger.log(payload)` **with no `step`**:
- A: `phase_a/<train stats>` (`:325`), then **`val_stats` UNPREFIXED** (`:326`) —
  `val_loss`, `val_macro_foreground_dice`, `val_dice_class_1..3` — then `headroom` -> `phase_a/*`
  (`:327`).
- B: everything prefixed `phase_b/*` (`:424`).
- C: `phase_c/<train stats>` (`:530`), **`val_stats` UNPREFIXED** (`:531`) —
  `initial_foreground_macro_dice`, `final_foreground_macro_dice`, `net_gain`,
  `harmful_acceptance_rate`, `audit_auroc`, ... — then `decomposition` (`:532`) which contributes
  `modes/*`, `stage_<t>/*`, `audit/*`, `gt_firewall/*`.

### Findings

1. **No `audit/on_policy/*` vs `audit/synthetic/*` split — FLAGGED.** The synthetic (Phase B)
   auditor quality lands under `phase_b/auroc` / `val/auroc`; the on-policy (Phase C) quality lands
   under `audit_auroc` / `val/audit_auroc`; and the on-policy gate decomposition lands under a bare
   `audit/*` namespace (`audit_decomposition.py:228-244`) that gives no hint it is on-policy. A
   reader of the W&B run cannot tell from key names which distribution a number came from. This is
   exactly the distinction the project's central claim depends on (does the auditor generalize from
   synthetic counterfactuals to its own proposals?) and the logging does not make it legible.

2. **Namespacing is inconsistent within a single run.** Phase A and Phase C validation keys are
   logged bare while Phase B is fully prefixed. `val/*` from the standalone scripts and bare
   `val_macro_foreground_dice` from the pipeline are the same quantity under two names.

3. **Non-finite metrics are silently dropped** (`_utils.py:1008-1011`): only values passing
   `not isnan and not isinf` reach `clean_metrics`. So `modes/headroom_capture_ratio = nan`
   (no row had headroom), `stage_2/*` = nan (no row survived to stage 2), and
   `audit/beneficial_capture_rate = nan` (no beneficial candidates) all **vanish from W&B entirely**
   rather than appearing as gaps. "Metric absent" and "metric not computed" and "metric
   undefined" are indistinguishable in the dashboard. For a project whose thesis is
   "the auditor blocks harmful edits", silently dropping `harmful_block_rate` when there were no
   harmful candidates hides the most important null result.

4. **`_log` passes no `step`** (`train_self_audit.py:180`) so W&B auto-increments; each phase calls
   `_log` 2-3x per epoch, so the x-axis is an opaque counter. `pipeline/epoch` and `pipeline/phase`
   are logged, so it is recoverable, but no chart will be right by default.

5. `wandb.enabled: false, mode: offline` in `configs/self_audit_joint.yaml:41-45` — nothing is
   logged at all unless overridden.

---

## 10. Halted-row aggregation

### Correct

- **`SelfAuditNet.infer`** maintains per-row `active` and only runs the expert/auditor on active
  indices (`self_audit_net.py:141-152`), with `attempted = active.clone()` recorded per turn
  (`:144-146`) and `num_attempted_turns` incremented only for attempted rows (`:148`).
  `active = accepted` for `self_audit`/`oracle_accept` (`:250-251`), so a rejection halts that row
  and only that row.
- **`decompose_self_audit_output`** selects active rows before computing anything
  (`audit_decomposition.py:175-177`) and returns NaN (not 0) for a stage with `active_count == 0`
  (`:141-159`).
- **`validate_phase_c`** builds transition targets only on active rows
  (`finetune_joint.py:361-384`) and skips stages with no active rows (`:369`).
- **`evaluate_threshold`** halts per-sample (`threshold.py:88`, comment at `:86-88`) and divides by
  `attempted.sum()`, not by `N*T` (`:96`).

I could not falsify the per-row masking. It is done correctly at every site I checked.

### F6 (P1) — the bug is one level up, in cross-batch aggregation

`evaluate_audit_decomposition` (`audit_decomposition.py:380-388`):

```python
keys = sorted(set().union(*(row.keys() for row in rows)))
for key in keys:
    values = torch.tensor([row[key] for row in rows if key in row and torch.isfinite(...)], ...)
    aggregate[key] = float(values.mean().item()) if values.numel() else float("nan")
```

An **unweighted mean over batches of per-batch rates**, with non-finite batches dropped. Two
consequences:

1. **Rate metrics are wrong whenever the active-row count varies across batches** — which is
   exactly what happens at stages 1..T-1, since rows halt. Executed with the real function:

```
batch with 8 active rows, 1 accepted -> stage_0/accept_rate = 0.1250
batch with 1 active row,  1 accepted -> stage_0/accept_rate = 1.0000
unweighted batch mean (what is reported) = 0.5625
transition-weighted truth ((1+1)/(8+1))  = 0.2222
```

   0.5625 vs 0.2222. The same defect applies to `accept_rate`, `reject_rate`,
   `beneficial_capture_rate`, `harmful_block_rate`, `candidate_gain`, `headroom`, `realized_gain`,
   `audit_gate_value` and the whole `audit/*` block — every one of them is a per-batch mean being
   averaged again without weights.

2. **NaN batches are dropped, so late-stage metrics are conditioned on survivorship.**
   `stage_2/accept_rate` is the mean over only those batches where at least one row survived to
   stage 2. Batches where everything halted early contribute nothing. The reported late-stage
   accept rate is therefore biased toward the batches where the gate was most permissive.

3. `stage_t/attempt_count` (`:139`) is a raw count per batch, then averaged over batches — it is
   "mean attempts per batch", not a total, despite the name. Depends on `batch_size`.

`evaluate_annotation_headroom` (`:503-540`) has the identical aggregation defect at `:534-538`.

### F10 (P2) — GT-firewall probe covers one batch

`probe_gt_leakage` is invoked only under `if batch_index == 0` (`audit_decomposition.py:363-372`).
`gt_firewall/passed` therefore certifies a single batch. The probe itself is well constructed
(`:391-452`): it runs `mode="self_audit"` twice with two different `oracle_target` tensors and
checks `max|logits_a - logits_b| <= 1e-7` **and** decision-mismatch rate `== 0` on
`accepted_count`/`halt_turn` (`:437-450`). `infer` does ignore `oracle_target` outside
`oracle_accept` (`self_audit_net.py:194-201`), so this is a genuine dynamic check, not a
tautology — it just is not run often enough to be a guarantee.

### F9 (P2) — hardcoded constant in `headroom_collapse_ratio`

`audit_decomposition.py:495-499`:
```python
result["phase_a/headroom_collapse_ratio"] = float(
    1.0 - min(max(result["phase_a/mean_stage_headroom"] / 0.005, 0.0), 1.0)
)
```
`0.005` is hardcoded even though `decompose_annotation_output` receives `neutral_margin` as a
parameter (`:459`) and uses it correctly at `:481-482`. Passing `--neutral_margin 0.01` changes the
beneficial/harmful rates but silently leaves the collapse ratio normalized against 0.005. The
metric is also a clipped linear rescaling with no stated justification — a "collapse ratio" of 1.0
means only "mean stage headroom <= 0.005", which the raw `phase_a/mean_stage_headroom` already says.

---

## Additional observations

- **`evaluation/__init__.py:12` imports `.visualizer`, which imports `matplotlib` unconditionally
  (`visualizer.py:23`).** `scripts/audit_checkpoint.py` cannot run on a machine without matplotlib,
  even though it produces no figures. Verified: `import self_audit.evaluation.audit_decomposition`
  raises `ModuleNotFoundError: No module named 'matplotlib'` in this environment.
- **`SelfAuditNet.infer` returns `"states": candidates`** (`self_audit_net.py:269`) — the
  *candidate* list, including rejected candidates, not the accepted state trajectory.
  `decompose_annotation_output` falls back to `[initial] + output["states"]`
  (`audit_decomposition.py:463-466`) when no `state_trace` is present. On `forward_annotation`
  output (`:87-89`) that fallback is never taken, so no current call site is wrong — but the key
  name is a trap for any future caller passing `infer` output.
- **`volume_inference.py:279-281,295-296`** annotate locals as `list[Tensor]` while only `torch` is
  imported (no `from torch import Tensor`). PEP 526 local annotations are not evaluated at runtime,
  so this does not raise — it is dead type information only.
- **`tests/test_self_audit_volume.py`** covers reconstruction shape, duplicate-index rejection,
  2.5-D boundary replication, and `annotation_metrics` field presence — but **`test_annotation_metrics_report_required_fields`
  (`:33-37`) asserts `result["dice"] == 1.0` on a `pred == target` input**, which is exactly the
  case the empty-empty convention makes uninformative. No test in the repo asserts anything about
  volume-level vs slice-level aggregation, or about ED/ES separation.
- **`select_threshold`** (`threshold.py:141-149`) maximizes `final_macro_dice` where `final` is
  built by *adding cached `actual_delta_dice` values to `initial_dice`* (`threshold.py:87`). This
  assumes Dice deltas compose additively across turns, which they do not in general — the cached
  delta for turn `t` was measured against the state reached under whatever gate produced the cache,
  not under the candidate `tau`. Calibration is simulating a counterfactual trajectory with
  on-policy deltas. Out of my assigned scope to trace fully; flagged for whoever owns
  `scripts/cache_validation_transitions.py`.

---

## Recommended priority for the author

1. **F1** — pick one aggregation unit. Report volume-level 3-D Dice per patient per phase
   (ED/ES separately) via the already-written `evaluate_comparison_modes`, and wire it into a
   script. This alone fixes F1 and F5.
2. **F2** — add an `empty_score=nan` / exclude-absent-classes reporting mode, and report the
   present-class-only Dice alongside the current number so the inflation is visible.
3. **F3** — replace `modes/headroom_capture_ratio` with `sum(self_gain)/sum(oracle_headroom)` over
   rows with `oracle_headroom > neutral_margin`.
4. **F4** — route `neutral_margin` into `metrics.py:286-287,316-317` and `threshold.py:85-86`;
   reconcile `epsilon_neutral=0.02` against `neutral_margin=0.005`.
5. **F6** — accumulate raw counts and deltas across batches and reduce once, instead of averaging
   per-batch means.
6. **F7** — namespace as `audit/synthetic/*` and `audit/on_policy/*`; log NaN as a sentinel
   (e.g. `-1`) or log a companion `*_count` so absent populations are visible.
