# Worker C — Auditor + Counterfactual Branch Audit

**Repo:** `/Users/alvinluong/Self-Audit` @ `a901eea` (branch `main`), read-only audit.
**Environment:** torch 2.13.0 available. `matplotlib` and `pytest` were absent and installed into the ambient env to enable evidence collection (no repo files touched). `PyYAML` still absent, so one config-loading test fails for environment reasons only.
**Test baseline:** `python -m pytest tests/test_self_audit_audit.py tests/test_self_audit_hardening.py tests/test_audit_decomposition.py -q` → **15 passed, 1 failed** (`test_checked_in_acdc_npy_convention_is_explicit`, `ImportError: PyYAML is required to load Self-Audit configs` — env, not logic).
**Data:** `preprocessed_data/` and `weights/` do not exist in this checkout. No trained checkpoint, no ACDC volumes. Every quantitative number below therefore comes from **synthetic phantoms fed through the repo's own production code paths**, not from real ACDC. Where that matters I say so explicitly.

Probe scripts used for the executable evidence live in the session scratchpad (`probeA.py` … `probeH.py`); each is reproduced in outline where it is cited.

---

## 1. Exact inputs of `CounterfactualAuditor.forward`

`src/self_audit/models/auditor.py:76-116`.

Signature (`auditor.py:76-84`):

```
forward(H, P_previous, P_candidate, delta_P=None, entropy_previous=None, entropy_candidate=None) -> AuditOutput
```

| Arg | Shape | Detached? | Where |
|---|---|---|---|
| `H` | `[B, feature_channels=96, H/4, W/4]` | **yes**, `H_audit = H.detach()` | `auditor.py:91` |
| `P_previous` | `[B, 4, H, W]` | **yes**, `.detach()` before `_probabilities` | `auditor.py:92` |
| `P_candidate` | `[B, 4, H, W]` | **yes** | `auditor.py:93` |
| `delta_P` | `[B, 4, H, W]`, default `candidate - previous` | **yes** when supplied | `auditor.py:94` |
| `entropy_previous` | `[B, 1, H, W]`, default computed internally | **yes** when supplied | `auditor.py:104` |
| `entropy_candidate` | `[B, 1, H, W]` | **yes** | `auditor.py:109` |

Concatenated input channel count is `feature_channels + 3*num_classes + 2 = 96 + 12 + 2 = 110` (`auditor.py:52`). Verified at runtime: `input_projection[0].in_channels == 110`.

**Does it see the image?** No. It sees only the **shared FPN feature map**, never raw pixels. In `train_auditor.py:67-75` `_feature_for_audit` calls `model.encode(images)` and returns `encoded["shared"]`; in `self_audit_net.py:183-190` `infer` passes `shared_active.detach()`. Measured stride: for a `256×256` input the shared map is `[1, 96, 64, 64]` → **stride 4** (probe: `SelfAuditNet.encode`).

**Does it see `A_t` and the candidate separately?** Yes — both, *plus* their difference. `delta_P` is information-theoretically redundant given `previous` and `candidate` (it is exactly `candidate - previous` in every production call site: `train_auditor.py:196`, `self_audit_net.py:187`, `train_auditor.py:598`). It is an inductive-bias convenience, not extra information.

**Executable detach evidence** (probeA): with `feat`, `prev`, `cand` all `requires_grad=True`, after `(local_logits.sum() + delta_q.sum()).backward()`:

```
local_logits (2, 3, 32, 32)  delta_q (2, 1)
grad->features: None  grad->prev: None  grad->cand: None
```

**Resolution finding (P2, structural).** The auditor downsamples `previous`/`candidate`/`delta` to feature resolution (`auditor.py:97-100`), runs the whole trunk at stride 4, and only then upsamples `local_logits` back to full resolution (`auditor.py:114`). The local cross-entropy target, however, is built at **full pixel resolution** (`targets.py:55-70`). The local head is therefore architecturally blurred by 4× relative to its own target. Probe G2 (oracle upper bound: take the *true* local target, push it through an `area`-downsample to 64×64 and a bilinear upsample back to 256×256, argmax):

```
positive     FIX      target_px=      1 (0.0004%)  oracle-through-stride4 RECALL=0.0000
negative     REGRESS  target_px=   2717 (1.0365%)  oracle-through-stride4 RECALL=0.9779
hard_neutral FIX      target_px=    153 (0.0584%)  oracle-through-stride4 RECALL=0.1830
hard_neutral REGRESS  target_px=    112 (0.0427%)  oracle-through-stride4 RECALL=0.1518
```

Thin FIX/REGRESS bands — exactly the "correction headroom" the project claims to localize — are **not representable** at the resolution the local head runs at. Bulk regressions (component deletion, class swap) survive.

---

## 2. Transition quality vs. segmentation quality of the candidate

**Verdict: the target is a genuine transition target, but nothing in the code verifies it, and there is a large generator-artifact shortcut.**

*Analytic argument.* The local target (`targets.py:65-69`) is a function of the pair `(previous_correct, candidate_correct)`. Knowing `candidate_correct` alone leaves the label ambiguous in both directions (`cand` correct ⇒ `{FIX, UNCHANGED}`; `cand` wrong ⇒ `{REGRESS, UNCHANGED}`). The global target (`targets.py:90-94`) is a literal difference of two Dice scores. Neither collapses to a candidate-only statistic in principle.

*Executable evidence — candidate-only shortcut is NOT available* (probeE, 222 generated transitions across positive/negative/hard-neutral kinds and heterogeneous `previous` quality):

```
std Dice(prev)=0.2472  std Dice(cand)=0.2322  std dDice=0.0734
corr(dDice, Dice(cand))            = -0.0535
corr(dDice, -Dice(prev))           = +0.3473
AUROC(score=Dice(cand), label=dDice>0)  = 0.5251   (pos rate=0.212)
```

So an auditor that ignored `P_previous` and only regressed candidate quality would score at chance on the sign task. **`P_previous` is load-bearing.** Good.

*Executable evidence — a different shortcut IS available* (probeF, 274 transitions). Six hand-crafted scalars computed **only from `delta_P` and the entropy channels** — no image features, no GT, no anatomy — fed to a logistic regression:

```
AUROC(dDice>0 | peak delta                                    ) = 0.7055
AUROC(dDice>0 | ALL 6 handcrafted delta/entropy stats, logistic) = 0.8198
```

The synthetic operations leave a **fingerprint in `delta_P`**. `_positive_single` blends toward a one-hot GT vertex (`counterfactual.py:318-320`) producing large peak deltas; `_transfer_to_background` / `_transfer_between` (`counterfactual.py:221-236`) move mass with a characteristically different profile. A convolutional auditor with full spatial access will exploit this far better than my six scalars do. **This shortcut does not transfer to on-policy transitions**, which carry no such signature.

*Ablation/symmetry evidence in the repo:* **none.** `grep` over `src/` and `tests/` for `ablat|symmetr|shuffl|permut` finds only the GT-firewall probe (`evaluation/audit_decomposition.py:392-451`), which permutes GT to prove GT does not reach inference — a different question. There is **no test that zeroes/swaps `P_previous`** and confirms the auditor degrades. There is **no test that evaluates the auditor on on-policy transitions alone.**

---

## 3. Local FIX / UNCHANGED / REGRESS target construction

`src/self_audit/audit/targets.py:55-70`. Class indices (`targets.py:12-15`):

```
FIX = 0 ; UNCHANGED = 1 ; REGRESS = 2 ; LOCAL_AUDIT_NAMES = ("FIX", "UNCHANGED", "REGRESS")
```

Exact per-pixel predicate (`targets.py:65-69`), where all three tensors are hard argmax labels via `_labels` (`targets.py:47-52`):

```
previous_correct  = (argmax(previous)  == gt)
candidate_correct = (argmax(candidate) == gt)
target = UNCHANGED                                        # default, index 1
target[~previous_correct &  candidate_correct] = FIX      # index 0
target[ previous_correct & ~candidate_correct] = REGRESS  # index 2
```

**There is no epsilon anywhere in the local target.** The class named "NEUTRAL" in the design spec is implemented as `UNCHANGED`, and it is not a margin band — it is the union of two semantically opposite cases: *both labels correct* (the overwhelming majority, background) **and** *both labels wrong* (an unrepaired error, arguably the most interesting pixels in the whole problem). Collapsing "correctly left alone" with "error not fixed" into one class is a **MISMATCH with the stated neutral-margin semantics** and destroys the signal that would distinguish a cautious edit from a useless one.

**Is the neutral class degenerate in practice? It is the opposite — it swallows everything.** Probe C/D over the repo's own generator at several `previous` confidence levels (`p_max` = mean max softmax of `previous`):

```
scale= 2.0 shift=1 p_max=0.711 | FIX%=0.336 UNCH%=99.664 REG%=0.000 | CE weights=[5.0, 0.502, 1.0]
scale= 6.0 shift=1 p_max=0.993 | FIX%=0.085 UNCH%=99.915 REG%=0.000 | CE weights=[5.0, 0.500, 1.0]
scale=12.0 shift=3 p_max=1.000 | FIX%=0.189 UNCH%=99.811 REG%=0.000 | CE weights=[5.0, 0.501, 1.0]
```

and pooled over all three kinds at 256×256 (probe G): `FIX=0.00020  UNCHANGED=0.99621  REGRESS=0.00360`.

**UNCHANGED is 99.6–99.9 % of pixels; FIX is 0.02–0.5 %.** That is a 200:1 to 5000:1 imbalance.

**The class weighting cannot compensate.** `train_auditor.py:131-142` computes inverse-frequency weights but clamps them to `(0.25, 5.0)` (`train_auditor.py:141`). Measured above: `FIX` weight pins at the **ceiling 5.0**, `UNCHANGED` sits at 0.50 — an effective 10:1 correction against a ≥200:1 imbalance. The local cross-entropy (`losses/audit.py:88`) is dominated by UNCHANGED by roughly one to two orders of magnitude. Combined with the stride-4 bottleneck (§1), `local_fix_f1` is close to structurally unlearnable.

Secondary metric caveat: `f1_for_label` (`evaluation/metrics.py:255-262`) returns **1.0** when the class is absent from both prediction and target. A local head that simply never predicts FIX will score `local_fix_f1 = 1.0` on any batch containing no FIX pixels. In `validate_auditor_epoch` the metric is computed once over the whole concatenated validation set (`train_auditor.py:413-418`), so this is unlikely to fire there, but it does fire in per-batch/decomposition contexts.

---

## 4. Global `dQ` target

`src/self_audit/audit/targets.py:90-94`, wrapped by `build_transition_targets` at `targets.py:97-107`.

```
delta_dice_target = multiclass_dice(argmax(candidate), gt) - multiclass_dice(argmax(previous), gt)
```

- **Yes**, it is exactly `Dice(candidate, GT) − Dice(previous, GT)`.
- **Per-class?** No. `multiclass_dice` (`targets.py:73-87`) loops `cls in range(1, num_classes)` — foreground classes 1,2,3 only, background excluded — and returns `torch.stack(values, dim=1).mean(dim=1)`, i.e. a **macro mean over the 3 foreground classes**. The auditor never sees a per-class breakdown; a +0.3 RV gain cancelling a −0.3 LV loss is indistinguishable from a no-op.
- **Per-volume or per-slice?** **Per-slice.** `pred.flatten(1).sum(1)` (`targets.py:80`) reduces over the spatial dims only, keeping the batch dim. Batches are 2-D slices (`configs/self_audit_auditor.yaml:12` `batch_size: 4`, dataset yields `[B,3,H,W]`). Result is `[B]`, unsqueezed to `[B,1]` at `targets.py:106`.
- **Empty-class convention:** when both prediction and GT are empty for a class, Dice is defined as **1.0** (`targets.py:81-85`). On ACDC's many empty apical/basal slices this pins `dDice` at exactly 0 regardless of what the model does.
- **Normalized / scaled / clipped?** **No.** Raw signed float in `[-1, +1]`. The auditor's global head is an unbounded linear output (`auditor.py:60-66`), regressed via `smooth_l1_loss` (`losses/audit.py:27, 30`) with no output activation or scaling.

**Which epsilon, and where — three different values are in play and they do not agree:**

| Constant | Value | Where set | Where used | Meaning |
|---|---|---|---|---|
| `neutral_margin` | **0.005** | `configs/self_audit_auditor.yaml:29`, `configs/self_audit_joint.yaml:32` | `losses/audit.py:20` `neutral = actual.abs() < neutral_margin` | pulls `delta_q → 0` |
| `epsilon_neutral` | **0.02** | `configs/self_audit_auditor.yaml:27` | `counterfactual.py:485` hard-neutral retry acceptance | generator target band |
| implicit **0.0** | 0.0 | hard-coded | `evaluation/metrics.py:286-287` `actual_sign = delta_dice > 0.0` | AUROC/AUPRC label |

The headline AUROC therefore uses **no neutral band at all**: a transition with `dDice = 0.0` exactly (a no-op — see §5) is labelled **negative**, and a transition with `dDice = +0.003` is labelled **positive** while the loss simultaneously trains its `delta_q` toward 0. The reported metric and the training objective disagree about the same samples.

---

## 5. `CounterfactualGenerator` — every emittable transition type

`src/self_audit/audit/counterfactual.py`. Kinds are `Literal["positive", "negative", "hard_neutral", "on_policy"]` (`counterfactual.py:20`).

| Kind | Operation string | Code | GT-derived? | Notes |
|---|---|---|---|---|
| `positive` | `local_connected_error_repair` | `counterfactual.py:310-335` | **GT-derived (synthetic)** — error region is `argmax(previous) != gt` (`:312`), blend target is `one_hot(gt)` (`:318`) | single operation only |
| `negative` | `local_erosion` | `:373-381` | **prediction-derived (synthetic)** | `ground_truth` is in `_negative_single`'s signature (`:340`) but **never referenced in its body** — verified by `grep` over lines 337-448 |
| `negative` | `local_dilation` | `:382-390` | prediction-derived | |
| `negative` | `boundary_displacement` | `:391-403` | prediction-derived | random shift `dy,dx ∈ [-2,2]`, forced non-zero |
| `negative` | `hole_insertion` | `:404-414` | prediction-derived | requires ≥4 interior px |
| `negative` | `false_island` | `:350-367` | prediction-derived | island capped at `max(4, min(32, HW/64))` px |
| `negative` | `component_deletion` | `:415-422` | prediction-derived | **partial** — strength ∈ [0.35,0.75], not a full delete |
| `negative` | `semantic_class_swap` | `:423-430` | prediction-derived | needs `num_classes > 2` |
| `hard_neutral` | `local_repair_plus_<negative op>` | `:450-493` | **GT-derived (synthetic)**, and GT-*optimized*: retries until `|dDice| < epsilon_neutral` using `multiclass_dice(..., ground_truth)` at `:462, :485` | superposition of one positive + one avoid-masked negative |
| `on_policy` | `annotation_expert_transition` | `:566-583` | **model-derived** | **DEAD CODE** — `grep` shows `on_policy_transition` has zero call sites in `src/` or `scripts/`. `generate(kind="on_policy")` returns `_invalid_edit` (`:561-562`). |
| — | `mixed` | `:512-513` | — | label emitted when a batch's per-item operations differ, not a distinct operation |

**Sampling probabilities** (`counterfactual.py:265-291`, `_pick_kind` at `:302-308`):

```
positive_fraction = 0.4   negative_fraction = 0.4   hard_neutral_fraction = 0.2
```

One `torch.rand()` draw per `generate()` call selects the kind for the **entire batch** (`:547`), not per sample. Within `negative`, the operation is drawn uniformly from the 7-tuple (`:292-300`, `:557`) → **1/7 ≈ 14.3 %** each.

**Config knobs and defaults.** `configs/self_audit_auditor.yaml:26-30` sets **only**:

```
counterfactual:
  epsilon_neutral: 0.02
  neutral_max_retries: 8
  neutral_margin: 0.005            # consumed by the loss, not the generator
  local_class_weighting: balanced_clamped
```

`train_auditor.py:505-509` constructs the generator with **only** `epsilon_neutral`, `neutral_max_retries`, `num_classes`. **`positive_fraction` / `negative_fraction` / `hard_neutral_fraction` / `min_repair_fraction` / `max_repair_fraction` are not exposed in any config and are not overridable from YAML** — they are hard-pinned at `0.4 / 0.4 / 0.2 / 0.3 / 0.6`.

`configs/self_audit_joint.yaml` has **no `counterfactual:` block at all** (`grep -n counterfactual configs/*.yaml` matches only the auditor config), and `finetune_joint.py` never imports or constructs a `CounterfactualGenerator` — Phase C is 100 % on-policy.

### 5b. P1 — the `positive` branch is largely a no-op at realistic model confidence

`_positive_single` (`counterfactual.py:316-320`) blends
`candidate = previous·(1−m) + one_hot(gt)·m` with `m ~ U(min_repair_fraction, max_repair_fraction) = U(0.3, 0.6)` (`:317`, defaults `:271-272`).

For a confidently-wrong pixel with `p_wrong ≈ 0.99`, flipping the argmax requires `m ≳ 0.5`. Only ~34 % of the `U(0.3,0.6)` mass clears that. Measured (probeH, 80 samples per row, repo generator, `p_max` = mean max-prob of `previous`):

```
prev p_max | N  | frac positive-CF with ZERO argmax change | frac |dDice|<0.005 | mean dDice
  0.475    | 80 |          0.000                          |      0.738        | +0.00632
  0.711    | 80 |          0.287                          |      0.863        | +0.00343
  0.948    | 80 |          0.575                          |      0.912        | +0.00175
  0.993    | 80 |          0.675                          |      0.912        | +0.00225
  1.000    | 80 |          0.675                          |      0.950        | +0.00138
```

A trained segmentation network operates at `p_max` ≈ 0.95–1.0. In that regime:

- **~58–68 % of "positive" counterfactuals change zero pixels** — `candidate == previous`, `dDice == 0.0` exactly. `_positive_single` still returns `valid=True` (`:332`); there is **no check that the edit actually altered the argmax**.
- **91–95 % of "positive" counterfactuals land inside the loss's neutral band** `|dDice| < 0.005` (`losses/audit.py:20`), so `signed_ranking_loss` actively pulls their predicted `delta_q` toward **0** (`losses/audit.py:36`).
- The gate accepts on **strict** `delta_q > tau_accept` with `tau_accept = 0.0` (`configs/self_audit_joint.yaml:30`, `self_audit_net.py:204`). The training signal therefore parks genuine repairs exactly on the reject side of the decision boundary.
- In the headline metric, those same no-ops are labelled **negative** (`metrics.py:287`, `> 0.0`) and are trivially separable (their `delta_P` is identically zero), padding AUROC/AUPRC with free true-negatives.

Net effect on the class prior: probeF measured `P(dDice > 0) = 0.230` over the generator's own mixture. **The auditor's curriculum is ~77 % "do not accept."** Mechanistically this predicts a gate that rejects almost everything at `tau = 0`, i.e. `self_audit_dice ≈ initial_dice` — which is precisely the failure mode that would falsify the "auditor captures the headroom" hypothesis while still producing a respectable AUROC.

---

## 6. Synthetic vs. on-policy ratio — the number

`build_auditor_transitions` (`train_auditor.py:88-128`) builds, per batch:

- **on-policy** groups: `len(trajectory) - 1` adjacent pairs (`:97-106`)
- **synthetic** groups: `len(trajectory)` — one `generator.generate(..., kind=None)` around **every** state including `A_0` (`:107-127`)

`forward_annotation` returns `state_trace = [initial] + states` with `max_turns = 3` (`self_audit_net.py:87`, `configs/self_audit_auditor.yaml:38`) ⇒ `len(trajectory) = 4`.

**Verified at runtime** (probeB, real `SelfAuditNet`, `max_turns=3`, batch 2):

```
state_trace len: 4
transitions: 7 {'on_policy': 3, 'synthetic': 4}
   on_policy turn 0 / 1 / 2                                   valid [True, True]
   synthetic:negative:hole_insertion              turn 0      valid [True, False]
   synthetic:negative:semantic_class_swap         turn 1      valid [True, True]
   synthetic:hard_neutral:local_repair_plus_local_erosion  turn 2  valid [True, True]
   synthetic:negative:mixed                       turn 3      valid [True, True]
```

The per-batch loss is a **uniform mean over transition groups** — `torch.stack(losses).mean()` (`train_auditor.py:217`) — so:

> ### **Synthetic : on-policy = 4 : 3. Synthetic carries 57.1 % of the Phase-B loss weight, on-policy 42.9 %. Identical per training step and per validation step.**

At the sample level the synthetic share is slightly lower because invalid rows are dropped (`train_auditor.py:180-185`) while on-policy `valid_mask` is always all-ones (`:104`); probeB measured 7 valid synthetic sample-transitions vs 6 on-policy = 53.8 %. But the *loss weight* is per-group, so 57.1 % is the operative number whenever each synthetic group retains ≥1 valid row.

Decomposing the synthetic 4 by kind at the 0.4/0.4/0.2 prior: **1.6 positive, 1.6 negative, 0.8 hard-neutral** per batch in expectation. Folding in §5b (~65 % of positives are no-ops at realistic confidence), the expected count of *genuinely improving synthetic transitions* is **≈ 0.5 out of 7 groups per batch (~7 %)**.

Phase C (`finetune_joint.compute_joint_losses`, `finetune_joint.py:220-256`) uses **zero** synthetic transitions — 100 % on-policy, drawn from the gated `infer` trajectory.

---

## 7. Does Phase-B validation use the same GT-driven generator?

## **Yes. Loudly yes. This contaminates every headline Phase-B auditor metric.**

`validate_auditor_epoch` (`train_auditor.py:358-442`) takes a `generator` argument (`:364`), defaults it to a fresh `CounterfactualGenerator()` if `None` (`:376`), and calls the **exact same** `_auditor_batch` used for training (`:392-402`) with `collect=True`. `main` passes the **same generator instance** to both (`train_auditor.py:522` train, `:530` val). `_auditor_batch` calls `build_auditor_transitions` unconditionally (`:170`), which calls `generator.generate(state, ground_truth)` (`:108`).

Consequences, in order of severity:

1. **57.1 % of the loss weight behind `val/auroc`, `val/auprc`, `val/local_fix_f1`, `val/local_regress_f1`, `val/correlation_delta_q_delta_dice` and `val/improve_regress_accuracy` comes from GT-synthesized edits**, not from anything the annotation expert would ever propose. These are logged as the headline Phase-B numbers (`train_auditor.py:550-573`).
2. `primary_metric` — the quantity that selects `best.pt` and the exported `phase_b_auditor.pt` (`train_auditor.py:428-429`, `:540-549`) — is **AUROC on this contaminated mixture**. Model selection is therefore optimizing partly for "detect the generator's fingerprint."
3. Per §2, an AUROC of **0.82 is reachable from `delta_P`/entropy summary statistics alone** on the generator's distribution. A large share of the reported Phase-B AUROC may be measuring generator-artifact detection, and that skill is worth nothing at inference, where only on-policy transitions occur.
4. The same GT-synthesized mixture appears in the Phase-B **visualization** path (`train_auditor.py:588`), so qualitative figures are also not on-policy.

**What is missing:** there is no code path anywhere in the repo that evaluates the auditor on **on-policy transitions only**. `_auditor_batch` has no flag to disable synthetic generation, and `build_auditor_transitions` has no such parameter. The on-policy-only slice is not computed, not logged, and not reported. That number — AUROC restricted to `provenance == "on_policy"` — is the only Phase-B metric that would actually support the central hypothesis, and it does not exist.

**Recommended fix (not applied — read-only audit):** thread a `use_synthetic: bool` through `_auditor_batch`/`build_auditor_transitions`; keep synthetic augmentation for training, set it `False` for validation, or at minimum report AUROC/AUPRC split by the `provenance` field already carried in the transition dicts (`train_auditor.py:103`, `:123`).

---

## 8. Threshold calibration — objective, split, GT, and the selection-bias P1

**Objective.** `select_threshold` (`evaluation/threshold.py:136-149`) maximizes the lexicographic key

```
(final_macro_dice, −harmful_acceptance_rate, −|tau_accept|)
```

i.e. **primarily validation final macro foreground Dice**, with harmful-acceptance and then smallest-|tau| as tie-breaks. `scripts/calibrate_threshold.py:60` labels it `"objective": "maximize final validation macro Dice"`. An optional hard constraint `max_harmful_acceptance_rate` filters rows first (`threshold.py:129-132`), defaulting to `None` (`calibrate_threshold.py:42`) — **off by default**.

**Grid.** `calibrate_threshold.py:39-41`: 41 thresholds over `[-0.20, +0.20]`. The in-pipeline path uses a finer grid: 81 thresholds over `[-0.02, +0.02]` (`scripts/train_self_audit.py:88-90`, `:579-583`).

**Split.** Validation only, and this part is clean. `cache_validation_transitions.py:56` builds the dataset from `config["val_split"]`; `train_self_audit.py:572` passes `val_c`, which `_make_loaders` (`scripts/train_self_audit.py:147-151`) builds from `config.get("val_split", "val")`. No test data is touched. `splits/acdc_patient_split_seed42.json` is patient-disjoint and `data/common.py:226` raises on cross-split patient leakage.

**Which GT.** Validation GT, twice over. `collect_validation_transition_cache` (`finetune_joint.py:487-555`) runs `mode="always_accept_refinement", tau_accept=-inf` (`:510-515`) to expose the full trajectory, then for each turn computes `build_transition_targets(previous, candidate, batch["mask"])` (`:536`) — **GT-derived `actual_delta_dice`** — and `_foreground_dice_per_sample(initial, batch["mask"])` for `initial_dice` (`:519`). `evaluate_threshold` (`threshold.py:77-90`) then simulates the gate with `active = accepted` halting and accumulates `final += actual[:, turn]` on accepted turns (`:87`). The telescoping is exact for a prefix of the always-accept trajectory, and the halting rule matches `self_audit_net.infer:249-250`, so the simulator itself is faithful.

**The P1.** `train_self_audit.py:570-598` calibrates on `val_c` and then reports `best_threshold["final_macro_dice"]` — **the maximized objective value, on the same split it was maximized over** (`:595-597`, stored to `report["calibration"]` at `:586-591`). Taking a max over 81 correlated threshold estimates on one split yields an upward-biased Dice. No held-out re-evaluation exists: `grep -n "test" scripts/train_self_audit.py` returns nothing, and `run_full_pipeline.sh` (lines 301, 311, 317) ends the pipeline at calibration. **`test_split: test` in all three configs is declared but never evaluated anywhere in the repo.**

**A second, opposite problem.** The calibrated `tau` is a **terminal artifact**. `calibrate_threshold.py:66` writes `calibration.json`; `grep` over `scripts/run_full_pipeline.sh` for `calibration.json|selected|tau_accept` finds only two echo/comment lines — nothing reads it back. Meanwhile every reported self-audit Dice — `validate_phase_c` (`train_self_audit.py:508-518`) and `evaluate_audit_decomposition` (`:519-528`) — uses `tau_accept` from `audit_cfg.get("tau_accept", 0.0)` (`:471-473`), i.e. **the hard-coded 0.0**, not the calibrated value. So the pipeline simultaneously (a) reports an optimistically-selected `final_macro_dice` that no deployed configuration uses, and (b) reports self-audit Dice at an **uncalibrated** `tau = 0.0`. The two numbers are not comparable and neither is a clean held-out estimate.

---

## 9. `audit/gate.py` — accept predicate, hysteresis, consistency with `infer`

**Exact predicate** (`gate.py:19-22`):

```python
def threshold_accept(delta_q, tau_accept=0.0):
    return delta_q.detach().reshape(-1) > float(tau_accept)
```

**Strictly greater than**, on the detached scalar global `delta_q`, default `tau_accept = 0.0` (`gate.py:30, 42`; `configs/self_audit_joint.yaml:30`).

**Does local evidence enter the predicate? No.** `accept_reject` (`gate.py:25-38`) takes only `previous, candidate, delta_q, tau_accept, active`. `local_logits` is not a parameter and appears nowhere in `gate.py`. The same holds in production: `self_audit_net.py:191-205` computes `local_logits_active` and `delta_q_active`, and the accept decision at `:204` is `delta_q > float(tau_accept)` — **`local_logits` never touches it**. The local head's only downstream use is as recurrent *input* to the annotation expert on the next turn (`self_audit_net.py:216-221`: softmaxed, detached, and zeroed on reject).

> The design spec says "local transition evidence {FIX, NEUTRAL, REGRESS} **+** global predicted dQ → ACCEPT/REJECT". The implemented gate is **global-only**. This is a real spec mismatch, not a naming quibble.

**Hysteresis / halting — where does it live?** There is **no hysteresis** anywhere (no dwell counter, no dual threshold, no minimum-turns rule). Halting is one-shot: the first rejection freezes the sample for the rest of the episode.

- `gate.py:80`: `active = active & decision.accepted`
- `self_audit_net.py:249-250`: `if mode == "self_audit" or mode == "oracle_accept": active = accepted`
- `threshold.py:90`: `active = accepted`

**Are the two consistent?** Behaviourally **yes**, and all three simulators agree:

| | accept predicate | halting | state on reject |
|---|---|---|---|
| `gate.ThresholdGate` | `delta_q.detach() > tau` (`:22`) | `active &= accepted` (`:80`) | `torch.where(accepted, candidate, previous)` (`:36`) |
| `self_audit_net.infer` | `delta_q.detach() > tau` (`:193, :204`) | `active = accepted` (`:250`) | `torch.where(accepted, candidate, state)` (`:210`) |
| `threshold.evaluate_threshold` | `quality[:,t] > tau` (`:80`) | `active = accepted` (`:90`) | no delta added (`:87`) |

Two structural notes rather than bugs:

1. **`gate.py` is dead code in production.** `grep` for `ThresholdGate|accept_reject|threshold_accept` outside `audit/` finds hits only in `tests/test_self_audit_audit.py:46` and `tests/test_self_audit_hardening.py:56`. `SelfAuditNet.infer` reimplements the same logic inline (`self_audit_net.py:191-250`). Two independent implementations of the accept rule that currently agree is a latent divergence risk; the tests exercise the copy that is not deployed.
2. Minor semantic difference: `ThresholdGate.run` calls `transition(state, turn)` on the **full** batch (`gate.py:66`) whereas `infer` `index_select`s only active rows (`self_audit_net.py:149-163`). Outputs match because halted rows are masked out of the state update either way, but compute and any stochastic transition function would differ.

---

## 10. Does anything in the auditor path leak gradient into the annotation branch?

**No. This is the cleanest part of the codebase.** Four independent barriers, verified:

1. **Inside the auditor** (`auditor.py:89-94`, the lines named in the brief):
   ```
   89  # This detachment is intentional and is part of the v1 gradient
   90  # contract: the auditor cannot teach the annotation network to collude.
   91  H_audit = H.detach()
   92  previous  = self._probabilities(P_previous.detach())
   93  candidate = self._probabilities(P_candidate.detach())
   94  delta = (candidate - previous) if delta_P is None else delta_P.detach()
   ```
   Note `delta` is derived from the already-detached `previous`/`candidate` in the default branch, so both branches are safe. Entropy inputs are detached at `:104` and `:109`. **Runtime-verified** (probeA): grads to `features`, `prev`, `cand` are all `None`.

2. **Phase B training loop** — belt, braces, and a second belt:
   - `freeze_annotation_network` (`train_auditor.py:52-64`) sets `requires_grad = name.startswith("auditor")` for every parameter and `.eval()`s encoder/FPN/initial head/annotation expert.
   - The annotation forward and feature extraction run under `torch.no_grad()` (`:164-166`), and `features` is additionally `.detach()`ed (`:166`).
   - Transition construction runs under `torch.no_grad()` (`:169-170`) and each state is `.detach()`ed (`:100-101`).
   - `CounterfactualGenerator.generate` detaches its inputs (`counterfactual.py:543-544`).
   - The optimizer is built from `requires_grad` parameters only (`train_auditor.py:492`).

3. **Phase C joint** (`finetune_joint.py:242-243`): `previous.detach()` / `candidate.detach()` before target construction and the auditor call; `infer` itself passes `shared_active.detach()` to the auditor (`self_audit_net.py:184-189`). So `audit_term` (`finetune_joint.py:247-257`) can only reach auditor weights.

4. **Auditor → annotation feedback is value-only.** `previous_audit = local_logits.detach().softmax(dim=1)` (`self_audit_net.py:216`), zeroed on reject (`:217-221`).

**But there is a non-gradient coupling worth flagging (P2, research risk).** In Phase C, `annotation_term = annotation_loss(final_logits, batch["mask"])` where `final_logits = output["logits"]` is the **gated final state** (`finetune_joint.py:209-214`). The gate is `torch.where(accepted[:,None,None,None], candidate, state)` (`self_audit_net.py:210`) with a detached boolean mask, so gradient flows only through whichever branch the auditor selected. Combined with §5b's prediction of a heavily reject-biased auditor:

- If the auditor rejects at turn 0, `final == initial`, and the **annotation expert receives zero gradient for that sample** — only the initial head is trained.
- Phase A supervises all four states with deep supervision `stage_weights: [0.5, 0.7, 0.8, 1.0]` (`configs/self_audit_annotation.yaml:29`); Phase C supervises **only the final state**.

So a conservative auditor can starve the annotation expert of training signal in Phase C — a routing-level feedback loop that no gradient audit will catch, and one that works directly against the "expert creates headroom" half of the central hypothesis. There is no test covering it.

**GT firewall.** `probe_gt_leakage` (`evaluation/audit_decomposition.py:392-451`) permutes/rolls GT and asserts `max_abs_logit_diff <= 1e-7` and `decision_mismatch_rate == 0.0` for self-audit inference. This is a genuine dynamic check and it is wired into the pipeline (`audit_decomposition.py:365`, printed as `GT_firewall=` at `train_self_audit.py:565`). `oracle_accept` mode requires an explicit `oracle_target` and raises otherwise (`self_audit_net.py:116-117`). **GT does not reach deployable inference. That claim holds.**

---

## Scorecard

| Component | Score | One-line justification |
|---|---|---|
| **Auditor** (`models/auditor.py`) | **PARTIAL** | Detach contract is correct and runtime-verified; sees shared features (not the image), and sees `A_t`, candidate, difference and both entropies. But the local head runs at **stride 4** while its target is per-pixel, making thin FIX/REGRESS bands unrepresentable (oracle recall 0.00–0.18, §1/§3); `delta_P` is redundant given the other two inputs. |
| **Counterfactual generator** (`audit/counterfactual.py`) | **RESEARCH RISK** | 9 distinct operations, clean simplex/locality discipline, and `P_previous` is genuinely load-bearing (§2). But at realistic model confidence **58–68 % of `positive` counterfactuals change zero pixels** and 91–95 % fall inside the loss's neutral band (§5b); the curriculum is ~77 % "reject"; `delta_P` fingerprints give **AUROC 0.82 with no image and no GT** (§2); sampling fractions and repair strengths are not configurable; `on_policy_transition` is dead code. |
| **Target builder** (`audit/targets.py`) | **MISMATCH** | Global `dQ` is exactly `dDice(cand,GT) − dDice(prev,GT)`, macro over foreground classes, per slice, unclipped — that part is right. The **local** target has **no epsilon at all**, and its class 1 conflates "both correct" with "both wrong", contradicting the stated `{FIX, NEUTRAL, REGRESS}` margin semantics. It is 99.6–99.9 % class-1, and the inverse-frequency weights are clamped at 5.0 against a ≥200:1 imbalance (§3). Three mutually inconsistent epsilons (0.0 metric / 0.005 loss / 0.02 generator) govern the same decision (§4). |
| **Gate** (`audit/gate.py`) | **PARTIAL** | Predicate is exactly `delta_q.detach() > tau_accept`, one-shot halting, no hysteresis; consistent across `gate.py`, `self_audit_net.infer`, and `threshold.evaluate_threshold`. **Local evidence never enters the accept decision**, contradicting the design spec's "local + global". `gate.py` is dead code — the deployed copy is the inline reimplementation in `infer`, and the tests exercise the wrong one. |
| **Threshold calibration** (`evaluation/threshold.py`, `scripts/calibrate_threshold.py`) | **RESEARCH RISK** | Mechanically sound and genuinely val-only, with a faithful gate simulator. But `final_macro_dice` is **maximized over 81 thresholds on val and then reported from that same val split** with no held-out re-evaluation, and `test_split` is never evaluated anywhere in the repo (§8). Worse, the calibrated `tau` is **never consumed** — all reported self-audit Dice uses the hard-coded `tau = 0.0`. |

## The three findings that most threaten the central hypothesis

1. **Phase-B validation is contaminated by the GT-driven generator** (§7). 57.1 % of the loss weight behind `val/auroc` — the metric that selects the exported checkpoint — comes from synthetic edits, and an AUROC of 0.82 is reachable from generator fingerprints alone. **No on-policy-only auditor metric exists anywhere in the repo.** Until one does, no reported auditor number supports the claim that the auditor captures on-policy correction headroom.
2. **The FIX curriculum is largely degenerate** (§5b, §3). Positive counterfactuals mostly do nothing at realistic confidence; those that do act land below the loss's own neutral margin and are simultaneously labelled negative by the AUROC. The auditor is trained to say "reject" ~77 % of the time and to output ≈0 for real repairs, against a strict `> 0` gate. This predicts `self_audit_dice ≈ initial_dice` — headroom left uncaptured — while AUROC still looks healthy.
3. **The local head cannot resolve its own target** (§1, §3). Stride-4 trunk, per-pixel target, 99.6 % single-class prior, weights clamped at 5.0. `local_fix_f1` is close to structurally unlearnable, and `f1_for_label` returns 1.0 for an absent class, so a head that never predicts FIX is not penalized in the degenerate case.

## What holds up

- The GT firewall is real, dynamic, and wired into the pipeline; GT does not influence deployable inference (§10).
- The gradient contract is airtight — four independent barriers, runtime-verified: no gradient flows from the auditor into the annotation branch in either Phase B or Phase C (§10).
- Patient-level split disjointness is enforced with an explicit leakage exception (`data/common.py:226`).
- `P_previous` is genuinely load-bearing; a candidate-only shortcut scores at chance (AUROC 0.525, §2).
- The threshold simulator's halting semantics exactly match deployed inference (§8/§9).

## Uncertainty and limits of this audit

- **No real data, no trained weights.** All quantitative probes use synthetic concentric-ring phantoms with a synthetically corrupted `previous`, run through the repo's production `CounterfactualGenerator` and `build_transition_targets`. The *mechanisms* (blend arithmetic, class priors, clamp ceilings, stride-4 bottleneck, transition counts) are structural and transfer; the *exact percentages* would shift on real ACDC with a real Phase-A checkpoint. §5b's confidence sweep is given across `p_max ∈ [0.475, 1.0]` precisely so the reader can locate their own model on it.
- The AUROC-0.82 shortcut result (§2) is a **lower bound** from six hand-crafted scalars on my synthetic mixture. I have not trained the actual auditor; the true shortcut magnitude on real data is unmeasured and could be higher or lower.
- Phase-A reported behaviour, the annotation expert's internals, `evaluation/audit_decomposition.py`'s attribution algebra, and the joint Phase-C training dynamics were traced only where they bear on the ten questions; they are other workers' scope.
- I did not run training. No file under `src/`, `scripts/`, `configs/`, or `tests/` was modified.
