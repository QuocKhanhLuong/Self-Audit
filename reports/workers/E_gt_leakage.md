# Worker E — GT-Leakage & Inference-Boundary Adversarial Audit

**Repo:** `/Users/alvinluong/Self-Audit` @ `a901eea` (branch `main`) — READ-ONLY audit, no repo file modified except this report.
**Environment:** `torch 2.13.0` importable. `matplotlib` and `pytest` are **NOT** installed in the active interpreter (`/Users/alvinluong/miniforge3/bin/python`), so `import self_audit.evaluation` fails at `src/self_audit/evaluation/visualizer.py:23` and the repo test suite could not be executed. All executable evidence below was gathered by importing `self_audit.models.self_audit_net` directly (which has no matplotlib dependency) and by loading `audit_decomposition.py` via `importlib.util.spec_from_file_location` to bypass the package `__init__`. Scratch scripts live outside the repo at
`/private/tmp/claude-501/-Users-alvinluong-Self-Audit/86453f80-e61c-4b1c-8019-1425966e3eeb/scratchpad/{gt_leak_probe.py,control_power.py}`.

---

## VERDICT (up front)

**IS THERE ACTUAL GT LEAKAGE IN DEPLOYABLE INFERENCE? — NO.**

In `mode="self_audit"` no ground-truth-derived value can reach A0, the candidate, auditor features, `delta_q`, the accept/reject decision, the `previous_audit_evidence` feedback, or `halt_turn`. This is proven three independent ways: (a) static tracing of every call site of every model entrypoint, (b) a GT-permutation invariance experiment that is **bitwise** zero-difference, and (c) a *positive control* on the same model/inputs showing the experiment has power (the oracle path diverges by 0.51 max-abs logit and flips every accept decision).

**However**, there are four real defects adjacent to the claim, listed by severity:

| # | Severity | Finding |
|---|----------|---------|
| E-1 | **High (methodology)** | No test-set evaluation exists anywhere in the repo. Every reported number is on `val`, and the headline calibration number is `argmax` over 81 τ on that same `val` data. See §5. |
| E-2 | **Medium (API design)** | `SelfAuditNet.infer` accepts `oracle_target` in *every* mode with zero validation; passing a string in `mode="self_audit"` raises nothing. One `if` edit away from real leakage. See §2/§4-TEST 3-4. |
| E-3 | **Medium (metric semantics)** | The stated neutral-margin semantics (`|dDice| <= eps` = NEUTRAL) is honoured in `audit_decomposition.py` but **violated** in `metrics.py:316` and `threshold.py:85`, where `dDice == 0` is counted as a *harmful acceptance*. See §8. |
| E-4 | **Low (latent leakage class)** | `VolumeSliceDataset(foreground_only=...)` at `src/self_audit/data/common.py:335` performs GT-based slice selection. It is currently dead (never enabled by any config or builder), but it is a live foot-gun. See §7. |

---

## 1. GT-token site classification

Grep over `src/` for `mask|target|ground_truth|gt|oracle|oracle_target|label|delta_dice|transition_target|counterfactual` returns 333 hits across 17 files. Only the sites where a GT-derived *value* actually flows are listed; pure parameter names, dataclass fields, and metric helpers that only ever receive GT after inference are collapsed into their owning function.

### 1a. Sites where GT enters a model call

There are exactly **three** call sites in the entire repo that pass anything GT-derived into a model, and all three name it `oracle_target`:

| path:line | Symbol | mode | Classification |
|---|---|---|---|
| `src/self_audit/evaluation/audit_decomposition.py:282-287` | `evaluate_audit_modes_batch` | `oracle_accept` | **DIAGNOSTIC-ONLY** (headroom ceiling) |
| `src/self_audit/evaluation/audit_decomposition.py:416-429` | `probe_gt_leakage` | `self_audit` (×2) | **DIAGNOSTIC-ONLY** — value provably inert (§3) |
| `src/self_audit/evaluation/volume_inference.py:294-300` | `evaluate_comparison_modes` | `oracle_accept` | **VALIDATION/ANALYSIS** |

Exhaustive list of *all* model entrypoint call sites (`grep -rn 'infer(\|forward_annotation(\|model(' src scripts`), each passing **only** `batch["image"]` unless noted:

| path:line | Call | Classification |
|---|---|---|
| `src/self_audit/training/finetune_joint.py:197-202` | `infer(mode="self_audit")` | TRAINING |
| `src/self_audit/training/finetune_joint.py:334-339` | `infer(mode="self_audit")` | VALIDATION |
| `src/self_audit/training/finetune_joint.py:510-515` | `infer(mode="always_accept_refinement", tau=-inf)` | CALIBRATION (cache build) |
| `src/self_audit/training/finetune_joint.py:898` | `infer(mode="self_audit")` | DIAGNOSTIC (visualiser) |
| `src/self_audit/training/train_annotation.py:156,235,418` | `forward_annotation` | TRAINING / VALIDATION |
| `src/self_audit/training/train_auditor.py:165,586` | `forward_annotation` | TRAINING / VALIDATION |
| `src/self_audit/evaluation/volume_inference.py:149` | `infer(mode=…)` | **DEPLOYABLE-INFERENCE** |
| `src/self_audit/evaluation/audit_decomposition.py:269,270-275,276-281` | `initial_only` / `always_accept` / `self_audit` | VALIDATION |
| `src/self_audit/evaluation/audit_decomposition.py:523` | `forward_annotation` | VALIDATION |
| `scripts/visualize_predictions.py:129` | `infer(mode="self_audit")` | DIAGNOSTIC |
| `scripts/self_audit_preflight.py:125`, `scripts/self_audit_memory_smoke.py:36` | `forward_annotation` | DIAGNOSTIC |

### 1b. GT-consuming code, by owner

| path:line | Symbol | GT use | Classification |
|---|---|---|---|
| `src/self_audit/audit/targets.py:55-70` | `local_audit_targets` | pixel-wise FIX/UNCHANGED/REGRESS vs GT | TRAINING (target construction) |
| `src/self_audit/audit/targets.py:73-87` | `multiclass_dice` | Dice vs GT | TRAINING + VALIDATION + DIAGNOSTIC |
| `src/self_audit/audit/targets.py:90-94` | `delta_dice_target` | `dDice` | TRAINING |
| `src/self_audit/audit/targets.py:97-107` | `build_transition_targets` | both of the above | TRAINING |
| `src/self_audit/audit/counterfactual.py` (whole module) | `CounterfactualGenerator` | GT-guided synthetic edits | **TRAINING-ONLY** — sole importers are `training/train_auditor.py:21` and `scripts/train_self_audit.py:26`; no evaluation or inference module imports it |
| `src/self_audit/losses/audit.py:67-96` | `audit_loss` | consumes `TransitionTargets` | TRAINING |
| `src/self_audit/losses/annotation.py` | `annotation_loss(logits, batch["mask"])` | TRAINING |
| `src/self_audit/training/finetune_joint.py:244-245` | `compute_joint_losses` | `build_transition_targets(prev.detach(), cand.detach(), batch["mask"])` — built **after** `infer` returned at :197 | TRAINING |
| `src/self_audit/training/finetune_joint.py:386-391` | `validate_phase_c` | targets built after `infer` at :334 | VALIDATION |
| `src/self_audit/training/finetune_joint.py:519,536` | `collect_validation_transition_cache` | GT deltas cached for τ sweep | **CALIBRATION** |
| `src/self_audit/evaluation/threshold.py:55-104` | `evaluate_threshold` | consumes cached `actual_delta_dice` | CALIBRATION |
| `src/self_audit/evaluation/audit_decomposition.py:169-171` | `decompose_self_audit_output` | Dice vs GT on stored transitions | DIAGNOSTIC-ONLY |
| `src/self_audit/evaluation/audit_decomposition.py:296-299` | `evaluate_audit_modes_batch` | Dice of 4 modes vs GT | DIAGNOSTIC-ONLY |
| `src/self_audit/evaluation/metrics.py:298-325` | `acceptance_metrics` | GT delta vs accept mask | VALIDATION |
| `src/self_audit/evaluation/volume_inference.py:274-281,316-319` | `evaluate_comparison_modes` | GT resized, scored | VALIDATION |
| `src/self_audit/evaluation/visualizer.py` | overlay rendering | DIAGNOSTIC-ONLY |
| `src/self_audit/data/common.py:377,381` | `VolumeSliceDataset.__getitem__` | `sample["mask"]` — a *separate dict key*, never concatenated into `sample["image"]` | data contract |
| `src/self_audit/data/common.py:335` | `VolumeSliceDataset.__init__` | `foreground_only` slice filter on GT | **latent DEPLOYABLE-INFERENCE risk** (§7) |
| `src/self_audit/models/self_audit_net.py:196-202, 340-357` | `infer` / `_candidate_improves` / `_dice` | GT Dice drives accept | **DIAGNOSTIC-ONLY**, gated by `mode == "oracle_accept"` |

**No other module in `src/` touches GT.** `models/encoder.py`, `models/fpn.py`, `models/annotation_head.py`, `models/annotation_expert.py`, `models/dynamic_window.py`, `models/auditor.py` contain **zero** occurrences of any GT token — verified by grep over `def forward|ground_truth|mask|target` across all six files, which returns only the `forward` signatures themselves plus `previous_audit_evidence` at `annotation_expert.py:98,112-117`.

---

## 2. `SelfAuditNet.infer` — the `oracle_target` boundary

`src/self_audit/models/self_audit_net.py:96-296`. Signature at **:96-105**:

```python
def infer(self, images, *, mode="self_audit", tau_accept=0.0,
          threshold=None, t_max=None, oracle_target=None) -> dict[str, Any]:
```

The only two statements in the whole 200-line body that read `oracle_target`:

- **:116-117** — `if mode == "oracle_accept" and oracle_target is None: raise ValueError(...)`
- **:196-202** — `elif mode == "oracle_accept": oracle_target_active = oracle_target.index_select(0, active_indices); accepted_active = self._candidate_improves(state_active, candidate_active, oracle_target_active)`

`_candidate_improves` at **:340-346** takes `argmax` of previous and candidate, computes `self._dice` (**:348-357**) against GT, and returns `candidate_score > previous_score`. The accept branch is a three-way `if/elif/else` at **:194-204**; `mode="self_audit"` lands in the `else` at **:203-204**, `accepted_active = delta_q > float(tau_accept)`, where `delta_q` (**:192-193**) comes from `self.auditor(...)` (**:183-190**).

### (a) Is the oracle path reachable in deployable `self_audit` mode?
**No.** The branch is a mutually-exclusive `elif` on a string literal. Reaching it requires the *caller* to pass `mode="oracle_accept"`. `src/self_audit/evaluation/volume_inference.py:179-180` explicitly hard-blocks that string in the deployable entrypoint:

```
if mode == "oracle_accept":
    raise ValueError("oracle_accept requires an explicit analysis wrapper; GT is not an inference input")
```

and `infer_patient_volume` has no `ground_truth`/`oracle_target` parameter at all (`:158-169`).

### (b) Can a caller pass `oracle_target` with `mode="self_audit"` and have it affect anything?
**Structurally, no — and executably, no.** There is no validation, so the call *succeeds silently*, but the value is never read. Proven two ways in §4: (i) bitwise-identical outputs for two different GT tensors; (ii) passing the Python string `"not-a-tensor"` as `oracle_target` with `mode="self_audit"` completes without raising — impossible if any code path had dereferenced it.

### (c) Is this an API / research-clarity risk even though it is not leakage?
**Yes, medium.** Three concrete reasons:

1. **No guard.** `mode="self_audit"` + a non-`None` `oracle_target` is accepted silently. The symmetric check exists for the *missing* case (:116-117) but not for the *spurious* case. A one-line `if mode != "oracle_accept" and oracle_target is not None: raise` would make the firewall structural instead of incidental.
2. **`forward` forwards everything.** `:359-360` — `def forward(self, images, **kwargs): return self.infer(images, **kwargs)`. Any `nn.Module` wrapper, DDP, `torch.compile` shim, or hook that calls `model(x, **batch_kwargs)` will happily route a `oracle_target` key straight in. The firewall's only defence is the mode string.
3. **The escape hatch sits inside the deployable class.** A reviewer reading `SelfAuditNet` sees GT Dice being computed in the same method that produces production predictions. The mode enum at `:108` advertises `oracle_accept` as a peer of `self_audit`.

**Recommendation: move the oracle path out of `SelfAuditNet.infer` into evaluation code.** Concretely: delete `oracle_target`, the `oracle_accept` mode, `_candidate_improves` and `_dice` from `self_audit_net.py`, and reimplement `oracle_accept` in `evaluation/` as a loop that calls the *public* per-turn surface. This is cheap because `infer(mode="always_accept_refinement")` already returns `transition_previous`, `transition_candidates` and `transition_active_masks` for every turn (`:271-272,282-283`) — everything an external oracle-gate simulator needs; `evaluation/threshold.py:69-90` already implements exactly this style of offline per-sample gate replay against a cached trajectory. After that change, `SelfAuditNet.infer` would have *no GT-typed parameter at all*, and the firewall becomes a type-system property rather than a runtime convention. Note the caveat: the offline replay is only exact for turn 0 unless the trajectory is re-expanded per accepted branch, because in `oracle_accept` the accepted state feeds the next turn's expert and the `previous_audit_evidence` (`:210,217-221`). The honest options are (i) accept a turn-0-exact / multi-turn-approximate oracle ceiling, or (ii) keep an oracle loop but put it in `evaluation/` calling `model.encode` / `model.annotation_expert` / `model.auditor` directly. Either removes GT from the deployable API.

---

## 3. Can any GT-derived value affect the seven decision points in `mode="self_audit"`? — call chains

Numbering follows `self_audit_net.py:118-250`.

**A0 (`initial`)** — `:118-120`: `encoded = self.encode(images)` → `self.encoder(images)` → `self.fpn(features)` (`:50-55`); then `initial = self.initial_head(shared, output_size=images.shape[-2:])`. Inputs: `images` and learned weights only. `encode` even *rejects* anything that is not `[B,3,H,W]` at `:51-52`, so a GT map could not be smuggled in as an extra channel. **No GT.**

**Candidate** — `:156-170`: `self.annotation_expert(shared_active, state_active, previous_audit_evidence=previous_audit_active, turn_index=turn, iteration_index=turn, return_metadata=False)`. `AnnotationExpert.forward` (`annotation_expert.py:93-117`) accepts `(shared, state, previous_audit_evidence, turn_index, iteration_index, return_metadata)` — no GT parameter exists. `state` is A0 at turn 0 and the retained state afterwards (`:210`). **No GT.**

**Auditor features** — `:183-190`: `self.auditor(shared_active.detach(), previous_probs.detach(), candidate_probs.detach(), (candidate_probs - previous_probs).detach(), entropy_previous=…, entropy_candidate=…)`. `CounterfactualAuditor.forward` (`auditor.py:76-116`) has signature `(H, P_previous, P_candidate, delta_P, entropy_previous, entropy_candidate)` — no GT parameter. Its concatenated input tensor at `auditor.py:111` is `[H_audit, previous, candidate, delta, entropy_previous, entropy_candidate]`; channel budget is fixed at `auditor.py:52` as `feature_channels + num_classes*3 + 2`, leaving no room for a GT plane. All six inputs derive from `images` + weights. **No GT.**

**`delta_q`** — `:192-193` reads it off the `AuditOutput` produced at `:183`; `delta_q` is `self.global_head(features)` (`auditor.py:115`). **No GT.**

**accept/reject** — `:203-204`, `accepted_active = delta_q > float(tau_accept)`. `tau_accept` is a Python float from the caller (`:101`/`:111-112`). The `oracle_accept` branch that *would* use GT is at `:196-202` and is unreachable here (§2a). **No GT** (see §5 for the separate question of where `tau_accept`'s *value* comes from).

**`previous_audit_evidence` feedback** — `:211-221`: `local_logits` comes from the auditor (`:191`), is scattered, softmaxed (`:216`), and gated by `accepted` (`:217-221`). `accepted` traces back to `delta_q` only. **No GT.**

**`halt_turn`** — `:238-244`: `rejected = attempted & ~accepted`; `halt_now = rejected & (halt_turn < 0)`. Depends only on `accepted` and `attempted`; `attempted = active.clone()` (`:144`) and `active = accepted` (`:249-250`). **No GT.**

---

## 4. Executable evidence

Model used throughout: `SelfAuditNet(pretrained_encoder=False, encoder_allow_fallback=True, shared_channels=16, num_classes=4, window_k=4, max_turns=3)`, `torch.manual_seed(1234)` for construction, `.eval()`, `torch.manual_seed(7)` then `images = torch.randn(4, 3, 64, 64)`.

### TEST 0 — determinism baseline (no GT anywhere)
Two identical `infer(mode="self_audit", tau_accept=0.0, t_max=3)` calls:

```
final_logits_maxdiff : 0.0
a0_maxdiff           : 0.0
halt_turn_equal      : true
```

Establishes that any non-zero diff in TEST 1 would be attributable to GT, not to nondeterminism.

### TEST 1 — GT-permutation invariance, `mode="self_audit"`
`g1 = randint(0,4,(4,64,64))`, `g2 = randint(0,4,(4,64,64))`, asserted `not torch.equal(g1, g2)`. Both runs `infer(images, mode="self_audit", tau_accept=0.0, t_max=3, oracle_target=g_i)`.

```
a0_logits_maxdiff            : 0.0        a0_bitwise_equal        : true
final_logits_maxdiff         : 0.0        final_bitwise_equal     : true
candidates_maxdiff           : [0.0, 0.0, 0.0]
transition_previous_maxdiff  : [0.0, 0.0, 0.0]
delta_q_maxdiff              : [0.0, 0.0, 0.0]
local_logits_maxdiff         : [0.0, 0.0, 0.0]
accepted_bitwise_equal       : [true, true, true]
active_mask_equal            : [true, true, true]
halt_turn_equal              : true       halt_turn_a = halt_turn_b = [-1,-1,-1,-1]
accepted_count_equal         : true
num_attempted_turns_equal    : true
n_turns_a = n_turns_b        : 3
```

All seven decision points from §3 are **bitwise identical**. Max-abs-diff is exactly `0.0`, not merely below a tolerance.

### TEST 1b — POSITIVE CONTROL (proves the probe has power)
A naive control with two random GT tensors is **degenerate**: on an untrained model both `oracle_accept` runs reject at turn 0 with `accepted = [false,false,false,false]` and `final_logits_maxdiff = 0.0`. That is an artefact of `_dice` returning `1.0` when `denominator == 0` (`self_audit_net.py:355`), so an all-background prediction scores 1.0 against every target. **A GT-permutation probe built this way would pass even on a leaking model.** This applies to the repo's own `probe_gt_leakage` when run early in training.

I therefore built a *powered* control (`control_power.py`): run `mode="always_accept_refinement"` once to obtain the real turn-0 previous/candidate label maps, then set `gt_a = candidate_labels` (candidate is perfect ⇒ must ACCEPT) and `gt_b = previous_labels` (previous is perfect ⇒ must REJECT), stamping non-degenerate foreground blocks of classes 1/2/3 into each so the `denominator == 0 → 1.0` shortcut cannot equalise them. Both label maps contained all four classes (`prev0 unique: [0,1,2,3]`, `cand0 unique: [0,1,2,3]`).

```
CONTROL  mode="oracle_accept"   gt_a vs gt_b
  final_logits_maxdiff : 0.5098123550415039     final_bitwise_equal : false
  halt_turn_a          : [1,1,1,1]              halt_turn_b         : [0,0,0,0]
  accepted_count_a     : [1,1,1,1]              accepted_count_b    : [0,0,0,0]
  accepted_turn0_a     : [true,true,true,true]  accepted_turn0_b    : [false,false,false,false]

TREATMENT mode="self_audit"     SAME gt_a vs gt_b
  final_logits_maxdiff : 0.0                    final_bitwise_equal : true
  a0_bitwise_equal     : true
  delta_q_maxdiff      : [0.0, 0.0, 0.0]
  candidates_maxdiff   : [0.0, 0.0, 0.0]
  accepted_equal       : [true, true, true]
  halt_turn_equal      : true                   accepted_count_equal: true
```

The identical GT pair that flips **every** accept decision and moves the final logits by 0.51 in `oracle_accept` produces **exactly zero** change in `self_audit`. This is the strongest available evidence: the probe demonstrably detects GT influence, and detects none.

### TEST 2 — Phase-A/B annotation invariance
Two `forward_annotation(images, turns=3)` calls with `g1`/`g2` present and touched (`g_i.float().mean()`) in the surrounding scope:

```
a0_maxdiff            : 0.0
state_trace_maxdiff   : [0.0, 0.0, 0.0, 0.0]     (A0, A1, A2, A3)
state_trace_len       : 4
all_bitwise_equal     : true
```

`forward_annotation` (`self_audit_net.py:57-94`) has no GT parameter at all, so this is expected; the test rules out any hidden global/module-level GT channel.

### TEST 3 — API boundary
```python
model.infer(images, mode="self_audit", oracle_target=g1, t_max=1)
# -> self_audit_accepts_oracle_target_without_error: true   (error: null)
```
**Risk classification: MEDIUM — API/research-clarity defect, not leakage.** The parameter is accepted in a mode that cannot use it, with no warning and no error. See §2c for the recommendation.

### TEST 4 — no validation of `oracle_target` in `self_audit`
```python
model.infer(images, mode="self_audit", oracle_target="not-a-tensor", t_max=1)
# -> completes, error: null
```
Passing a Python `str` where a `Tensor` is annotated raises nothing. This is simultaneously (i) the *strongest positive proof* that nothing dereferences the value in `self_audit` mode — a string has no `.index_select`, `.detach` or `.argmax`, so any read would have raised `AttributeError` — and (ii) confirmation that the parameter is completely unvalidated.

### TEST 5 — the repo's own firewall probe
`probe_gt_leakage` (`audit_decomposition.py:391-452`), loaded via `importlib` to skip the matplotlib-importing package `__init__`:
```
gt_firewall/max_abs_logit_diff     : 0.0
gt_firewall/decision_mismatch_rate : 0.0
gt_firewall/passed                 : 1.0
```
Agrees with my independent TEST 1. **Caveat, and it matters:** this probe reuses the same degenerate-control weakness identified in TEST 1b. It permutes GT across the batch (`:410-412`) rather than constructing GTs that provably change the oracle decision, so on an untrained or all-background model it can pass vacuously. It is also only invoked on `batch_index == 0` of the decomposition loop (`audit_decomposition.py:363-372`), and there is **no test in `tests/` that asserts GT-permutation invariance** — grep for `gt_firewall|leak|permut` over `tests/` returns only `test_self_audit_data.py:97-98` (patient split leakage) and `test_self_audit_masks.py:117-130` (which asserts `oracle_accept` *rejects*, i.e. it exercises the degenerate case). The firewall is a runtime metric, not a CI gate.

---

## 5. Threshold calibration leakage

**Finding E-1 (High, methodology): the τ-selection set and the reporting set are the same `val` split, and there is no test-set evaluation anywhere in the repo.**

Evidence chain in `scripts/train_self_audit.py`:

- **:149** — the Phase-C loaders are built from `split=str(config.get("val_split", "val"))`.
- **:471-473** — `tau_accept = args.tau_accept if not None else audit_cfg.get("tau_accept", 0.0)`; all four configs set `audit.tau_accept: 0.0` (`configs/self_audit_joint.yaml:29-31`, `configs/self_audit_acdc_to_mnms.yaml:30-32`).
- **:508-528** — every epoch, `validate_phase_c(...)` and `evaluate_audit_decomposition(...)` run on `val_c` at that fixed τ. These produce `modes/self_audit_dice`, `modes/oracle_headroom`, `audit_rescue_vs_always` etc.
- **:535** — checkpoint selection: `metric = float(val_stats["final_foreground_macro_dice"])`, i.e. best-on-`val`.
- **:570-584** — *after* training, `collect_validation_transition_cache(model, val_c, …)` then `sweep_thresholds(cache, np.linspace(-0.02, 0.02, 81))`.
- **:585** — `best_threshold = select_threshold(rows)`, which is `max(rows, key=final_macro_dice, …)` (`src/self_audit/evaluation/threshold.py:142-148`).
- **:594-598** — printed as `tau=… final=… net_gain=…` and written to `calibration.json` (**:593**).

So `calibration.json → best.final_macro_dice` is the **maximum over 81 thresholds of a Dice computed on the very transitions used to pick the threshold**. It is an optimistically biased estimate, and it is the most headline-looking number the pipeline emits.

**Confirmation that no test split is ever evaluated.** `grep -rn 'test_split|split="test"' scripts/ src/` returns only:
- `src/self_audit/training/_utils.py:369-372,379,406` — `validate_dataset_splits` reads `test_split` **solely to check patient disjointness** and to set `"test_available"`.
- `src/self_audit/data/{common,acdc,mnms}.py` — alias tables and `patient_level_split` construction.
- `scripts/visualize_predictions.py:50` — `--split` accepts `"test"` for *figure* rendering only.

No entrypoint (`scripts/train_self_audit.py`, `scripts/audit_checkpoint.py`, `scripts/cache_validation_transitions.py`, `src/self_audit/training/finetune_joint.py`, `src/self_audit/training/train_{annotation,auditor}.py`) ever calls `build_patient_dataset(..., split="test")`. `configs/self_audit_acdc_to_mnms.yaml:11-20` declares an M&Ms test protocol that no script implements.

**Two mitigating facts, stated honestly:**
1. The calibrated τ is **never written back** to any config and never applied to a subsequent evaluation. `grep -rn 'calibration.json|best_threshold'` shows it is written at `train_self_audit.py:593` and read by nothing. Every deployable path still uses the config's `tau_accept: 0.0`.
2. Consequently the per-epoch `modes/*` numbers are *not* τ-selected — they are ordinary val numbers at a fixed τ. The selection bias is confined to `report["calibration"]["best"]`.

**The residual risk in `scripts/audit_checkpoint.py`.** It builds `val_dataset` at `:86-90` from `config.get("val_split", "val")` and takes `--tau_accept` from the CLI (`:45`, `:100-102`). Feeding it the τ from `calibration.json` and pointing it at `val` closes the loop into a fully same-data selected report, with nothing in the code to stop or flag it. It has no `--split` argument, so it *cannot* be pointed at `test` without a config edit.

**Recommendation:** split calibration off `val`, or hold out a `test` split and report on it; and either persist the selected τ into the config so the numbers and the τ are traceable together, or drop `best.final_macro_dice` from the report and keep only `best.tau_accept`.

---

## 6. Volume-level inference and `scripts/audit_checkpoint.py`

### `src/self_audit/evaluation/volume_inference.py`
`infer_patient_volume` (`:158-233`) is the deployable path and is **clean**:
- Signature `:158-169` has **no** GT/`oracle_target`/`ground_truth` parameter.
- `:179-180` rejects `mode="oracle_accept"` outright.
- Input construction is `normalize_volume(canonicalize_depth_first(volume))` (`:183`) then `build_25d_batch` (`:184`, defined `:66-81`) — the 3 channels are slices `z-1, z, z+1` of the **image volume** (`:77`). No GT tensor exists in the function's scope.
- `_model_predict` (`:138-155`) calls `model.infer(batch, mode=mode, tau_accept=tau_accept, t_max=t_max)` at `:149` — no `oracle_target` kwarg.

`evaluate_comparison_modes` (`:236-320`) is the GT-holding analysis wrapper. It runs the three deployable modes through `infer_patient_volume` first (`:259-271`), only then loads/resizes GT (`:273-281`) and runs the oracle loop (`:290-305`). Metrics are computed at `:316-319`. GT never re-enters the three deployable runs — they were already completed and stored. **Correct isolation.**

**Low-severity defect:** `:283-285` annotate locals as `list[Tensor]`, but `Tensor` is never imported in this module (imports at `:8-12` are `numpy`, `torch`, `torch.nn.functional as F`, `..data.common.to_depth_first`). This is inert at runtime because PEP 526 local annotations are never evaluated, but it is a genuine `NameError` waiting for anyone who hoists these to module scope, and it breaks static type-checking of this file. Not a leakage issue.

### `scripts/audit_checkpoint.py`
No GT touches any decision path. It loads a checkpoint (`:82-84`), builds the val loader (`:86-97`), and calls `evaluate_annotation_headroom` (`:114-121`) and `evaluate_audit_decomposition` (`:122-131`). Both are `@torch.no_grad()` (`audit_decomposition.py:332,502`) and pass GT only as a *scoring* argument after `infer` returns. The `oracle_accept` run inside `evaluate_audit_modes_batch` (`:282-287`) produces `modes/oracle_dice` / `modes/oracle_headroom`, printed at `audit_checkpoint.py:155,159` — clearly labelled ceiling metrics, not deployable predictions. The GT firewall result is surfaced at `:161`. The only concern is the calibration-loop risk described at the end of §5.

---

## 7. Dataset / collate path

**Is GT concatenated into the model input tensor?** **No.**
`VolumeSliceDataset.__getitem__` (`src/self_audit/data/common.py:373-399`) builds `image` from `build_25d_triplet(volume, slice_index)` at `:376` — `volume` is the normalised **image** array returned by `_load` (`:348-367`, normalised at `:361`). The mask becomes a *separate* key: `sample = {"image": image, "mask": target, …}` at `:379-392`. `resize_sample` (`:203-213`) resizes them jointly with `bilinear`/`nearest` but returns them as a 2-tuple; nothing stacks them. The model's own guard at `self_audit_net.py:51-52` (`shape[1] != 3` → `ValueError`) would reject a 4-channel input anyway.

**Is there a custom collate?** No. `build_data_loader` (`src/self_audit/training/_utils.py:291-316`) passes only `batch_size`, `shuffle`, `num_workers`, `pin_memory`, and optionally `persistent_workers`/`prefetch_factor` — **no `collate_fn`**, so PyTorch's default collate stacks each dict key independently. `move_batch` (`_utils.py:935-939`) moves tensors device-side and preserves keys. Every training/eval call site then indexes `batch["image"]` for the model and `batch["mask"]` only for losses/metrics.

**Is GT used for cropping or foreground selection at eval time?** **Not currently — but the capability exists and is one config key from being enabled.**

`src/self_audit/data/common.py:335`:
```python
if self.foreground_only and not np.any(mask[slice_index] > 0):
    continue
```
This drops every slice whose **ground truth** has no foreground, at dataset-construction time, for train *and* eval alike. If enabled at eval it is a textbook leakage class: it removes exactly the slices where a false-positive prediction would be maximally penalised, and it uses test-time GT to decide what gets scored.

**Evidence that it is dead code today** — `grep -rn 'foreground_only' src scripts tests configs docs` returns only:
- `src/self_audit/data/common.py:304,317,335` (the definition and use, default `False` at `:304`)
- `src/self_audit/data/acdc.py:215,235` and `src/self_audit/data/mnms.py:161,182` (pass-through, default `False`)

**Zero** hits in `configs/`, `scripts/`, `src/self_audit/training/`, or `docs/`. Decisively, `build_patient_dataset` (`_utils.py:270-288`) constructs `ACDCDataset` with a fixed kwarg set — `data_root`, `split`, `image_size`, `augment` (`:279-284`) — plus an allowlist loop over exactly `("split_manifest", "seed", "depth_axis", "expected_slices", "max_cache")` at `:285-287`. `foreground_only` is **not** in that allowlist, so it cannot be turned on from YAML even by accident. That containment is real but incidental; it holds only because the allowlist happens to omit the key.

No cropping of any kind is GT-driven: the only spatial ops are `resize_in_plane` / `resize_sample` (`common.py:163-213`) and the geometric augmentations in `data/transforms.py`, which apply the *same* op to image and mask via `_transform_pair` (`transforms.py:11-23`) and are gated to `train=True` by `augment=bool(train and config.get("augment", True))` (`_utils.py:283`).

---

## 8. Neutral-margin semantics — a real inconsistency (Finding E-3)

The stated contract: `dDice > +eps` = IMPROVE, `|dDice| <= eps` = NEUTRAL, `dDice < -eps` = REGRESS.

**Honoured** in `src/self_audit/evaluation/audit_decomposition.py:185-187` and `:481-483`:
```python
beneficial = delta > float(neutral_margin)
harmful    = delta < -float(neutral_margin)
neutral    = ~(beneficial | harmful)
```
and in the training loss, `src/self_audit/losses/audit.py:20-21,33-39`, where `neutral = actual.abs() < neutral_margin` are excluded from the ranking pairs and regressed toward zero.

**Violated** in the two places that compute the gate-quality rates:
- `src/self_audit/evaluation/metrics.py:316-317` — `harmful = accepted & (delta <= 0.0)`; `beneficial_rejection = ~accepted & (delta > 0.0)`
- `src/self_audit/evaluation/threshold.py:85-86` — identical `<= 0.0` / `> 0.0` form

Both use a **zero** margin, and asymmetrically: a transition with `dDice == 0.0` is counted as a **harmful acceptance** if accepted, but is *not* counted as a beneficial rejection if rejected. Exact-zero deltas are the common case — most refinement steps leave the argmax label map unchanged — so `harmful_acceptance_rate` is systematically inflated and is not measuring harm. Impact: it is reported at `finetune_joint.py:468,876` and `threshold.py:99`; it is the **tie-break key** in `select_threshold` (`threshold.py:146`); and it is a **hard filter** when `calibrate_threshold.py --max_harmful_acceptance_rate` is used (`threshold.py:129-132`), where it can eliminate genuinely good thresholds. `scripts/train_self_audit.py:584` calls `sweep_thresholds` without that argument, so today the effect is limited to the tie-break and the reported rate.

Separately, `audit_decomposition.py:496-498` hardcodes `0.005` in `headroom_collapse_ratio` instead of using the `neutral_margin` parameter that the same function accepts at `:459` — so passing a different margin silently desynchronises that one metric. Diagnostic-only.

Also worth flagging for the "neutral margin" reader: `local_audit_targets` (`audit/targets.py:55-70`) uses **exact pixel correctness**, not a margin — `FIX` iff `~previous_correct & candidate_correct`, `REGRESS` iff `previous_correct & ~candidate_correct`, `UNCHANGED` otherwise. That is correct and intentional (the margin is a *global* dDice concept, not a per-pixel one), but it means "NEUTRAL" denotes two different things in the local and global heads.

---

## 9. Final verdict

### **NO — there is no actual GT leakage in deployable inference.**

Basis:
1. `mode="self_audit"` never reads `oracle_target`. Proven by the mutually-exclusive `elif` at `self_audit_net.py:196-204`, and executably by TEST 4: a Python `str` passed as `oracle_target` in `self_audit` mode raises nothing, which is only possible if the value is never dereferenced.
2. All seven decision points are **bitwise identical** under two different GT tensors (TEST 1: every max-abs-diff exactly `0.0`).
3. The probe has demonstrated power: the **same GT pair** flips every accept decision and moves final logits by 0.5098 under `oracle_accept` (TEST 1b control).
4. No GT-typed parameter exists on `encode`, `forward_annotation`, `AnnotationExpert.forward`, `CounterfactualAuditor.forward`, `InitialAnnotationHead.forward`, the encoder, the FPN, or the dynamic window.
5. The deployable volume entrypoint has no GT parameter and hard-rejects `oracle_accept` (`volume_inference.py:179-180`).
6. GT is never concatenated into the model input and is never used for eval-time cropping or slice selection (`foreground_only` is dead code, unreachable from YAML).
7. `CounterfactualGenerator` is imported only by `training/train_auditor.py:21` and `scripts/train_self_audit.py:26` — never by an evaluation or inference module.

### Strongest counter-argument to my own verdict

**The firewall is a convention, not a structure, and my proof is checkpoint-specific.**

Three concrete ways the verdict could be wrong or could stop being true:

1. **My evidence is one randomly-initialised model on one 4×3×64×64 batch.** GT-dependence that only manifests through a trained auditor's numerics — or an accept pattern that only diverges once `delta_q` sits near τ — would not show up here. I mitigated this by testing the *mechanism* (TEST 4 proves the value is never read at all, which is checkpoint-independent) rather than only the *outcome*, but the outcome tests themselves generalise only as far as the static trace does. I could not run the repo's own suite (no `pytest` installed) to check for a case I missed.

2. **The API invites the leak that does not yet exist.** `infer` accepts `oracle_target` in every mode with no validation, `forward` at `:359-360` splats arbitrary `**kwargs` into it, and the accept logic is a three-way branch on a string. Changing `elif mode == "oracle_accept"` to `if oracle_target is not None` — a plausible refactor, and one that would read as a *simplification* — converts a clean firewall into silent leakage with no test to catch it, because no test in `tests/` asserts GT-permutation invariance. The only guard is `probe_gt_leakage`, which is a logged metric on batch 0, not an assertion, and which (as shown in TEST 1b) can itself pass vacuously on a model whose predictions are degenerate.

3. **"No leakage into inference" is not the same as "the reported numbers are clean."** The verdict answers the narrow question. It does **not** clear the headline calibration number, which is a max over 81 τ on the same `val` transitions used to compute it (§5), on a pipeline that never evaluates a test split at all. A reader who takes `IS THERE GT LEAKAGE? NO` as validation of the reported Dice numbers is drawing a conclusion my evidence does not support.
