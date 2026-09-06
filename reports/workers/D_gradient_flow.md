# Worker D — Phase A/B/C Training + Gradient-Flow Audit

**Repo:** `/Users/alvinluong/Self-Audit` @ `a901eea` (branch `main`). Read-only audit; no files under `src/`, `scripts/`, `configs/`, `tests/` were modified.

**Executable evidence:** torch **2.13.0**, Python 3.11.16 — importable. All gradient numbers below are from real `backward()` calls on a real `SelfAuditNet`. Probe scripts live in the session scratchpad (`grad_probe.py`, `fw.py`, `dead.py`, `headroom.py`, `split.py`); they only import from `src/` and write nothing into the repo.

**Environment caveats that bound my claims:**
- `timm` is **not installed** on this machine. Every model I instantiated therefore used `_FallbackHierarchicalEncoder` (`src/self_audit/models/encoder.py:29-70`), reached via `encoder.py:103-111`. Autograd *topology* is identical (a plain conv stack, 52 leaf params), so all connectivity conclusions transfer. Claims that depend on ConvNeXt-specific layers (norm type, DropPath) are marked as **static-only**.
- `matplotlib` is not installed either; `self_audit.evaluation.__init__` imports `visualizer` which imports it. I injected a stub *on `PYTHONPATH` only*, never into the repo.
- I used a reduced model (`shared_channels=16`, `window_k=4`, `64x64` input) so runs finish in seconds. Gradient *magnitudes* are therefore indicative, not production values; gradient *presence/absence* (the firewall question) is exact and configuration-independent because it is decided by `detach()` placement.
- No training was run.

---

## 0. Executive summary

| # | Finding | Severity |
|---|---|---|
| F1 | Phase A supervises A0..A3 all against GT; the deep chain `L(A3) -> AnnExpert -> A2 -> A1 -> A0 -> initial_head -> FPN -> encoder` is **ACTIVE** (confirmed executably). | design-as-intended |
| F2 | **Headroom risk is real and quantified.** The A0 head receives only **7.8%** of its gradient from its own supervision term; **92.2%** comes from A1/A2/A3 losses backpropagating through the expert. Nothing in Phase A or Phase C selects for A0 quality. | **P1 (scientific)** |
| F3 | Gradient firewall **holds in both directions**, in Phase B and Phase C. Audit-only backward gives exactly `None` grads on all 120 annotation params; annotation-only backward gives exactly `None` on all 28 auditor params. No leak. | verified OK |
| F4 | In Phase B the local (pixel CE) term produces a **34x larger** gradient than the global ΔQ term at equal weight (1.0/1.0). ΔQ is the head that actually gates ACCEPT/REJECT. | **P2** |
| F5 | `LightweightFPN.forward` computes `self.refine[0..3]` but only uses `refined[0]` — **12 parameters are permanently dead** (`grad is None` in every phase) and 3 conv+GN blocks of compute are wasted every forward. | **P2** |
| F6 | No phase-tag validation on checkpoint load. Passing a Phase-A checkpoint to Phase C silently runs `mode="self_audit"` with a **random-init auditor** at `tau=0.0`. | **P2** |
| F7 | `amp_dtype: float16` + CUDA turns normal `GradScaler` warmup into a hard `FloatingPointError` crash. The checked-in configs use `bfloat16`, so the scaler is disabled everywhere and this is latent. | **P2 (latent)** |
| F8 | Phase C has **no auditor param group**. The auditor's LR silently drops 3e-4 -> 1e-5 (30x) and is bundled with the annotation heads. | **P2** |
| F9 | `resolve_stage_weights` **silently discards** a configured `stage_weights` list of the wrong length and substitutes defaults. | P3 |
| F10 | Phase C `--output` is parsed but never used; the config value always wins. Phase B recomputes the encoder a second time per batch. Neutral-margin boundary is `<` not `<=`, and two different epsilons (`0.02` / `0.005`) are both called "neutral". | P3 |

---

## 1. Phase A — every loss term, weight, and what it supervises

### 1.1 The loss

`phase_a_loss` — `src/self_audit/training/train_annotation.py:82-103`.

```
 91  initial = extract_initial_logits(output)          # A0
 92  states  = extract_annotation_states(output)       # [A1, A2, A3]  (refinement-only)
 93  logits  = [initial] + [s for s in states if s is not initial]   # [A0,A1,A2,A3]
 97  weights = resolve_stage_weights(len(logits), stage_weights)
 98  losses  = [annotation_loss(v, target) for v in logits]
 99  total   = sum(w*l for ...) / max(sum(weights), 1e-8)
```

`extract_annotation_states` (`_utils.py:419-446`) reads `output["states"]`, which `forward_annotation` sets to the **refinement-only** list (`self_audit_net.py:89`, comment at `:85-86`). So `logits` is exactly `[A0,A1,A2,A3]` with no duplication. Verified: `state_trace len: 4, states len: 3`, `A0 is state_trace[0]: True`.

Per-stage leaf loss = `annotation_loss` (`src/self_audit/losses/annotation.py:27-38`):

| Term | Weight (code default) | Source line |
|---|---|---|
| soft Dice, `include_background=True` | `dice_weight = 1.0` | `annotation.py:31,35,37` |
| cross-entropy | `cross_entropy_weight = 1.0` | `annotation.py:32,36,37` |

Neither weight is exposed in any config or plumbed from `phase_a_loss` — they are hard-coded 1.0/1.0. `include_background=True` means the background class is included in the Dice mean (`annotation.py:16,33,35`), which dilutes foreground Dice by ~4x for a 4-class problem.

### 1.2 Stage weights

`resolve_stage_weights` — `train_annotation.py:54-79`. `DEFAULT_STAGE_WEIGHTS = (0.5, 0.7, 0.8, 1.0)` at `:51`.

`configs/self_audit_annotation.yaml:29` — `stage_weights: [0.5, 0.7, 0.8, 1.0]`, i.e. **identical to the code default**. Read at `train_annotation.py:321`, forwarded at `:333`.

Because of the `/ sum(weights)` normalization at `:99`, the **effective** weights are:

| Stage | Raw w | Effective w/Σw (Σw = 3.0) | Supervises |
|---|---|---|---|
| A0 (`initial_logits`, `initial_head` output) | 0.5 | **0.1667** | `self_audit_net.py:62` |
| A1 (expert turn 0) | 0.7 | **0.2333** | `self_audit_net.py:68-77`, turn=0 |
| A2 (expert turn 1) | 0.8 | **0.2667** | turn=1 |
| A3 (expert turn 2, `output["logits"]`) | 1.0 | **0.3333** | turn=2, `self_audit_net.py:83` |

Executable confirmation (`phase_a_loss(o, y, stage_weights=[0.5,0.7,0.8,1.0])`):
```
parts: {'state_0': 2.160079, 'state_1': 2.182676, 'state_2': 2.219557,
        'state_3': 2.27162,  'loss': 2.218393, 'weight_sum': 3.0}
```

There is **no ramp, no decay, no epoch-dependent schedule**. `stage_weights` is a static list read once at `:321` and passed unchanged every epoch. The only ramp anywhere is the LR warmup (`_utils.py:563-569`, `warmup_epochs: 5` in `configs/self_audit_annotation.yaml:20`).

### 1.3 F9 — silent config drop

`train_annotation.py:63-70`:
```
if configured is not None:
    values = [float(v) for v in configured]
    if len(values) == count:  weights = values
    else:                     weights = list(DEFAULT_STAGE_WEIGHTS[:count]) ...
```
A length mismatch is **silently ignored** — no warning, no raise. If `model.max_turns` is changed to 4 or 5 without updating `stage_weights`, the configured list is thrown away and the defaults are used. **P3**, but it is exactly the kind of silent substitution that makes an ablation table wrong.

### 1.4 Model selection does not see A0

`validate_annotation_epoch` computes Dice from `output["logits"]` = **A3 only** (`train_annotation.py:239-242`), and `best.pt` / the Phase-A output target are gated on `val_macro_foreground_dice` (`:354, :365-386`). Nothing in Phase A ever selects a checkpoint on A0 quality. Relevant to §10.

---

## 2. Are A0, A1, A2, A3 ALL supervised directly against GT in Phase A?

**YES — all four, unconditionally, every step.** `train_annotation.py:93` builds the list, `:98` applies `annotation_loss(value, target)` to each element with `target = batch["mask"]` (`:157`).

The loop is the list comprehension at `:98`, not a Python `for`. Per-stage isolated backward (each stage's loss alone, all others discarded):

```
[A0 loss only] encoder=2.155818e-01 fpn=2.127618e-01 initial_head=3.098299e-01 annotation_expert=0.000000e+00
[A1 loss only] encoder=2.152244e-01 fpn=2.587853e-01 initial_head=5.081754e-01 annotation_expert=1.704537e-01
[A2 loss only] encoder=2.502753e-01 fpn=3.445463e-01 initial_head=6.831333e-01 annotation_expert=5.538737e-01
[A3 loss only] encoder=3.872929e-01 fpn=5.134525e-01 initial_head=8.244300e-01 annotation_expert=1.242964e+00
```

Two facts fall straight out:
1. `annotation_expert = 0.0` for the A0 term — correct, A0 is produced before the expert runs.
2. `initial_head` grad is **non-zero and monotonically increasing** for A1, A2, A3. Later-stage losses do reach the A0 head. This is §3 and §10.

---

## 3. Is `L(A3) -> AnnExpert -> A2 -> A1 -> A0 -> initial_head -> FPN -> encoder` active?

### 3.1 Static proof

`SelfAuditNet.forward_annotation` — `src/self_audit/models/self_audit_net.py:57-94`:

```
62      initial = self.initial_head(shared, output_size=images.shape[-2:])
63      state = initial
67      for turn in range(...):
68          expert_output = self.annotation_expert(
69              shared,
70              state,                       # <-- NOT detached
                ...
76          state = expert_output.candidate_logits
77          states.append(state)
```

Line 70 passes `state` **un-detached** into the next expert call, and line 76 rebinds `state` to the new output. There is no `detach()`, no `no_grad()`, and no `.data` access anywhere in `forward_annotation`. `AnnotationExpert` also does not detach its state input (`src/self_audit/models/annotation_expert.py:18` calls `.detach()` only inside a range *check*, on scalars, discarding the result). **CONFIRMED, not refuted.**

### 3.2 Executable proof

Built `SelfAuditNet(pretrained_encoder=False, shared_channels=16, num_classes=4, window_k=4, max_turns=3)`, ran `forward_annotation`, backpropagated **only** `annotation_loss(out["state_trace"][-1], y)` — the last-stage loss alone:

```
A0 requires_grad: True   grad_fn of A3: AddBackward0

[T3 last-stage-only backward]
  encoder           = 3.872929e-01  (52 params with grad)
  fpn               = 5.134525e-01  (16 params with grad)
  initial_head      = 8.244300e-01  ( 6 params with grad)
  annotation_expert = 1.242964e+00  (34 params with grad)
  auditor           = 0.000000e+00  ( 0 params with grad)
```

The full chain is **ACTIVE**. `initial_head`, `fpn` and `encoder` all receive non-zero gradient from a loss applied only to A3.

### 3.3 F5 — 12 FPN params are permanently dead

`fpn` shows **16** params with grad but the module has **28**. Enumerating `p.grad is None` after the same backward:

```
fpn.refine.1.0.weight  fpn.refine.1.0.bias  fpn.refine.1.1.weight  fpn.refine.1.1.bias
fpn.refine.2.0.weight  fpn.refine.2.0.bias  fpn.refine.2.1.weight  fpn.refine.2.1.bias
fpn.refine.3.0.weight  fpn.refine.3.0.bias  fpn.refine.3.1.weight  fpn.refine.3.1.bias
```

Root cause — `src/self_audit/models/fpn.py:53-54`:
```
53      refined = [block(feature) for block, feature in zip(self.refine, pyramid)]
54      return self.output(refined[0])
```
All four refine blocks are **executed**; only `refined[0]` is consumed. `self.refine[1..3]` are 3 x (Conv2d(96,96,3) + GroupNorm) = 12 leaf tensors that (a) never receive gradient in any phase and (b) burn forward FLOPs on every single batch of Phase A, B and C. They are also serialized into every checkpoint at their random-init values.

Not a correctness bug for the loss, but it is dead weight in the parameter-count table printed by `print_model_parameter_summary` (`_utils.py:246`) and a real throughput cost. **P2.**

---

## 4. Gradient connectivity table

`Y` = at least one leaf param in that module received a non-`None`, non-zero gradient. All rows measured by real isolated `backward()` calls unless marked. `n=` is the count of leaf tensors that got a grad.

| Loss | Encoder | FPN | A0 head | AnnExpert | Auditor |
|---|---|---|---|---|---|
| **PhaseA A0 loss** | **Y** 2.16e-1 (n=52) | **Y** 2.13e-1 (n=16/28) | **Y** 3.10e-1 (n=6) | **N** 0.0 (n=0) | **N** 0.0 (n=0) |
| **PhaseA A1 loss** | **Y** 2.15e-1 | **Y** 2.59e-1 (n=16/28) | **Y** 5.08e-1 | **Y** 1.70e-1 (n=34) | **N** 0.0 |
| **PhaseA A2 loss** | **Y** 2.50e-1 | **Y** 3.45e-1 (n=16/28) | **Y** 6.83e-1 | **Y** 5.54e-1 | **N** 0.0 |
| **PhaseA A3 loss** | **Y** 3.87e-1 | **Y** 5.13e-1 (n=16/28) | **Y** 8.24e-1 | **Y** 1.24e+0 | **N** 0.0 |
| **PhaseB audit local** | **N** grad=None x52 | **N** None x28 | **N** None x6 | **N** None x34 | **Y** 1.25e+1 (n=28) |
| **PhaseB audit dQ** | **N** None | **N** None | **N** None | **N** None | **Y** 3.64e-1 (n=28) |
| **PhaseC annotation** | **Y** 4.93e-1 (n=52) | **Y** 5.09e-1 (n=16/28) | **Y** 8.09e-1 (n=6) | **Y** 1.13e+0 (n=34) † | **N** grad=None x28 |
| **PhaseC audit** | **N** None x52 | **N** None x28 | **N** None x6 | **N** None x34 | **Y** 4.74e+0 (n=28) |

† **Conditional.** With `tau_accept = -1e9` (accept everything) the expert gets `1.13e+0`. With `tau_accept = +1e9` (reject everything) `compute_joint_losses` reports `transition_count: 1` and the expert grad is **exactly 0.0** — because `infer` sets `state = torch.where(accepted, candidate, state)` (`self_audit_net.py:210`) and with nothing accepted the final state *is* A0, so the graph never touches the expert. `initial_head` still gets `2.58e-1`. Evidence line: `[C reject-all ANNOT-only] ... initial_head=2.576138e-01 | annotation_expert=0.000000e+00`.

Evidence anchors:
- Rows PhaseA*: `train_annotation.py:93,98`; `self_audit_net.py:62,68-77`.
- Rows PhaseB*: `train_auditor.py:56` (freeze) + `auditor.py:91-94` (detach). Split measured separately per term.
- Row PhaseC annotation: `finetune_joint.py:209-214`.
- Row PhaseC audit: `finetune_joint.py:242-252` + `self_audit_net.py:183-190`.

### 4.1 F4 — the ΔQ head is starved

Same auditor, same batch, same transition, equal weights (`audit_loss(local_weight=1.0, global_weight=1.0)`, `losses/audit.py:71-72,95`):

```
[B LOCAL term only] auditor = 1.248067e+01
[B dQ    term only] auditor = 3.638188e-01     <-- 34x smaller
```

This is structural, not a seed artifact:
- The local term is `F.cross_entropy` over `B x 3 x H x W` pixels (`audit.py:88`) with balanced-clamped class weights (`train_auditor.py:131-142`, clamp `[0.25, 5.0]`).
- The global term is `F.smooth_l1_loss` on a scalar per sample (`audit.py:27,30,36`) with PyTorch's **default `beta=1.0`**, while `delta_dice` targets are on the order of 1e-3..1e-2. In that regime `smooth_l1` is in its quadratic branch: `0.5*x^2/beta`, so `d/dx = x ~ 1e-2`. The pairwise hinge (`audit.py:26`, `margin=0.05`) only fires when both a positive and a negative sample are present in the same transition batch (`audit.py:25`).

`delta_q` is the **only** signal the deployable gate consumes (`self_audit_net.py:204`: `accepted_active = delta_q > tau_accept`). The head that decides ACCEPT/REJECT is getting ~3% of the auditor's gradient. If the paper's claim is "the Auditor captures headroom while blocking harmful edits", the training signal for the blocking mechanism is the weakest term in the objective. **P2 — recommend an explicit `global_weight > 1` or `beta ~ 0.05` in the smooth_l1.**

---

## 5. Phase B — what is frozen, and how

`freeze_annotation_network` — `src/self_audit/training/train_auditor.py:52-64`.

**Mechanism 1 — `requires_grad`** (`:55-56`):
```
for name, parameter in model.named_parameters():
    parameter.requires_grad = name.startswith("auditor")
```
Note this is a plain attribute assignment, not `requires_grad_(False)`, and it is *prefix-string based*. Verified: `requires_grad True names: ['auditor']`. Fragile if a module is ever renamed, but correct today.

**Mechanism 2 — `.eval()`** (`:57-64`) on `encoder`, `fpn`, `initial_head`, `annotation_expert`. Verified after the call: `{'encoder': False, 'fpn': False, 'initial_head': False, 'annotation_expert': False, 'auditor': True}`.

**Mechanism 3 — `torch.no_grad()`** (`:164-166`) around the annotation forward and the feature extraction, and again at `:169-170` around counterfactual generation, `:188-190` around entropy.

**Mechanism 4 — module-level `detach()`** inside the auditor itself (`src/self_audit/models/auditor.py:89-94, 104, 109`), which is the actual load-bearing firewall.

### 5.1 Are frozen modules still in `train()` mode? — NO

`train_auditor_epoch` calls `model.eval()` at `:254` then `model.auditor.train()` at `:255` on **every epoch**, so even if something re-entered train mode between epochs the annotation side is put back into eval before the first batch. `validate_auditor_epoch` calls `model.eval()` at `:375`.

**No BatchNorm / dropout drift is possible in this model.** Grep across `src/self_audit/models/` returns only `nn.GroupNorm` — `auditor.py:17,55`; `annotation_expert.py:68,81`; `annotation_head.py:17`; `fpn.py:33,41`; `encoder.py:16,41,47`; `dynamic_window.py:70`. Zero `BatchNorm`, zero `Dropout`. GroupNorm has no running statistics and is train/eval-invariant, so the `.eval()` calls are belt-and-braces.

**Static-only caveat:** with `timm` installed, `self.model` is `timm.create_model("convnext_tiny", features_only=True, ...)` (`encoder.py:93-99`). timm ConvNeXt uses LayerNorm (also mode-invariant) and DropPath gated on `drop_path_rate`, which is **not passed** at `encoder.py:93-99` and therefore defaults to 0.0 — DropPath is an identity. I could not execute this branch (`timm` absent), so I flag it as unverified-but-almost-certainly-fine.

### 5.2 Are frozen params excluded from the optimizer, or just `requires_grad=False`? — BOTH

`train_auditor.py:492-493`:
```
auditor_parameters = [p for p in model.parameters() if p.requires_grad]
optimizer = build_adamw_optimizer(auditor_parameters, lr=config["auditor_lr"], ...)
```
Frozen params are physically absent from the param groups. `build_adamw_optimizer` (`_utils.py:511-533`) additionally **raises** if any supplied param has `requires_grad=False` (`:526-527`). Ordering is correct: freeze at `:486`, build optimizer at `:492`. Same in the single-process pipeline (`scripts/train_self_audit.py:362` then `:365-370`).

`finalize_optimizer_step` (`_utils.py:681-729`) iterates `model.parameters()` for the accumulation correction (`:701-703`) and `clip_grad_norm_` (`:706`), but both skip `grad is None`, and in Phase B **all 120 annotation params have `grad is None`** (measured). So the clip norm is computed over the auditor only. Correct.

### 5.3 Verified: Phase B audit loss cannot touch annotation weights

```
[B audit loss backward]
  encoder           = 0.000000e+00 (grad_n=0, none=52)
  fpn               = 0.000000e+00 (grad_n=0, none=28)
  initial_head      = 0.000000e+00 (grad_n=0, none=6)
  annotation_expert = 0.000000e+00 (grad_n=0, none=34)
  auditor           = 9.926813e+00 (grad_n=28, none=0)
```
`transitions: 7` (3 on-policy adjacent pairs + 4 synthetic). Grads are literally `None`, not merely zero — the graph is severed, not cancelled. **Firewall holds.**

### 5.4 Phase B inefficiency (P3)

`train_auditor.py:165-166` runs `model.forward_annotation(images)` and then `_feature_for_audit(model, images)` — the latter calls `model.encode(images)` (`train_auditor.py:68-69`), recomputing the encoder + FPN that `forward_annotation` already computed and returned as `output["shared_features"]` (`self_audit_net.py:92`). That is a **second full encoder pass per batch**, discarded work under `no_grad`. Trivially fixable by reading `output["shared_features"]`.

Also: `model.auditor(...)` runs inside `autocast_context(...)` (`:191`) but `audit_loss(...)` is computed **outside** it (`:200-206`), so the CE/smooth-L1 run on bf16 tensors without autocast's fp32 promotion for reduction ops. Minor numerics.

---

## 6. Phase C — param groups, LRs, summation, detach boundaries

### 6.1 Optimizer param groups and LRs

`finetune_joint.py:743-748` -> `encoder_head_optimizer` (`_utils.py:481-508`).

Exactly **two** groups (`_utils.py:501-505`):

| Group | Members | LR | Config source |
|---|---|---|---|
| 0 | everything under `model.encoder` | `joint_encoder_lr` = **1e-6** | `configs/self_audit_joint.yaml:26` |
| 1 | **everything else** — `fpn`, `initial_head`, `annotation_expert`, **and `auditor`** | `joint_lr` = **1e-5** | `configs/self_audit_joint.yaml:25` |

`weight_decay = 1e-4` applied to both (`_utils.py:508`, `configs/self_audit_joint.yaml:27`).

**F8 (P2):** the split is `id(p) in encoder_ids` vs. everything else (`_utils.py:493-497`). There is **no auditor group**. The auditor goes from `auditor_lr: 0.0003` in Phase B (`configs/self_audit_auditor.yaml:25`) to **1e-5** in Phase C — a silent 30x drop — and shares a group with the annotation heads whose optimal Phase-C LR is deliberately tiny. Identical in the single-process pipeline (`scripts/train_self_audit.py:461-469`). If the intent is "keep the auditor learning while the annotator barely moves", the code does the opposite.

### 6.2 Which losses are summed

`compute_joint_losses` — `finetune_joint.py:179-275`.

```
214  annotation_term = annotation_loss(final_logits, batch["mask"])[0]
247  audit_term, parts = audit_loss(selected_audit, targets, neutral_margin=..., local_class_weights=...)
257  audit_term = torch.stack(audit_terms).mean()  if audit_terms else _zero_audit_term(...)
258  total = annotation_term + float(lambda_audit) * audit_term
```

- Annotation term: Dice+CE (1.0/1.0) on **`output["logits"]` only** — the *retained final state* after the audit gate (`:209-211`). **A0/A1/A2 are NOT supervised in Phase C.** Deep supervision exists only in Phase A.
- Audit term: mean over per-transition `audit_loss`, each = `1.0*local_CE + 1.0*signed_ranking` (`audit.py:71-72,95`).
- `lambda_audit: 1.0` (`configs/self_audit_joint.yaml:28`), read at `finetune_joint.py:776`.
- `tau_accept: 0.0`, `t_max: 3`, `neutral_margin: 0.005` (`configs/self_audit_joint.yaml:30-32`), read at `:777-778, :792`.

`_zero_audit_term` (`:52-58`) builds `sum(p.sum()*0.0 for p in auditor.parameters())` so the total stays differentiable w.r.t. the auditor even when the hard cap yields no transition. Correct and deliberate — it contributes exactly zero gradient, not a fake one.

### 6.3 Can audit loss update annotation params? — NO. Can annotation loss update auditor params? — NO.

**Detach boundaries (three layers):**

1. `finetune_joint.py:242-243` — `previous.detach()`, `candidate.detach()` before target construction.
2. `self_audit_net.py:183-190` — the auditor is called with `shared_active.detach()`, `previous_probs.detach()`, `candidate_probs.detach()`, `(candidate_probs - previous_probs).detach()`, and both entropies `.detach()`. `previous_probs`/`candidate_probs` are themselves `.detach().softmax(...)` at `:173-174`.
3. `auditor.py:91-94, 104, 109` — the module re-detaches every input internally: `H_audit = H.detach()`, `P_previous.detach()`, `P_candidate.detach()`, `delta_P.detach()`.

**Reverse direction (auditor -> annotation):** the only feedback path is `previous_audit_evidence` fed back into the expert (`self_audit_net.py:159, 217-221`). It is built from `local_logits.detach().softmax(dim=1)` at `:216`. Severed.

**Measured (`tau_accept=-1e9`, all accepted, `transition_count: 3`):**
```
[C AUDIT-only backward] encoder=0.0(none=52) fpn=0.0(none=28) initial_head=0.0(none=6)
                        annotation_expert=0.0(none=34) auditor=4.744489e+00(n=28)
[C ANNOT-only backward] encoder=4.926821e-01(n=52) fpn=5.090779e-01(n=16) initial_head=8.089357e-01(n=6)
                        annotation_expert=1.133815e+00(n=34) auditor=0.0(none=28)
[C TOTAL backward]      encoder=4.926821e-01 fpn=5.090779e-01 initial_head=8.089357e-01
                        annotation_expert=1.133815e+00 auditor=4.744489e+00
```
TOTAL is the exact union of the two isolated backwards, with no cross-talk in either direction. **Firewall verified. No leak. Nothing to escalate here.**

### 6.4 EMA / stop-grad tricks

Grepped `ema|EMA|swa|polyak|stop_grad|stopgrad` across `src/self_audit/` and `scripts/train_self_audit.py`: **zero hits** (only substring noise from `schema`/`states`/`semantic_class_swap`). There is no teacher/student, no momentum copy, no target network. The only stop-grad mechanism is the explicit `detach()` set above, plus the non-differentiable boolean gate `accepted = delta_q > tau` (`self_audit_net.py:204`) which carries no gradient by construction — there is **no** straight-through estimator or Gumbel relaxation anywhere. The accept decision is a hard, gradient-free branch.

---

## 7. Checkpoint loading across phases

`save_checkpoint` — `_utils.py:762-825`. `load_checkpoint` — `_utils.py:880-920`.

### 7.1 What is saved

`payload["model"] = _cpu_state_dict(model.state_dict())` (`:787`) — the **full** state dict, all five submodules, every phase. No filtering, no phase-specific subsetting.

### 7.2 `strict=` flags and key remapping

`load_checkpoint(..., strict: bool = True)` (`_utils.py:888`), applied at `:906` as `model.load_state_dict(model_state, strict=strict)`.

**Every call site in the codebase uses the default `strict=True`:**
- `train_annotation.py:317` (resume)
- `train_auditor.py:481, 483, 513`
- `finetune_joint.py:760, 771`

Grepping the scope files for `strict=` returns no override. **There is no key remapping anywhere** — no prefix stripping, no `module.` handling, no `OrderedDict` rewriting. `_extract_model_state` (`:828-841`) only chooses between `payload["model"]`, `payload["state_dict"]`, or a bare tensor mapping.

**No keys can be silently dropped.** `strict=True` raises on both missing and unexpected keys.

### 7.3 Does Phase B actually load Phase A weights?

**Yes, and it is mandatory.** `train_auditor.py:480-485`:
```
480  if args.resume:                  load_checkpoint(args.resume, model=model, ...)
482  elif args.annotation_checkpoint: load_checkpoint(args.annotation_checkpoint, model=model, ...)
484  else: raise ValueError("Phase B requires --annotation_checkpoint or --resume")
```
Loaded at `:483`, frozen at `:486`. Correct order — freezing after loading.

Minor: on `--resume`, the checkpoint is loaded **twice** — model-only at `:481`, then again with optimizer/scheduler/scaler at `:513`. Wasteful (and re-restores RNG twice), not incorrect.

### 7.4 Does Phase C load Phase B?

**It loads *a* checkpoint, and it is mandatory** (`finetune_joint.py:736-738`), via `:771` (`--checkpoint`) or `:760-767` (`--resume`). But **nothing verifies it is a Phase-B checkpoint.**

### 7.5 F6 (P2) — a random-init auditor can silently survive into Phase C

`save_checkpoint` writes `extra={"phase": "annotation"|"auditor"|"joint"}` (e.g. `train_annotation.py:363`, `train_auditor.py:546`, `finetune_joint.py:828`). `load_checkpoint` **never reads `"phase"`** — grep for `"phase"` in `_utils.py` returns nothing.

The only guard is `_validate_checkpoint_architecture` (`_utils.py:861-877`), which compares only `num_classes`, `shared_channels`, `window_k`, `encoder_name` (`:870`). All four are identical across `configs/self_audit_annotation.yaml:32-38`, `self_audit_auditor.yaml:32-38`, `self_audit_joint.yaml:35-41`.

Consequence: `python -m ...finetune_joint --checkpoint weights/self_audit/phase_a_annotation.pt` loads cleanly under `strict=True` (the Phase-A checkpoint *contains* auditor keys — at their **random init**, since Phase A never trains them), and Phase C then immediately runs `model.infer(mode="self_audit", tau_accept=0.0)` (`finetune_joint.py:197-202`). Every ACCEPT/REJECT decision for the whole run is made by an untrained head. The training loop will not error, and the printed `audit_AUROC` (`:880`) is the only symptom.

This is the single most likely way to produce a silently-invalid result table. **Recommend asserting `payload["phase"] == "auditor"` in Phase C.**

Related weaker gap: `_model_architecture_signature` (`_utils.py:849-858`) reads `encoder.name`, which is hard-coded to `"convnext_tiny"` at `encoder.py:87` **even on the fallback path**. So the architecture check cannot distinguish a timm ConvNeXt from `_FallbackHierarchicalEncoder`. In practice `strict=True` catches it loudly (different key names), so this is P3.

### 7.6 `pretrained_encoder` inconsistency across configs (P3, benign)

`configs/self_audit_annotation.yaml:34` = `true`; `self_audit_auditor.yaml:34` and `self_audit_joint.yaml:38` = `false`. This is **correct** for the separate-script flow (B/C get their encoder from the checkpoint, not from ImageNet) and is not checked by `_validate_checkpoint_architecture` (not in the `:870` key list), so it does not error.

But it is a real trap for `scripts/train_self_audit.py`: `_assert_compatible_configs` (`:124-127`) compares `_architecture_signature` (`:113-121`) which also **excludes** `pretrained_encoder`, and the model is built from `config_a` **only** (`:242`). So `config_b`/`config_c`'s `pretrained_encoder: false` is silently ignored in the single-process path. Benign today (A's `true` is what you want) but the configs are misleading about what actually runs.

### 7.7 Other checkpoint notes

- `torch.load(..., weights_only=True)` (`:897`) — safe deserialization. Good.
- `restore_rng=True` default (`:889, :913-918`): `load_checkpoint` **mutates global RNG state** on every load, including the model-only loads at `train_auditor.py:483` and `finetune_joint.py:771`. Loading a Phase-A checkpoint therefore rewinds the global RNG to Phase-A's saved state *after* `seed_everything` already ran (`train_auditor.py:476`). Not a gradient bug; it does mean the Phase-B/C data order is determined by the checkpoint, not by `seed: 42`. Worth knowing when reproducing.
- Atomic write via tempfile + `os.replace` (`:813-821`) and a full finiteness sweep (`:806-809`). Solid.

---

## 8. AMP / grad-clip / accumulation — can grads be silently zeroed?

**Short answer: no silent zeroing. But there is one latent hard-crash (F7).**

### 8.1 Autocast placement — correct

| Phase | Autocast covers | Backward inside autocast? |
|---|---|---|
| A | forward + loss (`train_annotation.py:155-157`) | No — `:161/163` outside. Correct. |
| B | auditor forward only (`train_auditor.py:191-199`); loss at `:200-206` outside | No — `:294-298`, and `:295` explicitly wraps with `autocast_context(enabled=False, ...)` (a `nullcontext`, `_utils.py:659-660`). Correct. |
| C | forward + loss (`finetune_joint.py:627-636`) | No — `:645/647` outside. Correct. |

### 8.2 The GradScaler is disabled under the checked-in configs

`build_grad_scaler` (`_utils.py:648-655`):
```
651  scale_enabled = bool(enabled and device.type == "cuda" and dtype is torch.float16)
```
All three configs set `amp_dtype: bfloat16` (`annotation.yaml:19`, `auditor.yaml:19`, `joint.yaml:19`). So `scale_enabled = False` **always**, on every device. Every `scaler.is_enabled()` branch (`train_annotation.py:160`, `train_auditor.py:294`, `finetune_joint.py:644`, `_utils.py:697,722`) takes the plain-`backward()` path. `scaler.step()` on a disabled scaler is a passthrough to `optimizer.step()`, `scaler.update()` a no-op. **No scaling, no unscale, no silent skip.**

### 8.3 F7 (P2, latent) — `amp_dtype: float16` on CUDA turns normal scaler warmup into a crash

If anyone flips `amp_dtype` to `float16` on CUDA, `scale_enabled` becomes True and `_utils.py:709-714`:
```
709  gradients_finite = all(is_finite(p.grad) for p in model.parameters() if p.grad is not None)
714  step_ok = math.isfinite(norm) and gradients_finite
```
On a GradScaler's **normal** startup the initial scale (65536) routinely overflows for the first few steps — that is the designed behaviour, and the scaler is supposed to skip the step and halve the scale. Here `step_ok=False` propagates to `train_annotation.py:179-180` / `train_auditor.py:317-318` / `finetune_joint.py:664-665`, each of which **raises `FloatingPointError`** and kills the run. The code at `_utils.py:722-725` correctly feeds the overflow back to the scaler, but the caller has already decided to abort.

Not triggered by the committed configs; documenting because "set amp_dtype: float16 for speed" is a one-line change anyone would make.

### 8.4 Gradient accumulation — correct, no zeroing

`configs/*.yaml:21` all set `gradient_accumulation_steps: 1`, so accumulation is a no-op in the locked baseline. The general path is still correct:
- Per-batch loss is pre-divided: `loss / accumulation_steps` (`train_annotation.py:159`, `train_auditor.py:293`, `finetune_joint.py:643`).
- A trailing partial group is rescaled **before** clipping (`_utils.py:699-703`), which is the right order — clipping first would clip a systematically under-scaled gradient.
- `optimizer.zero_grad(set_to_none=True)` is called once at loop entry and once per step inside `finalize_optimizer_step` (`:728`), never mid-accumulation. Verified by reading `train_annotation.py:139,169-182`, `train_auditor.py:260,307-320`, `finetune_joint.py:616,654-667`.
- The `break` on `max_steps` (`train_annotation.py:152-153` etc.) fires **before** the batch is processed, so it cannot strand half-accumulated grads.

### 8.5 Grad clipping

`grad_clip: 3.0` in `configs/self_audit_annotation.yaml:30`; Phase B **hard-codes `grad_clip=3.0`** at `train_auditor.py:315, 334` (ignores any config); Phase C reads `config.get("grad_clip", 3.0)` at `:799` — and `configs/self_audit_joint.yaml` has **no `grad_clip` key**, so it falls back to 3.0. Consistent value across phases, but Phase B's hard-coding means a config change there is silently ignored (P3).

`clip_grad_norm_(model.parameters(), 3.0)` (`_utils.py:706`) is applied over the *whole* model. In Phase C this couples the annotation and auditor gradient scales through a single global norm — a large annotation gradient will shrink the auditor's update and vice versa. Given F4 (the ΔQ term is already 34x weaker than the local term), this is worth noting but is standard practice.

---

## 9. Gradient-firewall test — results

Both directions, both phases, real `backward()`. Reproduced verbatim from §5.3 and §6.3.

**Phase B — backprop only the audit loss:**
```
encoder=0.0(grad_n=0,none=52) | fpn=0.0(grad_n=0,none=28) | initial_head=0.0(grad_n=0,none=6)
annotation_expert=0.0(grad_n=0,none=34) | auditor=9.926813e+00(grad_n=28,none=0)
```
All 120 annotation-side params: `grad is None`. **PASS.**

**Phase C — backprop only `details["audit_loss_tensor"]`:**
```
encoder=0.0(none=52) fpn=0.0(none=28) initial_head=0.0(none=6)
annotation_expert=0.0(none=34) auditor=4.744489e+00(n=28)
```
**PASS.**

**Phase C — backprop only `details["annotation_loss_tensor"]`:**
```
encoder=4.926821e-01(n=52) fpn=5.090779e-01(n=16) initial_head=8.089357e-01(n=6)
annotation_expert=1.133815e+00(n=34) auditor=0.0(none=28)
```
All 28 auditor params: `grad is None`. **PASS.**

**No firewall leaks. Nothing at P0 or P1 in the gradient-separation dimension.** The separation is enforced redundantly at three independent levels (§6.3), so it is robust to a single-site regression — a future edit would have to remove `auditor.py:91-94` *and* `self_audit_net.py:184-189` *and* `finetune_joint.py:242-243` to break it.

Contrast: `finetune_joint.py:266-268` deliberately exposes `annotation_loss_tensor` / `audit_loss_tensor` (differentiable) alongside the detached scalars, with a comment at `:263-265` saying callers should only backprop the total or one explicit term. That is the hook I used. It is a reasonable diagnostic surface, though it is also a foot-gun: any caller that backprops *both* separately without `zero_grad` in between double-counts. No in-repo caller does this.

---

## 10. Headroom — is A0 encouraged to be optimal on its own AND optimized indirectly?

**Answer: yes to both, and the indirect pressure dominates by ~12x. This is a real mechanism for correction-headroom collapse.**

### 10.1 A0 is directly supervised — but weakly

`train_annotation.py:93,98` applies full Dice+CE to A0 against GT, at raw weight 0.5, effective **0.1667** after `/Σw = 3.0` (`:99`). That is the smallest of the four stage weights. See §1.2.

### 10.2 A0 is *additionally* optimized by A1, A2, A3 — confirmed executably

Because `state` is passed un-detached (`self_audit_net.py:70`), every later-stage loss backprops through the whole chain into `initial_head`. §2's per-stage table shows `initial_head` grad of `3.10e-1 / 5.08e-1 / 6.83e-1 / 8.24e-1` for the A0/A1/A2/A3 terms respectively — later terms hit A0 **harder** than A0's own term does.

### 10.3 Quantified split

Decomposed the `initial_head` gradient under the exact config weights `[0.5,0.7,0.8,1.0]/3.0`, batch of 4:

```
initial_head grad L2:  direct(A0 term)   = 4.645766e-02
                       indirect(A1..A3)  = 5.494189e-01
                       total             = 5.927494e-01
direct/total = 0.0784        cos(direct, indirect) = 0.9272
```

**Only 7.8% of the A0 head's gradient comes from "be a good segmentation on your own". 92.2% comes from "be a good *starting point* for the expert."** These are not the same objective.

### 10.4 The mechanism

`cos(direct, indirect) = 0.93` at initialization looks reassuring, but that is exactly what you expect from an untrained network where every term is pushing toward the same coarse GT. The two objectives diverge precisely where headroom lives: for a pixel the expert reliably fixes, the indirect gradient has **no incentive** to make A0 correct there — it only needs A0 to be a *legible input*. The 12:1 weighting means that wherever the two objectives conflict, the "good launchpad" objective wins.

Three amplifying factors, all verifiable:

1. **Nothing selects for A0.** `validate_annotation_epoch` scores only A3 (`train_annotation.py:239-242`) and `best.pt` is gated on that (`:354, :365`). A checkpoint where A0 collapsed and A3 is excellent is *preferred* by the selection rule.
2. **The shared trunk is shared.** `initial_head` and `annotation_expert` both consume the same `shared` tensor (`self_audit_net.py:62, 69`), and FPN+encoder receive the summed gradient from all four stage terms (§4 table, encoder column: `2.16e-1 / 2.15e-1 / 2.50e-1 / 3.87e-1`). The trunk is optimized for the *final* state, and A0 rides on whatever representation that produces.
3. **Phase C removes the direct term entirely.** `compute_joint_losses` supervises **only** `output["logits"]` (`finetune_joint.py:209-214`). A0 gets zero direct supervision for all 10 Phase-C epochs (`configs/self_audit_joint.yaml:13`), and its *indirect* gradient is now routed through the accept gate — measured: `2.58e-1` when all rejected, `8.09e-1` when all accepted.

### 10.5 The perverse coupling in Phase C

Combine §4's `†` note with §10.4. In Phase C:
- If the auditor **accepts**, gradient flows to the expert (`1.13e+0`) and A0 is pushed to be a good launchpad.
- If the auditor **rejects**, the final state *is* A0, the expert gets **exactly zero** gradient, and the full annotation loss lands on `initial_head` — pushing A0 to be a good *final answer*.

So Phase C's objective for A0 flips depending on the auditor's decision, and the expert only learns from transitions the auditor already likes. That is a self-reinforcing loop: an auditor that under-accepts starves the expert of gradient, which makes the expert worse, which makes the auditor accept even less. There is no exploration term, no entropy bonus, and no forced-accept schedule anywhere in `finetune_joint.py` to counter it. The `always_accept_refinement` mode exists (`self_audit_net.py:194-195`) but is used **only** in the no-grad calibration/diagnostic paths (`finetune_joint.py:510-515`, `:486` `@torch.no_grad()`), never in training.

### 10.6 What would falsify vs. confirm the headroom hypothesis

The project already computes `evaluate_annotation_headroom` and logs `phase_a/a0_dice`, `phase_a/total_refinement_gain`, `phase_a/mean_stage_headroom` (`scripts/train_self_audit.py:312-323, 352-357`), so the instrumentation exists. Concretely:

- **Falsifies collapse:** A0's standalone Dice tracks a Phase-A-trained *single-stage* baseline (stage_weights `[1.0, 0, 0, 0]`) within noise, while `total_refinement_gain` stays > 0.
- **Confirms collapse:** A0 Dice is materially below that baseline and `refinement_gain` is large — i.e. the model bought its headroom by making A0 worse, which is measurement theatre, not a correction mechanism.

That ablation is not in the repo. Without it, a positive `refinement_gain` is **not** evidence for the central hypothesis, because the 12:1 gradient asymmetry gives the model a direct route to manufacture gain by degrading A0. **This is the report's main scientific reservation (F2, P1).**

---

## 11. Neutral-margin semantics vs. spec

Spec: `dDice > +eps` IMPROVE, `|dDice| <= eps` NEUTRAL, `dDice < -eps` REGRESS.

Code — `src/self_audit/losses/audit.py:18-40`:
```
20  neutral   = actual.abs() < float(neutral_margin)      # strict <, spec says <=
23  positive  = ranking_predicted[ranking_actual > 0]
24  negative  = ranking_predicted[ranking_actual < 0]
```

1. **Boundary is `<`, not `<=`** (`:20`). `|dDice| == eps` exactly is treated as non-neutral. Measure-zero in float; note only.
2. **Default `neutral_margin = 0.0`** at `audit.py:16` and `:74`. With 0.0, `actual.abs() < 0` is never true — the neutral set is empty and the neutral calibration term at `:33-39` never fires. Both trainers do pass 0.005 from config (`train_auditor.py:525`, `finetune_joint.py:792`), but **any direct caller of `audit_loss` gets no neutral handling at all**, silently.
3. **Two different epsilons both named "neutral."** `configs/self_audit_auditor.yaml:27` `epsilon_neutral: 0.02` (fed to `CounterfactualGenerator`, `train_auditor.py:506`) and `:29` `neutral_margin: 0.005` (fed to the loss, `:525`). A 4x gap between "what counts as a neutral edit when generating counterfactuals" and "what counts as neutral when scoring them." That may be deliberate, but nothing in the configs or code says so, and the task brief describes a single eps. Worth a one-line comment at minimum.
4. **`local` targets correctly use no eps** — `local_audit_targets` (`audit/targets.py:55-70`) is pixel-exact FIX/UNCHANGED/REGRESS from label equality (`:65-69`). Matches spec; the eps applies only to the global ΔDice, as it should.

---

## 12. Smaller items

- **`--output` is dead in Phase C.** Parsed at `finetune_joint.py:712`, but `:779` reads `Path(config.get("output", ...))` and never consults `args.output`. Phase A (`train_annotation.py:322`) and Phase B (`train_auditor.py:516`) both honour it. Silent no-op flag. **P3.**
- **Phase B `audit_margin` is never configurable.** `train_auditor_epoch(audit_margin=0.05)` default at `:249`; `main` does not pass it (`:521-528`), and no config key exists. Same for `finetune_joint` (`audit_loss` default `margin=0.05`, `audit.py:73`, never overridden at `:247-252`).
- **`local_class_weighting` is a truthiness check on a string.** `cf_cfg.get("local_class_weighting", True) != "none"` (`train_auditor.py:526`, `finetune_joint.py:793`, `scripts/train_self_audit.py:405,496`). The configs say `balanced_clamped` (`auditor.yaml:30`, `joint.yaml:33`), which just evaluates to `True`. There is no other weighting scheme implemented — the string is decorative. Any typo (`balanced_clampd`) is also `!= "none"` and silently enables weighting. **P3.**
- **`include_background=True` in Phase A.** `annotation.py:33,35` — background is in the Dice mean, while every reported metric is *foreground* macro Dice (`train_annotation.py:253`, `finetune_joint.py:287` -> `targets.py:77` `range(1, num_classes)`). Train/report mismatch; deliberate-looking but undocumented.
- **`validate_phase_c` restores train mode with `model.train(was_training)`** (`finetune_joint.py:310, 421`), which recursively re-enables train mode on *all* submodules. Harmless in Phase C (everything is trainable) and correctly scoped by `try/finally`.

---

## 13. What I could not verify

- Anything requiring `timm` / real ConvNeXt weights (norm-layer behaviour under `.eval()`, DropPath). Static reasoning only, flagged in §5.1.
- Anything requiring the ACDC dataset — `validate_dataset_splits`, `build_patient_dataset`, real loss magnitudes, real acceptance rates, whether headroom actually collapses in practice. My gradient-share number (7.8%) is measured at initialization on synthetic data; the *ratio* is set by the loss weights and graph topology, but the exact figure will move during training.
- CUDA-specific paths: the enabled-`GradScaler` branch (`_utils.py:697-698, 722-725`) cannot execute on CPU because `build_grad_scaler:651` requires `device.type == "cuda"`. §8.3 is static analysis of that branch.
- `src/self_audit/audit/counterfactual.py` (642 lines) — only skimmed for the epsilon plumbing in §11. Counterfactual generation correctness is another worker's scope.

