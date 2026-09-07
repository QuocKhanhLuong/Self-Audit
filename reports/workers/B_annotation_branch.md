# Worker B — Annotation Branch Audit

**Repo:** `/Users/alvinluong/Self-Audit` · **HEAD:** `a901eea` · **Branch:** `main` · **Mode:** read-only audit
**Scope:** `src/self_audit/models/{encoder,fpn,annotation_head,annotation_expert,dynamic_window,self_audit_net}.py` + callers

## Evidence environment (read this before trusting numbers)

* `torch 2.13.0` is importable (`/Users/alvinluong/miniforge3/bin/python`). All dynamic probes below were actually executed.
* **`timm` is NOT installed** (`ModuleNotFoundError: No module named 'timm'`). Therefore every measured encoder number comes from `_FallbackHierarchicalEncoder` (`src/self_audit/models/encoder.py:29-70`), **not** from real ConvNeXt-Tiny. Encoder parameter counts below are explicitly labelled as fallback numbers and are **not** evidence about the intended pretrained backbone.
* **`pytest` is NOT installed** in this interpreter (`No module named pytest`). The repository's own test claims were read statically (`tests/test_self_audit_core.py`, `tests/test_self_audit_regressions.py`, `tests/test_self_audit_masks.py`) but **were not executed**. Where I make a gradient/sharing claim it is backed by my own probe script, not by their test suite.
* Probe scripts live in the session scratchpad (outside the repo). No file under `src/`, `scripts/`, `configs/`, `tests/` was modified.

---

## 1. What exactly produces A0

Chain: `SelfAuditNet.encode` (`src/self_audit/models/self_audit_net.py:50-55`) then `SelfAuditNet.forward_annotation` (`self_audit_net.py:62`).

**Step-by-step, measured for `images = [2,3,256,256]`:**

| Step | Code | Output shape (measured) |
|---|---|---|
| Encoder stage 0 | `encoder.py:113-117` | `[2, 96, 64, 64]` (÷4) |
| Encoder stage 1 | " | `[2, 192, 32, 32]` (÷8) |
| Encoder stage 2 | " | `[2, 384, 16, 16]` (÷16) |
| Encoder stage 3 | " | `[2, 768, 8, 8]` (÷32) |
| FPN lateral 1×1 ×4 | `fpn.py:28`, `fpn.py:48` | four maps, all `C=96` |
| FPN top-down | `fpn.py:49-52` | `P_i ← P_i + bilinear(P_{i+1}, size(P_i))`, i.e. **×2 upsample, applied 3 times** (i = 2,1,0) |
| FPN refine + output | `fpn.py:53-54` | `shared = [2, 96, 64, 64]` |
| Head conv stack | `annotation_head.py:15-20` | Conv3×3 96→48, GroupNorm(8,48), GELU, Conv1×1 48→4 ⇒ `[2, 4, 64, 64]` |
| Head upsample | `annotation_head.py:26-27` | bilinear to `output_size=images.shape[-2:]`, i.e. **×4** ⇒ **A0 = `[2, 4, 256, 256]`** |

Math, exactly as coded:

```
F_i        = Encoder(x)_i,                 i = 0..3
P_3        = W^lat_3 * F_3
P_i        = W^lat_i * F_i + Up_bilinear(P_{i+1}, size(P_i))     i = 2,1,0
shared     = GELU(GN(W^out * GELU(GN(W^ref_0 * P_0))))            # [B,96,H/4,W/4]
A0_low     = W_2 * GELU(GN(W_1 * shared))                         # [B,4,H/4,W/4]
A0         = Up_bilinear(A0_low, (H,W))                           # [B,4,H,W]   -- LOGITS, no softmax
```

`InitialAnnotationHead.forward` defaults `output_size` to `(shared.H*4, shared.W*4)` (`annotation_head.py:24-25`), which coincides with the true image size only when `H,W` are multiples of 4. `forward_annotation` passes the explicit size, so the deployable path is safe (`self_audit_net.py:62`).

**Parameter counts (measured, `shared_channels=96`, `num_classes=4`, `window_k=8`, `max_turns=3`):**

| Module | Params |
|---|---:|
| Encoder — **FALLBACK, not ConvNeXt-Tiny** | 7,906,272 |
| FPN (`LightweightFPN`) | 554,784 |
| Initial head | 41,812 |
| Annotation expert | 210,250 |
| Auditor | 514,660 |
| **Total (fallback encoder)** | **9,227,778** |

Annotation-expert breakdown (measured): `input_projection` 10,272 · `refinement_block` 65,621 · `refinement_residual` 92,544 · `delta_head` 41,716 · `gate_head` **97**.
Dynamic-window breakdown: `generator.state_net` 27,216 · `generator.parameter_head` 1,029 · `generator.turn_embedding` 64 · `generator.iteration_embedding` 64 · `query/key/value/output` 9,312 each. `heads=4`, `head_dim=24`, `k=8`.

---

## 2. What exactly produces A_candidate at stage t

`AnnotationExpert.forward` (`annotation_expert.py:93-158`). The literal update equation, from `annotation_expert.py:130-156`:

```
A_low^(t)   = Up_bilinear(A_t, size(shared))                              # [B,4,h,w]
E^(t)       = Up_bilinear(annotation_entropy(A_t), size(shared))          # [B,1,h,w]
U^(t)       = Up_bilinear(prev_audit_evidence or 0, size(shared))         # [B,3,h,w]
S_0         = GELU(GN(W_in * concat[shared, A_low^(t), E^(t), U^(t)]))    # 104 -> 96 channels
for i in 0..depth(t)-1:
    S_{i+1} = S_i + R( DWA(S_i; cond=U^(t), turn=t, iter=t+i) )           # SAME weights every i
delta       = Up_bilinear(W_delta * S_depth, size(A_t))
gate        = Up_bilinear(W_gate  * S_depth, size(A_t))
A_candidate^(t) = A_t + sigmoid(gate) ⊙ delta                             # annotation_expert.py:156
```

`sigmoid(gate)` is a **single-channel** map broadcast over all 4 classes (`gate_head = nn.Conv2d(C,1,1)`, `annotation_expert.py:90`). All quantities are logits; no softmax is applied to the state. Matches intended design.

---

## 3. Are AnnotationExpert weights truly shared across turns?

**YES — verified by module identity, not by naming.**

* `SelfAuditNet.__init__` constructs exactly one `AnnotationExpert` (`self_audit_net.py:42-47`). No `ModuleList`, no per-turn table anywhere in `self_audit_net.py`.
* `forward_annotation` calls `self.annotation_expert(...)` inside the turn loop (`self_audit_net.py:67-75`); `infer` likewise (`self_audit_net.py:156-163`).
* `AnnotationExpert.__init__` constructs exactly one `DynamicWindowAttention` (`annotation_expert.py:73-78`), reused by the depth loop at `annotation_expert.py:142-149`.

**Probe result (3 turns, `forward_annotation`):**

```
expert call ids:            [4928442832, 4928442832, 4928442832]  unique: 1
refinement_block call ids:  6 calls, unique: 1
total refinement_block invocations across 3 turns: 6   (= 1+2+3, matches depth schedule)
named_modules match for refinement_block: ['annotation_expert.refinement_block']
```

So both the across-turn loop and the within-turn depth loop reuse one Python object with one parameter set. **Verdict: genuine weight sharing at the module level.** See §4 for the caveat that partially undermines this.

---

## 4. Stage-dependent depth — formula, clamping, and the sharing violation

**Formula (`annotation_expert.py:135-136`):**

```python
depth = min(turn_value + 1, 3)
depth = max(depth, 1)
```

Measured: `turn_index 0→depth 1`, `1→2`, `2→3`, `3→3`, `4→3`. The cap `3` is a **hard-coded literal**, not derived from `self.max_turns`; a model built with `max_turns=5` still saturates at depth 3.

**The sharing violation — CONFIRMED, this is a real research risk.**

`DynamicWindowGenerator` holds two `nn.Embedding` tables indexed by the stage counters (`dynamic_window.py:64-65`):

```python
self.turn_embedding      = nn.Embedding(max(int(max_turns),1)+1, turn_dim)      # shape (4,16)
self.iteration_embedding = nn.Embedding(max(int(max_turns),1)+1, iteration_dim) # shape (4,16)
```

They are looked up per stage and concatenated into the generator state (`dynamic_window.py:114-118`). So the *window-parameter generator* has a **distinct 16-dim learned vector per turn index and per depth-iteration index** — a per-turn parameter table in everything but name. The convolutional weights are shared; the stage conditioning is not.

Measured consequences:

* Sampling coordinates for identical features change purely from the stage counters: `max |coords(turn=0,iter=0) − coords(turn=2,iter=2)| = 0.1906` (normalized units), vs. a full coordinate range of `[-1,1]`.
* Per-row gradient after a 3-turn backward: `turn_embedding` rows nonzero `[True, True, True, False]`; `iteration_embedding` rows nonzero `[True, True, True, True]`.

**Index-collision bug.** `annotation_expert.py:139-146` computes `iteration_value = iteration + iteration_index`, and both callers pass `iteration_index=turn` (`self_audit_net.py:73` and `self_audit_net.py:161`). `dynamic_window.py:115` clamps to `num_embeddings-1 = 3`. Enumerated:

```
turn 0, depth 1 -> raw iter idx [0]      -> clamped [0]
turn 1, depth 2 -> raw iter idx [1, 2]   -> clamped [1, 2]
turn 2, depth 3 -> raw iter idx [2, 3, 4]-> clamped [2, 3, 3]
```

Two distinct depth steps at turn 2 (`raw 3` and `raw 4`) silently collapse onto embedding row 3, and turn 1's second step already shares row 2 with turn 2's first step. The "stage identity" signal is therefore aliased. It is a silent clamp, not an error.

**Verdict on Q4:** depth schedule matches the intended "deeper at later stages" idea, but the parameterisation is **not** purely shared — it is shared-conv + per-stage embedding, and the per-stage index is aliased.

---

## 5. Every input channel of `AnnotationExpert.forward`

`input_channels = feature_channels + num_classes + 1 + audit_channels = 96 + 4 + 1 + 3 = 104` (`annotation_expert.py:65`), consumed by `nn.Conv2d(104, 96, 1)` (`annotation_expert.py:67`). Confirmed arithmetically by the measured 10,272 parameters (104·96 + 96 conv, + 192 GroupNorm).

| Channels | Tensor | Provenance | Grad? |
|---|---|---|---|
| 0–95 (96) | `shared_features` | `LightweightFPN(encoder(x))`, recomputed once per image, identical at every turn (`self_audit_net.py:54`, `:60-62`) | **yes** |
| 96–99 (4) | `annotation_low` | `F.interpolate(A_t)` — the *current* state logits, `annotation_expert.py:109` | **yes** (this is the recurrence) |
| 100 (1) | `entropy_low` | `annotation_entropy(A_t)` normalized by `log(num_classes)`, `annotation_expert.py:110-111`, `:15-23` | **yes** |
| 101–103 (3) | `audit_low` | `previous_audit_evidence` (3-way FIX/NEUTRAL/REGRESS softmax) or **all-zeros** when `None`, `annotation_expert.py:112-129` | **no** — always detached (§6) |

The image itself never re-enters; only `shared`. No positional/coordinate channel. No target channel (§10).

---

## 6. Does previous Auditor evidence affect the next candidate?

Trace of `previous_audit_evidence`:

* **Inference / Phase C (`infer`)** — `self_audit_net.py:216-221`:
  ```python
  local_evidence = local_logits.detach().softmax(dim=1)
  previous_audit = torch.where(accepted[:,None,None,None], local_evidence, torch.zeros_like(local_evidence))
  ```
  * **Detached: YES.** Measured `previous_audit_evidence.requires_grad == False` at turns 1 and 2.
  * **Carries gradient: NO.** `torch.autograd.grad(final_logits.sum(), auditor.parameters())` returned `None` for every auditor parameter. The auditor therefore cannot be trained through the annotation path — matches the stated anti-collusion contract (`auditor.py:89-91`).
  * **Zeroed on reject: YES**, explicitly (`self_audit_net.py:217-221`).
  * *Adversarial note:* "rejected" and "no evidence yet (turn 0)" are encoded by the **same all-zeros tensor**. The expert cannot distinguish them. In `self_audit` mode this is masked because a rejected row leaves `active` (`self_audit_net.py:249-250`), but in `always_accept_refinement` and `oracle_accept` it is a genuine ambiguity.

* **Phase A (`forward_annotation`) — evidence is NEVER available.** `audit_evidence = None` is set at `self_audit_net.py:64` and **is never reassigned** anywhere in the loop (`self_audit_net.py:67-78`). Probe confirms:
  ```
  forward_annotation per-turn previous_audit_evidence: [(0, None), (1, None), (2, None)]
  infer(always_accept)  per-turn: [(0, None), (1, [2,3,64,64] grad=False), (2, [2,3,64,64] grad=False)]
  ```
  **Consequence:** during the whole of Phase A (annotation pretraining, `train_annotation.py:156`) and Phase B (auditor training, `train_auditor.py:165`), 3 of the expert's 104 input channels are constant zero *and* the `DynamicWindowAttention` condition is constant zero. The audit-conditioned window is **inert for the entire annotation-training curriculum** and only becomes live in Phase C / deployment. This is a train/test input-distribution shift on the expert, introduced by architecture, not by data.

---

## 7. Is `DynamicWindowAttention` real attention or a disguised conv?

**It is real (deformable) attention, not a fixed conv — but `k` is static and at initialisation it behaves close to an 8-tap average.**

Actual computation (`dynamic_window.py:212-253`):

1. Generator predicts per-pixel `center ∈ ±0.25`, anisotropic `radius ∈ [0.03, 0.55]` (two values), `orientation ∈ ±π`, and `K` free residual offsets `∈ ±0.12`, all in normalized grid units (`dynamic_window.py:120-131`).
2. Support = a canonical `K`-point ring (`dynamic_window.py:76-77`) scaled by `radius`, rotated by `orientation`, offset by `center` + per-point `residual`, added to a base identity grid, then clamped to `[-1,1]` (`dynamic_window.py:196-209`).
3. `K` (`key`) and `V` (`value`) are sampled at those continuous coordinates with **`F.grid_sample` bilinear** (`dynamic_window.py:233-234`).
4. Multi-head dot-product attention of the query at the pixel against its own `K` sampled points, `softmax` over `K` (`dynamic_window.py:237-243`).

**There is no `topk` and no `gather` anywhere in the file** — sampling is continuous-coordinate bilinear, so the operator is differentiable w.r.t. the window geometry. `k` is **static**: `self.k = int(k)` fixed at construction (`dynamic_window.py:157`, `annotation_expert.py:75`, config `window_k: 8`). Nothing selects `k` from data.

**Correctness checks I ran (all pass):**

* The flattened `reshape(B,H,W*K,2)` / `view(B,C,H,W,K)` round-trip (`dynamic_window.py:230-236`) is index-correct: max abs difference vs. an explicit per-point `grid_sample` loop = **0.000e+00**. `coordinates.is_contiguous() == True`.
* `_base_grid` (`dynamic_window.py:174-178`) is `align_corners=True`-consistent with the `grid_sample` calls: identity-grid resample reproduces the input to `9.5e-07`.
* Attention sums to 1 over `K`.

**Data dependence, measured at init (feature map 16×16):**

* Coordinates differ across two different inputs: `max |Δ| = 0.3483` — so the window **is** input-conditioned, not fixed.
* Coordinates respond to the audit condition: `max |Δ| = 0.0289` (an order of magnitude weaker than the input response — note the condition is all-zeros in Phase A anyway, §6).
* Support offsets at init: mean radius **2.04 px**, max **2.89 px** — an 8-point ring roughly the size of a 5×5 neighbourhood.
* Attention at init is **near-uniform**: entropy 2.0562 nats vs. 2.0794 for uniform over `K=8`; mean max weight 0.1701 vs. 0.1250 uniform.

**Honest characterisation:** at initialisation it is functionally an 8-tap ring-averaging operator ≈ a dilated depthwise smoothing conv, and it must *learn* its way out of that. Structurally it is Deformable-DETR-style attention and is capable of much more. Calling it "a fixed conv" would be wrong; calling it "learned sparse attention over a data-dependent support" is accurate; but the design gives it no inductive push away from the near-uniform init.

---

## 8. HIGH PRIORITY — where does gradient from later stages reach A0 / initial_head / FPN / encoder?

**The chain is unbroken. There is no detach on the annotation path in `forward_annotation`.**

Exact tensor chain (`self_audit_net.py:60-78` → `annotation_expert.py:104-156`):

```
images
  -> encoder(x)                                       [no detach]
  -> fpn(features) = shared                           [no detach]
  -> initial = initial_head(shared, size)  == A0      [no detach]
  -> state_0 = initial
  for t in 0..T-1:
      annotation_low = interpolate(state_t)           [no detach]  <- carries A0 grad
      entropy        = annotation_entropy(state_t)    [no detach]  <- carries A0 grad
      audit_low      = zeros                          [constant, Phase A]
      S              = input_projection(cat[shared, annotation_low, entropy, audit_low])
      S              = S + refinement_residual(DWA(S, cond=zeros, ...))   x depth(t)
      state_{t+1}    = state_t + sigmoid(gate) * delta                    [annotation_expert.py:156]
  -> logits = state_T
```

`state_{t+1} = state_t + ...` is an `Add`, so `∂state_T/∂A0` contains an **identity path** through every stage in addition to the learned path.

**Measured (3 turns, `forward_annotation`, loss = `logits.square().mean()`):**

```
final.grad_fn = AddBackward0
d(final)/d(A0) is not None: True, abs-sum = 33250.4
  encoder        52/52 params with grad, grad_abs_sum = 1018.51
  fpn            16/28 params with grad, grad_abs_sum =  793.50
  initial_head    6/6  params with grad, grad_abs_sum =  269.32
  expert         34/34 params with grad, grad_abs_sum = 1151.67
  auditor         0/28 params with grad, grad_abs_sum =    0
```

The single `.detach()` calls in the annotation branch are:
* `annotation_expert.py:18` — inside a *Python-level* `if` predicate only (see §11 for why that is still a bug).
* `self_audit_net.py:173-174, 184-193, 216` — all on the **auditor** input/decision path in `infer`, never on the annotation state. Verified: `d(final)/d(auditor params) = None`.

**Phase C (`infer`) gradient behaviour, measured:**

* Accept-all (`tau_accept=-inf`, 3 turns): all four annotation modules receive nonzero gradient; `d(final)/d(initial_head)` present with magnitudes up to `5.86e5`.
* **Reject-all (`tau_accept=+inf`): `expert grad_abs_sum = 0.0`**, while encoder/FPN/head still receive gradient through `A0`. This is a genuine training-dynamics hazard, not a bug: in Phase C (`finetune_joint.py:197-214`, annotation loss on `output["logits"]` only) an expert whose candidates are being rejected gets **no annotation gradient at all** and cannot climb back out. The loop also breaks after turn 0 in that case (`self_audit_net.py:138-139`), so only 1 audit transition is produced.

---

## 9. Is the residual form capable of a null update, and how is `gate_head` initialised?

**Null update: not exactly reachable through the gate.** `sigmoid(·) ∈ (0,1)` is strictly positive, so `sigmoid(gate) ⊙ delta = 0` requires `delta_logits ≡ 0` exactly. There is no bias, no learned zero-init scale, and no straight-through/hard-gate anywhere in `annotation_expert.py:152-156`. A null update is reachable only asymptotically, by driving `delta_head`'s output conv to zero.

**Initialisation is *not* near-identity.** `gate_head = nn.Conv2d(96, 1, 1)` (`annotation_expert.py:90`) uses PyTorch's default Kaiming-uniform init — no zero-init, no negative bias. Measured on a freshly constructed model:

```
gate_head.weight.std = 0.0552,  gate_head.bias = [+0.0540]
sigmoid(gate):  min 0.2168  mean 0.3760  max 0.5587
delta_logits:   abs mean 0.1075,  abs max 0.4516
applied step:   abs mean 0.0402,  abs max 0.1778
A0:             abs mean 0.2557
step / A0 (abs mean ratio) = 0.157
argmax label changed by turn 0 at init: 3.23 % of pixels
```

So at initialisation each stage applies a ~16 %-of-signal edit with the gate sitting near 0.38, i.e. **near "half open", not near identity**. Over 3 stages this compounds. The design intent "gate starts closed so the expert begins as a no-op" is **not implemented**; a one-line `nn.init.constant_(self.gate_head.bias, -4.0)` would deliver it. (Reported as an observation — no fix applied, per the read-only rule.)

---

## 10. Can the annotation branch see a target/mask/GT tensor?

**No — the five annotation modules are clean.** A grep for `target|mask|label|ground_truth` across `encoder.py`, `fpn.py`, `annotation_head.py`, `annotation_expert.py`, `dynamic_window.py` returns **0 hits**.

The only GT-shaped tensor anywhere in `models/` is `oracle_target` in `SelfAuditNet.infer` (`self_audit_net.py:104`), and it is properly fenced:

* It is *only* read under `mode == "oracle_accept"` (`self_audit_net.py:196-202`), which raises if the target is absent (`self_audit_net.py:116-117`).
* It reaches only `_candidate_improves` → `_dice` (`self_audit_net.py:340-357`), which is a **discrete accept/reject decision on detached argmax labels**. It never enters `annotation_expert`, `initial_head`, `fpn`, or `encoder`, and it never enters any tensor that is differentiated.
* `volume_inference.py:179-180` hard-refuses `oracle_accept` in the deployable entry point: `raise ValueError("oracle_accept requires an explicit analysis wrapper; GT is not an inference input")`. The only callers that pass it are analysis code (`volume_inference.py:294-297`, `audit_decomposition.py:282-285`, `:416-428`).

**Caveat, stated honestly:** `oracle_accept` *does* change the state trajectory (it selects which candidates are retained), so any metric computed from it is GT-informed by construction. That is correct as an oracle upper bound, and the code labels it as such. I found **no** leakage into the deployable `self_audit` path.

---

## 11. Additional adversarial findings (not asked, but load-bearing)

**F1 — 45 % of the FPN is dead weight and dead compute.** `fpn.py:53-54` computes `refined = [block(f) for block, f in zip(self.refine, pyramid)]` for all four scales, then returns `self.output(refined[0])`. `refine[1..3]` are never consumed. Measured after a backward pass:

```
refine[0] params= 83,232  grad_abs_sum = 86.31
refine[1] params= 83,232  grad_abs_sum = 0.0
refine[2] params= 83,232  grad_abs_sum = 0.0
refine[3] params= 83,232  grad_abs_sum = 0.0
DEAD params in FPN: 249,696 of 554,784  (45.0 %)
```

These parameters are allocated, checkpointed, optimizer-state-tracked, reported in `get_model_parameter_summary` (`_utils.py:212-238`) as if they were part of the model, and their forward FLOPs are burned on every single training step. Any "parameter count" claim in the paper that includes the FPN is inflated by ~250 k.

**F2 — `annotation_entropy` has an input-dependent branch that silently changes its own definition.** `annotation_expert.py:18-21`:

```python
if logits_or_probs.min().detach() < 0 or logits_or_probs.max().detach() > 1:
    probs = logits_or_probs.softmax(dim=1)
else:
    probs = logits_or_probs / logits_or_probs.sum(dim=1, keepdim=True).clamp_min(eps)
```

The branch is decided by a **whole-tensor** min/max, so a single element crossing zero flips the semantics for the entire batch. Measured on the same tensor with one element nudged to `-0.01`:

```
entropy(all values in [0,1]) mean 0.8729
entropy(one element negative) mean 0.9825      ratio 1.126
max per-pixel absolute difference 0.5401
```

At initialisation `A0` has `abs mean ≈ 0.2557`, i.e. it sits right in the regime where whether any element is negative is close to a coin flip. Channel 100 of the expert's input can therefore change meaning between batches, and between turns of the same batch. `CounterfactualAuditor._probabilities` (`auditor.py:68-74`) carries the identical pattern.

**F3 — the encoder can silently degrade to a random 7.9 M-parameter CNN.** `encoder.py:87` sets `self.name = "convnext_tiny"` *before* the `timm` import is attempted, and the fallback path (`encoder.py:103-111`) is taken whenever `pretrained=False` and `timm` is unavailable. `_model_architecture_signature` (`_utils.py:217-224`) records only `encoder_name`, which is `"convnext_tiny"` in **both** cases, so `_validate_checkpoint_architecture` (`_utils.py:230-246`) cannot detect the substitution. Configs `configs/self_audit_auditor.yaml:34` and `configs/self_audit_joint.yaml:37` both set `pretrained_encoder: false`, which is exactly the condition that enables the silent fallback. `using_timm` (`encoder.py:89`, `:102`) is computed but never checked by anything. On this machine `timm` is absent, so this failure mode is live here.

**F4 — the "headroom" hypothesis is not supported by the Phase-A objective.** `phase_a_loss` (`train_annotation.py:82-103`) applies the **same GT loss to A0 and to every refinement state**, with weights `DEFAULT_STAGE_WEIGHTS = (0.5, 0.7, 0.8, 1.0)` (`train_annotation.py:51`; `configs/self_audit_annotation.yaml:29` sets the same values), normalized by their sum. Two consequences worth stating plainly:

* Every stage is driven toward the *same* optimum. The loss-minimising expert, once A0 is good, is the identity map. Nothing in the objective rewards A0 leaving correctable error behind. The only thing that creates headroom is the **loss weighting** (A0 at 0.5 vs. final at 1.0) — a soft, hand-set knob, not an architectural mechanism.
* That knob is weaker than it looks: A0 additionally receives gradient from all three downstream stage losses through the identity residual path (§8), so its *effective* supervision weight is well above its nominal 0.5/3.0 = 16.7 % share.

The central hypothesis "the Annotation Expert creates correction headroom" therefore rests on training dynamics that the code does not enforce. This is falsifiable and should be measured directly (per-stage Dice on A0 vs. A_T, and the fraction of transitions with `|ΔDice| > eps`), not assumed.

---

## Scorecard vs. intended design

| Component | Score | Justification |
|---|---|---|
| **Encoder** (`encoder.py`) | **PARTIAL** | Interface and multi-scale contract match the design (`[B,3,H,W]` 2.5-D input enforced at `encoder.py:85-86`, `:114-115`; correct `(96,192,384,768)` @ `(4,8,16,32)`). But F3: a random-weight 7.9 M fallback can silently replace ConvNeXt-Tiny under the exact config used for Phases B and C, and no checkpoint validation can catch it. Not verifiable here — `timm` is absent. |
| **FPN** (`fpn.py`) | **PARTIAL** | Top-down fusion to one `[B,96,H/4,W/4]` map is exactly as intended and measured correct. F1: 249,696 of 554,784 parameters (45 %) are constructed, optimized, checkpointed and forward-computed but never reach the output. Functionally correct, structurally wasteful, and it corrupts any reported parameter count. |
| **Initial head** (`annotation_head.py`) | **MATCH** | Produces `A0` logits at full image resolution via a 2-conv stack + ×4 bilinear, exactly as specified. Gradient from every later stage reaches it (measured `grad_abs_sum = 269.32`). The only nit is the `shared*4` default `output_size` (`annotation_head.py:24-25`), which the real caller never relies on. |
| **Annotation Expert** (`annotation_expert.py`) | **PARTIAL / RESEARCH RISK** | Sharing is genuine (§3: one module object, 6 invocations, one parameter set) and the residual update matches the spec exactly (`annotation_expert.py:156`). Against it: (a) §9 the gate initialises at `sigmoid ≈ 0.38`, so stage 0 is a ~16 % edit rather than a near-identity no-op, and an exact null update is unreachable; (b) §6 `previous_audit_evidence` is hard-wired to `None` for the whole of Phase A and Phase B, so 3 of 104 input channels and the entire window conditioning are dead during annotation training and only switch on in Phase C — a self-inflicted input-distribution shift; (c) F2 the entropy channel's definition flips on a whole-tensor sign test; (d) §8 an expert whose candidates are rejected receives exactly zero annotation gradient in Phase C. |
| **Dynamic window** (`dynamic_window.py`) | **RESEARCH RISK** | The operator is what it claims — differentiable deformable attention with a learned, input-conditioned, bounded `K`-point support; indexing and `align_corners` handling verified numerically exact. Two substantive risks: (a) §4 the `turn_embedding`/`iteration_embedding` tables give the generator **per-stage learned parameters**, which contradicts "one shared expert" in spirit even though the convolutions are shared, and the iteration index **aliases** (`turn 2` steps 2 and 3 both map to row 3 after clamping at `dynamic_window.py:115`); (b) §7 at initialisation the operator is a near-uniform 8-tap ring average (attention entropy 2.056 vs. 2.079 uniform; mean support radius 2.0 px), i.e. it starts life indistinguishable from a small dilated smoothing conv and nothing in the design pressures it away from that. `k` is static (`= 8`), so "dynamic" refers to the support *geometry*, not its cardinality — worth stating precisely in any write-up. |

## What I could not verify

1. Real ConvNeXt-Tiny parameter counts, features, and pretrained behaviour — `timm` is not installed in this interpreter.
2. The repository's own test assertions — `pytest` is not installed. Test *claims* were read; test *outcomes* are unknown. My gradient, sharing, depth, gate-init, and window findings come from independent probe scripts, not from that suite.
3. Any statement about trained behaviour. Every dynamic number above is at **random initialisation**; no checkpoint was loaded and no training was run. Findings §9 and §7 in particular describe initialisation, and a trained model may differ substantially.
