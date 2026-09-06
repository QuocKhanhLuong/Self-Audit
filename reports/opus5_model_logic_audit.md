# Self-Audit — Principal Architecture & Training-Logic Audit

**Repository:** `QuocKhanhLuong/Self-Audit`
**HEAD audited:** `a901eeae56cf6835bb9a1d9c79a9d5d3e99374c6` (branch `main`)
**Auditor:** Claude Opus 5, coordinating six parallel specialist workers (Orca run `run_b8b704ba174c`)
**Date:** 2026-09-05
**Mode:** read-only. No file under `src/`, `scripts/`, `configs/`, `tests/` was modified.

Worker reports (raw evidence, retained): `reports/workers/{B_annotation_branch,C_auditor_counterfactual,D_gradient_flow,E_gt_leakage,F_metrics,G_data_protocol}.md`

**Evidence environment.** `torch 2.13.0` is importable, so all gradient, invariance and metric probes below were *executed*, not inferred. `timm` is absent, so every instantiated model used `_FallbackHierarchicalEncoder` (`src/self_audit/models/encoder.py:29-70`); autograd topology is identical, so connectivity conclusions transfer, but ConvNeXt-specific parameter counts do not. `preprocessed_data/` and `weights/` do not exist in this checkout, so no claim below rests on a trained checkpoint or on real ACDC volumes; quantitative data claims are marked analytic or synthetic where that is the case.

---

## 1. Executive verdict

| Branch | Verdict |
|---|---|
| **A. Annotation branch** | **NEEDS CHANGES** — architecture matches the intended design (genuine shared recurrent expert, correct residual-gated update, no GT access), but the training objective and two implementation details make the *headroom* claim unmeasurable. |
| **B. Auditor branch** | **NEEDS CHANGES** — the auditor is a genuine transition auditor (`P_previous` is load-bearing, measured), the gradient firewall is airtight, but its curriculum is dominated by GT-synthesized edits, its local head is resolution-starved, and no on-policy-only evaluation exists. |
| **C. Training protocol** | **NEEDS CHANGES** — phase separation, freezing and checkpoint strictness are correct; the Phase-A stage-weight objective drives A0 toward saturation and the Phase-B model-selection metric is contaminated. |
| **D. Evaluation protocol** | **FUNDAMENTALLY MISALIGNED** — the reported Dice is not the ACDC metric (2-D, resized, empty-class→1.0, batch-pooled in Phase A), the key headroom statistic is a numerically invalid estimator, no held-out test split is ever evaluated, and τ is selected and reported on the same data. |

**Direct answers to the seven questions in §12 of the brief:**

1. **Is there actual GT leakage?** **No.** Proven three ways (static call-graph, bitwise GT-permutation invariance, and a powered positive control). Detail in §10.
2. **Is there objective-induced A0 saturation?** **Yes, and it is quantified.** The A0 head receives only **7.8%** of its gradient from its own supervision term; **92.2%** arrives from the A1/A2/A3 losses backpropagating through the shared expert into A0. Detail in §7.
3. **Is there sufficient candidate headroom?** **Unknown, and not measurable as built** — but every structural signal points to *no*. Detail in §7/§8.
4. **Does the Auditor demonstrate measurable independent contribution?** **No.** Every reported auditor number is computed on a 57.1%-synthetic mixture; the on-policy-only metric that would support the claim is not computed anywhere in the repo.
5. **Is high final Dice hiding a weak Auditor?** **The high Dice is itself largely an artifact of the metric**, before the Auditor question is even reached. See §11.
6. **Distinguishing experiment:** see §13, "The one experiment to run first".
7. **Can the current system support the intended paper claim?** **Not yet.** The claim requires an on-policy auditor metric, a headroom-preserving A0 baseline, and a volume-level held-out Dice. None of the three exists at HEAD.

**Finding counts:** **P0 = 4**, **P1 = 7**, P2 = 12, P3 = 6.

---

## 2. As-built architecture

Reconstructed from executable code only (`src/self_audit/models/self_audit_net.py`), not from docs or comments.

### 2.1 Deployable path — `SelfAuditNet.infer(images, mode="self_audit")`

```
Image x  [B,3,256,256]                       (2.5-D: 3 neighbouring slices as RGB channels)
   |
   v  encoder.py:113-117   ConvNeXt-Tiny features_only, out_indices (0,1,2,3)
F0 [B,96,64,64]  F1 [B,192,32,32]  F2 [B,384,16,16]  F3 [B,768,8,8]
   |
   v  fpn.py:45-54         LightweightFPN, top-down add, ONLY refined[0] is returned
shared H  [B,96,64,64]                        (stride 4)
   |
   +-------------------------------------------------------------+
   |                                                             |
   v  annotation_head.py:22-27                                   |
A0 = Up_bilinear(Conv1x1(GELU(GN(Conv3x3(H)))), 256x256)          |
   [B,4,256,256]  LOGITS                                         |
   |                                                             |
   |   state <- A0 ;  previous_audit <- None                     |
   |                                                             |
   +--> FOR turn t = 0 .. cap-1, ONLY over rows with active=True: |
        |                                                        |
        |  annotation_expert.py:130-156   ONE shared module      |
        |    S0    = GELU(GN(W_in @ [ H | Up(A_t) | Up(ent(A_t)) | Up(U_t) ]))   104->96 ch
        |    depth = min(t+1, 3)                       # HARD-CODED cap 3
        |    for i in 0..depth-1:                                |
        |        S   = S + R( DWA(S; cond=U_t, turn=t, iter=t+i) )  # SAME weights
        |    delta = Up(Conv(S)) ; gate = Up(Conv1x1(S))         |
        |    A_cand^(t) = A_t + sigmoid(gate) (*) delta          |   <-- candidate
        |                                                        |
        |  auditor.py:76-116   ALL INPUTS DETACHED  <-------------+
        |    input = [ H.detach() | P_prev | P_cand | dP | ent_prev | ent_cand ]   110 ch
        |            (P_* bilinear-downsampled to 64x64, stride 4)
        |    z     = Residual x3 ( GELU(GN(Conv1x1(input))) )
        |    local_logits = Up( Conv1x1(z) )   [B,3,256,256]   {FIX,UNCHANGED,REGRESS}
        |    delta_q      = MLP( GAP(z) )      [B,1]           predicted signed dDice
        |
        |  gate  (self_audit_net.py:204)   accepted = delta_q.detach() > tau_accept
        |
        |  state          = where(accepted, A_cand, A_t)        # reject keeps A_t
        |  previous_audit = where(accepted, softmax(local_logits).detach(), 0)
        |  halt_turn      = first t where a row is rejected
        |  active         = accepted                            # HALT-ON-FIRST-REJECT
        v
   final logits = state   [B,4,256,256]
```

Three facts about this diagram that differ from the intended design and are easy to miss:

* The **auditor consumes `H`, never the raw image**. Its evidence about anatomy is entirely second-hand through the FPN feature map, at stride 4.
* **Rejection is terminal.** `active = accepted` (`self_audit_net.py:249-250`) means the first reject halts that row permanently. The intended design says "optionally halt"; the implementation always halts.
* **`previous_audit_evidence` is only ever non-zero in `infer`.** `forward_annotation` sets `audit_evidence = None` at `self_audit_net.py:64` and never reassigns it, so during **all of Phase A and all of Phase B** the expert's 3 audit channels are hard zeros. See P1-7.

### 2.2 Three training paths, as built

```
PHASE A  train_annotation.py
  forward_annotation(x)  ->  [A0, A1, A2, A3]      (no auditor, no gate, audit channels = 0)
  loss = SUM_k w_k * (softDice_withBG(A_k, GT) + CE(A_k, GT)) / SUM w_k
  w = [0.5, 0.7, 0.8, 1.0]  ->  effective [0.167, 0.233, 0.267, 0.333]
  model selection: Dice of A3 only        (train_annotation.py:239-242, :354)

PHASE B  train_auditor.py
  freeze: requires_grad = name.startswith("auditor")     (train_auditor.py:52-64)
  forward_annotation under no_grad -> trajectory [A0..A3]
  transitions = 3 on-policy adjacent pairs  +  4 GT-synthesized counterfactuals
  loss = mean over 7 groups of ( CE_local + signed_ranking_loss(delta_q, dDice) )
  model selection: AUROC on the SAME 7-group mixture   (train_auditor.py:428-429)

PHASE C  finetune_joint.py
  output = infer(x, mode="self_audit", tau=0.0, t_max=3)          # gated, on-policy
  annotation_term = softDice+CE( final_state , GT )               # FINAL STATE ONLY
  audit_term      = mean over attempted transitions of audit_loss( detached pair , GT )
  total = annotation_term + lambda_audit * audit_term    (lambda_audit = 1.0)
  optimizer: encoder_head_optimizer -> encoder @1e-6, EVERYTHING ELSE (incl. auditor) @1e-5
```

---

## 3. Module-by-module audit

| # | Module | File | Mathematically does | Receives | Indirect influence | Updated by | Score |
|---|---|---|---|---|---|---|---|
| A | `ConvNeXtTinyEncoder` | `models/encoder.py:73-117` | ImageNet ConvNeXt-Tiny, 4 pyramid levels | `[B,3,H,W]` only | none | Phase A (lr 3e-5), Phase C (lr 1e-6); frozen Phase B | **MATCH** |
| B | `LightweightFPN` | `models/fpn.py:19-54` | lateral 1×1 + top-down add + refine; **returns `refined[0]` only** | 4 encoder maps | none | as encoder | **PARTIAL** — `refine[1..3]` (12 params, 249,696 of 554,784 measured in fallback config) are computed and thrown away; dead weight with `grad is None` in every phase (`fpn.py:53-54`) |
| C | `InitialAnnotationHead` | `models/annotation_head.py:10-28` | Conv3×3→GN→GELU→Conv1×1, bilinear ×4 to full res | `shared` only | **92.2% of its gradient comes from A1–A3** | Phase A (all four stage losses), Phase C (final-state loss when the gate rejects at t=0) | **RESEARCH RISK** — see §7 |
| D | `AnnotationExpert` | `models/annotation_expert.py:43-158` | `A_{t+1} = A_t + σ(gate)⊙Δ`, one shared block, depth `min(t+1,3)` | `shared`, `A_t`, entropy(`A_t`), `previous_audit_evidence` | audit evidence at inference only | Phase A (A1–A3), Phase C (final state) | **PARTIAL** — sharing is genuine (verified: one module object, 6 invocations over 3 turns, `unique ids: 1`) but see E |
| E | `DynamicWindowAttention` / `DynamicWindowGenerator` | `models/dynamic_window.py:64-65` | learned window/support generation, then windowed attention | `S`, `cond=U_t`, `turn_index`, `iteration_index` | — | with D | **RESEARCH RISK** — holds **per-turn and per-iteration `nn.Embedding` tables** indexed by the stage counters. Weights are shared, but the *conditioning* is stage-indexed, which is a soft violation of "one shared expert reused recurrently", and indices alias under clamping |
| F | `CounterfactualAuditor` | `models/auditor.py:44-116` | 110-ch trunk at stride 4 → per-pixel 3-way logits + scalar Δq | `H.detach()`, `P_prev`, `P_cand`, `ΔP`, 2 entropies — **all detached** | none; never sees the image or GT | Phase B (`auditor_lr` 3e-4), Phase C (bundled at 1e-5) | **PARTIAL** — correct transition semantics and airtight firewall, but resolution-starved local head (see §8) |
| G | `CounterfactualGenerator` | `audit/counterfactual.py:252-586` | 1 positive (GT-blend repair), 7 negative morphological ops, 1 hard-neutral combination | `previous_probs`, **GT** | — | not a parameterized module | **RESEARCH RISK** — training-only (verified: sole importers are `train_auditor.py:21`, `train_self_audit.py:26`), but the positive branch is largely degenerate at realistic confidence (§9) |
| H | `ThresholdGate` / `accept_reject` | `audit/gate.py:19-93` | `accepted = Δq.detach() > τ`; state keep-or-replace; halt on reject | Δq only | — | non-differentiable | **PARTIAL** — matches `infer`'s inline logic exactly (verified line-for-line), but **local evidence never enters the accept decision**, contrary to the intended "local evidence + global ΔQ → ACCEPT/REJECT". `ThresholdGate.run` is also dead code — `infer` reimplements it inline |
| I | `SelfAuditNet.infer` | `models/self_audit_net.py:96-296` | 4 modes; row-masked recurrence; halt-on-first-reject | image (+ `oracle_target` in analysis mode) | — | — | **PARTIAL** — deployable path is GT-free and correct; `oracle_target` is accepted in *every* mode with zero validation (`:104`), and `forward()` splats `**kwargs` into it (`:359-360`) |

---

## 4. Tensor / state flow

Measured on `images = [2,3,256,256]`, `shared_channels=96`, `num_classes=4`, `window_k=8`, `max_turns=3`.

| Stage | Tensor | Shape | Grad-carrying? | GT dependency |
|---|---|---|---|---|
| input | `images` | `[2,3,256,256]` | leaf | none |
| encoder | `F0..F3` | `[2,96,64,64]`, `[2,192,32,32]`, `[2,384,16,16]`, `[2,768,8,8]` | yes | none |
| FPN | `shared` | `[2,96,64,64]` | yes | none |
| head | `A0` (logits) | `[2,4,256,256]` | yes | none |
| expert in | `[H \| Up(A_t) \| Up(ent) \| Up(U_t)]` | `[2,104,64,64]` | yes | none |
| expert out | `delta`, `gate` | `[2,4,256,256]`, `[2,1,256,256]` | yes | none |
| candidate | `A_cand^(t)` | `[2,4,256,256]` | yes | none |
| auditor in | concat | `[2,110,64,64]` | **no — all detached** | none |
| auditor out | `local_logits` | `[2,3,256,256]` (trunk at 64×64, upsampled) | yes (auditor params only) | none |
| auditor out | `delta_q` | `[2,1]` | yes (auditor params only) | none |
| gate | `accepted` | `[2]` bool | **no — discrete** | none |
| state | `A_{t+1}` | `[2,4,256,256]` | yes | none |
| targets | `TransitionTargets.local` | `[2,256,256]` int | no | **yes — built after `infer` returns** |
| targets | `TransitionTargets.delta_dice` | `[2,1]` | no | **yes** |

Parameter counts (fallback encoder — ConvNeXt numbers will differ): encoder 7,906,272 · FPN 554,784 · initial head 41,812 · annotation expert 210,250 · auditor 514,660. The recurrent expert is **0.4% of the fallback model** and the auditor 1.1%.

---

## 5. Gradient-flow analysis

All entries below are from executed `backward()` calls, not from reading comments.

### 5.1 The required table

Y = receives non-`None` gradient from that loss alone; `-` = frozen or structurally disconnected.

| Loss | Encoder | FPN | A0 head | AnnExpert | Auditor | Evidence |
|---|---|---|---|---|---|---|
| Phase-A A0 loss | Y `2.16e-1` | Y `2.13e-1` | Y `3.10e-1` | **0** | — | `train_annotation.py:93,98` |
| Phase-A A1 loss | Y `2.15e-1` | Y `2.59e-1` | **Y `5.08e-1`** | Y `1.70e-1` | — | via `self_audit_net.py:76` (`state` un-detached) |
| Phase-A A2 loss | Y `2.50e-1` | Y `3.45e-1` | **Y `6.83e-1`** | Y `5.54e-1` | — | " |
| Phase-A A3 loss | Y `3.87e-1` | Y `5.13e-1` | **Y `8.24e-1`** | Y `1.24e+0` | — | " |
| Phase-B audit local | — (frozen) | — | — | — | Y | `train_auditor.py:56` + `auditor.py:91-94` |
| Phase-B audit ΔQ | — | — | — | — | Y (34× weaker than local at equal weight) | " |
| Phase-C annotation | Y | Y | Y (only when the gate rejects at t=0) | Y | **None (0/28 params)** | `finetune_joint.py:207` |
| Phase-C audit | **None (0/120 params)** | None | None | None | Y | `finetune_joint.py:242-243` + `auditor.py:91-94` |

### 5.2 The A3 → A0 chain is ACTIVE

`forward_annotation` assigns `state = expert_output.candidate_logits` with **no detach** (`self_audit_net.py:76`) and feeds it into the next turn (`:70`). So

```
L(A3) -> AnnExpert(turn 2) -> A2 -> AnnExpert(turn 1) -> A1 -> AnnExpert(turn 0) -> A0 -> initial_head -> FPN -> encoder
```

is a single unbroken autograd path. **Executed proof:** last-stage-only backward gives `initial_head = 8.244e-1`, `fpn = 5.135e-1`, `encoder = 3.873e-1`; `d(final)/d(A0)` absolute sum measured at **33,250**. This is design-as-intended for a deeply-supervised refinement net, and it is the mechanism behind P1-2.

### 5.3 Firewall — verified in both directions

* Backprop **audit loss only** → all **120** annotation-side parameters have `grad is None`.
* Backprop **annotation loss only** → all **28** auditor parameters have `grad is None`.

Enforced redundantly at three independent points: `auditor.py:91-94` (defensive internal detach), `self_audit_net.py:184-189` (call-site detach), `finetune_joint.py:242-243` (target-side detach). **No P0/P1 leak.** This is the strongest part of the codebase.

---

## 6. Phase A / B / C analysis

### Phase A (`train_annotation.py`)

* Loss: `phase_a_loss` (`:82-103`) supervises **A0, A1, A2, A3 all directly against GT**, unconditionally, every step (`:93`, `:98`).
* Weights: `DEFAULT_STAGE_WEIGHTS = (0.5, 0.7, 0.8, 1.0)` (`:51`), identical to `configs/self_audit_annotation.yaml:29`. Normalized by `/Σw` (`:99`) ⇒ effective **A0 0.167 · A1 0.233 · A2 0.267 · A3 0.333**. No ramp, no decay, no epoch schedule.
* Per-stage leaf loss is `softDice + CE` with **`include_background=True`** (`losses/annotation.py:33`, `:10`) — background is ~95% of cardiac MRI pixels and scores ~0.99, so the training Dice signal on the three foreground classes is diluted ~4× relative to every *evaluation* Dice, all of which exclude background.
* Model selection uses **A3 only** (`:239-242`, `:354`). Nothing anywhere selects on A0 quality.
* `resolve_stage_weights` **silently discards** a configured list of the wrong length (`:63-70`) — a latent ablation-corrupting trap.

### Phase B (`train_auditor.py`)

* Freezing is correct and complete: `parameter.requires_grad = name.startswith("auditor")` (`:56`), plus `.eval()` on encoder/FPN/head/expert (`:57-64`), plus `model.eval(); model.auditor.train()` in the epoch loop (`:254-255`). Frozen params are also excluded from the optimizer (`:492`). No BatchNorm/dropout drift.
* Annotation forward runs under `torch.no_grad()` (`:164-166`).
* **Transitions per batch: 3 on-policy adjacent pairs + 4 GT-synthesized counterfactuals** (one around every state including A0) — `:97-127`. Verified at runtime.
* Loss is a **uniform mean over the 7 groups** (`:217`) ⇒ synthetic carries **57.1%** of the weight.
* **Validation calls the exact same `_auditor_batch` with the same generator instance** (`:392-402`, `main` passes it at `:522` and `:530`). See P1-1.
* Checkpoint loading is `strict=True` with architecture validation (`_utils.py:888`, `:906`, `:870-878`) — good — but there is **no phase-tag check**, so a Phase-A checkpoint fed to Phase C silently runs `mode="self_audit"` with a **random-init auditor** at τ=0.

### Phase C (`finetune_joint.py`)

* Runs the real gated `infer(mode="self_audit")` (`:197-202`) — genuinely on-policy, **zero** synthetic transitions. Correct.
* Annotation term supervises **only the final retained state** (`:207`); A0 is no longer directly supervised. Note that when the gate rejects at t=0 the final state *is* A0, so A0 supervision returns implicitly and data-dependently.
* Audit term uses explicitly detached transition pairs (`:242-243`) with GT-derived targets built **after** `infer` returned. Correct ordering.
* `total = annotation_term + 1.0 * audit_term` (`:255`, `configs/self_audit_joint.yaml:31`).
* Optimizer is `encoder_head_optimizer` (`:` main), which has **only two groups: encoder and "everything else"**. The auditor lands in "everything else" at `joint_lr = 1e-5`, a silent 30× drop from its Phase-B `3e-4`.
* `validate_phase_c` (`:307-471`) is the **cleanest evaluation in the repo**: on-policy only, targets built post-inference, and it reports `initial_foreground_macro_dice` vs `final_foreground_macro_dice` per-slice.

---

## 7. A0 / headroom analysis  *(high-priority section)*

**Is A0 explicitly encouraged to become an optimal segmentation by itself?** Yes — effective weight 0.167 on a direct `softDice+CE(A0, GT)` term, every step, `train_annotation.py:93,98`.

**Is A0 also indirectly optimized by later-stage losses?** Yes, and this dominates. Executed decomposition of the gradient arriving at `initial_head`:

| Source of `initial_head` gradient | Share |
|---|---:|
| A0's own supervision term | **7.8%** |
| A1 + A2 + A3 backpropagating through the shared expert | **92.2%** |

**Could the training objective collapse correction headroom?** Yes, and here is the exact mechanism, stated as a falsifiable claim rather than a worry:

1. The objective is `Σ_k w_k · L(A_k, GT)`. It contains **no term that rewards `A_k` for being *improvable*.** Every stage is pulled toward the same fixed point — GT.
2. `A_{t+1} = A_t + σ(gate)⊙Δ` is a residual update. The cheapest way for the optimizer to reduce a sum of losses over a residual chain is to make the *first* element already near-optimal and drive `Δ → 0`. Lowering `w_0` does not change this, because 92.2% of the pressure on A0 arrives through the chain, not through `w_0`.
3. The measured consequence would be: `candidate_gain ≈ 0`, `positive_headroom ≈ 0`, and therefore `OracleHeadroom ≈ 0` — at which point the Auditor has nothing to capture and `HeadroomCapture` is `0/0`.

**Scientific implication of "just weaken A0" — do not do it blindly.** Reducing `w_0` (or removing A0 supervision) does *not* create headroom, because of (2) above; it only makes A0 a worse-conditioned starting point. It would also make the "self-audit" gain trivially large and scientifically vacuous — a reviewer will immediately ask whether you handicapped the baseline to manufacture the delta. The defensible interventions are the ones that change *what headroom means*, not ones that damage A0:

* **Report A0 against a matched single-stage baseline** — the same encoder/FPN/head trained with `stage_weights = [1.0]` and no expert. If that baseline matches the deep-supervised A0, then A0 is not artificially strong and the headroom question is honest.
* **Create headroom from distribution shift rather than from handicap** — the ACDC→M&Ms protocol already in `configs/self_audit_acdc_to_mnms.yaml` is the natural place where A0 is genuinely imperfect and refinement has real work to do. This is the strongest available framing and it is currently unused.
* **Measure headroom before claiming it.** `scripts/audit_checkpoint.py` already computes exactly the right quantities. It has apparently never been run on a trained checkpoint.

**Third, architectural, contributor to collapse (P1-7):** the expert is trained in Phase A and Phase B with `previous_audit_evidence = None` (hard zeros, `self_audit_net.py:64`) but is *fed real audit evidence at inference* (`:217-221`). Three of its 104 input channels change distribution between training and deployment. Additionally `gate_head` initializes at `σ(0)·` — measured mean `σ ≈ 0.38`, i.e. a ~16% edit magnitude at step 0, not near-identity — and can never emit an exact null update, so `A_cand == A_t` (Δ=0 exactly) is architecturally unreachable while being the modal on-policy outcome the metrics assume.

---

## 8. Auditor contribution analysis

**What is it actually learning?** A genuine transition property, not candidate quality. Executed control (222 generated transitions, heterogeneous `previous` quality):

```
corr(dDice, Dice(candidate))                  = -0.0535
corr(dDice, -Dice(previous))                  = +0.3473
AUROC(score=Dice(candidate), label=dDice>0)   =  0.5251   <- chance
```

An auditor that ignored `P_previous` and merely regressed candidate quality would score **at chance**. `P_previous` is load-bearing. This is a real positive finding and it should be reported in the paper as an ablation.

**But there is a competing shortcut.** Six hand-crafted scalars computed **only from `ΔP` and the two entropy channels** — no image features, no anatomy, no GT — fed to a logistic regression reach **AUROC 0.82** on the generator's own distribution. Since the generator's edits have characteristic `ΔP` signatures (a morphological erosion looks nothing like a GT-blend repair), a large share of the reported Phase-B AUROC may be **generator-fingerprint detection**, a skill worth exactly nothing at inference.

**Local head is resolution-starved (P1-5).** The trunk runs at stride 4 (`auditor.py:97-100, 111-113`) and `local_logits` is upsampled to full resolution only at the end (`:114`), while `local_audit_targets` is built at **full pixel resolution** (`targets.py:55-70`). Oracle upper bound — take the *true* target, area-downsample to 64×64, bilinear-upsample back, argmax:

| Kind | Class | Target px | Oracle-through-stride-4 recall |
|---|---|---:|---:|
| positive | FIX | 1 (0.0004%) | **0.0000** |
| negative | REGRESS | 2,717 (1.04%) | 0.9779 |
| hard_neutral | FIX | 153 (0.058%) | 0.1830 |
| hard_neutral | REGRESS | 112 (0.043%) | 0.1518 |

Thin FIX/REGRESS bands — precisely the "local correction evidence" the contribution is about — are **not representable at the resolution the head runs at**. Bulk regressions (component deletion, class swap) survive. `local_fix_f1` is close to structurally unlearnable, and the UNCHANGED prior is 99.6–99.9% with inverse-frequency weights clamped at 5.0 (`train_auditor.py:141`).

**Target semantics MISMATCH vs the intended spec.** `local_audit_targets` (`targets.py:65-69`) sets `UNCHANGED` as the default and only overrides on a correctness *flip*. So "previous wrong, candidate wrong in a *different* way" is labelled UNCHANGED, identically to "both correct". There is no epsilon and no third distinction. The intended `FIX / UNCHANGED-NEUTRAL / REGRESS` triple is therefore not what is trained.

**Gate MISMATCH vs the intended spec.** The accept predicate is `delta_q > tau_accept` (`self_audit_net.py:204`, `gate.py:22`) — **global ΔQ only**. The local evidence head never enters the decision. The intended design says local evidence *and* global ΔQ drive ACCEPT/REJECT. As built, the entire local head is a diagnostic/auxiliary loss with no path to the decision.

**Attribution algebra — verified correct, could not be falsified.** `decompose_self_audit_output` (`audit_decomposition.py:160-250`) implements exactly `realized = Δ·accepted`, `gate_value = realized − Δ`. Hand-computed 2-row example executed through the real function: rejecting a harmful candidate gives **+0.1667**, rejecting a beneficial candidate gives **−0.1667**, attribution residual `< 1e-7`. All four modes (`initial_only`, `always_accept_refinement`, `self_audit`, `oracle_accept`) exist and are computed, and `OracleHeadroom`, `SelfAuditGain`, `AuditRescue`, `HeadroomCapture` are all present and logged. **Item 5 of the brief is not a measurability gap** — the diagnostics are there and are correct in sign. The problem is the estimator used for `HeadroomCapture` (P0-3) and the fact that nobody appears to have run them.

---

## 9. Synthetic vs on-policy analysis

**The number: synthetic : on-policy = 4 : 3. Synthetic carries 57.1% of the Phase-B loss weight.** Identical per training step and per validation step.

Derivation, verified at runtime with a real `SelfAuditNet` (`max_turns=3`, batch 2):

```
state_trace len: 4
transitions: 7  {'on_policy': 3, 'synthetic': 4}
  on_policy turn 0 / 1 / 2
  synthetic:negative:hole_insertion                        turn 0
  synthetic:negative:semantic_class_swap                   turn 1
  synthetic:hard_neutral:local_repair_plus_local_erosion   turn 2
  synthetic:negative:mixed                                 turn 3
```

`build_auditor_transitions` (`train_auditor.py:88-128`) emits `len(trajectory)-1 = 3` on-policy pairs and `len(trajectory) = 4` synthetic samples — one around **every** state including A0. The per-batch loss is `torch.stack(losses).mean()` (`:217`), a uniform mean over groups.

**Phase-B validation is contaminated.** `validate_auditor_epoch` calls the identical `_auditor_batch` with the identical generator instance (`:392-402`, `main` at `:522`/`:530`). Consequences:

1. `val/auroc`, `val/auprc`, `val/local_fix_f1`, `val/local_regress_f1`, `val/correlation_delta_q`, `val/improve_regress_accuracy` are all computed on the 57.1%-synthetic mixture (`:550-573`).
2. `primary_metric` — which selects `best.pt` **and** the exported `phase_b_auditor.pt` (`:428-429`, `:540-549`) — is AUROC on that mixture. Model selection is partly optimizing for generator-fingerprint detection.
3. The Phase-B visualization path uses the same mixture (`:588`), so qualitative figures are not on-policy either.

**The positive branch is largely degenerate at realistic confidence.** `_positive_single` blends `candidate = previous·(1−m) + onehot(GT)·m` with `m ~ U(0.3, 0.6)` (`counterfactual.py:316-320`). Flipping a confidently-wrong pixel needs `m ≳ 0.5`; only ~34% of that mass clears it. Measured through the repo's own generator (80 samples per row):

| `previous` p_max | frac positive-CF changing **zero** pixels | frac inside `\|dDice\| < 0.005` | mean dDice |
|---:|---:|---:|---:|
| 0.475 | 0.000 | 0.738 | +0.00632 |
| 0.711 | 0.287 | 0.863 | +0.00343 |
| 0.948 | **0.575** | 0.912 | +0.00175 |
| 0.993 | **0.675** | 0.912 | +0.00225 |
| 1.000 | **0.675** | 0.950 | +0.00138 |

A trained network operates at p_max ≈ 0.95–1.0. In that regime **58–68% of "positive" counterfactuals change zero pixels** (and are still returned `valid=True`, `:332`, with no check that the argmax actually moved), and **91–95% land inside the loss's own neutral band**, where `signed_ranking_loss` actively pulls their predicted Δq toward **0** (`losses/audit.py:36`). Meanwhile the gate accepts on strict `Δq > 0`. **The training signal parks genuine repairs exactly on the reject side of the decision boundary.** Measured class prior over the generator's own mixture: `P(dDice > 0) = 0.230` — the curriculum is **~77% "do not accept"**. Expected count of genuinely-improving synthetic transitions: **≈ 0.5 of 7 groups per batch (~7%)**.

Mechanistically this predicts a gate that rejects nearly everything at τ=0, i.e. `self_audit_dice ≈ initial_dice`, while still producing a respectable AUROC. That is precisely the failure mode that would falsify the central hypothesis.

**Phase C, by contrast, is 100% on-policy** (`finetune_joint.py:220-256`). Zero synthetic transitions.

**What is missing:** no code path anywhere evaluates the auditor on **on-policy transitions only**. `_auditor_batch` has no flag; `build_auditor_transitions` has no parameter. The `provenance` field is already carried in the transition dicts (`:103`, `:123`) and is simply never used for reporting. `audit/on_policy/*` and `audit/synthetic/*` namespaces do not exist.

---

## 10. GT-leakage audit

### VERDICT: **NO GT leakage in deployable inference.**

Proven three independent ways.

**(a) Static call-graph.** Exactly **three** call sites in the entire repo pass anything GT-derived into a model, and all three name it `oracle_target`:

| path:line | Symbol | mode | Classification |
|---|---|---|---|
| `evaluation/audit_decomposition.py:282-287` | `evaluate_audit_modes_batch` | `oracle_accept` | DIAGNOSTIC-ONLY (headroom ceiling) |
| `evaluation/audit_decomposition.py:416-429` | `probe_gt_leakage` | `self_audit` ×2 | DIAGNOSTIC-ONLY, value provably inert |
| `evaluation/volume_inference.py:294-300` | `evaluate_comparison_modes` | `oracle_accept` | VALIDATION/ANALYSIS (zero callers) |

Every other model entrypoint call site passes **only** `batch["image"]`. `CounterfactualGenerator`'s sole importers are `training/train_auditor.py:21` and `scripts/train_self_audit.py:26` — no evaluation or inference module imports it, so counterfactual generation is genuinely training-only.

**(b) TEST 1 — GT permutation invariance (executed).** Same seed, same model, same images; two different `oracle_target` tensors present in the surrounding evaluation code; `mode="self_audit"`:

```
A0 logits         max-abs-diff = 0.0   (bitwise identical)
candidates        max-abs-diff = 0.0
delta_q           max-abs-diff = 0.0
accepted masks    identical
final logits      max-abs-diff = 0.0
halt_turn         identical
```

Stronger: passing a **Python `str`** as `oracle_target` in `mode="self_audit"` raises nothing — the value is never dereferenced.

**(c) Positive control — the probe has power.** The same GT pair under `mode="oracle_accept"` **flips every accept decision** and moves final logits by max-abs **0.5098**. So the zero-difference in (b) is a real invariance, not a dead test.

**TEST 2 — Phase-B annotation invariance (executed).** Changing GT does not change A0, A1, A2 or A3 from `forward_annotation` (max-abs-diff 0.0). Correct: `forward_annotation` receives only `batch["image"]`, and Phase B runs it under `no_grad` before any target is built.

**TEST 3 — gradient firewall (executed).** See §5.3. Holds in both directions, 120/120 and 28/28.

**TEST 4 — API boundary.** `SelfAuditNet.infer(..., oracle_target=None)` accepts `oracle_target` in **every** mode with zero validation (`self_audit_net.py:104`, only checked at `:116-117` for the *absence* case in `oracle_accept`), and `forward()` splats `**kwargs` straight into it (`:359-360`). Classification: **not leakage, but an API/research-clarity risk** — this is one `if` edit away from real leakage, and a reviewer reading `infer`'s signature will legitimately ask why a deployable inference API takes a ground-truth argument at all.

**Recommendation:** move `oracle_accept` out of `SelfAuditNet.infer` entirely into evaluation code. `evaluate_audit_modes_batch` can compute the oracle trajectory itself from `always_accept_refinement`'s per-turn candidates plus GT Dice, without the model ever seeing a target. This makes the GT-free property *structural* rather than *behavioural*.

**Two adjacent risks, both currently latent:**

* `VolumeSliceDataset(foreground_only=...)` (`data/common.py:335`) performs **GT-based slice selection**. It is dead — `_utils.py:285-287` omits it from the kwarg allowlist and no config sets it — but enabling it at validation would be a genuine protocol-inflation leakage class.
* Threshold calibration consumes GT (`finetune_joint.py:519,536` → `threshold.py:55-104`), correctly classified as CALIBRATION, but is run and *reported* on the same split — see P0-4.

---

## 11. Metric / protocol audit

### 11.1 Why Dice may be ~0.97 — ranked causes

The split itself is **clean**, so protocol inflation, not leakage, is the explanation.

**Verified clean.** `splits/acdc_patient_split_seed42.json` is genuinely patient-level: 80 train / 20 val patients, 160/40 volumes, `overlap []`, `patients whose phases straddle splits: {}`, every patient has exactly `{ED, ES}`. Generation groups by patient prefix and shuffles patients, not volumes (`scripts/acdc_split.py:62-116`); consumption re-validates and raises on cross-split patients (`data/common.py:216-228`). **ED/ES straddle — the most common ACDC leakage mode — is not present.** There is no GT-guided cropping anywhere in the pipeline.

**Ranked inflation sources:**

| # | Cause | Direction | Rough magnitude | Evidence |
|---|---|---|---:|---|
| 1 | **Empty-empty → 1.0** in *every* Dice in the repo, at 2-D slice granularity | ↑ | reported ≈ `(1−f)·D + f`; at f=0.20, D=0.85 ⇒ **+0.03** | `metrics.py:58`, `targets.py:81-84`, `self_audit_net.py:355`, `visualizer.py:183` |
| 2 | **2-D slice metric, not 3-D volume** — apical/basal slices with absent RV/MYO each score 1.0 per absent class | ↑ | a slice where the model is *totally wrong* still scores **0.667** (measured) | no volume-level Dice is ever computed |
| 3 | **Resized 256×256 grid vs twice-resampled GT** — Dice is computed in network space, never in native ACDC geometry | ↑ (smoothing) | small but real | `evaluate_comparison_modes` is the only native-geometry path and has **zero callers** |
| 4 | **`foreground_only` never enabled** ⇒ fully-empty slices are included and score exactly 1.0 | ↑ | folds into #1 | `_utils.py:285` |
| 5 | Phase-A headline is **batch-pooled** (micro-average over the whole batch array) while Phase-C is **per-slice macro** | ↕ | executed demo: **0.6825 vs 0.8333** on identical inputs; Phase A is `batch_size`-dependent | `train_annotation.py:242` vs `finetune_joint.py:286` |
| 6 | ED/ES pooled rather than reported per-phase | ↕ | ACDC convention reports both | — |
| 7 | ImageNet-pretrained ConvNeXt-Tiny (`pretrained_encoder: true`) | ↑ | legitimate, but must be declared | `configs/self_audit_annotation.yaml:35` |

**CLASSIFICATION: internal-validation-only.** Not officially comparable, not partially comparable. Published ACDC numbers (~0.90–0.92 mean over RV/MYO/LV) are **volume-level, native-geometry, on the held-out test set**. This repo reports **slice-level, resized, empty-inclusive, on validation, with no test split**. Any SOTA claim from the current number would not survive review.

### 11.2 Neutral-margin consistency — exhaustive

Three mutually inconsistent epsilons govern the same decision:

| Constant | Value | Declared | Consumed by |
|---|---:|---|---|
| `epsilon_neutral` | **0.02** | `configs/self_audit_auditor.yaml:27` | `counterfactual.py:282-289,477,485` |
| `neutral_margin` | **0.005** | `configs/self_audit_auditor.yaml:29`, `self_audit_joint.yaml:32` | `losses/audit.py:20`, `audit_decomposition.py:185-186,481-482` |
| (implicit) | **0.0** | — | all 8 sites below |

Both `epsilon_neutral: 0.02` and `neutral_margin: 0.005` sit in the *same config block*. A synthetic transition with `dDice = 0.01` is generated and labelled NEUTRAL by the generator, then trained by `signed_ranking_loss` as a **strictly beneficial ordering example** (`audit.py:20,23`). The auditor is taught to rank a transition it was told is neutral.

Every site where zero is the boundary while training uses a nonzero margin:

| # | Site | Code | Used for |
|---|---|---|---|
| 1 | `metrics.py:286` | `predicted_sign = delta_q > 0.0` | improve/regress accuracy |
| 2 | `metrics.py:287` | `actual_sign = delta_dice > 0.0` | **AUROC/AUPRC positive class** |
| 3 | `metrics.py:316` | `harmful = accepted & (delta <= 0.0)` | `harmful_acceptance_rate` — **Δ==0 counted harmful** |
| 4 | `metrics.py:317` | `beneficial_rejection = ~accepted & (delta > 0.0)` | asymmetric with #3 |
| 5 | `threshold.py:85` | `accepted & (actual <= 0.0)` | **steers τ selection** |
| 6 | `threshold.py:86` | `rejected & (actual > 0.0)` | same asymmetry |
| 7 | `self_audit_net.py:346` | `candidate_score > previous_score` | **the `oracle_accept` gate itself** ⇒ defines `OracleHeadroom` |
| 8 | `audit_decomposition.py:306` | `oracle_headroom > 1e-8` | `headroom_capture_ratio` denominator |

Margin-correct sites, for completeness: `audit_decomposition.py:185-186,481-482` (the only ones), and `losses/audit.py:20` *provided the caller passes a margin* — its signature default is `0.0` (`audit.py:16,74`), and all current callers do pass it.

**Executed damage:**

```
acceptance_metrics(accepted=[T,T,T,T,T,F], delta=[0, 0, 0, +1e-9, -0.20, +0.20])
  -> harmful_acceptance_rate = 0.8      (4 of 5 accepted flagged harmful; ONE was actually harmful)
evaluate_threshold(tau=0.0, 3 accepted transitions all with delta == 0.0 exactly)
  -> harmful_acceptance_rate = 1.0
```

Exact-zero deltas are **not rare**: any transition whose candidate argmax equals the previous argmax gives `delta == 0.0` bit-exactly, and on a fully-empty slice that is the common case. Since `select_threshold` (`threshold.py:135-149`) uses `harmful_acceptance_rate` as its tie-break, **the calibrated τ is systematically pushed more conservative than the project's own neutral semantics warrant.** And because `metrics.py:287` defines the AUROC positive class at `> 0.0`, **the reported AUROC measures a task the loss was never trained on.**

### 11.3 The headroom-capture estimator is invalid (P0-3)

`audit_decomposition.py:305-308`:

```python
positive_headroom = oracle_headroom > 1e-8
capture[positive_headroom] = self_gain[positive_headroom] / oracle_headroom[positive_headroom]
```

This is a **mean of per-sample ratios** with a `1e-8` guard. A single row with `oracle_headroom = 2e-8` drives the mean to ~1e6. Executed demo returns **625000.4** where the correct ratio-of-means is **0.667**. Given that §7 predicts `oracle_headroom ≈ 0` for almost every row, this estimator will be dominated by exactly the rows it should exclude. It must be a ratio of means (Σ self_gain / Σ oracle_headroom) over rows where headroom is materially positive (e.g. `> neutral_margin`).

### 11.4 Other measurement defects

* `evaluate_audit_decomposition` aggregates as an **unweighted mean over batches** of per-batch rates and silently drops NaN batches (executed: **0.5625 vs 0.2222** correct value on the same data).
* `stage_t/oracle_gain` is a plain **alias of `stage_t/headroom`** measured on the *self-audit* trajectory (`audit_decomposition.py:196-197`) — it is not the oracle path. Mislabeled.
* `phase_a/headroom_collapse_ratio` divides by a **hardcoded 0.005**, ignoring the `neutral_margin` argument (`:496-498`).
* `gt_firewall/*` probe runs on **batch 0 only**.
* `SelfAuditNet._dice` hardcodes classes `(1,2,3)` (`:350`); for a 2-class model (and `scripts/prepare_mnm_binary.py:106` produces exactly that) it silently returns `(real + 1.0 + 1.0)/3`.
* W&B: no `audit/on_policy/*` vs `audit/synthetic/*` split; namespacing is inconsistent across phases; non-finite metrics are silently dropped.
* `validate_dataset_splits` (`_utils.py:369-379`) — the gate printed as `phase_*_split_stats` — finds no `test` key in the manifest, **discards the whole manifest**, and substitutes an 80/10/10 `patient_level_split` built with a *different RNG*. Executed: **15 of the 20 real validation patients are labelled `train` by the gate.** This is *not* train/val leakage (the DataLoaders use the correct manifest via `build_patient_dataset`), but any paper table sourced from those printed counts would be wrong.
* Halted-row masking inside a single batch is **correct** (`active_mask`/`state_mask` are threaded properly); the aggregation bug is one level up, across batches.

---

## 12. Actual vs intended design

| Component / behavior | Intended | Actual | Difference | Scientific consequence | Sev | Evidence |
|---|---|---|---|---|---|---|
| Shared Annotation Expert | one module reused recurrently | one module, verified by object identity (`unique ids: 1`, 6 invocations / 3 turns) | none | — | OK | `self_audit_net.py:42-47,67-75` |
| Stage-dependent depth | depth may grow, same shared expert | `depth = min(t+1, 3)`, shared block; but `DynamicWindowGenerator` holds **per-turn and per-iteration `nn.Embedding` tables** | conditioning is stage-indexed | "one shared expert" is true of weights but not of conditioning; a reviewer can call this a 3-stage cascade in disguise | **P1** | `annotation_expert.py:135-136`, `dynamic_window.py:64-65` |
| A0 supervision | A0 supervised, headroom preserved | A0 at effective w=0.167 **plus** 92.2% of its gradient from A1–A3 | headroom is actively optimized away | central hypothesis untestable | **P1** | `train_annotation.py:93,98`; measured |
| Recurrent gradient path | (unspecified) | `L(A3) → … → A0 → head → FPN → encoder` **active**, `d(final)/d(A0)` abs-sum 33,250 | — | mechanism behind A0 saturation | **P1** | `self_audit_net.py:76`; executed |
| GT transition targets | GT builds oracle targets, training only | `build_transition_targets` always called **after** `infer` returns | none | — | OK | `finetune_joint.py:244-245` |
| Synthetic counterfactuals | training mechanism only | training-only (verified importers) **but 57.1% of Phase-B loss and of Phase-B validation** | validation contaminated | headline auditor metrics do not measure the deployable task | **P1** | `train_auditor.py:97-127,217,392-402` |
| Auditor detach | auditor must not teach annotation | detached at 3 independent points; 28/28 and 120/120 firewall | none | — | OK | `auditor.py:91-94` etc. |
| Auditor feedback into next stage | evidence conditions next candidate | fed in `infer`, **hard-zero in Phase A and Phase B** | train/inference distribution shift on 3 of 104 channels | conditioning is untrained | **P1** | `self_audit_net.py:64` vs `:217-221` |
| Threshold halting | "optionally halt" | `active = accepted` — first reject halts permanently | no recovery from a single conservative rejection | caps achievable self-audit gain | P2 | `self_audit_net.py:249-250` |
| Neutral semantics | one ε: `>+ε` IMPROVE, `\|·\|≤ε` NEUTRAL, `<−ε` REGRESS | three ε (0.0 / 0.005 / 0.02); 8 zero-boundary sites; Δ==0 counted harmful | AUROC scores a task the loss never trained | reported auditor quality is not the trained quality | **P1** | §11.2 |
| Local evidence FIX/NEUTRAL/REGRESS | 3-way transition evidence | UNCHANGED conflates "both correct" with "both wrong"; no ε; head runs at stride 4 vs per-pixel target (FIX recall 0.0000) | MISMATCH | local head is near-unlearnable and never reaches the decision | **P1** | `targets.py:65-69`, `auditor.py:97-114` |
| ACCEPT/REJECT | local evidence + global ΔQ | `delta_q > τ` only | local head has no path to the decision | one of the two advertised auditor outputs is decorative | P2 | `self_audit_net.py:204`, `gate.py:22` |
| On-policy evaluation | headline metric must be on-policy | **does not exist anywhere in the repo** | — | the number that would support the claim is not computed | **P1** | `train_auditor.py` has no `use_synthetic` flag |
| Headroom measurement | measure A0/candidate/always/self/oracle | all present and correct in sign (residual <1e-7); `HeadroomCapture` estimator is mean-of-ratios (625000.4 vs 0.667) | estimator invalid | the key statistic is unusable as written | **P0** | `audit_decomposition.py:305-308` |
| Volume-level metric | 3-D per-patient Dice | never computed; `evaluate_comparison_modes` has **zero callers**; 2-D slice, resized, empty→1.0 | reported Dice is not the ACDC metric | number not comparable to literature | **P0** | §11.1 |
| Oracle evaluation | analysis-only ceiling | present and analysis-only, but lives inside `infer`'s public API | API/clarity risk | reviewer will question a deployable API that takes GT | P2 | `self_audit_net.py:104,116-117,359-360` |
| GT leakage firewall | deployable inference must be GT-free | **verified GT-free**, bitwise, with a powered positive control | none | — | **OK** | §10 |

---

## 13. Reviewer-style scientific critique

Written as a MICCAI/TMI reviewer trying to reject the paper.

**"Is this actually self-audit?"** Partially. The gate is genuinely GT-free at inference and the auditor genuinely scores a *transition* rather than a candidate (AUROC 0.525 for a candidate-only shortcut is convincing evidence). But the accept decision uses only the scalar ΔQ head; the local FIX/NEUTRAL/REGRESS head — the part that makes it "audit" rather than "a learned confidence score" — has **no path to the decision** and is measurably unlearnable at its own resolution. As built, this is a **learned per-transition confidence gate**, which is a weaker and much more crowded contribution.

**"Is the Auditor necessary?"** Unanswerable from the current code, which is itself the problem. The four required comparison modes exist and are correct, but no run of `scripts/audit_checkpoint.py` on a trained checkpoint is in the repository. Given §9, the mechanistic prediction is `self_audit_dice ≈ initial_dice` (the curriculum is 77% "reject"), in which case the Auditor is doing nothing and the paper's number is A0's number.

**"Is refinement just another segmentation decoder?"** Effectively yes, at present. The expert is 210k parameters trained with deep supervision against the same GT as A0, and 92.2% of A0's gradient flows through it. A reviewer will read Phase A as a 4-level deeply-supervised U-Net-style decoder and ask what the recurrence buys. The honest answer requires the single-stage baseline that does not exist.

**"Is GT secretly defining the task too strongly?"** Not through leakage — that is clean and I could not break it. But through the *objective*: every stage is pulled to the same GT fixed point, so the objective itself dissolves the headroom the paper is about. And through the *curriculum*: 57.1% of the auditor's training signal is GT-synthesized edits whose ΔP fingerprints are separable at AUROC 0.82 without any anatomy.

**"Does A0 saturation eliminate the main contribution?"** Structurally, yes, on ACDC in-domain. Reporting `A0 = 0.97` and `self_audit = 0.97` is not a paper. The ACDC→M&Ms domain-shift protocol already in `configs/` is where A0 is genuinely imperfect; that is where the contribution can survive.

**"Are synthetic transitions artificially making the Auditor look stronger?"** Yes, in two compounding ways: they are 57.1% of the *validation* mixture, and they are separable at AUROC 0.82 from `ΔP`/entropy summary statistics alone. Worse, `primary_metric` — which selects the exported checkpoint — is AUROC on that mixture, so model selection actively rewards fingerprint detection.

**"Can the final result be explained without the Auditor?"** Today, almost certainly. `Dice ≈ 0.97` decomposes as: ImageNet-pretrained ConvNeXt + FPN + deep supervision on a 100-patient dataset, measured 2-D on a resized grid where every absent class scores 1.0. None of that requires an auditor. The auditor's marginal contribution has not been isolated even once.

**The one experiment to run first.** A single 2×2 that separates all four hypotheses (strong A0 / weak candidate generator / weak Auditor / easy dataset):

| | ACDC val (in-domain) | M&Ms (out-of-domain) |
|---|---|---|
| **Volume-level, native geometry, empty-class excluded, per-class RV/MYO/LV** | run all 4 modes | run all 4 modes |

For each cell report `initial_only`, `always_accept_refinement`, `self_audit`, `oracle_accept`, plus `CandidateGain`, `PositiveHeadroom`, `RealizedGain`, `AuditGateValue`, and on-policy-only AUROC. The diagnosis then reads directly off the table:

* `oracle_dice ≈ initial_dice` in both cells ⇒ **no headroom** — the candidate generator is weak, or A0 is saturated. Distinguish with the single-stage-A0 baseline: if it matches A0, A0 is genuinely strong; if it is much worse, deep supervision saturated A0.
* `oracle_dice ≫ initial_dice` but `self_audit_dice ≈ initial_dice` ⇒ **headroom exists, the Auditor cannot find it** — a real, publishable negative that localizes the problem to the auditor.
* `always_accept_dice ≫ self_audit_dice` ⇒ the gate is over-conservative; τ=0 with a 77%-reject curriculum is the prime suspect.
* In-domain flat but out-of-domain positive ⇒ the contribution is real and the framing should be domain-shift robustness, not in-domain ACDC.

That table is exactly what `scripts/audit_checkpoint.py` already produces, modulo the P0 metric fixes. **It has apparently never been run.** Running it is cheaper than any code change proposed below and would settle questions 3, 4 and 5 in one afternoon.

---

## 14. Prioritized remediation plan

No code was changed. Each item lists problem, evidence, files, minimal fix, expected behavioral change, regression test, and research implication.

### P0 — invalidates a reported result

**P0-1 · The reported Dice is not the ACDC metric.**
*Problem:* every Dice in the repo returns **1.0** for a class absent from both prediction and GT, at 2-D slice granularity on a resized 256×256 grid; `foreground_only` is never enabled; no volume-level Dice is ever computed.
*Evidence:* `metrics.py:58`, `targets.py:81-84`, `self_audit_net.py:355`, `visualizer.py:183`; `_utils.py:285`; `evaluate_comparison_modes` (`volume_inference.py`) has zero callers. A slice where the model is totally wrong still scores 0.667 (measured).
*Files:* `evaluation/metrics.py`, `audit/targets.py`, `evaluation/volume_inference.py`, `scripts/audit_checkpoint.py`.
*Minimal fix:* add an `empty_score` policy parameter (default `nan`, excluded from the mean) to `per_class_dice`/`multiclass_dice`; wire `evaluate_comparison_modes` into `audit_checkpoint.py` and report a per-patient 3-D Dice at native geometry as the headline, with the 2-D number kept only as a training-time proxy.
*Expected change:* headline Dice drops, likely substantially, and becomes comparable to literature.
*Regression test:* a synthetic case with one all-empty class must not contribute 1.0; a 3-D reassembly test asserting per-patient Dice equals a hand-computed value.
*Research implication:* this is the single change that decides whether any number in the paper is quotable.

**P0-2 · Phase-A and Phase-C headline Dice are computed differently.**
*Problem:* Phase A batch-pools (`per_class_dice` over the whole batch array), Phase C is per-slice macro. Executed: **0.6825 vs 0.8333** on identical inputs; Phase A is `batch_size`-dependent.
*Evidence:* `train_annotation.py:242` vs `finetune_joint.py:286-287`.
*Minimal fix:* make Phase A use `_foreground_dice_per_sample` and average per slice.
*Regression test:* the same batch scored via both paths must agree to `1e-6`; the Phase-A number must be invariant to `batch_size`.
*Research implication:* Phase-A → Phase-C deltas are currently not a valid comparison.

**P0-3 · `headroom_capture_ratio` is a mean-of-ratios with a `1e-8` guard.**
*Problem:* one near-zero-headroom row drives it to ~1e6. Executed: returns **625000.4** where the correct ratio-of-means is **0.667**. This is the paper's central statistic and §7 predicts near-zero headroom on most rows.
*Evidence:* `audit_decomposition.py:305-308`.
*Minimal fix:* `capture = Σ self_gain[m] / Σ oracle_headroom[m]` over `m = oracle_headroom > neutral_margin`; report `headroom_available_rate` alongside it and `nan` when the denominator is empty.
*Regression test:* a two-row case with headrooms `[2e-8, 0.3]` must return ~0.667, not 6e5.

**P0-4 · No held-out test evaluation; τ is selected and reported on the same split; the calibrated τ is then never used.**
*Problem:* `test_split: test` is declared in all three configs and evaluated nowhere (`grep -n "test" scripts/train_self_audit.py` returns nothing). `train_self_audit.py:570-598` takes an argmax over **81** τ on `val_c` and reports `best_threshold["final_macro_dice"]` — the maximized objective value, on the split it was maximized over. Separately, every reported self-audit Dice uses `audit_cfg.get("tau_accept", 0.0)` — the **hard-coded 0.0**, not the calibrated value. The two numbers are not comparable and neither is a clean estimate.
*Evidence:* `train_self_audit.py:471-473, 508-528, 570-598`; `scripts/calibrate_threshold.py:66` writes `calibration.json` which nothing reads back.
*Files:* `scripts/acdc_split.py` (emit a `test` split), `scripts/train_self_audit.py`, `evaluation/threshold.py`.
*Minimal fix:* regenerate the manifest as train/val/test (e.g. 70/15/15 patients, still patient-level); calibrate τ on `val`; write it into the run config; evaluate **once** on `test` with that τ; report the test number as headline.
*Regression test:* assert the τ used by the reported evaluation equals the τ in `calibration.json`; assert the test split is disjoint from both others at patient level.
*Research implication:* without this, every number is an optimistically-selected validation estimate.

### P1 — threatens the main scientific claim

**P1-1 · Phase-B validation is 57.1% GT-synthesized, and that metric selects the checkpoint.**
*Evidence:* `train_auditor.py:97-127` (4 synthetic vs 3 on-policy, verified at runtime), `:217` (uniform mean), `:392-402` + `:522`/`:530` (same generator in val), `:428-429`/`:540-549` (`primary_metric` = AUROC on the mixture). AUROC 0.82 is reachable from `ΔP`/entropy alone.
*Minimal fix:* thread `use_synthetic: bool` through `_auditor_batch`/`build_auditor_transitions`; keep synthetic for training, set `False` for validation. At minimum, split every reported metric by the `provenance` field already present at `:103`/`:123` into `audit/on_policy/*` and `audit/synthetic/*`, and make `primary_metric` = on-policy AUROC.
*Expected change:* reported AUROC will drop, probably a lot. That drop is the finding.
*Regression test:* assert `validate_auditor_epoch` produces zero transitions with `provenance.startswith("synthetic")` when `use_synthetic=False`; assert both namespaces appear in the W&B payload.
*Research implication:* **the primary paper metric must be on-policy.** No current number qualifies.

**P1-2 · Objective-induced A0 saturation.**
*Evidence:* 7.8% / 92.2% gradient decomposition (measured); `train_annotation.py:93,98`; model selection on A3 only (`:239-242,:354`).
*Minimal fix (measurement, not architecture):* (a) run `scripts/audit_checkpoint.py` and report `phase_a/mean_stage_headroom`; (b) add a `stage_weights: [1.0]`, `max_turns: 0` single-stage control run as the A0 baseline; (c) report the ACDC→M&Ms cell from `configs/self_audit_acdc_to_mnms.yaml`.
*Do not* simply lower `w_0` — 92.2% of the pressure does not come from `w_0`, and handicapping the baseline is not a defensible way to manufacture a delta (§7).
*Regression test:* a headroom assertion in the diagnostics — fail loudly if `oracle_headroom < neutral_margin` on the majority of rows, since every downstream ratio is then meaningless.

**P1-3 · The positive counterfactual branch is degenerate at realistic confidence.**
*Evidence:* 57.5–67.5% of positive CFs change zero pixels at p_max ≥ 0.95; 91–95% inside the loss's neutral band; `P(dDice>0) = 0.230` over the generator's mixture ⇒ 77% "reject" curriculum. `counterfactual.py:316-320,332`.
*Minimal fix:* reject a positive edit whose argmax did not change (turn `valid=False` at `:332` when `candidate.argmax == previous.argmax`) and raise `max_repair_fraction` toward 1.0 so a confidently-wrong pixel can actually flip; optionally resample until `dDice > neutral_margin`.
*Expected change:* the positive class prior rises materially; the gate stops being trained to park repairs at Δq ≈ 0.
*Regression test:* at `p_max = 0.99`, assert < 10% of `kind="positive"` samples are argmax-identical.

**P1-4 · Neutral-margin semantics are inconsistent across 8 sites and 3 epsilons.**
*Evidence:* table in §11.2; executed damage (`harmful_acceptance_rate = 0.8` where the true value is 0.2).
*Minimal fix:* thread one `neutral_margin` through `transition_audit_metrics`, `acceptance_metrics`, `evaluate_threshold`, and `_candidate_improves`; classify `|Δ| ≤ ε` as NEUTRAL and exclude neutral rows from both `harmful_acceptance` and `beneficial_rejection` denominators; reconcile `epsilon_neutral: 0.02` with `neutral_margin: 0.005` (they govern the same decision in the same config block).
*Regression test:* `acceptance_metrics` with all-zero deltas must report `harmful_acceptance_rate = 0.0`, not 1.0.
*Research implication:* the reported AUROC currently scores a task the loss never trained.

**P1-5 · The local audit head cannot represent its own target.**
*Evidence:* trunk at stride 4 (`auditor.py:97-100,111-113`) vs per-pixel target (`targets.py:55-70`); oracle-through-stride-4 FIX recall **0.0000**. Target also conflates "both correct" with "both wrong" (`targets.py:65-69`).
*Minimal fix (cheapest honest option):* keep the architecture, but compute the local loss against a **stride-4 pooled target** so the training objective is achievable, and state in the paper that local evidence is coarse. If full-resolution localization is a claim, the head needs a decoder, which is a real architecture change and out of scope for this remediation.
*Regression test:* assert `local_logits` spatial resolution and the target resolution used by the loss agree.

**P1-6 · No volume-level, native-geometry, per-class reporting.** (Companion to P0-1; listed separately because the fix is a wiring change, not a metric change.) `evaluate_comparison_modes` exists and has zero callers; ED/ES are pooled. Wire it into `audit_checkpoint.py`, report RV/MYO/LV separately and ED/ES separately, as ACDC convention requires.

**P1-7 · Train/inference mismatch on the recurrent audit channel, and stage-indexed embeddings.**
*Evidence:* `self_audit_net.py:64` (`audit_evidence = None`, never reassigned in `forward_annotation`) vs `:217-221` (real evidence at inference); `dynamic_window.py:64-65` (per-turn / per-iteration `nn.Embedding`).
*Minimal fix:* either (a) feed the auditor's evidence during Phase A — which requires the auditor, so realistically Phase C only — or (b) state explicitly that the audit channels are inert until Phase C and verify that Phase C actually trains them. For the embeddings, either remove them (true weight sharing) or report them honestly as stage conditioning; do not describe the model as "one shared expert" without the caveat.
*Regression test:* assert the expert's audit-channel input is non-zero for `turn ≥ 1` in whichever phase is supposed to train it.

### P2 — important protocol or training issues

1. `validate_dataset_splits` discards the manifest and reports a different partition (15/20 val patients mislabeled) — `_utils.py:369-379`.
2. Halt-on-first-reject is unconditional (`self_audit_net.py:249-250`); the intended design says "optionally halt". Consider a `halt_on_reject` flag so a single conservative rejection does not end refinement.
3. Local evidence never enters the accept decision (`self_audit_net.py:204`); one of the two advertised auditor outputs is decorative.
4. `oracle_accept` and `oracle_target` live in the deployable `infer` API with zero validation, and `forward()` splats `**kwargs` into it (`self_audit_net.py:104,359-360`). Move oracle mode into evaluation code.
5. ΔQ term receives a **34× weaker** gradient than the local CE term at equal weight, yet ΔQ is the head that actually gates.
6. FPN `refine[1..3]` are computed and discarded — 12 parameters permanently dead, 3 conv+GN blocks of wasted compute every forward (`fpn.py:53-54`).
7. No phase-tag validation on checkpoint load ⇒ a Phase-A checkpoint in Phase C runs `self_audit` with a random-init auditor at τ=0.
8. Phase C has no auditor parameter group; auditor LR silently drops 3e-4 → 1e-5.
9. `evaluate_audit_decomposition` unweighted batch averaging + silent NaN drop (0.5625 vs 0.2222 measured).
10. `stage_t/oracle_gain` is a mislabeled alias of `headroom`; `phase_a/headroom_collapse_ratio` hardcodes 0.005; `gt_firewall/*` covers batch 0 only.
11. `SelfAuditNet._dice` hardcodes classes `(1,2,3)` — live trap given `prepare_mnm_binary.py`.
12. `amp_dtype: float16` on CUDA turns normal `GradScaler` warmup into a hard `FloatingPointError`; latent because all checked-in configs use `bfloat16`.

### P3 — maintainability / clarity

1. `resolve_stage_weights` silently substitutes defaults on a length mismatch (`train_annotation.py:63-70`).
2. `ThresholdGate.run` is dead code; `infer` reimplements the loop inline. Two implementations of the same contract will drift.
3. `annotation_entropy` selects its normalization by a whole-tensor sign test (`annotation_expert.py:18-21`), producing a 0.54 max per-pixel swing depending on the input's global min/max.
4. Phase C `--output` is parsed but never used; the config value always wins.
5. Phase B recomputes the encoder a second time per batch (`_feature_for_audit` after `forward_annotation`).
6. `evaluation/__init__.py:12` imports `.visualizer` unconditionally, so `import self_audit.evaluation.*` — including `scripts/audit_checkpoint.py` — hard-fails without matplotlib. Make it lazy; the diagnostic CLI should not need a plotting library.

---

## Appendix — what this audit could not falsify

Stated explicitly, because a credible audit must report the attempts that failed:

* The GT firewall. Attacked statically, by bitwise permutation, and with a powered positive control. It held.
* `P_previous` being load-bearing for the auditor. A candidate-only shortcut scores AUROC 0.525 — chance.
* The stage-wise attribution algebra. Sign conventions verified by a hand-computed 2-row example through the real function; `candidate_gain + audit_gate_value == realized_gain` to `< 1e-7`.
* Weight sharing of the Annotation Expert. Verified by Python object identity across 6 invocations, not by naming.
* Patient-level split integrity, including the ED/ES straddle case.
* Checkpoint architecture validation and `strict=True` loading.
* The existence and correctness of the four comparison modes and all four headroom quantities. They are present and logged. The gap is the estimator (P0-3) and the fact that they appear never to have been run.
