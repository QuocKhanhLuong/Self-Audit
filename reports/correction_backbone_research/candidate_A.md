# Candidate A — conservative recurrent correction core

Decision: retain as an engineering/control architecture. No paper-level architectural novelty is claimed.

## 1. Exact state variables

`V` is the cached visual pyramid. Retain `L_t` (four-class logits), `C_t ∈ R[d,32,32]`, previous accepted three-channel audit `e_{t-1}`, turn index and active-row bit. `C_0=0`; `e_-1=0`. No history bank or class obligation ledger. Clear all state at each new center-slice sample.

## 2. Equations

\[
x_t=\operatorname{Proj}[V_8,\downarrow a_t,\downarrow e_{t-1},\downarrow H(a_t),\operatorname{time}(t)],\quad
\tilde C_{t+1}=\operatorname{ConvGRU}_\theta(x_t,C_t),
\]
\[
\tilde L_{t+1}=L_t+\sigma(g_t)\odot D_\theta(V_4,\uparrow\tilde C_{t+1},a_t,e_{t-1}).
\]

After the official detached transition audit, accept iff `q_t > tau`. Commit C, logits and predicted audit together; otherwise preserve the retained pair and HALT. The candidate is decoded before obtaining `e_t`; there is no circular audit input.

## 3. Visual encoder

First isolate recurrence with **pretrained ConvNeXt-Tiny and the existing FPN**, changing only the correction core in a future research implementation. Then test the ordinary GroupNorm CNN with `[w,2w,4w,8w]` at w=32/48/60 specified in [state_formulation.md](state_formulation.md). The image encoder runs once. Treat scratch and train-only SSL as separate initialization conditions; preserve strong A0 supervision in every condition.

## 4. Correction updater

One spatial ConvGRU at stride 8, d=64/96/112. Gates share weights over turns. A 1×1 input projection and bounded residual output prevent uncontrolled logit magnitude, but do not guarantee improved masks. No recurrent full-encoder pass. Include an equal-size residual updater without hidden state as a direct control.

## 5. Decoder

The stride-4 decoder combines stable visual detail with upsampled C and projected current annotation/audit. Four 3×3 convolutions plus a 1×1 projection/readout in the reference budget. A separate A0 head sees V and is supervised at unchanged strength. A0 must never require a deliberately poor starting mask to create headroom.

## 6. Auditor interaction

Use the baseline Auditor interface: detached V, old/new probabilities, differences and entropy; predict local FIX/UNCHANGED/REGRESS and global delta Q. Only **predicted** previous accepted evidence enters x. This prevents GT leakage but does not eliminate miscalibration. Compare no audit input, current audit map alone, and generic confidence input.

## 7. Commit/reject behavior

Atomic row-wise commit. Inactive rows do not execute the updater. Rejection is terminal; latent rollback alone should have no output benefit by the HALT equivalence lemma. The initial usefulness test concerns accepted hidden-state persistence, not rejected-state handling.

## 8. Compatibility with CC-restitution

Keep CC-restitution separate first, using current ConvNeXt/current expert as its control. To attach it to A, record the pre-GRU C in addition to the existing factual replay inputs. A generic hidden-state difference cannot safely be added after unrelated later updates; restrict to the immediately preceding ordinary action and compare against mask-only restitution. This is already more complexity than the conservative candidate needs. Prefer A without replay until it beats no-state refinement.

## 9. Parameter estimate

Analytical full compact systems approximately 4.15M / 9.31M / 14.27M before small norms/embeddings, corresponding to roughly 5M/10M/15M regimes. The recurrent core alone is about 0.24M / 0.52M / 0.71M. ConvNeXt-hosted A retains the larger pretrained baseline encoder and is not called a 5M model. Exact profiling awaits an authorized implementation.

## 10. Compute per refinement turn

At 256² and batch 1: approximately 1.08 / 2.40 / 3.27 GMAC, including the proposed decoder and Auditor; encoder once approximately 2.66 / 5.97 / 9.32 GMAC. One C tensor is 0.125 / 0.188 / 0.219 MiB in BF16. These are formulas, not GPU measurements. Training unrolls, full-resolution logits and optimizer buffers add memory; latency cannot be predicted from MAC alone.

## 11. Training schedule

Use the early-shadow schedule in [training_strategy.md](training_strategy.md): protect A0 while training an Auditor on detached evolving transitions, then expose the core to predicted evidence without hard rejection, finally use deployment-matched hard gates. Optional soft commits are an ablation, not required. Preserve an unchanged late-audit baseline and a ConvNeXt early-audit control. Train the GRU with per-turn segmentation plus regression/past-FIX penalties; paired utility objectives can be tested but are not required to define A.

## 12. Novelty risk

Very high overlap. This is explicitly **ConvGRU plus audit maps**, a useful falsification baseline for B. Fixed V and recurrence are already established. An external acceptance gate is an RNN gate, and rejection-HALT makes rejected-state discard observationally inert. If A wins, report engineering/data-efficiency evidence, not a new state abstraction.

## 13. Five closest methods

| Method | Mechanism threatening A | Necessary comparison |
|---|---|---|
| [Recurrent Mask Refinement, ICCV 2021](https://arxiv.org/abs/2108.00622) | Repeated prediction-conditioned relational feature/mask refinement in medical segmentation. | Same-image recurrent refinement adaptation under the same Auditor. |
| [RITM](https://arxiv.org/abs/2102.06583) | Iterative training with previous-mask input. | Predicted autonomous evidence replaces clicks; label it an adaptation. |
| [Mask2Former](https://arxiv.org/abs/2112.01527) | Predicted-mask-guided feature attention and evolving queries. | Compact mask-conditioned query/refinement decoder; same budget. |
| [SAM 2](https://arxiv.org/abs/2408.00714) | Mutable memory conditions segmentation features. | Lightweight memory-reader adaptation; distinguish video-frame memory from same-image turns. |
| [Skip RNN](https://arxiv.org/abs/1708.06834) | Hidden-state updates can be suppressed by a gate. | Gate matched to the Auditor; no novelty credit for identity on rejection. |

ConvGRU/recurrent U-Net families are additionally covered in the [prior-art matrix](prior_art_matrix.md); the component itself is prior art.

## 14. Experiments that falsify A

Reject learned persistence if resetting C after every accepted turn, with logits and last audit fixed, produces equivalent performance; if a same-capacity feed-forward expert matches it; if shuffled audit is as useful; or if its improvement disappears with matched A0/compute. Reject the compact encoder replacement if it trails pretrained ConvNeXt in final quality and safe correction at the same operating point. A may survive solely as a lower-cost engineering option if a preregistered noninferiority margin is met.
