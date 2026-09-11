# State formulation and limits of the claim

Status: proposed research equations, not an implemented or validated architecture. Exactly three architectures are specified in [A](candidate_A.md), [B](candidate_B.md), and [C](candidate_C.md). CC-restitution means the previously selected Candidate C mechanism; it is not a fourth architecture.

## Notation and temporal order

For one row, let `L_t` be retained four-class logits, `a_t = softmax(L_t)`, `y_t = argmax L_t`, and `V = E_visual(I)` a multiscale tensor computed once. Stable means **unchanged inside a rollout**, not frozen encoder parameters across training. The encoder may receive annotation gradients; a separate freeze-weights ablation tests whether doing so helps.

`S_t = (C_t, P_t, R_t, H_t)` contains a low-resolution correction tensor, class-indexed preservation strengths, class-indexed repair obligations and bounded accepted-event metadata. A uses only C; B adds P/R and class semantics; C additionally stores a replayable event. All initialize empty/zero; A0 has its own fully supervised decoder.

The only evidence available before proposing turn t is `e_{t-1}` from the previous accepted transition. No equation may condition a candidate on its own future audit without a separately counted second proposal/audit pass.

\[
\tilde C_{t+1}=C_t+F_\theta(V,C_t,a_t,P_t,R_t,e_{t-1},t),\quad
\tilde L_{t+1}=L_t+g_\theta\odot D_\theta(V,\tilde C_{t+1},a_t,P_t,R_t),
\]
\[
(e_t,q_t)=Q_\phi(\operatorname{sg}(V,a_t,\tilde a_{t+1},\tilde a_{t+1}-a_t)),\quad b_t=\mathbf1[q_t>\tau].
\]

Here `sg` is stop-gradient; entropy channels are omitted only from notation. The official Auditor reads annotation transitions and stable evidence, initially **not C**; otherwise state can become an unverified private communication channel. Local evidence and the acceptance gate are detached when used by the proposer. Train the Auditor on GT transition targets through its own loss.

After auditing, let `W(S_t, Ctilde, L_t, Ltilde, sg(e_t))` be the proposed persistent state including semantic record writes. Atomically:

\[
\boxed{(S_{t+1},L_{t+1})=
 b_t\big(W(S_t,\tilde C_{t+1},L_t,\tilde L_{t+1},\operatorname{sg}(e_t)),\tilde L_{t+1}\big)
 +(1-b_t)(S_t,L_t).}\tag{1}
\]

This shorthand means a row-wise branch over tensors and metadata, not arithmetic on history objects. Reject HALTs. The post-audit write affects the **next** proposal, not the candidate just audited. Diagnostic rejected proposals may be logged outside retained state. No cross-patient memory, mutable normalization statistics, random-number changes or hidden shared buffers may leak across row transactions.

## Why equation (1) is insufficient novelty

It is a binary gated RNN update: `(1-b)S+bW`. A GRU admits this form when its update gate is externally clamped. [Skip RNN](https://arxiv.org/abs/1708.06834) already learns to skip hidden-state updates. Segmentation memory selection also rejects low-confidence memories; [SAM2Long](https://openaccess.thecvf.com/content/ICCV2025/html/Ding_SAM2Long_Enhancing_SAM_2_for_Long_Video_Segmentation_with_a_ICCV_2025_paper.html) maintains and scores alternative memory pathways. An external transition critic is a task-specific gate, not a new computational primitive.

**HALT equivalence lemma.** Consider two inference systems with identical accepted updates, candidate masks and gates. They differ only in whether they store a rejected candidate's hidden state. If rejection terminates that row, hidden state is not an output, and there are no side effects shared with other rows or training, their returned annotations are identical. Proof: by induction the accepted prefixes agree; the first rejected step returns the same previous annotation; no later step observes the different hidden tensor. The same holds at the turn cap. This is an analytic argument, not an experiment.

Training through rejected states, continuing after rejection, replaying rejected events, or cross-row mutable buffers breaks these assumptions and changes the experiment. Do not manufacture such a change merely to claim rollback gains. An inference-only reject/discard ablation should tie; divergence reveals a violated assumption or bug. Accepted-history usefulness is the research target.

## Narrow hypothesis and equation worth testing

Let `P_t^k(x)` mean confidence that an accepted correction established class k at x, and `R_t^k(x)` mean an unresolved regression attributable to an accepted action whose previous class was k. These are predicted obligations, not truth. Newly wrong-to-wrong transitions fall into local UNCHANGED and are not guaranteed to be repaired by this representation.

In Candidate B, the value of a committed representation must be measured at a **fixed annotation** against a null write:

\[
\boxed{\mathcal U_t = J_Y\big(\mathcal T_K(V,L_{t+1},S_{t+1})\big)
-J_Y\big(\mathcal T_K(V,L_{t+1},\operatorname{NullWrite}(S_t))\big),
\quad \mathcal U_t>0\ \text{with less destruction of accepted FIX.}}\tag{2}
\]

`T_K` is the same bounded future proposer/Auditor policy in both arms, same weights/randomness/turn index, remaining horizon and complete visible logits. Only accepted nonterminal prefixes with K>=1 support this test; terminal writes have no future observation and are excluded, not labeled useless from missing outcomes. The null arm has the same non-state inputs (including latest audit when testing information beyond current evidence). Also test semantic-preserving C reset, complete history reset, and a parameter-matched unstructured writer separately. Outcome J includes beneficial corrections, regressions and past-FIX destruction; GT is used only to evaluate/train these targets. This is an operational hypothesis, not a theorem that latent state must help. State trained only with segmentation loss and carrying redundant mask information fails this test.

## Objectives that distinguish useful and unnecessary writes

For GT Y define hard masks `F_t = [y_t != Y, ytilde = Y]`, `Rgt_t = [y_t = Y, ytilde != Y]`. Let `B_t` be pixels ever fixed by an accepted transition and still correct before this proposal; destruction is `D_t = B_t & [ytilde != Y]`. Keep this evaluation history separate from predicted P/R. Report first-destruction and repeated-destruction rates separately. All normalizations and absent denominators are in [experiment_plan.md](experiment_plan.md).

Train with A0 and per-proposal Dice/CE plus differentiable surrogates. At pixels wrong before proposal maximize `p_tilde(Y)`; at previously correct pixels minimize `-log p_tilde(Y)`; weight the latter separately on B_t. Add a consistency/no-write cost on low-benefit/no-change examples, avoiding a large global zero-update penalty that suppresses useful correction. A possible research objective is

\[
\mathcal L=\mathcal L_{A0}+\sum_t w_t(\mathcal L_{seg}+\lambda_R\mathcal L_{reg}+\lambda_D\mathcal L_{destroy})
+\lambda_Q\mathcal L_{audit}+\lambda_U\mathcal L_{paired}+\lambda_N\mathcal L_{unnecessary}.
\]

For paired future rollouts use a differentiable future loss `ell+` with the proposed committed state and `ell0` with no write, at identical annotation. A stopped teacher target derived from bounded GT-scored future probes labels writes **good** when `J+ - J0 > epsilon`, **bad** when below `-epsilon`, and **unnecessary** inside the tolerance. A small utility head predicts these labels; ranking loss is `max(0, margin - (v+ - v0))` for useful pairs, reversed for harmful pairs. Direct future supervision minimizes `ell+`; utility classification alone can learn a diagnostic without improving state, so it is insufficient. Do not demand every write have a positive margin where no headroom exists. Bad/unnecessary writes get a null-write or minimum-change teacher; labels and masks are stop-gradient. Report the additional training rollouts and test usefulness out of sample.

Stopping gradients in the comparator prevents direct adversarial optimization of that branch but does not freeze shared parameters. Train and monitor zero-history/null inputs with independent segmentation supervision, report both arms' absolute outcomes, and repeat evaluation with frozen common proposal weights. An increased paired gap caused by degrading the null arm is not state usefulness. Compare exact B1 maps plus a single equal-capacity GRU with the same utility and destruction losses before crediting split latent state.

## Explicit analytical capacity and compute budget

These are proposed dimensioned reference designs, **not counts of an implemented model or measured speed/VRAM**. Batch 1, 256² input, two bytes per stored activation, ordinary turn at stride 8 (32²), final decoder at stride 4 (64²). One MAC means one multiply-accumulate; a convention counting multiply and add separately is approximately 2 FLOPs/MAC. No quadratic global attention.

Reference visual CNN: stages stride 2/4/8/16, widths `[w,2w,4w,8w]`; each stage has a stride-2 3×3 input convolution and two two-convolution residual blocks, GroupNorm/GELU. Ignoring small biases/norms, `Penc=3438w²+27w`. This deliberately ordinary encoder is a capacity control, not a claimed invention. Four lateral projections to d channels cost `15wd`. A0 uses a 3×3 head then four-class readout (`9d²+4d`).

For A, ConvGRU with input width `d+10` and hidden d costs `27(2d+10)d`; concatenate projected V, state, annotation and evidence at stride 4 through a 1×1 readout plus four 3×3 convolutions (`40d²+5d`, including four logits and a gate). Auditor: input projection, two residual blocks, local head and global MLP, approximately `37.5d²+18d`. Small mask/evidence embeddings, interpolation and normalization are omitted and must enter later profiler totals.

| Regime label | w / d | Encoder parameters | Approx A total | Encoder once, GMAC | A ordinary turn including Auditor, GMAC | One C buffer |
|---|---|---:|---:|---:|---:|---:|
| roughly 5M | 32 / 64 | 3.52M | 4.15M | 2.66 | 1.08 | 0.125 MiB |
| roughly 10M | 48 / 96 | 7.92M | 9.31M | 5.97 | 2.40 | 0.188 MiB |
| roughly 15M | 60 / 112 | 12.38M | 14.27M | 9.32 | 3.27 | 0.219 MiB |

The totals intentionally leave room for embeddings and B/C writers. A target is a size regime, not an instruction to pad parameters. B's class-ledger writers and split state add approximately 0.2/0.5/0.7M and 0.2/0.5/0.7 GMAC/turn under dense small-convolution implementations. C adds approximately 0.2–0.6M to B for event/support heads if needed, while frozen replay adds **zero** learned weights. Candidate C is near/above the 15M ceiling at the widest setting; reduce w/d instead of concealing auxiliary parameters.

V pyramid storage at strides 2/4/8/16 with d channels is approximately `2*d*21760` bytes (2.66/3.98/4.65 MiB), excluding raw encoder maps. Four-class full-resolution logits cost 0.5 MiB per tensor. State buffers are small; full-resolution outputs, stored encoder activations, optimizer state and unrolled autograd dominate training memory. FP32 parameters + gradients + two Adam moments cost about 16 bytes/parameter (76/153/229 MiB for 5/10/15M), before activations, master-weight choices and allocator overhead. C requires snapshots and a coordinate-gradient graph; peak VRAM cannot be inferred from parameter count.

Caching replaces `T * encoder_cost` by one encoder cost; **current Self-Audit already gets that saving**. A full mask-conditioned encoder rerun pays another ~2.66/5.97/9.32 GMAC per turn for this reference CNN, and carries deeper temporal gradient paths. Repeated whole-encoder adaptation might improve capacity but weakens evidence stability and data efficiency. Measure it as a control, not dismiss it a priori. Training BPTT remains costly even when V is numerically constant: annotation gradients from multiple turns accumulate into its graph.

A real 2.5D implementation must keep z-neighbor channels aligned, reset state for every center slice/patient, preserve through-plane spacing, and score stacked patient volumes. Cross-slice memory would be a separate protocol. Fine myocardium boundaries require stride-4 visual skips; an exclusively stride-8 correction bottleneck is a falsifiable resolution limitation.
