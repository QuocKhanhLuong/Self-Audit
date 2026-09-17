# Training strategy for correction-state research

Research proposal, 2026-09-11. Source anchor: `main` at `35aaba9335e0d8a0aa344356e20e93ae902b950d`, verified during this study. No training, implementation, parameter measurement, GPU profiling, or effectiveness evaluation was performed. Numerical schedules, budgets, margins, and gates below are proposals to preregister before collecting comparative results.

The recommended order is A (small ConvGRU), then B (structured repair/preservation memory), then C (replayable sparse correction events) only if simpler models establish correction and memory headroom. Keep the ImageNet-pretrained ConvNeXt baseline. Existing immutable visual features already solve repeated image encoding within a rollout; replacing the encoder requires evidence about A0 quality, data efficiency, and total cost, rather than this caching argument.

## 1. Current contract and research boundary

Current source evidence: [`docs.md`](../../docs.md), [`SelfAuditNet.encode`, `forward_annotation`, and `infer`](../../src/self_audit/models/self_audit_net.py), and [`ThresholdGate`](../../src/self_audit/audit/gate.py). `infer` computes shared features before the loop; accepted soft annotations become the next state, while rejection retains the previous annotation and halts. The documented default is three attempted turns maximum, `tau_accept=0`, and a 130-epoch curriculum: annotation bootstrap `[0,100)`, auditor-only `[100,120)`, joint gated training `[120,130)`. This document proposes alternative research runs; it does not change that baseline.

Here A/B/C name three **new correction-state architecture families**. They are not the existing three training intervals, and research C must not be confused with the repository's existing `candidate_c` support-geometry/replay implementation. An experiment must record both its architecture-family identifier and any existing `window_mode` setting.

The prediction contract stays `[z-1,z,z+1] -> center-slice four-class logits`, stacked to ED/ES volumes for patient outcomes. Class ids remain Background/RV/MYO/LV = 0/1/2/3. The shared image representation is immutable **within one rollout**; it can receive annotation-loss gradients during training between rollouts. The auditor receives detached features and soft previous/candidate probabilities, never GT at inference. All counterfactual masks, utility targets, and destruction labels derived from GT are training or retrospective analysis objects only.

## 2. Data scale and initialization controls

ACDC's original development set has 100 patients, with 20 in each of five groups; the separate original test set has 50. The official pages currently describe manual references for both sets, so historical hidden-test language must not be used as a claim about current access. ACDC's short-axis spacing is anisotropic. These facts support patient grouping and the retained 2.5D contract, not a presumption that thousands of correlated slices form thousands of independent examples. [ACDC training](https://www.creatis.insa-lyon.fr/Challenge/acdc/databasesTraining.html), [test](https://www.creatis.insa-lyon.fr/Challenge/acdc/databasesTesting.html), [acquisition](https://www.creatis.insa-lyon.fr/Challenge/acdc/databases.html).

The M&Ms primary paper describes 375 studies across four vendors/six centres, with 175 challenge training cases: 150 annotated from two vendors and 25 unannotated from a third; 40 validation cases and 160 final test cases follow. These published counts do not prove which files or labels are available locally. Freeze the actual external manifest and group any repeated acquisitions by verified patient identity. For ACDC-to-M&Ms, no M&Ms images enter SSL, normalization fitting, augmentation tuning, calibration, architecture selection, or checkpoint selection. Per-volume preprocessing fixed on ACDC is allowed at inference; learning target-domain statistics is a separate adaptation experiment. [M&Ms primary paper, Sections II and V](https://diposit.ub.edu/bitstreams/f2f7dc43-becc-4554-910d-360ed7fd339c/download).

Mandatory initialization comparisons:

| Arm | Initialization and exposure | Identifiable question |
|---|---|---|
| R-small | Lightweight residual stem, random initialization, no external pretraining | Can a compact complete system learn with the available patients? |
| S-small | Exactly the same small stem and downstream model; train-only global/local contrastive SSL | Does unlabelled **training-patient** exposure improve this same architecture? |
| P-CNX | Existing ImageNet-pretrained ConvNeXt encoder and baseline A0, retained at native size | Does the compact system improve the practical accuracy/cost frontier? |
| R-CNX | Same ConvNeXt architecture as P-CNX, random initialization | Separates initialization from encoder architecture/size; needed before a claim about pretraining itself |
| P-small, conditional | A compact ImageNet-pretrained encoder only if a legitimate checkpoint and identical random-initialized architecture can be supplied | Additional within-architecture initialization control; do not relabel an unrelated pretrained backbone as the custom stem |

For S-small, adapt global/local contrastive pretraining using consistent spatial transforms and anatomical slice-position information; local positives require mapped overlapping coordinates, not arbitrary neighbouring slices treated as the same structure. Split first, then construct SSL pairs, caches, or learned normalization. Unlabelled phases from training patients may be included only in a separately recorded all-train-images arm also available to matched controls. Report two regimes: restricted-image SSL (only images attached to the labelled training subset) and fixed-unlabelled-pool SSL (all outer-training patients, labels hidden outside the labelled subset). The latter is label efficiency with additional image exposure. [Chaitanya et al., NeurIPS 2020](https://arxiv.org/abs/2006.10511).

A proposed split of the 100 ACDC development patients is 60 fit / 10 model-selection / 10 calibration / 20 locked evaluation, stratified by the five original groups (12/2/2/4 per group). Preserve an existing authoritative patient split if one is already locked; register its exact counts instead of silently substituting this proposal. Use nested labelled fit subsets of 10, 20, 40, 60 patients, selected once with group stratification. ED/ES, all slices, adjacent-slice inputs, unlabelled frames, and repeated acquisitions of one patient stay together. A 10-patient calibration set may be insufficient for a precise risk bound; the outcome in that case is inconclusive, not threshold tuning on evaluation patients.

## 3. Common computational and gradient contract

Let `H=E_phi(x)` be fixed visual features, `A_t` soft logits, `P_t=softmax(A_t)`, `S_t` correction memory, and `e_t` the most recent accepted predicted audit evidence (zero plus a validity bit at the start). At active turn `t`:

```text
(delta_t, gate_t, S_draft) = R_theta(H, A_t, S_t, e_t, t)
C_t = A_t + sigmoid(gate_t) * delta_t
(q_t, pi_t) = Q_psi(stopgrad(H), P_t, softmax(C_t), softmax(C_t)-P_t)
pi_t = probabilities over FIX / UNCHANGED / REGRESS
accept_t = 1[q_t > tau]                  # final hard-policy phase only
if accept_t:
    A_(t+1) = C_t
    S_(t+1) = V_theta(S_t, S_draft, stopgrad(pi_t), C_t-A_t)
    e_(t+1) = stopgrad(pi_t)
else:
    preserve A_t and accepted memory; HALT
```

`Q` is a single shared lightweight transition auditor, not a second image encoder. Annotation loss never backpropagates through audit scores, local audit maps, gate decisions, or auditor parameters. Auditor training treats annotation probabilities and features as detached inputs. Within an accepted trajectory, gradients can flow through recurrent annotation and memory dynamics for bounded backpropagation through time; detach at explicit chunk boundaries and record the rule. A rejection produces no deployable future turn.

Ground-truth local targets at a pixel are `FIX = wrong(P_t) and correct(C_t)`, `REGRESS = correct(P_t) and wrong(C_t)`, and `UNCHANGED = otherwise`, with correctness determined by the four-class argmax. Wrong-to-different-wrong belongs to UNCHANGED, not FIX. Use GT only to calculate the training target. Requested perturbation names are never substituted for measured transition labels.

## 4. Architecture-specific training targets

### A: minimal ConvGRU correction state

Use a small state on the stride-8 feature grid, e.g. 64 channels for the first compact pilot, with stride-4 visual skips in the decoder. Let `u_t` project the fixed features and concatenate current probabilities/entropy, last accepted predicted audit, and turn information. A standard ConvGRU proposal is:

```text
z = sigmoid(W_z * [h_t,u_t]); r = sigmoid(W_r * [h_t,u_t])
h_draft = (1-z)*h_t + z*tanh(W_h * [r*h_t,u_t])
(delta_t, gate_t) = small_head(h_draft, H, A_t)
h_(t+1) = h_draft on accept; otherwise HALT
```

Use zero initialization with a validity bit; compare a learned image-conditioned initial state as a separately labelled ablation. Three 3x3 recurrent convolutions and a shallow output head are sufficient for the first experiment. The external `e_(t+1)` carries newly predicted audit evidence; A does not need a second recurrent pass merely to absorb it. Train deep supervision on A0 and proposals, with the same segmentation loss as the baseline. This is a direct recurrent-update adaptation of RAFT, without optical-flow inputs, an all-pairs cost volume, or a claim that RAFT demonstrated cardiac segmentation. [Teed and Deng, ECCV 2020, Section 3.3](https://arxiv.org/pdf/2003.12039).

### B: structured repair and preservation memory

Use [candidate_B.md](candidate_B.md)'s class-bound writer B1. In this training document `A` denotes logits and `P=softmax(A)` denotes probabilities; call the preservation ledger `K^k` to avoid collision with probabilities (`P^k` in the architecture document). State contains distinct latent branches `C^R,C^K` plus class-indexed repair obligations `R^k` and preservation strengths `K^k`, for all four classes. Repair binds the **old predicted class lost** by an accepted REGRESS; preservation binds the **new predicted class established** by an accepted FIX.

Let `Yold,Ynew` be detached one-hot predicted classes, using the baseline's fixed argmax tie rule. Qualified events require a changed predicted label, the relevant unique local argmax audit class, and a development-selected confidence threshold; tied audit maxima produce no event. Set `f=pi_FIX*chi_FIX` and `r=pi_REGRESS*chi_REGRESS`; these are predicted, detached maps. Compute B1 at the defined image grid, then downsample each class map by area weights for state reads:

```text
K_new^k = (1-r)*(1-f)*K_t^k + f*Ynew^k
R_new^k = clip((1-f*Ynew^k)*R_t^k + r*Yold^k, 0, 1)
n = 1[any qualified FIX or REGRESS in the row]
W_R = C_t^R + n*(C_draft^R-C_t^R + Psi_R(H,P_t,P_candidate,R_new-R_t))
W_K = C_t^K + n*(C_draft^K-C_t^K + Psi_K(H,P_t,P_candidate,K_new-K_t))
```

Each `Psi` is multiplied by its own ledger delta, so its value is exactly zero for zero delta. Commit `W_R,W_K,R_new,K_new` only with the globally accepted annotation; the write affects the next proposal. With `f=r=0`, the persistent correction state and ledgers are unchanged. The annotation can still accept a confidence-only improvement. Store a last-touch identifier per class/map/location; compressed B1 confidence summaries cannot attribute or replay every older contribution and must not be described as event-resolved provenance. The two latent readers remain separate and consume their corresponding class ledger; decoder cross-talk is allowed without erasing the write interfaces.

Accepted transitions can contain local regressions even when net quality improves; those are the intended repair signal. A rejected transition writes no accepted history and terminates. Predicted class obligations can be wrong, so preserve a soft penalty rather than an absolute class lock, bound history to the rollout, and test expiry/confidence on development data. A generic `r*V_R` / `f*V_K` soft accumulator without class binding is an ablation, not recommended B: soft noise can erode memory on no-change steps. Mandatory controls include exact B1 ledgers with one equal-parameter ConvGRU and identical losses, B1 ledgers without latent branches, two generic branches, and semantic permutations.

Train B with a **paired future-utility and previous-fix destruction objective**, beyond ordinary segmentation supervision. At an accepted **nonterminal** training prefix with at least one remaining attempt, clone the exact same `H,A_t,e_t,t` and remaining horizon into two arms; only memory differs: true `S_t` versus a specified intervention `I(S_t)` such as the no-history state. Let `P^+` and `P^-` be the next proposals and `ell_seg` mean CE + soft foreground Dice loss. Then:

```text
L_util = 1[valid accepted nonterminal history] * max(0, m_u + ell_seg(P^+,y)
                                             - stopgrad(ell_seg(P^-,y)))
L_destroy = mean_{v in F_live(t)}[-log P^+(v,y_v)]
L_B = L_seg + lambda_u*L_util + lambda_d*L_destroy
```

`F_live(t)` is the GT-derived set of previously accepted wrong-to-correct pixels that remain correct before this proposal; its exact recurrence is in `experiment_plan.md`. An empty set contributes zero loss and an explicit unavailable/zero-support flag; it is **not** a measured zero destruction rate. Use `m_u=0` initially; a positive margin is an optional registered ablation, since history need not help every prefix. Set proposed normalized auxiliary weights `lambda_u=lambda_d=0.1`, with just one sensitivity each at 0 and 0.3 on model-selection patients.

Reduce the incentive to worsen the comparator by detaching its paired-loss target and supervising its segmentation on interleaved training batches. Stop-gradient does **not** prevent shared parameter changes from degrading that comparator over time: monitor absolute true-arm and null-arm segmentation/gain throughout training, and reject an apparent utility increase explained by null-arm deterioration. Expose no-history states during training so reset interventions are not merely unfamiliar inputs. Early B tests freeze common H/A0 and use the same proposal trunk in both arms, disabling stochastic layer differences. Utility and destruction targets use GT during training; neither GT masks nor future targets enter the memory writer at inference. CE on future predictions is a differentiable surrogate; the claimed benefit must pass the hard-label, same-annotation interventions in the companion plan, especially the exact B1 writer plus one matched GRU and identical losses.

### C: accepted sparse additive events and restitution

Use [candidate_C.md](candidate_C.md)'s **same-action, same-pre-state residual replay**, integrating the existing CC mechanism; keep it high risk until simpler coordinate-only CC and B establish headroom. Maintain an additive latent correction accumulator `C_t=C0+sum_j delta_j`, actual retained logits `A_t`, and B's class ledgers. Only the most recent globally accepted **ordinary** event is replay-eligible. Record its pre-state, pre-annotation, prior evidence, factual support, latent increment, logit residual/gate, weight identity, dtype, normalization mode, turn/depth and batch identity. Earlier contributions remain accumulated but are not individually editable.

For the frozen recorded action, let support coordinates be `s` (not the probability symbol P):

```text
(delta(s), residual(s)) = F_theta(H,S_before,A_before,e_before,s)
C_t = C_before + delta(s_factual)
A_t = A_before + residual(s_factual)
delta_cf = delta(s_counterfactual) - delta(s_factual)
residual_cf = residual(s_counterfactual) - residual(s_factual)
C_proposed = C_t + beta*delta_cf
A_proposed = A_t + beta*residual_cf
```

The factual replay must reproduce both recorded increments before any support intervention is eligible. At beta=1 and with no intervening accepted action or nonlinear transformation of the accumulator, this replaces that immediate action's additive contribution. Beta<1 is interpolation. It does not reconstruct an alternative full history. Keep the class ledgers outside the additive accumulator and update B1 from the **actual current-to-restituted transition**, after its official audit accepts; a hypothetical historical ledger requires a different audit and is excluded. Replaying a nonlinear decoder from the recorded pre-state is allowed; asserting that decoding the edited accumulated state equals that historical residual is not.

The pilot has three maximum attempted turns and at most one ordinary event per accepted ordinary turn. Sparse support uses predicted quantities only; a proposed 5% active-grid budget is an ablation selected on development data and must be reconciled with the canonical support operator before implementation. Sparse sampling does not guarantee sparse output influence. Store indices, latent payloads, full action snapshots and actual replay graphs in the memory ledger; a dense masked implementation earns no sparse-memory claim. Bound realized support displacement and both latent/output innovations, and protect predicted previous FIX in output space. No GT selects inference coordinates or events.

Use the existing CC bound of one coordinate-gradient evaluation and at most two candidate **feasibility** checks initially, with a frozen auditor and fixed weights for the pair. These inner checks compare replay logits with recorded FIX/output/support constraints and make no official Auditor calls. Exactly one final official current-to-selected-candidate audit controls commit (or one final current-to-ordinary-fallback audit after a failed replay). Count every historical writer/decoder forward, backward and that final audit; extra alternative-candidate audits are excluded from this protocol and require a separate budgeted ablation. Clear replay eligibility after accepted restitution. Stale/numeric replay failure triggers a declared ordinary current-state proposal plus its final audit, with failed replay and fallback work both counted. Rejection HALTs.

Train C with annotation/auditor losses and B's paired future-utility/destruction targets. A normalized L1 event penalty is an optional sparsity surrogate, not proof of sparse runtime. The essential latent-transfer pair retains the **same accepted restituted A** in both arms, while one commits `C_proposed` and the other preserves its prior latent accumulator; subsequent proposals then test whether latent transfer helps. Use only nonterminal accepted prefixes. Event deletion with a changed A is a separate restitution comparison against no-op, direct rollback and equal-compute coordinate-only CC. Also compare A/B with ordinary deterministic snapshot replacement from the same saved pre-state and changed support, under the same output constraints and evaluation budget: immediate-event replacement at beta=1 can be algebraically equivalent to that operation. Additive bookkeeping therefore has no standalone novelty claim. A fixed linear renderer with direct event subtraction is only a simplifying C control, not the recommended replay definition or another architecture. Older-event editing, descendant recomputation and recursive replay are excluded from this deployment proposal. Invalidate records after any weight update and between rollout calls/patients.

## 5. Early-auditing curriculum

Use a common, finite exposure budget `U` measured in training-patient presentations, optimizer updates, and processed slices, not epoch count alone. For the initial alternative schedule, map 130 nominal epoch-equivalents to the same number of labelled presentations as the current 130 epochs, with the additional auditor/paired-forward MACs reported separately. No stage change is justified by a test score. Times below are fixed proposal defaults; a readiness failure retains the previous safe training mode through the remaining allotted budget and is recorded as a failed curriculum gate.

| Stage | Proposed interval | Rollout and feedback | Trainable components and transition source |
|---|---|---|---|
| S0: shadow audit | first 15% of U (about 20/130 epoch-equivalents) | Full annotation refinement is supervised without audit control. Auditor observes detached transitions but cannot reject, stop, or condition the expert. | Train annotation system from the start; train auditor from measured train-only on-policy and synthetic transitions. Preserve A0 supervision. |
| S1: detached predicted feedback only | next 55% (about 71/130) | Continue fixed training rollout; expose previous **predicted** audit maps after explicit commit-all training transitions. Feedback is detached. | Train expert/state and auditor in separated losses/gradient paths; progressively replace synthetic audit training pairs with actual model transitions. |
| S2: optional soft commit | next 10% (about 13/130), default OFF | If enabled, use detached scalar soft acceptance and the explicit soft-state dynamics below. Never label these trajectories deployable. | Train with the changed annotation distribution; compare with a no-soft arm using S1 for this interval. |
| S3: hard policy | last 20% (about 26/130) | Predicted hard accept/reject; first rejection HALTs; at most three attempts. | Joint low-LR fine-tuning on accepted histories; retain separate train-only ungated pair generation to prevent auditor starvation. Final selection uses the same hard-policy validation protocol. |

S0 avoids random early hard rejection while the annotator and auditor are untrained. Synthetic training counterfactuals cover local partial repairs, mixed repair/regression, neutral changes, erosion/dilation, missing structures, and class swaps; labels come from actual GT-measured effects. Propose an auditor-pair mixture of 50% actual model transitions / 50% synthetic at S0, moving to 80% / 20% in S1-S3. These are sampling proposals, not class frequencies. Report each source separately and validate on fresh model-generated pairs. Use patient-uniform sampling, then phase/slice sampling, to avoid giving deep volumes more training weight by accident.

S1 training commits are designated **commit-all training transitions**, not predicted accepted transitions. Its predicted feedback teaches the expert to interpret audit semantics without allowing an untrained threshold to extinguish learning. Separately generate hard-policy shadow traces to observe the accepted-history distribution. Mix zero-history and predicted-history training inputs with a fixed 50%/50% schedule for the first half of S1, then 25%/75%; never replace feedback with GT audit maps. History omission is an input intervention, not random hard rejection.

For optional S2, define `w_t = stopgrad(sigmoid((q_t-tau_train)/T))` and `A_next=(1-w_t)A_t+w_t*C_t`; interpolate A/B continuous memory with the same scalar. Proposed temperature decreases linearly 0.1 to 0.02 in Dice-score units, using the documented auditor target scale. This fractional logit commit differs from hard selection and can hide bad candidate decisions. Re-audit the actual previous-to-blended transition before writing its local predicted evidence; original candidate FIX/REGRESS maps do not automatically describe the blend. Count this extra audit. Do not enable S2 for C's event protocol in the first study: fractional event weights create a different semantics requiring its own replay contract. The default curriculum skips S2, extending S1 by 10%.

S3 uses `tau_train` chosen once from the model-selection subset before hard fine-tuning; the independent calibration subset remains unused. Proposed readiness checks on model-selection patients: both beneficial and harmful classes observed in fresh real transitions, patient-bootstrap lower confidence bound of beneficial AUROC above 0.5, finite local/global outputs, and no zero-coverage collapse. These checks are proposed diagnostics, not clinical safety claims. With no evidence for a working gate, finish the run in shadow/feedback mode and report S3 unavailable; do not invent random accepts/rejects to force coverage. Candidate-supervision and auditor-pair auxiliary branches may continue without controlling the deployment trajectory, with their costs and losses identified.

After the final weights/checkpoint are frozen on model-selection patients, fit the global score calibration/acceptance threshold once on calibration patients, then freeze everything for evaluation and M&Ms. A different final threshold changes the visited state distribution; calibration must rerun full sequential hard-policy trajectories for each threshold, not merely filter a stale transition list. Report the final-versus-training threshold, coverage and turn-distribution shift. No GT at inference includes no oracle choice of event, temperature, threshold, or stopping point.

## 6. Practical optimization and data-efficiency safeguards

Keep the baseline loss/augmentation recipe as a common control, then test curriculum changes one at a time. For compact random stems, propose GroupNorm or a fixed alternative suitable for small physical batches, AdamW at `3e-4`, 5% linear warmup, cosine decay, weight decay `1e-4`, gradient norm clipping at 1, and effective batch size 16 slices using accumulation if required. Preserve the pretrained baseline's lower encoder learning-rate group; do not handicap it by forcing a scratch learning rate. These settings require model-selection validation and are not claimed to be optimal. Report normalization and real physical batch size because accumulation does not reproduce BatchNorm statistics.

Use CE + soft foreground Dice on A0 and candidate states with the baseline stage weights retained for the fixed three-turn control. Give each patient equal expected sampling probability. Keep joint geometry consistent across three input slices and the mask; any added intensity/noise/bias-field augmentation is fit and selected on ACDC alone and must be applied to every matched arm. Neighbouring slices are context, not extra independent labels.

Freeze H/A0 in the first correction-mechanism experiments. Later compare complete jointly trained systems, recording initial quality and final-minus-initial gain. Small-model convergence needs a separate adequacy check: the main comparison matches training compute, while one registered scratch extension (2x updates, reported separately) distinguishes a slow-convergence explanation from a finite-budget failure. Natural-image evidence for longer scratch training does not establish feasibility in a small medical cohort. [He et al., ICCV 2019](https://openaccess.thecvf.com/content_ICCV_2019/html/He_Rethinking_ImageNet_Pre-Training_ICCV_2019_paper.html).

SSL pretraining updates, auxiliary heads, optimizer states, extra auditor passes, memory interventions, and replay work all count in the training resource ledger. External ImageNet pretraining is disclosed as external data/compute rather than assigned zero scientific cost; downstream compute comparisons report it separately. An EMA or teacher copy, if added later, must be counted and cannot silently enter the proposed system budget.

## 7. Analytical total-system 5/10/15M envelopes

**The following are hand-derived feasibility estimates and allocation ceilings, not measured model sizes or performance.** A tier is a maximum total parameter budget, not an encoder-only label. Count all encoder, FPN/A0, correction/state, auditor, normalization, embeddings, calibration, and training-only learnable heads; report deployment and training totals separately, each under the declared ceiling in a strict-tier experiment. Parameter sharing counts once; event caches are memory, not weights. The native pretrained ConvNeXt control is retained outside these compact tiers and must have its actual complete count measured before comparison.

Use the shared reference design in [state_formulation.md](state_formulation.md): four stages at strides 2/4/8/16 and widths `[w,2w,4w,8w]`, each with a stride-2 3x3 input convolution and two two-convolution residual blocks. Ignoring normalization/biases, `P_encoder=3438*w^2+27*w`. Four lateral projections cost `15*w*d`; the A0 head costs `9*d^2+4*d`. These architecture choices are illustrative capacity controls, not a selected replacement encoder.

For A, input width `d+10` and hidden width `d` give a dense 3x3 ConvGRU with `27*(2*d+10)*d` kernel weights. The stride-4 readout/decoder costs approximately `40*d^2+5*d` and the shared stride-8 auditor `37.5*d^2+18*d`. Summing these terms, with `(w,d)=(32,64),(48,96),(60,112)`, independently reproduces the reference values below. Small normalization, interpolation, evidence embedding, and auxiliary-head terms are omitted from these raw formulas; the explicit reserve row prevents claiming those omitted terms are free.

| Analytical component or total, millions | 5M regime | 10M regime | 15M regime |
|---|---:|---:|---:|
| Encoder | 3.521 | 7.922 | 12.378 |
| Lateral projections + A0 | 0.068 | 0.152 | 0.214 |
| A GRU + decoder | 0.403 | 0.893 | 1.210 |
| Auditor | 0.155 | 0.347 | 0.472 |
| **Raw A total** | **4.147** | **9.315** | **14.275** |
| Proposed omitted-term reserve | 0.100 | 0.150 | 0.200 |
| **A planning total including reserve** | **4.247** | **9.465** | **14.475** |
| B extra state/semantic writers | 0.200 | 0.500 | 0.700 |
| **B planning total including reserve** | **4.447** | **9.965** | **15.175: exceeds cap** |
| C extra event/support heads beyond B | 0.200–0.600 | 0.200–0.600 | 0.200–0.600 |
| **C planning total including reserve** | **4.647–5.047** | **10.165–10.565** | **15.375–15.775** |

These totals do not justify labelling every width combination a <=5/10/15M model. Before any strict-tier experiment, count all modules and reduce `w`, `d`, or the shared residual-block count where needed. For example, reducing only the reference stem from `w=60` to 56 saves about 1.595M encoder weights, which analytically creates room below 15M for the widest C envelope; reducing 48 to 44 saves about 1.265M and similarly creates room in the 10M regime. These changes also affect A0, so use identical reduced stems across matched A/B/C arms and keep the original-width system as a separately identified capacity control. A 5M C configuration must select the lower event-head allocation or reduce width. Actual parameter counts and A0 quality remain prerequisites; no padding or silently excluded auditor is allowed.

At 256x256, the stride-8 GRU alone costs approximately 0.244/0.536/0.725 GMAC per attempted turn and stores 0.125/0.188/0.219 MiB per FP16 hidden tensor. Including the stride-4 decoder and stride-8 auditor yields approximately 1.075/2.404/3.266 GMAC per ordinary A turn, before omitted small operations. Moving that same GRU to stride 4 multiplies its recurrent MACs and state storage by four. The encoder costs approximately 2.66/5.97/9.32 GMAC once; lateral/A0 costs are additional. B's illustrative writers add roughly 0.2/0.5/0.7 GMAC per turn; C's replay overhead cannot be inferred from weights. These calculations use the formulas above and the spatial grids, not timing measurements.

For a complete rollout use `MAC_total=MAC_encoder+MAC_lateral+A0 + sum_attempts(MAC_proposal+MAC_auditor+MAC_state_write+MAC_replay)`, never multiply the encoder by the turn count when it is cached. B may store multiple state/semantic tensors; C must count sparse indices, events, snapshots, renderer buffers and descendant replay. Full-system profiling must include them. Actual latency and peak VRAM remain unmeasured.

## 8. Decision recommendation

Begin with one compact A and the retained ConvNeXt baseline; establish A0 adequacy and matched-start correction headroom before scaling state complexity. B is a testable hypothesis that past accepted transition semantics improve future changes while preserving verified prior fixes. C is worthwhile only if exact replay, semantically useful memory, and bounded restitution cost survive dedicated controls. The companion [experiment plan](experiment_plan.md) defines falsifiers, exact metrics, minimal stages, calibration, paired patient uncertainty, and the evidence required before claiming any benefit.
