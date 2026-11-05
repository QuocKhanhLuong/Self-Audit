# Computation transfer council dossier

**Decision:** retain teacher-informed program induction as a risky, falsifiable lead; retain learned optimization as a separate alternative; reopen a single fixed compiled circuit. None is an established novelty or patient result. The preserved initial draft has SHA256 `262b1ebef0b587b6af5ef9c1e7c725b4672ac6306993598bdf7f139e48b36426`.

## Scope and teacher requirement

Source HEAD was verified as `9958d421ccae1da3b8527f74da5438dbb9f3f521`; ACTIVE_PIPELINE.json and docs/METHOD.md define the scientific boundaries. These proposals need separate protocols; they inherit no guarantee from joint_guard. All dense labels, including compiler/meta-training query labels, count toward the TRAIN patient budget. U supplies images only. No U/VAL/TEST masks may select programs, gates, label subsets or optimizer targets. Same original image x enters both encoders; any additional cine, neighboring-slice or full-volume context requires a separate declared input protocol. Student-only deployment is primary.

A strong teacher is a prerequisite, not an accomplishment of this dossier. First establish a reproducible supervised teacher under the same label/input budget; a full-TRAIN teacher is a separate comparison. The user’s 96 aspiration needs domain-specific foreground RV/MYO/LV, patient/phase aggregation and native-grid evaluation specified before results. Direct XML inspection: ABR Table 3 reports 96.03 with BG; its rounded foreground entries average **94.74**, while Table 10 prints 94.73. CASNet Tables 2–3 report 96.3 ACDC and 95.7 **M&Ms-2**, with insufficiently specified averaging and inconsistent split descriptions for our target protocol. Neither establishes 96 foreground Dice on both ACDC and M&Ms. All numbers are author-reported, not reproduced.

## Lead: teacher-informed induction of executable operator order

**Mathematical object:** a bounded program over typed operators, accompanied by estimated task regret. This is not exact decompilation of an arbitrary teacher. On aligned spatial locations define

    s_i=E_S(x)_i,  t_i=E_T(x)_i,
    C_i=(P_S s_i)(P_T t_i)^T.

C is a factored, same-image cross-encoder interaction field; spatial alignment is explicit interpolation using declared image geometry, never masks. A training-only compiler Q(C) predicts the consequences of six permutations of three fixed-shape primitives: local image-edge transport L, global token exchange G, and class-slot competition Cmp. Each permutation uses every primitive exactly once, giving identical operation counts; the order can matter because LG need not equal GL. Teacher vectors control executable choices rather than supplying decoder content.

**Compact architecture:** one fixed-resolution patch lattice, a few global semantic slots, the three shared operator modules, and a coordinate readout combining final state with local image samples. There is no encoder/decoder resolution pyramid or skip concatenation. Native local samples are necessary because a coarse patch lattice can erase thin myocardium. With N tokens, width d, m slots and local degree k, rough cost is O(Nd²+Nkd+Nmd), plus encoding, routing and native-query readout; these latter costs must be measured. Small parameter count alone says nothing about latency, activation memory or kernel overhead.

For each program pi, execute

    y_pi=D(O_pi3 O_pi2 O_pi1 E_S(x)).

With decoder/primitives frozen during target construction, labeled TRAIN images provide counterfactual regrets

    R_pi(x,y)=ell(y_pi,y)-min_r ell(y_r,y).

Fit Q to predict the regret vector from C using patient-disjoint TRAIN folds. On U, Q supplies **predicted**, unverified regrets. Train a student router p(pi|s) against these predictions and directly against supervised task loss; the latter is essential. For labeled cases, expected regret sum_pi p_pi R_pi avoids penalizing two equally useful programs merely for differing IDs. On U, using Q’s regret predictions remains fallible pseudo-supervision. Refresh targets only from a fixed current student snapshot; changing primitives invalidates old counterfactual meaning. The initial KL-only formulation is therefore insufficient.

**Full information graph:** labeled TRAIN trains teacher E_T and compact student primitives; x→(E_T,E_S)→C→Q→program-quality targets; x→E_S→student router→hard program→student decoder→TRAIN loss. U follows the same image branches without y. The teacher has no decoder-content bypass. Train and evaluate the hard student route throughout; a successful teacher-routed branch does not establish deployability. At inference: x→E_S→router→one hard permutation→D→segmentation. Q, E_T, labels and program search are absent. This is not progressive grafting, withdrawal or a residual image predictor.

**Prediction and limit:** benefit should occur when the same small operator vocabulary solves different observable regimes through different ordering. Teacher information is a deterministic function of x, so it contributes optimization/representation knowledge, not new observability. If two images collapse to the same s but require different programs, no router on s can identify both. Six route choices also can encode coarse mask archetypes; few bits do not prove algorithmic knowledge.

**Cheapest decisive test:** freeze analytically noncommuting operators A/B and a fixed readout; construct AB-required, BA-required and AB=BA regimes, with regime cues visible to the student. Hold out appearance/regime combinations. Compare teacher compiler, label-only hard router, equal-total-compute image-only compiler, best fixed order and shuffled assignments. Inspect task error and AB↔BA interventions, not route accuracy. Then unfreeze primitives as a separate factor. If one fixed order absorbs both regimes after unfreezing, report that the ordering explanation failed. If the image-only compiler ties, reject teacher-specific algorithm transfer. Compare ordinary output/feature distillation on the identical student and budget before claiming better transfer. Synthetic success would be FIXTURE_ONLY; a small prelocked cardiac pilot requires separate authorization.

**Failure case:** a teacher’s scanner shortcut selects global pooling before local discrimination, erasing a pathological thin-MYO boundary. Routing confidence cannot certify anatomy. Report per-domain/per-class boundary and volume metrics, including pathology strata where support permits, alongside Dice and actual inference cost.

## Distinct alternative: transfer a reusable optimization algorithm

Transfer an update rule rather than a per-image program:

    theta_(k+1)=theta_k-eta_k P_psi(z_k) grad ell_L(theta_k),
    P_psi(z)=B diag(softplus(h_psi(z))) B^T+epsilon I.

B is a fixed compact block/low-rank basis; z contains student gradients, moments and step index. A training-only meta-critic receives the same-image C and evaluates short learning episodes; optimize psi against query-label loss from other budgeted TRAIN patients, with any teacher-derived response criterion separately declared. After freezing psi, fresh students train using only z and labeled gradients; deployment uses only their encoder and decoder. U may enter the teacher meta-critic through images, never masks. This changes the optimization timeline and has no inherited audit guarantee.

The hypothesis is reusable geometry across patients and initializations. Test unseen TRAIN episodes and seeds against schedule-only, diagonal/block preconditioning, teacher-free meta-learning and extra ordinary optimization at equal total compute, including meta-training. If gains require the original initialization, call it parameter fitting. If gains disappear after compute matching, reject efficiency claims; any amortization claim must name the number of later students needed to repay meta-training cost. The supplied source packet does not cover learned-optimizer prior art, so originality is **unassessed**, not inferred from absence.

## Reopened alternative: a fixed nonlinear compiled circuit

Initially I preferred dynamic order and considered shared-atom compression derivative. The program-extraction source and peer challenge reopen a simpler possibility: find one bounded fixed program or shared nonlinear atom circuit that captures useful teacher computation. Teacher weights may initialize shared factors W_l≈sum_r a_lr U_r V_r^T; same-image C identifies which atom interactions matter to student decoding. Fit the resulting fixed circuit on budgeted TRAIN labels; deployment is x→student gates/shared atoms→coordinate readout, without teacher or router. Compare independent per-layer factorization and teacher-free initialization at identical depth, parameters and FLOPs.

This is plausible parameter/computation transfer, not a fresh compression claim. Nonlinearities must remain explicit: ReLU(1−ReLU(x)) has two breakpoints and cannot generally be represented by one affine-ReLU unit. Low individual matrix rank does not justify folding nonlinear compositions. Rare low-variance directions may carry MYO boundaries, making small weight error clinically irrelevant. The decisive manufactured test varies known atom reuse and holds out activation regions; separate weight reconstruction from function error. Reopening this route means accepting that dynamic selection may be unnecessary, not redefining a failed router as success.

## Closest inspected sources and what changed

[D-RASP, sections 3.1–3.2 and 6](https://www.alphaxiv.org/pdf/2602.08857?page=4) already converts GPT-2-style networks into executable primitives and simplifies them through causal pruning/replacement. Its exact theorem assumes exact linear layer-normalization substitution; the expanded program can be exponential. Consequently I removed any generic exact-small-compilation interpretation. Our possible distinction is constrained same-image program induction with student execution and order-intervention generalization, not executability itself.

[Codebook-guided CMKD, sections 3.3–3.5](https://www.alphaxiv.org/pdf/2609.37243?page=4) already selects teacher codes by task relevance and student compatibility. Discrete anchors or student-friendly selection are insufficient novelty. Here program choices must change computation; merely quantizing C would collapse toward that prior mechanism. [OctreeNCA, sections 3.1–3.3](https://www.alphaxiv.org/pdf/2508.06993?page=5) already supplies compact recurrent local segmentation rules and coarse-to-fine context. Our no-pyramid student is different structurally, but neither recurrence nor non-U-Net form is novel by itself. Its custom CUDA implementation also prevents borrowing its reported speed advantages.

## Peer exchange and evidence status

Dynamics challenged route identifiability and primitives absorbing order. **Accepted:** freeze semantics for the first test, include equivalent-program regimes, use task regret and causal interventions, and keep the fixed circuit alive. Their suggested tolerance-based equivalence requires correction: pairwise epsilon-closeness is not transitive. Use finite intervention-relative admissible sets, or an explicit exact partition; do not assert global program equivalence. The peer accepted this correction. My outgoing review challenged their finite-step solver versus claimed implicit optimum: they now specify a four-step normalized Richardson decoder, with a stated contraction bound and equal-matvec/image-edge rejection controls. I read their final response and accept the mathematical clarification; anatomical transfer remains unproven. Uncertain graph mixtures remain an optional later test. Agreement is not evidence.

Requirement accounting: the lead proposes joint same-image representation, computational transfer and a genuinely different compact decoder; it does **not** establish a strong teacher or superiority. The optimizer route addresses transfer but needs the separately specified compact architecture. The fixed circuit addresses compression/deployment while its useful joint representation remains the weakest part. All efficacy, speed and 96 claims are **NOT_RUN/UNVERIFIED**. Primary-text inspection and source HEAD are verified; no training, weights/data download, code edits or repository tests were performed. This bounded six-source packet is not novelty certification.


Primary-text files inspected directly: `papers/2602.08857.txt`, `papers/2609.37243.txt`, `papers/2508.06993.txt` under `/private/tmp/selfaudit-expanded-20261105/`; teacher checks use `sources/ABR.xml` and `sources/CASNet.xml` under `/Users/alvinluong/.local/share/openresearch/files/self-audit/teacher-student-redesign-20261105/`. No model weights were inspected or hashed.
