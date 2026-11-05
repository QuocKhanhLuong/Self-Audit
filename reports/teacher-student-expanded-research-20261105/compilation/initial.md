# Independent compilation hypotheses — preserved pre-evidence

Date: 2026-11-05. Investigator: compilation, dispatch ctx_a3fa3fd23a1c.
This document was written before receiving a literature packet, reading any prior council reports, or performing a literature search. No novelty assertion is made. Source boundaries read: `/Users/alvinluong/Self-Audit/ACTIVE_PIPELINE.json` and `docs/METHOD.md`; source HEAD verified as `9958d421ccae1da3b8527f74da5438dbb9f3f521`.

## Shared scientific contract

These are prospective protocols, not patches to the current joint-guard trainer. All dense masks, including meta-training/audit masks, belong to the declared TRAIN patient budget. U supplies images only; VAL/TEST masks supply neither routes, optimizer training targets, program selection nor gates. A fixed teacher trained under that budget is required; its strength and any 96 Dice aspiration remain unverified. Same original image x enters teacher and student encoders during training; no time series, neighboring slices, alternate modalities or multiview information is assumed. Deployment is x -> student only. Comparators share labeled/U schedules, initialization, pseudo-class normalization and compute accounting. Teacher errors are not ground truth.

## Idea A: a compiler produces executable segmentation programs

**Transferred object:** a short typed program, not a teacher activation tensor or logit target. Let teacher encoding t=E_T(x), student encoding s=E_S(x), and joint same-image interaction C=s^T W t / sqrt(d) after an explicitly learned, spatially aligned projection. A training-only compiler q_phi(pi|C) chooses a program pi=(a_1,...,a_K) from a finite library: local edge-conditioned transport, global low-rank token exchange, class-slot competition, and boundary readout. Every program operates on student tokens alone. The student router p_theta(pi|s) learns compiler decisions; the executable model is y=D_pi(s), with pi sampled/hardened from p_theta during both training and inference. There is never a teacher-feature bypass into D.

One possible constrained objective is

    min_phi,theta E_L[ell(D_{pi~q(C)}(s),y)]
       + beta E_{L,U}[KL(q_phi(pi|C)||p_theta(pi|s))]
       + lambda E_q[cost(pi)],
    subject to cost(pi)<=B and a fixed K at evaluation.

Compiler actions must be chosen by their *student execution consequence*, not similarity to teacher attention maps. Teacher evidence can additionally constrain candidate programs by teacher functional responses to declared image-preserving perturbations; that branch is a separate factor. To avoid an oracle training/deployment gap, evaluate the student-router path throughout training and report its loss separately. Compiler-distribution entropy is not confidence in anatomy.

**Architecture:** a constant-resolution token lattice, 3-4 semantic slots, sparse image-edge transport, and a small differentiable coordinate readout. No contracting/expanding U, convolutional skip pyramid, or residual image predictor. K rounds use tied primitives with input-dependent discrete ordering. Local fine-grid tokens retain thin boundaries; global slots pool context without full quadratic attention. The integer sequence is a first-class auditable artifact.

**Assumptions:** a small vocabulary captures a useful portion of the teacher's computation; x contains enough information for the router to predict the useful program; the compiler does not encode labels into an unrestricted per-image program. Restrict program entropy/capacity and use patient-disjoint TRAIN episodes.

**Predicted outcome:** benefit should concentrate on images requiring different orders of local versus global reasoning, survive replacing soft program mixtures with executable hard programs, and disappear when per-image program assignments are shuffled at matched compute. It should improve thin-MYO boundaries without merely increasing depth.

**Cheapest decisive test:** a small manufactured segmentation family with two image-observable regimes whose correct solution requires opposite operator ordering. Compare the learned compiler/router, a fixed best program, an image-only compiler of equal training compute, and shuffled programs, with identical primitives and K. Hold out combinations of appearance and regime. Success means route-specific causal generalization, not aggregate loss alone. This is FIXTURE_ONLY evidence; it cannot establish cardiac improvement.

**Killer counterexample:** for an input family where the best program is constant, q merely supplies regularization or extra optimization; matching gains from the image-only compiler or fixed program kills the claimed transfer of algorithmic selection. If useful teacher programs depend on information discarded by s, no student router can reproduce them. An unconstrained compiler can encode teacher predictions in program bits, turning the method into covert output distillation.

## Idea B: transfer a reusable training algorithm, not per-image predictions

**Transferred object:** a small learned update rule A_psi, trained from a frozen teacher's evaluation of short student-learning episodes. Teacher/student same-image joint coordinates C above serve a *training-only meta-critic* that scores which parameter subspaces learn useful relations. A small student is trained by updates

    theta_{k+1}=theta_k - eta_k P_psi(z_k) g_k,
    g_k=grad_theta ell_L(theta_k),
    P_psi(z)=B diag(softplus(h_psi(z))) B^T + epsilon I.

Here z_k contains deployable training statistics (student gradient norms, activation moments, step index), never U masks; B has a fixed low-rank/block-sparse structure. The meta-objective scores the final student's *held-out-within-TRAIN* supervised loss plus a teacher-derived functional criterion:

    min_psi E_episode[ell_Lquery(theta_K) + alpha R_T(theta_K; X_U)],

where R_T may evaluate how teacher/student joint relations support student decoding, with the exact choice predeclared. Any feature/output discrepancy in R_T is a meta-critic, not the claimed contribution: the hypothesis is that the learned update algorithm transfers across patient episodes and seeds. At ordinary student training A receives only z and g; teacher and C are absent. Deployment is the resulting E_S and D only.

**Architecture:** test the same non-U token/slot student as A so optimizer transfer can be isolated; later investigate a block-sparse compact operator network without changing the optimization claim.

**Assumptions:** useful optimization geometry is stable across TRAIN patient episodes and new initializations; the meta-training budget is not large enough to memorize patients; teacher information adds more than an ordinary validation-guided learned preconditioner. Query labels remain TRAIN labels and are fully counted. This replaces the current optimizer/guard timeline and needs a distinct protocol, not a claim of inherited audit guarantees.

**Predicted outcome:** at equal student update count and equal total training compute, a frozen A_psi accelerates learning or improves final sample efficiency for newly initialized students on held-out TRAIN episodes. A transfer that only helps its original student initialization is parameter fitting, not reusable learning-to-learn.

**Cheapest decisive test:** small artificial tasks with identical segmentation algorithm but rotated input statistics; meta-train on some rotations and initializations, freeze A, and compare on unseen combinations with tuned diagonal/block preconditioners, teacher-free meta-learning, and extra ordinary optimizer steps using the same total FLOPs. Measure loss-versus-compute and out-of-episode performance.

**Killer counterexample:** arbitrary teacher gradients across episodes, or teacher wrong on shifted anatomy, makes no reusable update rule identifiable. If A learns only a learning-rate schedule, a schedule-only control should tie it. If all gain vanishes after matching total meta-training compute, reject the compute-efficiency claim while retaining only a narrowly stated amortization hypothesis.

## Idea C: nonlinear circuit quotient with shared operator atoms

**Transferred object:** teacher parameter/operator structure. For teacher blocks h_{l+1}=sigma(W_l h_l), seek sparse atom expansions W_l≈sum_r a_lr U_r V_r^T and identify groups of blocks with common gates. Compile a compact recurrent network using the shared atoms and a bounded input-dependent gate program; do not collapse sigma across noncommuting layers. Joint teacher/student encodings of x estimate which atom interactions actually occur on TRAIN/U, defining a co-activation tensor H_{l,r,m}=E_x[g_l,r(t(x)) g_m(s(x))]. H determines shared parameter groups and allocation of a finite atom budget; final student forward uses its own gates and atoms.

The structural optimization is

    min_{atoms,assignments} sum_l ||W_l - sum_r a_lr U_r V_r^T||_F^2
      + lambda number_of_atoms + mu predicted_operator_cost,

followed by budgeted TRAIN supervised fitting of the compiled circuit. A more faithful activation-conditioned operator norm is a separate predeclared objective. Teacher weights alone do not identify the needed student width; sparse gating must preserve nonlinear region boundaries.

**Assumptions:** teacher blocks possess reusable low-dimensional operator structure and restricted activation patterns on the target image manifold. Equivalent parameter permutations/scalings must be aligned before making shared-atom claims. Incompatible teacher/student widths require explicit maps, not a silent claim of exact copying.

**Predicted outcome:** shared nonlinear atoms yield better parameter efficiency than a teacher-free compact circuit at equal depth and multiply-adds, and better accuracy than independent per-layer low-rank truncations at equal parameter count. Benefit should correlate with measured atom reuse, not just teacher accuracy.

**Cheapest decisive test:** a small known teacher synthesized from reused atoms plus controlled independent components. Sweep reuse strength and hold out activation regions. Compare recovered shared circuit, independent layer factorization, random atom initialization, and a compact network trained directly. Report operator error and function error separately.

**Killer counterexample and limitation:** two individually low-rank matrices separated by ReLU can generate a function requiring many distinct activation regions; replacing their composition by one matrix is generally impossible. Noncommuting gates defeat coefficient averaging. A teacher can hide a narrow but clinically important decision boundary in a low-variance direction, so low Frobenius error can coexist with complete MYO failure. This idea may reduce to existing network compression; prior art is required before retaining any novelty hypothesis.

## Independence checkpoint

The three objects are a per-image executable program, a reusable training update algorithm, and a shared nonlinear parameter circuit. None uses residual correction, teacher dropout/withdrawal, progressive grafting, or a matching loss as its central proposed novelty. Preferred risky lead before evidence: A, because its intervention test can distinguish learned computational order from mere extra computation. B is highest risk for cost and leakage; C is the most likely to be subsumed by compression prior art. No experiments or patient claims have been made.
