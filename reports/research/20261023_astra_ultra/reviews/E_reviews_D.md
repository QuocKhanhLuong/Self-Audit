# E independent review of D prior-art dossier

Reviewed: `workers/D_prior_art.md`, SHA256 `b949435be8c8949e3a4fb52b6d2c201a6d25f29e4035181a36887c59ae02d188`, 2026-10-23. Reviewer E / astra_design. Read-only review; no external method or patient run was executed. Independently checked the primary HCO Algorithm 1, CPO Algorithm 1/section 6, and DS3L Theorem 3 text. The GEM/OGD comparison is supported by D's source register and was not independently re-derived in this review.

## Verdict

**ACCEPT the central decision; tighten comparator definitions before implementation.** The broad novelty claims are occupied. What survives is a potentially publishable empirical characterization under an exact supervision/compute protocol, conditional on useful outcomes and a sufficiently informative negative-result analysis. A literal unlocated combination is not positive novelty evidence. D correctly calls this a proposed characterization and correctly denies a new theorem, independent-audit certificate, teacher contribution, or clinical safety claim.

HCO explicitly filters U coordinates using a labeled AdamW direction and updates its moment state; a repository sign rule at class level is not its algorithm. CPO already uses a local proposal and backtracking on sampled constraint estimates, but its policy/KL construction is not the present Brier problem. DS3L has a labeled empirical-risk comparison against a globally argmin-defined supervised estimator; this does not certify arbitrary finite practical iterates or patient Dice. These primary texts support D's mechanism-level collisions and its limits. [HCO Algorithm 1](https://papers.miccai.org/miccai-2026/paper/0945_paper.pdf), [CPO Algorithm 1](https://proceedings.mlr.press/v70/achiam17a/achiam17a.pdf), [DS3L Theorem 3](https://proceedings.mlr.press/v119/guo20i/guo20i.pdf).

## Requested corrections and implementation constraints

1. **Scalar control must match the restricted regularizer (material).** Along alpha=a·1 with m nonempty directions, the joint ridge is lambda·m·a²/2. The scalar comparator must therefore use m·lambda for a strict restriction. A scalar with lambda instead answers a different question. Use sum_jk A_cjk and sum_j b_cj and the identical finite guard; independent per-source coefficients are not this control. Add this explicitly to D's evidence item 2.

2. **Faithful HCO and a controlled HCO-inspired ablation answer different questions (material).** The original HCO evaluates both objectives before its update and retains its moment semantics. Moving U evaluation to the post-supervised theta_s, forcing direct SGD, discarding moment updates, or adding the current finite Brier guard makes an adaptation. Sharing data budget, architecture, label information and reasonable tuning/computation is possible, but exact update timeline equality is not. Compare faithful HCO as a whole training recipe. Use a separately named adaptation only for component attribution. Do not imply that sharing a protected patient/block alone produces algorithmic parity. This constraint follows directly from HCO Algorithm 1 and the inspected engine's split-step contract.

3. **Same-state mechanism contrasts versus trajectory outcomes (material).** Current directions and teacher probabilities cannot remain identical after different policies commit. Replay alternative policies functionally from frozen identical transaction states to identify the local role of curvature/compensation. Fresh full-run endpoints estimate the complete policy effect, including mediated teacher and candidate differences. D's narrow characterization needs both layers or explicitly narrower wording; a single paired final endpoint cannot isolate off-diagonal compensation.

4. **Norm/trial matching needs a predeclared secondary estimand (moderate).** State whether the norm is Euclidean parameter-block displacement, candidate coefficients, or weighted audit-probability response. They differ. Fixing the realized policy displacement post hoc can condition away the benefit the method is supposed to create. A useful secondary mechanism study holds same-state directions and compares predeclared update magnitudes and trial budgets; primary policy utility should retain natural chosen updates with honest total compute. Do not select a favorable matched norm from exposed VAL.

5. **KKT diagnostics need their regularity qualification (moderate).** B's rank/feasibility analysis contains zero-budget feasible problems with no KKT multipliers. Reference objective gaps, feasibility and analytic special cases are decisive; a missing KKT certificate alone is not failure unless the required constraint qualification/multiplier existence holds. Keep numerical liveness failures separate from genuine zero feasible motion and local-to-finite incompatibility.

6. **Stage the controls to keep the test economical (moderate).** D's eight evidence requirements are appropriate before a broad method/utility claim, but should not trigger eight simultaneous expensive training families. First qualify numerical scale and scalar parity. Next use frozen TRAIN transactions to ask whether separate coefficients change useful finite updates at all. Only then run J/S/plain policy endpoints; direct audit-gradient and a first-order-plus-same-guard adaptation are discriminators before expanding an asserted mechanism. Faithful HCO, strong SSL and independent-patient confirmation are required for the corresponding broad claims, not a reason to spend on them before liveness is established.

## What would falsify the remaining contribution

- Joint and scalar give the same effective finite predictions across meaningful multi-source opportunities, or all apparent differences disappear after matching the restricted ridge.
- Joint reduces the reused audit Brier but not independently evaluated patient outcomes, or any apparent gain is no better than additional labeled descent.
- An implementation repair removes apparent reject-all behavior without producing stable policy utility; that is a software finding, not a new learning principle.
- Benefits depend on the fixed L/U phase-lock, one labeled subset, one initialization, exposed checkpoint selection, or a compute advantage given only to J.
- Faithful prior methods or the simple scalar control attain the same useful outcome at lower measured cost. This supports a negative characterization, not an expanded novelty claim.

The current evidence cannot resolve these tests. D's recommendation to continue only as bounded falsification is consistent with E's staged protocol and the four completed, unguarded baseline endpoints.

## Author response and disposition

D accepted the requested scalar-ridge, faithful-versus-adapted HCO, KKT-qualification and secondary norm-matching clarifications. I reread the revised evidence requirements at SHA256 `b949435be8c8949e3a4fb52b6d2c201a6d25f29e4035181a36887c59ae02d188`; those changes are present. The remaining same-state-versus-policy and staged-compute constraints are supplied in E's protocol and should remain explicit in the joint synthesis. Final disposition: **accepted with the stated bounded-research claim; no novelty or efficacy upgrade**.
