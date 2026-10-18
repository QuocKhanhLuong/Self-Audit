# R5 — optimization, rare classes, and pretrained transfer

Independent initial draft, 2026-10-18. Author role R5; requested model `gpt-6-astra`, requested reasoning `ultra`; resolved backend identity is not independently exposed. Pinned source `f00e4faa52204f8e6863f76a493684b68ed3a132`. This draft was written before reading any other dossier or interrupted draft. Preserve this initial text verbatim; append reviews and revisions. AI-assisted literature search, mathematical drafting and local probes; no claim of exhaustive review or human verification.

**Decision position:** first separate missing/incorrect semantic evidence from inadequate optimization. A source/class-aware update policy may prevent avoidable proxy forgetting, but it cannot create named RV truth. Strict historical-loss nonincrease can prevent the very correction required to improve beyond the teacher. The first serious baseline should be normalized, observable stratum diagnostics plus a frozen-encoder head/decoder warm-up; any guard should apply to actual displacement with explicit rollback and a finite regression budget. All candidate changes below are **PROPOSED_NOT_APPROVED**.

## 1. Executed scope and source facts

Read STATE.md, AGENTS.md, ACTIVE_PIPELINE.json, CURRENT.md, RUNBOOK.md and ASTRA_NEXT.md. CodeGraph is absent per shared contract; used Semble semantic search, then direct file reads. Used the scientific-critical-thinking skill. Memory supplied historical pointers only; current facts below were rechecked against the pinned worktree. Read-only production inspection; only this dossier and `probes/R5/` written. No patient data, weights, dataset download, GPU training, evaluation, dependency change, commit or push. Downloaded papers are literature artifacts, not model/data artifacts.

**R5-C01 — CODE_FACT: active gradient paths.**

- Statement: the implemented policy optimizes global weighted Q-CE and full-view feature cosine using ordinary AdamW; semantic inactivity does not freeze semantic predictions.
- Scope/assumptions: pinned source; computational dependency claim, not runtime/patient evidence.
- Formal claim: `features = feature_project(feats[2])`; `coarse_logits = coarse_head(decoder(feats[0:4]))`. Therefore encoder parameters upstream of `feats[2]` can have both feature and semantic gradients. A feature-only update Δθ_E can produce Δz_coarse ≠ 0 even if Δθ_head = 0.
- Proof/evidence: `layer_upgrade/model.py:54-118`; `training_v1/runner.py:194-243`, `:383-388`, `:439-495`; weighted loss implementation `layer_upgrade/losses.py:30-42`. AdamW is constructed on `model.parameters()`, losses are combined, global norm clipping precedes `optimizer.step()`. The detail head has no enabled loss path. `semantic_updates` counts presence of semantic loss, not invariant predictions on inactive batches.
- Counterexample/qualification: when the upstream feature gradient is zero or lies in the coarse-logit Jacobian nullspace, coarse predictions need not move. A separate frozen encoder would remove this path.
- Closest prior: shared-parameter multitask optimization; this is an implementation fact, not a new method.
- Implications: instrument feature-only steps, class/source actual loss changes and encoder/head displacement separately. `head.grad is None` is insufficient to infer semantic stability.
- Limitation/prediction: toy D gives frozen-head semantic probability 0.731059 → 0.750260 from one feature-only AdamW step, but does not estimate the effect in DINOv3. Falsify the practical concern by showing negligible declared-stratum prediction drift on the actual approved schedule.

## 2. Loss measures must be explicit before balancing gradients

**R5-C02 — DERIVED_RESULT: normalized stratum accounting.**

- Statement: a global reliability-weighted mean can allocate negligible mass to RV, while arbitrary per-class normalization changes the objective and can amplify bad evidence.
- Scope/assumptions: frozen targets q_i ∈ Δ^3, weights w_i ≥ 0, fixed memberships m_ji ∈ [0,1], fixed observation geometry. Let Z_j = Σ_i w_i m_ji. A valid stratum requires Z_j > 0.
- Formal claim: ν_ji = w_i m_ji / Z_j and L_j(θ) = Σ_i ν_ji CE(q_i,p_θ(x_i)). If memberships form a partition Σ_j m_ji = 1, then L_global = Σ_j (Z_j / Σ_k Z_k)L_j. Equal group weighting instead defines L_bal = |J_active|^{-1}Σ_j L_j; it is a different target measure. For overlapping memberships this partition identity is unavailable unless memberships are explicitly partition-normalized.
- Proof: substitute ν_ji and exchange finite sums. No statistical assumption is needed for this identity.
- Counterexample: duplicating a source as two equally weighted groups doubles its vote under a naive group average. A tiny wrong RV group can dominate after inverse-mass weighting. Identical correlated views do not increase independent evidence. Effective weighted count n_eff = (Σw)^2/Σw² measures weight concentration, not independence among pixels/cines.
- Closest prior: GradNorm, Group DRO; their objectives concern declared tasks/groups, not the semantic validity of pseudo-groups.
- Code/evidence: current `soft_ce` divides by total transported reliability, not RV mass. `runner._semantic_gate` can omit semantic loss. `prepare_targets` describes transporting Q mass/support together. No source-resolved semantic loss is established by those interfaces alone.
- Implications: log Z_j, participating patients/cines, target entropy, normalized CE and KL, active-step fraction, gradient norm and actual displacement inner product for every declared group. For Z_j = 0 return INACTIVE/NA, not zero loss. Zero valid semantic support must not become BG. A full-Q coordinate q_c > 0 may be diffuse historical mass, not direct named-class support. Source identities require upstream lineage; they cannot be recovered by pretending Q was a source mixture.
- Limitation/falsifiable prediction: balancing will help a pure allocation bottleneck, but cannot separate collided features or repair systematically wrong source identities. Before calling a term “RV risk,” say whether it is (a) full-distribution loss on a fixed RV-evidence stratum or (b) the RV summand −q_RV log p_RV. These are different. The latter alone ignores false RV positives and is not a proper four-class scoring rule.

An inactive batch is not an unbiased zero observation of a group risk. Batch self-normalization E[Σwℓ/Σw] generally differs from the global ratio; changing which groups are active also changes effective weights. Compare fixed TRAIN evidence panels or declared inclusion probabilities and accumulate numerator/denominator separately. Such a panel is optimizer-facing TRAIN data, not unseen validation. Do not assume pixels are independent when estimating variability; use source/cine/patient clusters according to the sampling design.

**R5-C03 — COUNTEREXAMPLE: lower total KL, worse rare stratum.**

- Statement: total KL decrease does not imply RV-stratum improvement, even with fixed teacher entropy.
- Scope/assumptions: manufactured binary BG/RV subproblem; same predicted probability p for two strata due to a deliberately collided representation. q_RV = .02 in 99% mass and .98 in 1% mass. Not an assertion that current DINO features collide.
- Formal claim: moving shared RV logit 0 → −1 yields total KL 0.595108 → 0.244823, while rare RV-stratum KL 0.595108 → 1.195223. The common stratum KL becomes 0.235223. Every number uses the same fixed normalized measures before/after.
- Proof/evidence: direct Bernoulli KL calculation in `probes/R5/run_probes.py`, results `A_total_KL_vs_RV`. Binary example embeds into a four-class problem with fixed positive equal small MYO/LV mass and rescaled BG/RV masses.
- Counterexample/control: with separate logits, weighted gradient is [.4752, −.0048] and RV KL decreases to .592807; pure imbalance slows RV without forcing harmful sign. With equal group weights and collided features, logit zero is stationary and RV KL stays .595108. Increasing RV weight merely chooses another compromise.
- Closest prior: CAGrad §2.1 explicitly distinguishes scale and directional conflict; Group DRO distinguishes average from group risk.
- Implications: distinguish allocation (small gradient), unreliable source (wrong direction relative to latent truth), conflict (competing directions through shared parameters), and representation deficiency (no attainable satisfactory outputs). A single total loss curve distinguishes none of these.
- Limitation/prediction: no Dice/anatomical statement. If reweighting fixes the independent-logit case but fails at collision, the matched probe identifies mechanism rather than “rare-class failure” as one undifferentiated label.

## 3. The optimizer displacement is the object to certify

**R5-C04 — COUNTEREXAMPLE: raw-gradient guards do not survive AdamW in general.**

- Statement: a projection satisfying g_jᵀd ≥ 0 only controls the additive proposal θ − ηd; it does not certify an optimizer-transformed update.
- Scope/assumptions: differentiable fixed proxy L_j, Euclidean parameter block; candidate `d` may already be projected. With standard AdamW (no AMSGrad), actual Δ = −η[m_hat/(sqrt(v_hat)+ε) + λθ], including coordinate scaling, moment history and decoupled decay. The current installed implementation is PyTorch 2.14.0 and its relevant source was captured in `local_AdamW_source.py.txt`.
- Formal counterexample: θ = (0,0), L_guard = .5(1 + [2,1]ᵀθ)², d = (−1,3). Then g_guardᵀd = 1 > 0. First-step AdamW with β1 = β2 = 0, λ = 0, η = .1 gives Δ ≈ (.1,−.1); L_guard increases .5 → .605. No old momentum or weight decay is needed.
- Additional evidence: inherited moment probe increases .5 → .561021; a separate deliberately large decay probe increases .5 → 1.805. These isolate distinct mechanisms, not recommended hyperparameters. Parameter-only rollback changes the next parameter to .950582 vs .813223 for untouched control; restoring parameters AND optimizer state returns exactly .813223.
- Closest prior: [PyTorch AdamW specification](https://docs.pytorch.org/docs/2.14/generated/torch.optim.AdamW.html); GEM Eq. (7)-(8) is explicitly a local additive-gradient construction. `layer_upgrade/guarded.py:1-126` expressly limits its guarantee to deterministic selected-block additive steps without optimizer state.
- Implications: either use a genuinely additive method with a tested finite step, or stage the complete optimizer transition and accept/reject its actual Δ. Projecting raw gradients then applying ordinary AdamW is not the same method. Applying a frozen SPD preconditioner P requires guards g_jᵀPd ≥ 0; projecting in an appropriate metric is possible, but history/decay must still enter Δ and finite change must still be checked.
- Limitation/falsifiable prediction: these are existence counterexamples, not measured failure frequency. A correct transactional implementation should reproduce an untouched control after rejection, including subsequent steps.

A complete optimizer transaction must specify parameters, first/second/max moments, step counters, gradient scaler, scheduler/global counters, RNG and mutable model buffers. Deterministic closures or restored RNG/buffers are required for comparable trial losses. If a rejected batch is consumed, that is a declared schedule change even when tensors are restored. Current checkpoints save optimizer state for audit but do not implement exact resume. Snapshotting the whole network/state for every trial can be expensive; selecting a block does not protect predictions against simultaneous changes outside that block.

**R5-C05 — DERIVED_RESULT: a noise-tolerant finite-displacement bound (classical smoothness, not novelty).**

- Statement: a constructive guard is possible when proxy-gradient error and true-risk curvature are bounded, but those bounds are latent in this no-GT setting.
- Scope/assumptions: fixed group measure ν_j; true conditional distribution r_i; R_j(θ) = E_νj CE(r_i,p_θ(x_i)); proxy gradient ĝ_j = ∇L_j(θ); R_j has L_j^*–Lipschitz gradient on the segment [θ,θ+Δ]; ||∇R_j(θ) − ĝ_j||_2 ≤ δ_j. All quantities refer to the full changed parameter vector/block with other variables frozen.
- Formal claim: for any finite Δ in that segment,

  R_j(θ + Δ) − R_j(θ) ≤ ĝ_jᵀΔ + δ_j ||Δ||_2 + (L_j^*/2)||Δ||_2².      (R5.1)

  Hence requiring the right side ≤ ε_j implies true-risk regression at most ε_j. For Δ = −ηd and m_j = ĝ_jᵀd − δ_j||d||_2 > 0, every 0 < η ≤ 2m_j/(L_j^*||d||_2²) ensures nonincrease (strict if the inequality is strict). Simultaneous positive margins for finitely many j permit the minimum positive step threshold; if one margin is nonpositive this argument supplies no nonzero certificate. For L_j^* = 0 use the linear bound directly.
- Proof: descent lemma gives true gradient inner product plus quadratic remainder; Cauchy–Schwarz bounds the difference between true and proxy gradient. The existence statement follows by factoring −ηm_j + .5L_j^*η²||d||². This is a standard local sufficient condition, not convergence to a good network.
- Target-noise bridge: for softmax logits z_θ with Jacobian J_i, proxy/true CE gradients differ by E_νj[J_iᵀ(r_i − q_i)]. Therefore δ_j ≤ E_νj[||J_i||_op ||r_i − q_i||_2]. This requires the SAME measure ν_j. It does not address missing anatomy outside that measure, wrong stratum assignment, observation-resolution mismatch, or geometry errors.
- Counterexample: if q is a confident systematic wrong label, ||r−q|| can be large despite low entropy. Repeated source agreement does not bound δ when sources share bias. At a wrong proxy optimum ĝ = 0, δ need not vanish. A finite-set empirical curvature estimate is not a uniform L_j^* bound on an untested path.
- Closest prior: smooth descent lemma; GEM linearized constraints; CAGrad Eq. (3) adds a bounded deviation from average gradient and Theorem 3.2 gives stationary-average-loss convergence for 0 ≤ c < 1 under smoothness. None turns unknown pseudo-label error into a certificate.
- Evidence: `E_known_noise_bound` verifies (R5.1) on 12 manufactured quadratics with known truth, known L = 1 and known noise norm. All upper bounds were nonpositive and bounded the true change. The proof, not 12 cases, establishes the stated conditional result.
- Implications/limitation: δ_j and L_j^* are **latent assumptions**, not operational certificates derived from reliability, entropy, gradient agreement or view consistency. Same-batch proxy acceptance can be implemented without these constants, but then guarantees proxy retention only.
- Falsifiable prediction: in a manufactured noise sweep with known r, observed violation frequency must be zero whenever the declared uniform bounds hold, and can increase when underestimating δ or changing the group measure.

## 4. Hard nonincrease can fossilize wrong evidence

**R5-C06 — COUNTEREXAMPLE: guard feasibility and wrong-optimum lock.**

- Statement: a zero direction is always feasible for homogeneous linear guards; feasible protection does not imply progress or correctness.
- Scope/assumptions: one shared parameter; deterministic smooth fixed proxies.
- Formal examples: gradients +1 and −1 impose d ≥ 0 and d ≤ 0, hence d = 0. Separately, wrong historical proxy L_Q(θ) = θ²/2 and true objective R(θ) = (θ−1)²/2 at θ = 0: every Δ ≠ 0 increases L_Q, while 0 < Δ < 2 improves R. Exact historical nonincrease admits only Δ = 0.
- Proof/evidence: elementary inequalities; actual reference `project_direction` gives zero; `guarded_sgd_step` with atol = 0 rejects every tested positive correction, even though its zero guard gradient imposes no first-order constraint. `C_slack_escape`: Δ = .1 spends .005 proxy loss and lowers R .5 → .405.
- Closest prior: GEM constraints; PCGrad Theorem 1 permits the opposed-gradient stopping case; ConFIG Appendix A.3 Theorems 1-2 permit zero-direction stopping. Classical Pareto feasibility, not novelty.
- Implications: distinguish numerical infeasibility, a small feasible cone, opposed evidence and a wrong protected optimum. Report direction norm, projected correction, acceptance fraction and total accepted displacement; an accepted zero update is a stall, not successful learning.
- Limitation/prediction: allowing slack permits correction but also permits destruction. No-GT observations cannot distinguish these universally. A class/source budget must have a fixed baseline/window and cumulative accounting Σ_t ε_jt ≤ B_j; unlimited small per-step regressions can accumulate arbitrarily. A rolling baseline without debt accounting hides forgetting.

**R5-H01 — PROPOSED_NOT_APPROVED: budgeted, source-aware proxy retention on actual updates.**

- Plain-language hypothesis: retain useful historical evidence while permitting small, declared disagreements driven by a separate image-based objective; do not force every source never to worsen.
- Scope/assumptions: provenance-backed groups, frozen targets/memberships, full actual update evaluation, semantically useful new evidence on at least some support. That last assumption is a task-level gap, not an effect of optimization.
- Formal mechanism: stage (θ′,s′) = AdamW(θ,s,g); evaluate fixed normalized L_j(θ′) and candidate objective H(θ′). Accept if finite, H improves under the declared tolerance, and L_j(θ′) ≤ L_j(θ_ref) + B_j for protected groups. Store cumulative/anchor budget; reject with complete transaction rollback. Optional trust bound ||Δ|| ≤ ρ limits untested excursions. Do not derive B_j from confidence and call it true error tolerance. Soft penalties/CAGrad are alternatives, not equivalent certificates.
- Proof/missing step: accepted fixed proxy values obey the tested budget by construction. No theorem ensures B_j protects anatomy; R5.1 only gives a conditional bridge with latent δ/L. Repeatedly using the same TRAIN panel can overfit it.
- Counterexample: a wrong anchor plus B = 0 causes R5-C06 lock; B too large allows rare-class erasure. Highly correlated sources can collude. Omitted zero-support groups receive no protection and should remain explicitly unprotected.
- Nearest prior chain: claim → GEM Eqs. (6)-(8) and CAGrad Eq. (3) → protect individual objectives while optimizing a new/aggregate one → those methods use declared task losses, while here all named evidence is uncertain and source-dependent → unresolved source validity and finite-budget choice → simplest baseline: normalized fixed weights, smaller encoder LR and deterministic additive backtracking without optimizer momentum.
- Evidence/implications: toy C shows why slack is necessary for some correction; toy B shows why actual updates matter. These establish design constraints, not superiority of this policy.
- Falsifiable prediction: at equal accepted displacement and compute, the policy should reduce independently measured fixed-TRAIN proxy regressions relative to baseline without merely replacing most updates by zero. If a smaller-LR baseline matches the effect, remove the guard. If source permutation or duplication changes decisions unexpectedly, the source accounting fails.

Candidate alternatives deserving comparison are (i) loss balancing without projection, (ii) CAGrad-style compromise allowing individual losses to rise, (iii) reject/abstain on contradictory unreliable source groups, (iv) freeze a representation block, and (v) change the available representation when groups cannot be jointly fit. Clean-validation reweighting is an alternative supervision setting, not a no-task-annotation solution: Ren et al. explicitly use a clean unbiased validation set.

## 5. Pretraining is useful capacity, not a semantic guarantee

**R5-C07 — DERIVED_RESULT: feature preservation and semantic preservation differ.**

- Statement: cosine feature matching alone does not bound semantic logits, and a pretrained encoder can change before a randomly initialized decoder/head becomes useful.
- Scope/assumptions: local simplified head z = Wf + b. Bounds below concern this function, not all residual/normalization layers of the actual decoder.
- Formal claim: for updated W + ΔW, f + Δf, b + Δb,

  ||Δz||_2 ≤ ||W||_op||Δf||_2 + ||ΔW||_op||f||_2 + ||ΔW||_op||Δf||_2 + ||Δb||_2.      (R5.2)

  For any fixed target distribution r, |CE(r,z+Δz) − CE(r,z)| ≤ √2 ||Δz||_2 since ||softmax(z)−r||_2 ≤ √2. These are function stability bounds, not improvement or label correctness.
- Proof: expand (W+ΔW)(f+Δf) and use triangle/submultiplicative norms; integrate the CE logit gradient along the segment. No optimizer-specific assumption.
- Counterexample/evidence: f′ = 10f has zero cosine distance yet for W = [2,−1], f = [1,1] logits change 1 → 10 (`F_cosine_vs_logit`). Small feature drift cannot constrain unbounded head drift; stable predictions can be stably wrong. Projection-head cosine cannot certify every encoder direction or decoder output.
- Closest prior: Kumar et al., ICLR 2022, Theorem 3.3/Eq. (3.7), §3.4/Proposition 3.7: coupled feature/head adaptation can harm OOD transfer in a two-layer linear setting; LP-FT helps under specified feature/subspace assumptions. Their “perfect features” result is not a DINOv3 cardiac guarantee.
- Implications/limitations: measure representation norm, cosine, head norms and prediction drift; a random or warm-started projection head can absorb feature matching. Frozen-encoder success would show decodable proxy structure, not named anatomy. A frozen failure could reflect bad features OR insufficient decoder optimization/capacity.
- Falsifiable prediction: if freezing or warm-up prevents harmful early representation drift at matched update budget, later unfreezing may improve the proxy–capacity tradeoff; otherwise the mechanistic argument is unsupported in this task.

**R5-H02 — PROPOSED_NOT_APPROVED: head/decoder warm-up, then bounded encoder adaptation.**

- Statement: fit the newly initialized decoder/coarse head to allowed TRAIN evidence before large shared-encoder changes, then compare staged unfreezing with full tuning and permanently frozen controls.
- Scope/assumptions: approved identical pretrained initialization, same architecture/bank/geometry and declared schedule; fixed single-cine inference. No target-cohort adaptation or semantic source head is introduced. “Head/decoder warm-up” is not exactly linear probing because the current decoder is nonlinear.
- Formal hypothesis: choose warm-up θ_E = θ_E0 and optimize θ_D; then unfreeze selected encoder stages with a declared displacement budget or lower LR. Predict smaller encoder drift and no worse fixed proxy loss at matched accepted update magnitude, contingent on useful f_θE0 and learnable downstream mapping.
- Proof/missing step: R5.2 bounds functional drift only if its norms are controlled. There is no established theorem that this recipe improves RV or anatomy with noisy Q. LP-FT's linear assumptions do not supply that step.
- Counterexample: if frozen features collapse RV/BG or miss cine structure, warm-up cannot recover it; if wrong Q is fitted first, warm-up may entrench bad semantics. Freeze alone can also suppress useful adaptation. Similar results could come only from reduced effective LR.
- Nearest prior chain: hypothesis → LP-FT §3.4 and layerwise transfer practice → delay representation changes until head fit improves → here nonlinear decoder, dense uncertain targets and source errors replace labeled two-layer regression → unresolved semantic validity and finite-capacity adaptation → simplest baseline: same full tuning with smaller encoder LR, plus permanently frozen encoder.
- Evidence/implications: current model has approved pretrained encoder plus new decoder and projection modules; toy D shows shared drift exists. Actual relative benefit is NOT RUN.
- Falsifiable prediction: if warm-up's benefit disappears under equal encoder displacement/compute, attribute it to step size or budget, not a novel transfer mechanism. If no declared RV evidence remains active, do not credit optimization for identifying RV.

## 6. Primary literature ledger and nearest-method limits

Current search performed 2026-10-18. Publication metadata below was checked using primary proceedings/arXiv, not secondary summaries. Local downloaded PDF/text and SHA256 manifest are in `probes/R5/sources/`. An OpenReview PDF endpoint returned a browser challenge; the LP-FT arXiv version was accessible. An initial mistaken CAGrad arXiv identifier resolved to an unrelated physics paper and was rejected; the verified identifier is 2110.14048. No conclusion depends on the rejected page.

| ID | Verified primary source / status | Precise inspected result and boundary |
|---|---|---|
| R5-L01 | David Lopez-Paz and Marc'Aurelio Ranzato, **Gradient Episodic Memory for Continual Learning**, NIPS 2017; accessed arXiv v6 (2022-09-13). [Paper](https://arxiv.org/abs/1706.08840v6) | Eqs. (6)-(8), PDF pp. 3-4: past-task loss constraints, local gradient inequalities and Euclidean QP. Same projection family as the reference guard. Their episodic labeled memories and local approximation do not validate uncertain cardiac anchors. |
| R5-L02 | Tianhe Yu, Saurabh Kumar, Abhishek Gupta, Sergey Levine, Karol Hausman, Chelsea Finn, **Gradient Surgery for Multi-Task Learning**, NeurIPS 2020. [Proceedings](https://proceedings.neurips.cc/paper_files/paper/2020/hash/3fe78a8acf5fda99de95303940a2420c-Abstract.html) | Algorithm 1; Theorem 1 (convex two-task, smooth sum, step ≤ 1/L) allows opposed-gradient stall. Theorem 2 compares total-loss one-step performance under curvature/angle/magnitude/step conditions. It is not finite per-class nonincrease under arbitrary AdamW. |
| R5-L03 | Bo Liu, Xingchao Liu, Xiaojie Jin, Peter Stone, Qiang Liu, **Conflict-Averse Gradient Descent for Multi-task Learning**, NeurIPS 2021. [Paper](https://papers.nips.cc/paper/2021/file/9d27fdf2477ffbff837d73ef7ae23db9-Paper.pdf) | Eq. (3): maximize minimum local improvement in a ball around average gradient. Theorem 3.2: smooth losses, lower-bounded mean, α ≤ 1/H; 0 ≤ c < 1 bounds sum of squared mean gradients by 2(L0(θ0)−inf L0)/(α(1−c²)). Stationary mean is not true-label or per-task guarantee. |
| R5-L04 | Zhao Chen, Vijay Badrinarayanan, Chen-Yu Lee, Andrew Rabinovich, **GradNorm: Gradient Normalization for Adaptive Loss Balancing in Deep Multitask Networks**, ICML 2018, PMLR 80:794-803. [Paper](https://proceedings.mlr.press/v80/chen18a.html) | §3.2 Eqs. (1)-(2), Algorithm 1 adjust task weights using gradient norms and relative training rates. This is a necessary magnitude-only comparator. Soft-target CE entropy floors can distort relative loss rates; use declared KL diagnostics and avoid singular normalization near zero. No source-validity theorem. |
| R5-L05 | Shiori Sagawa, Pang Wei Koh, Tatsunori B. Hashimoto, Percy Liang, **Distributionally Robust Neural Networks for Group Shifts: On the Importance of Regularization for Worst-Case Generalization**, ICLR 2020; arXiv v2, 2020-04-02. [Paper](https://arxiv.org/abs/1911.08731v2) | Eq. (7), Algorithm 1, Proposition 2/Eq. (9): group minimax updates and convex bounded-loss O(1/√T) guarantee. Their empirical warning is that fitting worst-group training loss does not ensure worst-group generalization. Declared pseudo-groups here do not have trustworthy task labels. |
| R5-L06 | Ananya Kumar, Aditi Raghunathan, Robbie Jones, Tengyu Ma, Percy Liang, **Fine-Tuning can Distort Pretrained Features and Underperform Out-of-Distribution**, ICLR 2022 Oral; arXiv v1, 2022-02-21. [Paper](https://arxiv.org/abs/2202.10054v1) | Theorem 3.3/Eq. (3.7) in overparameterized linear regression; §3.4 Proposition 3.7/Eqs. (3.10)-(3.11) for perfect features. The paper explicitly leaves general imperfect-feature LP-FT upper bounds difficult. Use as a mechanism and simple baseline, not a theorem for nonlinear noisy dense prediction. |
| R5-L07 | Qiang Liu, Mengyu Chu, Nils Thuerey, **ConFIG: Towards Conflict-free Training of Physics Informed Neural Networks**, ICLR 2025. [Proceedings PDF](https://proceedings.iclr.cc/paper_files/paper/2025/file/94e85561a342de88b559b72c9b29f638-Paper-Conference.pdf) | §3.1 and Appendix A.3 Theorems 1-2; Appendix B.1 discusses solvability/linearly independent gradients. Positive-dot-product directions require feasible constraints; opposed or missing gradients here cannot be dismissed by parameter count. The convergence statements admit zero ConFIG direction. PDE residuals are not noisy named-class anchors. |
| R5-L08 | Bo Liu, Yihao Feng, Peter Stone, Qiang Liu, **FAMO: Fast Adaptive Multitask Optimization**, NeurIPS 2023. [Proceedings](https://proceedings.neurips.cc/paper_files/paper/2023/hash/b2fe1ee8d936ac08dd26f2ff58986c8f-Abstract.html) | §4/Algorithm 1 is an efficient task-weighting alternative to storing each task gradient. Relevant if many source/class groups make full per-group gradients expensive. The present review inspected the algorithm context, not a transferable finite-step guarantee; no such guarantee is claimed here. |
| R5-L09 | Mengye Ren, Wenyuan Zeng, Bin Yang, Raquel Urtasun, **Learning to Reweight Examples for Robust Deep Learning**, ICML 2018, PMLR 80:4334-4343. [Proceedings](https://proceedings.mlr.press/v80/ren18a.html) | Method learns example weights against a clean unbiased validation set. It addresses both imbalance and label corruption, but directly importing it would change this no-task-annotation setting. |

No novelty claim survives merely renaming projection “rare-class guard,” adding group normalization, or using smoothness bounds. A potential research contribution would need empirical and theoretical evidence about source uncertainty, valid observation support, and actual optimizer transactions that the listed methods do not already provide.

## 7. Matched control matrix and stopping criteria

These experiments are **PROPOSED_NOT_APPROVED**, not queued jobs. Before any patient execution, human selection and a versioned policy are required.

| Mechanism to isolate | Minimal matched contrast | Keep fixed / decisive observation |
|---|---|---|
| Pure mass imbalance | Global weighted mean vs fixed equal stratum means; independent-logit and collided-logit manufactured controls | Same source content, support, optimizer and displacement; distinguish slow improvement from opposite sign. |
| Wrong evidence | Known-noise manufactured sweep, including confident systematic RV/BG swap and duplicate source | Match masses/entropy, preserve latent truth for probe scoring only; balancing must not be credited when only wrong proxy improves. |
| Direction versus magnitude | Baseline, rescaled baseline with ||Δ|| matched, and direction-modified proposal | Match actual Δ norm, clipping, accepted step count and forward/backward compute; smaller steps alone may explain gains. |
| AdamW transformation | Additive SGD, raw-gradient projection + AdamW (negative control), actual-displacement check + full rollback | Identical initial state and batches; inspect g_jᵀΔ and actual ΔL_j. Parameter-only rollback is an expected-fail control. |
| Hard protection versus correction | B = 0 versus finite anchored B; proxy-optimum wrong-evidence toy | Same candidate evidence/objective; report truth in toys, proxy budget in actual data. Never assume relaxed proxy loss is true improvement. |
| Shared-encoder conflict | Feature/semantic gradients by block, feature-only steps, fixed encoder vs full tuning | Same pretraining, bank and capacity; a head-only guard cannot protect simultaneous encoder changes. |
| Pretraining vs schedule/capacity | Approved pretrained full tuning, permanently frozen encoder, decoder warm-up then unfreeze; smaller encoder LR control | Same initialization and architecture. Random-initialization experiments would be separately declared research controls, never canonical fallback; none executed here. Count warm-up compute. |
| Capacity deficiency | Same schedule with available distinct vs collided representations in toys; optional matched decoder capacities later | Separate inability to fit proxy from source incorrectness. Added parameters are not evidence for better semantics. |
| Source dependence | Original groups vs duplicated/correlated and shuffled-source controls | Same unique input evidence and total mass. Duplicates should not inflate independent evidence or alter budgets without declared policy. |

No-GT kill criteria: no valid named RV source support; invented source decomposition from Q; absent group reported as zero risk; mask/geometry change across acceptance comparisons; source duplication creates protection votes; guard passes raw gradient but fails actual displacement; rollback cannot reproduce control; zero accepted motion masquerades as learning; benefit disappears under matched LR/displacement/compute; semantic claims depend only on proxy loss or stable features. These can kill an engineering/mechanistic proposal without labels. Passing them cannot certify anatomical quality. If the task requires proving better anatomy, authorized independent reference evaluation remains necessary and outside Phase I.

## 8. Probe receipt

Command executed: `rtk proxy python reports/research/20261018T030143Z/probes/R5/run_probes.py`.

Exit status 0; Python 3.11.16, PyTorch 2.14.0; CPU float64; one thread; seed 0 for the random quadratic check. Script SHA256 `845587733d2a8ffa9ef44d5916f6391d5be48832e65da4d8098b8cacc4894356`. Results `probes/R5/RESULTS.json`; all assertions passed. Imported only the pinned reference guard for manufactured tests, not the trainer, bank or pretrained constructor. No full repository test claim is made by R5; coordinator owns those checks. These probes establish counterexamples and conditional arithmetic only.

Initial draft ends here. Subsequent text must be appended under dated review/revision headings.
