# Architecture dossier: adaptive geometric programs and categorical cellular operators

**Status: PROPOSED / DERIVED / NOT_RUN.** Two different student computation graphs remain live. Neither is validated, novel by assertion, or an established route to 96 Dice. Preserved independent `initial.md` SHA256 is `cc08fba0d93b8c8359268e8c546e96e444afd12a355448435637a4c3d4c7d612`; it predates the source packet and peer reading.

## Contract and relation to the four goals

Source HEAD `9958d421ccae1da3b8527f74da5438dbb9f3f521` and its active-pipeline/method documents were inspected read-only. These proposals are separate prospective protocols. Every teacher, planner, structural-head, audit and diagnostic label is charged to dense TRAIN patients L; U supplies images only. No U/VAL/TEST masks define targets or training gates. Both encoders receive the same original 2D image and validated geometry. Extra views, phases, slices, pretrained sources or full-TRAIN labels require separate declared information budgets. Inference uses only the student and image metadata.

Both routes redesign the compact decoder and define executable representations built jointly from teacher and student features during training. Their transfer mechanisms are hypotheses beyond feature/gradient matching; superiority is unmeasured. Neither route itself establishes a stronger teacher. First freeze a qualified native cardiac teacher, reporting ACDC and M&Ms separately, patient-level foreground RV/MYO/LV Dice, classwise results and physical boundary errors. A background-inclusive score cannot silently stand for foreground 96. Structural heads might strengthen the teacher, but must be compared with equal-compute teacher training without them; they cannot inherit accuracy from the original logits.

## A. Adaptive finite-element construction program

The student comprises a shallow constant-resolution image stem, a sparse triangular cell complex, a local action policy and a native rasterizer. A bounded program splits/moves cells and assigns vertex class coefficients; it does not correct a dense coarse segmentation. With barycentric basis functions b_v on the containing cell,

    f_c(p)=sum_v b_v(p)a_vc;  q(p)=softmax(f(p)).

There is no downsampling/upsampling feature pyramid. The output always comes from this mesh. Initial coverage spans the complete image, absent classes are legal, and no ventricular topology template is imposed. Enforce valid nonoverlapping triangles and minimum positive physical area; illegal moves are rejected. Refinement can reach native pixels, but fixed K cells cannot guarantee arbitrary detail.

**Transferred object and graph.** For a current state s, training constructs

    r_TS(C)=[pool_C E_T(x), pool_C E_Sbar(x), physical geometry, coefficients],
    pi*(s)=argmin_(legal a) R(Complete_H(s,a),t)+lambda cells(Complete_H(s,a)).

Here E_Sbar is a frozen student snapshot during target generation, H is bounded lookahead, and t is budgeted GT on L or teacher predictions on U. A frozen structural head from r_TS supplies class/interface costs; teacher distributions remain correlated estimates, not truth. Supervision transfers preferences between actions or terminal-mask equivalence classes, avoiding an arbitrary “correct” split order. The student executes its own policy during training and the planner evaluates visited states; teacher-forced trajectories alone would hide compounding error.

Training information graph: same x -> teacher features plus frozen student features -> joint cell representation -> bounded planner -> action/value targets; x -> live student stem -> actual geometric program -> rasterization -> native supervised loss on L. The inference graph is x -> student stem -> K-or-fewer actions -> mesh coefficients -> native q. The joint teacher/student object drives the training planner and decoder's geometric targets; it is not supplied at deployment or assumed exactly reconstructible there.

**Distinct prediction.** At equal cell and search budgets, teacher-feature planning should improve allocation around thin MYO and RV junctions beyond an identical logits-only planner. At equal native error, adaptive partitions might need fewer cells than uniform ones. Failure of the first prediction rejects the joint-feature-transfer claim even if adaptive segmentation works.

**Cost and physical detail.** Coordinates, edge lengths and loss quadrature use millimeters and native pixel support. Native samples enter the stem; a slice model does not acquire through-plane anatomy from z-spacing. Dense processing/output costs at least Ω(N). Naive all-cell action rescoring costs O(K²d), naive raster point-location O(NK); spatial indexing/local updates may reduce these, but need measurement. Training lookahead multiplies expense. Few parameters do not establish low latency.

**Cheapest decisive test.** Enumerate short programs on tiny TRAIN patches, including one-pixel walls, at fixed K; compute best attainable classwise Dice. Whole-image heuristic fitting provides an attainable result, not a certified capacity ceiling. Then hold search expansions fixed and compare logits-only, teacher-feature and same-capacity student-feature planners against the charged-GT oracle. Decompose representational error, teacher-cost error, search regret and student-policy regret. A missed tiny RV island or intricate wall can defeat compact K; no residual decoder may conceal the failure.

## B. Reopened: native categorical cellular operator

The initial unrestricted recurrent proposal is weakened by existing NCA segmentation. Reopen a narrower hypothesis: a native categorical solver with measurable intervention responses and a stability condition, rather than unexplained hidden-state dynamics. This is the contrasting route, not a claim that recurrence is new.

A shallow stem predicts unary terms u and sparse C-by-C neighbor interactions W on the native physical graph. A fixed small collection of class-mass/spatial moments supplies an explicit, ablatable global message. With A combining local interactions and that low-rank broadcast,

    F(q)=softmax(-u-Aq); q^(t+1)=(1-eta)q^t+eta F(q^t), 0<eta<=1.

Each node carries a categorical distribution; output is q^T itself. There is no U-Net decoder or dense correction branch. Starting beliefs come from image-derived unary terms, never GT seeds. Symmetric interactions permit an energy interpretation only with consistent pair counting and global terms; an arbitrary learned update must not be called energy descent.

A sufficient contraction condition in the block-vector infinity norm is `||A||_infty/2 < 1`, using the softmax Jacobian bound. Under this restriction the damped map is contractive with factor at most `1-eta+eta||A||_infty/2`. Damping alone does not guarantee convergence when the bound fails. This restrictive condition could remove the strong long-range coupling needed for useful segmentation. It is a falsifiable tradeoff, not a free guarantee.

**Transferred object and graph.** A structural module trained on L receives `[E_T(x), E_Sbar(x), displacement, image contrast]` and predicts a joint operator `(u_TS,A_TS)`. Freeze that module and the student snapshot per target-generation stage. Query it with several categorical initial states and localized unary interventions δ independent of U masks:

    R_TS(q0,delta)=F_TS,delta^T(q0).

Transfer these executable finite responses to the student operator `(u_S,A_S)`, together with native task supervision on L. This differs from coordinate feature equality, but remains distillation with losses; it cannot be advertised as loss-free knowledge extraction. Unseen intervention families distinguish operator learning from memorizing one teacher output.

Training graph: same x -> teacher/student-snapshot features -> joint operator -> fixed intervention trajectories -> student transition supervision; x -> live student stem -> native recurrent solver -> labeled loss on L. Inference: x -> student unary/interactions -> fixed T updates plus declared moments -> q^T. No teacher or per-patient fitting is present. The new joint structural module is not automatically a faithful description of the original teacher: compare its own native errors against teacher logits and an equally sized student-feature module before transfer.

**Cost, prediction, killer.** Physical edge lengths/face areas define conductance scaling; no silent isotropic or interslice adjacency. Work is O(T|E|C²) plus stem/moment costs; inference state O(NC+|E|C²), naive training history O(TNC). The distinguishing prediction is improved recovery across held-out categorical corruptions beyond output KD with the same final teacher probabilities. Test tiny grids first for contraction, finite propagation, cycles, anisotropic resampling and thin-wall preservation, then charged TRAIN-fold action quality. Two cavities with identical T-hop evidence and identical retained global moments are indistinguishable to this graph; coordinates can break symmetry but can encode brittle positional shortcuts. Attraction can erase weak thin walls, while a stable solver can converge confidently to the wrong answer.

## Evidence reconciliation and decisions

Primary texts supplied centrally were inspected; no independent retrieval occurred. [OctreeNCA](https://www.alphaxiv.org/abs/2508.06993), §§3.1–3.3, already has recurrent local image/state updates, coarse-to-fine context and specialized CUDA inference. It blocks any claim that non-U-Net recurrent segmentation or low parameter count is new; its reported memory results do not transfer here. Route B was therefore downgraded, then reopened around identifiable finite responses and an explicit stability/expressivity test.

[CineMesh4D](https://www.alphaxiv.org/abs/2605.13994), §§2–3, already maps image encoders to mesh decoding with render supervision. Its fixed-connectivity multiview/temporal mesh differs from a same-frame adaptive partition program; this distinction is a testable design, not a novelty certificate. [D-RASP](https://www.alphaxiv.org/abs/2602.08857), §3, provides executable reparameterization under exact linear layer-normalization assumptions, then empirical pruning; the unreduced program can be exponential. Route A is bounded planning/imitation, not exact teacher decompilation. [RKD](https://www.alphaxiv.org/abs/1904.05068), §3, already transfers invariant relations and explicitly retains task/individual semantics where needed; invariance alone is not our contribution.

**Peer response, accepted and sharpened.** The teacher investigator correctly objects that endpoint accuracy identifies neither action competence nor an energy. Hold a strictly positive teacher endpoint q* fixed and choose two equal-sparsity interactions A_1,A_2; set `u_k=-log(q*)-A_k q*`. Both satisfy `F_k(q*)=q*`, yet their local response Jacobians `-J_softmax(log q*) A_k` and finite-T recovery basins can differ. This is a mathematical counterexample to endpoint sufficiency, not patient evidence. Under the contraction restriction there is a unique fixed point but finite-T recovery rates/directions can still differ; no claim of multiple basins is needed in that restricted case.

Observable transfer targets are finite-horizon recovery error and action regret on patient-disjoint TRAIN folds, including thin-MYO gaps, ectopic islands and identity swaps. Fixed final teacher probabilities, interaction capacity, corruption distribution and compute isolate these from endpoint KD. Prediction: the lower-regret operator/planner produces lower student recovery error under the same budget. Falsifier: the ranking fails to predict student outcomes, or neither improves on equal-compute task-trained teacher-free rules. Reject an accurate teacher as a planner when its proposed actions increase charged-fold risk relative to that control; retain harmed cases and lock tolerance before scoring. Action comparisons unsupported by the fold remain unknown, not optimal.

I accept the peer's partial-order idea as an optional diagnostic, but reject “certified” as an unseen-patient claim. A control receiving identical comparison labels without teacher features is mandatory. My outgoing geometry review additionally requires causal teacher-feature ablation and separate representation, teacher-cost, search and imitation errors; its new face-flux alternative remains distinct and unvalidated. Independent convergence is not originality evidence.

## Evidence receipt

All proposed tests, training, patient efficacy, stability verification and runtime measurements are **NOT_RUN**. No code, datasets, weights, jobs or experiment nodes were changed. Repository checks and pytest were not run for these prose-only research artifacts; no software-validation claim is made. Source document SHA256: ACTIVE_PIPELINE `a451851b96ef382fd8de5f0542a6f44e0db61954c0adb66facec9a0e14ee904b`; METHOD `21ea10c4071c65816299b3b3f96d9167bfea6e5aeae256372908b7fa41f5be2e`. Bounded supplied literature leaves substantial adaptive-mesh, graphical-model and program-learning prior-art uncertainty.
