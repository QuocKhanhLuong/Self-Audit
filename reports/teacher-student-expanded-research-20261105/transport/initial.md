# Independent initial mechanisms — transport investigator

Written 2026-11-05 before receiving a literature packet or reading any council report. This file is preserved as the pre-evidence record. RESEARCH HYPOTHESES, NOT IMPLEMENTED OR EVALUATED. No literature search performed. Source boundaries inspected at `/Users/alvinluong/Self-Audit`, verified HEAD `9958d421ccae1da3b8527f74da5438dbb9f3f521`: ACTIVE_PIPELINE.json and docs/METHOD.md. Those govern evidence and information access, not the architecture of these prospective protocols.

Shared notation: x is one original image, t_i=T(x)_i and s_i=S(x)_i at spatial sample i. L is the complete charged dense TRAIN patient subset; U supplies images only. A strong frozen teacher must first be trained with the same declared labels; external data/context require a separate protocol. Names RV/MYO/LV/BG come from L, never from geometric intuition alone. Primary deployment receives x only, with no teacher, extra views, or masks. A 96 Dice target is an aspiration whose class aggregation and domains must be fixed, not a result.

## A. Compile a teacher transport process into a student response operator

**Object, rather than loss.** Replace a feature tensor with a finite-state diffusion process whose states are image-adaptive parts. Teacher and student independently produce local mass measures and affinities on the SAME spatial support. A coupling C maps teacher sites to K student states; use spatial support as the first correspondence, not an unconstrained semantic optimal-transport guess. Let P_T be a row-stochastic teacher affinity operator, C be an N-by-K nonnegative assignment with C1=1, and D=diag(C^T mu). The reduced transition is

    P_K = D^{-1} C^T diag(mu) P_T C.

Instead of matching P_K entries, train the student encoder to emit its own stochastic P_hat(x), assignments C_hat(x), and semantic source b_hat(x); its decoder computes a queryable response

    z = (I - alpha P_hat)^{-1} b_hat,
    p(y_i|x) = softmax(sum_k C_hat_ik z_k),  0 <= alpha < 1.

Train-time fused representation uses same-image student C and teacher P_T; deployed representation is its student-compiled operator. Teacher can provide responses to controlled source probes v, including local boundary impulses, through C^T diag(mu)(I-alpha P_T)^{-1}v. The transfer target is an input-output operator family with conservation and positivity constraints, not a basis-aligned embedding. Label-derived source and semantic head remain trainable on L. The teacher's own probability map is a comparator, not the complete target.

**Gauge and semantics.** Orthogonal rotations of t preserving affinity leave this object unchanged; invertible general gauges need a covariance metric or are NOT invariant. Permuting K latent states conjugates P and permutes sources and C, leaving output unchanged. Semantic output names cannot be permuted: anchors on L fix that gauge. Transport without spatial anchors admits swapped ventricles, so an unbalanced coupling cannot be trusted to name structures. Positional costs can also bake in pathological anatomy errors.

**Compact architecture.** One fixed-resolution patch stem; local linear attention or small mixing blocks; K=32–64 adaptive states; sparse state graph; a bounded polynomial approximation to the resolvent; a coordinate-conditioned rasterizer using original-image local intensity descriptors. No encoder-decoder pyramid or skip ladder. Rasterizer supplies boundary resolution but cannot silently carry a full U-Net's capacity.

**Assumptions/prediction.** Teacher affinities encode patient-specific connectivity that student logits alone do not capture. Prediction: at equal parameters, operator compilation improves thin-MYO separation under local contrast corruption, and its output remains unchanged under arbitrary orthogonal teacher feature rotations. Crucially, a model trained with only class-probability probes should be weaker than one taught nonsemantic localized response probes if structural transfer is real.

**Killer counterexample.** Teacher affinity strongly connects LV blood pool to RV through similar intensities. Diffusion washes out the separating myocardium, and exact operator transfer is actively harmful. Further, if all transferred queries lie in the four-class output span, this reduces to a dressed-up probability distillation scheme. Reject if localized operator queries add no value beyond matched KD and equal-compute graph decoding.

## B. A distribution over compositional partitions, decoded by marginalization

**Object.** A teacher representation determines a distribution over partitions pi of an image graph, not a single embedding or pairwise score matrix. A student representation supplies local evidence u_i(c) and a small set of split/merge choices. Let a random hierarchical partition have probability

    q_T(pi|x) proportional to exp[-sum_(i,j) cut_pi(i,j) w_T(i,j)] r(pi),
    p(y|x) = sum_pi q_S(pi|x) p_theta(y|pi, u_S(x)).

Training forms q_TS(pi|x) from teacher edge evidence and student fine-scale evidence on the same x; learns amortized q_S using supervised segmentation risk and posterior-predictive partition queries. Partition queries include probabilities that triplets share a region, connectedness after deletion of a bridge, and distributions of region sizes. They must probe beyond pairwise marginals, otherwise a partition-law claim collapses to relation matching. Keep a tractable hierarchical family with exact sum-product marginalization over binary split decisions; do not enumerate Bell-many partitions.

**Compact decoder.** The network predicts a fixed-depth spatial partition tree and class factors at leaves. A sum-product pass fuses alternative splits; coordinates and local pixel descriptors rasterize expected semantic occupancy. This is a probabilistic circuit with learned image-dependent factors, not an upsampling CNN. Fixed bounded depth controls cost, but spatial tessellation must preserve thin structures.

**Gauge and semantics.** The latent partition is label-free and invariant to arbitrary ordering of parts. Dense L attaches semantic labels to region factors. Independent naming per region is insufficient: allow MYO to be disconnected in a slice, allow absent RV/LV, and do not force one cluster per class. Teacher/student use a common spatial graph, so there is no unrestricted OT permutation problem. A region partition never establishes RV versus LV identity without semantic evidence.

**Assumptions/prediction.** Several plausible region decompositions survive low contrast, and averaging their semantic occupancy is better calibrated than forcing one feature match. Predicted distinguishing case: two partition distributions with identical pairwise co-membership but different triplet/higher-order constraints produce different outputs when a semantic query depends on joint region identity. Ordinary pairwise matching cannot distinguish them.

**Killer counterexample.** The tractable tree family cannot express two interleaved anatomical regions or a very thin curved annulus without excessive depth. Its calibrated average produces a blurred MYO boundary and worse Dice than a deterministic classifier. Reject if higher-order probes do not affect semantic inference, or if the model is only a soft tree used to generate a conventional feature tensor.

## C. A posterior over semantic sufficient statistics with an explicit missing-information test

**Object.** Teacher and student encoders jointly infer a compact random variable H of semantic measurement functions: class-conditioned signed boundary moments, occupancy fractions in learned spatial cells, and adjacency compatibility. Let h_m(y)=sum_i a_m(i) 1[y_i=c_m], with nonlinear connectivity summaries separated from these linear measurements. Train q_TS(H|T(x),S(x)) and a student-only q_S(H|S(x)); decode via an exponential-family conditional field

    p(y|u_S,H) proportional to exp[sum_i u_S(i,y_i) + sum_m lambda_m(H) h_m(y)].

The decoder enforces a posterior distribution of global constraints while local student evidence resolves boundaries. This changes the interface to probabilistic measurements and constrained inference. It is NOT merely making moments close: the decoded segmentation must actually depend on these constraints, and sampling H propagates uncertainty. Teacher/student fusion creates H during training; deployed student predicts its distribution from x.

**Assumptions/prediction.** A small set H carries task-relevant global information absent from a compact local encoder. On TRAIN only, estimate residual predictability of Y from teacher features conditional on (u_S,H); if substantial held-out-within-TRAIN predictability remains, H is not sufficient. Sufficient statistics are a hypothesis, never established by calling them sufficient. Compare equal-sized generic latent vectors and identical inference compute.

**Killer counterexample.** Two masks share every retained moment and adjacency statistic yet differ along a clinically important thin boundary. The decoder cannot recover the correct one from H, particularly if local evidence is also ambiguous. If augmenting H requires near-pixelwise measurements, the compactness premise fails. Moment constraints can impose impossible anatomy and must permit absent classes, pathology and uncertainty.

## Independent priority and decisive questions

A is the leading engineering hypothesis because its mathematical object admits exact gauge, permutation and query tests. B is the higher-risk scientific candidate: genuinely higher-order partition uncertainty could change the computation rather than add a metric, but tractability and narrow MYO support are serious blockers. C is retained as a falsifiable compression viewpoint, not a novelty claim. The evidence packet should be used to identify exact precedents for operator distillation, stochastic partition/circuit segmentation, and task-measurement bottlenecks; absence from a packet cannot establish novelty.
