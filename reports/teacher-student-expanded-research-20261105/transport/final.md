# Transport and compositional probability dossier

**Status, 2026-11-05:** source-informed hypotheses, no trained model or cardiac result. `initial.md` is preserved unchanged and was written before the supplied packet. Source HEAD independently verified as `9958d421ccae1da3b8527f74da5438dbb9f3f521`; ACTIVE_PIPELINE.json and docs/METHOD.md were read. No external search, repository edit, weights/data download, GPU work, or job mutation occurred. This prospective protocol does not modify native-v2.

## Recommendation and evidence-driven changes

Keep two competing mechanisms: **partition-distribution decoding**, the higher-risk main scientific hypothesis, and **transport-response compilation**, the more tractable alternative. Reopen the initial semantic-measurement idea only as a falsifiable compression experiment. The supplied RKD paper already covers general n-tuple relations, including triplet angles; therefore gauge invariance, higher order, and “relations instead of features” are insufficient distinctions. The codebook paper similarly removes discrete anchors or student-compatible concepts as standalone contributions. The proposed distinction must survive a test of actual inference through the transferred object.

Neither route establishes goal 1, a strong cardiac teacher. Both address goal 2 through a joint same-image representation, goal 3 through executable structured transfer, and goal 4 through a compact student without a U-Net pyramid. All three are hypotheses, not achieved benefits. A 96 Dice aspiration requires an explicitly declared metric: foreground RV/MYO/LV versus BG-inclusive Macro4, patient/class aggregation, domains, ED/ES handling and absent-class rules. No provided source inspected here establishes 96 foreground Dice on both ACDC and M&Ms.

## A. Decode a distribution of partitions

Let a small fixed-resolution image stem produce local class evidence u and merge/split evidence s. Construct B candidate spatial trees over the same image sites, using image-only deterministic adjacency/tessellation and bounded merge proposals. Each tree's leaves are disjoint; internal nodes choose either one region or a decomposition into children. Mixture gates choose among tree topologies. A probabilistic circuit sums mutually exclusive alternatives and multiplies disjoint child factors, admitting exact normalization and occupancy marginals for this restricted family.

The mathematical object is a joint law over partition pi and region labels a:

    q_TS(pi,a | x) = Z_TS(x)^(-1) w_T,S(pi | T(x),S(x))
                    product_(r in pi) exp[sum_(i in r) u_S(i,a_r) + v_T(r,a_r)].

Teacher features supply split/merge evidence and teacher semantic probabilities supply v_T; student features provide local evidence and candidate geometry. Both encoders receive exactly x. Named class factors are calibrated/trained on charged L. Treat this law as a designed teaching distribution, not a verified Bayesian posterior. Bound region factors to avoid area-dependent numerical domination; any area normalization is a declared experimental factor.

The deployed student predicts circuit factors q_S(pi,a|S(x)). Segmentation is computed by its sum-product marginals:

    p_S(y_i=c | x) = sum_(pi,a) q_S(pi,a|x) 1[a_region(i,pi)=c].

There is no unrestricted neural segmentation head after the circuit that can bypass this computation. Fine spatial leaves preserve boundaries; a native-resolution, small local stem supplies evidence without a downsampling/upsampling ladder. A cheaper superpixel version is a separate capacity factor because a leaf straddling thin MYO creates irreversible error. Complexity is O(B M C) circuit evaluation for M tree nodes and C=4 classes, plus stem and tree construction; count native leaves, topology-generation latency, intermediate memory and all B alternatives. Parameter count alone is insufficient.

**Train information graph:** budgeted (x_L,y_L) → frozen teacher training and student/circuit semantic training; x_L or x_U → T(x),S(x) → fixed candidate forest → q_TS → circuit queries and semantic predictions → student factor learning. Stop-gradient the whole fused target during each update. Supervision includes task risk on L, teacher probabilities, and partition-query response distillation; compare their contributions separately. No U masks, VAL/TEST masks, external anatomy atlases, temporal frames or LAX views enter. **Inference:** x → local stem and bounded image-only forest → q_S → exact occupancy marginals → mask. Teacher, fused targets and labels are absent. Changes to the label budget or image dimensionality create a separate protocol.

**Semantics and ambiguity:** latent region indices are exchangeable; semantic channels are not. Dense L fixes RV/MYO/LV/BG names. Multiple disconnected regions may share one name, and absent classes remain possible. A partition alone cannot name chambers. Same spatial support removes the need for unconstrained OT alignment; it does not eliminate ambiguity when intensities cannot distinguish structures. Because T(x) is deterministic, I(Y;T(x)|x)=0: transfer supplies learned inductive/computational structure, not new subject information. Irrecoverable image ambiguity must remain uncertainty.

**Decisive distinction:** law A puts 1/2 on {123} and 1/2 on {1|2|3}; law B puts 1/4 on each of {123},{12|3},{13|2},{23|1}. Every pair has co-membership 1/2 in both laws, but triple co-membership is 1/2 versus 1/4. Exact fraction arithmetic verified this. The decoder must produce distinct answers to the triple-region semantic query; a pairwise-marginal-only interface cannot. This does NOT defeat all RKD, whose general formulation permits higher-order potentials, nor prove non-identifiability for a fully specified pairwise exponential family. The new hypothesis is that retaining a valid joint partition law and marginalizing it gives task value beyond matched query losses on an ordinary decoder.

Circuit paths can duplicate a partition: define its probability as the sum over all derivations and report multiplicity-induced prior bias. A forced-full-split circuit is an independent-unary control; if it ties the full model, the composition hypothesis fails. A single fixed tree cannot represent all three two-site merges. Therefore the initial tree proposal is revised to a bounded mixture, with representational adequacy tested before training. The killer test uses equal pairwise marginals, then compares full circuit inference, pairwise-only summaries and an equal-compute deterministic decoder. On a later authorized small TRAIN pilot, require better held-out-within-TRAIN MYO boundary quality/calibration beyond equal-budget logit KD, RKD and feature matching. All those internal labels count in L. Reject if the difference is only generic regularization, or if B/M grow until compactness disappears. Curved thin MYO, unusual topology and blurred averages are concrete failures.

## B. Compile transport responses rather than feature coordinates

Teacher features define a stochastic local operator P_T. Student assignments C map N image sites to K states, with C1=1, positive image-site masses mu, and D=diag(C^T mu). The fused operator is

    P_K = D^(-1) C^T diag(mu) P_T C,
    R_alpha(P) = (1-alpha)(I-alpha P)^(-1),  0 <= alpha < 1.

Student-only deployment predicts C_hat, P_hat and named source scores b_hat from x, then computes softmax[C_hat R_alpha(P_hat)b_hat]. A fixed-resolution stem, K-state sparse mixer, bounded polynomial response and local coordinate rasterizer replace the U-Net. Train student responses on localized, nonsemantic source probes as well as semantic sources; otherwise four class probes can collapse the proposal into logit KD. Freeze each teacher/student-derived target within the update. Probe coverage and source normalization are prespecified.

Orthogonal teacher rotations preserve Euclidean affinity; arbitrary invertible basis changes generally do not. State permutations conjugate P and permute C/b, preserving output. Unconstrained OT on equal-mass LV/RV regions can exchange names at no cost; image-coordinate correspondence and L-derived semantic anchors are mandatory, and neither proves correctness under pathology.

Compression has a checkable obstruction. With E=P_T C-C P_K,

    R_alpha(P_T)C-C R_alpha(P_K)
      = alpha (I-alpha P_T)^(-1) E R_alpha(P_K).

Its infinity norm is bounded by alpha/(1-alpha)||E||_infinity. E=0 is exact lumpability; without it, perfect coarse response preservation is generally impossible. A three-state manufactured NumPy calculation gave ||E||=0.3 and response discrepancy 0.15617853 at alpha=0.6; the identity residual was 1.94e-16. This is algebra verification, not segmentation evidence. Root independently reports a projected-versus-reduced resolvent Frobenius gap 0.579636 in `PROBE_RESULTS.json` (FIXTURE_ONLY); its unnormalized operator and norm differ from this calculation, so the numbers are not comparable. Response consistency is a mandatory pre-transfer gate. Truncating the normalized power series at degree m has tail bound alpha^(m+1) for unit-bounded sources; long-range diffusion therefore carries real iteration cost.

The cheapest test measures E and probe error as K grows, before fitting a student. Include gauge/permutation invariance and teacher graphs with spurious blood-pool bridges. Reject if retaining thin MYO requires K near N, or if localized probes add nothing beyond class-probability probes. Teacher affinities may encode texture rather than useful anatomy; diffusion can amplify the wrong connectivity. The packet supplies no specific operator-compilation precedent, so novelty remains UNKNOWN, not established.

## Reopened C and closest provided mechanisms

The semantic-measurement bottleneck remains worthwhile as an adversarial experiment: transfer a distribution over class occupancy moments H, then decode an exponential-family field conditioned on H and local student evidence. Search for two masks with identical retained H but different MYO boundaries. If local evidence also agrees, no decoder can recover the difference. Reopen only if a small H improves task risk beyond a generic latent of equal size; never call H sufficient without conditional residual tests.

Primary methods inspected: **RKD**, `papers/1904.05068.txt`, section 3.2, transfers n-tuple relational potentials and requires individual/task supervision where output semantics matter. **Codebook-Guided Cross-Modal KD**, `papers/2609.37243.txt`, sections 3.3–3.5, quantizes teacher features, selects task/student-compatible codes and reconstructs feature/head predictions; our circuit must instead retain joint alternatives used directly in decoding. **OctreeNCA**, `papers/2508.06993.txt`, sections 3.1–3.3, already supplies non-U-Net recurrent segmentation with hierarchical state propagation; neither compact parameters nor a tree makes our architecture new or fast. **D-RASP decompilation**, `papers/2602.08857.txt`, introduction and executable-operator definitions, shows computation extraction in restricted algorithmic settings while acknowledging potentially exponential expansion; it supplies no guarantee of compact cardiac compilation. These are bounded provided comparators, not an exhaustive prior-art search.

**Evidence ledger:** source boundary and four primary-method inspections VERIFIED; two tiny algebraic diagnostics VERIFIED, synthetic only; architecture benefit, teacher strength, real Dice, calibration, latency and novelty NOT_EVALUATED. Repository checks and pytest NOT_RUN because this task only creates research documents outside the source checkout. No source/weight bytes changed; no weight artifact was used. The next decision is a cheap representation falsification, not a patient/GPU launch.

Preservation/source hashes (SHA256): initial.md `c204ef7ea7e919942e3599b65b3d126054be73b09cfd7b1d5ea53300c0903b7b`; ACTIVE_PIPELINE.json `a451851b96ef382fd8de5f0542a6f44e0db61954c0adb66facec9a0e14ee904b`; docs/METHOD.md `21ea10c4071c65816299b3b3f96d9167bfea6e5aeae256372908b7fa41f5be2e`.

## Peer challenge and response

Geometry correctly objected that a fixed independent-split tree can be identified by pairwise co-memberships; the unrestricted counterexample alone was inadequate. Accepted: explicitly require three candidate topologies ((12),3), ((13),2), ((23),1), each selected with probability 1/3; three image sites forming an 8-neighbor triangle make all pair merges spatially legal. With root-stop r and child-pair-stop s, the induced law has mass r on {123}, (1-r)s/3 on each pair partition, and (1-r)(1-s) on all-separate. Thus (r,s)=(1/2,0) and (1/4,1) realize laws A and B inside the actual circuit. Interior settings (9/20,3/11) and (3/10,6/7) also share all pairwise marginals 1/2 but have different triple probabilities, so the distinction does not depend on saturated gates.

Joint differences alone need not change pixel predictions. With four uniformly named region labels and identical soft local image evidence favoring class 0 by a factor of four at sites 1 and 2, exact enumeration gives posterior P(y3=0)=61/100 for A versus 139/250 for B. These are synthetic image-evidence factors, not inference-time GT. Both constructions and interior settings were checked using exact fractions. The circuit must represent these factors, preserve disjoint scopes, and match semantic supervision/compute against a pairwise-transfer mixture. This addresses the model-family objection but proves neither learning nor cardiac value. A binary shared regime is retained as the peer's alternative; it cannot realize missing spatial partitions on a single fixed tree.

Accepted also: compare operator decoding against equal-step image-anchored recurrence, and gate on projected-response consistency rather than calling transition compression exact. Rejected only the inference that fixed-tree identifiability rules out the explicitly expanded mixture. My separate `peer_review.md` challenges geometry's teacher-energy distortion and remaining-budget state equivalence; disagreement is preserved.
