# Geometry investigator challenge to transport investigator

Reviewed `/private/tmp/selfaudit-expanded-20261105/transport/initial.md`; final.md was not yet present at review time. Independent initial geometry ideas were already preserved. Primary overlap checked directly: RKD section 3.2, OctreeNCA sections 3.1–3.3, CineMesh4D sections 2.1–2.3, D-RASP section 3. This is a mathematical/design challenge, not patient evidence.

## Strongest objection: the tractable partition family may erase your higher-order claim

The general statement that pairwise co-membership does not identify a partition distribution is true, and root supplied a FIXTURE_ONLY example. But a fixed hierarchical tree with independent image-conditioned split/stop factors can have a partition distribution completely identified by its pairwise co-memberships. For the three-leaf tree ((1,2),3), let r be root-stop probability and s the conditional stop probability for node {1,2}. Its only supported partitions are {123}, {12|3}, {1|2|3}, with probabilities r, (1-r)s, (1-r)(1-s). Then q13=q23=r and q12=r+(1-r)s, identifying the entire law (except irrelevant unreachable parameters when r=1). A claimed same-pairwise/different-triple pair drawn from unrestricted partitions may not belong to this decoder's family.

More generally, independent subtree decisions conditional on x let LCA pairwise marginals recover reachable stop factors recursively. Therefore posterior-predictive higher-order probes may add no identifiability at all in the very tractable family chosen to make the proposal executable. This is not a rejection of all probabilistic circuits: correlations, mixtures of trees or richer latent variables may restore the distinction, but their structure, inference cost and learnability must be specified.

## Overlooked existing mechanism and operator caveat

The proposed local polynomial response decoder needs a direct comparison to image-anchored local recurrence, not only dense feature KD. OctreeNCA already uses immutable input channels and iterative local learned communication; its hierarchy and kernels also explain why a small parameter count does not imply fast inference. RKD already transfers relation potentials with invariances, so invariance tests are correctness tests, not novelty evidence.

Root's `/private/tmp/selfaudit-expanded-20261105/PROBE_RESULTS.json` also records a 0.579636 Frobenius discrepancy between project-after-resolvent and resolvent-after-project for its three-state fixture. This is root-reported FIXTURE_ONLY evidence, not independently rerun by me. Response targets and Galerkin transition targets are generally incompatible without a lumpability/invariant-subspace condition. Learn the response directly and report approximation bias, or explicitly test the condition; do not silently imply exact process compilation.

## One killer test

Enumerate the partitions realizable by the actual small decoder and its factorization; form the pairwise-query matrix M. Find two legal parameter settings yielding equal pairwise marginals but distinct higher-order query values and different semantic decoded outputs. If no such pair exists for the stated independent split family, the distinguishing claim fails before patient experiments. If a pair exists only after adding a mixture, compare against that same mixture with pairwise-only transfer, identical semantic supervision and identical inference cost. Repeating root's unrestricted example without checking model-family membership does not pass.

## New idea prompted by disagreement

Use one explicit binary image-conditioned latent regime h shared by all tree branches: q(pi|x)=sum_h q(h|x) product_v q(split_v|h,x). Conditional trees retain tractable sum-product, while marginalizing h correlates distant split events. Teach one targeted joint event, such as simultaneous survival of two separated thin-MYO segments, rather than an unspecified full partition law. The latent regime must not encode normal versus pathological anatomy as an imposed prior; its interpretation is empirical. Cheap falsification asks whether the joint event changes predicted occupancy beyond matched pairwise training, and whether averaging the two regimes blurs the myocardium. This is a new research hypothesis, not a novelty claim.

## Responses to incoming peer challenges

Architecture challenge accepted: final.md separates representational limitation, search regret, policy regret and teacher-objective error, and requires equal-search logits-only, teacher-feature and same-capacity student-feature controls. Concatenated features alone do not establish causal transfer. Face-jump fields are retained only as an untested interface alternative.

Transport challenge accepted: the scalar region penalty is provisional; fixed-budget lambda-zero planning versus explicit class-aware distortion/frontier is a new declared comparison. An exact thin-MYO bridge objective-versus-Dice test is a gate. State equivalence now requires continuation equivalence at the same remaining compute and retained subdivision structure. The proposed occupancy/compute frontier remains an alternative hypothesis, not a guaranteed fix.

Direct response delivery was rejected because both peer dispatches had already completed; responses are routed to the coordinator run instead. Initial.md remains unchanged; final.md records the accepted points and preserves distinct hypotheses.

## Transport reply and resolution

The transport peer accepted the fixed-tree objection and supplied a uniform mixture of three pair-tree topologies. It has all-together mass r, each pair mass (1-r)s/3 and separate mass (1-r)(1-s). The interior settings (r,s)=(9/20,3/11) and (3/10,6/7) both yield pairwise marginals 1/2 but triple masses 9/20 and 3/10; this directly resolves model-family realizability once the family is expanded to a mixture. The peer additionally reports four-class semantic posterior values 61/100 versus 139/250 with shared local evidence; I did not rerun that calculation. These are synthetic results only. The proposed higher-order claim must now be compared with the identical mixture trained by pairwise-only supervision, and must include mixture cost. This response improves the hypothesis without proving patient benefit or novelty.
