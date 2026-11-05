# Geometry council dossier

**PROPOSED / NOT RUN.** Preferred risky direction: transfer executable region decisions into an image-driven compact student. Retain a different transport decoder and a variable-incidence interface alternative. None is certified novel or demonstrated to reach 96 Dice.

## Scope and four requirements

Read-only source HEAD: `9958d421ccae1da3b8527f74da5438dbb9f3f521` at `/Users/alvinluong/Self-Audit`. `ACTIVE_PIPELINE.json` and `docs/METHOD.md` establish dense labeled TRAIN accounting, image-only U, and audit-as-optimization boundaries. Prospective architectures are new protocols, not current-v2 parity claims. All labels used for teacher heads, student fitting, diagnostics or gates belong to declared L; no U/VAL/TEST mask influences these operations. Primary inputs are the same original 2D MRI x. Temporal, native-volume, multiview, external anatomy or extra-label variants require separately declared information budgets.

The strong teacher remains a separate hypothesis: fit a high-capacity teacher with native semantic supervision and optional boundary/region heads derived from L, then compare identical backbone/compute without those heads. Freeze teacher versions for transfer. Before using the 96-Dice aspiration, predeclare foreground versus background-inclusive aggregation, patient/volume units, phase, domain weights and absent-class policy. No provided geometry paper establishes this teacher target on ACDC and M&Ms.

| Route | Strong teacher | Same-image teacher/student representation | Transfer hypothesis | Redesigned compact student |
|---|---|---|---|---|
| Region programs | Separate structural-head experiment | Joint region graph and planner | Useful operation returns | Sparse executable region policy |
| Transport | No independent quality guarantee | Teacher response graph/student coefficients | Normalized intervention responses | Finite absorbing recurrence |
| Interfaces | No independent quality guarantee | Joint interface/incidence alternatives | Incomplete until tested | Query/curve rasterizer |

All deployment paths receive x only, plus declared validated image geometry; no masks, contours, supplied seeds, atlas, teacher or extra views.

## A. Executable region programs

Begin with a coordinate partition and fine atomic grid. Legal operations are SPLIT, adjacent MERGE, semantic RELABEL and STOP, with action budget K. Merges retain subdivision trees. State includes remaining budget, geometry, semantic probabilities and legal continuations. Named semantics come from L, never from a presumed ventricular layout. Disconnected MYO, absent chambers and pathological openings remain legal.

The joint training representation is

    z_r = [pool_r E_T(x), pool_r E_S(x), geometry(r), incidence(r)].

Teacher features estimate uncertain semantic/interface evidence; student features propose realizable actions. A bounded planner evaluates

    E_T(P,c) = sum_r sum_(v in r) u_T(v,c_r)
                + sum_(r,s adjacent) b_T(r,s,c_r,c_s) + lambda |P|,
    A_T(s,a) = E_T(s) - E_T(Complete_H(s,a)).

Every candidate gets identical completion budget H and allowed search expansions. The transferred object is a distribution over executable operations and their achievable returns. A student action policy learns these preferences and values; region semantic logits receive native supervised CE/Dice on L. Discrete partition choices learn through action-return supervision, not an assumed differentiable hard rasterizer. U returns remain teacher pseudo-objectives, never certified anatomical improvement.

**Peer correction:** the initial scalar complexity penalty can sacrifice thin MYO even with excellent teacher evidence. The provisional energy therefore has no accepted production setting. The first objective comparison is fixed K with lambda=0 versus a declared class-aware distortion/computation frontier; select on charged TRAIN diagnostics only. Class weights, denominators and absent-class treatment are explicit experimental factors. A vector frontier is a viable alternative, not a proven cure for teacher bias.

Training graph: x -> frozen teacher -> structural evidence; same x -> student -> local descriptors/action shortlist; both -> joint graph -> bounded return targets -> student policy. L also supplies native semantic supervision and teacher-head calibration. Student-only executions are trained and evaluated directly; no inference-time teacher tensor is missing. This is learned policy compilation, not exact decompilation or teacher-dropout machinery. Feature concatenation alone proves nothing: teacher structural access must beat both logits-only and equal-capacity student-feature planners at identical search effort.

Inference graph: x -> depthwise stride-4 stem -> constant-width sparse region graph -> K operations -> native semantic rasterization. On-demand native image patches provide fine boundary descriptors. There is no contracting/expanding U-Net ladder. Count graph edges, visited regions, sequential decisions, native patch work and unavoidable O(HW) output cost. Parameter count alone cannot establish latency or peak-memory superiority.

### Decisive tests and killer cases

First test representation before teacher training. For a fixed partition, exact best pixel accuracy is

    Acc*(P) = (1/N) sum_r max_c n_(r,c).

This is not a Dice ceiling. Compute attainable class Dice separately; exact tiny-patch enumeration can establish a budget ceiling, whereas heuristic whole-image fitting cannot prove impossibility. Include thin bridges, cropped basal views, apical absence, dilation, unusual connectivity and annotation ambiguity using only TRAIN information.

Separate four errors: representational limitation at K; bounded-search regret versus exhaustive search within the same objective; deployed-policy regret versus its planner; and teacher structural-objective error versus charged native labels. These are diagnostic axes, not an additive Dice theorem. Compare supervised adaptive partitions, teacher-logit planning, teacher-feature planning, equal-capacity student-feature planning, and equal-step recurrence. Hold K and search expansions fixed. Compare current feature/gradient transfer under matched data, initialization and inference budget when models become available.

The cheapest decisive gate is exhaustive tiny-grid planning with a one-pixel MYO bridge: vary declared distortion and K, then score actual foreground Dice independently of E_T. Failure with perfect teacher evidence rejects that objective/budget before transfer. Next assess action ranking and harmful actions on small charged TRAIN folds. Freeze effect thresholds and strata before observing outcomes; later efficacy requires frozen patient predictions and uncertainty outside TRAIN.

Identical masks are not equivalent action states when budgets or subdivision trees differ. Quotient only continuation-equivalent states with the same remaining resources; commuting operations may be interchangeable. Otherwise imitation targets can erase future useful options. Confidently wrong teacher boundaries, excessive fragmentation and sequential runtime overhead remain killer cases. No topology repair may conceal them.

## B. Reopened after evidence: absorbing transport recurrence

Learn positive image-conditioned conductances w and semantic absorption rates a, including optional unknown absorption a0. With `D_ii=sum_j w_ij+sum_c a_ic+a_i0`, decode

    P^(t+1) = D^-1(W P^t + A).

Teacher/student training uses the same image graph: teacher responses provide targets while student features predict every deployed coefficient and semantic source. Transfer normalized finite source-intervention responses rather than raw conductances. Global rate scaling leaves the solution unchanged, so coefficient imitation is nonidentifiable. Every connected component needs absorption; if `rho=max_i sum_j w_ij/D_ii<1`, iteration error contracts in infinity norm by at most rho. This says nothing about Dice.

This alternative is reopened as a bounded local decoder after reading OctreeNCA. Strong absorption accelerates convergence but preserves wrong sinks; weak absorption enables distant propagation at greater iteration cost. Compare equal-step learned recurrence, no-transfer diffusion and a direct head. Kill it if extra iterations erase compactness or wrong image-predicted seeds dominate. Unknown is not background; report all-patient coverage and raw segmentation.

Root's `PROBE_RESULTS.json` reports a FIXTURE_ONLY 0.579636 Frobenius gap between projected fine resolvent and coarse-operator resolvent. I inspected the receipt, not reran it. Thus coarse transitions do not guarantee correct coarse responses: require lumpability/invariant-subspace conditions or directly measure response approximation. No patient claim follows.

## C. Variable-incidence semantic interfaces

Predict oriented fragments, left/right class labels, presence and a small distribution over valid connectivity, then rasterize. Joint teacher/student image features provide training alternatives; the student predicts all curves/connectivity from x at deployment. Native masks on L supply semantics and boundary supervision without extra free labels.

Correction to initial.md: an interface cannot terminate alone inside a fully labeled 2D domain. Ends must meet the image boundary, valid junction or explicitly censored region. Basal anatomical openness does not license inconsistent occupancy. Exact interface fitting under a fixed query budget is the first gate; tiny islands, partial volume and ambiguous conventions may defeat it. A peer proposed integrable face-jump fields with image-derived anchors; retain this as an untested variation, since integrability does not identify class names or ensure pathology preservation.

## Sources, peer decisions and evidence limits

Four centrally supplied primary methods were read directly:

- [RKD](https://www.alphaxiv.org/pdf/1904.05068), section 3.2, local text lines 385–715: relation transfer already exists and semantic outputs need task/individual signals. Our distinguishing prediction concerns changed executable allocation beyond RKD on identical descriptors.
- [D-RASP](https://www.alphaxiv.org/pdf/2602.08857), section 3, lines 540–941: exact reparameterization has a restrictive normalization assumption and potentially exponential size; simplification is empirical. No theorem here compiles arbitrary cardiac teachers into compact programs.
- [CineMesh4D](https://www.alphaxiv.org/pdf/2605.13994), sections 2.1–2.3, lines 95–465: image-to-mesh decoding already exists with fixed connectivity and multiview/temporal inputs. Variable incidence under one-image inputs is the remaining hypothesis, not geometry output itself.
- [OctreeNCA](https://www.alphaxiv.org/pdf/2508.06993), sections 3.1–3.3, lines 187–278: immutable-image local recurrence, hierarchical hidden-state transfer and kernel-dependent cost already exist. Quadtree structure and few parameters are insufficient distinctions.

Architecture peer objections on causal teacher access and four-way error separation are accepted. Transport objections on distortion and continuation equivalence are accepted. Their frontier and face-field alternatives remain hypotheses; I reject treating either as an established correction or requiring council convergence. My outgoing challenge (`peer_review.md`) shows independent tree splits can be identified by pairwise marginals. The peer responded with a three-tree mixture realizing identical pairwise but different joint laws and reported different semantic posteriors under shared evidence. This addresses family membership by changing the family; equal-mixture controls and computational accounting remain required. Its semantic posterior calculation is peer-reported synthetic evidence, not independently reproduced here.

Compared with preserved initial.md, A has narrower learned-policy claims and stricter gates; B is reopened as finite recurrence; C has valid endpoint semantics and reduced novelty scope. Two distinct 3D anatomies can yield indistinguishable single slices: missing information is separate from optimization failure. The bounded literature search cannot certify novelty.

All efficacy, teacher quality, GPU latency, learnability and 96-Dice aspirations are NOT RUN. No training, downloads, repository edits or commits occurred; repository tests were not run for research-only documents. Initial SHA256 remains `2cd41367e4efcbb3cb09b0673ba3ee512354f9aaa5975a544ea87f361586c097`; source ACTIVE_PIPELINE SHA256 `a451851b96ef382fd8de5f0542a6f44e0db61954c0adb66facec9a0e14ee904b`, METHOD `21ea10c4071c65816299b3b3f96d9167bfea6e5aeae256372908b7fa41f5be2e`.
