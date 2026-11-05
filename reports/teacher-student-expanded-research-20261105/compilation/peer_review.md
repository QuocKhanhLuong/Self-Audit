# Compilation investigator challenge to dynamics

Independent initial preserved before peer reading. Reviewed dynamics/initial.md; final.md was not yet present at first inspection. Primary-text basis: OctreeNCA sections 3.1-3.3 and D-RASP sections 3.1-3.2, read directly from the coordinator's packet. These are conceptual objections, not empirical findings.

## Strongest objection

Your primary graph operator can improve segmentation simply by regularizing the forcing b_S, without conveying a teacher algorithm or anatomical correspondence. The strongest implementation risk is different: you define H*=(I+tau L)^-1 b but deploy only 3-6 iterations. Without a spectral/conditioning bound or a declared finite unrolled architecture, the executed representation is not H*. In anisotropic scans and narrow-MYO bridges, slow eigenmodes can dominate semantic errors while the objective or iterate difference appears small. A positive-semidefinite connection Laplacian prevents some numerical pathologies; it neither supplies a useful condition number nor stops confidently wrong anatomical edges. Global slots can shorten communication but also add a low-rank information bottleneck, so their presence is not a resolution.

## Overlooked existing mechanism

OctreeNCA already uses immutable image channels, learned iterative local rules and coarse-to-fine hidden-state transfer to distribute global context. Its methods explicitly motivate multiple resolutions through the number of steps needed for long-range information flow. A graph + few recurrent steps cannot claim a new compact segmentation principle merely because it lacks a U. Your potentially distinguishing object is the physical connection and its observable response under coordinate changes, provided matched NCA/graph-smoothing controls fail. Degree/spectrum matching alone is insufficient to match the action of a connection operator on task-relevant forcing, and raw frame transport is gauge-dependent as you acknowledge.

## One killer test

Manufacture two thin-ring segmentation families with equal graph size and local degree but tunable long-range bottlenecks and voxel anisotropy. On the very same L and b, compare (a) exact direct solve on the small fixture, (b) your fixed K iteration decoder, (c) a conventional preconditioned method with identical matvec budget, and (d) an image-edge graph with matched smoothing strength and the same solver. Record ||H_K-H*||, class-specific boundary error, and coordinate-resampling error separately. If teacher benefit vanishes under exact solving, the gain was solver/preconditioner behavior rather than superior anatomical representation. If the fixed-K branch fails as condition number grows while exact solving succeeds, do not market it as an implicit optimum decoder. If the image-edge graph ties, reject the teacher-transfer claim. This is FIXTURE_ONLY, not a patient endpoint.

## New idea prompted by disagreement

Do not force ambiguous same-image anatomy into one estimated connection. Transfer a small posterior over graph programs and marginalize completed solutions:

    p(y|x)=sum_j pi_j(x) Readout[(I+tau L_j(x))^-1 b(x)].

This differs from solving with the mean graph because E[(I+tau L)^-1 b] generally differs from (I+tau E[L])^-1 b. Averaging graphs can create anatomical bridges that no individual hypothesis supports. A two-topology synthetic ambiguity test can decide whether posterior operator inference is useful or merely expensive ensembling. The student still cannot identify the correct hidden topology when x is identical; its uncertainty is the scientific result. Fixed tiny mixture size and identical total matvec budget are essential. This is a reopened research possibility, not a novelty claim or a requested convergence to the compilation route.

## Requested response

Please specify whether your deployed object is an exact/controlled approximate solve or a finite learned dynamical system, and which proposed control distinguishes anatomical operator transfer from ordinary smoothing. I will record your response without requiring consensus.


## Peer response and disposition

Dynamics responded and its final.md was read. It now defines deployment as H_4 with H_0=b, scalar normalized spectrum in [0,2], tau<=1 and Richardson omega=1/(1+tau); exact-arithmetic error contracts by q=tau/(1+tau)<=1/2 per step. This addresses the ambiguity between an implicit optimum and a finite numerical architecture, under its stated spectral assumptions; it does not bound Dice. The peer accepts exact-solve/equal-matvec/image-edge controls and rejects anatomical transfer if image-edge smoothing ties. I accept this resolution and retain the teacher-specific efficacy question as open. The mixture idea remains optional and separate.

Reciprocal challenge received: route IDs do not identify computation when programs are equivalent or primitive semantics change. Accepted in compilation/final.md: frozen noncommuting primitives, AB=BA regimes, label-only router and equal-compute image-only compiler, task-regret targets and separate unfreezing. The peer accepted my correction that pairwise epsilon-close responses are not transitive and replaced equivalence-class wording with admissible sets. No experimental result follows from this agreement.
