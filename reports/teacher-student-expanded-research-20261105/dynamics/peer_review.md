# Dynamics review of compilation proposal

Reviewed compilation/initial.md after preserving my own initial hypotheses and inspecting the central evidence packet. compilation/final.md was not yet present at this checkpoint. This is a mechanistic challenge, not a rejection by consensus.

## Strongest objection: program identities need not identify transferred computation

The proposed KL(q(pi|teacher,student)||p(pi|student)) treats a named action sequence as the knowledge target. Two programs may have identical execution on the target inputs, while the same named program can perform very different computations after its trainable primitive parameters change. Thus exact route imitation can be unnecessary, and improved route prediction can fail to indicate anything transferred. If primitives/readout are sufficiently expressive and trainable, one fixed order can absorb both manufactured regimes; a fixture claiming opposite required orders must prove this representational restriction rather than assume it. Program order is a causal claim only under fixed or explicitly constrained primitive semantics.

A second issue: eight or twelve program bits may encode coarse mask archetypes or patient groups, even without a pixelwise teacher bypass. Limited entropy reduces capacity but does not establish algorithmic content. A teacher-free compiler with the same search/meta-training compute is essential.

## Overlooked existing mechanism

The central packet's RASP source already concerns conversion into executable primitives and subsequent causal pruning/replacement; the proposal correctly needs something beyond mere executability. Based on the packet rather than an independent full-text read, I do not assert exact equivalence with RASP. More directly, our inspected OctreeNCA §§3.1–3.3 already uses repeated compact local rules plus global-context propagation, and RKD §3.2.4 stresses that relational targets need task semantics. Tied local transport/global exchange therefore cannot themselves establish the novel component. The surviving hypothesis is teacher-dependent selection of useful, constrained noncommuting computations that generalizes under interventions.

## One killer test

Freeze two genuinely noncommuting primitives A and B whose meanings are fixed analytically. Construct two observable input regimes requiring AB versus BA, plus an additional regime on which AB and BA are exactly equivalent. Give q and p the same student-visible regime information; the teacher adds only richer training knowledge. Hold out combinations of regime and appearance. Compare fixed best order, hard student router trained with labels alone, image-only compiler with identical search compute, teacher compiler, and shuffled teacher assignments. Evaluate task outcomes and explicit AB↔BA interventions, not route accuracy alone. If teacher advantage disappears against image-only search, or route KL penalizes the equivalent regime without improving outcomes, reject computational-transfer claims in this setting. Then unfreeze primitives as a separate factor: if dynamic ordering becomes dispensable, report that rather than retaining the original interpretation.

## New idea prompted by disagreement

Compile to an admissible set of bounded programs, not a unique teacher-selected program ID. For an input family and a small predeclared suite of coordinate/appearance interventions G, accept programs whose responses satisfy a predeclared task-regret tolerance across G. The training compiler returns that admissible set, and the student selects its cheapest executable representative. Pairwise epsilon-closeness is not transitive and must not be called an equivalence relation; exact response equality or an explicitly defined partition would be needed for equivalence classes. This is a finite, intervention-relative criterion, never a theorem of global equivalence. It could reduce impossible route-identification targets and connect compact compilation with physical-coordinate invariance, but risks becoming another matching objective unless representatives actually change deployed execution and total cost. The proposed killer test should come first.

## Implication for my own direction

My graph operator can suffer an analogous problem: raw operator differences can matter little for the forcing actually encountered, and projected teacher operators need not preserve responses. I accept the coordinator's FIXTURE_ONLY projection counterexample and require response validation on retained student forcing plus fine-boundary support. I will not call coarse graph transfer exact merely because its adjacency was copied correctly.

## Response to incoming compilation challenge

Accepted and integrated in final.md. The deployed representation is H_4, not an exact inverse: H_0=b; H_(k+1)=H_k+omega[b-(I+tau L)H_k], with a normalized scalar Laplacian spectrum in [0,2], 0≤tau≤1, and omega=1/(1+tau). Eigenvalue-wise contraction is at most tau/(1+tau)≤1/2, so four iterations give a norm error at most 1/16 of initial error; floating point and semantic errors are separate. The ideal inverse derivative in the dossier is now explicitly labeled ideal; implementation must differentiate the actual polynomial. This restriction does not automatically extend to arbitrary connection maps.

Accepted the exact-solve versus equal-matvec numerical-alternative test and image-edge smoothing control. If image-edge smoothing ties, no teacher anatomical-transfer claim survives; if gains exist only with the finite solver, they are solver behavior. Accepted graph mixtures only as a later uncertainty experiment at matched total compute, not as an identifiability cure or a new central convergence.

## Peer reply received and terminology corrected

Compilation accepted frozen noncommuting primitives, task-regret admissible program sets, equivalent-behavior regimes, and teacher-free matched-compute controls. I accept its mathematical correction: epsilon-close outputs are not transitive, so my original equivalence-class wording was too strong; the new-idea paragraph above now uses admissible sets unless exact equality or an explicit partition is supplied. This correction changes the proposed target definition, not the scientific conclusion.
