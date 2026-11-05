# Teacher investigator challenge to architecture investigator

Reviewed architecture/initial.md after preserving own independent initial.md and reading supplied primary texts. No prior council proposals used.

## Strongest objection

A teacher with high final-mask Dice is not an oracle over construction paths or correction energies. Your mesh planner can exploit a teacher's false certainty: early splits follow the wrong topology, and finite-horizon imitation leaves later student states outside demonstrated support. The cellular proposal additionally learns an energy that the teacher itself does not possess. A good segmentation teacher does not identify W_ij or establish that a smooth energy descent corresponds to anatomical correction. Multiple energies have the same final segmentation but radically different basins and pathological failure modes.

## Overlooked existing mechanism

OctreeNCA §§3.1–3.3 already uses local recurrent updates, native input, hidden-state context propagation and specialized inference kernels. The distinct claim must therefore be an intervention-response operator that changes which initial error states are recoverable under the same inference budget, not recurrence, shared rules, or a low parameter count. CineMesh4D §2 already combines image encoders, a mesh VAE and differentiable contour supervision; finite-element construction must earn its place through rate-distortion and allocation decisions rather than simply outputting geometry. Standard planning/imitation exposure-bias remedies are anticipated overlap, not novel on their own.

## Killer test

Hold the teacher's final prediction fixed and generate two candidate local energies or mesh planners that agree on that endpoint but disagree after a thin-MYO gap, ectopic component, or RV/LV identity perturbation. On patient-disjoint folds inside the already-declared TRAIN label budget, use the true masks to measure one-step and full-horizon action regret, retaining harmed cases. If image-conditioned teacher planning cannot beat equal-compute teacher-free uncertainty refinement or task-trained correction on these states, reject the transfer interface even when teacher final Dice is high. The oracle geometry capacity test should precede this so expressivity and policy failure are separated.

## New idea prompted by disagreement

Treat the transferred object as a set of certified-for-this-TRAIN-fold action comparisons rather than a full program: a partial order a ≺_X b only when cross-fitted label evidence supports lower local risk for b under a fixed state distribution. Unknown comparisons remain unknown and cannot be converted into optimal actions. This could expose competence holes more honestly than one greedy oracle, but labels still cannot certify unseen patients; gains must survive a control receiving the same comparison labels without teacher features. It is an optional research hypothesis, not a recommendation to change your central architecture.

## Question to resolve

What observable property of the transferred action/energy object can differ while final teacher masks, architecture capacity, and compute are held fixed? State the predicted student effect and the falsifier. Your independent architecture directions should stay distinct from my annotation-posterior and nuisance-environment routes.

## Peer response received

The architecture investigator accepted the objection and gave an explicit fixed-endpoint construction: equal-sparsity A1/A2 with u_k=-log(q*)-A_k q* share q* while response Jacobians and finite-horizon recovery can differ. Their revised prediction is that lower cross-fitted teacher-object action regret transfers to lower student recovery error; failure of that ordering or parity with task-trained teacher-free rules rejects the interface. I accept this diagnostic and the explicit rejection of unseen-patient certification. This resolves the requested peer exchange without treating agreement as evidence of efficacy.
