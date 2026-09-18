# Question and mathematical baseline

Can image-only partitions be named conservatively as cardiac anatomy, and can a
single scratch CNN amortize their computation without reducing meaningful coverage?
This is the immediate falsifiable question. Faster learning is downstream of a useful
target, not a substitute for it.

For normalized full-FOV image x, define image-only descriptor
v_i=(x_i,mean5(x)_i,0.10 y_i,0.10 z_i). C0 executes 15 Lloyd updates for four centers,
initializing intensity components at quantiles (0.125,0.375,0.625,0.875) and coordinates
at zero. It orders resulting group IDs by their mean image intensity. Empty groups
remain legal; no area-equality condition or class-presence condition is imposed.

Let P0=T(x) be this anonymous partition. The frozen resolver N returns draft anatomy
D and validity V. Export A_i=D_i if V_i>0, otherwise A_i=255. The order is permanently
BG=0,RV=1,MYO=2,LV=3. The resolver is run on the actual unpadded FOV, then output is
returned to the native grid using nearest-neighbour resampling and the original affine.

For C1, p_theta(x)=softmax(CNN_theta(x)). The cache arm minimizes

    Lcache = -sum_i M_i log p_theta(x)[i,P0_i] / sum_i M_i,

where M is the image-FOV support, not an anatomical mask. Targets are detached cached
integers. At inference argmax clusters are sorted by image intensity and passed to N.
One network forward does not mean zero postprocessing cost: N and native export remain.

The direct control uses c_k=sum_i M_i p_ik x_i / max(sum_i M_i p_ik,1e-6),

    Ldirect = sum_ik M_i p_ik (x_i-c_k)^2 / sum_i M_i
              + .01 TV(p) + .01 mean_M[-sum_k p_ik log p_ik].

TV is the sum of horizontal/vertical mean absolute probability differences on pairs
inside M, averaged over four channels. All means and probabilities are differentiable;
there is no auditor or detached pseudo-target in this arm. The conditional-entropy term
promotes confidence and can encourage collapse. A marginal-entropy term is deliberately
absent: anatomical area balance would be a different and unjustified assumption.

## Identifiability and the three user ideas

The image reconstruction/region objective is invariant under a permutation pi of
cluster IDs: L(x,p)=L(x,pi p). MRI blood pools can also share intensity statistics.
Thus optimizing it does not identify named RV/LV correspondence. A naming prior breaks
symmetry by assumption; its validity must be tested independently, with unknown allowed.

C1-cache approximates a deterministic teacher T(x). It adds no new observation of
anatomical truth. A network could improve a noisy teacher through a useful inductive
bias, but low CE or high agreement cannot prove that; independent frozen evaluation is
required. Lower inference latency must amortize the one-time cache and training cost.

An RNN h_(t+1)=R(h_t,x,A_t), A_(t+1)=U(h_(t+1),x,A_t) can encode optimization progress
or unobserved trajectory state. If the desired next teacher edit is already a function
of (x,A_t), a history benefit requires evidence beyond smoothing, momentum or extra
compute. It cannot correct systematically wrong semantics merely by remembering them.
No useful correction teacher has yet been admitted, so no C2 is implemented this cycle.

If revisited, memory is reset within each image episode, never across shuffled patients.
Compare equal-step no-memory refinement, final-mask vs trajectory distillation, EMA
prediction history, on-policy rollout, stale targets, stagewise harm/oscillation and
history interventions that distinguish lost information from out-of-distribution noise.
Teacher at training time need not be an inference input. This is PROPOSED, not tested.

The previous event proxy's negative result concerns the specific region/edge RF signal.
The previous CPU64/120-update experiments cannot rule out all auditors or establish
convergence. Here the baseline can drop auditor/history without claiming those families
are impossible. No mechanism-level novelty is claimed for these standard controls.
