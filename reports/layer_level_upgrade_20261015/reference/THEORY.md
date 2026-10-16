# Layer/function-level upgrade: statements, proofs and limitations

**Date:** 15 October 2026. **Reviewed Self-Audit:** `main@4d58f3aebe4f205b58b3184c8971b1d838109d82`.

This is an executable research prototype with mathematical/operator tests, not a trained medical segmentation model. The strong-student concept has been upgraded at input adaptation, output parameterization, optimization and image-grounded refinement. Generic projection, pyramid and mirror-descent techniques are prior art. The proofs below are derived for the specified operators; no novel-theorem or SOTA claim is made.

## 0. Objects, supervision and implementation scope

Let X be a cine image, A its permitted acquisition metadata, and Y its unobserved true segmentation. Y is never an input to the proposed training operators. A frozen representation teacher provides structural features; a separate image/anatomy readout creates potentially erroneous named evidence. These are not synonymous with a pretrained segmentation oracle.

A student outputs native logits l_theta(X) in four classes BG/RV/MYO/LV. UNKNOWN is an abstention decision, not BG. Cached teacher features and labels must be transported with their valid support and coordinate transforms. Images used for a deployment result must not secretly supply fitting statistics from the whole test cohort.

Actual run evidence motivating the design: on the fixed Z pilot probe, total KL fell from 0.569224894 to 0.250586301 while RV-conditional KL rose from 1.399583817 to 1.693750381. This is a single frozen probe, not a population theorem. The current proposal consequently targets the gradient function, rather than assuming a positive average feature/semantic gradient cosine protects every class.

### The built prototype

- Caller-supplied official DINOv3 ConvNeXt native-stage adapter, without hidden downloads or segmentation heads.
- Four-stage residual decoder, native detail path and coarse/detail logit factorization.
- Evidence and partial-label losses plus a masked-feature loss primitive.
- Selected-parameter, class-proxy gradient projection plus explicit finite-step acceptance.
- Detached entropic refinement for one precisely specified frozen image energy.

No automatic pseudo-label acceptance policy, recurrent neural controller, end-to-end ACDC trainer or clinical certificate is included. The image-energy primitive proposes corrections; it does not certify their anatomical correctness. A random full-size ConvNeXt-Tiny from torchvision was used for CPU shape/backward tests. It was NOT loaded with DINOv3 weights.

## 1. MRI input adaptation that does not randomly reinterpret RGB weights

Three adjacent slices are x_-, x_0, x_+. The image-only preprocessing maps their intensities into a declared common [0,1] range. Define the original repeated-center normalization

b_c(x_0)=(x_0-mu_c)/sigma_c,

and an input adapter

T_B(X)=b(x_0)+B*[x_- - x_0, x_+ - x_0],

where B is a bias-free 3x3 convolution from 2 to 3 channels, initialized to zero. The existing pretrained first convolution is not replaced.

**Proposition 1.** At initialization, for arbitrary neighboring slices, T_0(X)=b(x_0). Consequently any deterministic pretrained backbone F returns exactly its repeated-center output F(T_0(X))=F(b(x_0)), up to arithmetic identity. If all slices are equal, this equality holds for every B. Moreover dL/dB can be nonzero at B=0 whenever slice differences and the downstream gradient have a nonzero correlation.

**Proof.** The residual is identically zero at B=0. If all slices equal x_0, both residual inputs are zero and bias-free convolution remains zero for all weights. The convolution weight derivative is a cross-correlation between its inputs and its output derivative, which need not vanish merely because the weights are zero. This avoids initializing both a residual gate and residual weights at zero, which can create a dead adapter. QED.

**Scope.** This preserves the function of a repeated-gray baseline, not its anatomical accuracy. It says nothing about the original RGB input distribution or the quality of pretrained features on MRI. Correct normalization and alignment remain required. Dropout/randomness must be matched for a whole-backbone equality test. No claim is made that three MRI slices are RGB.

**Code:** `CenterDifferenceInput`.

## 2. Exact coarse/detail decomposition, not arbitrary bilinear resizing

On a native grid divisible by k, let D average disjoint k-by-k cells, U repeat each coarse element over its cell, and Pi=UD. Then DU=I. U=k^2 D^T under the ordinary Euclidean inner products, so Pi is symmetric and idempotent. Set H=I-Pi.

We parameterize logits as

l = U a + H r,

with coarse logits a and a raw native detail field r. The built version bounds r=b*tanh(v), where b>0 is a fixed recorded buffer (`detail_bound`, default 1.0 is an unvalidated design choice).

**Proposition 2a (exact resolution identity and direct gradient separation).** Dl=a. For any differentiable coarse guidance loss J(Dl), its derivative with respect to the detail variable v is exactly zero, even when r=b*tanh(v). This does not require a small loss or small detail amplitude.

**Proof.** DH=D-DUD=D-D=0; hence Dl=DUa+DHr=a. The chain derivative through the detail path is (dr/dv)^T H^T D^T grad J=0. QED.

An additional fixed resampling S AFTER D is allowed: J(SDl)=J(Sa). Replacing D by an unrelated interpolation applied directly to l destroys the guarantee. A teacher bank at 48x48 can be compared with a registered resampling of a, not with arbitrarily interpolated native logits while claiming the same identity.

**Proposition 2b (capacity and orthogonality).** Without the amplitude constraint, every field l* admits the representation a=Dl*, r=l*. Also Pi l* and H l* are orthogonal. For any coarse-only approximation Ua,

||l*-Ua||_2^2 = ||Pi l*-Ua||_2^2 + ||H l*||_2^2.

**Proof.** l*=Pi l*+H l*, Pi H=0 and both are orthogonal projectors. Ua lies in range(Pi), so the residual components in the equality lie in orthogonal subspaces. QED.

This is a representational statement on logits, not a proof that a finite CNN can learn every field or that squared-logit error determines Dice. The bounded implementation intentionally restricts large within-cell variations.

**Proposition 2c (bounded detail and an argmax certificate).** If |r_ic|<=b, then |(Hr)_ic|<=2b. If a cell's winning coarse class y has a_y-a_c>4b for every c != y, the native winning class is y throughout that cell.

**Proof.** A cell average of r is in [-b,b], so subtracting it gives an element in [-2b,2b]. At a pixel, l_y-l_c=(a_y-a_c)+(Hr)_y-(Hr)_c >4b-4b=0. QED.

**Scope and counterexamples.** Coarse probability is not mean native probability: softmax(Dl) generally differs from D softmax(l). The identity therefore does not guarantee native confidence, Dice or probability calibration. Shared encoder parameters can still change the detail output when coarse losses update them; zero direct detail-branch gradient is NOT invariance of the entire network's output. To protect parameters, an explicit optimizer scope is necessary. Conversely, a wrong high-margin coarse label can be hard to correct under a small b. The bound trades corrective capacity for bounded disturbance, not truth. The term 'high-pass' means cell-local zero mean, not a perfect Fourier band filter.

Odd native shapes are explicitly padded, with a returned validity mask. Algebraic identities apply to the padded grid before crop. Padded cells must not receive invented teacher evidence.

**Code:** `block_mean`, `block_lift`, `zero_mean_detail`, `compose_logits`, `LayerGuidedStudent`.

**Prior art:** multiresolution/Laplacian reconstruction in segmentation predates this work [2]. The contribution to investigate is the guidance contract and its empirical effects, not discovery of a pyramid.

## 3. Target information and confidence must not be conflated

Given class evidence mass M_i>=0 and support h_i=sum_c M_ic, define Z_i=M_i/h_i only where h_i exceeds a declared numerical minimum. These arrays must arise from the same nonnegative coordinate/graph transport, not from inverting a nearly uniform Q using an unstable denominator. Suppose the existing bank uses Q_i=(1-h_i)u+h_i Z_i with u uniform and h in [0,1].

**Proposition 3a.** CE(Q,p)=(1-h)CE(u,p)+h CE(Z,p), and the difference between the logit gradients for Q and Z is

g_Q-g_Z = (1-h)(Z-u).

**Proof.** CE is linear in its first argument and dCE(q,softmax(l))/dl=p-q. Thus (p-Q)-(p-Z)=Z-Q=(1-h)(Z-u). QED.

At an unconstrained per-input optimum, positive scalar reweighting does not change p*=Q. For Z=(0,0,0,1), h=.2, Q=(.2,.2,.2,.4): a perfectly fitted student remains below a .7 export gate. This is a counterexample to 'larger loss weight necessarily solves rejection'. Shared parameters and representation bias can alter actual outcomes; this is not a prediction of all trained neural networks.

**Proposition 3b.** With frozen targets/weights and the same normalization, CE(Q,p)-KL(Q||p)=H(Q) is independent of student parameters. Therefore replacing CE by forward KL alone leaves their gradients identical.

**Proof.** Expand KL=sum Q(log Q-log p). QED.

For genuine partial semantic evidence, a candidate set C has loss -log sum_{c in C}p_c. A singleton reproduces CE; the full class set produces zero semantic gradient. Such sets must actually include the true class under the declared information model. Brightness alone does not authorize a set {RV,LV}; it may wrongly exclude background. No balancing scheme can create a missing named class from nothing.

**Code:** `evidence_distribution`, `soft_ce`, `set_loss`.

## 4. Protect class proxies at the parameter step, not by hopeful scalar weights

Choose a specified parameter block theta_B, e.g. shared decoder parameters. Other parameters remain fixed during its protected step. Let L_c be differentiable losses on fixed, justified source strata of a fixed TRAIN witness batch. Membership, targets and weights are detached. Define g_c=grad_theta_B L_c and a proposed descent vector d0 from the total objective. Project

d* = argmin_d 0.5 ||d-d0||^2 subject to g_c^T d >= 0 for every active guard c.

**Proposition 4a (feasibility and first-order non-harm).** The constraint set is closed, convex and nonempty because 0 is feasible. The strictly convex objective has a unique minimizer. For theta'_B=theta_B-eta d*, each protected loss has directional derivative -g_c^T d*<=0 at eta=0.

**Proof.** Halfspaces have the stated convexity properties. Coercivity of the objective guarantees existence and strict convexity uniqueness. Differentiate L_c(theta_B-eta d*) at zero. QED.

For an active set S, the KKT candidate has d=d0+G_S^T lambda, where

(G_S G_S^T)lambda=-G_S d0,

lambda>=0, and all constraints must be verified. Enumerating at most four class guards requires at most 16 subsets. The code supports up to eight, normalizes nonzero rows (which preserves the halfspaces), solves in float64 and verifies KKT feasibility. A numerical failure returns a disclosed conservative zero vector, not a claim that an inaccurate vector is optimal.

**Proposition 4b (finite-step bound).** If grad L_c is L_c-smooth on the step segment,

L_c(theta_B-eta d*) <= L_c(theta_B)-eta g_c^T d* +(L_c eta^2/2)||d*||^2.

Consequently a sufficient condition for nonincrease is eta<=2 g_c^T d*/(L_c ||d*||^2), for every guard with nonzero d*.

**Proof.** Apply the smooth descent lemma to displacement -eta d*. QED.

If a guard's inner product is zero, this sufficient bound may permit no nonzero step. First-order feasibility is not finite-step safety. The reference `guarded_sgd_step` consequently evaluates every guard and the objective AFTER the actual additive trial update, halves eta on failure and restores all selected parameters if no trial passes. Its finite guarantee is solely that the *measured fixed proxy values* did not increase beyond the explicitly returned numerical tolerance.

**Important optimizer restriction.** Applying AdamW after projecting a raw gradient does not preserve this statement. Example: guard g=(1,-1), projected vector d=(1,1) has g^T d=0; a diagonal preconditioner diag(1,2) gives g^T M d=-1. Momentum and decay also change the actual step. A production AdamW integration must constrain or accept/reject the actual proposed displacement and restore parameter AND optimizer states. That integration is not silently implemented here.

**Counterexamples and cost.** Opposed gradients can make every feasible nonzero common improvement impossible; the layer can stall. Bad pseudo-label guards protect bad pseudo-labels. A fixed-batch certificate is neither all-TRAIN nor validation nor external-domain non-harm. Current default witnesses in the repository may favor early eligible batches; use predefined source-balanced witness batches, not VAL-derived selection. Several backward passes and gradient storage cost real compute. Start with a selected block, rather than claiming the full 33M network can be protected for free.

**Code:** `project_direction`, `guarded_sgd_step`.

**Prior art:** gradient halfspace projection is central to GEM [3]. The Self-Audit adaptation would be which grounded class/source proxies deserve protection, not a new generic QP.

## 5. Image-grounded self-learning as a verifiable operator

Let p_i be class probabilities at node i. Freeze an image-derived graph with nonnegative edge weights a_ij, a unary cost d_ic, source anchors z_i and source weights w_i. An edge is listed once. Consider

E_X(p)=sum_i,c d_ic p_ic
       +(lambda/2) sum_{(i,j)} a_ij ||p_i-p_j||_2^2
       +(beta/2) sum_i w_i ||p_i-z_i||_2^2.

The first term contains appearance evidence, the second structural affinity, and the third a soft source tether. Class semantics must come from justified unaries/anchors; an image graph alone cannot name anatomy. Temporal edges may be added ONLY if their correspondence and validity are supplied; this prototype builds spatial edges, not a motion estimator.

**Proposition 5a (smoothness).** With d_max=max_i sum_j a_ij and w_max=max_i w_i, the gradient is L-smooth in the joint Euclidean norm for

L=2 lambda d_max + beta w_max.

**Proof.** The Hessian is the graph Laplacian multiplied by lambda, plus the diagonal anchor term, repeated across classes. The graph Laplacian's largest eigenvalue is at most 2d_max: by ||v_i-v_j||^2<=2(v_i^2+v_j^2), its Rayleigh quotient is bounded by 2d_max. The diagonal term has largest eigenvalue beta w_max. The sum's operator norm is at most the sum of these bounds. QED.

Use entropic mirror descent:

p^+_ic = p_ic exp(-eta grad E_X(p)_ic) /
         sum_d p_id exp(-eta grad E_X(p)_id).

It preserves each simplex and positivity in exact arithmetic when p starts positive. The implementation performs the update in log space, uses float64, and rejects underflow/nonfinite conditions rather than inventing a probability floor.

**Proposition 5b (quantified descent).** For 0<eta<=1/L (or arbitrary eta>0 if L=0),

E_X(p^+) <= E_X(p) - (1/eta-L) KL(p^+||p),

where KL is summed over all nodes/classes.

**Proof.** The displayed update minimizes <g,q-p>+KL(q||p)/eta over the product of simplices. Comparing its objective with the feasible q=p gives <g,p^+-p><=-KL(p^+||p)/eta. Smoothness gives E(p^+)<=E(p)+<g,p^+-p>+(L/2)||p^+-p||_2^2. Nodewise Pinsker and ||v||_2<=||v||_1 imply ||p^+-p||_2^2<=2 KL(p^+||p). Substitute and simplify. QED.

This deliberately conservative bound allows a checkable step from the supplied graph, instead of a learned step size claimed safe without evidence. Numerical receipts also check nonincrease with tolerance 1e-10. Freezing the energy inside the loop is essential: updating appearance prototypes to make the candidate look good would change the objective being certified.

### A positive condition for correcting an initially wrong class

Let the externally defined true class at node i be y. Suppose its unary advantage over any competitor c is Delta_ic=d_ic-d_iy. For one update,

log(p^+_iy/p^+_ic)
 >= log(p_iy/p_ic) + eta[Delta_ic-2lambda deg(i)-2beta w_i].

**Proof.** Subtract the two log-update equations; the normalization cancels. Each class-probability difference is in [-1,1]. Therefore each neighbor term comparing two such differences has magnitude <=2, and the analogous source term has magnitude <=2. Use the worst-case signs and sum the nonnegative weights. QED.

In C classes, if the right side is at least m=log((C-1)tau/(1-tau)) for every c!=y, then p^+_iy>=tau. Indeed p^+_iy=1/[1+sum_{c!=y} exp(-log(p^+_iy/p^+_ic))] >=1/[1+(C-1)e^-m]=tau.

This result tolerates arbitrarily wrong anchors and cross-class edges when the informative unary margin is sufficiently strong relative to their influence and the initial error. A two-node synthetic construction in the tests satisfies it with two wrong anchors and an edge between different classes. It is a conditional semantic result: knowing that the unary favors the TRUE y is an assumption, not a label-free certificate that the code has verified on patients.

### Why the image operator still cannot certify anatomy unconditionally

A wrong unary can lower energy while changing a correct prediction into a wrong one; there is an explicit test. If unaries and anchors carry no semantic information, a constant uniform solution need not move. Similar-intensity regions are not necessarily the same anatomy. Label permutations remain possible under symmetric sources. A zero or low energy is not a correctness metric.

Use p^+ as a detached proposal. A self-learning loss can compare a strong-view native prediction with a geometrically transported p^+ from a weak view, but authorization/weighting of new labels remains an explicit experimental decision. The prototype does not silently authorize all proposals or re-label immutable banks. Teacher/source guards and audits must not be presented as independent evidence when they share data or errors.

**Code:** `FrozenImageEnergy`, `image_grid_graph`, `refine_distribution`.

**Prior art:** entropic mirror descent is classical; see [4]. This particular finite-dimensional proof was derived for the energy above. It is not transferred from an unrelated infinite-dimensional theorem.

## 6. What this upgrade can and cannot establish

Established for the operators: initialization equivalence, exact direct resolution-gradient separation, bounded detail perturbation, class-proxy directional constraints and checked finite steps, frozen-energy descent, and a conditional image-margin correction result.

Not established: better ACDC Dice, teacher correctness, minority-class improvement on unseen patients, external-domain generalization, calibrated uncertainty, superiority to a standard decoder, or novel theory. More parameters do not themselves prove any of these.

The new detail interface also does NOT explain the historical run by itself: prior stagewise scoring showed small raw-Dice differences between the historical common-grid and native outputs. It is a stronger control mechanism for a redesigned student, not a retrospective causal attribution.

## 7. Minimal experimental decision sequence

Use one strong backbone/initialization per matched comparison. First test an ordinary residual decoder versus the split head, with equal source targets and comparable parameter count. Next test selected-block guarding against the ordinary optimizer, checking BOTH fixed-proxy losses and exposed-development anatomy. Then test the frozen-energy proposer versus no proposer. The current package does not justify turning on every component in a new full run.

Freeze seeds, sample coverage, independent witness schedule and numerical tolerances before running. Report raw/gated scores, per-class PR and correct-class coverage, actual parameter displacement, rejected-step rate, source-stratum errors, and runtime. A refiner that always stalls or a guard that protects erroneous labels is a negative result. Domain generalization needs a recipe frozen before an external cohort is evaluated; train-distribution claims do not transfer automatically.

## Sources

[1] Official DINOv3 model implementation and loading/preprocessing contract:
https://github.com/facebookresearch/dinov3
https://github.com/facebookresearch/dinov3/blob/main/dinov3/models/convnext.py
API-inspected source blob: 6300de5a9af0b673e67ff7ae31258aa06c205080.

[2] Ghiasi and Fowlkes, Laplacian Pyramid Reconstruction and Refinement for Semantic Segmentation, ECCV 2016.
https://arxiv.org/abs/1605.02264

[3] Lopez-Paz and Ranzato, Gradient Episodic Memory for Continual Learning, NeurIPS 2017.
https://arxiv.org/abs/1706.08840

[4] Aubin-Frankowski, Korba and Leger, Mirror Descent with Relative Smoothness in Measure Spaces, 2022.
https://arxiv.org/abs/2206.08873

Project evidence:
https://github.com/QuocKhanhLuong/Self-Audit/blob/4d58f3aebe4f205b58b3184c8971b1d838109d82/reports/guidance_signal_execution_20261014/training_r2/RESULTS.json
