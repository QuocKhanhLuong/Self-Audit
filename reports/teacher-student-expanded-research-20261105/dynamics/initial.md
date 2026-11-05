# Independent initial hypotheses: dynamics and physical coordinates

Written before receiving or reading the coordinator evidence packet, prior council reports, or peer proposals. Source boundaries only: `/Users/alvinluong/Self-Audit` HEAD `9958d421ccae1da3b8527f74da5438dbb9f3f521`, ACTIVE_PIPELINE.json and docs/METHOD.md. Research hypotheses, not novelty or efficacy claims. Initial file is to remain unchanged.

## Shared protocol and information accounting

Primary deployment uses one original cardiac image x and its declared physical-coordinate metadata. Teacher encoder T(x) and compact student encoder S(x) both see that same image during training. Dense named-class supervision comes only from a declared TRAIN-patient budget, including all labels used for tuning/gates inside training. Unlabeled TRAIN cine can supply extra training information only in a separately declared temporal-privilege arm. No U/VAL/TEST mask is used for targets, label selection or training gates. The current repository method is semi-supervised; these prospective protocols are independent candidates rather than modifications falsely claiming existing guarantees. 96 Dice is an aspiration requiring a specified domain/class/patient aggregation, not a predicted result.

## Idea A: same-image physical-coordinate connection and material-slot decoder

**Object:** an image-conditioned connection/operator on physical-coordinate patches, not a target feature tensor. Let r_i be physical coordinates; student supplies patch vectors s_i and K semantic material slots h_a. Teacher supplies same-image pairwise conductances w^T_ij and local transport maps R^T_ij between learned patch frames. A joint representation is the graph operator

(L_T H)_i = sum_j w^T_ij [H_i - R^T_ij H_j].

Student features supply the forcing b_S(x), and the TRAIN decoder computes H* = argmin_H sum_i ||H_i-b_S,i||^2 + tau <H,L_T H>. Segmentation is read at native coordinates by a small coordinate-query network combining H*, local intensity and r. Teacher and student encoders therefore both causally participate in decoding the same image. A student operator head predicts w^S,R^S; separate supervised deployment-path optimization trains H*_S with L_S. Cross-branch decoding with shared slot definitions ties the two paths through label risk and response to known coordinate actions, not Euclidean feature matching. This is not residual correction: the implicit material representation is the entire decoder state.

**Symmetry:** for a declared passive physical-coordinate change g, require L(gx)=P_g L(x) P_g^-1, with vector/frame channels transformed by g. Distinguish passive coordinate changes from actively rotating the anatomy into different acquisition artifacts. Anisotropic resampling needs appropriate quadrature and cannot simply permute voxels. Avoid claiming diffeomorphism invariance: an arbitrary deformation can change anatomy and labels.

**Compact architecture:** one fixed-resolution depthwise patch encoder; sparse local graph plus K global material slots; 3-6 bounded iterative solves; coordinate-query rasterization. No down/up pyramid or encoder-decoder skip concatenation. PSD connection Laplacian uses R_ji=R_ij^T and compatible orthogonal transports, but graph anatomical edges need not imply literal tissue correspondence. Teacher training can use the same operator with a larger patch encoder.

**Assumption and prediction:** teacher geometry routes global class evidence more reliably than the compact encoder. Relative to equal-parameter local patch classification, gains should concentrate in thin myocardium and scanner spacing shifts; replacing physical coordinates by pixel indices should specifically erase spacing robustness. A teacher-randomized graph with degree/spectrum matched should fail if anatomy-specific correspondence is essential.

**Killer counterexample:** two touching structures share intensity and no trustworthy boundary evidence; a high-conductance incorrect edge spreads the wrong class globally. Connection frames have gauge freedom, so matching their raw coordinates is meaningless. If the graph only smooths, a tuned image edge filter can reproduce the gain and the teacher-transfer story fails.

## Idea B: temporal privilege transfers an observable generator distribution

**Object:** conditional distribution over coarse anatomical transport generators, not a deterministic inferred heartbeat. A cine teacher sees C=(x_1,...,x_T) in a separately declared protocol; its same-frame encoder T(x_t) and student S(x_t) anchor slots. Teacher infers material correspondence and a latent generator A_C with z_(t+dt) approximately exp(dt A_C)z_t, plus boundary uncertainty. The single-frame student learns q_S(A,z|x_t), an amortized posterior from which its segmentation decoder marginalizes p(y|x_t)=integral p(y|z,A,x_t)q_S(A,z|x_t)dA dz. A teacher-conditioned same-frame decoding branch learns useful slot semantics; the deployed student branch must be trained directly with the identical dense TRAIN labels.

Temporal targets train a likelihood for the operator's observable consequences (transport of image evidence and visible boundary evolution), and posterior calibration; matching an arbitrary latent A matrix is invalid under similarity transforms. Discretization should use physical time and avoid interpreting a stable recurrent matrix as a physiological law. Runtime is a small mixture of slot graphs or learned quadrature; continuous ODE solvers are not assumed cheap.

**Assumption and prediction:** motion patterns reveal which low-contrast boundary belongs to myocardium during training, improving an image-only anatomical prior on ambiguous images. Benefit should survive removal of phase labels and be localized to ambiguous boundaries, while patient-specific dynamics remain uncertain. Teacher quality itself should be evaluated in both equal-frame and privileged-cine settings so extra information is never credited to architecture alone.

**Identifiability counterexample:** two hearts have exactly the same x_t and segmentation but opposite velocity at that instant (contraction versus relaxation), or the same silhouette with distinct through-plane velocities. Any deterministic f(x_t) outputs the same generator for both and must be wrong for at least one. Even a perfect conditional mixture cannot recover an individual's dynamics without further information. If segmentation differs while images coincide, irreducible label ambiguity also remains; temporal knowledge cannot make a one-frame predictor an oracle.

**Killer test:** pair equal observed frames with different admissible trajectories in a controlled simulator. A valid student retains multimodal velocity uncertainty; confident branch selection falsifies the learned interpretation. On patients, compare frozen image-only deployment against a same-compute, same-label image-only teacher/student baseline and shuffled-within-patient temporal-order privilege.

## Idea C: equal-cine correspondence factorization instead of temporal consistency

**Object:** a material-space segmentation field a(u) and observation maps phi_t(u), not framewise consistent logits. Both teacher and student receive the same cine in a separate equal-input track. Decode y_t(r)=a(phi_t^-1(r)) with explicit visibility v_t(r) and an image-conditioned appearance branch; morphology changes correspond to deformation of named material, while visibility handles out-of-plane motion. A variational objective reconstructs observations plus TRAIN segmentation using a low-rank transport basis, with numerical cycle errors reported rather than treated as proof of correct correspondence.

Compact student: a fixed-resolution spatial patch front end, small temporal state-space bank, sparse transport basis, and native coordinate field decoder. Dense raster outputs are generated by querying the field, not a U-Net pyramid. Teacher and student same-frame encoders jointly seed material slots during training; the student-only cine bank supplies them at inference. This is a new task track, not evidence for single-image deployment.

**Prediction:** correspondence should improve segmentations during transient low contrast and reduce material identity swaps. Frame shuffling should hurt correspondence-sensitive metrics even if marginal per-frame Dice changes little. Long-window benefit must exceed duplicated-frame controls and equal-cine temporal attention at matched memory/latency.

**Killer counterexample:** 2D cine slices intersect different tissue over time; a single invertible map cannot explain appearing/disappearing structures. A perfectly cycle-consistent but wrong mapping can exchange symmetric regions. Visibility estimation may become an unconstrained escape hatch, so withheld-frame reconstruction and occlusion-budget diagnostics are mandatory.

## Independent preference before literature

A is the primary same-image, student-only research direction; B is the risky temporal-privilege hypothesis worth testing only with explicit uncertainty and observability; C is a useful separate deployment regime, not a substitute for the main goal. No method is claimed novel. Cheapest initial falsification uses synthetic physical-coordinate transformations and non-identifiable trajectories before patient training, then a small prelocked patient pilot if separately authorized. No computation has been run and no empirical gains are claimed.
