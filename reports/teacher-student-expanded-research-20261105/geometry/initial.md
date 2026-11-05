# Independent initial geometry proposals

Status: independent mechanisms before literature packet; PROPOSED / NOT RUN. Written UTC: 2026-11-05T12:12:54.629999+00:00.
Source boundary read: `/Users/alvinluong/Self-Audit` HEAD `9958d421ccae1da3b8527f74da5438dbb9f3f521`, `ACTIVE_PIPELINE.json`, `docs/METHOD.md`. No earlier council reports or forbidden directory read; no literature search, papers, peer proposals, or root evidence received. A lightweight memory lookup surfaced only older provenance/semantic-identifiability boundaries, not an architectural proposal; current source was used to verify those boundaries. These are prospective protocols, not modifications of current v2.

Common information contract: original single-frame MRI x enters both teacher E_T(x) and compact student E_S(x) during training. Named RV/MYO/LV/BG semantics come only from dense budgeted TRAIN labels L; audit labels are included in L. U provides images only. No U, VAL, TEST masks inform targets, selection or gates. A same-image 2D primary is distinguished from a separately declared native-volume experiment. At deployment only x enters the student. Training-time geometric objects must be predicted from image features at inference, never supplied as contours/masks. Ninety-six Dice is an aspiration whose class/patient/domain metric must be fixed, not evidence or a promise.

## A. Executable region programs with image-driven split/merge decisions

Object transferred: an equivalence-class distribution over sparse segmentation programs, not dense encoder features. Start from a fixed fine image grid G0 (not GT superpixels); a legal program pi consists of local SPLIT, MERGE, RELABEL and STOP operations. Regions may be disconnected across classes; no universal one-component or ring prior. Let R(pi) be its rasterization. Teacher features supply uncertain boundary/region evidence e_T; student features supply local texture and an image-adaptive action policy. A training-only planner searches

    pi* in argmin_pi [sum_v u_T(v,R(pi)_v) + sum_(i,j) b_T(i,j) 1[R_i != R_j] + lambda length(pi)]

within a fixed compute budget, with GT semantic energy replacing teacher unary evidence only on L. The student learns the value of actions using actual downstream energy improvement, Q_S(s,a|x) -> E(s)-min_completion E(s,a,...), alongside native supervised segmentation; equivalent action orders share the same state-value target. Thus the representation is a compressible constructive description and the decoder executes it. Teacher and student encoders of the same x jointly enter training planning through teacher structural evidence and student realizable action candidates, rather than independently matching tensors. Decoder inference uses E_S(x) -> region states -> action policy -> rasterizer, without teacher input.

Compact architecture: stride-4 depthwise image stem, sparse region tokens with moments and learned local descriptors, constant-width graph message passing, fixed action budget, direct native-grid leaf rendering. No contracting/expanding U-Net ladder. Refinement can stop early in homogeneous background but must account for worst-case native output cost.

Assumption: useful cardiac masks admit short region programs and teacher evidence provides ranking signal beyond its final logits. Prediction: at fixed peak memory and latency, compute concentrates on thin MYO, uncertain interfaces and open basal anatomy; region count tracks complexity rather than image dimensions. Cheapest decisive test: on existing frozen TRAIN images/labels compare optimal attainable region rasterization versus fixed equal-budget grids, then compare teacher-planned versus logits-only action rankings. Killer: highly trabeculated anatomy or a one-pixel MYO bridge needs more splits than the budget; compression irreversibly merges two semantic compartments. Do not hide this with post-hoc contour repair. Student learnability and search bias are large unknowns.

## B. Transfer image-conditioned transport operators, decode by absorption probabilities

Object transferred: a sparse stochastic transport operator and its response to source interventions, not an atlas or signed-distance field. Build image-grid or image-only adaptive region graph with symmetric positive conductances w_ij; learn class-specific absorption rates a_ic and escape/unknown rate a_i0. Its killed random walk solves

    (L_w + diag(sum_c a_c + a_0)) P_c = a_c,

with total probability including unknown. Teacher encoder predicts an ensemble of plausible operators; student image encoder predicts its own conductances/rates and a few latent right-hand sides. Training jointly evaluates teacher structural conductance proposals and student image evidence; labeled native CE/Dice train the resulting solve, while teacher transfer supervises finite perturbation-response experiments P(a+delta)-P(a), with perturbations fixed independently of label masks on U. Finite action response matters only if it changes the executable student operator, not if this collapses to matching one final probability map.

Student architecture: shallow depthwise encoder -> fixed-degree graph coefficients -> a fixed number of preconditioned local diffusion iterations -> native probabilities. No U-Net decoder, supplied foreground seeds, atlas registration, contour input, or temporal context. At inference all absorption rates are predicted from x. An explicit unknown route expresses insufficient evidence but reporting must include all-patient coverage and raw segmentation performance; abstention is not background.

Assumption: boundary uncertainty and connectivity can be encoded by a local operator without excessive diffusion. Prediction: teacher transfer improves long-range region assignment under localized intensity corruption more than ordinary feature KD at matched capacity; changing a tiny bridge conductance has measurable remote effects. Cheapest decisive test: compare impulse-response rank/conditioning and teacher versus student seed errors on a tiny frozen TRAIN subset, and compare with a same-parameter direct softmax head. Killer: in a homogeneous image with ambiguous class names, graph geometry cannot identify RV versus LV; wrong image-predicted absorption seeds spread confidently through the entire wrong compartment. A topology-changing pathological passage can be erased by smoothness. Operator nonidentifiability is serious: many conductances yield the same P.

## C. Open interface complexes with uncertainty over connectivity

Object transferred: a distribution over oriented inter-class interface fragments and their incidence, including endpoint/censoring states. Query k predicts curve gamma_k(t), left/right semantic labels (c_k-,c_k+), presence z_k, endpoints and uncertainty. No requirement that MYO is a closed ring in every slice. Teacher and student image encoders jointly propose and evaluate an interface complex C; differentiable rasterization maps it to class occupancy. On L, train against native masks; on U, teacher contributes uncertain incidence alternatives rather than one compelled topology. A mixture is retained:

    p(y|x) = sum_h p(h|x) Raster(C_h(x)),

where h indexes a bounded number of image-conditioned connectivity choices, including absent structures and open endpoints. Thin structures are represented as paired interfaces only when image evidence supports them; the incidence graph can change across slices or pathology. Student-only inference predicts curves and h from the original MRI, then rasterizes with explicit visibility/crop conventions.

Architecture: single-resolution patch stem + a small interface-query attention block + curve/control-point heads + native rasterizer. Boundary query count, not U-Net width, determines decoder complexity. Boundary supervision from L is derived from the same budgeted masks and does not count as free extra labels.

Assumption: interfaces are economical and trainable despite topology changes; semantic incidence is identifiable from L. Prediction: better thin-MYO boundary precision per parameter, and calibrated increased connectivity uncertainty at basal/apical views, compared with a closed-shape prior. Cheapest decisive test: oracle interface fitting to labeled TRAIN masks including absent-class/open-boundary strata; if even oracle representations cannot reproduce masks at the budget, reject before teacher training. Killer: multiple tiny islands, papillary ambiguity, through-plane partial volume or inconsistent annotation conventions require too many interfaces or make orientation/rasterization ambiguous. Closing curves by fiat would invalidate the proposal.

## Pre-evidence ranking

A is the preferred risky mechanism because it transfers a decision process and changes decoder computation; B is a useful contrasting operator hypothesis but may be ordinary graph diffusion with an extra transfer term; C is the most direct geometric compression but has severe representation and rasterization risks. None is claimed novel. First settle representational adequacy and semantics on L, then teacher benefit, then patient generalization. Prefer explicit negative results over tuning an unseen-test story.
