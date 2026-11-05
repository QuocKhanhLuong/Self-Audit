# Independent teacher hypotheses — frozen initial ideation

Written 2026-11-05 before reading the coordinator evidence packet or any council report. Source boundary only: `/Users/alvinluong/Self-Audit` HEAD `9958d421ccae1da3b8527f74da5438dbb9f3f521`, `ACTIVE_PIPELINE.json`, and `docs/METHOD.md`. No literature search or patient experiment performed. These are hypotheses, not novelty or efficacy claims.

## 1. Teacher as an inference engine for anatomy and annotation, not a label generator

Question: is the dominant limit imperfect image understanding, or collapsing physically plausible anatomy and annotation conventions into one hard mask? A stronger teacher should teach which latent anatomical worlds explain the image and which boundary decisions remain ambiguous. A single entropy map does not distinguish annotation uncertainty from domain unfamiliarity.

Let X be one native 3D image volume, Z an explicit anatomical representation, A an annotation convention, and Y the observed dense annotation. Construct a generative observation model p_psi(Y|Z,A) and a conditional student prior p_theta(Z,A|X). The teacher encoder and student encoder of this same X jointly parameterize a privileged training posterior q_phi(Z,A|X,Y_L,H_T(X),H_S(X)); Y_L exists only for declared labeled TRAIN patients. On U the posterior may use X and the frozen teacher but never Y. Training minimizes

    L_L = E_q[-log p_psi(Y|Z,A)] + beta KL(q || p_theta(Z,A|X)).

Teacher knowledge is the distribution over interpretable anatomy/convention states and their explanatory alternatives. The shared student decoder consumes sampled Z plus native image-local features; it never consumes a teacher feature map directly. At inference sample or deterministically integrate p_theta(Z,A|X), apply the same decoder, and return the specified decision rule. The teacher/student representation is a joint latent posterior, and the deployment route is an amortized generative student. Ordinary variational distillation is a serious anticipated prior; novelty would require evidence that the explicit annotation channel identifies a new failure mode and improves decisions under ambiguous boundaries.

Compact architecture: a shallow patch tokenizer creates a fixed-width native-resolution cell graph, six recurrent graph/state-space blocks exchange evidence, and 16–64 anatomical tokens specify local signed-distance basis coefficients. A coordinate-conditioned local surface decoder rasterizes class probabilities. There is no encoder pyramid, expanding convolution decoder, or U-Net skip ladder. Native coordinates and spacing remain explicit. A surface representation must allow disconnected structures and class absence; fixed healthy-heart topology is not assumed.

Assumptions: anatomy/convention can be partially identified from declared data; teacher alternatives are calibrated enough to be useful; the latent capacity covers pathology. Full-TRAIN route: train the teacher on all TRAIN labels and declare their union as the new budget. Same-budget route: teacher and student use exactly L; use multiple independently initialized teachers or self-supervised image pretraining with disclosed provenance, without treating agreement as truth. Without repeat annotations, A is not identifiable: restrict A to predeclared sensitivity analysis, do not claim learned interobserver uncertainty.

Prediction: compared with identical compact decoder and point-label KD, distributional anatomy improves boundary risk/coverage and native-volume coherence at similar mean Dice; the largest gains occur on ambiguous basal/apical slices. It need not increase Dice if the benchmark rewards one convention. Killer counterexample: a broad posterior merely conceals systematic teacher errors, or a normal-shape latent prior removes pathology; single-mask likelihood and held-out surface error worsen even as entropy appears sensible. Cheapest decisive test: on existing allowed TRAIN-only folds, compare point posterior versus multi-world posterior with identical teacher compute, then test whether alternatives cover actual held-out boundary errors at fixed volume/coverage. Genuine annotation claims require new independently repeated annotations and a new declared label budget.

## 2. Teacher as a designer of admissible nuisance environments

Question: can a powerful teacher expose the acquisition shortcuts a compact model should stop using, instead of supplying more confident labels? Teach a set of acquisition variations under which anatomy should be stable. This changes the student's training problem from imitation to robust identification.

Let g_eta(X) change only declared acquisition nuisance (smooth bias field, contrast mapping, bounded noise, resolution simulation with fixed geometry). Teacher and student encode the identical original image: H_T(X), H_S(X). Their joint representation R=[local student tokens, teacher context tokens, uncertainty/competence indicators] parameterizes a conditional environment generator a_omega(eta|R). The student decoder itself is teacher-free and is trained on orbit-aggregated student representations

    Z_S(X) = sum_k w_k E_S(g_eta_k(X)),
    eta_k ~ a_omega(.|H_T(X),H_S(X)).

The mathematical object transferred is an image-conditional admissible intervention set E_T(X), not teacher features or segmentation logits. Solve

    min_theta E_(X,Y in L) max_eta in E_T(X) ell(D_theta(E_theta(g_eta X)),Y)
         + lambda E_(X in U) dispersion_eta[p_theta(.|g_eta X)].

The supervised term anchors semantics; the U term cannot establish correctness. Teacher predictions can reject transformations that clearly break its own competence, but such rejection is only a heuristic, not an anatomy certificate. Geometrical invariance comes from the transformation construction, and intensity assumptions must be stated. At deployment replace the privileged generator with a fixed small quadrature of safe transforms selected using TRAIN-only folds, or use a student-only nuisance estimator trained on known synthetic eta; its cost counts. A primary single-pass variant integrates the nuisance variable inside a small recurrent cell graph; no teacher input survives deployment. The empirical question is whether the teacher-selected environments outperform equally costly teacher-free physics transformations.

Use the same non-pyramidal cell-graph architecture as an architecture-controlled first experiment, but this hypothesis is independent of the anatomy posterior. Full-TRAIN teacher may estimate acquisition-conditioned competence with all TRAIN labels under a new budget; same-budget teacher competence must be cross-fitted within L. Domain IDs or scanner metadata are privileged only if declared; the primary model uses X only. Temporal neighbors/multiviews are excluded from this primary protocol.

Prediction: gains concentrate in held-out scanner/style conditions, with little in-domain benefit; robustness survives replacing teacher masks with no masks, because the transferable object is environment support. Killer counterexample: teacher and student share the same texture shortcut, so the teacher selects harmless cosmetic changes and misses the actual domain shift; alternatively an assumed nuisance (intensity/partial volume) contains necessary pathological evidence, causing robust training to erase it. Cheapest decisive test: freeze one teacher and use disjoint TRAIN folds to compare random versus teacher-selected transforms with identical sampled severity/compute; assess anatomy-preserving synthetic stress first and held-out acquisition strata second, without selecting on TEST. If severity matching removes the advantage, reject the teacher-specific mechanism.

## Shared scientific boundaries

96 Dice is an aspiration with no specified aggregation or confirmed feasibility. Report foreground class-first patient Dice and each class/domain, with background-inclusive metrics separately, plus calibration and surface metrics where justified. ACDC and M&Ms can differ in label conventions, acquisition support, and pathology composition; neither high teacher confidence nor 3D smoothness certifies semantic truth. Teacher families and closest prior art remain pending the coordinator packet. Training/measurements/novelty verification: NOT_RUN. No active pipeline mutation, code changes, training, downloads, commits, or push.
