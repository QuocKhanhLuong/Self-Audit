# Teacher investigator: change the learning problem

**Verdict.** Keep two risky, distinct hypotheses: transfer a distribution of anatomies under an explicit annotation model, or transfer image-conditional acquisition environments for robust learning. Neither is established novel or effective. The first directly supplies a joint teacher/student representation to a student decoder; the second supplies that representation indirectly through environment construction and therefore only partly meets the representation requirement. Do not market either as a new name for feature matching.

## Evidence and teacher selection

Source checkout was read-only and HEAD verified as `9958d421ccae1da3b8527f74da5438dbb9f3f521`. `initial.md` was preserved before reading the supplied packet or any council report. Current v2 boundaries remain intact; these are prospective protocols.

The primary ABR-UNet3D XML Table 3 reports BG/RV/MYO/LV Dice of 99.89/94.85/92.11/97.26. Their four-class average is 96.0275%; foreground average is **94.74%**, while Table 10 prints 94.73%. Thus 96.03 is background-inclusive, not 96 foreground. ABR evaluates ACDC, not M&Ms. CASNet XML reports 96.3% ACDC and 95.7% **M&Ms-2**; class aggregation is unresolved. Its Table 1 nominal 100/50 and 250/125 splits coexist with an experimental description of five-fold 80/20 evaluation. These are protocol ambiguities, not evidence of misconduct. Neither source verifies the requested 96 foreground on both ACDC and M&Ms.

A credible teacher study would first nominate nnU-Net residual-encoder presets and MedNeXt as reproducible volumetric baselines identified by the packet, then verify exact public recipes/checkpoints before execution. Their primary recipes were not supplied/read here, so reproducibility is a candidate-selection rationale, not a completed audit. ABR is a boundary-context candidate; CASNet a 2D curvature/context candidate with unresolved protocol. MetaCLIP-CMR is a distinct acquisition-aware pretraining candidate: its primary Table 3 reports case-level Dice 90.21% ACDC and 83.68% M&Ms with full downstream labels. Its metadata pretraining uses a ResNet-50 and DistilBERT, not demonstrated 96-level segmentation. CMR-DINO/CineMA are mentioned comparators in that paper, not independently verified here; no historical CineMA-bank trainer is revived.

Teacher size or global Dice cannot establish useful posterior uncertainty or intervention competence. Rank candidates by the proposed interface on TRAIN folds. Separate class/domain metrics, boundary error, probability scoring, and patient aggregation; report background separately. Additional pretraining images, metadata, annotations and context are information factors even when no dense labels are added. Native 3D support may help within-volume continuity but cannot recover unobserved anatomy; multiple phases/LAX views belong to a separate protocol.

## Direction A: anatomy posterior with an annotation observation model

Let X denote one native image volume, Z spatial anatomical coefficients, A a predeclared annotation convention, and Y its observed mask. Define a teacher/student joint posterior

    q_phi(Z,A | H_T(X), H_S(X), Y_L)

and a student-only prior p_theta(Z,A|H_S(X)). The decoder produces a categorical anatomical field through a fixed-capacity local basis, followed by an annotation observation model p_psi(Y|Z,A). For labeled TRAIN patients:

    L = E_q[-log p_psi(Y|Z,A)] + beta KL(q || p_theta).

All Y_L are charged to the dense label budget. On U, Y is unavailable: the teacher supplies candidate Z states, not a fabricated annotation likelihood observation. A KL transfer term on U is an explicit variational-distillation component, not the novelty claim. Omit that term initially if the teacher posterior cannot pass the competence test. The substantive hypothesis is that preserving alternative anatomies through a declared observation model yields better decisions than compressing them to one mask.

**Information graph.** TRAIN X → frozen teacher encoder and student encoder → joint posterior → sampled anatomical Z → the student decoder → labeled likelihood. In parallel, X → student prior → the same decoder supplies a deployable supervised path. TRAIN masks affect the training posterior/likelihood only. Deployment is X → student encoder → prior → fixed-count latent samples → same decoder → averaged categorical probability → fixed argmax. No teacher, mask, neighboring phase, optimization loop, or external atlas is an inference input. Sample count and decoding cost are counted; one posterior-mean decode is a distinct ablation, not equivalent integration.

**Compact architecture.** Use a shallow local tokenizer on a native cell graph, fixed-width recurrent message blocks and a small global anatomical token set; no encoder pyramid or expanding skip decoder. Tokens define class-specific coefficients in overlapping compact-support spatial basis functions. A coordinate-conditioned categorical decoder evaluates these local fields at native physical coordinates. Class absence and disconnected components are possible because there is no fixed-connectivity heart template. Dense local processing/rasterization remains O(N); sparse basis lookup avoids an all-voxel/all-token product, while recurrent iterations and sample count multiply cost. Low parameter count does not imply speed. Basis support and capacity may still destroy one-voxel MYO or pathological components; oracle representation error must be measured before student learning.

Without repeated independent annotations, A and Z are generally nonidentifiable. The revised primary experiment therefore fixes A to the dataset convention and uses posterior alternatives as epistemic hypotheses. Varying A is sensitivity analysis only. A learned interobserver model requires new repeated labels and a separately declared budget. Better intensity reconstruction, broad uncertainty, or teacher agreement cannot validate anatomical truth.

**Prediction and falsifier.** At matched architecture and teacher compute, retaining alternative Z states should improve held-out TRAIN-fold boundary log score/coverage at a fixed uncertainty volume, especially near bases/apices, even if mean Dice changes little. Compare an identical single-state model, ordinary output KD, and teacher-free variational training. Reject if improvements vanish at matched uncertainty volume or if pathology/surface error worsens. This explicitly allows the hypothesis to fail despite a strong teacher Dice score.

**Closest supplied art.** NISF++ §§3.1–3.3 already models coordinate-conditioned cardiac intensity/segmentation with a patient latent and acquisition corrections; it uses SAX/LAX cine and 2,500 inference optimization steps, reporting about two minutes. It notes segmentation can plateau while reconstruction improves. CineMesh4D §2 already uses image encoders, mesh VAE and differentiable contour supervision; fixed mesh connectivity and atlas-derived references constrain its scope. Hence implicit anatomy, VAEs, and geometry decoding are not our novelty. A distinguishing prediction is benefit from calibrated alternative anatomical explanations under a fixed observation protocol, with one student inference graph. Probabilistic segmentation/privileged variational learning were not comprehensively searched in this packet: novelty remains unresolved.

## Direction B: teacher-designed acquisition environments

The transferred object is a conditional admissible set E_T(X) of acquisition interventions. Restrict g_eta initially to nonspatial, bounded contrast and smooth bias-field changes with known parameters. Noise and resolution degradation require separate severity studies because they remove information. A joint representation R(X)=[H_T(X),H_S(X),physical support] feeds an environment policy; a powerful acquisition-aware teacher proposes where the student relies on brittle cues.

    min_theta E_(X,Y in L) max_eta in E_T(X) ell(S_theta(g_eta X),Y)
              + lambda E_(X in U) dispersion_eta S_theta(g_eta X).

The teacher provides the environment support/ranking rather than target masks. On U, consistency is merely regularization. Do not let high confidence enlarge E_T or certify anatomy preservation. Transformation construction supplies the narrow invariance assumption; cross-fitted label evidence checks its plausibility within observed support.

**Information graph.** TRAIN X → both encoders → environment policy → bounded transformed X → student encoder/decoder → supervised or U-consistency objective. For an orbit-representation arm, the decoder consumes a weighted aggregate of aligned student cell tokens from these generated environments. This combines teacher and student information in training representation construction, but is not direct dual-encoder fusion at deployment. Freeze a small TRAIN-selected quadrature of transformations for student-only inference; use the same quadrature during training of the deployed path. Its K encoder passes count. A single-pass learned marginalizer is an unverified extension, not a free cost reduction. The compact cell-graph decoder above can serve as an architecture control; this direction does not independently solve student architecture design.

**Prediction and killer test.** Benefits should concentrate in held-out acquisition strata and survive eliminating teacher segmentation targets. Compare teacher-selected and teacher-free transformations with identical severity distributions, count and compute, plus output KD and ordinary robust training. If severity matching removes the gain, reject teacher-specific environment knowledge. A strong teacher may share the student's shortcut or protect its own mistakes. Worse, intensity differences may encode pathology; invariance would erase useful evidence. Holding the original teacher masks fixed while shuffling its environment rankings is a cheap test: unchanged downstream stress risk falsifies the proposed interface.

MetaCLIP-CMR supplies evidence that acquisition metadata can organize CMR representations, not that they identify causal nuisance. OctreeNCA §§3.1–3.3 establishes existing recurrent local segmentation rules and the importance of multiscale communication/custom kernels; our cell graph cannot claim novelty from recurrence. Robust optimization, adversarial augmentation and domain randomization are anticipated close prior art not adequately covered by the packet. The bounded search cannot settle this direction's novelty.

## Root challenge: competence under both label budgets

Accepted: an accurate endpoint teacher may be an invalid posterior/planner. Use patient-disjoint nested folds entirely inside declared TRAIN labels. Outer held-out folds evaluate interface usefulness; inner folds select thresholds, latent capacity or transformation policy without that outer patient's mask. Fit candidate teachers without the outer fold. Measure posterior log score and fixed-volume boundary coverage for A; label-preserving stress risk and ranking regret for B. Keep unsupported class/domain strata explicitly unknown. Compare teacher-free interfaces with identical labels/compute, and count calibration/audit patients in the budget.

For **same-budget**, every teacher, student, posterior, and competence model uses the same union L; image-only U never supplies masks. Small L may make nested competence estimates unusable: report insufficient evidence, rather than calling agreement calibration. For **full-TRAIN**, replace L by all declared TRAIN patients and charge the full union, even if the student directly receives fewer masks. Preserve cross-fitted teachers for U proposals so competence evidence refers to the evaluated training regime. A refit on all patients is a new teacher requiring explicit uncertainty; the fold estimate does not certify its interface. Neither route accesses VAL/TEST masks for targets, selection, or gates.

Root's challenge is answered without promising a generalization certificate. The cheapest rejection precedes student training: a high-Dice teacher whose alternative states have poor proper score, or whose environment ranking has no relation to fold-held-out harm, is rejected for that interface. Good final masks remain useful for a conventional KD control.

## Changes, peer discussion, and status

Preserved initial ideas, but narrowed annotation A after the identifiability objection, made the deployed stochastic/transform computation explicit, and rejected geometry/recurrence as stand-alone novelty after reading primary texts. The architecture peer challenge is recorded in `peer_review.md`: final masks do not identify an energy basin or optimal construction path; compare action regret at fixed masks/capacity/compute. Its new partial-order action-comparison idea remains separate from my two directions. The architecture investigator accepted the challenge and constructed equal-sparsity operators A1/A2 with u_k=-log(q*)-A_k q*, sharing the same endpoint q* but different finite-horizon dynamics. I accept the fixed-endpoint intervention test; I do not accept endpoint agreement or fold competence as unseen-patient certification.

Goal coverage: A addresses stronger teachers, same-image joint representation, a distinct probabilistic learning problem and a non-U-Net student, all hypothetically. B addresses teacher competence and robustness transfer; it only partly addresses the joint-decoding representation and relies on a separately tested compact architecture. No claim of satisfying all four empirically is made.

**Evidence:** primary text/table inspection VERIFIED; literature results REPORTED; mechanisms and predictions DERIVED; training, GPU/latency tests, efficacy and novelty verification NOT_RUN. No weights/data were downloaded, no source edited, no experiment node changed, and no commit/push performed. Repository tests were not run because this is an authorized research-only dossier outside the repository, not a software delivery.

### Primary source locations

- `/Users/alvinluong/.local/share/openresearch/files/self-audit/teacher-student-redesign-20261105/sources/ABR.xml`, Tables 3/10; https://doi.org/10.3390/diagnostics16111598
- Same source directory `CASNet.xml`, Table 1 and experimental/metric sections; https://doi.org/10.3389/fmed.2025.1688872
- Same source directory `2606.28991.txt`, §§3.1–3.4/Table 3; https://www.alphaxiv.org/pdf/2606.28991
- `/private/tmp/selfaudit-expanded-20261105/papers/2608.00752.txt`, §§3.1–3.3/results/limitations; https://www.alphaxiv.org/pdf/2608.00752
- Same papers directory `2605.13994.txt`, §§2–3; https://www.alphaxiv.org/pdf/2605.13994
- Same papers directory `2508.06993.txt`, §§3.1–3.3; https://www.alphaxiv.org/pdf/2508.06993
