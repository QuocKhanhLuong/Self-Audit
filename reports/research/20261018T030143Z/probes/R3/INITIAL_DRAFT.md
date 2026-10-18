# R3 — Native appearance evidence inside coarse teacher cells

Initial independent draft, 2026-10-18 03:12 UTC. All methods below are **PROPOSED_NOT_APPROVED**. Author dispatch requested `gpt-6-astra`, reasoning `ultra`; resolved backend identity is not exposed. No other dossier or interrupted draft was read before this draft. Preserve this initial draft verbatim; later review responses are append-only. AI-assisted primary-source search, derivation and manufactured probes are disclosed; this is not a systematic review or patient study.

Snapshot: `/Users/alvinluong/Self-Audit-research-20261018`, HEAD `f00e4faa52204f8e6863f76a493684b68ed3a132`, verified locally. Read STATE.md, AGENTS.md, ACTIVE_PIPELINE.json, CURRENT.md, RUNBOOK.md, ASTRA_NEXT.md and RTK.md. CodeGraph is absent per shared contract. Semble located current model/layers, then those exact files were read. No production edit, dependency install, pretrained/data download, patient training, target-label read, commit or push occurred. A quick memory pass informed historical boundaries only; current source was independently inspected.

## Decision and measurement model

Investigate ONE evidence mechanism: **frozen uncertain-anchor native intensity likelihood**, initially for the conditional blood-versus-myocardium interface. This is a tightly scoped candidate and a necessary cheap baseline, not a complete named cardiac method. Its promising part is that intensity can vary inside a teacher observation cell and can disagree with a wrongly placed teacher boundary. Its major defect is that the mechanism cannot tell LV blood from RV blood, cannot distinguish a nonheart ring with matching appearance, and cannot identify missing anatomy from low cost alone.

What is actually observed: native center-slice image samples `x_i` after the documented image-only normalization, acquisition-valid pixels, and frozen historical Q/reliability/permissions from the TRAIN bank. The bank probabilities supply uncertain names and support; they are not new measurements of tissue. Native `x_i` contains additional spatial samples relative to a coarse bank observation, although both originate from the same cine and are statistically dependent. Do not count them as independent witnesses or multiply their probabilities as independent sensors. No target labels, semantic pretrained head, image-level class labels or external scribbles enter this candidate.

For an explicit conditional positive model in one small supported region, let hidden `y_i = 1` denote blood and `y_i = 0` myocardium, and let

`x_i = mu_M + Delta y_i + e_i`, `Delta = mu_B - mu_M > 0`, `|e_i| <= sigma`.

`e_i` absorbs within-region acquisition noise, residual bias and partial-volume/model error. This is an assumption to test, not a verified MRI law. It is inappropriate when blood is not brighter in the relevant acquisition, tissue has heterogeneous appearance, papillary muscle creates an alternate semantic convention, pathology alters contrast, or local bias exceeds the contrast. Neither `Delta` nor the worst-case error `sigma` is certified by Q confidence. They are latent, label-dependent quantities.

Estimate frozen means `hat_mu_B, hat_mu_M` using positive-mass uncertain anchor weights, with complete source/selection lineage. The first admissible fitting population is one TRAIN cine/region with source permission; no target cohort fitting. Anchor selection and measurement normalization are frozen before native optimization. No iterative refitting from student labels in the first mechanism. A lack of either class's anchor mass is MISSING_EVIDENCE; never substitute zero, a global fabricated foreground, or UNKNOWN-as-background. Anchors named LV/RV may be pooled as blood only with their lineage and permissions retained. That pooling does not establish that a slice contains either ventricle.

Define costs `c_B(i) = (x_i - hat_mu_B)^2`, `c_M(i) = (x_i - hat_mu_M)^2`, `a_i = c_B(i) - c_M(i)`. This is the equal-variance Gaussian negative-log-likelihood contrast, up to a positive scale and label-independent terms. Operationally it is a squared-residual appearance score, not a calibrated posterior. All costs are stop-gradient. Anchor purity plus representativeness, not a sharp-looking cost map, links this measurement to anatomy.

Let the current native logits initially be lifted coarse logits `l0 = U z`. The documented architectural `D` averages disjoint **4 by 4 native pixels**; `U` repeats values; `H = I - U D`. These are student logit cells, not necessarily the teacher bank observation grid. Native measurements must be transported to the actual input coordinates with validity; the exact teacher observation operator requires the separate bank geometry contract. Tests below use these architectural cells explicitly and cannot certify teacher-grid observability.

For a controlled blood–MYO edit, use the class order BG,RV,MYO,LV and direction `v = (0, 1, -1, 1)`. Consider native candidate logits

`l_i(s) = l0_i + v s_i`, `H s = s`, `|s_i| < B`.

The conditional blood log odds are `t_i(s) = log(exp(l_RV) + exp(l_LV)) - l_MYO = t0_cell + 2 s_i`; `r_i = sigmoid(t_i)` is blood given blood-or-MYO. RV:LV odds stay fixed on this output path. **BG probability does not stay fixed**, because its softmax denominator changes. This collateral effect must be measured; this mechanism has no signed evidence for heart versus BG. A full four-class claim would require an additional, justified localization measurement and is outside R3's surviving hypothesis.

On complete, valid architectural cells with fixed nonnegative cell-constant gate `g_j`, use only the appearance energy

`E_X(s) = sum_j sum_{i in cell j} g_j [r_i c_B(i) + (1 - r_i) c_M(i)]`.

No perimeter, topology, temporal consistency, reconstruction decoder or learned energy is added. The output-space candidate is one projected gradient step with a checked step size/bound, then detached for any later transfer experiment. Clipping pixels independently after projection generally destroys `H s = s`; if the proposed step violates the bound, reduce the scalar step or abstain. Border cells with padding/partial validity are excluded from this first derivation, rather than claiming full-cell algebra on invalid pixels. A richer valid-measure projection would be a separate policy.

Inference contract if eventually approved for student training: fixed student, image-only single-image/slice-context inference; appearance fitting and the frozen TRAIN bank are training-time proposal construction only. Per-cine proposals at inference would instead be an explicitly different adaptation regime, not hidden inside this claim.

## Evidence cards

### R3-C1 — Actual native gradient and the information it retains

- **Statement/status:** DERIVED_RESULT. At zero detail, native intensity contrast can generate a finite nonzero within-cell output gradient; equal-variance fitting loses the anchor midpoint under H.
- **Scope/assumptions:** Full valid 4 by 4 cells, cell-constant `g_j > 0`, finite `t0_j`, fixed costs and means, positive estimated gap `hat_Delta = hat_mu_B - hat_mu_M`. Euclidean inner product, so H is an orthogonal projector. Arithmetic is exact unless numerical error is explicitly discussed.
- **Formal claim:** With `r0_j = sigmoid(t0_j)`,

  `H grad_s E_X(0) |_j = -4 g_j r0_j (1 - r0_j) hat_Delta H x |_j`.

  Therefore, for each such cell, the projected gradient is nonzero iff `hat_Delta != 0` and `H x != 0`. For arbitrary four-class logits, `grad_l E_X = g r(1-r) a (0, pi_RV, -1, pi_LV)` at initialization, where `pi` are the conditional blood-class odds. H is nonzero under the same conditions because these coefficients are cell-constant. A nonzero cost vector alone is insufficient: class-independent shifts and cell-constant costs disappear.
- **Proof:** Differentiate the sigmoid and use `a_i = -2 hat_Delta [x_i - (hat_mu_B + hat_mu_M)/2]`. H kills the constant midpoint and commutes with cell-constant multipliers. The controlled edit has `dt/ds = 2`. This is elementary calculus/projection algebra, not a novel theorem.
- **Counterexample:** Constant native intensities or coincident estimated means give zero. A highly varying field with no anatomical relation gives a strong gradient. Arbitrary pixel-varying reliability/gates can manufacture a gradient even from constant appearance; they must not be described as native image information.
- **Closest prior:** Kim–Ye, Section III-A, Eqs. (12)–(13), softmax region fitting; Chan–Vese, Section II, piecewise-constant region model. The H cancellation is a consequence of this repository's constraint, not prior-art-free segmentation machinery.
- **Evidence:** Current `layer_upgrade/layers.py:8–34`, `model.py:77–113`; manufactured probe and RESULTS.json below.
- **Implication/limit:** The first native signal is effectively centered intensity contrast scaled by anchor mean ordering. Fitting an elaborate midpoint estimator cannot improve this initial H direction under these assumptions. The scalar correction `s = k H x` is a mandatory simpler replacement.
- **Falsifiable prediction:** Shifting both estimated means equally while keeping their difference fixed changes raw costs but leaves the initial H gradient unchanged; replicated intensity eliminates it.

### R3-C2 — Conditional correction with noisy anchors and imperfect measurements

- **Statement/status:** DERIVED_RESULT, conditional on latent assumptions. A nonzero noisy-anchor appearance signal can correct coarse teacher errors rather than merely sharpen them.
- **Scope:** Binary blood–MYO truth inside a correctly localized mixed cell; controlled output edit only. No guarantee for BG or RV/LV naming, network optimization, all-native cells or generalization.
- **Assumptions/formal claim:** Let `rho_j = mean_{cell j} y_i` satisfy `0 < rho_j < 1`; suppose the model above holds and both estimated means obey `|hat_mu_c - mu_c| <= delta`. If

  `Delta > 2 delta` and `m_j = Delta min(rho_j, 1 - rho_j) - 2 sigma > 0`,

  then `hat_Delta > 0`, `(2 y_i - 1) (H x)_i >= m_j`, and the candidate

  `s+ = -eta H grad_s E_X(0) = 4 eta g_j r0_j (1-r0_j) hat_Delta H x`

  strictly increases every true signed conditional logit margin. Every initially correct binary pixel remains correct. Every initially incorrect pixel whose wrong-margin magnitude is less than

  `8 eta g_j r0_j (1-r0_j) hat_Delta m_j`

  is corrected, provided the step remains feasible `max_i |s+_i| < B`. Thus it corrects all such mistakes without adding binary mistakes on that cell. If those conditions cover all edited pixels, ordinary binary Dice for a nonempty class cannot decrease and strictly improves when a relevant error is removed. This last Dice conclusion is conditional on the truth margin, not on energy descent.
- **Proof:** Subtract the cell mean: `H x = Delta(y - rho_j) + H e`, with `|H e| <= 2 sigma`. For blood the deterministic signal is `Delta(1-rho_j)`; for myocardium it is `-Delta rho_j`. The gap bound follows by triangle inequality. C1 gives the step and `t+ = t0 + 2s+`. The threshold follows by comparing the positive signed change with the initial negative margin. Fixing binary false positives/false negatives without creating errors improves `2TP/(2TP+FP+FN)` whenever its denominator is nonzero.
- **Noisy anchors:** For x bounded in [0,1], a deterministic weighted anchor mean with contaminating weight fraction `epsilon_c` and clean-anchor conditional mean displaced from `mu_c` by at most `b_c` obeys `delta_c <= epsilon_c + b_c`. Finite-sample discrepancy may be included in `b_c`; a concentration theorem needs a genuine sampling/dependence model. Q confidence does not upper-bound either term. Correct anchor labels can still give large `b_c` through intensity-based selection. All these truth-dependent constants are **LATENT**, not deployment certificates.
- **Counterexample:** A pure cell has `rho=0` or `1`, so H removes the true class mean and retains only nuisance. An arbitrarily thin minority can violate the bound even at moderate contrast. Wrong mean ordering reverses every signed update. Misregistered pixels, if introduced later, enter `e`; near an edge an arbitrarily small coordinate error can violate the bounded-residual assumption.
- **Closest prior:** Classical two-region fitting and margin robustness; no novelty claim. Unlike ordinary thresholding, the fixed-coarse restriction introduces the prevalence-dependent `min(rho,1-rho)` requirement.
- **Evidence/implication:** Manufactured 16-pixel mixed cell with true means .75/.25 and noisy estimates .72/.28 fixes 8/16 initial errors to 0/16 using a bounded output step. This illustrates the theorem's possibility, not its cardiac prevalence.
- **Falsifiable prediction:** At fixed contrast/noise, minority occupancy approaching zero destroys the robust correction margin. Such cells should fail first, even when a direct unprojected intensity classifier remains accurate.

### R3-C3 — Numerical descent, feasible edits and teacher-error limits

- **Statement/status:** DERIVED_RESULT. A checked output step decreases the frozen appearance energy but says nothing unconditional about Dice.
- **Scope/assumptions:** Fixed costs/gates, Euclidean coordinates s, no active bound. `|sigmoid''| <= 1/4` is a conservative global bound. Let `L = ||g a||_infinity`, which is operationally computable; if `L=0`, gradient is zero.
- **Formal claim/proof:** Since `d²E/ds_i² = 4 g_i a_i sigmoid''(t0+2s_i)`, the gradient is L-Lipschitz. For `0 < eta < 2/L`, `s+ = s - eta H grad E(s)` satisfies

  `E(s+) <= E(s) - eta (1 - eta L/2) ||H grad E(s)||_2²`.

  This is the classical descent lemma plus H symmetry/idempotence. Use measured post-step energy in floating point and a finite difference/projection tolerance; exact descent can be smaller than rounding. No guarantee transfers automatically to AdamW, momentum, clipping or a neural parameter step.
- **Attainable edit:** `H s=s, |s|<B` is representable by the current bounded detail operator at the output level: set pre-tanh detail channel to `atanh(v_c s/B)`. This is an existence statement for raw detail tensors, not proof that a shared 1 by 1 head can fit them. The controlled contrast can move `t` by at most `2B`; sufficiently strong wrong teacher odds cannot be reversed. The larger current unrestricted H-tanh image has other bounds but is not the controlled v-path proved here.
- **Counterexample:** Swapped anchors lower energy from 1.7744 to 1.40458 while binary mistakes rise from 8/16 to 16/16. Correct means on a pure blood cell with nuisance .65/.85 lower energy from 2.11001 to 2.03087 and introduce 8/16 mistakes from zero.
- **Closest prior:** Standard smooth projected descent; GrabCut Section 3.2 explicitly separates iterative energy minimization from the supervision that names the object. No novel numerical guarantee.
- **Evidence/implication:** Output feasibility and energy checks are software conditions. They cannot authorize native anatomy claims or substitute for C2's latent assumptions.
- **Falsifiable prediction:** Descending cost and anatomical disagreement can coexist under wrong anchors/pure-cell nuisance, as the probe demonstrates.

### R3-C4 — Output information does not guarantee a network update

- **Statement/status:** CODE_FACT plus DERIVED_RESULT. At initialization, a native objective can update the zero detail head, while first-step upstream detail features receive zero gradient through that head.
- **Scope/assumptions:** Current `raw_detail_i = W h_i + b`, W=b=0, detail scale B, same H operator, frozen image cost. For clarity ignore independent auxiliary losses in this path. Use full padded tensors when applying H.
- **Formal claim:** Writing `G_i = (H grad_l E)_i`, `grad_W E = B sum_i G_i h_i^T`, `grad_b E = B sum_i G_i = 0`, and gradient into h through this branch is zero when W=0. Hence `H grad_l E != 0` is necessary but insufficient for `grad_W E != 0`; feature-gradient correlation is required. Constant h on each cell is an exact zero-head-update counterexample.
- **Proof:** Chain rule at tanh(0), H self-adjoint, and zero-sum cells. A spatially varying gate does not remove the zero-sum property of H. The feature branch has multiplier W^T.
- **Counterexample/positive case:** With manufactured h=1, output gradient is nonzero but scalar weight gradient is zero; h=x has gradient -0.433196. In both cases upstream feature gradient is exactly zero on the first step. A tiny current-model fixture produces nonzero weight gradient and a changed native detail after one SGD step; this does not prove DINO feature suitability.
- **Closest prior:** Standard Jacobian/tangent-space optimization; source's zero initialization. Not a novel theorem.
- **Evidence:** `model.py:77–81`, `104–109`; probe RESULTS.json. Current canonical coarse loss does not traverse native logits, per CURRENT.md; the candidate native energy in the probe is not enabled production.
- **Implication/limit:** Test candidate output quality separately from transfer. A detached proposal can be fitted by conditional CE or matching the declared v-coordinate only after defining the new policy. Full CE of a four-class proposal can also alter BG, coarse logits and RV/LV behavior; it is not automatically covered by this binary theory. The literal exact output step theorem does not describe ordinary AdamW parameter updates.
- **Falsifiable prediction:** First-step detail-head weight gradient can be positive while detail-stem/fuse gradients are zero. Real training needs a separate actual-displacement and transfer diagnostic.

### R3-C5 — Failures that a low image cost cannot identify

- **Statement/status:** COUNTEREXAMPLE. Native intensity fitting is not a complete named-anatomy measurement.
- **Scope/assumptions/formal claim:** If two admissible hidden labelings produce the same observed x, Q/anchor sources and metadata but exchange RV/LV identities, every deterministic estimator of those observations returns the same answer and cannot be correct in both worlds. An identical appearance ring assigned heart in one hidden world and nonheart in another similarly defeats this measurement alone. The construction concerns the available observation model, not a claim that every real cine is observationally identical.
- **Proof/missing step:** Identical inputs imply identical outputs; hidden truth differs. Additional valid semantic/localization evidence is required to break the symmetry. How often this occurs in the actual TRAIN setting is unmeasured.
- **Concrete failures:** (1) RV and LV blood have the same selected appearance distribution. (2) Anchors point to a nonheart bright cavity/dark ring. (3) MYO or ventricular blood is absent in a basal/apical slice but a two-class cost still partitions nuisance. (4) Partial volume or pathology destroys mean order. (5) Brightness-preserving correspondence fails; our chosen single-frame mechanism does not use flow, so no temporal claim follows. If cine pooling is added, out-of-plane motion/phase changes require a fresh measurement model. (6) A freely learned reconstruction decoder could reconstruct x while semantic logits collapse; this candidate avoids that bypass by freezing the measured costs, but does not solve semantic ambiguity. (7) High confidence at a wrong teacher boundary can suppress the feasible correction through saturated r0 and bounded B. (8) The candidate can introduce a within-cell tissue island from pure-cell noise.
- **Closest prior:** Chan–Vese/Kim–Ye are appearance partitioning mechanisms; GrabCut supplies user object localization. No primary source inspected establishes a no-task-annotation named RV/MYO/LV guarantee for this setting.
- **Evidence/implication:** Swapped-anchor and pure-cell counterexamples are executed; same-observation semantic ambiguity is a logical construction. Restrict the surviving claim to conditional interface correction under explicit latent conditions. Do not relabel binary tissue Dice as four-class Dice.
- **Falsifiable prediction:** Anchor permutation, nonheart-ring and empty-class controls can retain low energy and strong gradients despite incorrect anatomical edits.

### R3-C6 — Proposed candidate and cheapest decisive controls

- **Statement/status:** PROPOSED_NOT_APPROVED. Before training any student, test whether frozen native likelihood proposals carry useful corrections beyond their equivalent centered-intensity baseline.
- **Scope/assumptions:** Future, separately authorized TRAIN-only preparation; no labels for tuning or source mapping. The present phase runs only manufactured probes. Freeze source hashes, permitted samples, geometry, normalization, anchor estimator, support and policy identity before later measurement. Historical Q stays full Q in its existing coarse objective; this mechanism does not replace Q with Z.
- **Formal operational output:** Save candidate per-pixel delta, source/gates, invalid support, estimated means, appearance contrast, `||H grad E||`, proposed and actually realized delta, correction norm, achieved energy change, class odds changes, and saturation. Unknown means/sources yield abstention. Numerical nonzero/finite thresholds are implementation tolerances, not anatomy thresholds.
- **Simplest replacements:** (a) Lifted teacher/coarse student, no native objective; (b) direct scalar `s = k H x` with the same support/step magnitude, because C1 proves equivalence at initialization; (c) direct unprojected two-mean threshold or KL-proximal appearance update to expose damage from the coarse-nullspace restriction; (d) classical anchored appearance-only proposal with no neural student. Add the student only after proposal quality and transfer are separated. These are controls for the ONE appearance mechanism, not additional evidence mechanisms.
- **Negative controls:** Replicate native intensity within every cell; swap means/anchor names; hold contrast fixed and shift midpoint; noise-only pure cells; minority occupancies approaching zero; empty structure; nonheart ring; frozen random/constant head features; coordinate misalignment; apply the same augmentation to cost/gate/proposal with recorded transforms. Shuffling costs within a cell preserves their histogram while disrupting localization.
- **No-GT kill criteria:** Missing class/source support, nonfinite costs, failure to preserve valid geometry, a constant/replicated image still creating the claimed appearance signal, no head-weight update despite nonzero output signal, excessive saturation, BG effects outside declared support, or a fancy proposal numerically identical to `k H x` while claimed to add another mechanism. Also kill a scientific claim if only energy/entropy improves. These are falsification gates, not proof of anatomy when passed. Anchor mean order/coverage or round-trip consistency can be diagnostics but cannot certify purity or no false edits.
- **Proof/missing step:** C1–C4 justify the diagnostics. There is currently no observed patient correction precision, no independently established anchor error bound and no approved unseen quality measurement. All patient/GPU/student-generalization claims remain NOT RUN.
- **Closest prior/implication:** Frozen region fitting is already classical. A novel paper claim would need an actual advance in measurement validity, noise-adaptive support, or separately demonstrated transfer; adding this loss to DINO plus an H projection is not enough.
- **Falsifiable prediction:** If gains disappear relative to matched `k H x` or the classical proposal, the proposed complexity has no empirical support. If proposals improve but student outputs do not, the problem is transfer/parameterization rather than appearance evidence.

## Primary literature and collision ledger

All URLs accessed 2026-10-18. The summaries here are intentionally bounded; paper claims are not adopted as task guarantees.

| Source/version/publication status | Exact inspected location and result | Collision, differing supervision, gap, cheapest baseline |
|---|---|---|
| Tony F. Chan and Luminita A. Vese, **Active Contours Without Edges**, IEEE TIP 10(2), 266–277, February 2001, published; received 1999, revised 2000. [Primary paper](https://www.sfu.ca/~rpyke/research/vese_chan.pdf), DOI 10.1109/83.902291. | Section II, especially pp. 267–269, two approximately constant intensity regions, region fitting and relation to piecewise-constant Mumford–Shah; Section II-B level sets. PDF math extraction partly drops symbols; exact energy visual/text context was inspected, so no unverified equation-number attribution is used. | Same squared region residuals. Different task: an appearance-defined object, not noisy named cardiac sources plus a coarse-nullspace student. Unresolved: appearance-to-anatomy linkage. Baseline: frozen two-mean region score/direct threshold. |
| Boah Kim and Jong Chul Ye, **Mumford-Shah Loss Functional for Image Segmentation with Deep Learning**, arXiv:1904.02872v2, 9 September 2019; author manuscript says accepted IEEE TIP; published IEEE TIP 29, 1856–1866 (2020), DOI 10.1109/TIP.2019.2941265. [Version record](https://arxiv.org/abs/1904.02872), [primary full text](https://arxiv.org/html/1904.02872v2), [author institutional publication record](https://pure.skku.edu/en/publications/mumford-shah-loss-functional-for-image-segmentation-with-deep-lea/). | Section III-A Eqs. (12)–(13): softmax-weighted squared image residuals and prediction-dependent class means; III-C Eq. (16) supervised combination, Eq. (19) bias-field variant; III-C2 discusses absence of semantic labels. | Strongest collision: an image-region loss directly trains a CNN without pixel labels. R3 freezes uncertain-anchor means and analyzes a specific H restriction instead of estimating means from current predictions. This changes failure feedback but does not create named anatomy or establish novelty. Baseline: its appearance term with frozen means, zero perimeter, versus `k H x`. |
| Carsten Rother, Vladimir Kolmogorov and Andrew Blake, **“GrabCut” — Interactive Foreground Extraction using Iterated Graph Cuts**, ACM TOG 23(3), 309–314, SIGGRAPH 2004, published. [Primary Microsoft Research paper](https://www.microsoft.com/en-us/research/wp-content/uploads/2004/08/siggraph04-grabcut.pdf). | Sections 2.1–2.2, Eqs. (1)–(3): labeled-region histograms and negative-log data term; 3.1 Eqs. (7)–(10): Gaussian-mixture data cost; 3.2/Fig. 3 alternating estimation and segmentation. | Shared seeded appearance fitting and candidate refinement. Different supervision: user rectangle/trimap and correction brushes; the paper's Section 2.2 smoothness gamma was tuned against training ground truth. Those cannot be silently imported into this no-task-annotation setting. Gap: uncertain anchor localization/purity. Baseline: frozen one-pass appearance proposal before any iterative self-label refit. |
| Chunming Li, Rui Huang, Zhaohua Ding, J. Chris Gatenby, Dimitris N. Metaxas and John C. Gore, **A Level Set Method for Image Segmentation in the Presence of Intensity Inhomogeneities With Application to MRI**, IEEE TIP 20(7), 2007–2016, July 2011; published online 21 April 2011, DOI 10.1109/TIP.2011.2146190. [Primary full text](https://pmc.ncbi.nlm.nih.gov/articles/PMC6952214/). | Section III-A Eq. (3) image/bias/noise model; III-B assumptions and Eqs. (4)–(5) local bias approximation; III-C Eqs. (6)–(10) local intensity clustering/energy. | Same measurement-family requirement: local appearance clusters under slowly varying bias and sufficiently separated intensities. Different objective: joint bias/partition estimation rather than frozen noisy named anchors. Gap: no current evidence bounds cardiac residual sigma or anchor selection bias. Baseline: local frozen two-mean fit; adding learned bias estimation is deferred. |

Search leads not promoted to evidence: Grady **Random Walks for Image Segmentation** (2006) was located on the author's site but not used for an uninspected theorem; a graph affinity mechanism would be a different candidate. Chen et al. **Learning Active Contour Models for Medical Image Segmentation** (CVPR 2019) was located in the official CVF proceedings; detailed PDF extraction failed on subsequent calls, so its formula/supervision is not used as proof. No claim of an exhaustive 2026 prior-art search is made.

## Executed bounded probe receipt

Files owned by R3: `probes/R3/native_appearance_probe.py`, `probes/R3/RESULTS.json`, this dossier. Command: `rtk proxy python reports/research/20261018T030143Z/probes/R3/native_appearance_probe.py`. Final exit 0, CPU torch `2.12.0.dev20260923` as recorded in JSON (consult the JSON if runtime version string differs). No dataset or official weights read. One initial fixture attempt exited 1 because 8-channel GroupNorm at a 1 by 1 deepest stage had one value per group; the fixture decoder was changed to 16 channels, not production. Final run emitted a benign scalar-conversion warning when recording a tensor norm; the measured graph had already been backpropagated. No production/check-repo/full-suite pass is claimed from this scoped probe.

Results JSON SHA256 at initial draft: `193fcfa30131e2509933129264ad92758def80a55a023ad279c3b6880325c635`.

| Manufactured case | Energy before -> after | Binary errors before -> after | Interpretation |
|---|---:|---:|---|
| Mixed cell, correctly ordered noisy anchor means | 1.7744 -> 1.40458357 | 8/16 -> 0/16 | Conditional correction is possible. |
| Same image, swapped anchors | 1.7744 -> 1.40458357 | 8/16 -> 16/16 | Identical energy improvement can accompany total binary failure. |
| Constant native measurement | .7744 -> .7744 | 8/16 -> 8/16 | Zero native gradient. |
| Coincident estimated means | 1 -> 1 | 8/16 -> 8/16 | Zero discriminative appearance gradient. |
| Pure blood with intensity nuisance | 2.11001041 -> 2.03087045 | 0/16 -> 8/16 | H can create false tissue. |
| Shift both anchor means by .1 | 2.02194451 -> 1.64820490 | 8/16 -> 0/16 | Same initial H gradient as the first row. |

Current `LayerGuidedStudent` toy fixture: 79,826 parameters, manufactured encoder, 32 by 32 image, one head-only SGD step at learning rate .1. `||H grad_l E|| = .0016320625`; detail-head weight gradient norm `.0401258804`; head-weight displacement `.00401258795`; realized native-detail norm `.119479433`; energy `.110900015 -> .110739022`. Detail-head bias and detail-stem/fuse first-step gradients were zero. These are software/gradient observations, not patient segmentation, DINOv3 loading, training quality, or GPU evidence.

## Independent conclusion before cross-review

The candidate provides a real within-cell image force and a constructive conditional correction theorem with noisy anchors. It also exposes why apparent success can be misleading: under exact H and equal variance, its initial direction is merely centered intensity multiplied by an uncertain semantic sign, and even correctly signed contrast can hallucinate a minority class in a pure cell. The only defensible surviving hypothesis is **localized blood–MYO interface correction where the teacher supplied sufficiently good names/support and the cell is genuinely mixed**. Those truth-dependent conditions are not operationally certified today. Keep this mechanism as a transparent baseline/proposal study; do not make it the sole explanation for learning RV/MYO/LV or approve a new training policy from this dossier.

## Append-only review record

Awaiting three actual cross-review rounds. No reviewer response or consensus is fabricated.
