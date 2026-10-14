# Learning mechanism audit: executed stable guidance, 14 October 2026

The evidence rejects global permission blocking and zero semantic eligibility as explanations of these runs. It establishes sparse, uneven named hard support; an objective dominated by soft-target entropy; residual student-to-bank mismatch; and substantial baseline export sensitivity relative to the rare accepted support. It does **not** identify which named predictions are already wrong in R0, become wrong in Q, or become wrong during student/native transfer. The new checkpoint has no published native-versus-control or stagewise grounding measurements.

This is a research-only text-evidence analysis at repository SHA `b937e3b746608725e43d3bc3cef8861666d152dc`. No production files, datasets, weights, training, GPU execution, TEST scores, thresholds, or checkpoint selection were changed. Exposed VAL outcomes below are declared reference-informed development evidence.

## Evidence identity and how to reproduce this report

`E` below means `reports/stable_guidance_world_v2_20261014/`. Source citations always mean `E/executed_source/`, unless an explicit historical SHA is attached; they do not refer to historical top-level `src/`.

- **VERIFIED:** read the executed source and launcher patch; all 102 manifest-listed published files match their `MANIFEST.json.files[].published_sha256`; all 11 executed package source hashes match `run_evidence/run/RECIPE.json.source_sha256`. Patch hash is `ecbcd639fbaa010ac3f7fca9ac5ba3272278806ce92eecc8171c4833f70815ee`.
- **VERIFIED:** baseline `student.py` at `d5a45f31481f0a702a1f9d5d418889fe704e9105` hashes to `87c33df231348baa87193e84a80970ee825206d1f2eb01e2f349d3ebb3ce4888`, matching the baseline source receipt. AST comparison confirms unchanged `Block`, `GuidanceStudent`, `guidance_loss`, `contexts`, `load_student`, and `predict_cine`. The new student hash is `d216bdc7ca8e9b01b368457aed2b91065e8124a2d3be2a9875d1f6284b3387fa`.
- **REPORTED:** baseline checkpoint `516e569c9141e2d8edb00b598df5cf12c8a9a4741cf1ae0cca8cbf23d6d03353`; new checkpoint `90bc16bb9f89963b452355376673e3ba9aa24b38870e4595237d75e402ce873e`. The new source base is `7d1c6a04064cda473d1eccfbed818392c37b1a37` plus the recorded launcher patch. These checkpoints are not included locally, so their original binaries were not rehashed here.
- **DERIVED:** all numerical aggregates below are reductions of published measurements, with arithmetic independently checked. They are not newly executed model measurements. **BLOCKED** identifies absent measurements, not a request to run training.

Reproduce with `rtk proxy python reports/guidance_research_continuation_20261014/LEARNING_MECHANISM_CHECKS.py`. Outputs are [checks JSON](LEARNING_MECHANISM_CHECKS.json), [320 patient/class rows](LEARNING_MECHANISM_PATIENT_CLASS.csv), [80 baseline sampled batches](LEARNING_MECHANISM_BASELINE_BATCHES.csv), [50 epochs](LEARNING_MECHANISM_EPOCHS.csv), and [support-overlap bounds](LEARNING_MECHANISM_SUPPORT_OVERLAP.csv). The stdlib script reads text/JSON and immutable Git objects only. It asserts CE identities, permission counts, schedule-counter agreement, source hashes, and paired support bounds. Sanitized published manifests remain report copies; matching their published hashes is not verification of unavailable original artifacts.

## Exact route from R0 to native export

| Stage | Executed behavior | Evidence |
|---|---|---|
| Named initialization R0 | Frozen feature/intensity clusters produce wall/cavity/right-side hypotheses; votes name MYO/LV/RV and border BG. Votes are blended with uniform probability using strength `1-exp(-mass)`. Repeated multiscale votes are not asserted independent. | `src/self_audit_guidance/core.py:193–288`, especially `247–281` |
| Fixed guidance Q | Patient-local graph with spatial/temporal edges; row-stochastic restart iteration `Q <- alpha R0 + (1-alpha) P Q`, alpha `.35`. This is not clamped-boundary harmonic propagation. | `core.py:115–168,291–298` |
| Reliability/hard bank | Reliability is propagated strength times `1-H(Q)/log(4)`, clipped to `[0,1]`. Hard labels need maximum probability `.70`, margin `.20`, and reliability `.05`; otherwise label **255**, not BG. | `core.py:295–307`; run config |
| Global permissions | Class `c` allowed when at least two TRAIN patients have any hard pixel of `c`; any allowed foreground enables the semantic branch. Permissions are global, not patient-specific. | `student.py:72–82` |
| Soft weights | For each bank pixel `i`, `w_i = reliability_i * allowed[argmax Q_i]`. No hard acceptance mask and no `.05` reliability threshold is applied to the soft loss. All four components of Q remain in CE, including channels whose class would be disallowed if it won. | `student.py:97–103,38–58` |
| Batch activation | Batch active iff semantic branch enabled and at least one pixel has non-BG Q argmax and strictly positive `w`. A BG-only batch is semantically inactive even if it has positive weights. | `student.py:54–58` |
| Common-grid objective | Native image input uses neighboring z slices, with edge duplication. Student native logits are bilinearly resized **before** log-softmax to the bank grid `48x48`; student projected features are independently resized to teacher grid. Loss is unscaled `feature cosine loss + weighted soft CE`. | `student.py:19–58,62–64,93–106`; `pipeline.py:44–55` |
| Optimization/count | AdamW, lr `.0003`, decay `.0001`; norm clipping at 5. Inactive batches set decoder/head gradients to `None`; shared encoders still learn features. `successful_semantic += int(active)` counts eligibility, not nonzero gradients or parameter movement. | `student.py:78,104–110` |
| Checkpoint/export | Fixed last epoch. Native output softmax is gated at `.70` probability, `.20` margin and allowed winning class. Export has **no teacher reliability gate**. `REPRESENTATION_ONLY` short-circuits to all 255. Native TZYX labels are written with recorded geometry. | `student.py:122–158`; `pipeline.py:78–98` |

The launcher patch forwards `world_left_v2` and diagnostics, verifies the policy/logging receipt on resume, and forwards verified ACDC frame-grid handling to evaluation. It does not introduce a new student objective or change inference gates (`run_evidence/launcher_integration.patch:16–58`). Both prediction cohorts are frozen before reference evaluation in that launcher path (`executed_source/src/self_audit_guidance/full.py:203–243`). Target masks do not enter this naming/training route; the present audit does not establish the frozen upstream model's complete pretraining lineage.

## What support actually reaches learning

### Whole baseline bank versus hard labels

These are all 80 TRAIN patients, **46,575,360 bank pixels**, from `baseline_forensics/train_all/DIAGNOSTICS.json.patients.*.classes` and scalar bank fields. Counts concern the common bank grid, not native area or independent anatomical instances.

| Class | Informative R0 argmax | Q argmax, all pixels | Hard accepted | Positive-weight Q-argmax pixels | Of these, hard UNKNOWN | Patients hard / soft |
|---|---:|---:|---:|---:|---:|---:|
| BG | 6,882,154 | 44,614,861 | 795,189 | 8,454,581 | 7,659,392 | 62 / 80 |
| RV | 95,391 | 173,642 | 5,199 | 133,999 | 128,800 | 26 / 79 |
| MYO | 904,973 | 1,692,285 | 61,712 | 1,285,209 | 1,223,497 | 55 / 80 |
| LV | 86,706 | 94,572 | 6,777 | 90,158 | 83,381 | 47 / 80 |

There are **868,877 hard accepted pixels (1.8655%)**, **9,963,947 positive-weight pixels (21.3932%)**, and **9,095,070 positive-weight pixels whose hard label is 255**. Thus 91.2798% of positive-weight pixels are hard UNKNOWN; for RV that fraction is 96.1201%. Soft-only support appears in 53 RV patients, 25 MYO patients, 33 LV patients, and 18 BG patients without that class's hard acceptance. All permissions are nevertheless true. Positive weight can be extremely small; these counts are not effective sample size or named anatomy evidence.

All 868,877 hard labels are on initially informative nodes: `accepted_initially_neutral` sums to zero. This is an observed hard-bank fact and does not imply that neutral/UNKNOWN pixels are excluded from soft learning. Likewise, `Q.argmax==BG` includes ties at the uniform vector and is not proof that the pixel is anatomical background. The bank contains 36,611,413 exactly zero-reliability pixels and no permission-vetoed pixels.

Baseline mean Q confidence within each Q-argmax group is BG `.299036`, RV `.420422`, MYO `.436314`, LV `.571816`; corresponding mean entropy is `1.350649, 1.255982, 1.235542, 1.111426` nats. These are pixel-weighted reductions of patient summaries over **all** pixels in each argmax group, including zero weights, not the training-weighted entropy or the hard subset. Whole-bank reliability-weighted entropy is `1.058390` nats; it must not be substituted for an equal-active-batch entropy estimate.

### Change under world_left_v2

| Class | Baseline hard pixels / patients | New hard pixels / patients | Baseline/new hard pixels in top 5 patients |
|---|---:|---:|---:|
| BG | 795,189 / 62 | 795,381 / 62 | 32.41% / 32.42% |
| RV | 5,199 / 26 | 877 / 14 | 84.63% / 70.24% |
| MYO | 61,712 / 55 | 61,765 / 55 | 38.47% / 38.47% |
| LV | 6,777 / 47 | 6,777 / 47 | 37.70% / 37.70% |

**DERIVED:** new bank hard coverage is 1.8568%, with 864,800 accepted and 45,710,560 UNKNOWN pixels. All four new global permissions remain true, including RV supported in 14 patients. The median patient has zero hard RV in both runs. One baseline patient supplies 3,258 of the 5,199 hard RV pixels; maximum new per-patient RV is 215. Per-patient values and all class deltas are in the 320-row CSV.

The 52 coded-geometry patients have exactly unchanged hard totals `[428926,877,32101,3711]`. The 28 uncoded patients change from `[366263,4322,29611,3066]` to `[366455,0,29664,3066]`; their new RV raw vote mass is exactly zero. These facts come from baseline patient geometry fields and `run_evidence/run/bank/REPORT.json.patient_reports.*.{class_pixels,anatomy.raw_vote_mass_per_class}`, not inferred new Q arrays.

**DERIVED, complete support obstruction:** on any such patient, absent RV votes imply `R0_i,RV=(1-s_i)/4 <= 1/4` from `core.py:277–281`. Q starts at R0. If every current RV coordinate is at most 1/4, each row-stochastic average `(P Q)_i,RV` is at most 1/4; its convex combination with `R0_i,RV` is also at most 1/4. Induction proves the bound for every exact iteration, and the limit obeys it too. Patient-local edges prevent another patient's RV anchors from entering. The probability gate `.70` is therefore unattainable for RV in that patient's bank; numerical tolerances are far below the `.45` gap. This explains zero hard RV support in those 28 patients without claiming whether removed RV hypotheses were true or false. It does not prohibit a globally trained student from predicting RV there, and does not prove the cause of new VAL RV disappearance.

**BLOCKED:** new per-patient R0/Q-argmax, positive soft-weight, soft-on-UNKNOWN, and class-conditional confidence/gradient totals are not published. The epoch logger provides whole-bank channel-weight sums, not those spatial/class-specific statistics. Old-bank counts must not be relabeled as new-bank measurements.

### Soft channel mass does not mean positive named evidence

Baseline `sum_i w_i Q_ic` is `[338048.14,84381.57,132990.98,85815.28]`. New epoch-50 channel masses are `[337656.68,81770.89,132625.53,85458.25]`, or `[52.965,12.827,20.804,13.405]%` of total mass. These sums include inactive batches **before** activation (`experimental_diagnostics.py:33–38,46–56`); every TZ pair is visited once per epoch (`student.py:89–94`). Their near constancy across epochs corroborates a fixed bank, not improved labels.

For a coarse logit `z_ic`, define `p_i=softmax(z_i)`, `d_b=max(sum_i w_i,1e-8)`, and batch eligibility `A_b`. Differentiating `log softmax(z)_k=z_k-log(sum_j exp(z_j))` gives

`d CE_b / d z_ic = A_b (w_i/d_b)(p_ic-Q_ic)`.

Writing `u_c=1/4`, this is `A_b(w_i/d_b)[(p_ic-u_c)-(Q_ic-u_c)]`. Exactly uniform Q has zero entropy-based reliability and hence zero weight in this implementation. Instead, with positive propagated strength and allowed classes, a pixel with `Q=(.4,.2,.2,.2)` has positive entropy contrast and weight w: its RV channel mass is `.2w>0`, although `Q_RV-.25=-.05`. Include a second positive-weight pixel with `Q=(.1,.1,.7,.1)` so the foreground MYO argmax activates the batch. At the first pixel, from `p=(.25,.25,.25,.25)`, the RV logit derivative is `(w/d_b)(.25-.2)=.05w/d_b>0`, so gradient descent on that logit lowers it. Positive RV channel mass therefore need not be favorable named RV support and cannot refute weak RV grounding or imbalance. Aggregate signed centered mass is baseline approximately `[177739.15,-75927.42,-27318.01,-74493.71]`; new approximately `[178278.84,-77606.95,-26752.30,-73919.59]`, using the summed new channels as total weight, up to recorded floating-point reductions. These signed totals allow cancellation and include inactive batches; they are neither the batch-normalized gradient nor a measure of useful positive RV signal. **BLOCKED:** positive-part contrast `sum w max(Q_RV-1/4,0)`, margin-stratified evidence, active-batch class contrasts, and their anatomical validity require absent Q/weight arrays or richer logs.

The displayed `p-Q` derivative assumes normalized probability targets. With floating-point stored targets whose row sum is `s_i`, the exact symbolic derivative is `A_b(w_i/d_b)(s_i p_ic-Q_ic)`; the code validates row sums within `2e-4`, rather than renormalizing at this point. Thus the centered-target argument and nonnegative-KL proof below refer to the probability-simplex model, with numerical reduction error separately checked in the receipts.

## Exact loss, entropy floor, and update semantics

For batch b, define native logits Z, bank resampling D, `p_i=softmax((DZ)_i)`, detached Q and w, and the activation A above. Then

`CE_b = A_b / d_b * sum_i w_i[-sum_c Q_ic log p_ic]`,

`H_b = A_b / d_b * sum_i w_i[-sum_c Q_ic log Q_ic]`,

`KL_b = A_b / d_b * sum_i w_i[sum_c Q_ic log(Q_ic/p_ic)]`.

With `0 log 0=0`, adding the last two expressions cancels `Q log Q` termwise, hence **CE_b = H_b + KL_b**, including inactive zeros and the denominator clamp. For each probability vector Q and positive softmax p, `log x <= x-1` implies `sum_{Q_c>0} Q_c log(p_c/Q_c) <= sum_{Q_c>0}p_c-1 <= 0`; hence KL is nonnegative, CE is at least H, and p=Q attains the bound when representable (or in the limiting sense for zero Q components). If `sum w < 1e-8`, H is the correspondingly scaled entropy term, not a normalized probability-distribution entropy. This distinction matters in the actual baseline sample below. These derivations use the code's manual soft-label CE; the probability-target convention is also documented by [PyTorch CrossEntropyLoss](https://docs.pytorch.org/docs/2.14/generated/torch.nn.CrossEntropyLoss.html).

The feature term averages `1-cosine(student_feature,teacher_feature)` over teacher vectors with squared norm above `1e-10`, without semantic weights. `loss=feature+CE`; shared encoders receive both gradients when active. Cosine feature agreement does not certify the named semantic readout: the architecture has separate projection and decoder/head paths (`student.py:23–35`).

`epoch_terms` averages already-normalized batch CE/H/KL equally, inserting inactive zeros; it is not a pooled-pixel loss. If `a=n_active/n_batches>0`, each logged term equals `a` times its active-batch mean. Fixed Q does not make epoch H constant: shuffling changes grouping, normalization denominators, and which BG-containing batches become active. `semantic_updates` is exactly the sum of activation indicators; it can increment when gradients are zero at an optimum, arbitrarily small, cancel in shared parameters, or produce no detectable parameter movement. AdamW state and decay also make gradient presence different from measured parameter deltas. The inactive decoder/head `grad=None` skips their optimizer step, unlike a zero tensor; see [PyTorch AdamW, zero_grad](https://docs.pytorch.org/docs/2.14/generated/torch.optim.AdamW.html#torch.optim.AdamW.zero_grad).

### Baseline: sampled checkpoint forwards and whole-bank replay are distinct

`DIAGNOSTICS.json` records one first-eligible lexicographic TZ block per patient, chosen by bank eligibility rather than loss or reference masks; all 80 selected batches are active. This is not random sampling, a complete checkpoint epoch, or original optimization history (`diagnose_stable_guidance.py:195–239`).

| Baseline selected-batch statistic | Equal-batch mean | Range / other evidence |
|---|---:|---|
| CE | 1.178375 | `.0000146–1.471114` |
| Entropy term | 1.124532 | `.0000135–1.386276` |
| KL | .053843 | `.00000106–.268369` |
| Shared feature gradient norm | .443247 | `.050137–.857038`; nonzero in 80/80 |
| Shared semantic gradient norm | .243567 | `.000002934–1.174060`; nonzero in 80/80 |
| Gradient cosine | .196053 | `-.399604–.670428`; negative in 19/80 |
| Coarse argmax agreement with Q | 82.9129% | 76.1447–92.0085%; includes zero-weight pixels |

All gradients are fresh final-checkpoint **pre-clipping shared-encoder** autograd measurements with no weight updates (`experimental_diagnostics.py:84–97`); they say nothing directly about historical gradients or decoder/head gradient magnitudes. A positive mean cosine does not remove the 19 observed conflicts. Agreement is bank fidelity, not accuracy.

Patient024 demonstrates the eligibility caveat concretely: selected `weight_sum=9.761512616e-14`, denominator `9.999999939e-9`, CE `1.45948e-5`, entropy term `1.35323e-5`, semantic gradient norm `2.934e-6`, yet `active=true`. Its near-zero logged entropy term comes from the denominator floor and is not evidence of certain targets. Patient027 has weight only `5.03909e-8` but entropy `1.386276` and semantic gradient norm `.213300`; above the clamp, within-batch normalization can magnify low absolute weight. Evidence: `patients.patient024.batches[0].semantic` and `shared_encoder_gradients`, likewise patient027. The new epoch summaries omit denominator distributions, so prevalence of these phenomena in the new training schedule is **BLOCKED**.

Separately, the baseline whole-bank fixed-seed schedule replay reports **110,604 eligible / 127,650 steps = 86.6463%**, matching the checkpoint counter. It replays eligibility only; `original_epoch_replay=false` and `actual_historical_gradients=NOT_RECONSTRUCTIBLE_FROM_FINAL_CHECKPOINT` explicitly preserve the distinction. Baseline original epoch logs do not contain H/KL; multiplying a selected-batch CE by the historical active fraction does not reconstruct their objective.

### New run: 50 recorded training epochs

Evidence: `run_evidence/run/student/training.json[*].semantic_diagnostics`; 2,553 steps each epoch, batch size 8, fixed final checkpoint. Values below are nats.

| Epoch | Active fraction | CE including zeros | H including zeros | KL including zeros | Active CE | Active KL | Feature loss |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | .864473 | 1.045715 | .981936 | .063779 | 1.209656 | .073778 | .327987 |
| 5 | .866823 | 1.041730 | .987124 | .054606 | 1.201779 | .062995 | .261571 |
| 10 | .868782 | 1.037434 | .987862 | .049571 | 1.194125 | .057059 | .242209 |
| 20 | .857031 | 1.015698 | .972019 | .043679 | 1.185136 | .050965 | .220857 |
| 30 | .869565 | 1.025569 | .986576 | .038993 | 1.179404 | .044842 | .209091 |
| 40 | .859381 | 1.012939 | .977874 | .035065 | 1.178684 | .040803 | .200202 |
| 50 | .864865 | 1.013809 | .981306 | .032503 | 1.172216 | .037581 | .194335 |

**DERIVED:** 110,461 eligible / 127,650 optimizer steps = **86.5343%**, 143 fewer eligible steps than baseline; epoch active fraction spans 85.6639–87.9749%. Epoch-50 H is **96.7940% of CE**. Active H is `1.135878 -> 1.134635`; active KL drops about 49.06%. Last-ten mean active KL is `.038581`; loss is improving but not monotonically (11 upward transitions in zero-inclusive KL, 9 in active KL). CE=H+KL holds for all 50 records within `3.69e-9` of the reported reductions.

This supports fitting progress and an entropy-dominated plateau. It does not prove optimization has converged, that the final checkpoint achieves the logged epoch-average KL, or that semantic gradients were nonzero on all eligible steps: epoch logs combine changing parameter states. The nonzero residual KL is not a named-error estimate. New gradient conflict, native/control behavior, and final fixed-checkpoint whole-bank fitting remain **BLOCKED**. Comparing baseline `.053843` selected-final-checkpoint KL to new `.032503` training-epoch KL is not a matched improvement estimate.

New epoch-50 class CE contributions are BG `.277364`, RV `.240176`, MYO `.255231`, LV `.241038`. These sum to total CE; they are not per-class mean losses, class-specific fit quality, or positive named support. The all-channel diffuse target creates contributions even when a class rarely wins.

## Native export and rare-support sensitivity

The training objective observes `softmax(DZ)` on the coarse bank grid. Actual export observes `softmax(Z)` at native resolution. The diagnostic control instead computes `U softmax(DZ)`, with bilinear probability upsampling U, and applies the **same** gates (`experimental_diagnostics.py:100–123`). These operations generally differ; bilinear resizing and `align_corners=False` are specified in [PyTorch interpolate](https://docs.pytorch.org/docs/2.14/generated/torch.nn.functional.interpolate.html). These API links explain semantics, not the original runtime version or performance.

**DERIVED, complete logit-level counterexample:** on a 2x2 native grid let every logit vector be `log(.1,.1,.1,.7)`. Bilinear downsampling to 1x1 averages the four centers. Add RV perturbations `[[20,-20],[-20,20]]`, whose average is zero. The coarse logits, softmax, and every coarse CE/H/KL term remain exactly unchanged, while RV becomes native argmax at the two positive-perturbation pixels. Thus coarse semantic loss alone cannot determine all native labels. This is a statement about logit tensors, not proof the trained architecture realizes this perturbation or that the patient's errors take this form. The existing manufactured probe records the corresponding example (`diagnose_stable_guidance.py:89–110`; `baseline_forensics/math_checks/DIAGNOSTICS.json`).

For the 80 **baseline** selected checkpoint batches:

| Output | Pixels | Accepted | Accepted fraction | Probability rejected | Margin rejected | Permission rejected |
|---|---:|---:|---:|---:|---:|---:|
| Common grid | 1,474,560 | 8,716 | .591092% | 1,465,844 | 251,054 | 0 |
| Native | 35,335,936 | 215,428 | .609657% | 35,120,508 | 6,121,882 | 0 |
| Projected control | 35,335,936 | 121,350 | .343418% | 35,214,586 | 6,649,444 | 0 |

Veto counts overlap. With four nonnegative normalized probabilities and max at least `.70`, the runner-up is at most `.30`, so margin is at least `.40` and the `.20` margin gate is redundant wherever the probability gate passes. All classes are allowed here; hence probability rejection equals total rejection. This is a gate fact, not evidence that lowering the threshold would improve anatomy. The control accepts fewer pixels; it has no demonstrated accuracy advantage.

The equal-batch mean native/control ungated argmax disagreement is **2.78387%**, and gated disagreement is **.344966% of all pixels**. Those percentages alone obscure sparse acceptance. Multiplying each published gated disagreement by its batch's native pixel count and rounding reconstructs **119,330 differing pixels**, a pooled **.337702%**; the maximum integer-rounding residual is `.0001832` pixel. Batch sizes are below `2^24`, making float32 mean rounding unambiguous at the integer scale here. This is **DERIVED** count reconstruction, not a newly read joint label map.

Let native/control accepted sets have sizes `a,b`, overlap `i`, and total gated-label disagreement `d`. All pixels accepted by exactly one output disagree; pixels rejected by both have label 255 and agree. If `v` counts different class labels within the overlap, then `d=a+b-2i+v`, with `0<=v<=i`. Consequently `ceil((a+b-d)/2)<=i<=a+b-d`, additionally intersected with `max(0,a+b-N)<=i<=min(a,b)`. Applying these bounds per batch and summing gives:

- Accepted overlap: **108,724–121,350 pixels**, or **50.47–56.33% of native accepted support**.
- Native-only accepted: **94,078–106,704**, or **43.67–49.53% of native accepted support**.
- Total differing label pixels / native accepted pixels: **55.39%**. This last ratio includes control-only pixels and is not itself an error fraction on native support.

Therefore resolution handling is consequential on the baseline's accepted support even though its global disagreement percentage is small. The available marginal counts do not identify exact shared accepted labels, classwise support changes, or which route is anatomically correct. They do not establish the size or direction of this effect on the new checkpoint.

## Hypotheses, established failures, and unresolved stages

| Hypothesis | Evidence-bounded finding |
|---|---|
| No semantic permission | **Falsified for these runs:** baseline hard-patient counts `[62,26,55,47]`, new `[62,14,55,47]`, all above the permission floor. Positive new eligibility is logged. |
| CE near 1 means no fitting | **Falsified as a general interpretation:** CE=H+KL; new final-epoch H contributes 96.794% of CE. Selected baseline gradients are nonzero. |
| Fitting fully solved | **Not established:** new active KL remains `.037581`; selected baseline KL reaches `.268369`; class-specific final-checkpoint fit and whole-bank native transfer are unavailable. |
| Rare/imbalanced named evidence | **Established in counts, causal size unresolved:** few RV hard pixels, high concentration, and 28 patients with no new RV votes. Soft channel mass does not establish positive RV evidence; aggregate centered foreground mass is negative. |
| Export-transfer failure | **Established baseline sensitivity, unresolved anatomical contribution:** 43.67–49.53% of native accepted support is absent in control. Native predictions are weakly constrained by coarse CE; new checkpoint's analogous measurements are absent. |
| Wrong anatomical grounding | **Unresolved upstream:** R0 uses declared priors that can fit lookalikes; Q/CE/fidelity are self-consistency measurements. No paired R0/Q/native reference audit establishes first named error. Poor final VAL predictions establish end-to-end inadequacy, not its first cause. |

**Earliest established support bottleneck:** the baseline hard bank already covers only 1.8655% of bank pixels with severe class imbalance; the new run loses RV voting before propagation in 28 patients and thereby cannot create hard RV there. This is earlier than optimization/export. It is not the earliest established *anatomical misclassification*, which remains unlocalized. The hard bank is a permission source, not the full soft supervision, so its sparsity alone is not a proof of absent learning.

**Endpoint evidence, REPORTED and reference-informed:** exposed VAL has new mean foreground Dice `.0506852`, coverage `.00366846` (99.6332% UNKNOWN), and zero RV predictions in all 36 scored frames; baseline Dice `.0468441`, coverage `.00614758` (`summary.json.metrics`, `.comparison`, `.rv_val_frames_with_predictions`). The new precision macro excludes RV because its precision is undefined, so its apparent increase is not a comparable precision gain. No TEST scoring or tuning is used here.

**Why fidelity cannot certify grounding, complete counterexample:** take any image population with true named label RV, but let the fixed target be a high-confidence wrong LV distribution; a student that reproduces that target minimizes soft CE and agrees with its argmax while remaining wrong about anatomy. None of CE=H+KL, feature matching, optimizer completion, or contraction checks consults the true label. Accordingly those quantities cannot logically imply anatomical correctness without an independently justified link between Q and the named structures. This counterexample concerns the insufficiency of these certificates; it does not assert that every observed target is wrong.

## At most two discriminating next checks

1. **Frozen checkpoint fitting and transfer decomposition, on TRAIN only.** If the already-authorized original 4070/Downloads arrays and checkpoints are reachable, replay the same fixed TZ blocks for both checkpoints, then a declared full-bank pass if feasible, without optimizer steps. Report exact CE/H/KL with batch denominators and activation; class-conditional positive/negative centered Q contrast; shared-gradient conflict on the fixed blocks; common-grid/native/projected-control gates and joint accepted-label overlap by class and geometry stratum. Keep original thresholds and permission rules. This separates weak named supervision, residual fitting, negligible-weight eligibility, and native transfer; no data/weights download, training, or architecture change is proposed. Current status: **BLOCKED by arrays/checkpoints absent from this package or unreachable at their original location, not by missing permission**.

2. **Frozen stagewise grounding audit on exposed VAL as development evidence.** Before reading any new reference counts, declare fixed cases/geometry strata and unchanged teacher, hypotheses, thresholds, checkpoint and output definitions. Compare independently generated R0/Q and checkpoint native/control outputs against the already-exposed references solely in a read-only evaluator, with stage/class TP/FP/FN, UNKNOWN coverage and lookalike localization; never feed masks into naming, permissions, training, threshold choice, or checkpoint choice. This separates already-wrong anchors from propagation, student fitting and export error, and tests whether high-fidelity regions are nevertheless wrong. Current status: **BLOCKED by absent stagewise arrays and unexecuted frozen teacher audit**; no official TEST evaluation is authorized.

The report and checks are complete; those two proposed measurements remain unrun. No new architecture is justified by this evidence.
