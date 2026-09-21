# Astra final scientific verdict

**TEACHER_NOT_READY. NO-GO for advancing the current v3 recipe to long student training or a .91/special-issue claim.** A bounded redesign of the named-seed gate is justified; its success is unproven. This is not a theorem that all mask-free cardiac segmentation is impossible.

Audit base: `d4f503b7107eca4e8ac85f37a5dd01ec50479666` on main. Work is isolated on `QuocKhanhLuong/astra-pseudolabel-v3-audit`. Main is not merged. Root Astra inspected decisive source, reproduced defects, reran intervention/metric checks and rejected several worker conclusions. [Orca ledger](01_ORCHESTRATION.md) records AGY investigations and dispositions.

## Answers to the required decisions

| Question | Root judgment and evidence |
|---|---|
| Is the architecture coherent? | **As an interface scaffold, partly.** An expensive offline teacher feeding a cached-feature recurrent student is coherent. The teacher has no implemented named-anchor producer, training objective or full-cine data path. Its “motion” branch learns from frame differences, not estimated correspondence. Unmasked reconstruction can use the very image it predicts. [Code audit](02_CODE_AUDIT.md) |
| Can it produce useful named anatomy now? | **Not established; the current teacher does not ground the four names.** Class-symmetric objectives leave semantic permutations unresolved. Typed priors can supply names, but the frozen local precursor failed class precision/coverage gates. [Identifiability](03_SUPERVISION_AND_IDENTIFIABILITY.md), [Round-0](09_EXPERIMENT_PLAN.md) |
| Is DW correctly placed? | **Yes, in the final annotator.** `feature_only` needs no Auditor, retains image/logit/entropy conditioning, and reuses one encoder pass per slice batch. A0→A1→A2 is differentiable recurrence. Compact is exactly A0. Balanced executes one DW call; accurate executes three internal DW calls across two outer turns. [DW review](05_DYNAMIC_WINDOW_REVIEW.md) |
| Does DW add value beyond an ordinary refiner? | **NOT ESTABLISHED.** Content-dependent sampling is real, but K is fixed and there is no matched-compute accuracy benefit. A worker CNN comparison was confounded and is rejected. Keep DW only if the specified ordinary-CNN control loses at a matched deployment budget. |
| Are the profiles adaptive/scalable? | They are three real, manually selected compute levels. The current API has no hardware/uncertainty controller and no matched-budget adaptive benefit. CPU forward p50 is 9.73/15.28/24.99ms at 224 on this host; these are random-weight software timings, not a .91 operating point. [Resource audit](07_RESOURCE_ADAPTATION.md) |
| What is novel? | No validated scientific novelty yet. Early learning, prototypes, pseudo-label evolution, temporal consistency, deformable sampling and conditional compute have prior art. A class-specific semantic-conflict/transport rule might become a contribution only if it reduces swaps and harm at matched coverage across both datasets. UI is a system deliverable. [Research](04_PSEUDOLABEL_RESEARCH.md), [fit](10_SPECIAL_ISSUE_FIT.md) |
| Feasibility for the special issue? | **NO-GO on current evidence.** Missing named-label quality, full-cine/native-geometry verification, M&Ms execution, matched refinement controls and hardware-constrained results are substantive dependencies. A small seed/data redesign is the next authorized scientific step, not a long run. |

## Decisive results

Two software defects were independently reproduced and corrected only on the audit branch: pseudo CE evaluated UNKNOWN=255 before filtering, and confident but conflicting regions could yield an accepted, uniformly ambiguous semantic pixel. The fix selects valid non-UNKNOWN targets before CE and checks confidence/margin at final dense pixels. Original scoped tests:3 passed. Added regression tests exposed4 failures; corrected combined scope:16 passed. This establishes software behavior, not segmentation quality. [Tests](08_TEST_RESULTS.md).

The frozen **geometric seed diagnostic**, not a trained `CinePseudoTeacher`, used 4 train and 4 development patients,16 phase volumes. Seven cue arms were frozen before an independent evaluator opened masks. The preregistered combined candidate obtained patient/phase/class mean foreground Dice **.142947 train / .139116 dev**. On dev, RV precision1.00 covered only1.09% of true RV; MYO precision was.8726; LV precision1.00 covered32.97%. Every arm failed the per-class seed gate. No progressive teacher training or long student run followed. Root reproduced all metrics and tested actual rejection of tampered freeze inputs.

The local ACDC copy has 200 3D ED/ES images from 100 patients and no observed 4D cine in the audited root. Every image has unknown physical units and inconsistent affine/header scales. Stored-grid evaluation is possible; trusted millimetre/native-orientation processing is not established. M&Ms data and CUDA were unavailable. MPS forward timing was measured separately. The proposed4-core/8GB target was **NOT RUN** on this 12-core/24GB host.

C0(.01623), C1-cache(.02146/.02751), and C1-direct(0) remain negative controls. They are not teacher targets. No old reference-free Auditor is reinstated. Their historical phase aggregation differs from the newly locked protocol.

## One baseline to implement next

Lock **B1: physical/cine semantic anchors → image-only SSL U-Net teacher → A0-first student**, as specified completely in [09_EXPERIMENT_PLAN.md](09_EXPERIMENT_PLAN.md) and [baseline_B1.json](baseline_B1.json). First reacquire/verify full cine and geometry and test the named-seed gate. Only if it passes: masked image-only pretraining, a fixed early learner, one provenance-tracked prototype/temporal recruitment, then independent teacher admission near.88–.90. A required random-initialization control makes the scratch/SSL distinction explicit. Neither setting imports mask-supervised weights.

Only after TEACHER_READY: train width 32 student A0, freeze encoder+A0, then train the refiner with intermediate losses. Compare A0, ordinary CNN and DW under matched budgets; use the same frozen checkpoint family for compact/balanced/accurate. Evaluate native3D RV/MYO/LV Dice equally over patient, ED/ES and class, with UNKNOWN as false negatives, no BG and no GT Hungarian mapping. ACDC≥.91 needs sequestered independent evaluation. Run frozen ACDC→M&Ms before adaptation, and the same native recipe separately on M&Ms. Full modules, losses, epochs, thresholds, router proposal and stopping criteria are locked in report 09.

Strict scratch reaching the target is unsupported by this audit. SSL can improve representation but cannot supply class names by itself. Neither a decrease in loss nor higher teacher/student agreement can reopen the failed teacher gate.
