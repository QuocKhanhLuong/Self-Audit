# F cross-review of C — independent challenge

Reviewed initial stable `workers/C_empirical.md` (SHA256 `ddd066c555ad307d35ea8edc62bec7855f4b89ce7974eb3ee879daea1f32f986`) and `evidence/C_recomputation.json` against immutable experiment source/results at `8960b6baac4c6600553d900c303f192c2fcfa4f0`, plus main `637f38c`. The historical full50 appendix was not independently reconstructed by F; C provides a separate provenance/probe for that section.

## Verdict

The latest-v2 numerical and source-provenance conclusions survive this review. There is no contradiction in the reported table arithmetic or source fingerprint. Two study-design corrections and two precision refinements were requested, then incorporated by C. F directly reread the revised affected passages and verified all four changes. Revised C SHA256: `0d130eb77f12fa5973afecdd14afb5ebba0f4f6421e6a597a0a887c4a8355490`. No unresolved latest-v2 factual contradiction remains in this review. This is a review of evidence bounds, not patient-run verification or consensus validation of the proposed method.

## Required substantive corrections

1. **Same actual pseudo candidates are a frozen-replay condition, not a whole-training matching condition.** Initial C line153 requests identical pseudo candidates across methods. The EMA teacher evolves from each method's committed student (`engine.py:174-175,201-202` at the experiment SHA), so q, accepted pixels, class populations and directions become treatment-dependent after trajectories diverge. Match candidate-generation policy, threshold, accepted denominator, initial teacher state, U data and augmentation schedules. Actual q/d_j can be shared when comparing alternatives at one frozen transaction reference. A forced shared teacher during full training would be another experimental factor and must not enter silently.

2. **Add aggregate scalar allocation with the same finite guard as the parsimonious primary control.** Initial C line153 prioritizes joint versus plain block and supervised, then independent per-class scalars. This can detect a total guarded-policy effect and compare independent versus coupled allocation, but cannot rule out a gain explained by uniform shrinkage plus the shared finite guard. The scalar restriction alpha=a·1 with ridge m·lambda is the exact restriction of the joint regularized objective along the aggregate direction. This is currently unavailable and needs explicit implementation/parity qualification. It does not invalidate C's observed negative block result; it narrows what a future improvement could mean. E's proposed S/J contrast addresses this.

## Precision refinements

- **Per-frame metrics are volumetric.** Initial C line54 correctly describes the frame-mean hierarchy, but should explicitly note `evaluation.py:72-74` pools Z,H,W for each frame. The distinction is averaging frame Dice ratios versus pooling ED+ES frames or patients, not 2D versus 3D Dice. Reference-absent frames are deliberately omitted for that class; gt=0,pred>0 has ordinary Dice zero, so null is an endpoint convention. C already describes omission rather than mathematical undefinedness.
- **No tracked TEST result, not proof about every external file.** Initial C line131 should state that no tracked TEST endpoint/inference result exists and the execution policy cancels TEST. C did not inspect the external runtime, so absolute absence outside Git cannot be established.

## Independent checks and surviving limits

- F recomputed the exact source algorithm from immutable Git blobs and obtained `6e12885e44b725c132a01437dc6179cbf524b94c51dd5a5f5254701eb78cf442`, agreeing with C and all four retained measured receipts. The fingerprint covers the package/train subset only. It does not validate checkpoint bytes, effective external config, dependencies, patient data or GPU execution.
- F independently reaggregated all foreground patient Dice rows and exactly recovered the four classwise headline vectors and derived foreground means. `evidence/F_probe_results.json` records this. It corroborates arithmetic on the same saved data, not an independent replication of the patient experiment.
- C appropriately separates stored mean-frame Dice from metrics derived from pooled confusion. F additionally reconstructed an alternative pooled-frame-per-patient endpoint from saved TP/FP/FN (`evidence/F_pooled_patient_sensitivity.json`): supervised76.5916%, extra77.7334%, full83.0635%, block74.7268%. The rank is unchanged. This sensitivity result neither replaces the primary endpoint nor establishes how many frame/class entries were omitted.
- C correctly retains the harmful/mixed block outcome, limits full-EMA attribution because scope/optimizer/LR/clipping/state differ, and rejects optimized-latency claims from instrumented block timings.
- C correctly distinguishes reported 72-step guard execution from absent full-guard segmentation efficacy. Calling the guard never run would be false; claiming a completed quality trial would also be false. The pilot uses eight warmup steps and is not an estimate of full-protocol acceptance after 1,000 warmup steps.
- C's same-data Brier/Dice discussion is not used to claim patient failure. F's separate CPU test proves only that the finite guard's actual risk predicate does not imply hard Dice safety, even on observed audit pixels. The constructed direction is not evidence that the current EMA generator frequently realizes that case.
- C's phase-lock description is correct for L8/U72. Main and B52 paths have the same relationship; B52 repeats one patient per role. The audit excludes the paired supervised patient, so each U has one permanently ineligible L auditor. This is support design, not proven leakage or efficacy harm.
- Raw full METRICS, probabilities, checkpoints and steps are missing from the tracked branch. A hash of a missing artifact or a figure repeating it is not artifact verification. C maintains this distinction.

## Coordinator brief challenge

The initial brief's unrun-full-guard, phase-lock and manufactured Brier counterexample boundaries were sound. I requested replacing ambiguous “valid frames” with “frames containing GT for the class,” and replacing “absent-class Dice is undefined” with the explicit evaluator-null policy. Also specify that each frame's Dice pools its 3D volume. The rank-changing effect of absent-frame omission has not been established; F's alternate pooled-per-patient reaggregation preserves the rank.
