# Decision: C0, PROVISIONAL experimental baseline

Lock **C0: deterministic image-only partitioning plus the existing explicit,
abstaining anatomy resolver**. No neural encoder, student, auditor, trigger or RNN
belongs to the selected baseline. This is a reproducible negative control for
anatomical-label generation, **not a usable cardiac segmenter or a novel method**.

The selection was recorded before the independent GT evaluator ran. Subsequent
development evaluation found foreground mean Dice **0.01623** on 20 patients.
The baseline is implemented, but anatomical label generation remains unsolved.
Do not start long student training from this teacher or present it as evidence for
the earlier “3 → 2 → 1” ordering. That ordering did not survive its first utility gate.

## What is fixed

| Contract | Locked choice |
|---|---|
| Task | Image-only named cardiac segmentation with explicit abstention |
| Input | Provided ACDC ED/ES images; 100 patients, 200 volumes, every one of 1902 slices; existing 80/20 patient split |
| Resolution | Full FOV resized proportionally to fit 224×224, zero padding excluded from clustering/naming; return to native grid |
| Normalize | Per-volume image percentiles 1/99; no ROI, manual mask, class-dependent slice filtering or atlas |
| Encoder / head / initialization | None. Four Lloyd centers, quantile initialized; image intensity, mean5, 0.10 normalized y/x |
| Optimization | Exactly 15 local Lloyd updates per image, deterministic; no neural test-time optimization |
| Naming | Unchanged `self_audit_maskfree.ontology.resolve_roles`; BG0/RV1/MYO2/LV3, invalid output UNKNOWN255 plus validity map |
| Priors | Border share .10; at most three foreground groups; enclosure .90; adjacency/intensity fallbacks; source and failures in ledger |
| Losses / weights | Squared Euclidean distance in the descriptor above; coordinate coefficients .10. No segmentation, audit or recurrent loss |
| Teacher / history | No teacher at training or inference, no RNN, no hidden state, no second student |
| Schedule / checkpoint | No training checkpoint. Fixed configuration and source hashes; outputs frozen before scoring |
| Inference | One deterministic partition pass (15 iterations), one resolver call per slice, native export; no candidate bank or audit search |
| GT | No masks in current method. Separate evaluator only; historical cohort-collection caveat below |

## Why the neural alternatives were not admitted

The preregistered admission required ≥0.90 anonymous agreement, ≥95% of C0 valid
non-BG coverage, and ≥20% lower warm inference median in **both** seeds. These are
engineering tests, not semantic validation. C1-cache reached agreement 0.8479/0.8389;
seed17 also failed coverage retention. Both were faster, but neither passed the
whole rule. C1-direct collapsed to one anonymous group and all-UNKNOWN named output.
The decision is in `evidence/SELECTION_BEFORE_GT.json`; its timestamp precedes all
five successful evaluation receipts. The protocol hash is unchanged.

Posthoc C1-cache Dice was 0.02146/0.02751, slightly above C0, with 4/20 and 5/20
patients worse than C0. These are very poor absolute results; they do not justify
changing the locked decision. The observed maximum checkpoint scores are **not** selected: final update1024 remains the only checkpoint rule.

The tested direct objective is rejected for this recipe, not all unsupervised CNNs.
The cache arm may be underfit after 4096 exposures (~2.68 dataset equivalents).
Longer training might reproduce C0 better; it has not been shown to fix C0's semantics.

## Which ideas stop here

* **A — GT-free teacher + RNN:** not admitted. There is no independently useful
  correction teacher/trajectory to distill. Neither the failed old event proxy nor
  the current poor anchor supports an additional auditor or memory module.
* **B — scratch lightweight unsupervised annotator:** the direct loss tested here
  collapsed in both seeds. The cached variant amortizes some computation but does
  not solve named anatomy. Keep its implementation as a control, not the baseline.
* **C — clustering to named anatomy:** retained only as the C0 falsification baseline.
  Appearance partitions and their naming both fail; clustering alone is not a
  validated route to a useful anatomical teacher.

## Status and next gate

| Status | Verdict |
|---|---|
| BASELINE LOCKED | YES: recipe/selection rule fixed before new results; no post-GT tuning |
| IMPLEMENTATION VERIFIED | YES within scoped CPU unit tests, native export, actual resolver interventions and real-data runs |
| SCIENTIFIC SUPPORT for useful named segmentation | NO: very poor foreground Dice and unstable naming |
| Novel core method / faster convergence / clinical utility | NOT ESTABLISHED |
| Strict no-GT provenance across historical data acquisition | NOT CERTIFIED; see below |

Measured C0 preparation: 2.795s; anonymous cache for 1902 slices: 82.915s; naming
376 development slices: 9.116s; native export: 1.034s. These stage sums are 95.861s,
excluding startup, diagnostics and evaluation. Warm B1 image-to-named median/p90:
59.58/75.92ms on local CPU4 threads; host peak RSS about 861MiB in anchor process.
GPU, M&Ms, full-pipeline comparison with maskfree150 and converged neural performance
are UNKNOWN/NOT RUN. There is no claim that this replaces a measured whole-pipeline
maskfree benchmark with a speedup.

**One next research problem:** establish image-conditioned anatomical correspondence
and defensible unknown coverage before learning a teacher or recurrent student.
The current metrics do not isolate a single repair: anonymous ARI is low as well as
named Dice. Therefore do not prescribe a hole-closing patch, more clusters, stronger
shape prior or RNN as a proven fix. Any changed design starts a new, explicitly
development-informed protocol; independent semantic evidence must accompany useful
coverage, and a separate final cohort must remain untouched. Stop any new teacher
that merely increases its own image score without improving independently assessed
anatomy. No long run is authorized by this report.

The current method reads image-only inputs, but the old local-copy script chose
images by paired `_gt` filename availability. All 200 images independently match
the 100 Info.cfg ED/ES declarations, and raw hashes match; no crop was found. This
mitigates missing-frame concerns but does not erase indirect collection provenance.
Before a strict no-mask pipeline claim, re-ingest from the original image release
using image/phase metadata only and retain an untouched final cohort. Development
labels were already seen in this project, and are used for the present falsification.

Run the selected baseline end-to-end from this worktree with the command in
[RUNBOOK.md](RUNBOOK.md). Full evidence and limitations: [RESULTS.md](RESULTS.md),
[SUPERVISION_LEDGER.md](SUPERVISION_LEDGER.md), [PRIOR_ART.md](PRIOR_ART.md).
