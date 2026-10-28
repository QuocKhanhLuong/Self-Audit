# Baseline recovery: evidence and recipe comparison

This delivery is code-only at the user's explicit request. No patient training,
GPU profile, checkpoint-payload recovery, or new patient inference was executed.
Software checks do not establish segmentation quality or justify increasing L.

## Executed sources

The coordinator fetched origin before implementation. Local main remained at
`637f38ced217fe1db2347a87924695533056d8f0` with its existing untracked files.
The fetched report snapshot is `9958d421ccae1da3b8527f74da5438dbb9f3f521`.
That main snapshot does not contain the full pilot implementation. The isolated
implementation branch starts from `93f7c7a1f9606a68955f849dd580b90462ad329e`.

Historical execution used `0f54455b67bb11a72fbfe2a8ac6fae4073f3e9ca`.
Recomputing its source closure from Git objects gave
`3c7fa85f00d54a6e087af3cd6c7045351fb72d532f1069bf09d95f9cd72359b7`,
exactly matching the saved protocol. The PRO4000 source overlay hash recomputed
from the report is
`aedb2cd50c3a59a78be0da30e973de22894f9ccee7ed08613d515166094dc561`.
See `HISTORICAL_EVIDENCE.json` for pinned file hashes.

## What the reports actually measure

| Run | Foreground patient-macro | All classes including BG | Full-model supervised warm-up | Later outer steps | L batch | U batch | Window |
|---|---:|---:|---|---:|---:|---:|---|
| Historical baseline | 0.8318620630 | 0.8729522993 | 1,000, included in total | 9,000 | 8 | 8 | none |
| Historical fixed | 0.8385200133 | 0.8780314990 | 1,000, included in total | 9,000 | 8 | 8 | fixed |
| PRO4000 full EMA equal-update | 0.4814871792 | secondary, not used here | none; 512 head-only updates first | 128 | 1 | 128 | fixed |
| PRO4000 full EMA equal-time | 0.5751228649 | secondary, not used here | none; 512 head-only updates first | 151 | 1 | 128 | fixed |

The historical metrics are from `reports/window_pilot_test50_20261024/RESULTS.json`
and the PRO4000 metrics from `reports/automatic_pilot_pro4000/COMPARISON.csv`,
at the pinned revisions above. All use the exposed ACDC Drive TEST50 cohort as
a development reference; it is not a newly unseen confirmation set, and its
official challenge-label provenance is not independently certified.

The historical primary is a mean over patients of each patient's mean of the
available RV/MYO/LV Dice values. Frame-volume confusion is accumulated before
frame, class and patient reductions. BG-inclusive 0.872952 is not foreground Dice.

## Scope diagnosis from executable code

The canonical historical `engine.train` constructs AdamW over every trainable
non-head parameter and SGD over the head. Supervised CE plus present-foreground
Dice therefore updates encoder, input adapter, decoder (`lateral/up/refine`),
detail, fuse and head. `update_scope=head` controls guarded pseudo directions;
it does not freeze the historical supervised body or the `ema_full` comparator.

The automatic pilot instead warm-starts only the head on a new segmentation
feature path. Its adaptation supervised step selects one support slice; the
`batch_size=128` argument controls U. The full-EMA continuation does train its
non-head parameters, including encoder, despite an earlier runbook description
of a frozen encoder. The source overlay microbatches only that pseudo step.

These differences include training scope, duration, labeled exposure, sampling,
window policy and supervised clipping. The pilot is not a reproduction of the
historical baseline and its result alone cannot establish that low-label L8
training fails. The reports establish a failed head-pilot initialization and
budget combination, without isolating the causal contribution of each factor.

## Locked recovery recipe

Use the no-window baseline recipe, not a mixture of the two historical arms:
L8/U72, label seed 42, 10,000 outer iterations including 1,000 full-model warm-up
iterations; effective supervised batch 8 and U batch 8; crop 256; body LR 3e-5,
head LR 0.003, weight decay 1e-4 on the body, EMA decay 0.99, pseudo threshold 0.7.
The original `pseudo_lr=0.001` is retained in the config; `ema_full` uses the
body/head optimizer learning rates for its pseudo optimizer update.

Labeled foreground slice sampling probability is 0.5 within the already-budgeted
TRAIN pool. Random crop locations use image geometry; horizontal flips align
image, labels and validity; U uses image-only intensity augmentation.
Historical code samples audit and U images even during warm-up to preserve its
RNG schedule, but performs no pseudo optimizer update during warm-up. Logs must
distinguish sampled slices, optimized slices, outer iterations, optimizer calls
and EMA updates. A full 10,000-iteration run can have 10,000 supervised updates
plus up to 9,000 pseudo updates, with 10,000 EMA updates.

The exact historical L8 IDs are patient007, patient010, patient011, patient024,
patient026, patient041, patient077 and patient099. The locked split excludes
patient055 and patient076. The training manifest SHA is
`e64f3b4009e7250632491ac634e15657692b58d38b97a0a2d975396865e80de4`.

DINO source commit is `6876159a11b4df116f30f667f8c9888617df0751` and pretrained
weight SHA is `21b726bb286e037f00a23fb4699fa9bda9c75b6a615bd57ce3013cec1b528d54`.
The historical final student payload SHA is
`b36bcafe9613325d85ee70dabd6a36fef7c17429882967831a151c0cce6d8437` and its manifest
SHA is `03c1730f88941ada898bed7fcf1b9f13cdd8f08bcce4390539fec665c82e97de`.
These are verified declarations in Git artifacts, not a claim that their payload
bytes were available or rehashed during this code-only delivery.

## Gate policy and shared run

First try to recover and validate the historical checkpoint, including inference
in its exact source checkout. Do not retrain to recreate an already valid model.
The implementation witness must show actual nonzero gradients and parameter
deltas for all learned module groups. A fixed tiny TRAIN-L fit is a separate
diagnostic, never the initialization of the shared experiment. Stop on failure.

Choose the final step-10,000 student before training, not the best TEST score.
The development recovery tolerance is foreground patient-macro >= 0.8118620630
(historical minus 0.02), complete coverage, and all foreground classes active.
It does not convert exposed TEST50 into confirmatory evidence. Do not retune the
threshold after observing a new result.

Only a verified baseline gate opens shared training. The shared experiment has
exactly ACDC L4 plus MnM L4, total L8, with deterministic metadata-only selection,
explicit namespace/class mapping and duplicate quarantine. It starts from the
same SSL DINO weights, never from the ACDC-L8 segmentation checkpoint. Keep a
single student and one EMA teacher, the declared 10,000/1,000 schedule and L8/U8
batches; alternate domains 50/50 and log actual patient coverage. Any prepared-L8
candidate-pool restriction must be disclosed in the generated manifest.

Initial shared development thresholds are foreground patient-macro >= 0.75 for
each dataset, each RV/MYO/LV Dice >= 0.60, and complete prediction coverage.
Report each dataset separately, with equal-domain average only supplementary.
The same checkpoint must generate both domains' frozen predictions before
references are scored. This is multi-dataset training, not external-domain
generalization. No gate failure authorizes more labels, a longer schedule, or
an expanded audit matrix.

Missing evidence is BLOCKED; optimizer/loader violations are implementation
errors; an unfinished or failed tiny-fit budget is not convergence; a completed
protocol below its preregistered threshold is empirical NO_GO. None should be
collapsed into a general claim that few-label learning is ineffective.
