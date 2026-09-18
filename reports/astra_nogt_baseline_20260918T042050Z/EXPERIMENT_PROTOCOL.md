# Locked protocol — 2026-09-18, before new results

Status: BASELINE SELECTION RULE LOCKED; no new experiment metrics inspected at lock.
Base: origin/main 826a11ca9c37fd024c9d7d83df594f8b051ec9ca. Existing event
results are development knowledge, not evidence against all reference-free methods.

## Contract and candidates

Method processes cannot open manual reference masks. No supervised/pretrained weights,
GT crop, mask-dependent slice selection, GT cluster matching, GT checkpoint selection.
Use all available ED/ES image slices and the existing patient split (80/20).
These 20 patients are DEVELOPMENT: their labels were used in previous supervised/event
research. We do not claim an untouched test or development that never saw labels.

C0: deterministic four-cluster Lloyd segmentation of image intensity, local mean
(5 pixels) and weak normalized spatial coordinates (0.10 each); 15 iterations,
quantile initialization, deterministic image-only ordering of cluster means.
No over-clustering. Existing frozen maskfree ontology supplies conservative names;
unresolved pixels export UNKNOWN=255 with validity separately. No forced presence,
equal anatomy areas, GT assignment, or invented confident semantic labels.

C1-cache: one small CNN from scratch, trained on fixed cached anonymous C0 partitions
with unweighted cross entropy. One forward followed by the SAME fixed ontology.
No online teacher, second student, critic, recurrent state, per-test-image optimization.
C1-direct control: identical CNN/initialization/data order/updates, soft piecewise
constant region distortion plus 0.01 total variation plus 0.01 conditional entropy.
No marginal class-balancing term. This objective may collapse; report it unchanged.
C2 is NOT admitted unless a useful semantic teacher and nontrivial trajectory exist.

## Data and compute budget

Raw local copy: /tmp/astra_event_acdc_training/training. Inventory only image filenames,
NIfTI headers and image intensities. Verify/hash image provenance. Construct an image-only
input mirror excluding references and log file reads. Existing patient split file is
copied by content/hash; do not regenerate the split. Full field of view, aspect-ratio
preserving resize/pad to 224x224; normalization from image percentiles only; include
apical/basal/empty slices. Native affine/shape retained for inverse export. Resolution
may still lose thin anatomy; quantify native/resized scale factors, do not claim native
quality from resized images.

CPU only while SSH vast-gpu is unavailable; 4 Torch intraop threads, interop 1, loader
workers 0, physical/effective B4 (no accumulation), Adam lr 0.001, no weight decay.
Pilot: at most 8 updates for software/timing only, discarded weights. Main: 1024 updates
(4096 image exposures) per arm, seeds 17 and 29; four sequential runs. Last update is
checkpoint, irrespective of GT or validation score. Evaluate image-only holdout at
steps 0,128,256,512,768,1024; freeze exports at final. No concurrent benchmark.
Wall cap 45 minutes per arm; if reached retain explicit INCOMPLETE result, no assertion
of convergence. No 150 epoch run, rental, job termination or existing checkout changes.

## Selection rule fixed before GT evaluation

This is an engineering feasibility decision, NOT certification of anatomy correctness.
C0 is the fallback experimental baseline. C1-cache can replace C0 only if on development
images it reproduces anonymous partitions at >=0.90 pixel agreement for BOTH seeds,
has at least 0.95 of C0's valid non-BG coverage, and reduces warm end-to-end inference
median (including the fixed resolver) by >=20%. Direct C1 can replace cache only if it
passes the same coverage/latency tests and its independent C0-partition agreement is
no worse by >0.02. Anonymous agreement tests amortization, never correctness.
No architecture/weight/threshold will be changed after GT scoring in this cycle.
If naming fails, retain the selected baseline as PROVISIONAL and explicitly state
anatomical label generation is unsolved; do not add auditor/RNN to conceal failure.

## Falsifiers and measurements

Software: reference access denied, absent/swapped-reference invariance, source-read
ledger, deterministic seed/cache, finite loss/gradients, no test optimization, patient
disjointness, absent classes, native geometry, unknown preserved, independent frozen
evaluation. Synthetic actual-resolver ring-gap test with 4-neighbour exterior path;
then deterministic gap perturbations on real image-only partitions where enclosure
exists. Report denominator, rules_fired, R2 merges, R3 enclosure, unresolved rate,
per-class valid pixels and unresolved reasons. Prior score constant zero is diagnostic
design, not by itself a bug.

Freeze config, source hashes, split, checkpoint, mapping, prediction hashes before
separate reference evaluator. Posthoc development evaluation: per-patient named Dice
(unknown counts as missed anatomy), coverage and conditional quality, class-specific
coverage, anonymous ARI and oracle permutation alignment ANALYSIS ONLY, volume/native
geometry. Oracle mapping never changes exported labels. Perturbations include ID
permutation, dilation/erosion, missing class, all-empty, displacement and smoothing;
stability is not correctness. No slice-independent confidence intervals.

Measure prepare, C0 cache, neural training, image-only validation, naming/native export,
inference; parameters, peak host RAM, device, threads, samples/s and p50/p90 latency.
Neural-only latency is separately reported from end-to-end. Account one-time cache
cost, exposures and optimizer steps. No speedup over maskfree150 without a matched
whole-pipeline run. Convergence remains UNKNOWN if 1024 updates do not stabilize.

Continuation requires semantic coverage and independent correctness evidence, not
low loss or agreement. M&Ms and a final untouched test are NOT RUN in this cycle.
