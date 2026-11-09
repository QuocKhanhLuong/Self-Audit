# Active method: Candidate A

`train.py` is the only training entry point. Its versioned recipe is
`self_audit.candidate_a.v2`. The old joint_guard/EMA optimizer is retired from the
active tree. Current implementation details are in [CANDIDATE_A.md](CANDIDATE_A.md).

1. Both teacher and student receive the same normalized three-slice MRI context.
2. Shared teacher features feed separate broad-context region and native-detail
   boundary branches. Teacher losses use budgeted TRAIN segmentation and SDF targets.
3. The fixed bank identifies measurements by type, class, location and scale.
   Student values, finite-level quantization and actual packet bytes form the code.
4. K1 proposes a bounded utility-trial set. Utilities use the same decoder state,
   quantized student values and TRAIN segmentation improvement minus added bits.
   Detached utilities teach the candidate-renormalized policy. Actual actions
   use the entire feasible bank plus STOP.
5. The decoder has no image-feature bypass. Student segmentation gradients reach
   its encoder/value head through STE code values; discrete query IDs are not
   differentiated. Teacher parameters receive no student gradients.

An epoch traverses every labeled TRAIN frame/slice once, with a reproducible
shuffled order. TRAIN metrics aggregate online predictions. Development metrics
are separately frozen/scored at the epoch boundary. Neither validation nor test
labels create targets, select a checkpoint, tune a gate or stop training.
Final-epoch selection is fixed. Exposed development remains development evidence.

Source and weight hashes bind checkpoints to their implementation. All teacher
stage labels count in the same exact hashed TRAIN manifest. ACDC/M&Ms labels are
normalized explicitly to BG=0, RV=1, MYO=2, LV=3. Distances are 2D physical in-plane
cell-interface distances; neighboring input slices do not imply a 3D SDF.
