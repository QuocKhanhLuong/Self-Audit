# ACDC L8 compact experiment, 2026-10-28/29

This directory archives the measured ACDC L8 experiment evidence. `RESULTS_INDEX.json` lists every archived file, its original server path, SHA256, and size. Training logs are gzip compressed. The 10,000-step batch-152 log is split into three parts below GitHub's per-file limit; concatenate its numbered parts before decompressing. Patient images, masks, prediction arrays, caches, and model weights are not in Git.

The frozen KD teacher was the recovered **raw student B8** (TEST50 Macro4 0.880049, foreground Dice 0.841210). The B52 raw student scored Macro4 0.861421 and foreground 0.816664 and did not pass the teacher gate. B52 and recovered B8 are distinct runs, not exact reproductions.

All four matched compact arms completed 10,000 steps with the same initialization, ACDC L8 labeled budget, support IDs, and L/U schedule. VAL18 Macro4: supervised 0.730006, plain KD 0.754042, scalar guard 0.729833, joint guard 0.727036. A wider plain KD student at batch 8 reached VAL18 Macro4 0.777413 and TEST50 Macro4 0.759673 (foreground 0.682498). It was selected by VAL18 and exported as the independent student on Drive; its TEST50 evaluation was exposed development evidence, not an untouched confirmation set. The acceptance threshold was Macro4 >= 0.87, so the result is **NO_GO**.

The separate batch-152 width experiment completed 10,000 steps and reached VAL18 Macro4 0.748240, TEST50 Macro4 0.729063, foreground 0.642139. It was a changed batch-size factor and was not selected. Its training peak GPU memory was 21.480 GiB allocated / 23.131 GiB reserved, with 27,912.5 seconds elapsed. The per-volume benchmark compares student and B8 reference on the same GPU, float32, batch 8, and native TEST50 cases. See `experiment_20261028/WIDE_QUALITY_SPEED_MEMORY.json` and `maxbatch_20261029/BATCH152_QUALITY_SPEED_MEMORY.json` for latency, GPU/RAM peaks, and weights-only sizes.

Selected export: `KhanhDrive:Self-Audit-v2-artifacts/compact_ACDC_L8_S42_selected_197dd436_20261029/selected_compact_student/`. `weights.pt` SHA256: `32a35c1fc3b83d57d083de4ed17ab80bda6ba38a8ebc50133e5aede17ab00380`; `MANIFEST.json` SHA256: `197dd4367877c51bc21b91d172f25442549ef9a3d0e422d28118bb07308067de`. See `DRIVE_UPLOAD_RECEIPT.json` for verification.

To restore the split log from `maxbatch_20261029/wide_kd_batch152_full/`:

```sh
cat TRAIN_LOG.jsonl.gz.part* > TRAIN_LOG.jsonl.gz
gzip -t TRAIN_LOG.jsonl.gz
```

The archived receipts, per-arm configs/reports/manifests, evaluator metrics, benchmark reports, source hashes, and test logs provide the detailed evidence. No ACDC+M&Ms joint experiment was run because its label budget and teacher lineage were not locked.
