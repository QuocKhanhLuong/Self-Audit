# Completed teacher-memory experiment, 25 October 2026

All seven arms completed 10,000 steps with 1,000 warmup, seed42, L8, batch24, crop256 and FP32. All use the fixed window and full EMA recipe; A0 is the matched fixed-window reference. No encoder freezing, AMP, shortened training, early stopping or TEST-based checkpoint selection was used. The separate earlier window/meta-Dice study remains separately reported.

## Final TEST50 Dice

Dice values below are percentages. Macro4 includes BG and gives each patient equal weight; foreground macro is reported separately. All50 patients contain evaluated foreground classes.

| Arm | Control | BG | RV | MYO | LV | Macro4 | Foreground macro |
|---|---|---:|---:|---:|---:|---:|---:|
| A0 | Fixed window + EMA | 99.6067 | 75.9868 | 80.8233 | 88.8873 | 86.3260 | 81.8991 |
| A1 | External soft targets | 99.5938 | 77.4576 | 79.6848 | 89.7742 | 86.6276 | 82.3055 |
| A2 | Unpaired memory | 99.6123 | 79.2253 | 79.6338 | 88.6652 | 86.7842 | 82.5081 |
| A3 | Paired memory | 99.5873 | 77.2994 | 79.6097 | 89.9228 | 86.6048 | 82.2773 |
| A4 | Shuffled endpoints | 99.5529 | 76.0172 | 79.5271 | 88.8621 | 85.9898 | 81.4688 |
| A5 | Student-only memory | 99.6181 | 77.5580 | 80.5509 | 89.8677 | 86.8987 | 82.6589 |
| A6 | Random vectors | 99.6167 | 76.1340 | 81.2724 | 90.2354 | 86.8146 | 82.5473 |

A5 has the largest observed final TEST Macro4 (86.8987%). This is a descriptive maximum across predeclared final checkpoints, not independent model selection or confirmation. A3 paired memory is lower than A2 unpaired by0.1793 percentage points and lower than A6 random vectors by0.2098 points. It exceeds A4 shuffled by0.6150 points but fails the prespecified requirement to exceed both A2 and A4. These results do not support a pairing-benefit claim. A4 is below the matched A0 reference. All negative results and midpoint outcomes are retained.

## Validation and checkpoint protocol

VAL18 was evaluated at5,000 and10,000 steps for every arm; TEST50 used every prespecified final10,000-step student. All training finished before any of these new evaluations. No early stopping or selected best-VAL checkpoint replaced the final model.

| Arm | VAL Macro4 step5000 | VAL Macro4 final |
|---|---:|---:|
| A0 | 88.2332 | 87.6962 |
| A1 | 88.4654 | 88.3537 |
| A2 | 87.6225 | 88.5749 |
| A3 | 88.2182 | 88.0261 |
| A4 | 88.6549 | 87.1601 |
| A5 | 87.2551 | 87.2884 |
| A6 | 88.2845 | 87.7221 |

## Actual training time and VRAM

| Arm | Training hours | Peak PyTorch allocated GiB | Sampled device-used GiB |
|---|---:|---:|---:|
| A0 | 2.2833 | 9.3053 | 11.6758 |
| A1 | 2.2574 | 9.3437 | 11.6758 |
| A2 | 2.4315 | 9.4232 | 11.9102 |
| A3 | 2.5884 | 9.3389 | 11.6758 |
| A4 | 2.4497 | 9.3389 | 11.6758 |
| A5 | 2.3778 | 9.3389 | 11.6758 |
| A6 | 2.8481 | 9.3389 | 11.6758 |

PyTorch allocated peaks come from every training step. Device-used peaks and RAM in COMPUTE.csv are sampled from nvidia-smi/cgroup logs at approximately30-second intervals, so they are not certified high-water marks. Evaluation elapsed times total11.2607 minutes across21 evaluations. Training hours exclude bank extraction, diagnostics and evaluation.

## Measured runtime optimization

The original full-transaction sweep selected batch24/threads8 by measured suite throughput, with2GiB spare VRAM plus context allowance. Batch52 failed with CUDA OOM; the negative receipt remains in BENCHMARK.json. The longer B24 canonical probe measured0.8961s/step and26.7821 samples/s; B48 measured1.7791s/step and26.9804 samples/s. At fixed10k updates, B48 doubles exposure and almost doubles wall time. It was not applied.

CPU cache/prefetch input, augmentation and RNG parity passed on real L8 data. The longer prefetch probe reached0.6832s/step (35.1314 samples/s), but student, EMA and optimizer hashes did not match canonical CUDA runs, so cache/prefetch stayed disabled. Two repeated canonical A3 probes also had differing state hashes while RNG hashes matched. The cause is unresolved; this receipt does not attribute the mismatch to caching or establish numerical equivalence. Thread probes1/2/4/8/16/24/32/64 and all negative gates are retained. The measured cgroup CPU-time quota is26.87999 logical-CPU equivalents; affinity exposes224 host logical CPUs.

Teacher extraction selected12 workers from2/4/8/12, with0.6077 timepoints/s on its fixed cost subset. A5 selected12 workers from1/2/4/8/12/16, with0.8123 versus0.1632 timepoints/s for worker1 on the same16-patient/32-timepoint benchmark (~4.98x). Worker16 was slower than12. Selected bank vectors/confidences/validity/endpoint metadata matched the serial A5 benchmark bitwise. Full A5 extraction covered80 TRAIN patients and2,148 timepoints in1,090.3232s (18.17min), with8.0840GiB sampled device-used VRAM and95.0GiB cgroup RAM. Both complete banks and the A1 cache are hash-pinned; no U/VAL/TEST masks built them.

## Verified evidence and source identity

All70,000 training rows and the common patient/frame/slice/crop schedule were verified. The shared schedule SHA256 is `d185ed73928b8925a6c8c7f4630204ef63b94934e3a4741cbe92de0bde0b8866`. Fourteen midpoint/final checkpoint payload hashes were checked. All602 frozen prediction-file hashes and all1,204 labeled-frame confusion matrices were replayed from predictions and hash-checked references without new inference. Patient-level Dice aggregation matches the reports. Shared initialization was reconstructed on CPU from real pinned weights/source and seed42 and matched bitwise across all arms; initial GPU states were not serialized.

Canonical self_audit/ plus train.py SHA256: `7f5856216d59bc787633f8a703fa362cf83582427b830ab1ce09c15aeb55e8a9`. Execution base Git SHA: `8252760092e5dc97e0c441bd2d3d306146dcff64`. DINO source: `6876159a11b4df116f30f667f8c9888617df0751`; weight SHA256: `21b726bb286e037f00a23fb4699fa9bda9c75b6a615bd57ce3013cec1b528d54`. External teacher source: `c10daa1d93f0ea28d8b9ad9206b0f673d25805c1`; weight SHA256: `ee11e5392cc89f0a03f7cd2c171e87de0771b839d414835cb59044077679acb8`. Per-file, config, bank, dataset and checkpoint identities are in ASSET_LINEAGE.json, VERIFIED_EVIDENCE.json and CHECKPOINT_HASHES.csv.

Actual required checks passed: check_repo and pytest166 passed with one existing FutureWarning. Original end-of-run and publication check logs are retained. Compact training logs sample every100 steps plus step1; full step-log hashes and server paths preserve the connection to the complete logs.

## Limits and retained artifacts

This is one seed and one dense TRAIN-patient budget including audit labels, not no-GT training. VAL18 and the reused Drive TEST50 were exposed; TEST references are not independently certified official challenge GT, and external-teacher subject overlap remains unknown. No independent confirmation, repeated-seed significance, HD95/physical boundary metrics or probability calibration is claimed. The prior raw duplicate TEST118/TRAIN055 was handled by excluding055 before TEST exposure; the evaluated TRAIN/VAL manifests contain no overlapping IDs.

Checkpoint/bank tensors, patient arrays and full frozen predictions remain only under `/root/self-audit-memory-full-20261024` and the pinned source data directories. Git contains configs, checkpoint manifests/hashes, complete numeric metrics, frozen-file hash receipts, benchmarks and compact logs. No binary patient data or checkpoint payload is added.
