# Native fixed-window teacher-memory pilot

This opt-in pilot extends `NativeStudent` + fixed window + `ema_full` on branch `feat/native-teacher-memory-pilot`. It starts from window-pilot commit `9341cc81fc046d61135b928f5ac51ebed19c19ed`, which includes the user-checked `0f54455b67bb11a72fbfe2a8ac6fae4073f3e9ca`. Existing baseline/fixed/meta recipes, results and canonical defaults are preserved. The archived CineMA-bank trainers are not imported or resumed.

## Model and controls

The query feature h is the full stride-4 decoder tensor immediately after `refine[0]`. A trainable projection W maps its96 channels to32 normalized query coordinates. The bank contains at most8 normalized endpoint pairs in each canonical group `(BG,RV),(BG,MYO),(BG,LV),(RV,MYO),(RV,LV),(MYO,LV)`. Missing groups stay empty. Each endpoint retains patient, timepoint, native index/world coordinates and confidence. Selection requires confidence>=0.8, physical distance<=3mm and exactly one c-to-d transition along the sampled segment, with no third class; selected candidates are balanced deterministically across patients.

For each pair group, with tau=0.2, `sc=q·uc`, `sd=q·ud`, reliability=`confidence_c*confidence_d`:

```
a_i = softmax_i((sc_i + sd_i)/(2*tau) + log(reliability_i))
r_cd = sum_i a_i * tanh((sc_i-sd_i)/tau)
g_cd = stopgrad(4*p0_c*p0_d)
delta = sum_cd 0.5*sigmoid(alpha_cd)*g_cd*r_cd*(e_c-e_d)
z = z0 + upsample(delta)
```

`p0=softmax(z0)` is padded to the full decoder footprint, then pooled in aligned4x4 cells. Validity is geometric, pooled conservatively and masked again after native cropping. Alpha starts at−4. The residual sums to zero across classes; each class magnitude is bounded by0.5 and initially by approximately0.008993. The bank is persistent frozen buffers; W and six alphas add3078 trainables for the official96-channel feature. These parameters learn only through the existing CE+Dice and EMA pseudo-CE losses. EMA still updates once after the final student state.

| Arm | Implementation |
|---|---|
| A0 | Fixed window + full EMA, no external memory/targets. |
| A1 | Additional external soft pseudo-CE (default coefficient0.25) inside an existing EMA-eligible U optimizer transaction. External-only support does not create another step. |
| A2 | Unpaired class memory using exactly A3's vectors, projection and six alpha parameters. Independently attend to c and d endpoints with logits `s/(2*tau)+log(confidence)`, then average `tanh((sc_i-sd_j)/tau)` over all independent combinations. N² response work is measured, not hidden. |
| A3 | Paired retrieval using the observed correspondence above. |
| A4 | Permute d endpoints, their confidence and metadata across entries within each class pair. Preserve semantic classes and marginal vectors. Groups with fewer than two entries cannot test pairing. |
| A5 | Build from a frozen non-synthetic A0 student, with the same TRAIN manifest, L/U sets and label budget. No external teacher supplies labels, positions, confidence or features. Sources trained with A1–A6 are rejected. Uses the student's32-channel fused features. |
| A6 | Seeded normalized random vectors, retaining bank occupancy and metadata as an explicit control. |

A2/A3 use the same valid-vector budget and trainable capacity. All recipes share batch/update/schedule settings. Profile selection uses measured compute only, never TEST scores. A pairing claim requires A3 to beat both A2 and A4 under a later matched patient study; otherwise retain the negative result and drop that claim. [Prior-art audit](TEACHER_MEMORY_PRIOR_ART.md) establishes substantial overlap with prior prototype/memory/distillation work; novelty is unconfirmed.

## Teacher, geometry and information access

Use only the pinned M&Ms SAX seed0 segmentation checkpoint, not an ACDC substitute. [Provenance](TEACHER_PROVENANCE.md) records source/HF revisions and asset checksums. The teacher is frozen eval/inference-mode with strict state loading and no optimizer. Its32-channel `decoder_dict["sax"]` output is tapped before `pred_head_dict["sax"]`; both direct-head and hook-off logit parity are checked.

Each input is a single native XYZ timepoint. The adapter resamples using the full physical affine to1x1x10mm, tiles192x192x16 with50% overlap, end padding and support tracking, then inverts probabilities/features to native coordinates. NIfTI spatial units are explicitly converted to mm or rejected. The image-only full-FOV percentile domain differs from upstream's GT-LV-centered crop. Upstream M&Ms preprocessing is never called. NaN/Inf, untrusted module roots, bad hashes and unsupported geometry fail closed.

Public lineage reports UKB pretraining and M&Ms fine-tuning. Patient-level cross-dataset overlap remains UNKNOWN; code/weight identity does not prove ACDC was unseen. This delivery supports engineering execution, not a clean-external-teacher or efficacy claim.

The native manifest is derived from the hash-pinned TRAIN image records and explicit NIfTI image paths. No U/VAL/TEST mask is opened. Prepared TZYX image values must match the declared native XYZT image and its canonical image-only normalization. A1 cache writing is opt-in, streamed to disk and requires complete declared TRAIN/timepoint coverage; probability channels and hashed geometric support are explicit. A5 requires matching A0 provenance before loading its model. Bounded banks are marked incomplete and are not presented as the full experimental bank.

## Runtime and deployment

Teacher inference, bank construction and optional A1 target caching happen offline. Online training imports neither the external teacher nor its weights. A1 alone reads the fixed TRAIN target cache. Student checkpoints include bank tensors and metadata; `load_checkpoint` reconstructs the bank from the payload. `scripts/export_teacher_memory_student.py` writes a student-only artifact, and `load_exported_student` verifies and loads it without the external teacher, cache or original bank file. The student's pinned DINO architecture source and current backbone-construction dependency remain explicit; this is not a portable TorchScript/ONNX export.

The seven JSON files in `configs/teacher_memory_pilot/` are guarded templates. A1–A6 require real paths and checksums; placeholder hashes cannot pass validation. The guarded templates retain a conservative batch of 1; the completed Pro4000 cost probe selected a common batch of 32 for the measured smoke/profile. To bind a template, copy it to a new launch directory, set the same `batch_size`, `steps`, `warmup_steps`, seed and crop for every arm, and set `teacher_memory.bank_path/bank_sha256` or `cache_path/cache_sha256`. A2/A3/A4/A6 consume the same base bank path; the loader applies A4/A6 controls deterministically. A5 requires its independently built student bank. Pass the pinned DINO JSON via `train.py --model-config`.

## Server commands for subsequent work

Commands below are **not a launch authorization for long training**. Execute on the separate server worktree `/root/Self-Audit-teacher-memory-pilot` using `/venv/main/bin/python`. The additional CineMA dependencies are isolated at `/root/self-audit-teacher-memory-assets/deps`; set that directory in PYTHONPATH only for offline teacher commands. Existing `/venv/main` packages were not replaced.

Prepare a new complete image-only manifest (omit the smoke limits):

```sh
/venv/main/bin/python scripts/prepare_teacher_images.py \
  --train-manifest /root/self-audit-pilot-artifacts/data/cache/L8_S42/train.json \
  --train-manifest-sha256 e64f3b4009e7250632491ac634e15657692b58d38b97a0a2d975396865e80de4 \
  --native-root /root/self-audit-pilot-artifacts/data/raw/ACDC/training \
  --out /root/self-audit-teacher-memory-execution/native-train-full.json
```

Hash that new file, then build a complete teacher bank (and optionally the complete A1 cache) with `scripts/build_teacher_memory.py --mode teacher --native-manifest <file> --native-manifest-sha256 <sha> --teacher-source /root/self-audit-teacher-memory-assets/CineMA --teacher-config /root/self-audit-teacher-memory-assets/config.yaml --teacher-weights /root/self-audit-teacher-memory-assets/mnms_sax_0.safetensors --device cuda --out <new-directory> [--write-a1-cache]`. Full extraction is **NOT RUN** in this delivery. Never substitute the three-timepoint smoke bank for a complete build without declaring that experimental factor.

Build A5 separately with `--mode a5 --student-checkpoint <matched-A0-folder> --student-checkpoint-manifest-sha256 <sha>` and no teacher arguments. For scientific comparison, freeze the A0 snapshot selection rule before examining outcomes; a two-step smoke checkpoint is only a plumbing witness.

After binding all intended assets and shared budgets, the launch command is:

```sh
/venv/main/bin/python train.py \
  --config /root/self-audit-teacher-memory-execution/launch/A3.json \
  --model-config /root/self-audit-pilot-artifacts/BACKBONE.json \
  --data /root/self-audit-pilot-artifacts/data/cache/L8_S42/train.json \
  --data-sha256 e64f3b4009e7250632491ac634e15657692b58d38b97a0a2d975396865e80de4 \
  --device cuda --out /root/self-audit-teacher-memory-execution/runs/A3-pilot
```

Use the corresponding A0–A6 config and a new output directory for each arm. Runs beyond128 updates additionally require the CLI's explicit `--allow-full-run`; no such run was started. [Delivery report](../reports/teacher_memory_pilot/REPORT.md) lists actual tests, hardware timings, hashes, coverage and NOT RUN items.

## Subsequently authorized full experiment

The user subsequently authorized full A0–A6 training, validation and TEST. The
execution worktree is `/root/Self-Audit-memory-full-20261024`, based on
`8252760092e5dc97e0c441bd2d3d306146dcff64`; receipts are under
`reports/teacher_memory_full_20261024/`. The initial delivery's NOT RUN statements
above describe that delivery and are not completion claims for the new job.

`scripts/run_teacher_memory_full.py --root /root/self-audit-memory-full-20261024
--steps 10000 --warmup-steps 1000` waits for a complete 80-patient/2,148-timepoint
teacher bank and A1 cache, benchmarks real full transactions on A0/A1/A2/A3,
checks dense memory capacity, then selects one shared batch and thread count by
measured suite throughput. Profile threshold0 forces the pseudo path for cost
measurement; every full recipe retains threshold0.7, crop256, L8 and seed42.
A0's final checkpoint is the prespecified A5 source, irrespective of outcomes.
All seven training schedules must match exactly. No exact resume or early
stopping is introduced.

The raw mirror lacks spatial-unit tags in 52/80 TRAIN cines. An explicit
dataset-level mm declaration is recorded against image hashes and the pinned
CineMA ACDC preprocessing convention. `declare_acdc_spatial_units.py` creates
new gzip copies changing only NIfTI-1 header byte123, verifies identical payload
bytes/affine/spacing, and leaves raw files and the strict unknown-unit guard
unchanged. This adjudication is explicit provenance, not independent per-file
clinical geometry certification. No masks or diagnostic fields resolve units.

The complete FP32 A1 cache uses approximately17.6GiB of shared RAM and is
rebuildable; it is not durable across container restart. Banks, midpoint/final
checkpoints and reports remain on disk. Validation uses midpoint and final
checkpoints; TEST uses all seven preselected final students after all training
finishes. `evaluate_frozen_labels.py --compress-predictions` retains full native
uint8 argmax volumes with lossless gzip and hashes before opening references.
Dice is checked against the canonical probability evaluator. Probability
calibration analyses are unavailable from these compressed label outputs.
Reporting includes BG/RV/MYO/LV, macro over all4 and foreground macro separately.
Previously exposed VAL18 and Drive TEST50 are not independent confirmation;
external-teacher subject overlap and official TEST-label provenance remain
unresolved.
