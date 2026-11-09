# Candidate A: typed measurement code

This opt-in implementation follows A1–A3 of the supplied design. It is a new experimental method behind `train.py`, selected by `method: candidate_a`. The default native v2 `joint_guard` trainer and its transaction/EMA semantics remain separate. Candidate A currently uses the declared dense labeled TRAIN subset; U images are recorded for split integrity but are **unused**. It is neither no-GT training nor a validated semi-supervised improvement.

The teacher proposes useful measurement IDs. It does not supply code values or fused feature tensors to the student decoder. No result here establishes novelty, patient accuracy, SOTA, 96 Dice, or clinical usefulness.

## Inputs and geometry

The existing cache is normalized `[T,Z,Y,X]`. Both teacher and student receive the **same** `[1,3,H,W]` context (`z-1,z,z+1`, replicated at volume ends), and predict the central `[1,H,W]` mask. Labels retain `0=BG, 1=RV, 2=MYO, 3=LV`. The native aligned horizontal flip is applied jointly to image and label; positive in-plane axis lengths remain unchanged. There is no resize/resampling or label-dependent query placement.

`prepare_acdc.py` now writes a top-level patient geometry map containing native ZYX affine, shape, spacing in mm and class mapping. Old native v2 manifests remain accepted by the original trainer. Candidate A rejects missing geometry, inconsistent class mapping, sheared/non-orthogonal axes and nonpositive spacing. Re-prepare old caches from their original metadata; do not invent unit spacing. M&Ms needs a separately verified adapter to this same manifest/ontology; this delivery does not claim a raw M&Ms importer.

Distances are **2D in-plane**, in mm, matching this repository's slice prediction protocol. They are not 3D SDFs. Foreground masks are unions of pixel cells. Distance is to the nearest observed interface segment, with half-pixel edges and anisotropic spacing, negative inside and positive outside, divided by fixed `radius_mm` and clipped to `[-1,1]`. An absent class has REGION=0 and BOUNDARY=+1. A class filling the FOV has BOUNDARY=-1 because no observed in-FOV interface exists. The FOV edge is not invented as an anatomical boundary. Query targets reject ignore=255: this reference requires dense labeled slices for measurement supervision, rather than treating unknown voxels as background.

## Representation and gradient paths

The finite bank is reconstructed from `BankConfig`: type, foreground class, normalized grid center and region scale. REGION queries average a fixed FOV fraction; BOUNDARY queries sample the fixed normalized location on the current native grid. IDs and descriptors depend only on configuration. Configuration and query definitions are SHA256 fingerprinted.

The student encoder produces image features for its value head and policy. Its value vector is `[1,Q]`; REGION values are bounded to `[0,1]`, BOUNDARY values to `[-1,1]`. The code consists of selected integer IDs and **student** values. Fixed `baseline_ids` form R0 and consume budget. The decoder takes only IDs, quantized values, output shape, spacing and coordinates. Its attention over measurement tokens is a reference implementation choice, not a novelty claim. No encoder skip or raw-image bypass reaches the decoder.

Uniform finite-level quantization uses `value_bits`. Training uses a straight-through estimator: forward values are exactly dequantized integer levels; backward passes gradients to the bounded value prediction. Hard query choices receive no segmentation gradient. CE + present-foreground Dice trains the decoder and, through nonempty selected code values, the value head and encoder. GT measurement MAE supervises all configured queries; BOUNDARY MAE is divided by its range 2, REGION by 1. Candidate-renormalized policy CE trains selection. STOP may be chosen before adding any adaptive query; decoding still uses R0 (or the learned null token when R0 is explicitly empty).

Packets contain integer value levels, IDs, shape, spacing, count, protocol fields, bank fingerprint and integrity checksum. The v1 format uses a 57-byte header, an 8-byte checksum, a 32-bit ID and `ceil(value_bits/8)` bytes per selected value. Thus total storage is `8 * (65 + N * (4 + ceil(value_bits/8)))` bits; unused bits in each value byte are padding and are still charged. This simple format is not entropy compression. All serialized bytes, including metadata and padding, count toward `bit_budget`. `codec.cost` measures packet size; `incremental_cost` measures the actual padded difference for a new ID. Evaluation reports serialized bytes, not a nominal latent dimension. The decoder reads dequantized levels, including during utility trials and exported inference.

## Teacher and K1

The teacher shares the native hash-pinned backbone, then has separate region and boundary processing. The boundary path preserves spatial detail and predicts segmentation plus normalized per-class SDF. Teacher CE+Dice losses apply to both segmentation branches; SDF MAE uses `lambda_distance`. Only teacher losses update teacher parameters.

The stages are `teacher_pretrain`, `teacher_warmup`, and `joint`. A real run starts from a verified local backbone or compatible segmentation checkpoint; no download or random fallback occurs. Pretraining learns the segmentation branches from the dense TRAIN budget. Warm-up freezes the shared encoder and trains branch parameters. Joint training updates the teacher at `teacher_update_every` cadence, then computes one detached teacher snapshot for that complete student rollout. The `teacher_frozen` control skips teacher updates. Teacher stages must use exactly the same hashed TRAIN manifest and labeled/U cohort; all pretraining labels count in the joint run's declared label budget.

At each visited student state, teacher discrepancy proposes IDs separately by measurement type (normalized by range); student proposals, random exploration and STOP supplement them. Duplicate or unaffordable IDs are removed before selection. For each trial the same decoder parameters and quantized **student** values compute segmentation loss before/after; utility subtracts `cost_lambda * incremental_bits`. STOP utility is exactly zero. Utility targets are detached. Policy probabilities are renormalized over the actually tested candidate subset for its loss. Rollout actions come from the student's policy over the full feasible bank, not the utility oracle. Inference uses the student alone and performs no labeled utility trials.

## Commands

Use the repository environment with `requirements.txt`. All commands below run from the repository root. Outputs must be new folders outside the input data directory. Full patient training was **not run** in this delivery.

First prepare ACDC with the existing `prepare_acdc.py` CLI (`python prepare_acdc.py --help`). Preserve TRAIN/development/test separation and retain the emitted hashes. Set a local model JSON with `backbone=dinov3_convnext_tiny`, local `source_repo`, pinned `source_commit`, `weights`, and `weights_sha256`. Local weights/source are required; none are bundled. `TRAIN_SHA` and checkpoint SHA values below are the SHA256 of their JSON manifests.

```bash
python train.py --config configs/candidate_a/teacher_pretrain.json \
  --model-config /path/to/local_model.json \
  --data /path/to/train.json --data-sha256 "$TRAIN_SHA" \
  --out /path/to/runs/a-pretrain --device cuda

python train.py --config configs/candidate_a/teacher_warmup.json \
  --model-config /path/to/local_model.json \
  --teacher-checkpoint /path/to/runs/a-pretrain/final \
  --teacher-checkpoint-sha256 "$PRETRAIN_SHA" \
  --data /path/to/train.json --data-sha256 "$TRAIN_SHA" \
  --out /path/to/runs/a-warmup --device cuda

python train.py --config configs/candidate_a/full.json \
  --model-config /path/to/local_model.json \
  --teacher-checkpoint /path/to/runs/a-warmup/final \
  --teacher-checkpoint-sha256 "$WARMUP_SHA" \
  --data /path/to/train.json --data-sha256 "$TRAIN_SHA" \
  --out /path/to/runs/a-joint --device cuda

python scripts/export_candidate_a.py --checkpoint /path/to/runs/a-joint/final \
  --checkpoint-sha256 "$JOINT_SHA" --out /path/to/export/student

python evaluate.py --checkpoint /path/to/export/student \
  --checkpoint-sha256 "$EXPORT_SHA" --data /path/to/development.json \
  --data-sha256 "$DEV_SHA" --out /path/to/evaluation/a --device cuda

python scripts/candidate_a_infer.py --checkpoint /path/to/export/student \
  --checkpoint-sha256 "$EXPORT_SHA" --image /path/to/context_3xHxW.npy \
  --spacing-yx "$DY_MM" "$DX_MM" --out /path/to/inference/a --device cuda
```

To initialize from a native v2 segmentation checkpoint, pass `--teacher-checkpoint-kind native_v2` with its folder/hash. Architecture/state shapes, provenance, ontology and the exact TRAIN-manifest hash are checked before loading. An older checkpoint from a geometry-less/different manifest is rejected; it needs a separately verified provenance migration, which is not performed here. Historical frozen-Q/CineMA checkpoints are unsupported. DINO backbone pretraining alone is not pretrained cardiac segmentation; the recorded provenance distinguishes these cases.

Configurations are small reference recipes, not optimized accuracy schedules. Increase `steps` explicitly for experiments; `train.py` requires `--allow-full-run` above 128 steps. Stage transfers initialize teacher weights only; optimizer/RNG resume is not supported. Student export embeds all student weights and reconstructible inference configuration, with no teacher checkpoint or external backbone dependency. Candidate-A loaders also require the pinned source inventory recorded at training; use the matching repository commit.

## Controls and evidence

`configs/candidate_a/` includes full, teacher-free direct measurement learning, fixed/random selection, REGION-only, BOUNDARY-only, frozen teacher, teacher stages and CPU smoke. Teacher-free K1 retains student/random candidates and TRAIN-label utility supervision; this isolates teacher proposals. Fixed/random selection disables policy loss. Budgets, value precision, student widths, input schedules and label cohorts are held fixed in these recipes. REGION/BOUNDARY-only banks change available measurements; actual parameter count, code length, query count and measured time are reported. Equal maximum bit budgets do not imply equal realized code length or equal total teacher-training compute. Record those differences in comparisons.

Evaluation freezes and hashes full-cine predictions before opening held-out reference masks. It reports per-class patient Dice, explicitly 2D physical boundary metrics, actual packet size/query count and measured inference time. Exposed development scores remain development evidence. No inference speed is inferred from parameter count.

Run `python scripts/smoke_candidate_a.py --out /tmp/self-audit-a-smoke` for the complete CPU check. Exact checks and outcomes are recorded in the delivery validation receipt. Synthetic masks test software contracts; they provide no patient/GPU efficacy evidence. No compatible local teacher checkpoint or patient dataset was used for this delivery.

Validation for this delivery is recorded in [VALIDATION.json](../reports/candidate-a-implementation-20261110/VALIDATION.json): repository check PASS; 87 tests passed (one existing PyTorch deprecation warning); CPU synthetic stages/export/evaluation passed. Check the receipt for exact source and synthetic weight hashes.
