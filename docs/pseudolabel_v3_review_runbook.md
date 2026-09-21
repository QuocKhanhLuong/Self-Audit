# Self-Audit v3 integrity review and runbook

Audited base: `103ce77ce59cc99c8935dee507dd0ba5f487ec6a`.
This revision changes the scientific recipe and artifact schema; start a fresh run.
Do not reinterpret old freezes or the old C0/C1 results using the new protocol.

## Reproduced defects

The exact original system file was reconstructed and verified against Git blob
`eee050b20e9d0936ef79d686ee74ea839dbedcf3` before an isolated CPU probe.

- A mixed target containing UNKNOWN=255 raises `Target 255 is out of bounds`:
  CE was evaluated before validity indexing.
- A confidently biased random semantic head can accept every pixel without
  any image-only semantic evidence.
- Target-added semantic logits make the seed loss nearly zero for an untrained
  uniform head; raw neural CE is log(4), approximately 1.386294.
- No explicit patient split existed in the previous teacher/student CLI.
- Global random batches stack unlike native shapes; indexing raw Z then loading
  canonically reoriented arrays can change the slice axis.
- M&Ms discovery could open `_gt` files before identifying images.
- Old consistency averaged unregistered pixels and implicitly wrapped the cycle.
- Old CI ran `test_self_audit_pseudolabel_v3.py`, but excluded
  `test_pseudolabel_*.py`. A green old run did not cover the complete v3 suite.

## Repairs and intentional limits

UNKNOWN is masked before CE; empty batches skip AdamW. No foreground seeds means
student training fails clearly instead of writing a misleading trained checkpoint.
Raw neural logits are supervised by detached raw evidence. The compatibility key
`semantic_prob` remains evidence-guided; `raw_semantic_prob` and `semantic_logits`
exclude that evidence, and only the raw logits enter seed CE. Prototype history cannot
create or rename independently unsupported seeds. Training and inference each encode
the teacher once. Dense labels also require class agreement after region mixing.

Patient splits are required. Teacher trains only on train patients; frozen exports may
contain train and val, but student uses only train entries. Each batch stays within one
patient/native shape. Native acquisition axes are preserved, with affine/source hashes
in the freeze. Only usable in-plane orientation is provided to the prior.

Consistency now rejects only, respects validity and uses estimated temporal warp fields
in the production path; there is no implicit periodic wrap. Unregistered cross-slice
voting is disabled. Learned flow and handwritten priors remain unvalidated on real data.
This is NOT a completed full-cycle anatomical reasoning engine or a new accuracy result.

Freeze schema 3 binds prediction contents, identities, cohort, config, split, source
hashes, input images and checkpoint artifacts. Consumers verify before reference access
or student training. This protects against accidental changes, not malicious re-signing.

No scribble reader/input is introduced. Seeds are automatic proposals from image-only
heuristics, NOT expert strokes. Manual scribbles, including scribbles derived from GT
masks, would make a separate weakly supervised setting. No-GT does not mean no prior.

## Install and test

Use an existing compatible PyTorch/CUDA environment; do not replace it with CPU torch.

```bash
python -m pip install numpy scipy nibabel pytest pyyaml tqdm
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest -q \
  tests/test_self_audit_*.py tests/test_pseudolabel_*.py
```

Local isolated CPU regression tests do not substitute for repository CI. CI also runs
native NIfTI fixtures and canonical Dynamic Window tests. Synthetic tests are not ACDC
or M&Ms performance measurements.

## ACDC bounded training + complete export

Use raw full cine, including `patientXXX_4d.nii.gz`; an ED/ES-only processed folder is
insufficient. Full cine training is offline and can be expensive. The max-batch limit
below limits TRAINING, not the complete export of the selected patients.

```bash
export ACDC_ROOT="${ACDC_DATA_ROOT:-$PWD/data/ACDC}"
export RUN="$PWD/runs/pseudolabel_v3/acdc_review_$(date +%Y%m%d_%H%M%S)"
python scripts/train_pseudolabel_v3.py \
  --dataset acdc --root "$ACDC_ROOT" \
  --split-manifest splits/acdc_patient_split_seed42.json \
  --config configs/pseudolabel_v3.json --export-split train,val \
  --out "$RUN" --epochs 1 --max-train-batches 20 \
  --batch-size 1 --threads 4 --device cuda

python scripts/evaluate_pseudolabel_frozen.py \
  --run "$RUN" --references "$ACDC_ROOT" --split val \
  --out "$RUN/evaluation_val.json"
```

Use `--device cpu` explicitly for CPU. Missing declared full-cine patients are errors,
not silently dropped. Held-out development is not an untouched final test; the project
has already used development-label feedback. Every new recipe must disclose that.

## Student: only after inspecting pseudo-label quality

```bash
python scripts/train_student_v3.py \
  --dataset acdc --root "$ACDC_ROOT" --manifest "$RUN/FROZEN.json" \
  --out "$RUN/student_balanced.pt" --profile balanced \
  --epochs 1 --max-train-batches 20 --batch-size 1 --threads 4 --device cuda
```

`NO_FOREGROUND_SEEDS` and `NO_VALID_UPDATES` are valid scientific stop outcomes.
Do not lower thresholds just to make these checks pass. Nonzero seeds do not certify
teacher quality; the training command's coverage guard is NOT a Dice-0.91 gate.
The checkpoint is not a deployment-ready medical model. No UI is added by this patch.

Compact/balanced/accurate invoke 0/1/3 internal Dynamic Window blocks respectively.
Profile names are compute choices, not accuracy guarantees. Grid sampling is still
evaluated at every feature-grid query: this is not sparse hard-region-only execution.

## M&Ms

The loader supports image-only 4D `*_sa.nii.gz` and rejects mask/scribble names before
header reads. Supply a patient-ID train/val manifest matching the actual release.
Multiple scans with ambiguous patient identity fail closed. Evaluation requires an
explicit `--reference-manifest`, e.g. `{"references":[{"patient_id":"ABC","t":0,
"path":"/references/ABC_mask.nii.gz","reference_frame_index":0,
"label_map":{"0":0,"1":3,"2":2,"3":1}}]}`. Verify the mapping and phase indices for
your release; the example is not a silently assumed universal M&Ms schema.

## Unresolved scientific questions

Image/flow/topology/orientation scores remain correlated heuristic evidence, not
calibrated probability of correctness. The raw seed quality, pseudo-label Dice,
coverage, external generalization, GPU memory/latency and ACDC >=0.91 are NOT established
by this audit. This revision fixes software and measurement integrity, not anatomical
identifiability. No long training, threshold selection using GT, or manual-scribble
fallback is performed automatically.
