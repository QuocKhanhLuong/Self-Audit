# Actual compact checkpoints: S0, S1, S2

Snapshot captured on the RTX PRO 4000 server at **2026-11-02 14:54:27 UTC**.
All three arms completed **10,000 optimizer steps**. This publication contains
real PyTorch weights, config, provenance, compressed training logs, and the
existing VAL18 evaluation of each selected checkpoint.

**Final evaluation is pending.** `best.pt` means the highest VAL Macro4 among
checkpoints already evaluated at snapshot time. Step 10,000 VAL and TEST50 are
still gated by the active backup/evaluation queue. These are provisional best
selections, not a completed final comparison. No TEST result is claimed here.
The original server status receipts remain unchanged (`UPLOAD_PENDING`).
Git publication does not certify Drive backup.

| Arm | Method | Crop | Training steps | Selected step | VAL Macro4 | VAL foreground Dice |
|---|---|---:|---:|---:|---:|---:|
| S0 | Supervised CE + foreground Dice | 160 | 10,000 | 8,000 | 0.7652113356890722 | 0.6898516386728967 |
| S1 | Plain fixed-teacher KD | 160 | 10,000 | 9,000 | 0.7701132218223803 | 0.6963755574253855 |
| S2 | Plain fixed-teacher KD | 256 | 10,000 | 5,000 | 0.7184355177224724 | 0.6278563947428106 |

These results do not meet the Macro4 >= 0.87 deployment gate. They do not establish
an inference speed advantage, audit benefit, or joint-dataset performance.
The unfinished ACDC+M&Ms arm is not represented as completed here.

## Contents and provenance

Each arm contains:

- `best.pt`: the actual raw compact student export payload (`{"student": state_dict}`).
- `export/MANIFEST.json` and `export/weights.pt`: the unchanged standalone export.
  `weights.pt` and `best.pt` have identical bytes; Git stores their identical blob once.
- `selected_val/METRICS.json`: existing actual patient/class metrics, coverage and
  confusion counts. `PREDICTIONS_FROZEN.json` binds them to the predictions scored
  on the server. The prediction volumes and patient images are not in this bundle.
- Locked recipe/config, `RUN_LOCK.json`, `PROVENANCE.json`, `BEST.json`,
  `BEST_SELECTION.json`, validation log, final training report and source manifests.
- `segments/*/TRAIN_LOG.jsonl.gz`: byte-preserving compressed training logs. The
  original and archive hashes are recorded in `PUBLICATION.json`. S0's initial
  failure and subsequent exact resume are retained in its segment records.
- `FULL_RESUME_CHECKPOINT.json`: location and hashes of the full final optimizer/RNG
  checkpoint retained on the server. `best.pt` itself is for inference, not exact
  training resume.

Training source commits:

- S0: `ac66aef38cc67d520762f7b9a71427ffe961770b` (source/resume details in receipts).
- S1/S2: `543c968504e0ddda8a14813a11b790b0a0f266a5`.

Common compact architecture: residual 2.5D U-Net, widths `[32,64,128,256]`,
GroupNorm, 2,410,596 parameters, random initialization seed 42. Training uses
labeled batch 32; KD arms additionally use unlabeled batch 32 and 1,000 supervised
warm-up steps within the 10,000-step total. Exact settings are in each locked config.
The TRAIN manifest SHA256 is
`e64f3b4009e7250632491ac634e15657692b58d38b97a0a2d975396865e80de4`.
L8 is patient007/010/011/024/026/041/077/099; U72 supplies images only.

S1/S2 use the **new B8 reproduction**, raw student SHA256
`d76d693ecb974bcd2c83964491e8f59b4e0d0f5d55c659f80e72618a4b1d1130`.
It is not the unavailable historical `0ae593ec...` teacher. The local teacher
receipt preserves the actual quality/lineage evidence and pending Drive backup.
The teacher is not included in or needed by these compact inference exports.

## Verify and load

From the repository root, with its PyTorch/NumPy environment:

```bash
python artifacts/server_compact_20261102/verify_exports.py
```

This verifies server-snapshot file hashes, gzip round trips, standalone loading,
finite outputs, and exact logits/prediction parity with directly loaded `best.pt`
on a CPU fixture. It does not run patient evaluation or establish patient quality.
`verification/` records the actual checks executed before publication.

Example standalone S1 loading:

```python
from self_audit.compact_artifacts import load_compact_export

model, metadata = load_compact_export(
    "artifacts/server_compact_20261102/S1/export",
    "179990dcb2eb086cb3e01f42f03d2a727593dfe909ef416d0c6e10454fde869e",
    device="cpu",
)
# model expects B x 3 x H x W and returns B x 4 x H x W logits.
# Use metadata['preprocessing'] and metadata['class_map'] unchanged.
```

The export loader needs the repo's pinned `compact_model.py` plus PyTorch/NumPy.
It does not need teacher, pretrained DINO, support masks, or pseudo-cache assets.

Native inference on the existing server VAL18 image manifest, with a NEW output
directory (do not reuse a directory being written by the active evaluation job):

```bash
python scripts/infer_compact_student.py \
  --export artifacts/server_compact_20261102/S1/export \
  --export-sha256 179990dcb2eb086cb3e01f42f03d2a727593dfe909ef416d0c6e10454fde869e \
  --manifest /workspace/selfaudit-20261101/prepared/acdc_l8_s42_fresh/val18_fresh_native.json \
  --manifest-sha256 070fbfd2cf9f25689fd4206ad07ee33003a3b7bd5a6207d47f976a22fadd067a \
  --out /workspace/selfaudit-20261101/publication_s1_val_predictions \
  --device cuda --batch-size 8 --normalization prepared
```

This command is documented for reproduction, not launched by this publication.
All SHA256 pins for S0/S2 are in `PUBLICATION.json`; use the matching export hash
when switching arms. The native inference workload remains batch 8, independently
of training batch 32.

## Server locations

```text
S0 /workspace/selfaudit-20261101/runs/s0_acdc_l8_b32_seed42
S1 /workspace/selfaudit-20261101/kd_local_20261102/arms/S1/run
S2 /workspace/selfaudit-20261101/kd_local_20261102/arms/S2/run
snapshot /workspace/selfaudit-20261101/kd_local_20261102/git_publication_20261102T145427Z
```

Existing full checkpoints, logs, backups and jobs were retained. No new training
run, extra seed, hyperparameter change or TEST-based selection was performed.
