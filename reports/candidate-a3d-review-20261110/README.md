# Candidate A / A-3D review receipt

This directory records the final software and synthetic checks for the current
Candidate A (2-D) and Candidate A-3D implementation.  It is an evidence
receipt, not a clinical result or a claim of SOTA, 96 Dice, novelty, or teacher
quality.

## What was verified

- `scripts/check_repo.py`, the full test suite, the focused Ruff unused/undefined
  symbol check, and `git diff --cached --check` passed.  The final suite had
  **117 tests**; the exact commands and source-file SHA-256 inventory are in
  [`evidence/CHECKS.json`](evidence/CHECKS.json).
- The ablation harness completed **88 CPU synthetic jobs**: A has 19 arms plus
  two shared teacher stages for each of ACDC and M&Ms profiles (21 per profile);
  A-3D has 21 arms plus the same two stages (23 per profile).  All jobs report
  `COMPLETED`; the matrix receipt checks four-class per-epoch metrics, pooled
  confusion counts, separate macro-patient fields, code budget, and unique
  query IDs.  See [`evidence/MATRIX_CHECKS.json`](evidence/MATRIX_CHECKS.json).
- Both tracks completed synthetic teacher pretrain, teacher warm-up, joint
  training, export, and evaluation.  Export/load parity is checked in the two
  smoke receipts.  The tracked representative metric files include per-class
  Dice, F1, recall, and precision fields:
  [`evidence/metrics/a2d_joint_metrics.csv`](evidence/metrics/a2d_joint_metrics.csv)
  and [`evidence/metrics/a3d_joint_metrics.csv`](evidence/metrics/a3d_joint_metrics.csv).
- Student-only A-3D CLI inference ran from an image, explicit full-volume
  geometry, and the exported student checkpoint.  No label or teacher argument
  was supplied; output was `[4, 8, 16, 16]` and serialized-code round-trip
  parity passed.  See [`evidence/CLI_CHECK.json`](evidence/CLI_CHECK.json).
- The exported student manifests for both tracks are retained as
  [`evidence/a2d_student_MANIFEST.json`](evidence/a2d_student_MANIFEST.json)
  and [`evidence/a3d_student_MANIFEST.json`](evidence/a3d_student_MANIFEST.json).
  Both carry the same 32-file framework source inventory and framework
  `source_sha256` `f9ad328fd27add333ea296edeb482e0279e752fc85faa334e312c43503573c67`,
  while retaining their own data and payload hashes.  The tiny raw logs are
  [`evidence/pytest.txt`](evidence/pytest.txt),
  [`evidence/check_repo.txt`](evidence/check_repo.txt), and
  [`evidence/ruff.txt`](evidence/ruff.txt).
- The optional official MedNeXt source check ran on CPU against upstream commit
  `0b78ed869fbd1cc2fd38754d2f8519f1b72d43ba`.  It used random initialization,
  verified exact region-path parity, replicate padding from `16^3` to `32^3`,
  and finite checkpointed backward.  This is an adapter/software check, not a
  pretrained-quality result.  See [`evidence/mednext_cpu.json`](evidence/mednext_cpu.json).
- The TRAIN-GT code diagnostic uses privileged synthetic labels only to measure
  actual serialized packets at fixed bank prefixes.  It is explicitly not
  deployment, efficacy, or an upper bound.  See
  [`evidence/DIAGNOSTIC.json`](evidence/DIAGNOSTIC.json).

## Scope and limitations

No patient-data training or evaluation, GPU run, pretrained MedNeXt weights,
RefineFormer verification, or real ACDC/M&Ms result was available in this
receipt.  The A-3D protocol is native full-volume: no crop, resampling, hidden
patch coordinate, or augmentation is introduced.  The MedNeXt channel
semantics remain metadata-bound and require a reviewed checkpoint lineage and
class mapping before any real experiment.  Synthetic metric values must not be
reported as patient performance.

The implementation review checklist and file/line evidence are maintained in
[`REVIEW.md`](REVIEW.md).  The source inventory in
`evidence/CHECKS.json` and `evidence/MATRIX_CHECKS.json` is identical.  The
official MedNeXt adapter receipt and both exported student manifests carry
that same 32-file inventory; equality was checked before this report was
copied.  Its framework source hash is
`f9ad328fd27add333ea296edeb482e0279e752fc85faa334e312c43503573c67`, and its
canonical mapping hash is
`0966cc6be0b74db068e1d9225f381d812d7ba92972d7f5db3e527977c8c93a96`.

Evidence files are copied from the immutable run directory
`runs/candidate-a3d-verified-20261110`; model weights and fixture arrays are
intentionally not copied into this tracked report.

Tracked CSV copies normalize CRLF to LF without changing parsed records.
`RECEIPT.json` records both byte hashes; original run files remain untouched.
