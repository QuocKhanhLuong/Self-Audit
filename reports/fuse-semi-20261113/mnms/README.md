# Original M&Ms Track B header audit

The current remote original_2020 image set was audited from NIfTI headers only.
No voxel arrays or mask headers/payloads were opened. The CPU audit wrote its
machine-readable receipt to `/root/mount_data/receipts/fuse-mnms-20261113/`.

## Result

- 175 TRAIN images and 34 validation images were found through the staging
  index; TEST is absent and its archive is still opaque/unextracted.
- All 175 TRAIN and all 34 validation headers have unknown spatial units.
  Therefore all TRAIN records are excluded, leaving zero eligible TRAIN
  patients (minimum required: 8), and validation blocks Track B/B0.
  `C8J7L5` additionally fails affine orthogonality; its selected affine is
  recorded verbatim in `per_patient.json`.
- Historical staging receipt listed 34 nonorthogonal cases; current
  re-audit found one in the extracted 34-person validation set. The source
  receipt and present result differ in scope and should not be conflated.
- No geometry repair or unit waiver was applied.

## Provenance and unresolved evidence

The local `index.csv` SHA256 is
`8b285d11e920602b48a1dd87002cc2f360eb6853e66879916c325fd24ba95d7e`.
Staging reports 150 annotated TRAIN, 25 image-only TRAIN, and 34 validation
patients, but does not establish the official Track B train/val/test assignment.
The test archive is `Testing-003.zip`, retained opaque; no official TEST IDs
are available. The local diagnosis metadata is not authoritative label-value
ontology evidence. Explicit source mapping to BG/RV/MYO/LV remains unverified.

Status: **BLOCKED**. Do not create a randomized split, infer labels, or use
unknown-unit geometry.

The reusable summary keeps status BLOCKED while official split and class-map
evidence are unverified, even if a future audit finds otherwise valid headers.
TRAIN header failures are recorded as exclusions and do not independently
block if at least eight valid TRAIN patients remain.
