# FUSE A semi-supervised TRAIN data build

The new TRAIN v2 manifest was built from the existing full-label FUSE TRAIN
source and the locked Q3D K8 source. The builder used only source JSON metadata
and NumPy image headers; it did not open mask arrays or NIfTI files, copy
payloads, or write into either source directory. The output U records contain
only `id`, image path, and image SHA; the only retained TRAIN mask references
belong to the eight locked L patients.

## Identity and hashes

- Output manifest: `/root/mount_data/prepared/ACDC/fuse-k8-20261113/train.json`
- Output manifest SHA256: `ef7562a251e5c30622ab00dfa46a5b62e0c4552cc0aac06b24caa3e63ff23363`
- Server receipt: `/root/mount_data/prepared/ACDC/fuse-k8-20261113/BUILD_RECEIPT.json`
- Full-label FUSE source SHA256: `3b30c0006218b2aa348730b1667ca2a78fe47742af509871d9a860bb9a8d5b9b`
- Locked K8 source SHA256: `60dd11747570f3ac183e8636050a3ac3d73651cb43b37deba9ff7422b442a433`
- Sorted L-ID SHA256 (SHA256 of compact JSON list): `ee4b4c34f5a4018626ac935c2570a3da01e446a103b98628b71bd5f47e47153c`

The sorted L8 IDs are `patient007`, `patient010`, `patient011`, `patient024`,
`patient026`, `patient041`, `patient077`, and `patient099`. The builder checked
the K8 source IDs, image SHAs, and mask SHAs against the full-label FUSE source
before emitting the subset. It checked the exact observed K8 metadata shape and
its source SHA. The K8 source class map includes background and ignore entries;
the emitted FUSE manifest retains the locked FUSE BG/RV/MYO/LV class map.
The full-label source's original-mask hash inventory was reduced to the 16
selected L source masks, so no non-L TRAIN or evaluation mask source path is
carried in the generated source-mask inventory.

## Counts

| Item | Count |
|---|---:|
| TRAIN patients | 80 |
| L patients | 8 |
| L ED/ES frames | 16 |
| All TRAIN cine frames | 2,148 |
| U patients | 80 |
| U frames | 2,132 |

U includes every cine frame for the 72 non-L patients and every non-ED/ES frame
for L8. Existing normalized image arrays, L8 mask arrays, and geometry remain
referenced at their original paths.

## Software verification

`python -m pytest -q tests/test_fuse_semi_data.py tests/test_fuse_data.py`:
**16 passed**. The synthetic builder-and-loader spy rejected any attempted
non-L TRAIN mask read, verified the v2 L/U frame sets, exercised one 80-patient
sampler epoch with 80 unique U patients, checked locked-ID tampering and source
L consistency, and confirmed v1 synthetic manifest bytes and loader behavior
remain unchanged. `git diff --check` passed.

These are software and manifest-construction checks. No training, GPU, or new
patient-outcome evidence was produced.
