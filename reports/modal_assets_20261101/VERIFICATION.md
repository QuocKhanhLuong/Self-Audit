# Modal staging execution receipt

Status: **BLOCKED**. CPU staging actually executed; the Volume and its staged
subset are persistent. This is not a ready diagnostic environment.

- Volume: `selfaudit-assets`, mounted `/assets`.
- Final actual app: https://modal.com/apps/quockhanhluong/main/ap-nE9MPrESdwb0CnaSdeyrml
- Receipt: `/assets/receipts/STAGING.json` (exact downloaded metadata copy beside this file).
- Receipt SHA256: `c4ffe94c921820f0bc3c67586f3ad64f5c6ec02fe2359146c1f2fa8c7b22029d`.
- Registered files: **17**, **9825642 bytes**, all read and SHA256-verified
  in a different fresh CPU container.
- Stage container: `ta-01M6B4YAYTM9JDDG5AXWS83DWR`.
- Verify container: `ta-01M6B4YEDWDZ2MP5VDAV308TTR`; no Drive Secret present.
- Temporary config mode: `0600`, outside Volume, removed before verification.
- No GPU allocated. No patient data payloads traversed the Mac. No trainer changed.

## Staged assets

The wide plain-KD weights have full SHA256
`32a35c1fc3b83d57d083de4ed17ab80bda6ba38a8ebc50133e5aede17ab00380`.
The pinned teacher Python runtime has source closure
`3c7fa85f00d54a6e087af3cd6c7045351fb72d532f1069bf09d95f9cd72359b7`.
Original teacher/student manifests and VAL18 manifest bytes are verified too.
See `STAGING.json` for every individual path, source, size and full SHA256.

## Missing prerequisites

1. Historical B8 `weights.pt`, SHA256 `0ae593ec...`, originally
   `/workspace/compact-acdc-l8/20261028/b8_recovery/full/final/weights.pt`.
2. Compact runtime closure `bbfbdab0...`, originally the dirty execution source
   under `/workspace/Self-Audit-compact/`. The clean Git commit does not match.
3. Original TRAIN manifest `e64f3b40...`, originally
   `/workspace/Self-Audit/data/prepared/ACDC_L8_S42/train.json`.
4. The exact manifest-referenced prepared ACDC TRAIN/VAL18 files. Only raw ACDC
   was found in the scoped data inventory. Preparation and large transfer remain
   stopped until the preceding source/protocol prerequisites are resolved.

The original DINO source/pretrained dependencies also remain unstaged, so the
teacher runtime statement concerns Python source integrity, not model loading.
`SOURCE_FINDINGS.json` includes full expected hashes and server-to-Volume targets.
The search was scoped to observed/receipt-backed locations, not the whole Drive.
No B52/retrained/focal checkpoint was substituted. Raw ACDC is not prepared data.
No M&Ms or TEST50 payloads were copied.

## Verification

All final checks ran in conda `selfaudit-modal`:

- Staging tests: **19 passed**.
- Full repository suite: **1064 passed, 2 existing warnings, 153.19s**.
- `python scripts/check_repo.py`: **PASS**.

`VERIFICATION.json` binds source bytes, environment, test logs and the actual
Modal receipt. Tests ran on the listed base plus these new staged-file hashes;
no future commit ID is fabricated. Earlier environment/CLI issues and their
resolution are recorded in `execution_notes`.

## Inventory and recovery commands

```bash
conda activate selfaudit-modal
python -m modal volume ls selfaudit-assets / --json
python -m modal run scripts/modal_stage_assets.py --action inventory
```

The second Python command re-verifies the current persisted subset and updates
the receipt's verification container. It does not load a model or launch A100.
See [the runbook](../../docs/MODAL_ASSET_STAGING.md) for the actual repeated-stage
requests and exact old-server direct upload inputs. No large dataset transfer,
preparation, model inference or training was run during this staging attempt.
