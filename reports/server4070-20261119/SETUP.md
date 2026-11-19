# Server 4070 setup, Phase S0 (2026-11-19)

Measured facts only; interpretation is marked as such.

## Host
- Host `linhdang-System-Product-Name`, user `aidev` (key-based SSH from the Mac). Workspace `$WS=/home/linhdang/workspace/quockhanh_workspace/SpecMamba`.
- Driver 570.133.07, CUDA 12.8. nvidia-smi GPU 0 = RTX 5070 Ti (16 GB, busy with other users' jobs, never touched), GPU 1 = RTX 4070 (12,282 MiB, idle, 15 MiB used).
- **Device-order deviation from the prompt:** by default torch orders CUDA devices fastest-first, so `CUDA_VISIBLE_DEVICES=1` selected the 5070 Ti (checked). Every shell here also sets `CUDA_DEVICE_ORDER=PCI_BUS_ID`, after which index 1 is the RTX 4070 (11.6 GiB visible to torch, checked).
- RAM 62 GB; `$WS` disk 1.7 TB, 95 % used (about 88 GB free). System python is 3.8 (too old), so the venv is a conda prefix env with Python 3.11 at `$WS/.venv-selfaudit` (no `activate` script: the PATH is set explicitly).

## Folders created (only these; nothing in `$WS` was modified or deleted)
`$WS/Self-Audit` (clone), `$WS/selfaudit_root/{raw,prepared,runs,hf,logs,bundles,receipts}`, `$WS/.venv-selfaudit`, `$WS/Self-Audit/external/MedNeXt` (clone of MIC-DKFZ/MedNeXt at the pinned `0b78ed869fbd1cc2fd38754d2f8519f1b72d43ba`, required by `teacher.source_repo`).
`selfaudit_root/raw/ACDC` is a symlink to `$WS/data/ACDC`.

## Code
- Server clone branch `claude/server4070-20261119`, created from `1d8242a480e93d3c06fbb435804f957899c92818` (A4-SA code of the Modal session). **Code SHA used for the run: 1d8242a.** Not updated mid-run.

## Environment (versions in `selfaudit_root/logs/pip_freeze.txt`)
torch 2.11.0+cu128, torchvision 0.26.0+cu128, numpy 1.26.4, scipy 1.15.2, nibabel 5.3.2, timm 1.0.15, huggingface_hub 2.4.0, MedNeXt 0b78ed8, CineMA c10daa1.
CineMA's install pulled a CPU-index torchvision that did not match torch (`operator torchvision::nms does not exist`); fixed by reinstalling `torchvision==0.26.0` from the cu128 index with `--no-deps`. `pip check`: no broken requirements. `import torch, nnunet_mednext, cinema` passes on the 4070.

## Data verification (see `LOCAL_VERIFICATION.json`)
- SHA256 of every file under `$WS/data/ACDC` (1053 files) compared with `reports/pg-server-20261116/ACDC_FILES.json` (sha256 `01d71e80…`, the inventory the Modal receipt binds): **1053/1053 match, no missing, no extra files.**
- `ACDC_SOURCE.json` sha256 `24a50c93…` equals the `source_inventory_sha256` in the committed download receipt and in the Modal receipt.
- No receipt of this machine's own claims `DOWNLOADED_VERIFIED`. The committed `ACDC_DOWNLOAD.json` (status field verbatim) was copied to `selfaudit_root/receipts/ACDC_DOWNLOAD_server4070.json` with only `target` and `files_receipt` rebound to server paths, the same path rebinding the Modal copy has; the preparer's binding check needs that. `ACDC_FILES.json` and `ACDC_SOURCE.json` are the repo's own files.
- Preparer: `python scripts/pg_prepare_acdc.py --raw-root selfaudit_root/raw/ACDC --raw-inventory reports/pg-server-20261116/ACDC_FILES.json --source-inventory reports/pg-server-20261116/ACDC_SOURCE.json --download-receipt selfaudit_root/receipts/ACDC_DOWNLOAD_server4070.json --out selfaudit_root/prepared/ACDC/cbkd-sa-a4-20261119` finished with status READY.
- Server vs Modal `PREPARATION_RECEIPT.json`: identical `split_lock_sha256` (`cdec2b85…`), counts (train 80, val 18, test 50, labeled masks 16 from 8 patients), split ids, labeled ids, normalization, geometry policy, historical K8 hash. Manifest file hashes differ (train `eda96afa…` vs `44274592…`, val `898a38f9…` vs `65839328…`, test `e218a41b…` vs `75cec96a…`) because the manifests embed absolute paths. Not re-derived beyond that; reported as measured.

## Config (path diff vs the Modal session's `A4_SA_CONFIG.json`)
Generated with `scripts/cbkd_sa_make_config.py` (unchanged code). The only differences are the three `data.manifests.*.{path,sha256}` entries and `data.materialization.{receipt,sha256}`: Modal paths `/__modal/volumes/…` and `/root/mount_data/…` against `$WS/selfaudit_root/prepared/ACDC/cbkd-sa-a4-20261119/…`. Every other field is equal.

## Tests (CPU, `python -m pytest -q`)
- `python scripts/check_repo.py`: PASS.
- pytest: **1223 passed, 7 skipped, 5 failed** (924 s). All 5 fail on `git init -b main` (server git is 2.25.1, which lacks `-b`): four `test_candidate_pg_refresh_main.py` cases and `test_q3d_repository_clean.py` (which runs the same suite in an isolated clone). Environment limitation, no code defect found; tracked tree stayed clean. Not fixed (no code changes on this branch).
- Log: `selfaudit_root/logs/pytest_cpu.log`.

## Bundle
`pg_k8_A1-A4_bundle.zip` is not on the server. Old A2/A4 checkpoints are unavailable here; comparison with the old arms uses reported numbers only (labelled as such) or the Modal session's evaluation.
