# CPU asset staging for the pinned diagnostics

The diagnostic contract is `docs/DIAGNOSTICS_RUNBOOK.md` at
`8dc9e7446be23e2b8533871c2a1e418b4a115da7`. Staging is separate from training.
No trainer, GPU function or A100 launch is added. Data payloads are never routed
through the Mac client. Python source and small verification metadata are the
only client-side artifacts in this workflow.

Use the existing conda environment:

```bash
conda activate selfaudit-modal
python -m modal volume ls selfaudit-assets / --json
python -m modal run scripts/modal_stage_assets.py --action inventory
```

The second command launches a fresh CPU verification container, with no Drive
Secret, reloads `/assets`, reads the persisted receipt, hashes every registered
file and verifies pinned runtime closures. It prints `READY` or `BLOCKED`, missing
assets and persistence results. `/assets/receipts/STAGING.json` is authoritative;
`persistence_verified: true` alone does **not** mean all required assets are ready.

## Security and persistence

The app creates/reuses Volume `selfaudit-assets` mounted at `/assets`.
`stage_assets` requests one CPU and 1024 MiB, with no `gpu` argument. Its image
uses `apt_install("rclone")`. Only this function receives Secret
`selfaudit-rclone`, containing `RCLONE_CONFIG_TEXT` and `RCLONE_REMOTE_NAME`.
The selected remote must match a section in the supplied config.

The config is written under `/tmp`, mode 0600, passed via `rclone --config`, and
removed in `finally`. It is never placed on the Volume or returned to the client.
Child-process environments omit the secret text. Raw rclone errors are withheld;
only fixed error categories are returned. File transfers use `copyto`; there is
no `sync`, `move`, Drive deletion or recursive Drive-root traversal.

`volume.commit()` precedes a separate verification function using a new
single-use container. Container identity is compared using `MODAL_TASK_ID`, not
the shared hostname `modal`. See the official [Volume documentation](https://modal.com/docs/guide/volumes)
and [runtime environment variables](https://modal.com/docs/guide/environment_variables).

## Actual staged subset and blockers

The executed requests and evidence are in `reports/modal_assets_20261101/`.
Confirmed assets include the wide plain-KD export (full weights SHA256
`32a35c1fc3b83d57d083de4ed17ab80bda6ba38a8ebc50133e5aede17ab00380`), its original
manifest, the original B8 manifest, native VAL18 manifest and teacher Python
runtime. The teacher runtime was fetched directly from GitHub to Modal at
`0f54455b67bb11a72fbfe2a8ac6fae4073f3e9ca`; its full native source digest matches
`3c7fa85f00d54a6e087af3cd6c7045351fb72d532f1069bf09d95f9cd72359b7`.

This does not restore the missing teacher weights. The inspected selected
compact backup explicitly excludes its teacher. B52, B8 reproduction and focal
backups have different identities and were not substituted. The clean compact
Git revision in the historical receipt also has a different closure; the receipt
records uncommitted execution-source changes. No historical model load has been
claimed: original DINO source/pretrained dependencies remain unstaged too.

Drive has raw NIfTI ACDC under the observed `Dataset/ACDC/training` directory.
No raw images or masks were downloaded because the exact teacher, compact runtime
and original TRAIN manifest prerequisites are unresolved. No preparation was
executed. Raw files/ZIPs are not prepared data. M&Ms and TEST50 payloads were not
downloaded. Limited directory metadata used to inspect backup identities is not
an evaluation or a dataset transfer.

## Exact recovery inputs from an old server

Run these on the server **holding the historical files**, with authenticated
Modal CLI. Do not route them through the Mac, use a replacement checkpoint, or
copy an unrelated current worktree. These commands are provided, **not executed**.

```bash
sha256sum /workspace/compact-acdc-l8/20261028/b8_recovery/full/final/weights.pt
# Must be 0ae593ecb6b6f9dcb88d0a41d4a77d7b3b3a3edd87396624c3648afb1281f1b6
python -m modal volume put selfaudit-assets \
  /workspace/compact-acdc-l8/20261028/b8_recovery/full/final/weights.pt \
  /b8_recovery/full/final/weights.pt

sha256sum /workspace/Self-Audit/data/prepared/ACDC_L8_S42/train.json
# Must be e64f3b4009e7250632491ac634e15657692b58d38b97a0a2d975396865e80de4
python -m modal volume put selfaudit-assets \
  /workspace/Self-Audit/data/prepared/ACDC_L8_S42/train.json /manifests/train.json
```

Also recover `/workspace/Self-Audit-compact/train.py` and its complete
`self_audit/` package, including the original uncommitted changes. Its source
digest must be `bbfbdab02cf7703904121f39720adcb1dc358b6d76b60fbbcb4366fa30244b3c`.
Recover the manifest-referenced prepared ACDC files and required raw geometry
sources only. Preserve JSON bytes. The historical absolute paths need explicit
container aliases to the matching `/assets` locations; rewriting JSON would
invalidate its hash. The current staging receipt must remain BLOCKED until a
subsequent staging/preparation pass registers and fully verifies these files.
Manual `volume put` does not change a receipt into READY.

If only raw ACDC can be recovered, use the repository's exact historical
preparation pipeline in a CPU container after the missing protocol inputs are
resolved. Verify resulting files and manifest hashes; do not guess serialization,
normalization, label subset or native geometry. No alternate prep path or fake
prepared flag is implemented here. Any future M&Ms staging must separately
verify its archive and extraction before deleting only the temporary Modal ZIP;
the Drive original must remain intact.

## Repeat the bounded staged subset

After inspecting `STAGE_REQUEST.json`, these commands repeat only the already
identified metadata, teacher Python source and pinned compact payload. Existing
different bytes are refused. They cannot silently download the data cohort.

```bash
conda activate selfaudit-modal
python -m modal run scripts/modal_stage_assets.py --action batch \
  --request-file reports/modal_assets_20261101/STAGE_REQUEST.json
python -m modal run scripts/modal_stage_assets.py --action git_metadata \
  --request-file reports/modal_assets_20261101/GIT_METADATA_REQUEST.json
python -m modal run scripts/modal_stage_assets.py --action inventory
```

Bounded discovery takes an explicit JSON request with observed/receipt-backed
paths, `recursive` and `depth`. It lists metadata, never copies directories.
`metadata` permits exact small files; `payload` only accepts the two pinned
weight identities after their original manifests; `runtime` checks every Git
source file and the complete declared source digest. No mode launches a model.

```bash
conda activate selfaudit-modal
python -m pytest -q tests/test_asset_staging.py
python scripts/check_repo.py
python -m pytest -q
```
