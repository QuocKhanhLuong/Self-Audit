# M&Ms archive preparation and audit

## Scope

This workflow audits the three downloaded M&Ms archives for the PG server
queue. It does not extract or alter source archives, open mask payloads, build
training manifests, or start target training. Archive integrity failure aborts
the CLI. `RAW_VERIFIED` means container/member CRC checks and image-header
inspection passed; `GEOMETRY_READY` additionally means that the committed
FUSE source inventory and geometry receipt exactly match all archive image
header hashes and the locked mm/polar policy. Neither status means
`training_ready`; that field remains false.

Official open-release partitions are TRAIN175, VAL34 and TEST136. The source
label mapping is BG/RV/MYO/LV = `{0:0, 1:3, 2:2, 3:1}`, recorded only when the
committed source receipt matches. The FUSE policy preserves each raw header
finding and uses the pinned acquisition paper for a dataset-level mm
interpretation plus bounded polar geometry; it never changes source bytes or
claims that raw headers declared mm. Invalid VAL/TEST geometry blocks future
M&Ms B/B0 use. It does not block the independent ACDC queue.

## Command

From the repository root, with Python, NumPy, nibabel and pytest installed:

```bash
python scripts/pg_audit_mnms_download.py \
  --archives \
    /root/dataset/MNMS/archives/Testing-003.zip \
    /root/dataset/MNMS/archives/Training-001.zip \
    /root/dataset/MNMS/archives/M\&M-20251118T142205Z-1-002.zip \
  --out reports/pg-server-20261116/mnms-audit
```

The archive basenames are required so each is unambiguously assigned to TEST,
TRAIN or VAL. Output is exclusive: use a new `--out` directory for each audit.
The default committed receipts are under
`reports/fuse-mnms-geometry-20261113/`. When the pinned publication file is
available separately, supply `--source-document PATH`; its SHA256 must equal
`c3584856b3933cdeb440d3d2215f1bcdc6d90a7bbe5803bee615e7a823b4c7fd`. The
publication bytes are unavailable in this checkout, so the current workflow
records source-document status and will not apply geometry until those bytes
are supplied and verified. A `SOURCES.json` receipt alone does not substitute
for that SHA check. The expected archive files are also absent from
`/root/dataset/MNMS/archives` in this checkout, so no real archive audit or
patient header result was run here.

Optional receipt overrides are `--source-snapshot PATH` and
`--geometry-summary PATH`. Every ZIP member is streamed to force CRC checking;
image members are then decompressed only through their NIfTI header. No files
are extracted. The source ZIPs remain in place. Runtime is bounded by full
archive read throughput, since verifying CRC for every member requires reading
all compressed members. The CLI hashes each source archive as well.

## Geometry and raw header reporting

Raw findings include units, pixdim, coded sform/qform presence, selected affine
finiteness/nonsingularity, orthogonality and affine/pixdim agreement. Unknown
units and other flags stay visible even if a matching FUSE geometry receipt
allows an explicit dataset-level mm interpretation. No unit is guessed and no
affine is synthesized. Geometry status is reported separately from archive
integrity, and no patient is silently excluded.

The CLI verifies identity using patient ID, split, member path and the raw
348-byte header SHA from the existing immutable snapshot. The committed
snapshot currently binds TEST to archive SHA
`941296c7a357cf1bc3b61225541438ecd1a3b6c1de48caa43c8f973d4f1d099e`, but its
TRAIN/VAL source provenance is from a prior preparation inventory. Therefore
all three downloaded archives must match the snapshot's per-image header
digests before the combined geometry receipt can be reused. A release/member
layout or image-header mismatch keeps the outcome at `RAW_VERIFIED` and requires
a separately governed refreshed source receipt.

When the paper is supplied, the CLI calls the existing FUSE
`adjudicate_geometry` helper for every raw header, using only the committed
policy and source identities, then checks the resulting geometry against the
per-patient FUSE receipt. It marks `GEOMETRY_READY` only if all 345 rows agree.
Without the paper, raw integrity/header checks can still finish as
`RAW_VERIFIED`, with geometry not applied.

The organizer evidence in the existing FUSE report says source labels 1=LV,
2=MYO, 3=RV. Preserve mask dtype and NIfTI scaling if masks are inspected in a
later authorized step: casting to `uint8` before handling non-unit
`scl_slope`/`scl_inter` can alter fractional source values and silently change
label IDs. This audit does not inspect masks or infer training targets.

## Verification

Focused synthetic tests cover path traversal, unknown-unit/header findings,
partition counts, CRC failures and separation of raw/geometry readiness:

```bash
python -m pytest -q tests/test_candidate_pg_audit_mnms_download.py
```

These tests are software-only; they do not verify patient archive contents or
training readiness.
