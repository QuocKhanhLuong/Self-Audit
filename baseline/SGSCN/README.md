# SGSCN native baseline (GPL-3.0)

This isolated package preserves pinned official behavior, including the signed
context-loss arithmetic. Read PROTOCOL_LOCK.md and PATCHES.md before running.

The PH2/SYSU-US paper profiles intentionally return BLOCKED_PROTOCOL.
Executable *_official_reference profiles support local verification only;
they do not establish paper reproduction or select an ACDC protocol.

From repository root, in the canonical environment (environments/self-audit-canonical):
~~~
python baseline/SGSCN/scripts/run_native.py --config baseline/SGSCN/config/native/ph2_paper.yaml --check-protocol
python baseline/SGSCN/scripts/run_native.py --config baseline/SGSCN/config/native/ph2_official_reference.yaml --images-manifest /path/to/image_inventory.json --image-root /path/to/image_only_staging --output /new/output/raw --seed 1 --device cpu --threads 2
python scripts/verify_native_raw.py /new/output/raw
python scripts/verify_native_raw.py /new/output/raw --write-receipt /receipts/raw_receipt.json
python scripts/verify_native_raw.py /new/output/raw --receipt /receipts/raw_receipt.json
python scripts/verify_native_raw.py /new/output/raw --repeat /other/frozen/raw
python -m pytest baseline/SGSCN/tests tests/native_baselines
~~~

The image-only inventory has exactly schema, dataset and records:
~~~
{
  "schema": "medical-native.image-only.v1",
  "dataset": "PH2",
  "records": [
    {"sample_id": "IMD017", "role": "image", "image_path": "/images/IMD017.bmp",
     "image_sha256": "<64 lowercase hex characters>"}
  ]
}
~~~
SYSU-US uses dataset="SYSU-US" and sysu_us_official_reference.yaml.
Inventory paths may be relative to its file. No GT fields, globbing or annotation
existence filtering. Reject links and annotation paths before reading images.
Original image preparation/inventory evidence is still required.

Producer isolation: --image-root must be an image-only staging directory that
holds exactly the inventory images (plus, optionally, the inventory file). Copy
images out of native layouts such as PH2, where the lesion GT is a sibling
directory; a native dataset root is refused before any output is created. Supply
a writable empty output root outside the staging root and do not mount GT.
The Python guard fails closed: only the listed images and readable files may be
opened or stat'ed; listing/globbing is allowed only in code and output trees,
never in or above the staging root; a denial swallowed by a caller (for example
os.path.exists) still fails the run. It supplements, rather than replaces, OS
mount isolation.

Before any evaluation, finalize the verified raw seal in a write-once receipt
stored outside the raw root and commit/archive it. Evaluators verify that
external receipt and the raw seal before protocol or GT access; a modified and
resealed raw run no longer matches its receipt.

Local synthetic smoke labels its inventory/profile binding PH2 or SYSU-US for
I/O verification, not a real dataset cohort or a native score. Do not count
these artifacts as native reproduction.

A failed/nonfinite original loss leaves an incomplete run, never an invented
numerical fix or RAW_COMPLETE. min_labels is a stopping trigger, not a class floor.
Native Track B is gated until exact paper evaluation dependencies are resolved.
Native Track A is independently BLOCKED_ADAPTER; it does not invalidate faithful
original producer + paper Track B reproduction.

All derived SGSCN code remains GPL-3.0. No common module imports this algorithm.
