# SGSCN native baseline (GPL-3.0)

This isolated package preserves pinned official behavior, including the signed
context-loss arithmetic. Read PROTOCOL_LOCK.md and PATCHES.md before running.

The PH2/SYSU-US paper profiles intentionally return BLOCKED_PROTOCOL.
Executable *_official_reference profiles support local verification only;
they do not establish paper reproduction or select an ACDC protocol.

From repository root, with Python 3.12 and this package's environment:
~~~
python baseline/SGSCN/scripts/run_native.py --config baseline/SGSCN/config/native/ph2_paper.yaml --check-protocol
python baseline/SGSCN/scripts/run_native.py --config baseline/SGSCN/config/native/ph2_official_reference.yaml --images-manifest /path/to/image_inventory.json --output /new/output/raw --seed 1 --device cpu --threads 2
python scripts/verify_native_raw.py /new/output/raw
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

Producer isolation: supply image-only mounts and a writable empty output root;
do not mount GT. Python audit/stat guards trace approved image reads and deny
unexpected access, including GT probes. They supplement, rather than replace,
OS mount isolation. Only the separate evaluator may receive GT after raw sealing.

Local synthetic smoke labels its inventory/profile binding PH2 or SYSU-US for
I/O verification, not a real dataset cohort or a native score. Do not count
these artifacts as native reproduction.

A failed/nonfinite original loss leaves an incomplete run, never an invented
numerical fix or RAW_COMPLETE. min_labels is a stopping trigger, not a class floor.
Native Track B is gated until exact paper evaluation dependencies are resolved.
Native Track A is independently BLOCKED_ADAPTER; it does not invalidate faithful
original producer + paper Track B reproduction.

All derived SGSCN code remains GPL-3.0. No common module imports this algorithm.
