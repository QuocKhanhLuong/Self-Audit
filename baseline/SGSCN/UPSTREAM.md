# SGSCN upstream

URL: https://github.com/osmond332/Spatial_Guided_Self_Supervised_Clustering
Commit: 592efb6e72ceeef15c8be0630a4673eda5dce6f5. License: GPL-3.0.

upstream/demo_final.py, upstream/src/center.py, upstream/README.md and LICENSE
are preserved byte-for-byte; hashes are in upstream/receipt.json. No data/image,
pyc, git metadata or binary is vendored. Derived modules remain inside SGSCN
and are licensed GPL-3.0. Common benchmark modules never import SGSCN.

The original demo is an audit/parity reference, not a safe producer entrypoint.
Use scripts/run_native.py, which accepts an explicit image-only inventory,
never probes GT, writes integer raw maps outside the input tree, and seals runs.
