# ASTRA / DINOv3 RTX 4070 training results

Published 2026-10-19 from `runs/astra_dinov3_c4973553_4070_20261016_214836`.

Training completed 50 epochs and 1,010,750 optimizer steps, with 314,900 semantic updates. Runtime: 98,250.47 seconds (27.29 hours). Peak PyTorch allocated memory: 2,818,285,056 bytes. Final step training loss: 1.1690037251.

**Validation and test were not run. Segmentation quality is not evaluated.** The native detail branch has no native training objective in this run.

- `summary.json`: completion information and final step metrics.
- `EPOCH_METRICS.csv`: per-epoch means over non-null logged values, with counts for each metric; missing semantic values are excluded.
- `full50/`: configuration, startup provenance, completion report and checkpoint manifests.
- `pilot32/` and `receipts/`: pilot and execution evidence.
- `PLAN.json`: source commit, branch, runbook and backbone provenance.
- `MANIFEST.json`: SHA-256 hashes of copied artifacts.

All 1,010,750 metric rows were parsed and checked against the completion report. The raw metrics hash is recorded in `summary.json`. Raw logs, weights, datasets, dependencies and external repositories remain local. Checkpoint manifests reference local weights, which are not included here.
