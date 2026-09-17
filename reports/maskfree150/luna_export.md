# W7 export handoff — Luna

Date: 2026-09-15

## Delivered

* `export_prediction` rejects fractional labels, boolean labels, non-finite
  values, negative probabilities and source probabilities whose per-pixel class
  sums differ from one by more than `1e-3`. Source probabilities are never
  silently renormalized.
* Unit and volume paths use W3 `volume_id` (then `acquisition_id`, then the
  legacy `study_id`) together with the method. A patient identifier is never a
  grouping key. Freeze lineage records all role-scoped `partition_ids` and the
  per-unit method coverage map.
* Native NIfTI writes preserve the input affine's spatial units when nibabel
  supports them. Raw header zooms remain separate from converted `spacing_mm`,
  and `spacing_valid` is carried into the sidecar.
* `freeze_predictions` accepts arbitrary checkpoint labels, explicit expected
  methods/unit ids/checkpoints/nuisance files, assembles streamed unit shards by
  volume, and reports method, unit and artifact completeness. The final caller
  must use `validate_freeze(..., require_complete=True)`.
* `verified_freeze_session` performs a complete physical hash pass on entry and
  exit. Inside the context, only `validate_freeze` calls receiving the exact
  yielded mapping can skip the repeated corpus hash; copied/reloaded mappings
  still hash fully. A single full validation reuses each distinct file digest
  across the structured and flat manifest indexes, so each phase is linear in
  the distinct frozen files. Exit detects any in-session mutation.
* The full launcher resolves one inherited numeric or UUID
  `CUDA_VISIBLE_DEVICES` selector against an unmasked physical `nvidia-smi`
  inventory, exports the selected real UUID, records RTX 4070 or an explicit
  rented-GPU override, writes `reports/gpu_identity.json`, and exports
  `CUBLAS_WORKSPACE_CONFIG=:4096:8` before Python starts.
* Completion verification requires the primary and deployment freeze manifests,
  all image-only summaries, `finalization.available`, GPU `hardware_qualified`,
  exact verification row/unit coverage and every per-unit compared method.
  A zero-foreground scientific outcome remains visible and does not get turned
  into a runtime pass/fail shortcut.
* The sequential launcher always derives deterministic per-dataset ids, checks
  the exact expected run path and `last_run_dir.txt`, and has no newest-run
  glob. Optional reference evaluation runs after both freezes and only for
  explicitly supplied dataset reference configs.

## Verification

* `rtk pytest -q tests/test_maskfree_export.py` — **9 passed**.
* `rtk proxy bash -n scripts/run_maskfree_full.sh` — **passed**.
* `rtk proxy bash -n scripts/run_maskfree_acdc_mnms.sh` — **passed**.
* `rtk proxy python -m py_compile src/self_audit_maskfree/export.py` —
  **passed**.
* A mocked two-GPU `nvidia-smi` table resolved
  `CUDA_VISIBLE_DEVICES=1` to `GPU-bbb`, `NVIDIA GeForce RTX 4070`; the plan
  recorded that physical index/name/UUID and exited without execution.
* Duplicate dataset ids (`--datasets acdc,acdc`) are rejected before any run.
* The focused combined read-only check was **15 passed, 7 failed**. The seven
  failures are in concurrent W3 data/trainer changes (`tests/test_maskfree_data.py`,
  `tests/test_maskfree_firewall.py`), before this export layer is reached; they
  are not export failures.

## Integration contract for root

Use the original mapping object for the authoritative verification phase:

```python
from self_audit_maskfree.export import verified_freeze_session

with verified_freeze_session(original_manifest, require_complete=True) as frozen:
    # Pass `frozen` unchanged to load_verification_unit and verify_frozen_bank.
    ...
```

The context's entry and exit passes are the evidence for the single-process
read-only verification phase. Do not pass a JSON reload or `dict(frozen)` to
the authoritative validator inside the loop, or it will correctly perform a
full hash pass for every call.

## Limits

All evidence above is CPU/synthetic or mocked launcher evidence. No GPU, rented
GPU, ACDC/M&Ms image root, real native metadata, 150-epoch run, or clinical
quality result was observed here. v1 remains `spatial_predictive`; source frame
availability and geometry are unconfirmed until W3 inventory runs.
