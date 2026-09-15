# ACDC inventory geometry and terminal visibility fix — 2026-09-15

## Supplied execution evidence

The user supplied a stage-2 inventory failure for
`maskfree150_seq_20260915T081721Z_3262760_acdc` under
`/home/linhdang/workspace/quockhanh_workspace/SpecMamba`.
`acdc:patient001` had two distinct source geometries. The previous check compared
affine lists exactly; duplicate detection also required exact geometry, so a
byte-identical frame with a rewritten header could reach that refusal.

The supplied GPU record names RTX 5070 Ti, index 0,
`GPU-33f45df8-0bbb-fd21-48a0-af323522bf89`, with an explicit cu128 override.
This is user-supplied execution evidence. The conflicting real NIfTI headers
and that device were not accessed by this local fix session.

| Stage, for the supplied sequence | Status |
|---|---|
| ACDC inventory | FAILED in the supplied log |
| ACDC batch-8 preflight | NOT STARTED in that sequence |
| ACDC 150-epoch training | NOT STARTED in that sequence |
| Subsequent M&Ms run | NOT STARTED in that sequence |
| Local software checks below | COMPLETED |

## Implementation

* Exact decoded shape/dtype/content identifies native NIfTI 3D frame reexports
  of a rank-4 cine, even with different headers. Both geometries and the proof
  are recorded. Distinct rank-4 acquisitions remain counted; matching one frame
  does not prove two acquisitions are duplicates.
* All retained sources undergo pairwise study-grid checks. Full spatial shape,
  depth axis, orientation, format, affine availability and declared spatial unit
  must agree. Affine precision differences are accepted only if the maximum
  bidirectional displacement over eight spatial corners is <= 1e-4 voxel.
  There is no resampling and each original affine survives for native exports.
* `discovery_contract` version `maskfree150.data.discovery.v2`,
  `geometry_checks`, and per-record `study_grid_compatibility` document the rule.
  A material conflict exposes `MixedStudyGeometryError.details`; the inventory
  CLI writes `FAILED` / `training_status: NOT_STARTED` with all paths/geometries.
* CLI imports, inventory, image decoding/normalization, transfer, producer,
  audit, both students, optimizers, checkpoint IO, freeze hashing/assembly,
  verification and reports have live phase context. A 10-second heartbeat
  identifies the current work even before a batch finishes. Diagnostic updates
  are scoped so a previous file cannot label a later operation.
* A plain terminal dashboard survives `tee` without ANSI cursor escapes.
  It displays existing measured metrics, denominators, actual update counts,
  coverage/collapse information, stage timing and actual CUDA memory when
  available. Final computed image-only aggregates are emitted by split only
  after freeze. Logging does not score extra observations or change RNG state.
* Removed duplicate loss entries in the producer/student metric accumulators;
  their loss means and training gradients are unchanged, and counts now record
  each batch once. Diagnostic journals append independently of checkpoint and
  canonical epoch metric recovery.

Existing supervised Self-Audit and Candidate C files are outside this change.
No mask reader, teacher, GT threshold/checkpoint rule, or temporal evidence was
added. Physical batch remains 8 with accumulation 1 in the production templates.

## Independently executed checks

Environment: local CPU Python from `/Users/alvinluong/miniforge3/bin/python`;
`PYTHONPATH=src OMP_NUM_THREADS=1 MKL_NUM_THREADS=1`.

1. `pytest -q tests/test_maskfree_progress.py tests/test_maskfree_trainer.py tests/test_maskfree_reporting.py`
   — **10 passed in 4.46 s**. Includes background heartbeat during a blocked
   load, context cleanup, error output, readable redirected dashboard,
   byte-identical model states with telemetry enabled/disabled, exact resume,
   and CLI failure JSON before training.
2. `pytest -q tests/test_maskfree_data.py tests/test_maskfree_firewall.py tests/test_maskfree_pipeline.py`
   — **17 passed in 16.59 s**. Includes the reheader duplicate, bounded drift,
   material displacement/shape/unit rejection, withheld-value mutation checks,
   synthetic native ACDC and M&Ms export/freeze/verification with live aggregate
   output, and resume preserving frozen bytes. Reference fixtures in these
   tests are synthetic and created only after prediction freeze.
3. Actual `scripts/train_maskfree.py` CLI on synthetic ACDC fixtures, CPU,
   physical batch 2, one optimizer group: **PASS**, exit 2 (`partial`). This is
   a bounded software smoke, not a production batch fallback or GPU gate.
   Captured `terminal.log` and `dashboard_excerpt.txt` reside under
   `runs/maskfree150/software_checks/geometry_progress_20260915_cli_smoke/`.
4. Both launcher shell syntax checks, config-only CLI resolution, and
   `git diff --check` passed. Orca run `run_d19ed61b546b` check: `No messages.`

Root/Astra made architecture decisions and reviewed the combined tree.
The existing `/root/luna_data` worker owned discovery and its geometry tests;
`/root/luna_reporting` performed a read-only metrics/phase audit. No new worktree,
GPU job or SSH session was created. Worker-reported pass counts are not added
to the independent totals above.

## Operator recovery

Update `main` on the rented host and relaunch with a new run ID, the same chosen
image root, explicit GPU identity override, physical batch 8 and accumulation 1.
Keep the failed run and its read-only YAML as evidence. There is no training
checkpoint from the supplied stage-2 failure to resume. The actual source may
still contain a material conflict; if so, the new inventory JSON reports the
exact sources and discrepancy instead of silently bypassing the check.
The full run must pass a fresh real batch-8 preflight before training.
