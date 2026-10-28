# Compact implementation delivery — 2026-10-28

**IMPLEMENTED / VERIFIED (software only). Deployment: BLOCKED.** No patient
training, full run, sweep, external download or active-job modification was done.
The user will run patient experiments. The original checkout and untracked work
were preserved; delivery lives on `QuocKhanhLuong/compact-kd-audit-20261028`.

## Delivered behavior

- Canonical train.py mode with a fully learned compact residual2.5D U-Net,
  all-module supervised AdamW and independent frozen-teacher output KD.
- Strict teacher/source/payload/TRAIN lineage, K1 reuse semantics, bounded
  image/probability memmaps, native aligned targets and no U masks.
- Matched plain additive SGD, scalar and joint finite whole-network trials;
  per-class hard-Dice/absent-FP checks protect only the extra U update.
- Standalone raw-student export; native manifest inference; historical
  class-first frame/patient evaluation after all predictions are frozen.
- Isolated paired benchmark with real mask writes, all native cases, CUDA
  synchronization, allocated/reserved peaks, sampled host RAM and file bytes.
- Explicit small-K and blocked joint4+4 templates, exact historical ACDC L8
  pins and a full opt-in10000/1000 plain-KD template. No automatic sweep.

Run commands and protocol: [COMPACT_STUDENT.md](../../docs/COMPACT_STUDENT.md).
Exact reference evidence: [SOURCE_INVENTORY.json](SOURCE_INVENTORY.json).
The historical baseline is Macro4 **87.2952%**, FG **83.1862%**; inference used
the **raw student** from a full-EMA run. It is not a compact result.

## Actual validation

`python scripts/check_repo.py`: PASS. `python -m pytest -q`: **789 passed,
2 existing warnings** (duplicate-optimizer fixture and torch.jit deprecation).
Full log: [PYTEST.txt](PYTEST.txt); source/file hashes: [VALIDATION.json](VALIDATION.json).
Initial worker drafts failed independent integration checks; their intermediate
reports were superseded. AGY workers implemented the first pass; Luna completed
trainer/inference after AGY quota failures. The coordinator reviewed the final
source, added adversarial regressions and ran public CLI smoke independently.
All five task areas settled and no worker terminal remained reclaimable.

The final bounded CPU CLI smoke built a real on-disk **manufactured** teacher
cache, trained all four modes for two updates (one supervised warmup), exported
plain KD, reloaded it, inferred/evaluated two odd-sized native cines, benchmarked
them and ran the acceptance assessor. Initial weights and L/U schedules matched;
encoder/decoder/head each changed weights; the fixed cache stayed unchanged.
Complete coefficients, trial outcomes, denominators, timing and all artifact
hashes are in [SOFTWARE_SMOKE.json](SOFTWARE_SMOKE.json).

| Manufactured arm | Updates | Nonzero U accepts | Reject-all | No candidates |
| --- | ---: | ---: | ---: | ---: |
| supervised | 2 | 0 | 0 | 0 |
| plain_kd | 2 | 1 | 0 | 0 |
| scalar_guard | 2 | 1 | 0 | 0 |
| joint_guard | 2 | 1 | 0 | 0 |

Manufactured CPU profile only: model-only median 0.000748s,
E2E median 0.005909s/p95 0.006139s
per full4D cine; throughput 32.899cine/s;
sampled process-tree peak 210.641MiB;
2288 parameters; 18060 deployed bytes.
This tiny smoke architecture uses widths4/8 and is not the configured patient
architecture. CUDA metrics and the historical paired benchmark are NOT_RUN;
these CPU numbers are not evidence of patient speed or memory improvement.

## Remaining empirical gates

1. Verify the pinned physical teacher payload and its original DINO assets;
   restore native geometry from pinned raw images if the NPY manifest lacks it.
2. Check supervised fit on budgeted TRAIN, then profile and run the locked plain
   KD recipe. Select checkpoints using the declared development rule only.
3. Evaluate the exported artifact on the same historical protocol. Require
   Macro4≥.87 and separately report BG/RV/MYO/LV and foreground macro.
4. Measure paired inference on PRO4000 with identical cases/geometry/workload.
   Require lower E2E time, GPUallocated/reserved, peak host RAM and deploy bytes.
5. Only then run explicitly selected matched audit controls. Audit benefit and
   joint superiority remain unproven; reject-all is not successful self-learning.

There is no final patient checkpoint in this delivery. Compact exact resume is
not implemented and is rejected, rather than silently restarting. Teacher
training cost is NOT_KNOWN unless supplied from evidence; cache-generation,
student and audit costs are separate. CPU test success does not certify Dice87,
novelty, clinical performance or research superiority. The synthetic acceptance
receipt correctly returns **BLOCKED**, not GO or an empirical NO_GO.
