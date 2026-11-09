# Candidate A cleanup — 10 November 2026

This is the validated 2D/2.5D baseline immediately before the requested A-3D extension. It is software evidence from manufactured data, not ACDC/M&Ms patient accuracy.

- `check_repo.py`: PASS; complete pytest: **72 passed**.
- End-to-end teacher pretrain → warm-up → joint → student-only export → frozen evaluation: PASS. Export/load logits match bitwise; payload has only student weights.
- All **19 component arms plus two shared teacher stages**, one synthetic epoch per job, completed under **both dataset profiles: 42 jobs**. Every run emitted per-class epoch metrics, validation rows and bounded, duplicate-free codes.
- Actual `.nii.gz` fixture preparation tests cover ACDC and original M&Ms, ED/ES, class remapping, anisotropic spacing, physical units and split isolation.

Resolved findings: pooled versus macro metric mixing; teacher-only validation measuring an untrained student; non-finite feasible policy actions; teacher proposals starving the other type; dense TRAIN ignore-pixel failure discovered too late; false surfaces at ignored evaluation pixels; random-sampler and materialized-epoch memory overhead.

The old joint_guard/EMA/control runtime, its three test modules, obsolete planners/entrypoint stubs and duplicated ablation configs were removed. Reused backbone, segmentation loss and geometry/data primitives remain. Git history and unrelated local reports/data/checkpoints are preserved.

`RECEIPT.json` pins source and synthetic exported weights. `CHECKS.json`, `pytest.txt`, and `MATRIX_CHECKS.json` record actual checks. Local ignored run directories are listed in the receipt. Later source changes require new validation; this receipt is historical.

Not run: patient training, licensed/pretrained teacher reproduction, CUDA profiling, seed-level efficacy, SOTA/96 Dice or novelty confirmation. Full training still requires prepared locked cohorts and the declared local pretrained backbone assets.
