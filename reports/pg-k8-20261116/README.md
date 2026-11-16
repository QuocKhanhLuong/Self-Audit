# Candidate PG v2 — code-only delivery

The user narrowed this delivery to implementation and local checks: “chỉ code thôi chưa test được server”. No patient training, GPU test, or campaign execution on `vast-gpu` was performed. S1, S2, S3 and A1–A4 remain **NOT_RUN on real data**. The design has no measured efficacy result.

## Implementation

Implementation commit: `088b18ba82b97dea853bcb658666e6532596e965`. The following report commit contains only software evidence and provenance.

The additive `self_audit/candidate_pg/` package contains CineMA initialization, the anisotropic MedNeXt-L teacher, the PG student, views/CutMix, reliability weights, codebook/CBKD, training steps, stage runners, epoch logging, export, evaluation, finalization and paired analysis. `configs/candidate_pg/pg_k8_base.json` holds the campaign settings; the four arm files contain only the two switches. `train.py` and `evaluate.py` remain the canonical entrypoints. FUSE remains the default track.

Inspection and source evidence: [INSPECTION.md](INSPECTION.md), [CINEMA_SOURCE.md](CINEMA_SOURCE.md), [REQUEST.md](REQUEST.md). The sole Claude reviewer records findings and dispositions in [CLAUDE_REVIEW.md](CLAUDE_REVIEW.md). Parallel coding workers used Orca with `gpt-5.6-luna`.

## Measurements

| Model | Parameters | Evidence |
|---|---:|---|
| Student-only deployment | 3,074,852 | Local model construction; zero fusion/mimic parameters |
| PG training student, without code branch | 3,425,028 | Local model construction |
| MedNeXt-L teacher, including compatibility heads | 61,907,084 | Official pinned source; CPU forward/backward with skip dropout |
| CineMA pretrained SAX encoder | Not measured | Full weights not loaded in this delivery |
| GPU peak memory / throughput | Not measured | No GPU/server execution |

The official MedNeXt check used commit `0b78ed869fbd1cc2fd38754d2f8519f1b72d43ba`, a random initialization, and a synthetic native volume of size `5×17×19`. Its five measured feature phases match the requested schedule. See [MODEL_SOFTWARE_CHECK.json](MODEL_SOFTWARE_CHECK.json). This establishes software execution, not patient performance.

The unchanged baseline initially passed **991 tests, with 4 skipped and 3 subtests passed**. The Q3D smoke completed with its explicitly synthetic gate override; the measured tokenizer gate was `NOT_READY`, not a passed quality gate. See [BASELINE_CHECKS.json](BASELINE_CHECKS.json). Final integrated verification passed **1085 tests, 4 skipped and 3 subtests**, plus `check_repo` and the complete synthetic PG smoke. All four deployed exports contain 3,074,852 parameters and pass bitwise cold-load checks. See [SOFTWARE_CHECKS.json](SOFTWARE_CHECKS.json), [SOURCE_FILES.json](SOURCE_FILES.json), and [ORCHESTRATION.json](ORCHESTRATION.json).

## Campaign results

| Arm | PG | CBKD | TRAIN16 | VAL18 | TEST50 |
|---|---|---|---|---|---|
| A1 Base | Off | Off | NOT_RUN | NOT_RUN | NOT_RUN |
| A2 +PG | On | Off | NOT_RUN | NOT_RUN | NOT_RUN |
| A3 +CBKD | Off | On | NOT_RUN | NOT_RUN | NOT_RUN |
| A4 +PG +CBKD | On | On | NOT_RUN | NOT_RUN | NOT_RUN |

There are no measured Dice, HD95, ASSD, learning curves, Wilcoxon/bootstrap comparisons, or code-shuffle effects from this campaign. Synthetic fixture metrics are software checks and are excluded from this table.

The implemented protocol logs TRAIN, VAL and TEST every epoch. **Only VAL18 selects checkpoints; TEST50 is monitoring only and must never determine a checkpoint, hyperparameter, or arm choice.** The final `TEST_ONCE` transaction locks the four selected exports for verification. It does not mean TEST was previously unseen: the explicit per-epoch monitoring requirement supersedes that interpretation.

## Resolutions and limitations

- The unchanged `q3d_teacher_loss` expects four SDF channels and a boundary head. PG keeps its public three-channel foreground SDF, adds an internal background SDF channel and four-class boundary head, and passes the four-channel concatenation only to the existing supervised loss adapter.
- `best_val` is appended after the requested fixed CSV column prefix. Labeled padding uses the existing FUSE convention; pooled code losses exclude invalid padded cells.
- S1 uses the last 10% of the stable TRAIN image list as its image-only holdout, explicitly recorded as `cinema_init.holdout_fraction`. It reads native frames lazily and logs cosine similarity (one minus cosine loss).
- CutMix coordinates stay aligned to 16. The box lies inside the common in-plane support of acquired slices. When the partner has fewer slices, full-depth mixing keeps its entirely padded z slices invalid with zero loss weight. An unaligned box is never substituted.
- S3 uses one labeled frame and the mean contribution of two image-only U frames per update. The explicit S3 optimizer is AdamW with learning rate `1e-3` and weight decay `1e-4`.
- S2 crops/pads partner B to A's canvas before drawing B's independent weak transform. S1 computes full TRAIN-subset cosine each epoch, so its evaluation cost still needs real-data profiling. The review's earlier S1 eager-RAM note is superseded by the lazy image sequence in the final implementation.
- Tiny smoke budgets, synthetic encoders/teachers and warmup overrides are identified in their own receipts. They are not campaign settings or pretrained evidence.
- No memory-driven reduction in strong views was measured. Real runs require the locked ACDC manifests, the pinned MedNeXt checkout, and the official CineMA package/dependencies plus only its two permitted pretrained files.
- The uniform one-strong-view setting is available through `runtime.oom_reduce_strong_views` and writes an adaptation receipt for S2/all arms. Automatic recovery from a CUDA OOM is not implemented; choosing or validating this setting requires the deferred GPU check.
- PG reuses FUSE's deterministic policy: strict CPU, best-effort CUDA with `warn_only` for unavailable deterministic kernels, cuDNN benchmark off, and TF32 off. Receipts distinguish these modes; the CUDA flag path is unit-tested without initializing a GPU.
- Cleanup removed three recorded `.DS_Store` files. Historical source tracks, results, and unrelated untracked work were preserved; see [CLEANUP.md](CLEANUP.md).

## Interpretation

This delivery provides an experimental implementation and local correctness evidence. It does not establish that PG, CBKD, or their combination improves segmentation. That conclusion requires the complete locked campaign and the paired patient analysis after all four arms finish.

## Execution commands (not launched on the server)

After preparing the locked manifests and the pinned dependencies described above, the canonical campaign entrypoint is:

```bash
python train.py --config configs/candidate_pg/pg_k8_base.json \
  --out /root/mount_data/runs/pg-k8-20261116 --device cuda --pg-stage all
```

`--pg-stage S1`, `S2`, `S3`, or `arms` can address a stage; later stages require the preceding hash-verified artifacts. Switch-only arm configs select their matching arm and default to `arms`; they cannot rerun shared prerequisites. Run S1, S2 and S3 once with the base config before separate arm commands. A base-config `--pg-stage arms --pg-resume` collects all four finished arms and performs finalization.

`--pg-resume` restores S2/S3/arm checkpoints and reuses a completed, hash-verified S1 artifact. An interrupted S1 restarts its distillation because its optimizer/projectors are not checkpointed mid-stage. A single-arm execution does not open the final all-four TEST transaction. CUBLAS and deterministic/TF32 settings are installed before CUDA initialization and recorded in receipts.

The local fixture command is `python scripts/smoke_pg.py --out /tmp/pg-software-smoke --updates 1`. Its tiny injected source/teacher, budget and warmup overrides are written to software-only receipts. This command cannot establish pretrained CineMA compatibility, GPU memory requirements or clinical performance.
