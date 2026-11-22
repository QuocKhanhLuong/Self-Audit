# SliceSSMStudent: implementation, software checks and server queue (2026-11-22)

Branch `claude/slicessm-student-20261122`, based on `claude/w2s-ssl4mis-20261120` at `516bd0c`.
This report covers software evidence and one CPU probe on real ACDC images. No Dice result exists yet for this
architecture; the screening runs are queued on the server (see the last section).

## What was built

`self_audit/candidate_pg/slicessm_student.py`, selected by `student.arch = "slicessm"`.

- In-plane encoder/decoder of `AnisoStudent`: kernels `1x3x3`, no z down-sampling, the same deep-supervision/SDF
  train-only heads and the same stage-3 tap.
- **B1 `PhysZSSM`.** A bidirectional selective state-space scan along z for every in-plane cell, at stage 3 (1 block)
  and stage 4 (2 blocks). The step is `softplus(dt(x)) * (s_z / 10 mm)`, where `s_z` comes from `geometry["spacing_zyx"][0]`.
  Padded slices get a zero step. `spacing_mode = "none"` is the ablation without the physical factor.
- **B2 `SliceAlign`.** At stage 3, predicts in-plane offsets between adjacent slices, warps both neighbours onto each
  slice and fuses them residually. Zero-initialised, so the module starts as the identity.
- **Shift-recovery auxiliary.** `shift_recovery_loss` and `slice_shift_aux` inject known per-slice shifts and require the
  predicted offsets to change by the injected relative shift. The clean-view offsets are detached. Wired into
  `w2s_step.py`, active only when `student.slicessm.shift_aux.weight > 0`, so other arms are unchanged. Deterministic in
  `(seed, update)`.
- Config arms `S0slicessm` (UniMatch, no teacher) and `S1kdM3ss` (KD from frozen M3HL), both 150 epochs;
  SliceSSM export in `w2s_finalize_ssl4mis.py`; `scripts/slicessm_pipeline.py` modelled on `aniso_pipeline.py`.

## Size and cost

| Student | Parameters (train) | Deployed | GPU s/iter | Peak GB |
|---|---:|---:|---:|---:|
| PGStudent (unetrpp_lite_fuse), current baseline | 13,294,084 | – | – | – |
| AnisoStudent | 2,719,807 | – | 0.069 | 0.84 |
| SliceSSMStudent | 2,553,001 | 2,531,734 | 0.118 | 1.48 |

The GPU rows were measured on the server's RTX 5000 Ada while two other jobs were running, with bf16 autocast and a
1×1×10×256×224 volume (fwd + bwd + AdamW, plus the shift auxiliary for SliceSSM). They are indicative only.
SliceSSM is about 1.7× slower per iteration than Aniso, mainly because of the Python-loop scan and the two extra stage-3
encoder passes of the auxiliary.

## Software evidence

- `tests/test_slicessm_student.py`: 18 tests (parameter cap, shapes for D = 1/6/13, missing spacing raises, the
  physical step depends on spacing while the "none" mode does not, padded slices do not leak, identity init, warp
  correctness, shift injection on images and labels, zero loss for exact offsets, gradient reaches every parameter,
  code-branch contract, export cold load, dispatch, auxiliary determinism).
- Local: PG/W2S/aniso/slicessm selection 302 passed, 2 skipped. The 2 failures are environment-only
  (`medpy` and `h5py` are not installed on the Mac) and are unrelated to this change.
- Server (`/root/Self-Audit-slicessm`): slicessm + aniso + w2s_ssl4mis + pg_config tests, 46 passed.
- Server smoke with real data: S1 (CineMA distillation) 20 updates completed; arm A2 with the M3HL teacher
  (`S1kdM3ss` config), 830 updates (crosses the update-800 onset of the U path); see the final message for the outcome.
- Note: `cudnn_grid_sampler_backward` has no deterministic implementation, so SliceAlign adds one more
  warn-only nondeterministic kernel, consistent with the existing `BEST_EFFORT_GPU_WARN_ONLY` status.

## CPU probe on real ACDC images (B2 trainability only)

Encoder up to stage 3 + SliceAlign trained only with the shift-recovery loss, 400 steps, 40 ACDC training patients
(images only, no masks), evaluated on 10 held-out patients (041–050). Injected shifts are uniform in [−8, 8] px per slice.

| Step | MAE of recovered relative shift (px) | Zero predictor (px) |
|---:|---:|---:|
| 0 | 6.21 | 6.21 |
| 100 | 4.62 | 6.21 |
| 200 | 4.33 | 6.21 |
| 300 | 4.66 | 6.21 |
| 400 | 3.30 | 6.21 |

The offset head learns injected shifts from images alone and was still improving at 400 steps.
This shows the auxiliary is learnable. It says nothing about segmentation accuracy.

## Server queue (vast RTX 5000 Ada)

- Code: bundle `/root/slicessm.bundle`, worktree `/root/Self-Audit-slicessm` (the running jobs' checkout
  `/root/Self-Audit` is untouched; `external/M3HL` and `external/MedNeXt` are symlinked from it).
- tmux session `slicessm`, script `/root/slicessmF.sh`, log
  `/root/mount_data/runs/slicessm-screen-20261122/pipeline_vast_slicessm.log`.
- Configs written (SSCONFIG done): `ssl4mis_S0slicessm.json` (sha256 `1acee870…`), `ssl4mis_S1kdM3ss.json` (`3f3aeb8f…`).
- Each group takes `/root/gpu_heavy.lock`, so it starts when S2-arb-150 releases the lock and runs next to the aniso chain:
  `S1SS SSLINK` (CineMA S1, 10,000 updates), then `S0SS S0SSFINAL`, then `S1KDSS S1KDSSFINAL`. All FINAL steps are
  `--val-only`; test20 is not opened.

Comparisons once finished (val10, one seed, n = 10, screening only):
- S0-slicessm vs S0-aniso vs S0-unetrpp-150 (architecture, no teacher, same 150-epoch recipe);
- S1kd-M3HL-slicessm vs S1-kd M3HL (val10 0.8913).

Ablations not yet queued: `spacing_mode = "none"`, `align = false`, `shift_aux.weight = 0`.
