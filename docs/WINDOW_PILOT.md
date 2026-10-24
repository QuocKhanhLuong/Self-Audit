# NativeStudent adaptive-window + meta-Dice pilot

Status: implemented opt-in research pilot. The initial delivery contained
CPU/manufactured verification only. Base: experiment commit
`8960b6baac4c6600553d900c303f192c2fcfa4f0`.
The canonical `joint_guard` defaults and existing experiment matrix stay intact.

Subsequent server execution on 2026-10-24 completed all three arms with 10,000
steps, 1,000 warmup steps, seed42, dense TRAIN budget L8, crop256 and batch8.
Frozen final-student TEST50 evaluation receipts are under
`reports/window_pilot_test50_20261024/`. All three final checkpoints were pinned
before TEST preparation; no TEST masks influenced training or model selection.
The testing copy and references came from the configured Drive mirror; their
official upstream provenance and challenge-server certification are unavailable.
These measurements do not establish novelty or a general Dice guarantee.

## Matched arms

| Recipe | Method | Support | Geometry training |
|---|---|---|---|
| `configs/window_pilot_baseline.json` | `ema_full` | Original decoder | No geometry parameters |
| `configs/window_pilot_fixed.json` | `ema_full` | Fixed K=9 reader | Frozen regular grid |
| `configs/window_pilot_meta.json` | `ema_full_meta_window` | Adaptive K=9 reader | Update-mediated meta-Dice after warmup |

These three recipes use 8 total steps, 4 warmup steps, batch1, crop256, the same
seed, pretraining requirement, losses and schedules. They are templates for a
bounded pilot, not measured GPU-feasible configurations. Their backbone paths
are intentionally unset until an explicit hash-pinned local model config is
provided. No downloads, random patient-model fallback or license acceptance.

An additional ordinary-adaptive control is supported by selecting `ema_full`
with `window.mode=adaptive`. Its geometry is frozen during warmup and trained
by ordinary L+U gradients afterward. It is not the meta arm. Supervised arms
also support the operator; guarded/head transactions reject it in this pilot.
Missing window/meta config fields normalize to disabled defaults for old v2
recipes; unknown fields and unsupported combinations are rejected.

## Operator and sparse meaning

Immediately before the final stride8-to-stride4 `refine[0]`, keep the original
`L0 + up0(interpolate(D1))` and add a residual semantic support reader. Queries
come from that fine-grid sum; keys/values come from `D1`. Each query samples
nine positions on a canonical 3x3 grid transformed by a learned 2D center and
two bounded radii. The raw full-resolution detail branch remains unchanged.

With the full-width decoder, d=32 adds 14,916 projection/geometry parameters
plus a scalar residual gate. The gate and Q/K/V/output projections belong to
ordinary student parameters theta. Only `window.geometry.*` belongs to psi.
This is sparse spatial attention inside a dense model, not weight pruning.

The sampler uses gather and bilinear arithmetic rather than relying on
`grid_sample` double backward. Tests compare its interior values to
`grid_sample` and check first/second derivatives away from cell/validity
boundaries. Validity is piecewise discrete; derivatives at a cell boundary or
when a sample enters/leaves the valid region are not certified by those tests.

Image/crop geometry supplies a separate boolean mask for all roles. Internal
padding is marked invalid even when inference callers omit that mask. Coarse
cells require full image coverage; bilinear support touching invalid cells is
rejected. All-invalid queries return a zero residual, including projected bias.
This avoids sampling padding directly but cannot remove all encoder receptive
field effects from padding. Inference needs only the student and image.

## Exact training scope and timeline

1. Sample the existing supervised/audit/unlabeled roles. Audit is a different
   TRAIN patient in the same declared labeled budget; it is not held-out VAL.
   Valid masks follow the same image flip without consuming another RNG draw.
2. Perform the ordinary supervised CE+Dice step. In the meta arm, freeze psi,
   clear its stale gradients and clip only theta. Feature-input gradients still
   flow through the frozen geometry predictor.
3. Freeze EMA hard pseudo-labels, confidence/valid mask and the strong-intensity
   tensor. Preserve the post-supervised theta/optimizer reference.
4. In eval mode, compute a functional virtual full-theta AdamW-body/SGD-head
   pseudo step with current moments, bias corrections, decoupled weight decay
   and the same global clipping. Parameters and state are cloned/detached from
   history; live model, optimizers and teacher are not advanced by the trial.
5. Evaluate batch-aggregated **present-foreground soft Dice loss** on the TRAIN
   audit batch. Recompute outer geometry with detached psi. The only psi
   gradient path is `psi -> pseudo gradient -> virtual theta -> outer Dice`;
   the indirect `virtual theta -> features -> geometry` path remains active.
6. Apply one separate momentum-free, decay-free SGD geometry step (default
   lr0.001, clip norm1). Clear psi gradients. Discard virtual theta/state.
7. From the unchanged post-supervised theta/state, commit one real pseudo step
   using the new geometry and the same targets/masks/strong tensor. Psi remains
   frozen during this ordinary step. Do not add another supervised update.
8. Update all EMA teacher parameters once after the final student state;
   copy buffers as in the baseline. The final student is the prediction model.

No pseudo support: skip geometry and real pseudo updates. No foreground in
the audit batch: skip geometry, retain the ordinary pseudo update. An absent
class receives no positive Dice term; logs identify present classes. Empty
support is not a successful meta update. Nonfinite gradients/losses abort.

The functional optimizer supports the actual plain AdamW and momentum-free SGD
family only. Unsupported optimizer variants, geometry in ordinary optimizers,
duplicate owners and missing trainable theta owners fail closed. Meta ordinary
optimizers explicitly use `foreach=False`. Adam second-moment sqrt keeps the
exact forward value and uses a zero derivative extension at exactly zero to
avoid NaNs; cold/warm parity and finite-difference tests cover this convention.
There is no AMP, decoder-only proxy, virtual-SGD substitution or first-order
detach approximation hidden in the meta arm.

This is a truncated one-step hypergradient, not a gradient through all training
history and not a guarantee of improved Dice. Geometry changes between trial
and commit. `steps.jsonl` therefore records reference/virtual/committed TRAIN
soft Dice losses, predicted/actual loss reduction, gradient/update norms,
accepted coverage and present classes. Negative gains are retained.

## Verification and reproducible short runs

Run from this worktree with its supported Python dependencies installed:

```bash
python scripts/check_window_hypergradient.py --device cpu --out /tmp/window-gradient-new.json
python -m pytest tests/test_window.py tests/test_meta_window.py -q
python scripts/run_window_smoke.py --out /tmp/window-smoke-new
python scripts/check_repo.py
python -m pytest -q
```

Output paths must be new. `run_window_smoke.py` runs the numerical gate first
and refuses to continue if it fails or the runtime source changes. It then
runs exactly three steps per arm (one warmup) on manufactured 32x32 inputs
with crop40 to exercise padding, TinyEncoder, d=4, confidence0.25. It checks
matched patient-role schedules, label boundaries, geometry updates and EMA
counts. The manufactured threshold and small backbone are explicitly different
from patient recipes; its losses are not cardiac accuracy results. No patient
dataset or pretrained weights are read by either verification script.

The numerical gate uses actual NativeStudent + sampler in double precision
with TinyEncoder. It compares center and radius directional hypergradients to
central differences, keeping outer geometry fixed, in cold and warm optimizer
states. It separately compares the functional update with the real optimizer.
Unit tests additionally exercise active gradient clipping and transaction
immutability. A CUDA gate can be requested explicitly with `--device cuda`;
the delivered receipts are CPU only. CUDA full-model memory/throughput remains
unmeasured and must be established before considering a long run.

Checkpoint config records the architecture and geometry optimizer; loading
reconstructs baseline/fixed/adaptive models correctly. Runtime source hashes
remain enforced and exact resume remains unsupported. An old checkpoint must
be used with its pinned old source, not bypassed through this new loader.

## Delivered evidence and limitations

`reports/window_pilot/` contains numerical, baseline-parity, matched-smoke and
test receipts. Baseline parity compares the disabled-window branch with an
isolated copy of the exact base commit using identical manufactured data.
Student, teacher and optimizer tensors are compared directly, not only losses.

The initial tests establish software/numerical behavior for the inspected backend.
They do not establish useful geometry, patient Dice, calibration, GPU capacity,
or a new algorithm relative to prior work. The original real-data baseline
remains one seed on exposed VAL18. No cancelled real-data jobs are resumed.

Algorithmic attribution still requires ordinary-adaptive, supervised-only
geometry, meta-pixel weighting, extra labeled exposure, outer-CE, reduced-U,
and shared frozen-teacher controls. Compare patient-level Dice and per-class
harm on a locked protocol; do not tune against exposed VAL18 and call it test.

Related work informing these controls: [Primus](https://arxiv.org/abs/2503.01835),
[DySample](https://arxiv.org/abs/2308.15085), [FaPN](https://arxiv.org/abs/2108.07058),
[BATFormer](https://arxiv.org/abs/2206.14409), and
[MLB-Seg](https://arxiv.org/abs/2307.11604).
