# Bounded `_batch_log_mixture` density refinement

Date: 2026-09-15
Scope: private `src/self_audit_maskfree/observation.py::_batch_log_mixture` and
targeted tests only. No fitting equation, capacity, iteration budget, prior,
loss, RNG, or public API was changed.

## Design

The previous helper evaluated the full `[B, M, N]` density separately for each
of the four semantic regions, then selected the pixels belonging to that
region. The refinement gathers each pixel's hard-label parameters from the
existing `[B, 4, M]` tensors into `[B, M, N]` (means, variances, Python
`math.log` normalizers, and log weights), evaluates `_log_density_batch` once,
and performs the same mode-axis `logsumexp(dim=1)`. Region-specific padded
`-inf` log weights remain in place for missing modes; component validation and
the NaN/unscored-pixel guard remain in `_component_batch_tensors` and the
helper. Invalid labels are converted to a safe gather index but are restored to
NaN before the existing fail-closed guard, so they cannot silently score.

## Exactness and focused tests

`tests/test_maskfree_observation_fast.py` now compares the gathered helper to
an independent explicit four-region reference for `B=1, 8, 32`, one or two
background modes, both Gaussian and Student-t families, all labels, missing
labels, and tiny non-contiguous views. It also checks missing-mode `-inf`,
omitted-region and over-capacity validation, invalid-label rejection, and the
NaN guard.

```
PYTHONPATH=src pytest -q tests/test_maskfree_observation_fast.py
23 passed
```

All twelve parameter combinations and three input layouts were
`torch.equal` bitwise; the returned tensors stayed float64 with the input
shape.

## Bounded primitive microbenchmark

CPU-only, seeded synthetic `B=32, H=W=128 (N=16,384)` fixture. Components
were produced by the actual `ObservationModel.fit_many` path from that same
fixture (not hand-entered), then held fixed while timing the old explicit
four-region helper and the gathered helper. `torch.set_num_threads(1)` was
used; each case had 3 warmup calls followed by 15 measured calls, and the
table reports median seconds per call. Dense labels cycle through all four
regions; sparse labels contain three small foreground runs and a background
remainder. The comparison is a primitive density result, not an end-to-end
pipeline claim.

| labels | background modes | family | old (s) | gathered (s) | old / gathered |
|---|---:|---|---:|---:|---:|
| dense | 1 | Gaussian | 0.022095 | 0.017747 | 1.245x |
| dense | 1 | Student-t | 0.029103 | 0.019433 | 1.498x |
| dense | 2 | Gaussian | 0.034839 | 0.016427 | 2.121x |
| dense | 2 | Student-t | 0.050853 | 0.019534 | 2.603x |
| sparse | 1 | Gaussian | 0.023215 | 0.018043 | 1.287x |
| sparse | 1 | Student-t | 0.029742 | 0.019799 | 1.502x |
| sparse | 2 | Gaussian | 0.036403 | 0.016241 | 2.242x |
| sparse | 2 | Student-t | 0.050872 | 0.020087 | 2.533x |

The old and gathered outputs were `torch.equal` in all eight benchmark cases
(maximum absolute difference `0.0`). These timings justify retaining the
small private primitive change; they do not establish a full score, training,
CUDA, or production throughput claim.

## Verification boundary

The focused observation, auditor, and trainer-performance suites passed after
the change (`36 passed`), as did the four-test observation contract suite and
the three-test trainer suite. The coordinator still owns the combined-tree
151-test gate and end-to-end benchmark; no full training or 20-batch benchmark
was run here.
