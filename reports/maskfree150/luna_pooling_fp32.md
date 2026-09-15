# Mask-free FP32 pooling baseline

Status: **CPU software baseline ready; GPU baseline not run** (2026-09-15).

## Scope

The model runtime now uses `torch.nn.functional.avg_pool2d(kernel_size=2,
stride=2)` whenever both spatial dimensions are even and at least two pixels.
This makes the production ladder `128 -> 64 -> 32` use a fixed 2x2 stencil. The
previous adaptive expression remains the correctness-compatible fallback for
odd or singleton dimensions, preserving the existing small/non-square input
contract. No warning-only or alternate behavior was added, and model weights,
channels, image size, and architecture are unchanged.

Both 150-epoch templates set `amp: false`; the runtime and tests remain FP32.
No BF16 or mixed-precision path was introduced.

## Evidence

Focused model checks:

```text
PYTHONPATH=src /Users/alvinluong/miniforge3/bin/python -m pytest tests/test_maskfree_models.py -q
14 passed, 1 skipped
```

The new CPU checks compare fixed pooling with the former adaptive expression on
`2x2`, `8x6`, and `128x128` grids, including input gradients. They also compare
Producer and Student forward outputs, input gradients, and parameter gradients
on even `8x6` and `32x24` grids. The tested CPU cases were bitwise equal
(forward and backward), not merely within a numerical tolerance. Existing
singleton/odd model coverage remains passing, and the production `128 -> 64 ->
32` ladder is checked for shape and deterministic results. These checks do not
establish GPU performance; the CUDA repeatability check is present but skipped
when CUDA is unavailable in the local environment.

Config/parser and syntax checks:

```text
maskfree_acdc_150.yaml amp= False image_size= 128
maskfree_mnms_150.yaml amp= False image_size= 128
PYTHONPATH=src /Users/alvinluong/miniforge3/bin/python -m py_compile \
  src/self_audit_maskfree/models.py tests/test_maskfree_models.py
```

This report is software evidence only: no real-data training, checkpoint, CUDA
device, or scientific/clinical claim is made.
