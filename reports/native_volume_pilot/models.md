# Native-volume models owner report

## Scope

Owned implementation is limited to `self_audit/volume/models.py`,
`self_audit/volume/fusion.py`, `tests/test_volume_models.py`, and
`tests/test_volume_fusion.py`. Existing `self_audit/model.py` and all historical
training paths remain unchanged.

## Implemented API (corrected G1/G2/G3/G6/G8 scope)

`build_volume_model(config, *, synthetic=False, architecture_only=False)` returns
`(model, provenance)`. The deployment-only `architecture_only=True` seam
verifies a pinned clean local DINO source and constructs it with
`torch.hub.load(..., source="local", pretrained=False)` without opening or
hashing the configured pretrained weight path. Normal training retains strict
source/weight/hash validation; callers must load the complete exported state
dict with `strict=True` afterward. Tiny architecture-only construction is
synthetic-only; CNN3D remains scratch-built.
The model accepts a normalized `B,3,Z,H,W` tensor and an optional boolean
`B,Z,H,W` geometric support mask, and returns `B,4,Z,H,W` logits. Each model
also exposes `parameter_groups()` with disjoint `encoder`, `body`, and `head`
lists; no external teacher or network access is attached.

* `volume25d`: G1 control; wraps the unchanged `NativeStudent` independently
  for each Z slice. Both synthetic TinyEncoder and non-synthetic
  `dinov3_convnext_tiny` routes call the unchanged hash-pinned
  `self_audit.model.build_model`; missing source/commit/weight/hash fields fail
  closed, without an unconditional synthetic-only rejection. Fixed-window
  configs are passed through unchanged and the three-channel center triplet is
  preserved.
* `hybrid`: G2/G6 path; reuses the actual NativeStudent input adapter, all four
  encoder stages, lateral/up/refine modules, per-slice fixed SemanticWindow,
  and native detail/fuse/head. New residuals are learned at all four
  multiscale decoder resolutions (`depth_mode=off|mix|planar_control`) inside a
  forked RNG. Zero residuals are exactly equal to the old per-slice function;
  mix reads cross-Z neighbours, while planar_control has active per-slice
  parameters and no cross-Z read. G6 skip modes are legacy, delayed nonlinear,
  and near-capacity matched additive; stop-gradient and a separate frozen
  checkpoint intervention are explicit.
* `cnn3d`: G3/G8 scratch residual 3-D U-Net with learned decoder upsampling and
  encoder skip fusion, configurable odd anisotropic `z_kernels` and positive
  `z_strides`, center-channel input only, and exact native shape restoration for
  Z=1/variable-depth/odd XY inputs.

The provenance record includes architecture/depth and skip factors, synthetic
status, scratch/native initialization, input backbone declaration, parameter
count, disjoint group counts, `external_teacher=false`, and analytical
convolution/MAC proxies, actual encoder/decoder width provenance, and exact
whole-model parameter matching. `architecture.widths` means NativeStudent
decoder/lateral widths: DINO `[96,128,192,256]` versus encoder
`[96,192,384,768]`, and tiny `[8,16,24,32]` versus encoder `[8,16,32,64]`.
CNN3D stage widths remain its own explicit control. Hybrid depth kernels are
depthwise `(kz,1,1)`; arbitrary XY entries fail closed rather than being
ignored. Capacity reports include an executed Conv/Linear MAC proxy at the
declared synthetic shape `(1,3,3,32,32)`, including spatial output sizes and
group divisors while declaring normalization/interpolation/activation costs
omitted.

At the same synthetic tiny decoder widths and input shape `(1,3,3,32,32)`,
the current receipts are: V04 (`mix`+legacy) **129,207 parameters / 122,164,032
executed MACs**; V05 (`planar_control`+legacy) **128,551 / 122,139,648**;
S01 (`mix`+delayed) **131,663 / 122,630,976**; S02 (`mix`+matched_additive)
**131,653 / 122,624,832**. These are software capacity proxies, not runtime
throughput or patient evidence; the production DINO arms must use decoder
widths `[96,128,192,256]` and regenerate receipts after configuration update.

Zero-residual controls only zero the terminal residual projection in each
CrossZ/SkipFusion block. Upstream projections remain randomly initialized, so
the old per-slice function is exact at construction while the block can learn
input-dependent residuals after subsequent SGD updates. Hybrid models default
to a fixed SemanticWindow; an explicitly supplied `window.mode=none` remains a
named disabled control rather than being silently rewritten. Primary V03/V04/
V05 registry recipes must still set fixed windows and matching skip modes.

G6 frozen interventions are operational: `SkipFusion.branch_intervention()`
(alias `inference_intervention`) is a reversible context manager that zeros a
selected added residual input (`skip`, `context`, or `both`) without changing
parameters, optimizer state, or trainability. `set_frozen_intervention()` is a
persistent checkpoint diagnostic that freezes only the added skip projection
and uses the same branch-zeroing behavior; `clear_frozen_intervention()`
restores trainability. Native decoder/detail paths remain live, so the scope is
the added residual branch rather than a claim to remove the native detail path.

## Verification

Exact owned-test command:

```text
rtk python -m pytest -q tests/test_volume_models.py tests/test_volume_fusion.py
```

Result: **28 passed** (CPU, torch 2.14.0, 2026-10-26). Tests cover output
shapes, anisotropic downsampling and learned decoder skip gradients,
center-channel policy, finite encoder and multiscale cross-Z gradients,
disjoint optimizer groups, enabled-neighbour influence, exact zero-residual
NativeStudent parity (legacy, delayed, and matched-additive skip modes; with
and without fixed window), depth-off independence, active planar-control
gradients, terminal-only zero initialization, two-step input-dependent residual
learning, terminal-only SkipFusion zero-init alias behavior, CNN stage-width tails,
fixed/disabled window controls, architecture-only local-DINO construction with
missing-weight/state-replay and source-dirty rejection, reversible branch
interventions, persistent intervention gradient scope,
canonical encoder/decoder width provenance, depthwise XY validation, executed
same-shape MAC proxies with grouped-convolution divisor and V04/V05/S01/S02
parameter/MAC comparisons,
capacity-changing width/depth-kernel knobs, matched-capacity nonlinear fusion,
stop-gradient versus frozen intervention, malformed input/grid rejection,
hash-pinned DINO field validation, and the mocked non-synthetic loader seam.
`rtk python -m compileall -q self_audit/volume` also completed successfully.

Repository checks run in the shared worktree were `rtk python scripts/check_repo.py`
(**PASS**) and `rtk python -m pytest -q` (**246 passed, 1 warning** after the
native-volume workers landed). The full suite was CPU-only; the warning was the
pre-existing TorchScript deprecation in `tests/test_pipeline.py`.

After concurrent owner fixes, the latest shared full-suite run was **299 passed,
3 failed, 1 warning**. The three failures are correction-owner witnesses: the
strict TRAIN-L click-simulation contract now rejects legacy tests/CLI invocations
that still pass role `TRAIN` without an explicit label budget and patient ID.
No model/fusion test failed. Earlier geometry/region and registry failures were
resolved by their owners; no missing pretrained asset was downloaded or
substituted.

## Boundaries and unresolved dependencies

This is software/synthetic verification only: no patient data, GPU, real
weights, U/VAL/TEST masks, or external teacher were accessed. Normal production
training remains blocked until local source/weight assets are supplied with
exact hashes; architecture-only deployment now uses the explicit source-only
seam and strict exported-state replay. Integration still needs to route the
explicit `self_audit.volume.v1` schema and preserve the historical trainer;
optimizer, geometry, loss, EMA, and registry owners must consume the returned
groups and capacity provenance without changing their contracts.
