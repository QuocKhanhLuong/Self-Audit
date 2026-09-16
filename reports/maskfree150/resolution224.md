# Maskfree150 resolution 224x224

Status: **configuration and CPU software checks pass; no production run was
started** (2026-09-16).

## Contract change

Both production templates now resolve `image_size: 224`. The requested physical
batch and training budget remain unchanged: `batch_size=8`,
`accumulation_steps=1` (`effective_batch=8`), `amp=false`, 150 epochs,
`width=16`, `feature_dim=16`, `lr=0.001`, `weight_decay=0.0001`, and
`warmup_epochs=5`.

The existing dense models already support the even 224 -> 112 -> 56 pooling
ladder, and the data layer receives the resolved image size through its existing
image-only dataset APIs. No architecture, trainer, or observation code changed.

## Evidence

```text
PYTHONPATH=src pytest -q tests/test_maskfree_models.py
19 passed, 2 skipped in 1.53s

PYTHONPATH=src pytest -q tests/test_maskfree_data.py
11 passed in 1.95s
```

The focused model coverage checks deterministic CPU Producer and Student
forward/shape/backward at 224x224, including finite input and parameter
gradients. The fixed-pooling and CUDA repeatability smoke is parameterized for
both 128 and 224; both CUDA cases in this focused test were skipped because
`torch.cuda.is_available() == False` here. Config checks load both templates and
verify the 224 scientific identity plus the unchanged batch, precision, epoch,
optimizer, and model budgets.

Coordinator CPU224 canonical smoke (synthetic image-only data) also passed:

```text
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:src python scripts/profile_maskfree.py \
  --synthetic --image-size 224 --device cpu --warmup 0 --measured 1 \
  --timing-mode instrumented --kind early \
  --output /private/tmp/selfaudit-root-resolution224-smoke \
  --run-id resolution224-root-cpu
```

Receipt: all three component steps were `1`; finite loss/gradients were true;
32 fits, 160 fitting steps, 32 primary and 693 regional score calls, 0
regional refits, and 8 units were observed. The synthetic fixture reported
`semantic_unresolved=8/8`, so this is software evidence only and carries no
anatomical or label-quality claim. The combined suite initially reported 172
passed, 3 skipped, and 1 failure in the stale export-test `image_size=128`
assertion; that single assertion is now corrected to 224 and the final rerun is
complete. Final independent combined suite: **173 passed, 3 skipped in 43.80s**
using:

```text
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:src python -m pytest -q \
  tests/test_maskfree*.py \
  --basetemp=/private/tmp/selfaudit-root-resolution224-tests-final
```

No real GPU, remote host, real data, full training job, or representative
throughput benchmark was run; the bounded CPU smoke above executed one
synthetic training batch. Existing 128-resolution benchmarks do not establish
224 throughput, equivalence, or scientific performance. A fresh 224 run is
required; a 128 checkpoint must fail the scientific resume guard because
`image_size` is part of the scientific identity.

The profiler now supports `--image-size 224`, retaining historical default 128.
Use a fresh output/run ID and record its own result (FP32, physical batch 8),
for example:

```bash
PROFILE_ID="resolution224-$(date -u +%Y%m%dT%H%M%SZ)-$$"
PROFILE_OUTPUT="reports/maskfree150/resolution224/$PROFILE_ID"
CUDA_VISIBLE_DEVICES="GPU-33f45df8-0bbb-fd21-48a0-af323522bf89" \
CUBLAS_WORKSPACE_CONFIG=":4096:8" \
PYTHONPATH=.:src python scripts/profile_maskfree.py \
  --config configs/maskfree_acdc_150.yaml --device cuda \
  --image-size 224 --warmup 5 --measured 50 --timing-mode instrumented \
  --output "$PROFILE_OUTPUT" \
  --run-id "$PROFILE_ID"
```

The config supplies FP32 (`amp=false`) and physical batch 8. This command is a
future-run recipe, not execution evidence from this change.
