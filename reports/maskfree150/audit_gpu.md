# Maskfree150 auditor device gate (resolution 224)

Status: **CUDA NOT RUN** on this checkout.  The requested CUDA gate is a
non-success status when `torch.cuda.is_available()` is false; the run does not
fall back to CPU and does not make a GPU, RTX, speedup, scientific, clinical,
or production claim.

## What the gate checks

`[scripts/check_maskfree_audit_device.py](/Users/alvinluong/Self-Audit/scripts/check_maskfree_audit_device.py)` is an independent hardware/equivalence gate.  It
generates the four-candidate bank once from CPU fitting observations, freezes
that content, keeps public banks/views on CPU as the trainer does, sets
`ObservationModel(execution_device=...)`, and calls the canonical `audit_banks`
API.  It compares candidate IDs, labels, probabilities,
and validity bitwise; fit parameters and fit/selection scores in strict FP64
tolerances; regional margin and validity in strict FP32 tolerances (including
maximum absolute differences); and selected/accepted/rejected, search,
semantic-unresolved, evaluation-order, and budget decisions exactly.

CUDA also requires public inputs/banks to remain CPU while private heavy helper
inputs execute on CUDA (portable fitted hypotheses and regional/topology outputs
are expected to serialize to CPU), the model's declared execution device to
match, and profiler/private-helper evidence of heavy CUDA operations.  Timing
synchronizes before and after each pass, performs symmetric bounded warm-up and
repeats on both CPU and GPU, and emits an audit-only speedup ratio only when
the full work counters, decisions, floating comparisons, and residency checks
all pass.  Fresh output paths are required; defaults use the portable system
temp directory (`tempfile.gettempdir()`).

## CPU evidence

The bounded image-only fixture is deterministic, non-degenerate, and frozen at
`224 x 224`, seed `42`, with disjoint fit/selection supports and no mask or
reference inputs.  The CPU selfcheck command completed with
`CPU SELFCHECK PASS`:

```bash
python scripts/check_maskfree_audit_device.py \
  --device cpu --cpu-selfcheck --image-size 224 \
  --synthetic-units 1 --warmup 0 --repeats 1 \
  --output /private/tmp/maskfree150-audit-device-cpu-selfcheck.json
```

The full logical work recorded for this one unit is:

| counter | value |
| --- | ---: |
| candidate bank items | 4 |
| fit calls/items | 4 / 4 |
| primary score calls/items | 4 / 4 |
| `fit_many` calls/items | 1 / 4 |
| `score_many` calls/items | 1 / 4 |
| regional `prepare_score` calls | 4 |
| regional `score_region` calls | 52 (fixture observation) |
| total score calls including regional work | 56 |
| fitting iterations per candidate | 5 |
| audit rounds | 2 |
| deterministic algorithms effective | true |
| `CUBLAS_WORKSPACE_CONFIG` | `:4096:8` (validated before CUDA context) |

The four fits are the complete fixed budget: two rounds partition challenger
slots and do not multiply the fit count.  The regional ceilings remain the
canonical 32 scored regions and 128 extra score calls.  CPU selfcheck output is
software reproducibility evidence only and intentionally has
`speedup = null`.

The synthetic one-unit receipt selected the grouping incumbent, retained the
learning sample, had `search_inconclusive=false`, scored 21 challenged regions
(11 additional regions unobserved), and produced nonzero validity (mean
`0.3125658035`, max `0.9369222522`).  These are bounded self-audit decisions,
not correctness or clinical labels.

The coordinator supplied the final frozen CPU224 matched-harness receipt
(`matched=true`, all 45 checks, with no speed inference because the runs
overlapped other tests):

| artifact | SHA-256 |
| --- | --- |
| `/private/tmp/selfaudit-root-auditgpu-finalcpu224-20260916/profile_report.json` | `0a87c41be9570aef01ab772eec342f07a54a411cc82ad3d2c47963ab7d677a1f` |
| final frozen package combined SHA-256 | `7eee010e8cec18e54ad9e741852cf19f1b83f3a351404d20ef9e8862edd20d04` |

Those receipts are CPU/software comparisons only and do not substitute for
this gate's CUDA residency or timing run.  The gate JSON additionally records
the runtime package source identity, git HEAD/dirty state, gate-script digest,
resolved device identity, CUDA-visible environment, torch/CUDA versions, and
per-unit fitting/selection image and support hashes.

## GPU run (not available here)

No CUDA device/runtime was available locally, so no GPU pass, residency
observation, timing, equivalence, or speedup result exists.  On the target RTX,
first inventory the UUID and then run the fresh-output gate (substitute the
actual UUID; do not reuse an existing output):

```bash
nvidia-smi --query-gpu=index,uuid,name,memory.used,memory.total,utilization.gpu,utilization.memory,power.draw \
  --format=csv,noheader,nounits

CUDA_VISIBLE_DEVICES=GPU-33f45df8-0bbb-fd21-48a0-af323522bf89 \
  python scripts/check_maskfree_audit_device.py \
  --device cuda --image-size 224 --seed 42 \
  --synthetic-units 2 --warmup 1 --repeats 3 \
  --output /tmp/maskfree150-audit-device-GPU-33f45df8-0bbb-fd21-48a0-af323522bf89-224.json
```

The target command uses two distinct deterministic units to exercise the bulk
ordering path; the CPU table above records the one-unit receipt.

The expected successful status is `PASS` only if candidate content, decisions,
strict floating tolerances, complete work counters, model declaration, tensor
residency, and private heavy-helper CUDA residency all pass with a
positive-duration profiler event for a heavy CUDA token.  Any mismatch is `FAIL` and
must not be converted into a speedup claim; unavailable CUDA remains `CUDA NOT
RUN`.

The canonical `scripts/profile_maskfree.py` command runs complete scientific
training batches (producer, audit, both students, and optimizer).  It is not
the new audit-only gate timing.  For a bounded, same-config audit-device
comparison on the target RTX, run both fresh outputs with ordinary timing and
the same CUDA-visible UUID; the optimized arm must reference the CPU receipt:

```bash
RUN_TAG="$(date -u +%Y%m%dT%H%M%SZ)"
GPU_UUID="GPU-33f45df8-0bbb-fd21-48a0-af323522bf89"

CUDA_VISIBLE_DEVICES="$GPU_UUID" \
python scripts/profile_maskfree.py --config configs/maskfree_acdc_150.yaml \
  --device cuda --audit-device cpu --image-size 224 \
  --warmup 5 --measured 50 --timing-mode ordinary \
  --kind baseline --run-id "auditgpu_acdc_cpu_${RUN_TAG}" \
  --output "/tmp/maskfree150-profile-acdc-cpu-224-${RUN_TAG}"

CUDA_VISIBLE_DEVICES="$GPU_UUID" \
python scripts/profile_maskfree.py --config configs/maskfree_acdc_150.yaml \
  --device cuda --audit-device cuda --image-size 224 \
  --warmup 5 --measured 50 --timing-mode ordinary \
  --kind optimized --run-id "auditgpu_acdc_cuda_${RUN_TAG}" \
  --reference-report "/tmp/maskfree150-profile-acdc-cpu-224-${RUN_TAG}/profile_report.json" \
  --output "/tmp/maskfree150-profile-acdc-cuda-224-${RUN_TAG}"
```

These 5-warmup/50-measured calls measure complete scientific training-loop /
batch throughput with ordinary timing; they are not a 150-epoch launch and do
not measure the whole 150-epoch job including per-epoch validation, final
freeze, or export.  The full 150-epoch run remains gated/unlaunched until this
equivalence/residency review passes.

## Production configuration boundary

The production YAMLs retain `image_size: 224`, candidate bank 4, fitting
iterations 5, two audit rounds, up to 32 scored regions and 128 extra score
calls per unit, batch 8, gradient accumulation 1, and 150 epochs.  `audit_device: auto` follows the
resolved trainer CUDA device when available; explicit `cpu`/`cuda` values are
reported in the run identity.  Neural models remain FP32 and observation
arithmetic remains FP64.  CPU public topology, tensor serialization, and
NIfTI/data-loader I/O are still expected; this gate only establishes the
private tensor-heavy observation route and does not authorize an exact resume
across a changed execution device or source identity.

## Optional real-data mode and gaps

Provide a frozen image-only manifest to exercise bounded real `TrainingUnit`
loading through `load_manifest` and `ImageOnlyDataset` (no verification access):

```bash
CUDA_VISIBLE_DEVICES=GPU-33f45df8-0bbb-fd21-48a0-af323522bf89 \
python scripts/check_maskfree_audit_device.py \
  --device cuda --manifest /path/to/manifest_acdc.json \
  --split train --max-units 1 --image-size 224 --seed 42 \
  --warmup 1 --repeats 3 \
  --output /tmp/maskfree150-audit-device-acdc-GPU-33f45df8-0bbb-fd21-48a0-af323522bf89-224.json
```

No real-data manifest was supplied for this review.  The gate therefore makes
no ACDC/M&Ms, trajectory, verification-mask, clinical-efficacy, or production
throughput claim.  A real run still needs a readable frozen manifest and source
files, matching worker `ObservationModel(execution_device=...)` support, and an
RTX runtime; a missing execution declaration or private-helper residency event
fails closed.  Banks in this independent observation gate use `features=None`
(intensity-only candidate generation); this is not full producer-training
equivalence.
