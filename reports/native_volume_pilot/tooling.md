# Tooling and ablation registry delivery

Status: **IMPLEMENTED / CPU-software verified**.  This worker owns the
registry/configuration and bounded preflight, smoke, and profile commands.
No patient data, U/VAL/TEST masks, network assets, server, or GPU was accessed.

## Delivered paths

- `configs/native_volume/REGISTRY.json` declares all 23 requested rows
  (`V00..V08`, `O01..O03`, `S01..S02`, `R00..R02`, `T00..T01`, `I00..I01`,
  `C00..C01`) with one parent, changed factors, status, and explicit config
  paths and truthful update/exposure/initialization/loss/geometry/prediction
  specs.  V00 stores the historical A0 recipe and SHA-256 of
  `reports/teacher_memory_full_20261024/completed_20261025/configs/A0.json`
  (`efa18176...9938bef`); its preflight receipt marks runtime bounds
  `NOT_APPLICABLE_HISTORICAL_REFERENCE`, and smoke/profile never instantiate it.
- `configs/native_volume/*.json` are strict `self_audit.volume.v1` production
  recipes with canonical DINO provenance and fixed-window mode.  Each row makes architecture,
  geometry, loss reduction, optimizer, EMA/pseudo settings, regions/temporal
  mode, precision, microbatch/accumulation, and prediction model explicit.
  Defaults are synthetic, FP32, and at most eight training steps; DINO decoder
  rows declare canonical widths `[96,128,192,256]` and hybrid rows use
  depthwise Z kernels.  CNN3D rows declare a separate four-stage anisotropic
  scratch topology `[16,32,64,128]` with matching kernels/strides.  CPU
  smoke/profile records family-aware overrides (DINO decoder
  `[96,128,192,256] -> [8,16,24,32]`; CNN `[16,32,64,128] -> [8,16,24,32]`).
  Scientific provenance is pinned to DINO source commit
  `6876159a11b4df116f30f667f8c9888617df0751` and weight SHA-256
  `21b726bb286e037f00a23fb4699fa9bda9c75b6a615bd57ce3013cec1b528d54`;
  those assets are not present locally and are not downloaded by tooling.
- `scripts/volume_preflight.py` validates registry coverage, parent acyclicity,
  canonical DINO provenance/fixed-window policy, factors, schema, bounded
  budgets, and forbidden mask/external-teacher keys; it bootstraps the repo
  before importing the strict integration validator and never falls back to
  local checks.  It emits config/registry hashes and `NOT_RUN` evidence fields
  without loading data.
- `scripts/volume_smoke.py` builds a temporary deterministic
  `VolumeTrainingData` labeled/U fixture and executes the actual
  `VolumeTrainer` supervised -> fixed-target -> pseudo -> one-EMA transaction.
  It checks finite gradients/update counts, L-only label reads and U mask-free
  provenance, and fails closed unless enabled R/T/C factors leave region,
  temporal, or extra-supervised-update witnesses.  I00/I01 receive identical
  frozen STUDENT probabilities and explicit fixture affine/spacing: I00 has
  zero prompts and unchanged baseline, while I01 has a declared click budget,
  manual/propagated provenance, and native NIfTI/JSON export; V00 returns a
  historical-reference-only receipt without running 10k steps.
- `scripts/volume_profile.py` performs synchronized warmup-excluded timing of
  those end-to-end L/U transactions and reports effective batch volumes,
  slices, valid voxels, measured supervised/pseudo/EMA updates, wall/CUDA
  time, fixture hashes, logical batch/microbatch/accumulation, and CUDA peak
  memory/free-VRAM headroom from `cuda.mem_get_info`.  The GPU cap is
  hard-limited to 20 minutes; rates are measured, not estimated.
- `docs/native_3d/RUNBOOK.md` documents exact CLI invocations, future hash-pinned
  TRAIN command syntax, safety gates, and evidence boundaries.

## Verification receipts

Commands run from the repository root (with the required `rtk` wrapper):

Final tooling source snapshot (owned scripts plus registry):
`f77cc178cf9362b61c4b537358cdab7b9d6420e83b7d89c1071cb89b54e81f8a`.
The shared worktree base is `486fae40c490e382d58896442215476d7d32095e`; no
commit or push was performed.
Configuration snapshot (all native-volume JSON, including registry and
correction protocols): `98f8d69d86cd24b2c1f9b50b2d8b399f02e15fd59f539ff153f5b2113f601e17`.

```text
rtk python scripts/check_repo.py
PASS (architecture dinov3_convnext_tiny_native_decoder_v2; semi-supervised)

rtk python -m pytest -q tests/test_volume_registry.py tests/test_volume_cli.py
24 passed after Astra negative probes: production width/lateral mock,
automatic zero-prompt versus interactive correction, finite/huge runtime
rejection before builder allocation, exact C01 extra-update accounting,
absolute-path preflight, logical-batch, factor-control, and explicit-correction tests

Registry Astra controls: V01 now enumerates batch/prediction/schedule/horizon/
delay/exposure confounds against immutable A0; O02 enumerates optimizer family,
encoder/decoder LR, head LR, and scheduler/warmup factors. V06/V07 targeted
CNN3D smoke/profile PASS with production widths `[16,32,64,128]`, four active
anisotropic stages, and synthetic `[8,16,24,32]` overrides.
O02 now uses conservative grouped-AdamW `head_lr=3e-4` (encoder `2e-5`,
decoder `3e-5`); its optimizer/LR confound is explicitly unvalidated.

rtk python -c '...load_registry() and validate all 23 configs...'
23 registry rows preflight-pass (cwd and absolute script from /tmp)

rtk python scripts/volume_smoke.py --config configs/native_volume/V01.json --steps 1 --crop-size 32 --device cpu --max-seconds 60
PASS; actual fixture trainer transaction, output [1,4,3,32,32], supervised+pseudo updates, finite EMA teacher

rtk python scripts/volume_profile.py --config configs/native_volume/V01.json --iterations 1 --warmup 1 --crop-size 32 --device cpu --max-minutes 1
PASS; end_to_end_trainer=true, measured supervised/pseudo/EMA deltas=1/1/1,
logical_batch=1, measured L/U geometric and label-valid voxels, examples/s, GPU NOT_RUN

rtk python scripts/volume_profile.py --config configs/native_volume/V06.json --iterations 1 --warmup 1 --crop-size 32 --microbatch 1 --device cpu --max-minutes 1
PASS; CNN3D end_to_end_trainer=true, effective batch=2 volumes/10 slices,
measured valid voxels=2261, examples/s=26.58, supervised/pseudo/EMA deltas=1/1/1, GPU NOT_RUN

rtk python scripts/volume_profile.py --config configs/native_volume/C01.json --iterations 1 --warmup 1 --crop-size 64 --microbatch 1 --device cpu --max-minutes 1
PASS; supervised=2, supervised_extra=2, pseudo=0, EMA=2, effective batch=2

selected smoke arms V01..C01 (one bounded fixture transaction each; V00 historical witness only)
all 22 non-historical rows PASS in direct CPU CLI checks, including R01/R02
region, T01 motion temporal (T00 has no temporal objective), C01 actual
supervised-extra, V06/V07 3D CNN, and I01 explicit correction/export witnesses

Retained receipts: `smoke_final_V01.json`, `profile_final_V01.json`,
`tooling_smoke_R01.json`, `tooling_smoke_T01.json`,
`tooling_smoke_C01.json`, and `tooling_smoke_I01.json` under this report
directory.  The receipts are synthetic/software evidence only.
Receipts generated before the production-width, correction, runtime-bound, and
exposure-counter fixes are explicitly **SUPERSEDED**; current retained receipts
were regenerated for V01/I00/I01 after those fixes.  V00 remains a
provenance-only historical witness and is never executed.
```

## Boundaries and unresolved dependencies

Evaluation receipt repair: frozen-evaluation JSON now records declared versus
effective per-record slab/tile windows (including FOV-derived dimensions),
overlaps, spacing/FOV, flip/resampling, all inference CLI flags, and the
executed evaluation source SHA-256. Image-only configured evaluation receipts
retain `labels_read=0`; no prediction or mask behavior was changed.
Final evaluation source identity: `0b5e612d2823bb6b74d4ab1aa7db70de016e39c5db8d163ea12afbc7265c4dc3`;
owned evaluation snapshot (`evaluation.py`, CLI, and tests):
`d7524e8fc23a5f477747105cd2af98c855d470a2276646410ed8f793b37482d3`.

The profile and smoke call the integration/model/loss/data/engine APIs and fail
closed when those APIs are absent; the strict preflight imports the package
after bootstrapping the repository and has no fallback path.  The checked
integration config validator
uses strict top-level sections, so row identity/parent/factor/update/exposure
metadata remains in `REGISTRY.json`; preflight hash-checks the immutable
canonical A0 and verifies its 10000-step/1000-warmup/batch24/crop256 fields.
No real TRAIN smoke, GPU profile, development Dice, or patient efficacy was
run; a future run needs an approved hash-pinned TRAIN manifest and geometry
sidecar plus an idle PRO4000 (currently not rented).  The runbook's future
command is intentionally explicit about `--data`, `--data-sha256`, `--geometry`,
and `--geometry-sha256`.

## Modified files

`configs/native_volume/REGISTRY.json`, all 23 JSON row configs plus explicit
`I00_correction.json`/`I01_correction.json` protocol files,
`scripts/volume_preflight.py`, `scripts/volume_smoke.py`,
`scripts/volume_profile.py`, `tests/test_volume_registry.py`,
`tests/test_volume_cli.py`, and `docs/native_3d/RUNBOOK.md`.
