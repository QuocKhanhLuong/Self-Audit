# Native-volume integration receipt

Status: IMPLEMENTED (CPU/software scope only; no patient run and no GPU run).

## Owned implementation

- `self_audit/volume/config.py` validates the opt-in `self_audit.volume.v1`
  schema, rejects unknown controls and unbounded smoke budgets, and normalizes
  registry aliases without changing the historical `self_audit.semi_v2`
  configuration. Unsupported scheduler/precision aliases and model/architecture
  overlap are rejected rather than migrated, and I00/I01 correction protocol
  fields are explicit. `V00` remains an explicit historical descriptor; all
  other defaults are bounded to eight outer iterations.
- `self_audit/volume/engine.py` provides `VolumeTrainer` and `train_volume`.
  Each transaction is supervised update -> detached/fixed EMA target -> at
  most one pseudo/auxiliary update -> one EMA update.  Empty pseudo support and
  AMP overflow fail closed without optimizer or EMA mutation.  Region and
  temporal terms, when configured and available, are included in the one
  pseudo transaction and never receive an extra optimizer step; auxiliary
  support can commit the combined U update even when pseudo CE has zero
  confident voxels.  RegionTargets/region_consistency_loss are called
  directly, and temporal consistency uses a teacher prediction on the
  adjacent frame from `sample_temporal_pair`.  Supervised and pseudo logical
  batches perform real volume-weighted accumulation; CE, region and temporal
  terms retain independent supported-volume/voxel denominators, while legacy
  pooled accumulation fails before any optimizer update. A zero pseudo weight
  with no active auxiliary support skips the U optimizer entirely, and
  outer-index scheduler rates are applied before both labeled and U updates.
  Successful receipts expose finite supervised/extra/pseudo gradient norms and
  actual parameter-update norms, plus valid-voxel pseudo coverage and class
  acceptance counts/disagreement.
  The real/fixture data loop requires hash-pinned manifest and geometry
  sidecar arguments, uses L/U batches of `microbatch*accumulation` including
  adjacent temporal frames, and reads no labels from unlabeled records.
- `self_audit/volume/checkpoint.py` saves student/EMA, optimizer/scheduler/
  scaler, trainer/sampler/region state and Python/NumPy/Torch/CUDA RNG state.
  Recursive source and payload hashes are checked on resume; source identity
  covers `train.py`, canonical model/window/data/loss/evaluation modules, and
  all volume modules. Data provenance, sampler, region cache (when enabled),
  scaler, counters and RNG fields are mandatory; CPU RNG tensors are restored
  from CPU payloads. Resume compares config/data/geometry identities, preserves
  the planned scheduler horizon while treating `--steps` as a stop target, and
  writes real intermediate checkpoints at `save_every`; existing checkpoints
  are never overwritten.
- `self_audit/volume/evaluation.py` exposes frozen student-vs-EMA evaluation,
  teacher-free student export/load, and native-grid probability/support export
  before argmax.  `scripts/volume_evaluate.py` and
  `scripts/volume_export.py` provide callable frozen CLI paths.
  `legacy_compat.py` requires explicit historical source/data/payload/model and
  a complete external canonical-file map (from pinned ASSET_LINEAGE/START
  data); it hashes actual current canonical files, rejects partial/manifest-
  invented maps, and loads verified historical student/EMA/optimizer states
  directly without the current v2 source-digest gate or any monkeypatch.
- `train.py` dispatches only `self_audit.volume.v1` configs to the opt-in
  router; the historical path remains the canonical v2 engine.  Real volume
  dispatch requires `--data --data-sha256 --geometry --geometry-sha256`;
  `--resume --resume-sha256 --steps` restores and continues exact cadence.

## Verification

Owned tests:

```
rtk python -m pytest -q tests/test_volume_config.py tests/test_volume_engine.py \
  tests/test_volume_resume.py tests/test_volume_legacy.py \
  tests/test_volume_integration_followup.py tests/test_volume_cli.py
```

Result: `47 passed` (CPU) in the owned integration run. Follow-up integration
tests cover accumulation, additive auxiliary support counts, empty
support/no-decay, auxiliary gradients with zero pseudo support, C01 ordering,
scheduler timing, pseudo coverage, fixture execution, strict historical
loading and canonical-source mutation rejection. The router now rejects
native-only flags on the historical schema and legacy-only flags on the volume
schema before dispatch. C01 all-UNKNOWN extra supervision returns complete
zero metrics without a second optimizer step, while preserving one EMA update.
A bounded fixture witness
also confirms configured horizon 4 with stop target 2, real `save_every=1`
intermediate checkpoints, and exact continuous-vs-resumed student/teacher
equality through step 4.

Bounded executable checks:

- Hash-pinned synthetic fixtures completed real L/U execution through
  `train_volume`, including `microbatch*accumulation` batches and adjacent
  temporal pairs; no random B1/no-U path remains.
- A synthetic fixture path passed real L/U batches through `VolumeTrainingData`, and a T01 fixture path used adjacent-frame pairs.  A two-step run resumed to step four through `train.py --resume --resume-sha256 --steps`, restoring optimizer/scheduler/RNG cadence.
- Fresh synthetic checkpoint save/load completed on CPU with one optimizer and one outer-iteration scheduler; stale checkpoints correctly fail closed after source changes. Frozen evaluation and teacher-free export/load CLIs both completed successfully.
- Checkpoint corruption witnesses verify RNG/sampler/active-region-cache,
  optimizer/scheduler/scaler counts and state types before loading a supplied
  model; malformed payloads leave the supplied parameters unchanged. The
  actual historical fixture at
  `/var/folders/zk/7qfmzgnd1c9fqnzv4lfsm8t00000gn/T/native-volume-v2-parity-q690k5c0/handoff/final`
  loaded student, teacher and both optimizer states using the externally pinned
  ASSET_LINEAGE canonical map, with no fabricated manifest canonical field.
- `rtk python scripts/check_repo.py` returned `PASS`. The latest repository-wide
  gate returned `328 passed, 1 failed, 1 warning`; the remaining failure is
  outside these owned files in correction affine-mismatch error ordering (the
  correction CLI reports sidecar image-hash mismatch before the expected prior
  affine mismatch). Neither result is patient/GPU evidence.

Source hashes at receipt time:

Public executed-dependency digest: `c2b200f6ae1d236c337a0f7983527041a11db382b4033ed7f5a11ea49a66a23e`.

| file | SHA-256 |
|---|---|
| `self_audit/volume/__init__.py` | `43b22499363362f3a6c57ad21cf6c2cbf35d83678f9d396671b2985570a1335d` |
| `self_audit/volume/config.py` | `df656a539bc4f483ed9b8b5ea025f7378722891e0ffaaa35aec7cbfdfb09755b` |
| `self_audit/volume/engine.py` | `a10cbd855ec30592df3a648e5adfc199c2338fb546a49ebf870f51565d5b1b2e` |
| `self_audit/volume/checkpoint.py` | `9fafdf175bd0e5b6523c5cdd964bcf7b7a70972987a7f798c43b64e26aa04c5c` |
| `self_audit/evaluation.py` | `abc805c1f94e31be948720882bbeba87fdeaed430bc2cbd219d50293d66c9898` |
| `self_audit/volume/evaluation.py` | `5629115355fbcf7d9d4a3807a514fb6de667a6916bc07440dae6381c2a6f0562` |
| `self_audit/volume/legacy_compat.py` | `47a455396033fee4048bb5f2970a443007f43e7147270d11478fbd75367e8e6e` |
| `train.py` | `49a18abff36210d351ad74b09c8c47704ffc290125fbeeeef97f115da737140f` |
| `scripts/volume_evaluate.py` | `c29b3f54f2614c936c7d91288980ffde718bd06c43ddbcd54d2d8344104ec645` |
| `scripts/volume_export.py` | `e7c2ee25f8427af9009cfc5f0721221e1389cf59445b05694798332b05d72497` |
| `tests/test_volume_cli.py` | `c69a2b92052b6f737133caff8273e7f436c04c07d7a0a7cf214e6c3dc25c0746` |
| `tests/test_volume_integration_followup.py` | `30c4712a25c3e151b8fa642d15877aa179c0e463ee01a40ff2c34949a5ecd6b5` |
| `tests/test_volume_legacy.py` | `b042a0905be6870587d2b8f85d88d3ca71a76387b0a5f1c51de1b84a882bdaec` |
| `tests/test_volume_config.py` | `25735d0c6f3b5d139413c057d8e09f803985362909b7ca062f83236bc66b1aa0` |
| `tests/test_volume_engine.py` | `431e5826622d7f94df2370f4744adfbc5857d13e2f77ce2dafc01e4e16f1affc` |
| `tests/test_volume_resume.py` | `551dfc08936aaf6a4029dd073870d88aa1ee492a20db78026e64ad077931d2c4` |

## Boundaries and unresolved dependencies

- CPU/synthetic execution verifies tensor shapes, finite gradients, transaction
  ordering, source-pinned resume and explicit compatibility checks only.  It is
  not patient Dice, anatomy, calibration, clinical, or GPU throughput evidence.
- A real run needs an existing hash-pinned TRAIN manifest plus versioned
  geometry sidecar and a verified non-synthetic model/backbone.  Missing assets
  must remain `NOT_RUN`; no download, license acceptance, or server access was
  attempted.
- Tooling/evaluation owners must route I00/I01 correction workflows from the
  explicit validated `correction` section (not config filenames), and the
  evaluation owner must continue importing the public `source_sha256` helper.
  The current correction exporter/CLI affine failures remain outside this
  integration ownership and must be resolved by that owner.
  No GPU, patient training, or external asset retrieval was attempted.
- Current tooling configs still use tiny decoder widths in production paths;
  ctx_b826a43cc764/ctx_afc99ff2def0 must update them to DINO widths
  `[96,128,192,256]` with explicit tiny synthetic overrides. V08 must declare
  configured physical resampling rather than native-Z plus spacing. These are
  outside this integration owner’s file allowlist and remain unresolved.

## Final strict-validation repair (Luna, 2026-10-26)

The final narrow repair closes the semantic checkpoint/configuration findings
from the independent Astra review. Checkpoint load now validates Python,
NumPy, Torch and applicable CUDA RNG payloads with isolated generators before
restoring a caller-supplied model or process-global RNG; malformed states fail
with `ValueError`, and CUDA state is rejected when it cannot be restored rather
than silently dropped. Checkpoint provenance now requires an exact `bool`
`synthetic` field and non-null 64-hex `manifest_sha256` and `geometry_sha256`
pins on both save and load. Configured geometry requires finite positive
`spacing_xyz_mm`; `configured_physical_fov` additionally requires an explicit
`fov_xyz_mm`; native modes continue to reject declared spacing.

Focused verification:

```
rtk python -m pytest -q tests/test_volume_config.py tests/test_volume_resume.py \
  tests/test_volume_engine.py tests/test_volume_integration_followup.py \
  tests/test_volume_cli.py
```

Result: **57 passed** in 5.99s (CPU), including the 16-test native CLI suite.
The resume tests reseal malformed
Python/NumPy/Torch RNG payloads and verify unchanged supplied weights and
global Python/NumPy/Torch RNG states; provenance tests use explicit manufactured
hex pins. The `train_volume` resume path preflights exact config/data/synthetic
identities before checkpoint RNG restoration, and its identity-mismatch witness
leaves global RNG streams unchanged. `rtk python scripts/check_repo.py` returned
`PASS`. No patient, GPU, server, download, or long training was run.

Strict-repair source hashes:

| file | SHA-256 |
|---|---|
| `self_audit/volume/checkpoint.py` | `3245000e3b894167ce86dafd6717560b40cdd4ced5829e89d6caf83cbe4fe80a` |
| `self_audit/volume/config.py` | `23e9acf9d514d7f36665c9089bc0220a95ab393196e45cd47636f0e3aeea7673` |
| `self_audit/volume/engine.py` | `13a7b82ee75e12ded3d7e08f292ec26f14e85034f048e72e5ee7b7da2f6fdf27` |
| `tests/test_volume_resume.py` | `0d9b77dd3984c01e34bf77548b532dd42747966cdd19268601353369b06631d5` |
| `tests/test_volume_config.py` | `1ab6563a4f414f5f8497203fa61fe14d1ffb7d9418c11aba9d352e3ec632e6f9` |

The narrow engine edit is limited to preflighting exact config/data/synthetic
identity before `load_volume_checkpoint` can restore RNG; accepted transaction
ordering and horizon/sampler/cache/optimizer/scaler counters are unchanged.
Remaining production registry width/resampling edits and unrelated
correction/evaluation work stay outside this allowlist.
