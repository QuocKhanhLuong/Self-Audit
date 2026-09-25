# STEGO and PiCIE Shared-Benchmark Port Plan

## Goal

Make STEGO and PiCIE first-class producers in the same cardiac benchmark as
CUTS and DFC.  Each producer must consume the frozen image-only shared
manifest, use its declared whole-FOV grid, create a sealed raw partition with
the common artifact layer, and optionally hand that partition to the existing
post-freeze semantic adapter.

The target result is four comparable producer commands.  It is not a rewrite
of either research baseline or a retraining project.

## Authoritative starting point

Create the implementation branch from `main@817f2ba`.

Use the following snapshots only as source references:

| Reference | Commit | Intended use |
| --- | --- | --- |
| `baseline/stego-picie-recovered` | `9f9545d` | Preserve useful cardiac wrapper intent, existing local tests, and documented recovery decisions. |
| `baseline/stego-picie-clean` | `8ec3942` | Restore original STEGO/PiCIE source files that the recovered wrapper imports but does not carry. |

Do not merge either snapshot into `main`.  They predate the current
shared-benchmark contract and can overwrite or bypass hardened CUTS/DFC code.
Before changing code, record the exact source commit and path for every file
copied from a reference snapshot in the port PR description.

## Non-negotiable contract

The port must meet these conditions before a scientific result is accepted:

- The only producer input is a `shared_benchmark_manifest.v1` record and the
  separately mounted image-only source root.
- Sample membership, patient split, context `[z-1,z,z+1]`, source hash,
  full-FOV grid, and spatial transform come from the record.  A baseline may
  choose its declared channels and intensity normalization, but may not
  rediscover files, construct a second split, crop from image content, or
  resize through an independent path.
- Producer code does not open masks, labels, metrics, Hungarian mapping, or
  semantic class names.  It emits an anonymous raw partition only.
- All generated output goes through `shared_benchmark.artifacts.run_generation`.
  Raw identity must bind the manifest, shared-grid, source image, checkpoint,
  effective config, source code, repository, seed, and execution receipt.
- The common `cardiac_adapter_v1` runs only after the raw artifact is sealed.
  The producer never implements its own semantic conversion.
- `--device cpu` must work on a bounded fixture without an implicit CUDA move.

## Task 0: establish the import boundary

Files:

- Create `docs/stego-picie-port-provenance.md`.
- Add the needed original source under `baseline/STEGO/` and `baseline/PICIE/`.
- Do not modify shared CUTS/DFC code in this task.

Implement:

1. Compare `recovered` and `clean` file inventories.  Import only files needed
   by the callable STEGO/PiCIE paths; retain licenses, notices, and upstream
   README attribution.
2. Keep original research entry points as legacy entry points.  New cardiac
   code must live in isolated `cardiac_benchmark` modules and top-level
   scientific runner scripts.
3. Make every import deterministic from repository root.  New runners add
   repository `src` and the relevant baseline source directory explicitly;
   they must not rely on the current directory or an operator-set
   `PYTHONPATH`.
4. Verify that every `modules.*`, `utils.*`, DINO, FPN, and checkpoint import
   resolves from the ported tree before writing adapters.

Acceptance:

- A source/provenance table identifies each imported file and its source
  snapshot.
- Import smoke tests can import the cardiac entry modules from repository root.

## Task 1: add image-only shared-manifest views

Files:

- Create `baseline/STEGO/src/cardiac_benchmark/dataset.py` and
  `baseline/STEGO/src/cardiac_benchmark/manifest.py`.
- Create `baseline/PICIE/src/cardiac_benchmark/dataset.py` and
  `baseline/PICIE/src/cardiac_benchmark/manifest.py`.
- Create focused tests under `baseline/STEGO/tests/cardiac/` and
  `baseline/PICIE/tests/cardiac/`.

Implement one narrow view per baseline:

1. Load with `load_shared_manifest`; reject every schema other than
   `shared_benchmark_manifest.v1`.
2. Select records only by declared `split` and `sample_id`.
3. Decode with `read_context_stack(record, source_root=...)` and resample with
   `resize_values_to_grid(..., record["shared_grid"])`.
4. Return the image tensor and a provenance dictionary containing the shared
   identity fields, source-image SHA-256, transform, normalization version,
   target shape, and grid hash.
5. Preserve method fidelity explicitly:
   - STEGO may create its three-channel input from the declared stack or use a
     documented centre-only profile, but it must derive it after canonical
     stack decoding and before/through the shared resize.
   - PiCIE must receive its required three-channel image through the same
     canonical stack/grid path.  Its old `metadata.json`, `volumes/`,
     `masks/`, and private split-manifest readers are legacy-only and may not
     be imported by the scientific path.
6. Never construct PIL labels, inspect a mask directory, or resize labels in
   these modules.

Tests must prove both consumers select the same record identity and grid as
CUTS/DFC, preserve endpoint replication, reject a grid override, and reject a
manifest without a valid image-only locator/hash.

## Task 2: make STEGO device-safe

Files:

- Update the callable DINO/model loading paths in `baseline/STEGO/src/`.
- Create `baseline/STEGO/src/cardiac_benchmark/stego_runner.py`.
- Add CPU/device tests under `baseline/STEGO/tests/cardiac/`.

Implement:

1. Remove direct `.cuda()` calls from `DinoFeaturizer.__init__`, model-loading
   helpers, and the new cardiac callable path.  Construction remains device
   neutral.
2. Pass an explicit `torch.device` into the cardiac loader/runner and perform
   the sole `model.to(device)` operation at runner setup.  Inputs follow the
   same device.
3. Load all checkpoints with `map_location="cpu"`, validate their expected
   keys, then move the fully loaded model once.
4. Keep legacy training/demo scripts out of the scientific runner; do not
   refactor unrelated original entry points merely to remove their historical
   GPU assumptions.

Tests must monkeypatch CUDA as unavailable and show construction, checkpoint
loading through a fixture, and one bounded producer call succeed on CPU.

## Task 3: isolate PiCIE raw-partition inference

Files:

- Create `baseline/PICIE/src/cardiac_benchmark/picie_runner.py`.
- Add a minimal checkpoint compatibility helper beneath the same package.
- Add tests under `baseline/PICIE/tests/cardiac/`.

Implement:

1. Reuse only PiCIE backbone/FPN and cluster-head inference needed to obtain
   an anonymous partition.  Keep training, centroid fitting policy, and
   checkpoint interpretation explicit in a validated effective config.
2. Eliminate the scientific-path dependency on `eval_picie.py`,
   `medical_metrics.py`, dataset mask readers, and Hungarian remapping.
3. Return an integer partition at the shared grid resolution.  Do not restore
   it to native geometry inside the producer.
4. Use a caller-provided device and `map_location="cpu"`.  `DataParallel` is
   optional; the CPU path must not require it.
5. Treat the existing PiCIE evaluation code as legacy.  The defect where it
   resizes images to 224 while retaining native-size labels must have a
   regression test demonstrating that the new producer cannot recreate that
   mixed-resolution flow.

Tests must cover checkpoint key normalization, CPU inference with a tiny mock
backbone, output dtype/shape, and no import/reachability of mask, metric, or
Hungarian modules from the producer path.

## Task 4: implement scientific orchestration

Files:

- Create `scripts/run_stego_scientific.py`.
- Create `scripts/run_picie_scientific.py`.
- Add runner tests in `tests/shared_benchmark/` and baseline cardiac test
  directories.

Make both runners mirror the externally visible CUTS/DFC runner contract:

```text
validated scientific manifest + image-only root + checkpoint + effective config
    -> method-specific image-only view and raw partition
    -> run_generation raw artifact and terminal ledger
    -> optional frozen shared adapter and semantic artifact
```

Required CLI arguments:

- `--manifest`, `--image-root`, `--output-root`, `--checkpoint`, `--split`;
- `--device`, `--limit`, `--sample-list`, `--config-hash`;
- `--apply-adapter`, `--adapter-spec`, `--semantic-root`, and
  `--no-retry-failed` with the same meanings as CUTS/DFC.

Required runner behavior:

1. Call `validate_scientific_execution` and reject any non-frozen 224x224
   whole-FOV grid before a checkpoint is loaded.
2. Verify the effective configuration hash and checkpoint SHA-256.  Include
   upstream source identity and every newly ported source file in
   `code_identity`.
3. Use `select_manifest_records`; fail if the baseline view inventory cannot
   account for every selected shared record.
4. Use `run_generation` for raw sealing, retry/resume, terminal accounting,
   execution receipt, and optional adapter invocation.  Do not retain the old
   baseline bundle/seal implementation as a parallel scientific protocol.
5. Record method-specific but non-semantic metadata only, such as architecture,
   feature/profile choice, clustering hyperparameters, checkpoint identity,
   seed, and device/runtime details.

## Task 5: add cross-baseline gates

Files:

- Extend `tests/shared_benchmark/test_baseline_consumers.py`.
- Extend `tests/shared_benchmark/test_execution_artifacts.py` only where a
  generic shared invariant is missing.
- Update `.github/workflows/self-audit-ci.yml`.

Add the following tests:

1. CUTS, DFC, STEGO, and PiCIE consume an identical fixture record with the
   same sample/patient/split/frame/slice/context identity and shared grid.
2. STEGO and PiCIE producer runners run one synthetic sample on CPU, seal a
   raw artifact, then resume without regenerating a valid artifact.
3. A checkpoint hash or code-identity change invalidates the prior raw result.
4. `--apply-adapter` creates a semantic output bound to the raw artifact and
   frozen adapter specification; a plain run creates no semantic claim.
5. Static/reachability tests reject GT-shaped fields and producer imports of
   labels, masks, Dice, IoU, HD95, ASSD, or Hungarian mapping.
6. PiCIE's fixture uses a non-224 native shape and proves that every producer
   tensor and raw partition remains on the declared shared grid.

CI must execute these tests, rather than merely compiling `src`, `scripts`,
and `tests`.  Keep real checkpoints and clinical data out of CI.

## Task 6: document execution and handoff

Files:

- Update `README.md` and `docs.md`.
- Create `reports/STEGO_PICIE_PORT_AUDIT.md` after implementation.

Document:

- required image-only mount and checkpoint paths;
- exact commands for fixture CPU smoke, preflight, raw generation, resume, and
  optional semantic handoff;
- declared STEGO and PiCIE profiles and all method-specific preprocessing;
- what remains deferred until a server-side image-only preflight and runtime
  measurement are completed.

The final audit must state evidence, command results, commit hashes, artifact
identities, and whether real data were used.  It must not claim real-data or
clinical readiness based on synthetic tests.

## Verification order

After each task, run the relevant focused tests.  Before merge, run:

```text
python -m pytest -q tests/shared_benchmark baseline/STEGO/tests/cardiac baseline/PICIE/tests/cardiac
python -m compileall -q src scripts baseline/STEGO/src baseline/PICIE
git diff --check
python scripts/run_stego_scientific.py --help
python scripts/run_picie_scientific.py --help
```

Then run one bounded synthetic CPU generation for each baseline, inspect the
raw ledger and hashes, rerun it to verify resume, and repeat with
`--apply-adapter`.  A server run is a separate later gate: it requires an
image-only mount, validated scientific manifest, supplied checkpoint, and a
bounded runtime preflight.

## Commit checkpoints

1. `baseline: import pinned STEGO/PiCIE sources and provenance`.
2. `baseline: add shared-manifest image-only views`.
3. `baseline: make STEGO cardiac inference device-safe`.
4. `baseline: isolate PiCIE raw-partition inference`.
5. `benchmark: add STEGO/PiCIE scientific artifact runners`.
6. `test: gate four-baseline consumer and artifact parity`.
7. `docs: record STEGO/PiCIE execution protocol and audit`.

## Merge gate

Do not merge until all four baselines share manifest and grid identity in
tests; STEGO completes CPU fixture inference without implicit CUDA; PiCIE has
no label/metric/Hungarian reachability in producer code; raw artifacts seal,
resume, and invalidate correctly; and CI executes the new integration suite.
