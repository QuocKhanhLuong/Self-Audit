# PiCIE Re-audit — 2026-09-21

## Summary

- Branch: `baseline/stego-picie-recovered`
- Head when audited: `46efa4c`
- Baseline: `PICIE`
- Scope: scientific producer path only
- Audit conclusion: runnable shared-benchmark consumer, closer to the intended PiCIE variant, but still only a **compat/public-checkpoint** run

This re-audit treats the PiCIE baseline as the shared-benchmark consumer defined
by `scripts/run_picie_scientific.py`, `baseline/PICIE/src/cardiac_benchmark/`,
`baseline/PICIE/tests/cardiac/`, `src/shared_benchmark/`, and
`docs/stego-picie-port-provenance.md`.

The root `README.md` is monorepo-level `Self-Audit` documentation and is not
the authoritative context for the PiCIE baseline.

## Source of Truth

The active PiCIE baseline in this repo is bounded to:

- `scripts/run_picie_scientific.py`
- `baseline/PICIE/src/cardiac_benchmark/config.py`
- `baseline/PICIE/src/cardiac_benchmark/dataset.py`
- `baseline/PICIE/src/cardiac_benchmark/picie_runner.py`
- `baseline/PICIE/tests/cardiac/`
- `tests/shared_benchmark/test_baseline_consumers.py`

`docs/stego-picie-port-provenance.md` correctly describes the import boundary:
the scientific path may use the PiCIE cardiac wrapper, minimal upstream model
source, and `src/shared_benchmark/`, but not broader legacy reader/metric code.

## Current Behavior

The current PiCIE scientific entrypoint:

1. validates the shared manifest and frozen grid/profile contract
2. selects records through shared-benchmark utilities
3. seals the effective baseline configuration hash
4. computes repository/code identity for artifact provenance
5. loads the upstream FPN/classifier path through the constrained cardiac runner
6. generates raw partitions on the shared target grid
7. optionally publishes semantic handoff artifacts for SA224 profiles only

The dataset adapter uses the shared three-slice context stack in compat mode and
the Self-Audit 224 contract in SA224 mode. Scientific mode also locks
augmentation off and fixes `K_train = K_test = 4`, which is closer to the
intended PiCIE clustering variant than the current STEGO producer is to STEGO.

## Findings

No blocking shared-benchmark contract defect was found in the reviewed PiCIE
scientific path. The code runs and produces contract-valid raw and semantic
artifacts.

The main limitation is not contract invalidity but **evaluation regime**. The
current run is a `benchmark_tier = compat` public-checkpoint run, not a fair
in-domain ACDC checkpointed run. The repository currently has only the public
checkpoint file and no fair checkpoint sidecar declaring `in_domain_acdc`.

That means the current low Dice on ACDC must not be treated as a clean number
for an in-domain PiCIE baseline. It is a valid compat result, but still
confounded by checkpoint/domain mismatch.

## Runtime Evidence

The committed shared-benchmark summary report under
`reports/shared_benchmark_sa224_dev2/picie_report/` shows:

- `metric_contract = foreground_dice_exclude_v1`
- `volume_metric_contract = foreground_dice_volume_resized_v1`
- `final_foreground_macro_dice = 0.0`
- `coverage_mean = 0.695230787627551`

Even with much higher semantic coverage than STEGO, slice-level inspection on
the same report showed the semantic output is still typically `BG/VOID` only,
with no persistent `RV/MYO/LV` emission after the shared resolver. That points
to raw partitions that are too coarse or anatomically mismatched for the frozen
MRI resolver to recover cardiac roles.

## Verification

The following repository evidence currently anchors this baseline:

- `baseline/PICIE/tests/cardiac/test_picie_shared_manifest_view.py`
- `baseline/PICIE/tests/cardiac/test_contract_artifacts_picie.py`
- `baseline/PICIE/tests/cardiac/test_scientific_hardening_picie.py`
- `baseline/PICIE/tests/cardiac/test_smoke_picie.py`
- `tests/shared_benchmark/test_baseline_consumers.py`

These tests collectively protect shared-manifest consumption, artifact contract
shape, scientific-entrypoint hardening, and cross-baseline identity consistency.
They do **not** certify that the current checkpoint provenance matches a fair
in-domain ACDC baseline.

## Cleanup Notes

This branch-local cleanup removed local/generated PiCIE artifacts that were not
part of the tracked scientific contract, including cache directories, logs, and
local smoke-test outputs.

## Follow-up Rules

- Keep PiCIE reporting separate from STEGO reporting.
- Treat changes under `scripts/run_picie_scientific.py`,
  `baseline/PICIE/src/cardiac_benchmark/`, and `baseline/PICIE/tests/cardiac/`
  as baseline-contract changes.
- Treat broader vendor-tree edits as provenance or reference edits unless the
  provenance boundary is intentionally expanded.
- Do not present current PiCIE ACDC Dice as a fair in-domain baseline number
  until the checkpoint provenance satisfies the fair checkpoint contract.
