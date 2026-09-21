# STEGO Re-audit — 2026-09-21

## Summary

- Branch: `baseline/stego-picie-recovered`
- Head when audited: `46efa4c`
- Baseline: `STEGO`
- Scope: scientific producer path only
- Audit conclusion: runnable shared-benchmark consumer, but **not faithful to the intended STEGO variant**

This re-audit treats the STEGO baseline as the shared-benchmark consumer defined
by `scripts/run_stego_scientific.py`, `baseline/STEGO/src/cardiac_benchmark/`,
`baseline/STEGO/tests/cardiac/`, `src/shared_benchmark/`, and
`docs/stego-picie-port-provenance.md`.

The root `README.md` is monorepo-level `Self-Audit` documentation and is not
the authoritative context for the STEGO baseline.

## Source of Truth

The active STEGO baseline in this repo is bounded to:

- `scripts/run_stego_scientific.py`
- `baseline/STEGO/src/cardiac_benchmark/config.py`
- `baseline/STEGO/src/cardiac_benchmark/dataset.py`
- `baseline/STEGO/src/cardiac_benchmark/stego_runner.py`
- `baseline/STEGO/tests/cardiac/`
- `tests/shared_benchmark/test_baseline_consumers.py`

`docs/stego-picie-port-provenance.md` correctly describes the import boundary:
the scientific path may use the STEGO cardiac wrapper, minimal upstream model
source, and `src/shared_benchmark/`, but not legacy data readers, legacy metric
remapping, or broader vendor logic.

## Current Behavior

The current STEGO scientific entrypoint:

1. validates the shared manifest and frozen grid/profile contract
2. selects records through shared-benchmark utilities
3. seals the effective baseline configuration hash
4. computes repository/code identity for artifact provenance
5. loads the upstream `DinoFeaturizer` through the constrained cardiac runner
6. generates raw partitions on the shared target grid
7. optionally publishes semantic handoff artifacts for SA224 profiles only

The dataset adapter converts only the shared-owned central slice into a
three-channel image for compat mode, and uses the Self-Audit 224 contract for
SA224 mode. The runner emits topology metadata and checkpoint identity per
sample, which aligns with the shared artifact contract.

## Findings

No blocking shared-benchmark contract defect was found in the reviewed STEGO
scientific path. The code runs and produces contract-valid raw and semantic
artifacts.

However, the current STEGO producer is **method-mismatched** relative to the
intended baseline variant. The intended variant keeps the DINO recipe and then
trains a STEGO head on train images only, with train-only support/KNN protocol.
The current scientific runner does not do that. Instead, it loads
`DinoFeaturizer(... pretrained_weights=checkpoint)` and emits `code.argmax(...)`
directly as the anonymous partition.

That means the current low Dice on ACDC must not be interpreted as evidence
about the intended STEGO baseline. It is evidence only about the current compat
producer implementation.

## Runtime Evidence

The committed shared-benchmark summary report under
`reports/shared_benchmark_sa224_dev2/stego_report/` shows:

- `metric_contract = foreground_dice_exclude_v1`
- `volume_metric_contract = foreground_dice_volume_resized_v1`
- `final_foreground_macro_dice = 0.0`
- `coverage_mean = 0.17251275510204084`

Slice-level inspection on the same report showed the semantic output is usually
`BG/VOID` only after the shared resolver. That failure is consistent with the
current producer emitting partitions that do not preserve cardiac enclosure
structure well enough for the frozen topology-only adapter.

## Verification

The following repository evidence currently anchors this baseline:

- `baseline/STEGO/tests/cardiac/test_stego_shared_manifest_view.py`
- `baseline/STEGO/tests/cardiac/test_contract_artifacts_stego.py`
- `baseline/STEGO/tests/cardiac/test_scientific_hardening_stego.py`
- `baseline/STEGO/tests/cardiac/test_smoke_stego.py`
- `tests/shared_benchmark/test_baseline_consumers.py`

These tests collectively protect shared-manifest consumption, artifact contract
shape, scientific-entrypoint hardening, and cross-baseline identity consistency.
They do **not** certify that the current producer is faithful to the intended
STEGO training/evaluation recipe.

## Cleanup Notes

This branch-local cleanup removed local/generated STEGO artifacts that were not
part of the tracked scientific contract, including cache directories, local
checkpoint/result dumps, and the duplicate scratch copy outside the baseline
tree.

## Follow-up Rules

- Keep STEGO reporting separate from PiCIE reporting.
- Treat changes under `scripts/run_stego_scientific.py`,
  `baseline/STEGO/src/cardiac_benchmark/`, and `baseline/STEGO/tests/cardiac/`
  as baseline-contract changes.
- Treat broader vendor-tree edits as provenance or reference edits unless the
  provenance boundary is intentionally expanded.
- Do not present current STEGO ACDC Dice as a faithful baseline number until
  the producer is changed to the intended variant protocol.
