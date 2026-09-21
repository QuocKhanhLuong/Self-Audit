# STEGO in This Repo

This directory serves two roles:

1. It preserves a minimal upstream STEGO source snapshot for provenance and
   model loading.
2. It hosts the repo-specific cardiac shared-benchmark port used by
   `Self-Audit`.

## What Is Authoritative Here

For this repository, the active STEGO baseline is the scientific producer path:

- `scripts/run_stego_scientific.py`
- `baseline/STEGO/src/cardiac_benchmark/`
- `baseline/STEGO/tests/cardiac/`
- `src/shared_benchmark/`
- `docs/stego-picie-port-provenance.md`

Those files define how STEGO consumes the shared manifest, validates the frozen
grid/profile contract, loads a checkpoint, generates anonymous partitions, and
publishes shared-benchmark artifacts.

## What Is Not Authoritative

Most of the remaining vendor tree is kept as upstream reference material:

- generic STEGO training/evaluation scripts under `baseline/STEGO/src/`
- upstream configs and demo assets
- DINO support code needed by the scientific runner

Changes outside the scientific producer path should not be treated as changes to
the active benchmark baseline unless the provenance document is updated to say so.

## Role in the Monorepo

STEGO is one of several external baselines wired into the shared cardiac
benchmark. In this repo it is not the main product; it is a controlled baseline
consumer used to:

- read the shared benchmark manifest
- generate raw anonymous partitions on the frozen grid
- optionally hand off to the shared semantic adapter for SA224 profiles
- provide cross-baseline contract evidence alongside CUTS, DFC, and PiCIE

## Reporting Policy

- STEGO changes should be documented in STEGO-specific reports, not merged into
  a combined STEGO/PICIE memo.
- The current branch-local audit report is
  `reports/STEGO_REAUDIT_2026-09-21.md`.
