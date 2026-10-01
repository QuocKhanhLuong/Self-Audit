# PiCIE in This Repo

This directory serves two roles:

1. It preserves a minimal upstream PiCIE source snapshot for provenance and
   model loading.
2. It hosts the repo-specific cardiac shared-benchmark port used by
   `Self-Audit`.

## What Is Authoritative Here

For this repository, the active PiCIE baseline is the scientific producer path:

- `scripts/run_picie_scientific.py`
- `baseline/PICIE/src/cardiac_benchmark/`
- `baseline/PICIE/tests/cardiac/`
- `src/shared_benchmark/`
- `docs/stego-picie-port-provenance.md`

Those files define how PiCIE consumes the shared manifest, validates the frozen
grid/profile contract, loads a checkpoint, generates anonymous partitions, and
publishes shared-benchmark artifacts.

## What Is Not Authoritative

Most of the remaining vendor tree is kept as upstream reference material:

- legacy PiCIE support modules at the baseline root
- alternate module namespaces such as `picie/` and `modules/`
- non-scientific training/evaluation assumptions from the original project

Changes outside the scientific producer path should not be treated as changes to
the active benchmark baseline unless the provenance document is updated to say so.

## Role in the Monorepo

PiCIE is one of several external baselines wired into the shared cardiac
benchmark. In this repo it is not the main product; it is a controlled baseline
consumer used to:

- read the shared benchmark manifest
- generate raw anonymous partitions on the frozen grid
- optionally hand off to the shared semantic adapter for SA224 profiles
- provide cross-baseline contract evidence alongside CUTS, DFC, and STEGO

## Fair Variant Entry Points

The fair ACDC variant work is now split into explicit repo-owned entry points:

- `scripts/train_picie_sa224_fair.py` trains a train-only ACDC fair checkpoint
  with a frozen PiCIE-style augmentation preset and emits a matching
  checkpoint-contract sidecar.
- The scientific producer path under `baseline/PICIE/src/cardiac_benchmark/`
  remains the authoritative inference path and can consume the fair checkpoint
  once the sidecar contract is supplied.

## Reporting Policy

- PiCIE changes should be documented in PiCIE-specific reports, not merged into
  a combined STEGO/PICIE memo.
- The current branch-local audit report is
  `reports/PICIE_REAUDIT_2026-09-21.md`.
