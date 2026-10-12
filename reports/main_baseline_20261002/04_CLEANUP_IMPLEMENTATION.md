# Cleanup implementation

Date: 2026-10-02. Base: `c31825a7f90df47f9f9382a9c9595e43f7f71236`.
Before publication, integrated the new main tip
`53fa6182ff8f5d262c2a45a8644e9b850a4d5c49`, preserving its 12-file ADNet addition.
It does not alter the audited v3 model or config. This cleanup's allowlist and
ten deletions remain unchanged.
Status: **IMPLEMENTED** by the coordinator after the follow-up Orca worker
failed to start/authenticate. Validation is recorded separately in
[05_VALIDATION.md](05_VALIDATION.md); this report is not an independent review.

## Changes

Removed exactly ten tracked files, totaling 72,877 bytes and 1,804 source lines
before extraction. The [deletion manifest](cleanup_manifest.json) records each
path, byte/line count and original SHA-256:

- `scripts/train_self_audit_legacy.py`
- `scripts/run_full_pipeline_legacy.sh`
- `scripts/run_full_pipeline_legacy.ps1`
- `baseline/CUTS/src/datasets/tempCodeRunnerFile.py`
- `baseline/.gitkeep`
- `baseline/CUTS/.gitkeep`
- `baseline/DFC/.gitkeep`
- `baseline/PICIE/.gitkeep`
- `baseline/STEGO/.gitkeep`
- `baseline/STEGO/src/wget-log`

Before removing the old Python executable, moved the five still-used calibration
and checkpoint-binding functions plus two constants into
`scripts/self_audit_post_training.py`. Every function/constant AST is identical
to the pinned original, including defaults and fallback behavior. The new
module has only the imports needed by those helpers and no training CLI.

The canonical runner re-exports the same helper names from the new module.
Checkpoint-binding and calibration-lineage tests now import/patch the new
module. The retired runner's CLI-acceptance test was removed because that CLI is
intentionally no longer supported; canonical CLI rejection tests remain, with
their documentation target updated. No behavioral assertions were relaxed.

Current Bash/PowerShell/Python help messages point to historical reproduction
documentation instead of deleted executables. The reproduction guide pins an
immutable Git revision and explains the difference between live-weight
single-process and checkpoint-reloading multi-process recipes.

README now leads with the fully no-GT v3 flow, target RTX 4080 Super 16GB, and
unverified quality/speed objectives. Existing supervised commands are explicitly
labeled as the reference baseline. The cleanup removes obsolete executable
entrypoints; it does not remove active supervised dependencies or comparison
methods merely because they are not the primary no-GT research candidate.

## Preserved contracts

- No changes to `src/`, model forward, losses, dataset code or configs.
- Active phase modules/YAMLs remain; they are used by current code, tests and
  frozen benchmark provenance.
- `UnifiedTrainer`'s strict selected-`best.pt` calibration is unchanged. The
  extracted historical helpers preserve their distinct, declared fallback
  semantics only for compatibility with historical artifacts.
- CUTS/DSS-US/SGSCN and historical comparison families remain. CUTS diffusion is
  comparison code, not the v3 student path. Frozen benchmark files, evaluation
  evidence, checkpoints and datasets were not deleted or regenerated.
- The main checkout at `/Users/alvinluong/Self-Audit` has unrelated dirty work.
  All integration changes were made in `/private/tmp/self-audit-main-20261002`,
  based on fetched current main; unrelated changes were not staged.

Deleting source files does not establish faster inference. Parameter count and
CPU timing are separate engineering measurements; no inference algorithm was
optimized in this cleanup.
