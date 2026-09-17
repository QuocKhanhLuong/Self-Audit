# Maskfree150 implementation review

## Scope and execution status

Authority: [complete implementation brief](../../self_audit_maskfree_150_orca_prompt.md)
and the later instruction: the RTX 4070 is occupied; finish code, rent hardware
for testing later. Work remained on the existing `main`; no worktree was created.
The supervised Self-Audit and Candidate C implementations were preserved.

| Item | Status | Evidence or reason |
|---|---|---|
| New namespace and integration | COMPLETED | `src/self_audit_maskfree`; root-reviewed |
| Focused CPU software checks | COMPLETED | 45 combined passes; 5 affected checks passed after final reference hardening |
| Actual dataset inventory / full cine availability | NOT INSPECTED / UNKNOWN | Linux dataset roots are absent from this local checkout; remote inspection was deferred |
| Actual RTX 4070 physical batch 8 | NOT STARTED | Deferred by user; no GPU or MPS used |
| Native ACDC, 150 global epochs | NOT STARTED | Deferred by user |
| Independent native M&Ms, 150 global epochs | NOT STARTED | Deferred by user |
| Real reference evaluation | NOT STARTED | Separate optional process after all compared predictions freeze |

Runtime defaults point to
`/home/linhdang/workspace/quockhanh_workspace/SpecMamba`.
Code and retained CPU fixtures are in `/Users/alvinluong/Self-Audit`.
The earlier SSH sandbox denial and interrupted retry do not establish a remote
authentication or hardware failure. No claim is made about the remote data.

## Implemented behavior

- Image-only NIfTI/NPY discovery; patient splits; all acquired Z/T units; exact
  frame/volume identities; conservative duplicate detection; mixed native grids
  fail closed. NPY exports remain stored-grid exports. Full cine is never inferred
  from paired ED/ES images.
- Fit-only normalization and producer input; guarded spatial observation blocks
  shared across neighboring slices; typed fitting/selection/verification access.
  No baseline imports, pretrained teachers, mask-derived crop, threshold, or
  checkpoint selection in the generating path.
- Randomly initialized producer plus identically initialized independent
  students, trained within one continuous timeline and global schedule. Bounded
  bank generation/auditing starts at epoch zero. Ignored targets produce no
  optimizer or weight-decay update; accumulated student gradients normalize by
  actual valid support.
- Common-bank E0–E5, feature-grouping control, negative controls, and equal-budget
  random-edit comparison. Fixed-bank E4/E5 ties remain ties. Each compared fit,
  prediction and checkpoint is frozen before verification.
- Versioned unit and native-volume label/probability/validity exports; all-volume
  freeze completeness; separate full-input student deployment freeze. Verification
  uses per-unit atomic cached receipts and a read-only freeze session with full
  hash validation on entry and exit.
- Image-only NLL with raw denominators; patient-level aggregation/bootstrap;
  split-specific reports; real natural/matched-support student NLL; degenerate
  challenges; perturbation sensitivity; pre-freeze repeated annotation under
  fixed fit-only noise. Candidate agreement is a separate diagnostic.
- Atomic checkpoints with optimizer/scaler/RNG/sampler, exact cursor, partial
  epoch accounting, source/config/data/device identity, and log recovery.
  A capped resume cannot overshoot its step limit. Successful finalization
  preserves frozen bytes and archives prior failure receipts.
- Sequential launchers with explicit run IDs, actual GPU UUID resolution,
  physical batch-8 preflight, explicit `4x2`/`2x4` fallback before execution,
  immutable resolved config, and strict completion gates. No automatic batch,
  resolution, learning-rate, or checkpoint substitution.

## Material limits

This is an executable **spatial_predictive v1** implementation. Temporal transport,
temporal cycles, and biomechanical evidence are unavailable; all acquired frames
may be processed spatially. The ontology runs per unit; study-level continuity
is not implemented as an anatomical resolver. These limitations are recorded,
not substituted with invented measurements.

Appearance NLL and repeat stability are not segmentation accuracy. Semantic
ambiguity and zero useful foreground coverage remain possible; runtime completion
and label-generation scientific outcome have separate fields. No real accuracy,
GPU capacity/throughput, clinical utility, or novelty result has been established.
Published CUTS reproduction remains pending; the executable control is explicitly
named `cuts_inspired_control`.

Reference evaluation is isolated and requires fixed raw-label-to-named-role
mapping and unambiguous volume identities. Missing reference metrics are null
with availability reasons and cannot affect training or model selection.

The observed software environment is Python 3.11.16 / PyTorch 2.14.0 on CPU.
The target Python 3.10 / PyTorch 2.4.1 CUDA environment is untested. A working
W&B SDK was not verified locally; focused checks use disabled W&B and local reports.

## Validation and publication

Implementation commit: `7607eeac7b185998af030af57208a0902632706a` on `main`.
Baseline source diff against initial HEAD
`a3737081fe10c1e4c38ba3e2fbeaddc79f1c4e65` is empty.

**Root verdict: code ready for hardware/data qualification; real execution remains
NOT STARTED.** Earlier worker pass counts and the historical red-team report are
provisional snapshots, not this final verdict.

- Combined focused suite: **45 passed in 16.28 s**. See [JUnit receipt](focused_checks.xml).
- After final reference completeness/sidecar hardening: **5 affected checks passed
  in 13.51 s**. See [final reference receipt](reference_checks.xml). These include
  both native NIfTI pipeline fixtures, independent reference CLI execution,
  native-mm surface availability, and same-shape/wrong-affine rejection.
- Each synthetic dataset probe has 4 patients, 8 volume/frame identities and 16
  Z/T units; one software epoch, 15 compared methods, both student arms, primary
  and deployment freezes, 16 verification rows, and a successful exact continuation
  without changing frozen bytes. Both are correctly reported `partial`, not 150-epoch
  completion. Actual student updates and skips are retained in the receipt.
- Shell syntax for both launchers, strict config parsing (150 / 8 / 1), sequential
  dry run, fatal Ruff checks and static forbidden-baseline-import scan passed.
  No supervised repository-wide suite or GPU stress test was run.

Initial integration attempts exposed and then closed missing deployment slice
metadata, string/integer foreign-bank slot mismatch, and missing explicit nuisance
completeness. Reference review closed incomplete-freeze acceptance and frozen
volume-sidecar discovery. These were software failures in temporary synthetic
checks, not failed real ACDC/M&Ms runs.

The historical W8 mixed-study-grid and metadata-whitelist defects are closed by
the fail-closed data path and current focused checks. Root also reviewed exact
resume, student valid-support accumulation, per-split aggregation, verification
reuse, and per-volume reference identity.

[Validation receipt](validation_receipt.json) records the final package source
hash, observed environment, exact fixture paths and real-execution states.

See [operator guide](../../docs/maskfree150.md),
[architecture contract](architecture_contract.md), and
[real worker/session registry](worker_registry.json).
Eight Opus workstreams were dispatched through Orca. AGY authentication/quota
was unavailable; Claude later exhausted its session quota. User-authorized Luna
workers completed bounded implementation tasks. Orca exposed the `opus` alias
but not an independently resolved underlying model version; it remains null.
User-owned/external terminals were retained and were not forcibly closed.
