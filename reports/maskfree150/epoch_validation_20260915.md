# Per-epoch validation Dice — 2026-09-15

## User-authorized change

The user explicitly enabled reference Dice every epoch, after that epoch's
compared predictions freeze. Native masks are colocated with images; M&Ms has
a separate validation folder and ACDC development patients are split from
training. This supersedes the original brief's final-only reference timing.
It does not permit mask supervision, GT-selected thresholds/checkpoints,
reference-driven training crops, or training on validation patients.

## Implementation

- Image-only discovery preserves M&Ms official Training/Validation/Testing
  membership, checks patient conflicts and ignores unrelated absolute parent
  folder names. ACDC retains the deterministic patient split already declared.
- Each completed epoch saves its exact resume checkpoint, predicts both
  full-input students over the same development units using the configured
  physical batch, exports native volumes and freezes both arms completely.
- Every epoch has separate immutable student checkpoints and image-manifest
  provenance. Its freeze never points to rolling `last.pt`.
- Only a separate CPU evaluator can load references, after revalidating the
  complete freeze. Scalar reports are outside checkpointed training history.
  Validation preserves RNG, model weights/modes and optimizer state.
- Default terminal output has train/validation tqdm bars, timings and two
  student metric lines. Detailed load/export/reference phases stay in the
  journal. Missing or failed reference evaluation has null metrics and a reason.
- Epoch monitoring uses development data. Final common-bank comparisons and
  image-only O_verify keep their final-freeze boundary; native ACDC and M&Ms
  remain independent, with supervised Self-Audit/Candidate C preserved.

## Worker provenance

Existing user-authorized Luna sessions were reused on the shared `main`:
`/root/luna_data` (split discovery), `/root/luna_eval` (initial reference schema;
coordinator completed the evaluator), `/root/luna_export` (configuration/launchers),
`/root/luna_reporting` (terminal), and `/root/luna_audit` (read-only review).
Their previously resolved model is `gpt-5.6-luna`; these are native session
names, not newly claimed Orca/Opus IDs. Coordinator remains sole architect,
integrator and final reviewer. No worktrees were created.
The requested Orca check for `run_d19ed61b546b` returned `No messages.`

## Validation and execution status

Validation evidence:

- Combined focused CPU suite: **41 passed** in 25.13 seconds. Coverage includes
  native image manifests, reference evaluation, trainer, both final pipeline
  freezes, config/launchers, progress and the strict config allowlist.
- After review fixes, **18 affected checks passed** in 9.17 seconds: missing
  reference report recovery, cache integrity, training-only M&Ms split fallback,
  native ACDC image-index proof, M&Ms sparse ED/ES, and training/resume isolation.
- Final terminal tests: **9 passed**, including a real PTY with zero reported
  width/height. That PTY originally hid tqdm bars; the fallback now sets both
  dimensions, and a direct PTY probe shows Train and Val bars.
- A real two-epoch CPU CLI smoke completed its bound with exit **2 / partial**
  on a declared 150-epoch timeline. Both epochs emitted actual synthetic
  reference Dice/IoU and RV/MYO/LV Dice for both students. Configured batch was
  8; this tiny fixture has only a final short training batch, so it is not an
  eight-example hardware qualification.
- Shell syntax and `git diff --check` passed. Ruff was not installed in the
  selected root Python environment; no dependency was installed for linting.

Artifacts: `runs/maskfree150/software_checks/epoch_val_final_combined/`,
`epoch_val_review_fixes/`, `epoch_val_progress_final/`, and
`epoch_val_terminal/` (CLI logs and actual reference reports).
Synthetic tests are software evidence, not cardiac segmentation quality or
GPU throughput evidence. Counts above describe separate runs, not an additive
unique-test total.

Review corrections also preserve the old reference evaluator's strict default
for unmatched volumes, fix unavailable single-patient bootstrap reporting,
and retain complete frozen prediction bytes when repairing missing metrics.

Real rented-machine data layout, CUDA execution and physical batch 8 have not
been tested in this local macOS checkout. Full ACDC/M&Ms GPU runs are
**NOT STARTED here**: the user's GPU run was deferred to rented hardware.
No GPU/SSH job was launched and no unrelated process was stopped.

Completed validation receipts hash their reference metrics artifact. Missing
or altered reports cannot be reported completed; the isolated evaluator can
regenerate them from the same freeze into a new attempt directory, retaining
the prior receipt. Changing an explicit epoch reference-config path likewise
creates a new evaluation attempt without changing prediction/checkpoint bytes.

## Native reference conventions

The fixed M&Ms map is raw 0=BG, 1=LV, 2=MYO, 3=RV, converted once to this
pipeline's BG/RV/MYO/LV order. The implementation reference is the
[M&Ms participant repository](https://raw.githubusercontent.com/MarioProjects/MnMsCardiac/master/README.md).
Native folder and sparse ED/ES annotation conventions are also documented by
the [CineMA M&Ms loader](https://github.com/mathpluscode/CineMA/blob/main/cinema/data/mnms/README.md).
The original M&Ms website was temporarily unavailable during verification.
No supervised preprocessing or cropping code from those projects is imported.
