# Scalar Cross-Patient Transfer Utility — implementation delivery

Date: 2026-10-30. Branch: `QuocKhanhLuong/cptu-scalar-20261030`.
Source base: `d93c4dea4f42895297ec5941bb938b47140aa9e4`.
Verified training source SHA-256:
`f44e1d12bda0eaacab2bdea26c3960a803a7f179e9273466be40f5c1ca4362d1`.

**IMPLEMENTED; software checks VERIFIED; real pilot BLOCKED.**
The four patient arms have not run. No empirical GO/NO_GO or deployment success
is claimed. `vast-gpu` still resolves to `141.0.85.200:42465`; the final read-only
SSH attempt returned exit 255, `Connection refused` at 10:34:18 UTC.
See [SSH receipt](SSH_RECEIPT.json). No remote jobs were changed or started.

## What changed

- `self_audit/compact_control.py`, `compact_utility.py`: executable original,
  larger-tie and cross-patient utility selectors; unchanged hard-Dice/absent-FP
  guard; fixed probe tensors; functional no-grad trials; one additive commit;
  separate logical-zero/no-candidate/reject-all/accepted-nonzero outcomes.
- `self_audit/compact_probe.py`: patient-disjoint image-only TRAIN-U probes with
  an independent checkpointed generator; optional explicit same-batch control.
- `self_audit/compact_training.py`, `compact_artifacts.py`, `engine.py`, `train.py`:
  canonical trainer integration, CLI, full immutable resume, verified log-prefix
  continuation, cumulative coverage/costs, CUDA telemetry, native export reuse.
  Legacy checkpoint export remains supported; incomplete old metadata cannot
  be relabeled as an exact training resume.
- `self_audit/transfer_preflight.py`: exact protocol/asset pins, L8 label union,
  raw B8 teacher payload and quality-receipt checks, bounded cache validation,
  read-only GPU/job checks, fresh-output requirement. The actual archived VAL18
  manifest uses `self_audit.data.v2`; this schema is supported explicitly.
- `scripts/run_transfer_utility_pilot.py`, `check_transfer_utility_preflight.py`,
  `analyze_transfer_utility.py`: bounded profile, four sequential arms, exact
  checkpoint segments, export/evaluation at locked boundaries, watchdog,
  partial-run preservation, conditional same-batch dispatch, provenance-aware
  comparison, patient/class metrics, observed equal-time comparison and plots.
- `scripts/verify_compact_original_parity.py`, `audit_transfer_utility_evidence.py`,
  `smoke_compact_pipeline.py`: independent old-source oracle, archived-log
  recomputation and four-arm manufactured end-to-end smoke.
- `configs/transfer_utility/`: immutable 2,000-step pilot, 32-step separate GPU
  profile, four primary arm configs and conditional same-batch config.
- `ACTIVE_PIPELINE.json`, `docs/METHOD.md`, `docs/COMPACT_STUDENT.md`,
  `docs/CPTU_IMPLEMENTATION_CONTRACT.md`, `docs/TRANSFER_UTILITY_RUNBOOK.md`:
  current source-of-truth semantics, ownership, protocol and exact CLI commands.
- Seven new test modules cover selectors, probe/resume, preflight, archived
  evidence, runner, actual segmented subprocess training and guard witnesses.

The complete changed-source file inventory and hashes are in
[SOURCE_FILES.json](SOURCE_FILES.json). AGY workers implemented assigned scopes
through Orca orchestration; the coordinator reviewed the integrated code,
corrected failures and ran the final checks independently. No quota fallback
was needed. The original main worktree and its untracked user data were preserved.

## Actual verification

| Check | Result |
|---|---|
| `python scripts/check_repo.py` | PASS |
| `python -m pytest -q` | **899 passed, 2 warnings**, 132.26 s |
| Original implementation oracle from pinned `d93c4de` source | **80/80 PASS**, committed tensors and original receipt fields match |
| Actual three-segment CLI resume, first boundary inside warm-up | PASS in pytest; decisions, schedule, model tensors and optimizer match uninterrupted run |
| Four-arm synthetic CLI smoke | PASS; 4 supervised steps and 3 nonzero U updates per arm |
| Matched initialization and L/U schedule | PASS in smoke |
| Compact encoder/decoder/head actual weight changes | PASS |
| Export/reload and inference with owned teacher/cache/TRAIN assets hidden | PASS; native predictions byte-identical |
| Native evaluation, geometry and coverage | PASS on two manufactured cines |
| Bounded CPU profile | PASS; 8 steps, including 4 warm-up steps |
| Bounded PRO4000 profile | NOT_RUN — SSH blocked |
| Physical recovered B8 weights/cache on server | NOT_VERIFIED — SSH blocked |
| Four-arm patient pilot / VAL18 comparison | NOT_RUN |

The final source SHA stayed identical before and after verification. Commands,
exit codes, runtimes and stdout/stderr hashes are in [VERIFICATION.json](VERIFICATION.json).
The two warnings are an existing deliberately duplicated optimizer-parameter
negative test and PyTorch's `torch.jit.script` deprecation warning. Neither was
suppressed. An intermediate integrated test failed because source changed
between resume segments during worker edits; it passed on the final frozen
source. No test was removed to make the suite pass.

Required risk coverage includes: old selector parity; feasible equal-Dice ties;
maximal feasible utility; high-gain guard rejection; empty/nonfinite/no-positive
gain zero; U patient disjointness and no mask reads; frozen probe identity;
independent RNG; BatchNorm/optimizer/RNG atomicity; single commit; exact resume;
asset-free export; legacy CLI/config regression. Synthetic witnesses establish
software mechanics only.

## Bounded measurements and artifact identities

The smoke/profile use **widths [4,8], manufactured arrays and CPU**. These are
small software witnesses, not the 603,956-parameter patient pilot architecture.
No EMA reference ran in this benchmark, so no speed/memory superiority is known.

| Manufactured inference measurement | Observed value |
|---|---:|
| Device / precision / batch | CPU / FP32 / 2 |
| Cases / complete native cines | 2 / 2 |
| Model-only median seconds/volume | 0.000596 |
| E2E median / p95 seconds/volume | 0.011313 / 0.011664 |
| Measured aggregate throughput | 42.424 volumes/s |
| Peak inference process-tree host RAM | 208.063 MiB |
| GPU allocated/reserved | null / null |
| Export parameters / tensor bytes / serialized weights bytes | 2,288 / 9,152 / 18,060 |
| Cold model load | 0.003660 s |

Timing uses one warm-up and one steady repeat with OS caches retained. It is
too small to support a patient or hardware performance claim. The full raw
receipt is [SYNTHETIC_CPU_BENCHMARK.json](SYNTHETIC_CPU_BENCHMARK.json).
The eight-step CPU profile measured 0.124 s of training; its 1.384 s process
elapsed time includes startup. Its linear projection is not a PRO4000 estimate.

| Manufactured artifact | SHA-256 |
|---|---|
| Export manifest | `2155053ca1d884509cdee497dc661d28b7d041cedcb085c236994a506ce643d4` |
| Export weights | `c7de65da0ab4de179cd2a44d273d35b081da754fac47713dcd9eda2dfd36b328` |
| Evaluation manifest | `b43ac84265704eb196a375922eabbb944646cde67220e037e0e71ed9032cec63` |

All configuration, training-manifest, cache, checkpoint, prediction and source
hashes are retained in [SMOKE_REPORT.json](SMOKE_REPORT.json),
[SYNTHETIC_CPU_PROFILE.json](SYNTHETIC_CPU_PROFILE.json), and
`synthetic_manifests/`. Binary manufactured weights remain locally under
`/tmp/cptu-final-verify-20261030-1034/`; they are not a delivered patient model.

## Patient pilot status and next execution

| Arm | Status | VAL18 Macro4 | Foreground Dice | Patient checkpoint |
|---|---|---|---|---|
| Plain KD, original LR | NOT_RUN | null | null | none |
| Original scalar guard | NOT_RUN | null | null | none |
| Larger-tie scalar guard | NOT_RUN | null | null | none |
| Cross-patient utility scalar guard | NOT_RUN | null | null | none |

The protocol was locked before any new patient result: 2,000 total steps,
1,000 warm-up, evaluation at 1,000/1,500/2,000, final step primary, watchdog
7,200 s per arm. Screening requires utility Macro4 to exceed **each** control
by >0.002. Only a real utility-over-larger-tie signal triggers the same-batch
control. Missing evidence is BLOCKED, not an empirical NO_GO. No joint expansion.

The archived 761/7,028 finding was independently recomputed. The teacher pin is
the **raw student from recovered full-EMA B8**, not the missing original b36
artifact and not EMA-state deployment. Recorded lineage is ACDC L8; physical
validation remains pending. The old TEST50 metrics are already-observed
development references, and were not used for pilot tuning.

Use the exact train/resume/export/infer/VAL/benchmark commands in the
[runbook](../../../docs/TRANSFER_UTILITY_RUNBOOK.md). Once a reachable server
endpoint is available, run read-only preflight first, then the locked `--mode all`
runner in tmux on an idle GPU. It checks physical teacher/cache assets before
launch. Do not change steps, labels, batch or guard to force acceptance.

Final Macro4 >=0.87 and faster/lighter inference than the EMA reference remain
**NOT_VERIFIED**. No completed patient checkpoint, real VAL curve, PRO4000
benchmark, or scientific GO result exists for this delivery.
