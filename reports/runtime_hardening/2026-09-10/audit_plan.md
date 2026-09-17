# Independent runtime audit before implementation

Base HEAD and freshly fetched origin/main: `25f429a46255ceb59997ae5178939d5d40b2ea27`. Reviewer: Astra Medium; implementation assigned only after this audit to AGY via Orca. No production edits were made during these probes.

## Confirmed bugs

| ID | Base file:line | Evidence / root cause |
|---|---|---|
| R1 | training/_utils.py:1047 | Actual Python 3.10.21 / PyTorch 2.4.1 reproduction: `_rng_state()['numpy'][1].dtype == torch.uint32`; `save_checkpoint(Linear(2,2))` raises `KeyError: dtype torch.uint32 is not recognized`. Destination absent, temp cleanup succeeds. |
| R2 | training/_utils.py:1164-1185 | Payload accepts `Path` metadata, writes it successfully on local torch 2.13, then `torch.load(weights_only=True)` raises UnpicklingError. Validation covers finite model/state tensors only, not recursively portable metadata. |
| R3 | training/_utils.py:1187-1191 | Injected `OSError(28)` from fsync is swallowed; old destination is replaced despite durability failure. Cleanup can also mask the primary error if unlink fails (inspection). |
| R4 | training/unified_trainer.py:106-120,1639 | Actual strict JSON probes fail on NaN ndarray, Inf tensor, NumPy NaN scalar and Path. Array/scalar conversion is not recursive; default JSON permits nonstandard NaN/Infinity. |
| R5 | training/unified_trainer.py:1491-1640 | No exception guard or per-epoch durable report; W&B log occurs before checkpoint save, logger finish only on successful final report path. Checkpoint failure loses report/validation evidence and cleanup status. |
| R6 | evaluation/threshold.py:1421; evaluation/transition_bank.py:1573; scripts/audit_checkpoint.py:1305; scripts/evaluate_external_mnms.py:328 | Direct final-file writes can truncate existing artifacts. Bank JSON validation occurs while final file is open. |
| R7 | training/unified_trainer.py:1739; scripts/cache_validation_transitions.py:114; scripts/train_self_audit_legacy.py:617 | Transition caches bypass checkpoint atomic writer and recursive serialization checks. |
| R8 | training/unified_trainer.py:1271-1274,1427-1430 | Safe checkpoint load failure triggers unrestricted pickle fallback. This is unsafe and obscures unsupported payload bugs. |
| R9 | training/unified_trainer.py:1532-1573 | Best tracker and best.pt change before last.pt commits. If last save fails after best replacement, the old last hash references an overwritten best, breaking resume. |
| R10 | training/_utils.py:1561-1564 | W&B scalar NaN/Inf rejected by first branch but then accepted by the following broad numeric branch; nested payload values silently dropped. |
| R12 | scripts/calibrate_threshold.py:61-65 | Active standalone cache loader explicitly uses weights_only=False, then an implicit unrestricted fallback; discovered by exhaustive torch.load boundary scan. No safe-load gate for transition metadata. |
| R11 | training/unified_trainer.py:1344-1358; configs/self_audit_full.yaml | Canonical augmenting persistent workers are explicitly rejected on resume; changing worker config then fails strict execution-config identity. Default command is not exactly resumable as currently configured. |

Paths above are relative to `src/self_audit/` unless prefixed `scripts/` or `configs/`.

## Likely bugs / additional review targets

- Loading complete checkpoint and best sibling with map_location=GPU can allocate duplicate optimizer/model state on a 12GB GPU. CPU staging is preferable with explicit optimizer-device restoration.
- Final calibration can publish an artifact before later diagnostics fail; completion must be committed only after all requested work finishes.
- Early output/init/resume failures after W&B initialization do not close it; stale completed reports must not describe a new failed attempt as successful.
- Full report memory and checkpoint provenance must preserve undefined Dice as null with explicit semantics, never zero. Internal best=-inf sentinel must be handled separately from invalid model/optimizer values.
- RNG restore must validate range/shape before uint32 cast and must not silently ignore missing CUDA state in exact GPU resume.

## Not reproduced / limits

Additional independent fault probes: after completed validation, injected checkpoint OSError produced logger events `['log']` with no `finish` and an empty report directory. In a four-epoch fixture, failure saving epoch-4 last.pt left epoch-3 last.pt intact but its best hash no longer matched public best.pt (`BEST_LAST_FAILURE 3 False`). Both lifecycle and best/last linkage failures are reproduced, not merely inferred.

- CUDA 12.1, RTX 4070, bf16 GPU execution, OOM and real power-loss filesystem behavior are not available locally. A separate Python3.10 / torch2.4.1 CPU environment is installed at `/private/tmp/self-audit-torch241`; newer local torch2.13 is a second test matrix, not the compatibility authority.
- Existing code uses the torch2.4 torch.amp GradScaler API, bf16 scaler disabled by design, and autocast. Actual target-device validation remains required.
- No architecture, objective, split, threshold tuning or long retraining changes are authorized.

## Implementation waves and gates

1. Checkpoint portability/atomicity: lossless int64 MT19937 keys and explicit uint32 restore; recursive safe CPU payload handling; no unsafe loads; do not swallow fsync errors; serialization/disk/replace cleanup tests, actual torch2.4.1 RNG/optimizer/scheduler/scaler round trips.
2. Shared strict JSON and atomic artifact IO: recursively handle supported types, explicit null semantics, fail on unsupported custom objects, avoid key collisions; preserve calibration/bank schemas and signatures; update all active report/cache writers and mocked W&B conversion.
3. Lifecycle: truthful per-epoch commit and failure artifact, nonzero exit/rethrow, logger cleanup even on init/resume/finalization failure; separate validated metrics from checkpoint-committed epoch status. Never mask primary failure with failure-writer/logger errors.
4. Transactional best/last selection and resume: preserve referenced best across failed commits; resume all intervals/boundaries without unsafe pickle or duplicate GPU residency. Resolve canonical persistent-worker exact-resume limitation without changing scientific augmentation/objectives. Verify calibration after resume and W&B/report/checkpoint consistency.
5. Independent final matrix: full suites on torch2.4.1 and current environment; bounded actual subprocess training/calibration/diagnostics/bank and frozen external M&Ms; fault injection subprocess; documented GPU canary command and limits. Commit logical changes and push main only after Astra GREEN.

Each worker delivery receives PASS / REVISE / BLOCK based on actual diff and independently run tests. Synthetic fixtures must remain clearly labeled and disable pretraining only within fixtures.
