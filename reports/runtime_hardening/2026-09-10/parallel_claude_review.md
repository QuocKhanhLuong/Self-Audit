# Parallel independent review: checkpoint / RNG / AMP / resume runtime

Reviewer: parallel Claude worker (read-only on production code).
Base: `main` @ `25f429a46255ceb59997ae5178939d5d40b2ea27`, working tree as of 2026-09-10 including the
reviewed Wave 1/2 primitives (`src/self_audit/serialization.py`, `src/self_audit/artifact_io.py`)
and the in-progress Wave 3 trainer lifecycle edits.

Reference environment for every number below:
`/private/tmp/self-audit-torch241/bin/python` — Python 3.10.21, torch 2.4.1, CPU build,
run with `PYTHONPATH=src:. PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1`.

Owned deliverables: this report and `tests/test_runtime_amp_resume_review.py`.
No production file was edited. No GPU was available; nothing here claims a measured GPU result.

## Test delivery

`tests/test_runtime_amp_resume_review.py` — **14 passed in 1.11s** on Python 3.10.21 / torch 2.4.1.

Command:

```
PYTHONPATH=src:. PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  /private/tmp/self-audit-torch241/bin/python -m pytest tests/test_runtime_amp_resume_review.py -q
```

What it covers, and why it is not a duplicate of `tests/test_runtime_checkpoint.py`:

| Test | Gap it closes |
|---|---|
| `test_enabled_cpu_scaler_full_state_roundtrips_under_weights_only` | First and only exercise of a **non-empty** scaler payload. `build_grad_scaler` enables scaling only for CUDA + float16, so every other test in the repo saves `scaler == {}`. Uses `torch.amp.GradScaler("cpu", enabled=True)` (works on 2.4.1), takes a real optimizer step, asserts `weights_only=True` roundtrip of model + optimizer (`exp_avg`) + scheduler (`get_last_lr`) + scaler (`scale`, `_growth_tracker`), integer optimizer state keys survive normalization, and that a further step from restored state matches the uninterrupted step. |
| `test_saved_checkpoint_stages_every_training_tensor_on_cpu` | Asserts the write-side CPU staging invariant across `model`/`optimizer`/`scaler`/`rng_state` recursively. |
| `test_disabled_scaler_serializes_empty_and_enabled_scaler_refuses_it` | Pins the torch 2.4.1 root cause behind finding **C2** in both directions, without asserting the buggy call site. |
| `test_cuda_rng_capture_stages_states_on_cpu_and_preserves_values` | `_rng_state()` CUDA branch, previously unreachable on a CPU-only host; asserts CPU staging and value preservation. |
| `test_cuda_rng_restore_passes_cpu_tensors_for_matching_device_count` | Asserts `set_rng_state_all` receives exactly one **CPU** tensor per device. |
| `test_cuda_rng_restore_refuses_wrong_state_count` (3 cases: 2/1, 1/2, 0/1) | The wrong-count branch, with a message assertion that names both counts and never calls `set_rng_state_all`. Replaces the coverage the stale regex in `test_runtime_checkpoint.py` claims but does not deliver (**C5**). |
| `test_cuda_rng_restore_refuses_non_tensor_and_non_sequence_states` | Non-tensor entry, mapping instead of sequence, and `None`. |
| `test_cuda_rng_restore_exact_requires_present_cuda_state` | `exact_cuda=True` refuses a CUDA-less payload, **and** documents that the default path silently no-ops — the premise of **C1**. |
| `test_cuda_rng_restore_without_cuda_reports_unavailable_for_exact_resume` | Exact resume on a CPU-only host fails loudly. |
| `test_checkpoint_roundtrip_restores_cuda_rng_through_mocked_devices` | End-to-end `save_checkpoint` → `weights_only=True` file → `load_checkpoint(exact_cuda=True)` with mocked CUDA; asserts the restored bytes and CPU staging. |
| `test_python_numpy_torch_streams_continue_exactly_after_reload` | Bit-exact host stream continuation with a cached gaussian left in both Python and NumPy (`has_gauss == 1`). |
| `test_bf16_amp_resolution_and_scaler_policy_on_cpu` | bf16 AMP API contract (see "Target API assessment"). |

Every assertion is a contract that should still hold after the Wave 4 changes; none of them encodes
the current defective behaviour of a call site, so a fix will not break this file.

## Confirmed findings

### C1 — HIGH: CUDA RNG is silently not restored on GPU resume

`src/self_audit/training/unified_trainer.py:1679` calls `_restore_rng_state(rng)` with no
`exact_cuda`, and the required-component check on the two lines above only demands
`python`, `numpy`, `torch` — never `cuda`.

`_restore_rng_state` (`src/self_audit/training/_utils.py:1226-1231`) raises on a missing CUDA state
**only** when `exact_cuda=True`; otherwise the CUDA generator is left exactly as it was.
The auto-detection in `load_checkpoint` (`_utils.py:1392-1399`, which does set `exact_cuda=True`
for a CUDA `map_location` or CUDA parameters) is bypassed, because `resume_from_checkpoint` reads
the file with a direct `torch.load` (`unified_trainer.py:1469/1471`) rather than the shared loader.

Consequence on the RTX 4070 target: resuming from any checkpoint whose `rng_state` lacks `"cuda"`
(one produced on a CPU host, or produced while CUDA was unavailable) proceeds with an
un-restored CUDA generator. The resumed run is not a bit-exact continuation and nothing says so.

Repro (passes as a test):
`tests/test_runtime_amp_resume_review.py::test_cuda_rng_restore_exact_requires_present_cuda_state`
— under mocked CUDA, `_restore_rng_state({"torch": ...})` raises `ValueError: Missing 'cuda' RNG
state in checkpoint for exact GPU resume` with `exact_cuda=True`, and returns silently without it,
with `set_rng_state_all` never called.

Suggested direction for the Wave 4 owner: resume should require `"cuda"` in `rng_state` and pass
`exact_cuda=True` whenever `self.device.type == "cuda"`, or route resume through `load_checkpoint`
so the existing auto-detection applies.

### C2 — MEDIUM/HIGH: an empty scaler state passes the resume presence check, then torch rejects it

`src/self_audit/training/unified_trainer.py:1580-1585`:

```python
if self.scaler is not None and self.scaler.is_enabled():
    if "scaler" not in payload or payload["scaler"] is None:
        raise ValueError(f"Checkpoint missing required scaler state for non-boundary resume ...")
    self.scaler.load_state_dict(payload["scaler"])
```

`build_grad_scaler` (`_utils.py:943-950`) enables scaling only for `device.type == "cuda"` **and**
`dtype is torch.float16`. A disabled `GradScaler.state_dict()` is `{}`, and `save_checkpoint`
stores it verbatim, so the checkpoint contains `"scaler": {}` — present, not `None`. The guard
therefore accepts it and torch 2.4.1 raises from inside `load_state_dict`:

```
RuntimeError: The source state dict is empty, possibly because it was saved from a disabled instance of GradScaler.
```

Measured on the reference environment; the intended, actionable
`"missing required scaler state for non-boundary resume at epoch N"` error never fires.

Reachability without any config edit: `amp.enabled: auto` with `amp_dtype: float16`.
`resolve_amp` (`_utils.py:918-940`) returns `(False, float16)` on CPU — the
`CPU AMP requires amp_dtype=bfloat16` guard is skipped precisely because `enabled` is already
`False` — and `(True, float16)` on CUDA. `compare_execution_configs` sees the identical YAML on
both hosts and passes. So a checkpoint written on a CPU host and resumed on the RTX 4070 crashes
at resume with the message above.

Mirror direction (silent, not a crash): a checkpoint written on CUDA + fp16 carries a populated
scaler state; resumed on CPU the outer `self.scaler.is_enabled()` is `False`, so the whole block is
skipped, and torch's own `load_state_dict` early-returns for a disabled scaler. `scale` and
`_growth_tracker` are dropped with no diagnostic.

**Canonical configs are not affected today**: `configs/self_audit_full.yaml:47-49` and
`configs/self_audit_full_mnms.yaml:47-49` use `dtype: "bfloat16"`, so the scaler is disabled on
both hosts and `{}` loads into `{}`. That is also why the whole "required scaler state" branch is
currently dead code and why no existing test ever moved a non-empty scaler payload — the gap
`tests/test_runtime_amp_resume_review.py` now closes.

Repro: `tests/test_runtime_amp_resume_review.py::test_disabled_scaler_serializes_empty_and_enabled_scaler_refuses_it`.

Suggested direction: treat a falsy `payload["scaler"]` the same as absent (raise the explicit
error), and when the live scaler is disabled but the payload is populated, fail or warn explicitly
rather than dropping the state.

### C3 — MEDIUM: evaluation-only binds forward a CUDA `map_location`, materializing state they never use

> Root has confirmed this one is already assigned. Kept here only for the exact call-site
> inventory and the write-side CPU-staging evidence, not as a new finding.

`bind_evaluation_checkpoint` / `bind_existing_evaluation_state` (`_utils.py:1478-1560`) pass
`map_location` straight into `load_checkpoint`, which passes it to `torch.load`. The docstring is
explicit that optimizer, scheduler, scaler and RNG state are *deliberately not restored* — but
`torch.load` has already deserialized all of them onto the mapped device.

Confirmed by instrumenting `torch.load` on the reference environment: calling
`bind_evaluation_checkpoint(model, [("best", p)], map_location=torch.device("cuda", 0))` forwards
`device(type='cuda', index=0)` verbatim, and the file being read contains
`['epoch', 'format_version', 'global_step', 'model', 'optimizer', 'optimizer_step', 'provenance', 'rng_state']`.

Active call sites passing a device rather than `"cpu"`:

- `src/self_audit/training/unified_trainer.py:1942` (`map_location=self.device`)
- `scripts/audit_checkpoint.py:1198` (`map_location=device`)
- `scripts/cache_validation_transitions.py:78` (`map_location=device`)
- `scripts/evaluate_external_mnms.py:225` (`map_location=target_device`)
- `scripts/export_transition_bank.py:134` (`map_location=target_device`)
- `scripts/train_self_audit_legacy.py:315` and `:368` (`map_location=device`)
- `scripts/visualize_predictions.py:73` (`load_checkpoint(..., map_location=device)`)

The write side is already correct: `_normalize_training_tree`
(`src/self_audit/serialization.py:59-80`) calls `.detach().cpu().clone()` on every training tensor,
so the file on disk is CPU-staged and a `"cpu"` `map_location` plus an explicit `model.to(device)`
loses nothing. Verified by `test_saved_checkpoint_stages_every_training_tensor_on_cpu`.

No GPU peak-memory claim is made here; the finding is that unused optimizer/RNG tensors are
deserialized onto the CUDA device by construction.

### C4 — LOW/MEDIUM: `load_checkpoint` finite-validates model and optimizer but not scheduler or scaler

`_utils.py:1381-1390`: `_finite_tree` runs on `model` and on `normalized["optimizer"]`, then
`scheduler.load_state_dict(...)` and `scaler.load_state_dict(...)` are called with no check.
`save_checkpoint` does validate all four on the way out (`_utils.py:1296-1299`), so this only
matters for a tampered, hand-built or foreign checkpoint — e.g. a `scheduler` payload carrying
`_last_lr: [nan]`, or a `scaler` payload with a non-finite `scale`, is accepted on load. Wave 4's
brief already asks for finite validation of scalar non-finite values before accepting resume;
this is the exact pair of lines where it is missing on the shared-loader path.

### C5 — Stale assertion in `tests/test_runtime_checkpoint.py` (not my file; reported only)

`tests/test_runtime_checkpoint.py:150-158` (`test_cuda_rng_state_handling`):

```python
with pytest.raises(ValueError, match="does not match available devices"):
    _restore_rng_state({"cuda": [torch.zeros(10, dtype=torch.uint8)] * (dev_count + 1)})
```

The implementation raises (`_utils.py:1230`):

```
Checkpoint contains 2 CUDA RNG state(s), but 1 device(s) are available
```

The regex cannot match, so the assertion is wrong. It is invisible locally because that whole
branch is inside `else: # torch.cuda.is_available()`, which never runs on a CPU-only host — the
test would fail the first time the suite runs on a CUDA machine. The CPU branch it does take
asserts only the `CUDA RNG restore requested but CUDA is not available` message.

I did not edit that file. `tests/test_runtime_amp_resume_review.py::test_cuda_rng_restore_refuses_wrong_state_count`
covers the same branch under mocks so it runs everywhere; the owner of
`test_runtime_checkpoint.py` should either fix the regex to the real message or delete the
now-redundant CUDA-only assertion.

## Target API assessment: torch 2.4.1 / CUDA 12.1 / RTX 4070 / bf16

Assessed from the local torch 2.4.1 APIs and the repository source only. **No GPU was available;
no GPU behaviour, throughput or memory figure is claimed.**

- `torch.amp.GradScaler(device, enabled=...)` — the 2.4.x device-scoped constructor — exists and
  works. `build_grad_scaler` uses it with a hardcoded `"cuda"` string and falls back to the
  deprecated `torch.cuda.amp.GradScaler` on `AttributeError`/`TypeError`; on 2.4.1 the fallback is
  never taken. Constructing the `"cuda"` scaler on a CPU-only host with `enabled=True` emits
  `UserWarning: torch.cuda.amp.GradScaler is enabled, but CUDA is not available. Disabling.` and
  self-disables, so the code path is safe off-target.
- `torch.autocast(device_type=..., dtype=torch.bfloat16, enabled=True)` via `autocast_context`
  works for `device_type="cpu"`; a `nn.Linear` under it returns `torch.bfloat16` and returns to
  `torch.float32` outside. bfloat16 is the canonical `amp_dtype` in both full configs, and the
  RTX 4070 (Ada, SM 8.9) supports bf16 tensor cores, so the dtype choice is consistent with the
  target — but that is an architecture fact, not a local measurement.
- Gradient scaling is correctly **not** used for bfloat16: `build_grad_scaler` requires
  `dtype is torch.float16`, and bf16's exponent range makes a loss scaler unnecessary. Asserted by
  `test_bf16_amp_resolution_and_scaler_policy_on_cpu`.
- `resolve_amp` on the reference environment: `(True, bfloat16)` for `amp: true` on CPU,
  `(False, bfloat16)` for `amp: auto` on CPU, and `ValueError: CPU AMP requires amp_dtype=bfloat16`
  for `amp: true` + `float16` on CPU. Note the `auto` + `float16` hole described in **C2**.
- Payload safety on 2.4.1: an enabled scaler's `state_dict` is five plain Python scalars
  (`scale`, `growth_factor`, `backoff_factor`, `growth_interval`, `_growth_tracker`), all
  `weights_only`-safe; optimizer state keys stay Python `int`s through
  `_normalize_mapping_key`. Both verified against a real `weights_only=True` load.
- Still requires the actual device to certify: any bf16 numerical parity claim, CUDA RNG
  restoration against a real generator, GPU peak memory for **C3**, and OOM behaviour.

## Scope and limits

- Read-only on all production code, tests other than my own, and configs. Only
  `tests/test_runtime_amp_resume_review.py` and this report were written.
- Only the one focused test module was executed; the full suite is root's at final integration.
- No commit, no push, no training run.
- `src/self_audit/training/unified_trainer.py` was being edited concurrently for Wave 3/4, so the
  line numbers cited for that file are as-of the final re-verification pass in this review
  (they already drifted by roughly +80 lines while the review ran) and may shift again. Every
  cited finding was re-confirmed present at those lines after the concurrent edits: the trainer
  still contains no `exact_cuda` reference at all, and both `weights_only=False` fallbacks remain.
- `scripts/calibrate_threshold.py:63` now reads `weights_only=True` with no unrestricted
  fallback, so audit-plan item **R12** appears already addressed in the current tree. The two
  remaining `weights_only=False` fallbacks are `unified_trainer.py:1471` and `:1631`
  (audit-plan **R8**, in Wave 4's brief).
