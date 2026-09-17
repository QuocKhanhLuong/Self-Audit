# Candidate C runtime and checkpoint review — code PASS, CUDA blocked

**Reviewer:** Worker 6, runtime and checkpoint. Read-only on production code except `src/self_audit/provenance.py`, transferred to this stream mid-dispatch together with `tests/test_candidate_c_runtime.py` (see §9).
**Base commit:** `9a612485b7ead1273082d7dd52879f8e9e7a3e83` (`9a61248`, "Document runtime audit and PyTorch 2.4.1 validation").
**Working-tree snapshot investigated:** uncommitted Candidate C work in progress, read at 2026-09-11 19:40–20:05 local, while other streams were actively editing (see §7).

**Status of this revision.** Report-only: the investigation is closed, the code review verdict is **PASS** (§1), and no production file was changed for this revision. Sections that quote what the tree looked like during the investigation are marked *historical* and describe the hole that was found, not current behaviour; every current claim below has been re-checked against the tree as it now stands. Test results stay provisional — the root runs the authoritative gates on a quiescent tree.

Files carrying uncommitted Candidate C work at the time of reading:

```
configs/self_audit_full.yaml               |  26 +
configs/self_audit_full_mnms.yaml          |  12 +
scripts/evaluate_external_mnms.py          |   6 +
src/self_audit/data/mnms.py                |  35 +-
src/self_audit/models/__init__.py          |  47 +-
src/self_audit/models/annotation_expert.py | 288 +-
src/self_audit/models/dynamic_window.py    | 161 +-
src/self_audit/models/self_audit_net.py    | 946 +-
src/self_audit/training/_utils.py          | 239 +-
src/self_audit/training/finetune_joint.py  | 347 +
src/self_audit/training/unified_config.py  | 123 +-
src/self_audit/training/unified_trainer.py | 221 +-
```

Before the ownership extension arrived, this review modified no production file. Afterwards it changed exactly one, `src/self_audit/provenance.py`, and added `tests/test_candidate_c_runtime.py`; §9 records that patch in full. Everything else here remains read-only inspection.

---

## 1. Verdict

**Code PASS. CUDA BLOCKED. The controlled experiment is not yet authorized.**

Runtime and checkpoint posture is sound: replay state is ephemeral, the diagnostics contract is JSON/W&B-clean, the solver leaves no gradient or state residue, and the mechanism-identity hole is closed. Two behaviours are fail-closed by design and the coordinator has **accepted** them rather than softening them (F1, F3). What remains open is not code: no CUDA device, dataset or checkpoint has been reached, and no effectiveness claim is authorized or supported by anything in this report.

| ID | Finding | Severity | Status |
|---|---|---|---|
| F1 | Every pre-Candidate-C `last.pt` is unresumable once the new config keys land | By design, fail-closed | **Accepted by the coordinator** — no silent migration |
| F2 | `model_identity.signature` was mode-blind, so a `current`-mode `tau_accept` bound and was reused under `candidate_c` | High | **Fixed** (§2 F2, §9) |
| F3 | Landing C invalidates every existing calibration artifact through `source_content_signature` | By design, fail-closed | **Accepted by the coordinator** — recalibrate on ACDC |
| F4 | Replay state is correctly ephemeral: no parameters, no buffers, no checkpoint bytes | Pass | Verified empirically |
| F5 | Diagnostics rows are JSON/W&B-safe and must never enter `report["epochs"]` | Pass with a standing constraint | Verified empirically |
| F6 | C1 replay is bitwise exact on CPU, in fp32 and under bf16 autocast; CUDA is unverified | Pass on CPU, GPU gate open | Verified empirically |
| F7 | Solver leaves no `.grad` on model parameters and does not mutate module state | Pass | Verified empirically |
| F8 | `selected_best/` snapshots are never pruned; ~350 MB each, monotonic growth | Medium, operational | Confirmed by code path |
| F9 | No CLI override exists for `model.window_mode`; a bounded Candidate C run needs a copied config | Low, operational | Confirmed; no checked-in config required |
| F10 | Exact-resume test coverage silently skips while workers edit `src/` | Medium, process | Observed in a real run |

No real-GPU gate was executed. There is no CUDA device, no dataset and no trained checkpoint on this host, and the coordinator's own read-only probe of the remote alias was refused (§5), so every GPU item is **BLOCKED / NOT RUN**, never a pass.

**Remaining limits, explicit.** (1) *Numerical*: C1 bitwise identity is proven only on CPU, fp32 and bf16 autocast; CUDA replay identity, and therefore the reachability of the 1e-5 tolerance on GPU, is unverified (F6). (2) *Resume*: pre-C checkpoints cannot be resumed under the new configs at all, and a pre-C calibration artifact cannot supply a threshold to a post-merge evaluation (F1, F3). (3) *Scope*: nothing here measures Dice, transition quality, or whether Candidate C helps; that is the controlled experiment, which is not yet authorized.

---

## 2. Confirmed findings

### F1 — Adding the new config keys makes every pre-C checkpoint unresumable

`compare_execution_configs` (`src/self_audit/training/unified_trainer.py:175`) delegates to `_diff_keys` (`:204`), which raises whenever a key exists on one side and not the other:

```
if k not in d1: raise ValueError(f"Config mismatch on resume: missing key '{pfx}' in saved config")
```

The canonical configs now add `model.window_mode`, `model.candidate_c` (ten scalars) and `training.rollout.predicted_history_exposure` / `predicted_history_weight` (`configs/self_audit_full.yaml`, `configs/self_audit_full_mnms.yaml`). Reproduced against the live code:

```
$ python -c "compare_execution_configs({'model':{'num_classes':4,'window_k':8}}, <current>)"
ValueError: Config mismatch on resume: missing key 'model.candidate_c' in saved config
```

`compute_config_signature` (`:163`) hashes the same resolved config, so the recorded signature changes as well.

**Impact.** Any run started before this merge cannot be resumed after it, including a partially completed 130-epoch curriculum.

**Coordinator decision: accepted as-is.** The fail-closed refusal stands and no compatibility shim is added. A pre-C run is resumed from its **original checkout**, or it is restarted under the new configuration; it is never migrated silently into a config it was not trained under. This matches `reports/runtime_hardening/2026-09-10/runtime_runbook.md`, which already requires historical checkpoints that cannot establish the exact-resume contract to fail clearly rather than migrate. Operators need to know the rule, not work around it: the error names the missing key, and the remedy is the original checkout or a restart.

### F2 — The checkpoint identity signature cannot distinguish execution modes

*Historical, as observed at the time of investigation:* `resolve_model_identity` built the identity dict and signed `class_name`, `num_classes`, `shared_channels`, `window_k`, `max_turns`, `encoder_name`, `encoder_pretrained_requested`, `encoder_backend`, `entropy_version`, `buffer_count`, `torch_version` — with `window_mode` absent. A temporary `_bind_execution_mode` helper in `training/_utils.py` attached `identity["window_mode"]` *after* the signature had been computed, so the signature stayed mode-blind. That helper has since been removed by the config stream, and identity is now signed mode-aware inside `provenance.py` itself (§9); the measurements below are the record of the hole, not a description of current behaviour.

Measured on the live code, two models differing only in `window_mode`:

```
sig equal: True
state_digest equal: True
params equal: True   (29,140,098)
buffers: 1 1
live window_mode: candidate_c
```

`src/self_audit/evaluation/calibration_lineage.py` lists `checkpoint.model_identity.signature` in `IDENTITY_FIELDS` and compares it when a calibration artifact is reused. Because Candidate C adds zero parameters, `checkpoint_sha256`, `state_digest` and `file_state_digest` are also identical across modes.

**Impact.** A `tau_accept` calibrated on ACDC with `window_mode: current` passes every lineage check when reused for an evaluation running `window_mode: candidate_c` on the same weights and the same source tree. Candidate C changes the accepted-transition distribution the threshold was calibrated against, so this is a silent semantic mismatch in exactly the place the lineage machinery exists to prevent.

**Resolution.** The coordinator transferred `src/self_audit/provenance.py` to this stream and this dispatch closed the hole; §9 has the patch, its compatibility rule and its tests. In short: `window_mode` and a signature over the live solver settings are now always recorded in the identity, and they are folded into the signed `signature` exactly when the live model runs a non-default mechanism — so a baseline model's signature is unchanged byte for byte and existing artifacts stay comparable, while any Candidate C mode signs differently and is refused by the existing `IDENTITY_FIELDS` comparison. `calibration_lineage.py` needed no edit.

### F3 — Landing Candidate C invalidates every existing calibration artifact

`SOURCE_SIGNATURE_ROOTS = ("src/self_audit", "scripts", "configs")` (`src/self_audit/provenance.py:62`) and `source_content_signature` (`:236`) hash the content of every `.py`, `.yaml`, `.yml`, `.sh`, `.ps1` under those roots. `verify_source_identity` (`src/self_audit/evaluation/calibration_lineage.py:499`) raises `LineageMismatchError` when the artifact's signature differs from the runtime's.

**Impact.** The moment any Candidate C source or config edit lands, every previously produced calibration artifact and every threshold JSON that carries a lineage block stops being reusable.

**Coordinator decision: accepted as-is.** Source-bound calibration stays fail-closed and nothing is migrated. The consequence is a scheduling dependency, not a code change: a fresh calibration is produced by the post-merge source, **on ACDC only**, and that value is what the frozen external M&Ms flow consumes. No threshold is ever fitted or re-fitted on external labels, and an artifact from before the merge may not supply one.

Note also that `SOURCE_SIGNATURE_EXCLUDED_DIRS` excludes `reports` and `tests`, so this report and the new `tests/test_candidate_c*.py` files do not perturb the signature.

### F8 — `selected_best/` snapshots accumulate without bound

Every improving epoch writes a fresh uniquely named snapshot (`unified_trainer.py:2662-2667`, `selected_best/epoch_{n}_{uuid}.pt`) and nothing deletes older ones: `grep -n "unlink\|prune\|rmtree" src/self_audit/training/unified_trainer.py src/self_audit/training/checkpoint_commit.py` finds only the temp-file cleanup in `checkpoint_commit.py:338` and two guards that forbid removing pre-existing checkpoints (`unified_trainer.py:920`, `:931`).

Measured size: 29,140,098 parameters, 116.6 MB of fp32 weights, ≈349.7 MB per checkpoint once AdamW's two moment buffers are included. A 130-epoch run with twenty improving epochs therefore leaves ≈7 GB in `selected_best/` plus `last.pt` and the published `best.pt` alias. Retention is deliberate — `_require_committed_best_alias` (`_utils.py:1763`) refuses to bind a `best.pt` whose committed snapshot is missing — so **do not** delete referenced snapshots to save space. Provision disk instead, and multiply by the number of Candidate C baseline arms if several are trained.

### F9 — A bounded Candidate C run needs a copied config, not a checked-in one

`scripts/train_self_audit.py` exposes no `--window_mode` override (`add_argument` list at `:80–:117`); the mode is reachable only through `model.window_mode` in the YAML. Both canonical configs ship `window_mode: "current"` (`configs/self_audit_full.yaml:49`, `configs/self_audit_full_mnms.yaml:45`), so the baseline is what runs unless an operator says otherwise.

**No dedicated Candidate C config needs to be checked in.** For a bounded run, copy a canonical YAML to a run-local file and set `model.window_mode` in the copy. Never edit `configs/self_audit_full.yaml` in place — that silently redefines the baseline for every other run.

The copy is a different execution recipe and says so in the recorded signature. Measured here with `compute_config_signature` on the resolved configs:

| Config | `model.window_mode` | `compute_config_signature` |
|---|---|---|
| `configs/self_audit_full.yaml` | `current` | `afd2395644c925b38063c2ffe7ffb50bd4b825ba17a4beabb59bdf8bcf2d0164` |
| run-local copy, mode changed only | `candidate_c` | `edb91e14fec4105cf994a13069d43ecdd71e0596921c882d48292ee3590ba489` |

One changed scalar, a different recipe signature, so the two arms can never be confused for one another on resume or in a report. A run-local copy that lives outside `configs/` also leaves `source_content_signature` untouched, so it does not by itself invalidate the post-merge ACDC calibration; a copy placed *inside* `configs/` would change that signature and force another recalibration. Cross-*mode* reuse of a threshold remains refused regardless, by the mode-aware identity of F2/§9 — that refusal is the intended behaviour, not a side effect of where the file lives.

### F10 — Exact-resume coverage silently skips while the tree is being edited

Running the resume suite during this review:

```
$ python -m pytest tests/test_runtime_resume.py tests/test_canonical_best_alias.py -q
FAILED tests/test_runtime_resume.py::test_source_tree_was_quiescent_while_the_baseline_trained
1 failed, 17 passed, 31 skipped in 22.36s
```

The failure is the intended guard (`tests/test_runtime_resume.py:474`): concurrent edits to `src/` or `configs/` change `source_content_signature` mid-run, `resume_from_checkpoint` then refuses the checkpoint, and the 31 exact-resume comparisons skip. This is not a Candidate C defect, but it means **the final integration gate must be run on a quiescent tree**; a green resume suite collected while workers are editing proves almost nothing.

---

## 3. What already holds — verified, keep it holding

### F4 — Replay state is ephemeral and adds no checkpoint bytes

`AnnotationExpert._replay_generation` is a plain Python integer attribute, not a registered buffer, and `ExpertReplayRecord` is a frozen dataclass held only for the duration of one `infer` call. Measured before and after a `candidate_c` inference:

```
digest stable: True   state_dict keys: 274 -> 274
```

This must stay true. `provenance.state_digest` (`provenance.py:121`) hashes every parameter **and buffer** entry; `resolve_model_identity` records `buffer_count` inside the signed identity; `bind_evaluation_checkpoint` (`_utils.py:1859`) compares the live digest against the on-disk digest, and `verify_bound_state` (`:1957`) re-checks at every consumer boundary. A single `register_buffer` for a generation counter, a cached record or a diagnostic accumulator would:

1. change `state_digest` for every mode, so no existing checkpoint binds;
2. change `model_identity.signature` through `buffer_count`;
3. break `load_state_dict(..., strict=True)` for every pre-C checkpoint (`_utils.py:1594`).

**Invariant for core: zero new parameters, zero new buffers, no replay state reachable from `state_dict()`.** The current implementation satisfies it.

**Where the records live.** They are local structures of one `infer` invocation, not attributes of `AnnotationExpert`. The expert module carries exactly one added attribute, the plain integer `_replay_generation` (`annotation_expert.py:315`), which is why the state digest and the `state_dict` key count are unchanged. The record itself (`ExpertReplayRecord`) is built inside the turn (`self_audit_net.py:1124`), handed to `_update_records` (`:1341`) and held only in the inference call's own scope, superseded each turn and dropped when `infer` returns.

**One batched record, not per-sample duplication.** `_update_records` keeps every row of the factual ordinary forward — including rows that halted — and marks eligibility with a separate `accepted` mask, precisely so a replay reproduces the original forward's exact batch composition (the property F6 depends on). So the live cost is *one* record covering the ordinary subset of the batch, not one record per sample.

**Memory accounting corrected by root.** `previous_audit_evidence` / accepted `audit_evidence` have 3 channels, while the class-shared update gate has 1, not 4. Coordinates are material: two tuples × J × Hf × Wf × K × 2 scalars. A direct CPU tensor-storage probe at B1, FP32, output256², feature64², C96, K8, J3, with previous evidence found 7,340,032 bytes in the expert record plus 786,432 bytes for accepted audit evidence. That pre-cleanup measurement included an unused 1,048,576-byte factual_delta_logits clone. See replay_memory_probe.json. This is tensor storage arithmetic, not process/GPU peak memory, and excludes solver activations and the live rollout. Root removed the unused field after red-team agreement. The post-cleanup measured record is 6,291,456 bytes plus 786,432 accepted-evidence bytes: 7,077,888 bytes (6.75 MiB) per row under these assumptions.

### F5 — Diagnostics are JSON- and W&B-clean

Measured on live output: `any tensor in rows: False`, ~713 JSON bytes per row, unmeasured fields are `null` rather than `0`, and `json.dumps(json_safe_artifact(rows), allow_nan=False)` succeeds. A representative ineligible and eligible row:

```
{'eligible': False, 'record_kind': None, 'c1_passed': None, 'fallback_reason': 'no_accepted_history',
 'accepted_path': 'ordinary', 'evals': {'factual_replay': 0, 'coordinate_backward': 0, 'candidate_checks': 0, 'total_forward': 0}}

{'turn': 1, 'eligible': True, 'record_kind': 'ordinary', 'c1_passed': True, 'c1_max_abs_err': 0.0,
 'regress_mass': 0.00132, 'feasible': False, 'improved': False, 'fallback_reason': 'no_improvement',
 'accepted_path': 'factual_support', 'evals': {'factual_replay': 1, 'coordinate_backward': 1,
 'candidate_checks': 2, 'total_forward': 3}, 'delta_q': -0.0488, 'accepted': True}
```

This matches the strict contract in `src/self_audit/artifact_io.py:44`: `json_safe_artifact` **rejects** dataclasses, enums and any custom object; it silently expands tensors into full nested lists; integer mapping keys are stringified and collide with equal string keys; non-finite floats become `null`, never zero. `clean_wandb_payload` (`:190`) is the same function, so anything JSON-clean is W&B-clean.

**Standing constraint.** Per-sample rows must never enter `self.report["epochs"]`. `unified_trainer.py:2781` embeds the entire epoch history inside `last.pt` under `extra["epoch_history"]`, and `_restored_epoch_history` (`:502`) validates it row by row on resume. At ~713 bytes per row and (active samples × turns) rows per epoch, per-sample diagnostics in the report would add megabytes to *every* checkpoint and grow `last.pt` monotonically. The training worker's current approach is correct: only scalar counters reach `train_stats` (`unified_trainer.py:1523–1531`, `rollout/candidate_c_*`), and `train_stats` is an open dict, so no schema change is needed.

Checkpoint serialization itself stays safe as long as diagnostics stay out of the payload. `normalize_checkpoint_payload` (`src/self_audit/serialization.py:227`) routes `model`/`optimizer`/`scheduler`/`scaler`/`rng_state` through a **finiteness-enforcing** path and everything else through the metadata path, which preserves non-finite sentinels. Both reject sets, frozensets, bytes, custom tensor subclasses and unsupported dtypes with exact tree paths. Note the asymmetry: a NaN counter placed under `extra` survives as NaN in the checkpoint but becomes `null` in the JSON report, so prefer `None` for "unmeasured" in both.

Atomicity is unchanged and correct: `_atomic_save_torch_raw` (`serialization.py:258`) writes a same-directory temp file, fsyncs it, `os.replace`s it and best-effort fsyncs the parent directory; `atomic_write_json` (`artifact_io.py:116`) does the same for JSON with `allow_nan=False`. Candidate C touches none of this and must not.

### F6 — C1 replay is bitwise exact on CPU, including under bf16 autocast

This was the risk I flagged first, because `configs/self_audit_full.yaml` sets `training.amp.enabled: true` with `dtype: bfloat16` and `experiment.deterministic: false`, and bf16 machine epsilon (≈7.8e-3) is three orders of magnitude larger than the configured `replay_atol/replay_rtol` of 1e-5. Non-bitwise replay would make C1 unpassable.

Measured on this host:

| Context | `c1_passed` | `c1_max_abs_err` |
|---|---|---|
| fp32, `torch.no_grad()` | True | 0.0 |
| `torch.autocast('cpu', bfloat16)`, `torch.no_grad()` | True | 0.0 |

Exactly zero, not "within tolerance". The design achieves this by replaying the recorded transition with the **same active-row batch composition** inside the same autocast region, through `torch.func.functional_call` with detached parameters. That is the load-bearing property:

**Invariant for core: the replay forward must use the identical active-row batch composition as the factual forward.** Replaying a single sample out of a batch of B changes reduction order and kernel selection and would produce errors far above 1e-5 in fp32 and catastrophically above it in bf16. If a future revision narrows the replay to a per-sample call, C1 will start failing and the correct response is to restore batch identity, not to loosen the tolerance.

**Open GPU gate.** CUDA is unverified. `seed_everything` (`_utils.py:323`) only pins `cudnn.benchmark=False` / `cudnn.deterministic=True` when `deterministic: true`, and the canonical config sets it false, so cuDNN algorithm selection is at torch defaults (`benchmark=False`, heuristic choice, stable for fixed shapes) and TF32 policy is untouched. Identical shapes and a deterministic heuristic make bitwise replay likely, but "likely" is not evidence. Report `c1_max_abs_err` from the first real GPU smoke before trusting the 1e-5 tolerance there.

**Recommendation for core:** record the autocast state (enabled flag and dtype) inside `ExpertReplayRecord` and compare it at replay. `state_identity()` covers parameter identity, `_version`, dtype, device, train/eval mode and the invalidation generation, but autocast is ambient and invisible to it; a record captured inside autocast and replayed outside it would pass the identity check and then fail C1 for a reason the diagnostics would attribute to numerics rather than context.

### F7 — The solver does not contaminate training gradients or module state

Measured after a `candidate_c` inference: `grads on params: False`, and the state digest was unchanged (F4). `AnnotationExpert.replay` uses `torch.func.functional_call` with `_frozen_state()` (all parameters and buffers `.detach()`ed) inside `torch.inference_mode(False)`, and `_freeze_tensor` (`annotation_expert.py:268`) clones under `inference_mode(False) + no_grad()`, which is the correct way to lift an inference tensor into ordinary autograd territory.

This matters because `SelfAuditNet.infer` carries **no** `@torch.no_grad` decorator and is used for joint gradient training (`finetune_joint.py:233`, `:649`, inside `autocast_context`). Every evaluation caller wraps it explicitly instead: `volume_inference.py:485,657`, `transition_bank.py:447,545`, `audit_decomposition.py:511,596,660,807`, `audit_checkpoint.py:413,561`, `finetune_joint.py:369,577,1174`.

**Invariant for core: take the coordinate gradient with `torch.autograd.grad(objective, [coords])`, never `.backward()`.** A `.backward()` inside the solver would accumulate into the encoder, annotator and Auditor `.grad` buffers and silently corrupt the next joint optimizer step. `.grad` is not part of `state_dict`, so neither `state_digest` nor `verify_bound_state` would catch it — there is no safety net behind this one.

`torch.inference_mode` appears nowhere in the repository today, so the `inference_mode(False)` handling is forward-looking hardening for a deployment wrapper. It needs its own test rather than inspection: wrap an `infer` call in `torch.inference_mode()` and assert the solver still runs and still leaves parameters untouched.

---

## 4. Checkpoint state size, ephemerality and invalidation — summary

| Question | Answer |
|---|---|
| Does Candidate C add checkpoint bytes? | No. Zero new parameters, zero new buffers, `state_dict` stays at 274 entries, `state_digest` unchanged across modes. |
| Where does replay state live? | In the local scope of one `infer` call. `AnnotationExpert` gains only the plain integer `_replay_generation`; no record is ever an attribute of it, and nothing is serialized. One batched record per turn covers the ordinary subset, superseded each turn. |
| What invalidates a record? | An optimizer step (parameter `_version` bump), a device/dtype change, a train/eval flip, or an explicit `invalidate_candidate_c_records()` call, which the trainer issues around optimizer steps. Records must never span a batch or a weight update. |
| Checkpoint size | 116.6 MB weights; ≈349.7 MB per file with AdamW state. `last.pt` + every `selected_best/` snapshot + the `best.pt` alias. |
| Growth risk | `selected_best/` is never pruned (F8). Referenced snapshots must not be deleted — `_require_committed_best_alias` refuses to bind a `best.pt` whose committed snapshot is missing or hash-mismatched. |
| What Candidate C invalidates on disk | Every pre-C `last.pt` for resume (F1) and every pre-C calibration artifact for reuse (F3). |

---

## 5. Environment inventory and missing prerequisites

Read-only inventory of this host, 2026-09-11:

| Item | State |
|---|---|
| Host | macOS (Darwin 25.2.0), Apple silicon. No CUDA device, no NVIDIA driver. |
| Interpreter | `/private/tmp/self-audit-torch241/bin/python` — Python 3.10.21, torch 2.4.1, `torch.cuda.is_available() == False`. |
| Packages present | timm 1.0.29, numpy 1.26.4, pytest 9.1.1, PyYAML 6.0.3, nibabel 5.4.2, scipy 1.15.3, scikit-learn 1.7.2, matplotlib 3.10.9, tqdm 4.70.0, wandb. |
| Packages absent | SimpleITK, pandas. Neither is imported anywhere under `src/` or `scripts/`, so neither blocks anything. |
| ACDC data | **Absent.** `preprocessed_data/` does not exist; `configs/self_audit_full.yaml` expects `preprocessed_data/ACDC`. |
| M&Ms data | **Absent.** `configs/self_audit_acdc_to_mnms.yaml` expects `preprocessed_data/mnm`. |
| Trained checkpoint | **Absent.** `checkpoints/teachers/{cinema,medsam2}` are both empty; no `weights/` directory; no `best.pt` or `last.pt` anywhere in the tree. |
| Split manifest | Present: `splits/acdc_patient_split_seed42.json`. |
| Pretrained encoder | Not cached locally as far as this review can tell; `pretrained_encoder: true` in the ACDC config means timm will need network access on first construction. External M&Ms evaluation correctly sets `pretrained_encoder: false`. |
| Orca environments | Empty list. |
| Remote GPU route | An SSH alias `4070` exists in the user's SSH configuration; no address, user or credential is reproduced in this report. This worker attempted no connection. The **coordinator** subsequently attempted a read-only `nvidia-smi` over SSH in batch mode against that alias: it failed with **exit 255, `Permission denied (publickey,password)`**. No GPU, dataset or checkpoint on that host has been verified to exist or be reachable. No project runbook, config or report names a remote host, user, path or launcher; `reports/runtime_hardening/2026-09-10/runtime_runbook.md:39` says only "on the declared server". |

### Exact missing prerequisites for the requested GPU work

Everything below is required before any real-GPU gate can move off BLOCKED:

1. **Working non-interactive access to a machine with an RTX 4070**, which does not currently exist: the coordinator's batch-mode SSH probe of the `4070` alias was refused with exit 255 (`publickey,password`), so an authorized key or another access path has to be established by the user before anything else on this list can be checked. Then CUDA 12.1 and Python 3.10 + torch 2.4.1 (the repo's validated pair), plus confirmation that `CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1` selects the 4070 there — `nvidia-smi -L` under `CUDA_DEVICE_ORDER=PCI_BUS_ID` settles it.
2. The repository checked out on that machine at the reviewed revision plus the Candidate C work, with a declared absolute path.
3. `preprocessed_data/ACDC` prepared there (`scripts/preprocess_acdc.py`) together with `splits/acdc_patient_split_seed42.json`.
4. `preprocessed_data/mnm` prepared there (`scripts/prepare_mnm_binary.py` is for the binary derivative, which the four-class flow refuses; the paired-3D M&Ms preparation and the raw→ACDC mapping `0→0, 1→3, 2→2, 3→1` must be in place).
5. For the frozen external evaluation, an ACDC-trained checkpoint whose metadata declares `acdc` — `_inspect_checkpoint_dataset` (`scripts/evaluate_external_mnms.py:140`) refuses an M&Ms-trained or contradictory checkpoint and marks a metadata-less one `uncertified_historical_checkpoint`.
6. Network access on first ACDC construction, or a pre-populated timm cache, because `pretrained_encoder: true`.
7. A decision on F1 before resuming anything produced before the merge, and a post-merge recalibration before quoting any frozen external number (F3).
8. Disk headroom: ≈350 MB per committed checkpoint, unpruned (F8).

Until 1–4 exist, the only honest evidence available is CPU fixture evidence, which is what §6 separates.

---

## 6. Exact commands

Everything here is prefixed for the local torch 2.4.1 interpreter. Substitute the server's interpreter on the remote.

### 6a. Compile and import — safe anywhere, seconds

```bash
PYTHONPATH=src /private/tmp/self-audit-torch241/bin/python -m compileall -q src/self_audit scripts
PYTHONPATH=src /private/tmp/self-audit-torch241/bin/python -c "import self_audit, self_audit.models.self_audit_net, self_audit.training.unified_trainer, self_audit.training.unified_config, self_audit.training._utils, self_audit.evaluation.calibration_lineage, self_audit.serialization; print('imports ok')"
```

### 6b. CPU fixture evidence — bounded, this is *not* a GPU result

```bash
# checkpoint/serialization contract (27 tests, ~2.3 s observed)
PYTHONPATH=src /private/tmp/self-audit-torch241/bin/python -m pytest tests/test_runtime_checkpoint.py -q

# best/last commit protocol and alias publication (~22 s observed together with resume)
PYTHONPATH=src /private/tmp/self-audit-torch241/bin/python -m pytest tests/test_canonical_best_alias.py tests/test_checkpoint_commit.py -q

# exact-resume suite — ONLY on a quiescent tree, see F10
PYTHONPATH=src /private/tmp/self-audit-torch241/bin/python -m pytest tests/test_runtime_resume.py -q
```

### 6c. Bounded native ACDC smoke on the real GPU — **not executed here**

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 \
python scripts/train_self_audit.py \
  --config configs/self_audit_full.yaml \
  --device cuda --max_steps 2 --max_val_batches 1 \
  --output_dir weights/candidate_c_canary_current \
  --report_dir reports/candidate_c_canary_current \
  --no_wandb --no_tqdm
```

For the Candidate C arm, copy this YAML to a run-local file, set `model.window_mode: candidate_c` in the copy, and point `--config` at it (F9); there is no CLI override, and the canonical file must not be edited in place. As the runbook already states, this canary writes an intentionally incomplete, non-resumable `last.pt` and skips calibration. It proves CUDA initialisation, bf16 forward/backward and first-save portability — nothing about Dice, effectiveness or full-curriculum resume.

What to read out of the Candidate C canary, in order: `c1_passed` and `c1_max_abs_err` on GPU (F6), `rollout/candidate_c_*` counters, and peak memory.

### 6d. Bounded native M&Ms smoke — **not executed here**

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 \
python scripts/train_self_audit.py \
  --config configs/self_audit_full_mnms.yaml \
  --device cuda --max_steps 2 --max_val_batches 1 \
  --output_dir weights/candidate_c_canary_mnms \
  --report_dir reports/candidate_c_canary_mnms \
  --no_wandb --no_tqdm
```

Separate experiment with its own membership and outputs; it is not an extension of ACDC training.

### 6e. Frozen external ACDC → M&Ms — **not executed here**

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 \
python scripts/evaluate_external_mnms.py \
  --config configs/self_audit_acdc_to_mnms.yaml \
  --checkpoint weights/self_audit_full/best.pt \
  --data-root preprocessed_data/mnm --split testing \
  --tau-accept 0.0 --device cuda \
  --output reports/candidate_c/external_mnms_<arm>.json
```

Two operational notes. First, the evaluator has **no case-count limit flag** (`--config --checkpoint --data-root --split --tau-accept --device --output` are the whole surface), so a "few-case" bounded run means pointing `--data-root` at a prepared directory holding only those cases; a subset directory changes `split.membership_signature` and is therefore not interchangeable with the full cohort in any lineage comparison. Second, `tau` stays frozen at the protocol value or at the post-merge ACDC calibration value, decided before looking at M&Ms output.

---

## 7. Evidence log

Everything in this report came from reading the tree and from these executions on the local CPU interpreter:

| # | What | Result |
|---|---|---|
| 1 | `pytest tests/test_runtime_checkpoint.py -q` | 27 passed in 2.32 s |
| 2 | `pytest tests/test_runtime_resume.py tests/test_canonical_best_alias.py -q` | 1 failed, 17 passed, 31 skipped in 22.36 s — the failure is the quiescence guard (F10) |
| 3 | Parameter/buffer census of the baseline model | 29,140,098 params, 1 buffer, 116.6 MB fp32, 349.7 MB with AdamW |
| 4 | `compare_execution_configs` on a pre-C saved config | `ValueError: Config mismatch on resume: missing key 'model.candidate_c' in saved config` (F1) |
| 5 | `resolve_model_identity` / `state_digest` across `current` vs `candidate_c`, **before** the §9 patch | signatures and digests identical — this is the historical measurement of the F2 hole; after the patch the digests still match and the signatures differ |
| 6 | `infer(mode="self_audit", t_max=2..3)` in `candidate_c` on random `[2,3,64,64]` | digest stable, 274 state_dict keys before and after, 4 diagnostics rows, no parameter `.grad` (F4, F7) |
| 7 | Same, JSON round-trip through `json_safe_artifact` + `json.dumps(allow_nan=False)` | no tensors in rows, ~713 bytes/row, nulls for unmeasured (F5) |
| 8 | Same under `torch.autocast('cpu', bfloat16)` | `c1_passed=True`, `c1_max_abs_err=0.0` (F6) |

All of this is **CPU fixture evidence on random tensors**. It is not evidence about GPU numerics, memory, throughput, real data, Dice, or Candidate C's effectiveness. No GPU smoke, no native ACDC or M&Ms run and no frozen external evaluation was executed; all remain BLOCKED for the reasons in §5.

**Every test result in this report is provisional.** It was collected while other streams were actively editing the tree — the tree changed twice during the review (first `annotation_expert.py`, `dynamic_window.py`, `_utils.py`; then `unified_config.py`, `self_audit_net.py`, `finetune_joint.py`, `unified_trainer.py`, `mnms.py`, `models/__init__.py`, both canonical configs, `evaluate_external_mnms.py`, and new `data/dataset_resolution.py` / `tests/test_candidate_c_config.py`) — so these runs indicate health, not a gate. The authoritative gates are the root's, run on a quiescent tree against the final integrated diff. **This revision is report-only: no further tests are needed from this stream for it.**

## 8. Follow-up

Settled in this revision, no action outstanding from this stream:

* **F1** — the fail-closed resume refusal is accepted; a pre-C run uses its original checkout or restarts.
* **F3** — source-bound calibration is accepted; recalibrate on ACDC after the merge and feed that frozen value to the external flow.
* **F2** — closed by the §9 patch; the temporary `_bind_execution_mode` helper has been removed by the config stream, so the duplicate mode check no longer exists.
* **F9** — no checked-in Candidate C config is required; copy a canonical YAML per bounded run.
* Tests — provisional (§7); the root runs the quiescent gates. Nothing further is needed for this report-only revision.

Still open, owned elsewhere:

1. Root's quiescent-tree gates against the final integrated diff, including `tests/test_candidate_c_runtime.py`.
2. An `inference_mode` regression test for the solver (F7), and recording the autocast state in the replay record (F6) — core stream.
3. On the first real GPU run, publish `c1_max_abs_err` before anyone relies on the 1e-5 tolerance (F6).
4. Working non-interactive access to the RTX 4070 host; the coordinator's batch-mode probe was refused (§5). Until then the controlled experiment stays unauthorized and unrun.

---

## 9. Mechanism identity patch (`src/self_audit/provenance.py`)

Ownership of `src/self_audit/provenance.py` and `tests/test_candidate_c_runtime.py` was transferred to this stream mid-dispatch to close F2 with a minimal compatibility patch. No other production file was touched, and `src/self_audit/evaluation/calibration_lineage.py` needed no edit.

### The problem restated

Candidate C adds no parameter and no buffer. Two models differing only in `window_mode` therefore share every tensor, so `state_digest`, `file_state_digest` and `checkpoint_sha256` are all equal, and `resolve_model_identity` signed only architecture. Every field `calibration_lineage.IDENTITY_FIELDS` compares agreed, and a `tau_accept` calibrated under `current` bound silently under `candidate_c`.

### What the patch does

1. **Records the mechanism, always.** `resolve_model_identity` now reports three more fields: `window_mode` (the live `model.window_mode`, or `"unknown"` for a module that predates the switches — never silently `"current"`), `candidate_c_settings` (the live solver settings as plain primitives, or `None` if unreadable) and `candidate_c_signature` (a `candidate_c_settings.v1` signature over those settings, or `"unknown"`). Settings are read duck-typed through `as_dict()` or a plain mapping, so `provenance.py` still imports nothing but stdlib and torch — no cycle into the model or config layers.

2. **Signs the mechanism only when it is non-default.** The signed payload gains `window_mode` and `candidate_c_signature` exactly when the live mode is neither `current` nor unknown. Consequences, all deliberate:
   * a baseline model and a legacy module that cannot report a mode keep the **historical signature byte for byte**, so existing calibration artifacts stay comparable and no recalibration is forced by this patch alone;
   * every non-default mode (`candidate_c`, `candidate_c_no_fix`, `direct_rollback`, `feature_only`, `free_offsets`) signs differently from the baseline and from each other;
   * under a non-default mode, changing any solver setting — including `lr: null` versus `lr: 0.0` — changes the signature;
   * under `current` the solver never runs, so differing inert `candidate_c` values do **not** change the signature; two such runs compute the same thing;
   * unreadable settings sign as `"unknown"` rather than as absent, so they cannot collide with a model whose settings were readable.

3. **Verifies the mechanism, failing closed.** `verify_model_config` now also compares `model.window_mode` and, under a non-default live mode, each configured `model.candidate_c` key against the live settings. A configured non-default mode against a module that cannot report one is refused ("refusing to bind without verification") rather than run as the baseline under another mode's name; the configured default against such a module is allowed, which is the one claim a pre-mechanism module can honestly satisfy. A configured key the live model does not expose is refused. Mechanism findings are appended to the same mismatch/unresolved lists as architecture findings, so a mode disagreement never masks a `num_classes` disagreement — both appear in one refusal.

`lr: null` is treated as a sentinel, equal only to itself. Numeric settings compare by exact float value, so `1` and `1.0` agree; booleans never compare equal to numbers.

### Why the compatibility rule is the honest one

Signing `window_mode` unconditionally would have changed every signature, invalidating artifacts produced by models that compute exactly what they always computed. Excluding the default mechanism keeps the signature's meaning stable — "this is the architecture, plus any non-default mechanism that changes what the weights mean" — and still makes every Candidate C arm non-interchangeable with the baseline.

Note that F3 still applies independently: any source or config edit changes `source_content_signature`, which invalidates pre-existing artifacts for a different and unavoidable reason.

### Tests and evidence

`tests/test_candidate_c_runtime.py`, 34 tests, **34 passed in 4.23 s** on CPU. Coverage: unknown-versus-declared mechanism reporting; the baseline signature recomputed independently against the historical formula; the same instance signing identically with and without the mechanism attributes; inert settings not changing the baseline signature; each non-default mode and each solver setting changing it; the `lr` sentinel; matching and disagreeing modes; a non-default claim against a mechanism-less module; unreportable and unknown settings; a mechanism disagreement not masking an architecture one; and a real `SelfAuditNet` pair proving equal `state_digest`, equal buffer count and equal parameter count with different signatures. The last test is end-to-end: it saves a mode-less checkpoint, binds it twice, builds both lineages and asserts `verify_calibration_lineage` accepts baseline-against-baseline and raises `LineageMismatchError` on `model_identity.signature` for baseline-against-`candidate_c`.

Regression evidence on the patched tree:

| Suite | Result |
|---|---|
| `tests/test_checkpoint_binding.py` + `tests/test_calibration_lineage.py` | 240 passed in 386.77 s |
| `tests/test_runtime_checkpoint.py` + `tests/test_self_audit_core.py` | 31 passed in 2.88 s |
| `tests/test_candidate_c_runtime.py` | 34 passed in 4.23 s |

### Limitations of this patch

* CPU only, on small modules and random tensors. It says nothing about GPU behaviour, real data, or effectiveness.
* It binds the mechanism **identity**; it does not validate that a mode name is one of the six legal modes. That validation lives in the config and model layers, and duplicating the list in `provenance.py` would create a drift risk this module cannot detect. `provenance.py` knows only which single value is the historical no-op.
* The temporary `_bind_execution_mode` helper that `training/_utils.py` carried during the investigation has since been **removed** by the config stream, so `verify_model_config` is now the single place the mechanism is verified; there is no duplicate check left to reconcile.
* `_validate_checkpoint_architecture` in `_utils.py` compares only the four historical keys. A checkpoint-declared mode is therefore verified through `verify_model_config` at binding time, not by that function. Unchanged by this patch, and noted for the final review.
* The settings signature covers whatever the live model reports through `as_dict()`. If a future revision adds a solver setting that is not in that mapping, it will not be signed. Keep `as_dict()` exhaustive.
