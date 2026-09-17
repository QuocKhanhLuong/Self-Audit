# Joint-from-epoch-1 profile: what it is, how to run it, what it does not promise

Status: **new, unvalidated configuration profile.** Nothing in this document
reports a measured result. No training was run to produce it.

## 1. What was added

| File | Role |
| --- | --- |
| `configs/self_audit_joint_from_start.yaml` | ACDC joint-from-epoch-1 profile |
| `configs/self_audit_joint_from_start_mnms.yaml` | M&Ms joint-from-epoch-1 profile |
| `scripts/run_acdc_mnms_candidate_c.sh` | Native runner, now `CURRICULUM`-selectable |
| `tests/test_joint_from_start_configs.py` | Schema and runner-wiring contract tests |

The canonical staged configs `configs/self_audit_full.yaml` and
`configs/self_audit_full_mnms.yaml` are **unchanged** and remain the staged
baseline. `scripts/run_full_pipeline.sh` still defaults to
`configs/self_audit_full.yaml`; only its help text and header comment were
extended to mention the new profiles.

## 2. What the profile changes, and only that

The single substantive change is the schedule.

Staged baseline (3 intervals over 130 epochs):

| Epochs | Name | Trainable | Objective | Rollout |
| --- | --- | --- | --- | --- |
| 0–99 | `annotation_bootstrap` | `annotation` | `weighted_a0_a3` | `propagate_no_audit` |
| 100–119 | `auditor_training` | `auditor` | `counterfactual_audit` | `annotation_eval` |
| 120–129 | `joint_self_audit` | `all` | `retained_final_annotation` | `threshold_gate` |

Joint-from-start profile (1 interval over 130 epochs):

| Epochs | Name | Trainable | Objective | Rollout |
| --- | --- | --- | --- | --- |
| 0–129 | `joint_self_audit` | `all` | `retained_final_annotation` | `threshold_gate` |

Consequences that follow directly from that interval, and are asserted by the
tests:

* **Both modules are trainable from the first optimizer step.** `trainable: all`
  makes `encoder_lr`, `annotation_lr` and `auditor_lr` all strictly positive;
  the schedule validator rejects a zero on any of the three under `all`. There
  is no frozen bootstrap stage and no frozen auditor stage.
* **The auditor accept/reject gate is consulted from the first step.**
  `rollout: threshold_gate` runs the model in `self_audit` inference mode with
  `tau_accept = training.rollout.tau` (`0.0`). At epoch 0 the auditor is
  randomly initialised, so its early `DeltaQ` values are arbitrary; a
  `DeltaQ > tau` from an untrained auditor is an accepted transition all the
  same. Early acceptance is therefore essentially random feedback, and that is
  intended by this profile, not an accident of it.
* **Base (peak) learning rates are the early-training rates**, `encoder 3e-5 /
  annotation 3e-4 / auditor 3e-4`, taken from the staged bootstrap and auditor
  intervals. They are **not** the staged joint rates (`1e-6 / 1e-5 / 1e-5`),
  which exist only because the staged joint stage begins at epoch 120 on
  already converged weights. Those fine-tune rates are not tuned for training
  heads from initialisation. These early rates are an
  **explicit provisional recipe**: they were chosen by analogy to the existing
  schedule and have not been tuned or validated for this setting.
* **The optimizer LR ramp is inherited unchanged** (`warmup_cosine`,
  `warmup_epochs: 5`, `min_ratio: 0.0`). That ramp is *not* a curriculum
  freeze — both modules are trainable and the gate is consulted throughout it —
  but it does mean the rates above are the values reached at the **end** of the
  ramp. The first optimizer step runs at the scheduler's floor (`~1e-8` of
  base), rises linearly over the first 5 epochs, then cosine-decays. Do not read
  "3e-5 / 3e-4 / 3e-4" as the LR in force at step 1.
* **`checkpoint.best_selection_min_epoch: 0`.** The selection rule itself is
  unchanged: selection happens only inside a `threshold_gate` interval and only
  on `final_foreground_macro_dice`, with no fallback to `primary_metric`. The
  cutoff moves to 0 purely because the gated interval now starts at 0; the
  config loader in fact requires the cutoff to lie inside the gated interval.

Everything else is copied verbatim from the matching canonical config: dataset
class contract (`data_root`, `split_manifest`, `train`/`val`/`test` splits,
`image_size`, `depth_axis`, `representation`, class mapping, preprocessing,
dataloader), `seed: 42`, the pretrained ConvNeXt-Tiny encoder with
`fallback: false`, bfloat16 AMP, the AdamW settings and `grad_clip: 3.0`, the
annotation/audit/counterfactual loss settings, the calibration contract and the
diagnostics flags. A test compares those sections key-by-key against the staged
baseline.

Two deliberate settings worth stating plainly:

* **`model.window_mode: current`.** The profile isolates the schedule change,
  so it does not also change the network. `current` is the existing baseline
  configuration; no claim of tuning is made for it. The Candidate C native
  runner sets `candidate_c` explicitly in
  its own generated run config; the checked-in profiles never do.
* **`training.rollout.predicted_history_exposure: false`.** This flag controls
  the **auxiliary** predicted-history rollout, a *second* rollout pass some
  objectives add on top of the primary one. It is **inert for this profile**:
  the trainer's `retained_final_annotation` branch of `compute_batch_loss` in
  `src/self_audit/training/unified_trainer.py` never consults the flag and never
  runs the auxiliary pass, whatever the flag says. The gated joint rollout
  is its own forward and always consumes accepted predicted auditor feedback,
  and the rollout counters are measured from that forward. So `false` records
  only that no auxiliary pass is requested — it is **not** a feedback switch,
  and setting it `true` would **not** add a duplicate joint `infer` here.

## 3. How feedback actually flows, and where gradients do and do not go

Per image, per turn, inside the gated rollout:

1. The initial head produces `A0` from the encoded image.
2. The recurrent refinement expert proposes a candidate from the current state
   and the accumulated history.
3. The auditor scores the transition and produces `DeltaQ` (detached at the
   gate: the gate decision is discrete and carries no gradient).
4. If `DeltaQ > tau_accept` the candidate is **accepted**: it becomes the new
   state, and only then does it enter the history that later turns on the
   **same image** consume. Predicted feedback is only ever available after an
   accepted transition on that same image.
5. If the transition is **rejected**, that row **halts**: `active = accepted`,
   so a rejected row attempts no further turn and its final retained state is
   whatever it held before the rejected proposal.

Under Candidate C (`window_mode: candidate_c`, which the native runner selects),
the solver additionally requires an **accepted ordinary replay record** for the
row. A row with no such record is reported as `record_row_not_accepted` /
`no_accepted_history` and falls back to the ordinary annotation path. Any
optimizer step invalidates every replay record, because the weights moved.

What that implies for the first epochs, stated without overclaiming:

* If a row is rejected on its **first** turn, that row's retained final state is
  `A0`. The annotation loss on the retained final state therefore delivers a
  gradient to the **initial head and the encoder**, and the **refinement expert
  receives zero gradient from that row on that SGD step**. It is zero for that
  row and that step — not zero forever, and not zero for rows that were
  accepted.
* The **auditor still learns from that rejection.** The transition population is
  `active_attempted`: the attempted transition is supervised against its true
  improvement/regression target regardless of whether the gate accepted it.
* So a heavily-rejecting early auditor starves the refinement expert of
  gradient while the encoder, the initial head and the auditor keep training.
  Whether the expert subsequently recovers gradient depends on the auditor
  starting to accept, which this profile does not and cannot guarantee.

## 4. What is explicitly **not** claimed

* **No guaranteed improvement.** There is no evidence that joint-from-epoch-1
  reaches a better final Dice than the staged curriculum, or than any other
  baseline.
* **No guaranteed speed-up.** Nothing here was timed. The gated rollout, the
  threshold gate and the auditor loss now run for all 130 epochs rather than the
  last 10, and no claim is made about whether the profile converges in fewer
  epochs, or about wall-clock cost in either direction.
* **No guarantee of gradient on every module at every step.** See §3: the
  refinement expert's gradient is contingent on acceptance.
* **No validated hyperparameters.** The learning rates are a provisional recipe
  by analogy, not a tuned one.
* **`BATCH_SIZE=8` is not validated for this profile.** The native runner keeps
  the existing `BATCH_SIZE="${BATCH_SIZE:-8}"` override and applies it to every
  interval of the generated run config. Under the staged curriculum the gated
  joint stage ran only for the final 10 epochs; here the gated rollout runs from
  epoch 0, so whatever its memory profile is, it applies for the whole run.
  **No memory measurement was taken** — no GPU was used in this work at all. The
  value was left at 8 rather than silently lowered; lower it at invocation if
  the run runs out of memory. The checked-in profile YAML keeps `batch_size: 2`
  (the staged profiles keep their own `4 / 4 / 2`); the runner's override
  replaces whichever of those is in the source profile.

## 5. Relationship to the existing 130-epoch run

The currently running staged 130-epoch job is **untouched**. Its config, its
schedule, its checkpoints and its output directories are unchanged by this work.

A run of the joint-from-start profile is a **fresh training run from
initialization**. It cannot strictly resume a staged checkpoint: the schedule,
the interval identity, the config signature and the source signature all differ,
and the lineage guards that reject such a resume must never be bypassed. Neither
the runner nor the profiles pass `--resume`.

The reasonable use of the existing run is therefore to **finish it as the staged
baseline** and compare the two profiles as separate runs. Output identities are
disjoint by construction so the two can never be mixed:

| | Staged | Joint from start |
| --- | --- | --- |
| ACDC weights | `weights/self_audit_full` | `weights/self_audit_joint_from_start` |
| ACDC reports | `reports/self_audit_full` | `reports/self_audit_joint_from_start` |
| M&Ms weights | `weights/self_audit_full_mnms` | `weights/self_audit_joint_from_start_mnms` |
| M&Ms reports | `reports/self_audit_full_mnms` | `reports/self_audit_joint_from_start_mnms` |
| Native run dir | `runs/{acdc,mnms}_candidate_c_staged_4070_<stamp>` | `runs/{acdc,mnms}_candidate_c_joint_from_start_4070_<stamp>` |
| Generated run config | `run_configs/{acdc,mnms}_candidate_c_staged.yaml` | `run_configs/{acdc,mnms}_candidate_c_joint_from_start.yaml` |

## 6. How to run it

### Native Candidate C, both datasets, sequentially

```bash
# Joint from epoch 1 — this is now the DEFAULT for both native flows.
bash scripts/run_acdc_mnms_candidate_c.sh

# Old staged curriculum, explicitly.
CURRICULUM=staged bash scripts/run_acdc_mnms_candidate_c.sh
```

`CURRICULUM` is a closed enum: `joint_from_start` (default) or `staged`.
Anything else exits 2 before any config is generated or any run starts.

The runner generates its run configs from the selected profiles, sets
`window_mode: candidate_c`, applies `BATCH_SIZE` to every interval, validates
each generated file through the real strict Schema Version 1 loader, and prints
the resolved schedule inline as it goes — no intermediate artifact is written.
The strict loader is imported at module scope in that block: if it cannot be
imported the preflight fails outright rather than reporting a skipped
validation as success. The banner prints the curriculum, the source profiles,
the native batch-size override, and per interval the epoch range with
`trainable`, `objective`, `rollout` and the three base learning rates, plus
`total_epochs`, `best_selection_min_epoch` and `warmup_epochs`. Example lines
for the default:

```
[config] run_configs/acdc_candidate_c_joint_from_start.yaml: strict schema validation OK window_mode=candidate_c predicted_history_exposure=False batch_schedule=0-129:8
[schedule] run_configs/acdc_candidate_c_joint_from_start.yaml: epochs [0, 130) name=joint_self_audit trainable=all objective=retained_final_annotation rollout=threshold_gate encoder_lr=3e-05 annotation_lr=0.0003 auditor_lr=0.0003
[schedule] run_configs/acdc_candidate_c_joint_from_start.yaml: total_epochs=130 best_selection_min_epoch=0 warmup_epochs=5 (optimizer LR ramp only; no curriculum warmup interval)
[schedule] run_configs/acdc_candidate_c_joint_from_start.yaml: auditor feedback and accept/reject are live from epoch 0; predicted_history_exposure=False means only that no auxiliary rollout is requested, and the joint objective ignores that flag anyway
```

ACDC runs first, then M&Ms, as two independent runs. M&Ms never loads the ACDC
checkpoint, no resume argument is passed, and no external tuning is shared
between them. Under `CURRICULUM=staged` the runner keeps its original behaviour
exactly, including forcing `predicted_history_exposure: True` at weight `0.1` —
which is what gives the frozen staged intervals any predicted-history exposure
at all. The joint profile leaves the flag off, which for its objective is inert
either way (see §2).

### Baseline `window_mode: current`, single dataset

```bash
bash scripts/run_full_pipeline.sh --config configs/self_audit_joint_from_start.yaml
bash scripts/run_full_pipeline.sh --config configs/self_audit_joint_from_start_mnms.yaml
```

`run_full_pipeline.sh` still defaults to `configs/self_audit_full.yaml`; the new
profiles are opt-in via `--config` only.

## 7. Verification performed

* `configs/self_audit_joint_from_start.yaml` and
  `configs/self_audit_joint_from_start_mnms.yaml` load under
  `self_audit.training.unified_config.load_unified_config` without error.
* Both `CURRICULUM` branches of the native runner were exercised through their
  config-generation block; all four generated run configs re-validate under the
  strict loader.
* `bash -n` passes on `scripts/run_acdc_mnms_candidate_c.sh` and
  `scripts/run_full_pipeline.sh`.
* `tests/test_joint_from_start_configs.py`,
  `tests/test_unified_config_contract.py` and `tests/test_candidate_c_config.py`:
  **118 passed** (`PYTHONPATH=src python -m pytest ... -q`), of which 25 are the
  new file. The two existing suites passing unchanged confirms the staged
  baselines are unaffected.

No model was constructed, no encoder was downloaded, no GPU was used, no memory
measurement was taken, and no training step was executed. The runner's dry-run
coverage exercises the config-generation block only and asserts that no training
command is reached.
