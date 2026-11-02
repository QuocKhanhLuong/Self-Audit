# Server campaign queue

The campaign controller runs a finite, sequential queue on one visible GPU. It
is a separate orchestration layer: `self_audit.server_campaign` does not import
or change `server_runner`, trainer code, or recipes. The plan supplies the exact
commands and the hashes that authorize their inputs.

The coordinator fills the real absolute paths, hashes, stage order, process
identity, receipts, and preregistered commands before starting the queue. This
runbook describes the exact `self_audit.server_campaign.v1` JSON contract.

## CLI

First validate the JSON and obtain its exact file hash. Validation checks the
schema; individual source, recipe, and asset hashes are checked immediately
before their stage runs.

```bash
python scripts/run_server_campaign.py validate \
  --plan /workspace/server_campaign/campaign.json
```

`validate` prints `plan_sha256`. Pass that value explicitly when starting the
detached supervisor. Reusing the same state directory with a changed plan hash
is refused.

```bash
python scripts/run_server_campaign.py start \
  --plan /workspace/server_campaign/campaign.json \
  --plan-sha256 PLAN_SHA256

python scripts/run_server_campaign.py status \
  --state-dir /workspace/server_campaign/state

python scripts/run_server_campaign.py stop \
  --state-dir /workspace/server_campaign/state
```

`start` holds an OS queue lock for the supervisor lifetime. A second start
reports `already_running`; after a supervisor restart it reattaches to the
recorded stage process. A completed, stopped, or blocked campaign is terminal.
Do not reset its state directory to retry a stage. Any new continuation needs
a new campaign ID, plan hash, and state directory.

`stop` writes a durable request. The supervisor stops scheduling after the
active stage exits; it does not signal or kill that stage. The JSON plan must
set `max_wall_clock_seconds` to an integer from 1 second through 14 days. At the
deadline, the controller likewise waits for the active stage to end, then
records pending work as stopped.

For the current S0, inspect the process without changing it and copy these
fields into the first `adopt` stage's `process` object:

```bash
python scripts/run_server_campaign.py identify \
  --pid S0_PID \
  --receipt /workspace/server_compact/run_s0/RUN_STATUS.json
```

The command prints `pid`, `start_ticks`, `cwd`, `argv_sha256`, and
`receipt_sha256_before`; it does not print the process arguments. Set `log_path`
to the existing S0 stdout log. The
adopt stage only polls this exact process and its completion receipt. It never
submits S0. If the process has already exited, adoption succeeds only when the
declared receipt passes; otherwise the stage fails without a replacement run.
Linux `/proc` gives the strongest identity. macOS uses `ps` and `lsof` as a
local development fallback.

## JSON plan shape

All paths are absolute. JSON objects reject duplicate or unknown keys. The plan
file itself is pinned externally with `--plan-sha256`; its contents do not
contain a self-referential hash.

Required top-level fields:

| Field | Contract |
| --- | --- |
| `schema` | Exactly `self_audit.server_campaign.v1`. |
| `campaign_id` | Unique ID: letters, digits, underscore, dot, and dash. |
| `campaign_mode` | `training` for the queued experiment; `backup_recovery` for an upload-only continuation. Backup recovery rejects every training stage. |
| `state_dir` | Absolute persistent state and log directory. Keep it with the run. |
| `progress_receipts_jsonl` | Absolute output path for controller step/ETA receipts. |
| `max_wall_clock_seconds` | Integer from 1 through 1,209,600 (14 days). |
| `poll_seconds` | Number from 0.1 through 300. |
| `continue_independent_on_failure` | `false` stops after any failed/blocked stage; `true` permits later stages whose prerequisites succeeded. Dependencies still block. |
| `gpu.visible_device` | One numeric device ID. The controller sets `CUDA_VISIBLE_DEVICES` to this single device and runs one stage at a time. |
| `pins` | Unique named hash pins, described below. |
| `stages` | Finite ordered list of 1–128 named stages. Prerequisites must refer to earlier entries. |
| `env_allowlist` | Optional map of environment names to existing absolute nonsecret configuration paths. Names containing token, secret, password, credential, API key, or auth are rejected. |
| `tmux_target` | Optional simple `session[:window]` target. A separate tmux window follows each stage's stdout log. |

Each pin is exactly:

```json
{"id":"runner_source","kind":"source","path":"/abs/repo/self_audit/server_runner.py","sha256":"64 lowercase hex characters"}
```

`kind` is `source`, `recipe`, or `asset`. Pins identify files. A stage checks
the exact hash of every ID it lists before launch; training also checks while it
runs. Changed or missing inputs stop subsequent scheduling.

Every stage has `id`, `kind`, `role`, `requires`, and `completion`:

```json
{
  "id":"teacher_eval",
  "kind":"command",
  "role":"evaluation",
  "requires":["teacher_train"],
  "pin_ids":["teacher_export","val_manifest"],
  "argv":["/abs/python", "scripts/evaluate_teacher.py", "--export", "/abs/export", "--out", "/abs/receipt.json"],
  "cwd":"/abs/repo",
  "completion":{"path":"/abs/receipt.json","field":"status","equals":"COMPLETED"}
}
```

`kind` is `command` or `adopt`. `role` is `supervised`, `joint`, `teacher`,
`kd`, `evaluation`, `backup`, or `maintenance`. Evaluation and backup command
stages require nonempty `pin_ids`. Each command is a literal argv array and
absolute executable; the controller uses `shell=False`. It rejects shell
programs, Python `-c`, and variable-like `$NAME`, `${NAME}`, or `%NAME%`
arguments. It also rejects credential-bearing argv options. Do not put credentials
or tokens in argv or environment. The child
receives only a small base environment, the one-device setting, and explicitly
allowlisted config paths.

For a training role, add exactly this training declaration and list exactly its
recipe, source, and asset pins in the stage's `pin_ids`:

```json
"training": {
  "steps":10000,
  "recipe_pin":"s0_lock",
  "source_pin_ids":["runner_source"],
  "asset_pin_ids":["train_manifest"],
  "progress_source":{
    "path":"/abs/run/TRAIN_LOG.jsonl",
    "step_field":"step",
    "step_duration_seconds_field":"timings.step_elapsed_seconds"
  }
}
```

Use exactly one of `eta_seconds_field`, `step_duration_seconds_field`, or
`elapsed_seconds_field`. Cumulative elapsed gives `(10000-step)*elapsed/step`;
it includes overhead already incurred by that trainer. For zero-based logs set
`step_offset: 1`; omit it for one-based completed-step counts.
With an ETA field, the controller records that finite, nonnegative value. With
a per-step duration field, it estimates ETA from the rolling mean of the last
100 durations and the fixed 10,000-step budget. Both are written as
`step`, `total_steps`, and `eta_seconds` rows to the configured
`progress_receipts_jsonl`. The source must be JSONL and provide the configured
fields. A malformed progress line fails the stage after its active process
finishes.

For an adopted stage, use `kind: "adopt"`, a training declaration, and this
`process` object instead of `argv`, `cwd`, `pin_ids`, or `resume`:

```json
"process": {
  "pid":12345,
  "start_ticks":987654,
  "cwd":"/workspace/Self-Audit",
  "argv_sha256":"64 lowercase hex characters",
  "log_path":"/workspace/server_compact/run_s0/runner.stdout.log",
  "receipt_sha256_before":null
}
```

Set `receipt_sha256_before` to the current completion receipt's SHA-256, or
`null` if the receipt does not exist yet. The controller refuses adoption if
that file changed after the plan was pinned; successful completion must then
produce a fresh receipt.

Its `completion` receipt is polled until the process exits. PID reuse, cwd or
command-fingerprint changes, missing logs, and exit without the expected
receipt fail closed. Adopt stages never have a command or resume vector.

### Dependencies, teacher gate, and uploads

Every prerequisite ID must appear earlier in `stages`. A KD stage must include
`quality_gate` and include its evaluation stage in `requires`:

```json
"quality_gate": {
  "stage":"teacher_eval",
  "metric":"Macro4",
  "field":"metrics.Macro4",
  "minimum":0.87
}
```

The controller accepts only the exact minimum `0.87`; a missing, nonfinite,
changed, or lower teacher receipt blocks KD. No command can lower this gate.

A backup stage's completion receipt should require verified upload. To retain a
failed upload explicitly, add a failure outcome:

```json
"failure_outcomes":[
  {"field":"status","equals":"UPLOAD_PENDING","label":"UPLOAD_PENDING"}
]
```

If that stage exits unsuccessfully or misses its success receipt while the
receipt says `UPLOAD_PENDING`, state records `remote_verified: false` and
`local_artifacts_retained: true`. The failed stage is terminal and is never
replayed. Stages depending on it are blocked; independent stages run only when
`continue_independent_on_failure` is explicitly `true`. A backup retry must be
a separately reviewed, hash-pinned maintenance campaign and must not include a
training command.

### Exact resume

A command training stage may optionally list exactly one resume command:

```json
"resume": {
  "argv":["/abs/python", "scripts/run_server_compact.py", "run", "--lock", "/abs/lock.json", "--lock-sha256", "LOCK_SHA", "--out", "/abs/run", "--resume-checkpoint", "/abs/run/segments/segment_0001/step_004000", "--resume-sha256", "MANIFEST_SHA"],
  "checkpoint_pins":[
    {"path":"/abs/run/segments/segment_0001/step_004000/MANIFEST.json","sha256":"MANIFEST_SHA"}
  ],
  "required_argv_values":["/abs/run/segments/segment_0001/step_004000","MANIFEST_SHA"]
}
```

List every exact resume artifact in `checkpoint_pins`; its file hash is
rechecked before the resume vector runs. Every checkpoint path and hash must
also appear as an exact argv item, along with all `required_argv_values`. The
controller allows one resume attempt only after an unexpected process
disappearance with no recorded exit result. A known nonzero exit, OOM marker,
receipt failure, or second interruption is terminal. Without this exact block,
an interrupted started stage fails and is never restarted from zero.

## State and logs

The queue lock prevents duplicate supervisors. `STATE.json` and receipt files
are atomically replaced; `EVENTS.jsonl` and progress rows are append-only. Each
attempt has a unique `stages/<id>/attempt_NNN/stdout.log` and execution/result
identity receipts. The stage stdout log is linked to its own tmux viewer window
when `tmux_target` is configured. The controller stores process IDs and hashes,
not command arguments or inherited shell environment.

## PRO4000 continuation (2026-11-02)

The authorized finite order is: adopt the existing S0, one new full EMA B8
reproduction, independent joint supervised ACDC4+M&Ms4, teacher verification,
TRAIN-U cache, S1/crop160, S2/crop256. All compact runs use wide
`[32,64,128,256]`, random seed-42 initialization, batch 32 and 10,000 steps.
The separate B8 reference retains its historical batch 8, crop256 and full
1,000-step supervised warmup inside 10,000 steps. It is a new reproduction;
it does not recover the missing historical teacher identity.

`scripts/run_server_ready_arm.py` consumes one SHA-pinned
`self_audit.server_ready_arm.v1` plan. The `verify` action checks current source
and all already-existing recipe/data pins without allocating GPU. Its
`teacher-binding` action verifies the one declared B8 run, the raw state against
the full checkpoint, lineage, physical evaluation outputs and verified backup.
The existing Macro4 >= 0.87 teacher gate remains enforced. A teacher failure
blocks cache/S1/S2 but does not block the independent joint supervised arm.

The `cache` action builds float32 native TRAIN-U targets with the canonical
cache CLI (B8 inference batch 8), after an image-header-only disk estimate with
a 12 GiB reserve. It never reads U masks. Each compact action derives its
immutable config only from its pinned template and validated teacher/cache,
runs a discarded seven-step profile, checks the common initial-state hash,
then locks and runs the canonical trainer. Profiles cannot initialize training.
Existing outputs are refused: use `run_server_compact.py run` with that arm's
generated config, lock and exact latest checkpoint hashes for an explicit
resume. Do not delete outputs to restart from zero.

Joint supervised uses a metadata-only namespaced development view of ACDC18
and M&Ms32. The canonical evaluator freezes all predictions before scoring.
Selection averages the two dataset Macro4 values equally; per-domain metrics
remain separate. A namespaced ACDC TEST50 view is scored only after best VAL
selection and verified weights backup. Original manifests and payloads remain
unchanged. No M&Ms TEST has been authorized by this queue.

The running S0 remains on its original source/lock. Its step-1000 evaluation
metadata error was fixed by `scripts/resume_server_s0_repaired_eval.py`, which
records and hashes the precise evaluator-only replacement, redoes the failed
scheduled evaluation, and resumes the exact checkpoint with original trainer,
optimizer and RNG state. This does not count as a new S0 initialization.

Actual server paths for the continued queue are under
`/workspace/selfaudit-20261101/campaign_20261102/`. The pinned plan is
`CAMPAIGN.json`; binding inputs are `READY_ARMS.json`; outputs live in `arms/`.
Use the campaign worktree, leaving the active S0 and B8 source checkouts intact:

```bash
cd /workspace/Self-Audit-campaign-20261102
/venv/selfaudit-pro4000/bin/python scripts/run_server_campaign.py status \
  --state-dir /workspace/selfaudit-20261101/campaign_20261102/state
tail -f /workspace/selfaudit-20261101/logs/training_live.log
tmux attach -t ssh_tmux
```

The controller's `stop --state-dir .../state` command stops scheduling after
the active run ends. A stage marked `UPLOAD_PENDING` retains local weights;
it is not silently retrained or called completed. The queue has a finite
48-hour scheduling deadline, not permission to add runs, seeds or joint KD.

The queue is sequential, including evaluation and backup commands. It does not
run GPU discovery or start extra GPUs. OOM markers or nonzero exits fail a
stage; there are no loops, periodic sweeps, implicit retries, or recipe
rewrites. Manufactured subprocess tests exercise controller behavior only;
they are not patient or GPU evidence.
