# W&B Tracking for the Self-Audit v3 Pipeline

## Goal

Add optional W&B tracking to the current `run_full_pipeline_v3.py` orchestration without changing the no-GT data contract, stage order, artifact contents, or default offline/local behavior.

## Scope and success criteria

- The v3 runner exposes explicit W&B CLI controls for enablement, mode, project, entity, and run name.
- One parent W&B run represents a complete teacher → frozen evaluation → optional student pipeline.
- Teacher and student epoch metrics, stage lifecycle events, final evaluation metrics, runtime, GPU identity, and the final summary are logged when W&B is enabled.
- W&B authentication is supplied through the normal `WANDB_API_KEY` environment variable; no secret is committed.
- W&B initialization or logging failures do not corrupt or abort the scientific pipeline; local logs and JSON artifacts remain authoritative.
- Existing invocation behavior is unchanged when W&B is not enabled.

## Design

### Run ownership

The parent orchestrator owns exactly one W&B run. Child processes remain responsible for training and artifact generation; the parent reads their existing JSON outputs after each stage and sends normalized metrics to W&B. This avoids fragmented child runs and keeps W&B telemetry separate from the no-GT artifact and freeze contracts.

### CLI

Add to `scripts/run_full_pipeline_v3.py`:

- `--wandb`: enable tracking.
- `--no-wandb`: explicitly disable tracking.
- `--wandb-mode {online,offline,disabled}`: select backend mode; default remains disabled unless `--wandb` is supplied.
- `--wandb-project`: project name, default `self-audit-v3`.
- `--wandb-entity`: optional W&B entity/team.
- `--wandb-run-name`: optional run name; otherwise derive a stable name from the output directory.

No API key flag is added. The SDK reads `WANDB_API_KEY` from the environment.

### Logged data

The adapter will sanitize JSON-compatible values and use namespaced keys:

- `pipeline/*`: dataset, stage, epoch, global progress, runtime, device/GPU.
- `teacher/*`: metrics from `teacher/train_metrics.json`.
- `evaluation/*`: values from `evaluation_val.json`.
- `student/*`: metrics available from the student run artifacts.

At completion, the parent logs the final summary as a W&B table/config payload and optionally uploads only small JSON/text artifacts. Model checkpoints and pseudo-label arrays are not uploaded automatically.

### Failure behavior

The adapter is a no-op when disabled or when the `wandb` package is unavailable. Initialization and log exceptions are captured and printed as warnings; the subprocess pipeline continues. The final local `pipeline.log` records the telemetry status and any error message.

### Testing

Add focused unit tests that mock the W&B SDK and verify:

- disabled mode performs no SDK calls;
- enabled mode initializes one run with the requested project/entity/name/mode;
- JSON metrics are flattened/sanitized and logged with stage prefixes;
- SDK initialization/logging failures degrade to local-only execution;
- CLI parsing preserves the existing default-disabled behavior.

Run the focused tests, the v3 CLI help/import smoke test, and the existing v3-related test subset before committing.

## Alternatives rejected

- Separate W&B runs in every teacher/student subprocess: gives fragmented experiment identity and makes resume/failure interpretation harder.
- Hard-coding W&B settings or the API key in YAML/source: risks credential leakage and prevents per-server/project configuration.
- Modifying the no-GT freeze schema to carry telemetry: couples optional observability to scientific artifact verification unnecessarily.
