# Shared W&B run-identity wrapper API for Wave 4

Worker: parallel Claude. Owned paths only — the `WandbLogger` class in
`src/self_audit/training/_utils.py`, `tests/test_runtime_wandb_identity.py` (new), and this report.
`unified_trainer.py` and the configs were not touched, and nothing outside the `WandbLogger` class
body in `_utils.py` was changed, so the concurrent `_utils` fixes are preserved.
No commits, no push, no recipe/model/loss change, and no network logging anywhere.

Delaying logger initialization until a validated resume identity is known, and preserving RNG
around telemetry, remain the main AGY's integration work; this is the wrapper it calls.

Environment: `/private/tmp/self-audit-torch241/bin/python` (Python 3.10.21, torch 2.4.1 CPU),
`PYTHONPATH=src:.`, single-threaded.

## Test run

```
PYTHONPATH=src:. PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  /private/tmp/self-audit-torch241/bin/python -m pytest tests/test_runtime_wandb_identity.py -q -p no:randomly
```

**40 passed in 0.46s.** Fully mocked: no real run is created and no request leaves the process.

Because the change lives inside a class that other tests already exercise, I also ran the two
existing files that construct `WandbLogger` directly —
`tests/test_wandb_and_tqdm.py` and `tests/test_artifact_io.py` — **24 passed in 0.88s**. A final
combined run of those four files plus `tests/test_checkpoint_commit.py` gave **129 passed in 0.99s**.
No historical broad suite was run; root's final integrated suite covers the rest.

## The concern this addresses

Wave 4's W&B item: `UnifiedTrainer.__init__` builds the logger before
`resume_from_checkpoint` can recover a run id, so every resume starts an orphan run. Fixing that
needs two separable things — *when* the logger is constructed (trainer) and *what* it can be told
and asked (this wrapper). The wrapper also has to be honest about a limit of the backend: an
offline run cannot be resumed server-side, so reusing an id offline must never be reported as a
continued run.

## API (contract sent to root before implementation)

Two optional keyword parameters, appended after the existing ones so every current call site and
every positional order is unchanged:

```python
WandbLogger(
    enabled=False, project=None, entity=None, run_name=None, group=None,
    tags=None, config=None, mode=None, dir=None,
    run_id: str | None = None,       # new
    resume: str | None = None,       # new
)
```

### Resume value validation, against the current official reference

`ALLOWED_RESUME_VALUES = ("allow", "never", "must", "auto")`.

Source: the current `wandb.init` reference (docs.wandb.ai/ref/python/init) documents exactly those
four literals — `allow` resumes if a run with the given `id` exists and otherwise creates a new
one, `never` errors if the run exists, `must` errors if it does not, `auto` resumes a run that
crashed locally — with `None` as the default and `True`/`False` as **deprecated**.

The wrapper accepts the four literals case-insensitively (and strips surrounding whitespace) plus
`None`, and refuses everything else — booleans included — with a `ValueError` naming the allowed
set. A typo must not silently degrade into a fresh run. `run_id` must be a non-blank string or
`None`.

### Forwarding rules

| Condition | `id` forwarded | `resume` forwarded |
|---|---|---|
| `mode="online"` | when `run_id` given | when `resume` given |
| `mode="offline"` / `"disabled"` / anything else | when `run_id` given | **never** |

An id is still meaningful offline — it names the local run directory — so it is forwarded in any
mode. `resume` is meaningless without a backend, so it is recorded and reported, never forwarded.
Mode comparison is case-insensitive.

### Identity attributes and `identity_summary`

`identity_summary` is a property returning a fresh dict of only `str`/`bool`/`int`/`None` values —
no logger object, no exception object, nothing torch — so the trainer can put it straight into
checkpoint `extra` metadata or a JSON report. The same names are also plain attributes.

| Key | Type | Meaning |
|---|---|---|
| `schema_version` | int | `1` (`WandbLogger.IDENTITY_SCHEMA_VERSION`) |
| `enabled_requested` | bool | what the caller asked for |
| `enabled_effective` | bool | what survived init |
| `mode` | str | the mode as configured |
| `requested_run_id` | str \| None | validated `run_id` |
| `actual_run_id` | str \| None | the id on the run object the SDK returned, **only when it is genuinely a non-blank string**; never copied from the request, never invented |
| `requested_resume` | str \| None | normalized `resume` |
| `effective_resume` | str \| None | exactly what reached `wandb.init`, so `None` whenever offline/disabled/not initialized |
| `backend_resume_performed` | bool | `True` only when resume was forwarded **and** the SDK's own `run.resumed` is literally `True`; a matching run id is never evidence |
| `resume_limitation` | str \| None | plain-text reason a requested resume was not or could not be honored |
| `sdk_available` | bool \| None | `None` when init was never attempted |
| `init_status` | str | `not_attempted` \| `ok` \| `sdk_missing` \| `failed` |
| `init_error` | str \| None | message text only |
| `finish_status` | str | `not_finished` \| `ok` \| `failed` |
| `finish_count` | int | effective finish calls; idempotent, so at most 1 |
| `failed_log_count` | int | unchanged existing counter |
| `telemetry_error_count` | int | `len(telemetry_errors)` |

`backend_resume_performed` follows the correction root sent after reviewing the contract: a
matching run id is **not** proof of a resume, because `resume="allow"` creates a new run under the
requested id when none exists and `resume="auto"` starts fresh when there is nothing to recover.
The only explicit proof is the SDK's own `run.resumed`, so:

* `run.resumed is True` → `backend_resume_performed = True`, no limitation.
* `run.resumed is False` → `False`, and `resume_limitation` says the SDK reports
  `run.resumed=False` and a new backend run was created.
* no usable `resumed` attribute (older or partial SDK) → `False`, and `resume_limitation` says the
  resume is *unconfirmed* and that a matching run id is not evidence.

The raw tri-state is also available as the `sdk_reported_resumed` attribute (`True`/`False`/`None`).
It is not one of the accepted `identity_summary` keys, so the summary key set is unchanged; the
distinction it carries is already in the `resume_limitation` text that gets persisted. Say the word
if you want it as its own field.

### `finish` idempotence

The first effective `finish` sets `finish_count = 1` and `finish_status` to `ok` or `failed`;
every later call returns immediately without touching the SDK, without incrementing a counter and
without overwriting the recorded status. A lifecycle that finishes on both a normal and a failure
path can therefore no longer double-count telemetry failures. A logger that never started stays at
`finish_count = 0` / `not_finished`, which is the truthful reading of "there was nothing to
finish".

### Preserved behaviour

`enabled`, `project`, `entity`, `run_name`, `group`, `tags`, `config`, `mode`, `dir`, `_run`,
`last_error`, `failed_log_count`, `telemetry_errors` and the `log` / `set_summary` / `log_images` /
`finish` signatures are unchanged, as are the strict `clean_wandb_payload` conversion (non-finite
floats become explicit nulls, not zeros) and the graceful optional-telemetry failure policy
(a telemetry error is counted and recorded, never raised into training).

## Environment finding worth passing on

In this repo `import wandb` from the repo root resolves to the local `./wandb` **run-output
directory** as an empty namespace package, because the current directory is on `sys.path`:

```
python -c "import wandb; print(wandb.__file__, list(wandb.__path__))"
None ['/Users/alvinluong/Self-Audit/wandb']
```

The import therefore succeeds while `wandb.init` does not exist. The previous code caught the
resulting `AttributeError` in its generic `except Exception` arm and reported it as an
initialization failure, incrementing `failed_log_count`. The wrapper now checks that `wandb.init`
is actually callable and classifies this as `init_status="sdk_missing"` with
`sdk_available=False` and no telemetry-failure count, which is the truthful reading.

**Correction to an earlier version of this report:** the local `./wandb` directory does *not* have
to be removed once the SDK is installed. A regular installed package wins over a namespace-package
candidate — the import system only falls back to a namespace package when no regular module or
package was found on the path — so installing `wandb` is sufficient on its own, and the directory
can stay where the SDK puts it. The shadowing above is only what happens with the SDK **absent**.
So the practical note is narrower than first stated: in an environment without `wandb` installed,
the wrapper reports `sdk_missing` rather than a spurious init failure, and nobody should read that
as "online resume does not work".

## Test coverage map

| Area | Tests |
|---|---|
| Actual id read from the SDK, never guessed | `test_actual_run_id_is_read_from_the_sdk_and_never_guessed` (non-string id → `None`) |
| Resume proven only by `run.resumed` | `test_online_resume_is_confirmed_only_by_the_sdk_resumed_flag` |
| Matching id alone is never a resume (root's correction) | `test_matching_run_id_alone_is_never_treated_as_a_resume` (`allow` + matching id + `resumed=False`), `test_matching_run_id_with_no_resumed_flag_stays_unconfirmed` |
| Online resume that the backend did not honor | `test_online_resume_with_a_different_returned_id_is_not_claimed_as_resumed`, `test_online_resume_without_an_id_cannot_confirm_continuation` |
| Offline requested resume recorded but not forwarded | `test_offline_requested_resume_is_recorded_but_never_forwarded` |
| Offline id reuse is not a backend resume | `test_offline_reusing_the_same_id_does_not_claim_a_continuing_backend_run` |
| Non-online modes never forward resume | `test_non_online_modes_never_forward_resume` (`offline`, `disabled`, `OFFLINE`), `test_online_mode_detection_is_case_insensitive` |
| Documented resume literals accepted, unknown and deprecated bool refused | `test_documented_resume_values_are_accepted`, `test_unknown_or_deprecated_resume_values_are_refused` (`"yes"`, `""`, `"continue"`, `True`, `False`, `1`, `0`, object) |
| `run_id` validation | `test_malformed_run_id_is_refused` (blank, whitespace, `True`, int, object) |
| Disabled logger starts nothing | `test_disabled_logger_never_starts_a_run` |
| Missing SDK vs shadowed module | `test_uninstalled_sdk_is_reported_as_missing_not_as_a_failure`, `test_shadowed_wandb_module_without_init_is_reported_as_missing_sdk` |
| Init failure recorded truthfully, telemetry stays a no-op afterwards | `test_init_failure_is_recorded_truthfully_and_disables_logging` |
| `finish` idempotent on success and on failure | `test_finish_is_idempotent_and_records_success_once`, `test_finish_failure_is_recorded_once_and_not_retried`, `test_finish_on_a_logger_that_never_started_is_a_noop` |
| Backward compatibility | `test_existing_construction_without_id_or_resume_is_unchanged` (no `id`/`resume` keys, NaN still becomes null), `test_positional_construction_still_matches_the_previous_signature` |
| Checkpoint-safe summary | `_assert_summary_is_checkpoint_safe` runs in 6 tests: exact key set, scalar-only values, no `BaseException`, `json.dumps` round-trip |

## Notes for the integrating AGY

- **Construct late.** Nothing here defers construction; `_initialize` still runs from `__init__`
  when `enabled` is true. Build the logger after resume has validated the run identity, then pass
  the recovered `run_id` (and `resume="must"` if you want a hard failure when the backend run is
  absent).
- **Persist `identity_summary`, not the logger.** It is already scalar-only; drop it into
  checkpoint `extra` or the report as-is. Do not put the logger or `last_error` anywhere near
  checkpoint metadata.
- **Offline semantics to surface in the report.** When `mode != "online"`,
  `backend_resume_performed` is `False` and `resume_limitation` carries the sentence explaining why.
  A resumed offline attempt is a *new local run that reuses an id*, and the report should say that
  rather than implying continuity.
- **Do not read a resume off the id.** `backend_resume_performed` is the only field that claims a
  resume, and it is `True` only on an explicit `run.resumed is True`. When it is `False`, read
  `resume_limitation` to see whether the backend said "not resumed" or said nothing at all.
- **`resume="must"` is the strict option** if a missing backend run should fail the attempt rather
  than silently start fresh; the wrapper forwards it unchanged and the SDK raises, which surfaces
  as `init_status="failed"` with the backend's message in `init_error`.
- **RNG around telemetry is not handled here.** `wandb.init` and `wandb.log` may consume global
  random streams; preserving RNG across telemetry, and not logging a completed epoch before its
  checkpoint commits, stay on your side.
- **`failed_log_count` semantics unchanged**, so the existing
  `"telemetry_error_count": self.logger.failed_log_count` uses in `unified_trainer.py` keep
  working. Note that `identity_summary["telemetry_error_count"]` is `len(telemetry_errors)`, which
  also counts a `finish` failure — `failed_log_count` does not.
