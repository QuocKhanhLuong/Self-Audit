# PG readiness-gated queue

## Scope and engine compatibility

`scripts/pg_server_queue.py` is a durable local supervisor for the canonical
PG engine. Its stages are shared S1, S2, and S3; one selected-arm `arms`
invocation each for A4, A3, A2, A1; then one base-config `arms` invocation for
finalization. `train.py` accepts `--pg-stage arms --pg-arm A# --pg-resume`, and
the engine sets `selected_arms=(arm,)` when a single arm is supplied. Those
calls do not finalize. Each writes a checkpoint whose identity includes the
effective arm config, arm name, switches, source/data identity, and runtime
flags. The final base-config `arms --pg-resume` call loads each arm checkpoint
with the same effective identity, restores its best model, skips training when
its update cursor is at budget, then calls the finalizer once after assembling
all four selected models.

The queue tests the actual argv for each stage. It stores a `RUNNING` stage
before launch and prints exact argv with flush enabled for the live log.
Completed queue stages are skipped. FAILED or interrupted RUNNING queue state
blocks another invocation; there is no silent retry or injected resume after
failure. `--resume-failed ROOT_APPROVED_REASON` is the explicit recovery
switch; it appends the prior status, active stage, stage receipts, and reason to
`attempt_history`, then reuses completed stages and resumes the failed or
interrupted stage via its normal `--pg-resume` command. Root must review logs
and engine checkpoints before supplying the reason.

## Readiness receipts

Both `--data-ready` and `--source-ready` point to JSON objects with
`{"status":"READY"}`. Optional `kind` must match `data` or `source`.
Optional `artifacts` may be a list of `{path, sha256}` objects or a mapping
from path to SHA256 (or `{sha256: ...}`); paths must exist and pinned bytes
must match. Optional `config_pins` (or `config`) may include `path` and
`sha256`/`config_sha256`, which are checked against the selected queue config.
Data receipts may include `manifests` pins, checked against the base config.
Source receipts may include `source_sha256` entries for `config`, `split_lock`,
and `source_k8`, checked against the same identities. Unknown receipt fields
are retained by the producer and ignored by the queue.

The queue polls while either receipt path is absent, a recognized preparation
status is present, or a pinned expected artifact path is not yet present; it
persists `status: WAITING` and does not launch `train.py`. FAILED, malformed
JSON/status, or a hash mismatch fails promptly. `--max-wait-seconds` sets a
finite wait (default 24 hours); timeout leaves WAITING evidence and returns a
nonzero status. The queue requires exact READY statuses, all supplied pins,
and matching bytes for all three config-pinned manifests before launch.

## State, locking, and evidence

The queue holds an exclusive nonblocking `fcntl` lock on `QUEUE.lock`, so a
second concurrent writer fails immediately. `QUEUE_STATE.json` is replaced
atomically after readiness changes, before launch, and after every subprocess.
It records config and queue-owned source hashes, ordered argv, stage status,
timestamps, return codes, log paths and log SHAs, readiness receipt SHAs, and
the arm order. Each stage gets an append-only `logs/<STAGE>.log`. Nonzero
return code or runner exception stops the pipeline. No automatic rerun follows
an error, OOM, or TEST failure.

The queue's own source hash list includes `train.py`, `scripts/pg_campaign.py`,
and `self_audit/candidate_pg/engine.py`; config SHA is the exact input JSON
bytes. Stage command lines always use the canonical base config; the engine
owns its fixed four-arm switch configs and finalization. Completed queue stages
are not replayed when the queue is invoked again with the same output/config.

## Example

```sh
python scripts/pg_server_queue.py \
  --config configs/candidate_pg/pg_k8_base.json \
  --out /path/to/new-pg-run \
  --data-ready /path/to/data-ready.json \
  --source-ready /path/to/source-ready.json \
  --device cuda
```

`--device` defaults to `cuda`. This tool does not SSH, launch a shell task
through a supervisor, download data, or start training until both READY
receipts are verified. Implementation tests use temporary paths and mocked
subprocesses only; they are not server, GPU, or patient evidence.

## Remaining limitation

The engine has no finalization-only command. The last base arms call rebuilds
all four selected models from arm checkpoints in memory; when checkpoints are
at budget, it does not repeat their optimizer updates. On explicit recovery,
if a prior TEST transaction is COMMITTED, the queue verifies its TEST manifest
pin, all four export bytes/SHAs, and committed `FINALIZATION.json` and
`FINAL_METRICS.json`; then it records the final stage as reused and does not
invoke the finalizer or duplicate TEST. If TEST is committed but those final
receipts are incomplete or inconsistent, the queue stops and leaves the
committed TEST intact for manual completion; it never reruns TEST. A failed or
interrupted final call without committed TEST can be explicitly resumed after
checkpoint review.
