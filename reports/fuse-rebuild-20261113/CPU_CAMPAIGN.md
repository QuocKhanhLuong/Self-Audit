# FUSE campaign integration verification

## Actual synthetic campaign

- Config/manifests: `/tmp/fuse-campaign-task_4761/config.json` and `/tmp/fuse-campaign-task_4761/data/`
- Command: `python scripts/fuse_campaign.py --config /tmp/fuse-campaign-task_4761/config.json --out /tmp/fuse-campaign-task_4761/run --execute --resume --device cpu`
- Final campaign output: `status=COMPLETED`, `device=cpu`, completed stages `T`, `J`, `VAL3`, `EXPORT`, `TEST_ONCE`.
- Evidence: `/tmp/fuse-campaign-task_4761/run/RUN_STATE.json` says campaign COMPLETED; `/tmp/fuse-campaign-task_4761/run/TEST_ONCE/TEST_TRANSACTION.json` says `status=COMMITTED`, `role=test`, and contains exactly `teacher`, `student`, and `fusion` metrics. The campaign one-shot ledger says `COMMITTED`, `attempt=1`.
- Campaign needed two resumes after real integration failures: the initial synthetic T validation was fixed by the data owner; the engine T-to-J optimizer/sampler resume path was fixed by its owner. The final completion traversed the preserved T checkpoint into J, export, and one TEST transaction; earlier errors remain in `previous_error`/logs as failure history.

## Focused checks

- `python -m pytest -q tests/test_fuse_operations.py` — 11 passed.
- `python -m py_compile scripts/fuse_campaign.py scripts/fuse_watchdog.py scripts/fuse_profile.py scripts/fuse_switch.py scripts/fuse_watch_metrics.py` — passed.
- `git diff --check -- scripts/fuse_campaign.py scripts/fuse_watchdog.py scripts/fuse_profile.py scripts/fuse_switch.py scripts/fuse_watch_metrics.py tests/test_fuse_operations.py` — passed.

## Owned changes

- `scripts/fuse_campaign.py`: `--device` through T/J/TEST; fixed CPU campaign handoff integration support and committed-checkpoint recovery safeguards; successful terminal state clears current error.
- `scripts/fuse_watchdog.py`: campaign status takes precedence over per-stage engine status through VAL/export/TEST.
- `scripts/fuse_watch_metrics.py`: handles type/event, per-class Dice, nested fusion gates, dropped pair gate values, independent loss output, and one-based displayed train epoch.
- `tests/test_fuse_operations.py`: regression tests for device threading, metrics, and overall campaign watchdog lifecycle.
- `scripts/fuse_profile.py`, `scripts/fuse_switch.py`: remain within assigned ownership and passed compile/diff checks.

This synthetic CPU campaign is software integration evidence only; it is not real-data, GPU, or model efficacy evidence.
