# Independent review — SOFTWARE_ONLY

Orca run: `run_9e1f9e97a935`. After the user's request to replace Claude, the
remaining implementation and independent reviews used `gpt-6-astra`.

The engine/helper, artifact/export consumers, CLI/campaign/profile, watchdog,
and queue were checked with focused CPU fixtures. The independent reviewer
verified strict-T import into waived Q/P/J, unchanged-update failed-Q resume,
all-zero physiology fallback, paused/resumed J, and bitwise student export
cold-load in a fresh interpreter on three synthetic native volumes.

Two queue findings were raised and resolved before deployment:

1. **P1:** the first liveness check recognized trainers but missed legacy queue
   writers between child processes. The final check excludes only its own PID
   and recognizes legacy/current queue scripts, including the actual
   `q3d_resume_epoch_queue.py` wrapper.
2. **P2:** a completed experiment could be skipped solely from run-state status.
   The final skip path revalidates producer policies, artifact SHA pins, and the
   J checkpoint/receipt before accepting completion.

The follow-up also confirmed that a failed bounded J profile is preserved with
its original bytes and retry receipt before retry; valid completed profiles
are reused. Malformed watchdog restart history blocks recovery, and restart
attempts are durably reserved before invoking Supervisor.

Final focused queue implementation tests: **36 passed**. The independent
follow-up ran the then-current **35 tests**, separately probed the added legacy
wrapper name, and reported no remaining P1/P2 in its review scope. The final
baseline/watchdog/CLI recheck was **68 passed**. Full-suite evidence is recorded
separately in `pytest.txt` after it completes.

No reviewer ran real patient training or GPU work. Real-checkpoint recovery,
server execution and monitor activation require the separate launch receipt.
