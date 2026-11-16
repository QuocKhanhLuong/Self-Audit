# PG S1/S2 code boundary

The S1 refresh hook is a no-op: S1 must run against the reviewed old code. At the S1/S2 boundary, the hook requires a clean tracked worktree, fetches `origin/main`, checks that fetched `main` descends from the current HEAD, and uses only a fast-forward merge. It records both SHAs, the complete Git diffstat, and validation command outputs under `<out>/code_updates/`.

S2 validation runs `scripts/check_repo.py` followed by all discovered `tests/test_candidate_pg*.py` files. A failed check retains the refreshed checkout and its output artifacts, writes a `FAILED` receipt, and returns nonzero. A successful receipt is reused only while HEAD remains exactly at its recorded new SHA; changed code requires explicit review and a new output directory.

No S2 refresh was performed while preparing this implementation. The shared worktree contained unrelated tracked edits in `self_audit/candidate_pg/cinema_init.py` and `self_audit/candidate_pg/config.py`; those edits were preserved. No SSH, GPU, data download, commit, or push was performed.
