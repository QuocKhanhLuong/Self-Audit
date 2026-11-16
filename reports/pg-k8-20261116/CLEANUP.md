# Candidate PG cleanup audit

Date: 2026-11-16
Worktree: `/Users/alvinluong/Self-Audit`
Base observed: `17efb5f` (`main`, matching `origin/main`)
Scope: the coordinator-authorized deletion of exactly three untracked Finder files; no other deletion, rename, commit, push, GPU, SSH, or training action was performed.
Evidence boundary: `SOFTWARE_ONLY` — filesystem/Git status and hash checks only; no patient, server, GPU, or training evidence.

## Decision boundary

The repository contract says Candidate FUSE is the active default and Candidate A, A-3D, and Q3D are preserved comparison tracks. `README.md:3` and `ACTIVE_PIPELINE.json:3-6,8-73` identify the canonical entrypoints and package/config contracts; therefore this report does **not** authorize deletion from `self_audit/candidate_a/`, `self_audit/candidate_a3d/`, `self_audit/candidate_q3d/`, `self_audit/candidate_fuse/`, their `configs/`, tests, or historical docs/reports. All report/run history, datasets, weights, and unrelated dirty files remain protected.

The three Finder files in the completion section were explicitly authorized after an untracked-file check. Every other item below remains a proposal only; no other removal is authorized by this report.

## Approved action completed

The following three paths were verified untracked (`git ls-files --error-unmatch` exit 1 for all three) and then removed explicitly:

| Removed path | Pre-removal bytes | Pre-removal SHA-256 | Post-removal proof |
| --- | ---: | --- | --- |
| `.DS_Store` | 8,196 | `caaadf78b42380e8f61e682e2887585e30d43a7fa7289c82d81b3613ed0fd2ef` | `ABSENT`; not in `git ls-files` |
| `docs/.DS_Store` | 6,148 | `3c17719305fd967ab67a9af702d691186b04bc55c284ce8ed325d7603ba44078` | `ABSENT`; not in `git ls-files` |
| `tests/.DS_Store` | 6,148 | `c566433cbc4503c3ec6faf6fe126771c8e83916dbab634ed0718bf4ca802a34f` | `ABSENT`; not in `git ls-files` |

Post-cleanup `git status --short --untracked-files=all` for these three paths produced no entries. The remaining Finder files under `external/` and `runs/**` are ignored and were intentionally preserved because `external/` is an asset boundary and all run directories are protected history.

## Proposed generated-clutter removals (high confidence, pending coordinator approval)

| Candidate path(s) | Evidence | Safe-removal boundary |
| --- | --- | --- |
| Root `__pycache__/`, `scripts/__pycache__/`, `self_audit/**/__pycache__/`, `tests/__pycache__/`, and `.pytest_cache/`, `.ruff_cache/` | `.gitignore:1-3` ignores all of these; they contain only bytecode/tool caches. Current sizes are approximately 156 KiB, 1.9 MiB, 2.9 MiB, 15 MiB, 304 KiB and 56 KiB respectively. | Remove only the named cache directories; do not traverse `reports/` or `runs/` while doing so. Re-run the focused software checks if any test process is still active. |
| `src/` (only if the coordinator confirms no external process uses this stale cache tree) | `git ls-files src` is empty; `find src -type f` found only `.pyc`; no non-report/non-run source or config references `src/`. The tree is 4.6 MiB and its bytecode timestamps are September 2026, while canonical top-level `self_audit/` sources are dated November 2026. `.gitignore:1,3` explains why Git does not show it. | Treat the whole `src/` tree as ignored stale bytecode, not as source code. Verify no process has set `PYTHONPATH` to this path before removal. |
| ~~`.DS_Store`, `docs/.DS_Store`, `tests/.DS_Store`~~ | **Completed above.** They were untracked Finder metadata with no code/config references. | No further Finder-file removal was performed. `runs/**/.DS_Store` and `external/.DS_Store` remain protected. |

## Temporary duplicate downloader helpers (approval + receipt check required)

| Candidate path | Evidence | Required check before removal |
| --- | --- | --- |
| `scripts/_acdc_download_tmp.py` | Untracked, 6,677 bytes, older timestamp (2026-11-12 15:44). It emits `self_audit.acdc.drive_download.v1`; `rg` found no reference outside the file itself. | Check the authorized ACDC receipt and transfer status first. This is a one-shot helper, but its local receipt path is outside this checkout and is not present at `/root/mount_data/receipts/setup-20261112` in this worktree. |
| `scripts/_acdc_download_full_tmp.py` | Untracked, 10,834 bytes, newer timestamp (2026-11-12 15:56). Its module docstring explicitly calls it a temporary deployment helper outside the reusable pipeline and says it is removed after transfer (`:2-7`); it emits the newer `self_audit.acdc.drive_download.v2`; `rg` found no caller. | Do not remove until the coordinator verifies the transfer is complete, the v2 receipt is durable, and no worker/process still references the path. Once verified, this supersedes the older v1 helper above. |

These two files are the only source-looking temporary/duplicate candidates found by filename, Semble discovery, and repository-wide reference search. Because the receipt is external and absent locally, this report does not claim that either transfer completed.

## Source-code audit result: no safe old-code deletion found

No tracked duplicate implementation was deleted from this worktree:

- The canonical package is the top-level `self_audit/`; the apparent `src/` tree has no source files, only stale ignored bytecode (see above).
- Semble/reference checks show historical and operational scripts remain live. Examples include `scripts/check_mednext_adapter.py` (documented in `docs/baselines/CANDIDATE_A3D.md:78-90`), `scripts/candidate_a_infer.py` (the historical runbook invokes it at `docs/baselines/RUNBOOK.md:122-131`), `scripts/q3d_watch_status.py` (used by the read-only training viewer in `reports/q3d-exploratory-20261112/README.md:92-98`), `scripts/q3d_profile_gpu.py` (referenced by the Q3D plan and queue/watchdog receipts), and `scripts/fuse_transfer_queue.py` (loaded by `tests/test_fuse_transfer_operations.py:123-132`). These are not obsolete merely because they are historical or operational.
- The previous Candidate A cleanup is already recorded in `reports/candidate-a-cleanup-20261110/README.md`; it explicitly says old runtime pieces were removed while reports/data/checkpoints were preserved. Do not repeat that historical deletion against the current checkout without a new reference audit.

No tracked script, config, test, or package was therefore added to a deletion allowlist. Candidate PG should be implemented in its own package/config namespace and must not reuse deletion as a shortcut for namespace separation.

The only filesystem mutations in this follow-up were the three hashes listed under **Approved action completed**. All caches (including `src` bytecode), both untracked downloader helpers, every candidate track, and all reports/runs/datasets/weights remain untouched.

## Protected paths (explicit)

`self_audit/candidate_a/`, `self_audit/candidate_a3d/`, `self_audit/candidate_q3d/`, `self_audit/candidate_fuse/`; `configs/candidate_a/`, `configs/candidate_a3d/`, `configs/candidate_q3d/`, `configs/candidate_fuse/`; `train.py`, `evaluate.py`; all `tests/` source; `docs/`; all `reports/**`; all `runs/**`; `splits/`; `external/`; `checkpoints/`; any dataset/weight/manifest/receipt; and all unrelated untracked files currently shown by `git status` (including `CODEX_PROMPT_SELF_AUDIT_DIAGNOSTICS.md`, the untracked Q3D/maskfree reports, and any worker-owned Candidate PG files).

## Coordinator action requested

The two temporary downloader removals remain blocked pending external receipt/process verification, and all cache removal remains prohibited while the full pytest workflow is active. No tracked source-code deletion is recommended or performed.
