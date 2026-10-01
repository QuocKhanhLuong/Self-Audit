#!/usr/bin/env python
"""Validate the immutable, superseded v10 historical-224 baseline freeze.

v10 is bound to its original source snapshot (commit 8937fe7, where all 26
bound repository files match). Its repository bindings are therefore checked
against those immutable Git objects, not the current checkout, and the result
never claims that v10 validates the current source. The active freeze for the
current checkout is v12 (``validate_cardiac_benchmark_v12_historical_224_freeze.py``).
Freeze payload/file tamper detection and the grid/manifest/adapter semantic
re-checks are unchanged. ``--against-worktree`` runs the original strict
working-tree binding check, which is expected to fail on newer source.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import historical_224_freeze_snapshot as _snapshot


REPO_ROOT = Path(__file__).resolve().parents[1]
FREEZE = REPO_ROOT / "benchmark_freezes" / "cardiac_benchmark_v10_historical_224"
SCHEMA = "shared_benchmark.cardiac_benchmark_freeze.v10_historical_224"
PREFIX = "cardiac-benchmark-v10-historical-224"
SOURCE_SNAPSHOT = "8937fe7248a6d4f5bb10530fa0b78817af6a40f8"
FreezeValidationError = _snapshot.FreezeValidationError


def validate_freeze(
    freeze_dir: str | Path = FREEZE, repo_root: str | Path = REPO_ROOT, *, against_worktree: bool = False,
) -> dict[str, object]:
    return _snapshot.validate_superseded_freeze(
        freeze_dir, repo_root, schema=SCHEMA, prefix=PREFIX, label="v10",
        source_snapshot=SOURCE_SNAPSHOT, superseded_by="cardiac_benchmark_v11_historical_224",
        against_worktree=against_worktree,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze-dir", type=Path, default=FREEZE)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--against-worktree", action="store_true",
                        help="strict original check of bound files against the current checkout")
    args = parser.parse_args()
    import json

    print(json.dumps(validate_freeze(args.freeze_dir, args.repo_root, against_worktree=args.against_worktree),
                     sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
