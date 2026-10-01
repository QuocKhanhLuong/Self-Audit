#!/usr/bin/env python
"""Validate the immutable, superseded v10 historical-224 baseline freeze.

v10 is bound to its original source snapshot (commit 8937fe7, where all 26
bound repository files match). Its repository bindings are therefore checked
against those immutable Git objects, not the current checkout, and the result
never claims that v10 validates the current source. The active freeze for the
current checkout is v11 (``validate_cardiac_benchmark_v11_historical_224_freeze.py``).
Freeze payload/file tamper detection and the grid/manifest/adapter semantic
re-checks are unchanged. ``--against-worktree`` runs the original strict
working-tree binding check, which is expected to fail after the merge.
"""
from __future__ import annotations

import argparse
import hashlib
import subprocess
from pathlib import Path
from typing import Any, Mapping

import validate_cardiac_benchmark_v7_historical_224_freeze as _v7


REPO_ROOT = Path(__file__).resolve().parents[1]
FREEZE = REPO_ROOT / "benchmark_freezes" / "cardiac_benchmark_v10_historical_224"
SCHEMA = "shared_benchmark.cardiac_benchmark_freeze.v10_historical_224"
PREFIX = "cardiac-benchmark-v10-historical-224"
SOURCE_SNAPSHOT = "8937fe7248a6d4f5bb10530fa0b78817af6a40f8"
FreezeValidationError = _v7.FreezeValidationError


def _snapshot_hash(repo: Path, relative: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), "show", f"{SOURCE_SNAPSHOT}:{relative.as_posix()}"],
        capture_output=True,
    )
    if result.returncode != 0:
        raise FreezeValidationError(f"v10 source snapshot unavailable for: {relative}")
    return hashlib.sha256(result.stdout).hexdigest()


def _snapshot_bindings(original):
    def validate(root: Path, values: list[Mapping[str, Any]], label: str) -> None:
        if label != "repository":
            return original(root, values, label)
        for value in values:
            relative = Path(str(value.get("path", "")))
            if relative.is_absolute() or ".." in relative.parts:
                raise FreezeValidationError(f"invalid bound path: {relative}")
            if _snapshot_hash(root, relative) != value.get("sha256"):
                raise FreezeValidationError(f"repository snapshot hash mismatch: {relative}")
    return validate


def validate_freeze(
    freeze_dir: str | Path = FREEZE, repo_root: str | Path = REPO_ROOT, *, against_worktree: bool = False,
) -> dict[str, object]:
    prior = (_v7.SCHEMA, _v7.PREFIX, _v7._validate_bindings)
    try:
        _v7.SCHEMA, _v7.PREFIX = SCHEMA, PREFIX
        if not against_worktree:
            _v7._validate_bindings = _snapshot_bindings(prior[2])
        result = dict(_v7.validate_freeze(freeze_dir, repo_root))
    finally:
        _v7.SCHEMA, _v7.PREFIX, _v7._validate_bindings = prior
    result["repository_binding_mode"] = "worktree" if against_worktree else "source_snapshot"
    result["repository_source_snapshot"] = None if against_worktree else SOURCE_SNAPSHOT
    result["validates_current_checkout_sources"] = against_worktree
    result["status"] = "ACTIVE_WORKTREE_MATCH" if against_worktree else "IMMUTABLE_HISTORICAL_SUPERSEDED_BY_V11"
    return result


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
