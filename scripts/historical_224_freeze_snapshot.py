"""Validate a superseded historical-224 freeze against its immutable source snapshot.

A superseded freeze binds repository files that later commits legitimately
changed. Its bound repository files are therefore hashed from the Git objects
of the commit it was generated from, never from the current checkout, and the
report states that the current checkout's sources are not validated. Freeze
payload/file tamper detection and the grid/manifest/adapter semantic re-checks
run unchanged. ``against_worktree=True`` restores the original strict check of
bound files against the current checkout.
"""
from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path
from typing import Any, Mapping

import validate_cardiac_benchmark_v7_historical_224_freeze as _v7


FreezeValidationError = _v7.FreezeValidationError


def _snapshot_hash(repo: Path, snapshot: str, relative: Path, label: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), "show", f"{snapshot}:{relative.as_posix()}"],
        capture_output=True,
    )
    if result.returncode != 0:
        raise FreezeValidationError(f"{label} source snapshot unavailable for: {relative}")
    return hashlib.sha256(result.stdout).hexdigest()


def _snapshot_bindings(original, snapshot: str, label: str):
    def validate(root: Path, values: list[Mapping[str, Any]], kind: str) -> None:
        if kind != "repository":
            return original(root, values, kind)
        for value in values:
            relative = Path(str(value.get("path", "")))
            if relative.is_absolute() or ".." in relative.parts:
                raise FreezeValidationError(f"invalid bound path: {relative}")
            if _snapshot_hash(root, snapshot, relative, label) != value.get("sha256"):
                raise FreezeValidationError(f"repository snapshot hash mismatch: {relative}")
    return validate


def validate_superseded_freeze(
    freeze_dir: str | Path, repo_root: str | Path, *, schema: str, prefix: str, label: str,
    source_snapshot: str, superseded_by: str, against_worktree: bool = False,
) -> dict[str, object]:
    prior = (_v7.SCHEMA, _v7.PREFIX, _v7._validate_bindings)
    try:
        _v7.SCHEMA, _v7.PREFIX = schema, prefix
        if not against_worktree:
            _v7._validate_bindings = _snapshot_bindings(prior[2], source_snapshot, label)
        result = dict(_v7.validate_freeze(freeze_dir, repo_root))
    finally:
        _v7.SCHEMA, _v7.PREFIX, _v7._validate_bindings = prior
    result["repository_binding_mode"] = "worktree" if against_worktree else "source_snapshot"
    result["repository_source_snapshot"] = None if against_worktree else source_snapshot
    result["validates_current_checkout_sources"] = against_worktree
    result["status"] = "ACTIVE_WORKTREE_MATCH" if against_worktree else "IMMUTABLE_HISTORICAL_SUPERSEDED"
    result["superseded_by"] = superseded_by
    return result
