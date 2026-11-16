#!/usr/bin/env python3
"""Refresh the PG local checkout at the reviewed S1/S2 boundary.

S1 is intentionally a no-op.  S2 performs one fast-forward-only refresh and
pins both the boundary change and its validation output under ``out``.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Callable


ROOT = Path(__file__).resolve().parents[1]
RECEIPT_SCHEMA = "self_audit.candidate_pg.main_refresh.v1"


def _run(runner: Callable[..., subprocess.CompletedProcess[str]], args: list[str], *, cwd: Path) -> subprocess.CompletedProcess[str]:
    return runner(args, cwd=cwd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)


def _git(runner: Callable[..., subprocess.CompletedProcess[str]], repo: Path, *args: str) -> str:
    result = _run(runner, ["git", *args], cwd=repo)
    if result.returncode:
        raise RuntimeError(f"git {' '.join(args)} failed ({result.returncode}):\n{result.stdout}")
    return result.stdout.strip()


def _timestamp() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _status(repo: Path, runner: Callable[..., subprocess.CompletedProcess[str]]) -> list[str]:
    return [line for line in _git(runner, repo, "status", "--porcelain=v1", "--untracked-files=all").splitlines() if line]


def _atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def refresh(stage: str, out: Path, repo: Path = ROOT, *,
            runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run) -> dict:
    """Execute one stage boundary; injectable runner supports hermetic tests."""
    repo = repo.resolve()
    out = out.resolve()
    if stage not in {"S1", "S2"}:
        raise ValueError("stage must be S1 or S2")
    if stage == "S1":
        return {"schema": RECEIPT_SCHEMA, "status": "NOOP", "stage": "S1", "reason": "S1 runs on the verified old code"}

    updates_dir = out / "code_updates"
    updates_dir.mkdir(parents=True, exist_ok=True)
    receipt_path = updates_dir / "S2_MAIN_REFRESH.json"
    if receipt_path.exists() and not receipt_path.is_file():
        raise RuntimeError(f"S2 receipt path is occupied by a non-file artifact: {receipt_path}")
    if receipt_path.exists():
        prior = json.loads(receipt_path.read_text(encoding="utf-8"))
        if prior.get("schema") != RECEIPT_SCHEMA or prior.get("status") != "PASS" or prior.get("stage") != "S2":
            raise RuntimeError("existing S2 refresh receipt is not a successful recognized receipt")
        current = _git(runner, repo, "rev-parse", "HEAD")
        if current != prior.get("new_sha"):
            raise RuntimeError(f"S2 refresh receipt pins {prior.get('new_sha')}, current HEAD is {current}; refusing changed code")
        return prior

    status = _status(repo, runner)
    tracked_dirty = [line for line in status if not line.startswith("??")]
    if tracked_dirty:
        raise RuntimeError("S2 refresh requires a clean tracked working tree; tracked changes found:\n" + "\n".join(tracked_dirty))

    old_sha = _git(runner, repo, "rev-parse", "HEAD")
    _git(runner, repo, "fetch", "origin", "main")
    target = _git(runner, repo, "rev-parse", "FETCH_HEAD")
    ancestry = _run(runner, ["git", "merge-base", "--is-ancestor", old_sha, target], cwd=repo)
    if ancestry.returncode != 0:
        raise RuntimeError(f"origin/main is not a descendant of old HEAD {old_sha}; refusing non-fast-forward update")
    _git(runner, repo, "merge", "--ff-only", target)
    new_sha = _git(runner, repo, "rev-parse", "HEAD")
    if new_sha != target:
        raise RuntimeError(f"fast-forward ended at {new_sha}, expected fetched origin/main {target}")
    post_merge_status = _status(repo, runner)
    if post_merge_status:
        raise RuntimeError("worktree changed during S2 refresh; refusing validation:\n" + "\n".join(post_merge_status))
    diffstat = _git(runner, repo, "diff", "--stat", old_sha, new_sha)
    pg_tests = sorted(str(path.relative_to(repo)) for path in (repo / "tests").glob("test_candidate_pg*.py"))
    if not pg_tests:
        raise RuntimeError("no tests/test_candidate_pg*.py paths discovered; refusing to report refresh success")
    validation_commands = [
        [sys.executable, "scripts/check_repo.py"],
        [sys.executable, "-m", "pytest", "-q", *pg_tests],
    ]
    results = []
    for command in validation_commands:
        result = _run(runner, command, cwd=repo)
        results.append({"command": command, "returncode": result.returncode, "output": result.stdout})
        (updates_dir / ("CHECK_REPO.txt" if command[1:3] == ["scripts/check_repo.py"] else "FOCUSED_TESTS.txt")).write_text(
            result.stdout, encoding="utf-8")
        if result.returncode:
            failed = {
                "schema": RECEIPT_SCHEMA, "status": "FAILED", "stage": "S2", "old_sha": old_sha,
                "new_sha": new_sha, "diffstat": diffstat, "validation": results,
                "recorded_utc": _timestamp(),
            }
            _atomic_json(receipt_path, failed)
            raise RuntimeError(f"S2 validation failed: {command!r} returned {result.returncode}; artifacts retained under {updates_dir}")
    receipt = {
        "schema": RECEIPT_SCHEMA, "status": "PASS", "stage": "S2", "old_sha": old_sha,
        "new_sha": new_sha, "diffstat": diffstat, "validation": results,
        "validation_sha256": {p.name: _sha256(p) for p in (updates_dir / "CHECK_REPO.txt", updates_dir / "FOCUSED_TESTS.txt")},
        "recorded_utc": _timestamp(),
    }
    _atomic_json(receipt_path, receipt)
    return receipt


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True, choices=("S1", "S2"))
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--repo", type=Path, default=ROOT)
    args = parser.parse_args(argv)
    try:
        result = refresh(args.stage, args.out, args.repo)
    except Exception as exc:
        print(f"[pg-refresh] failed: {exc}", file=sys.stderr)
        return 2
    print(f"[pg-refresh] stage={args.stage} status={result['status']}" +
          (f" old={result['old_sha'][:12]} new={result['new_sha'][:12]}" if args.stage == "S2" and result["status"] == "PASS" else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
