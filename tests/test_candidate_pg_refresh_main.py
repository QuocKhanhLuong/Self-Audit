from __future__ import annotations

import importlib.util
import subprocess
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "pg_refresh_main.py"
SPEC = importlib.util.spec_from_file_location("pg_refresh_main", SCRIPT)
pg_refresh_main = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(pg_refresh_main)


def git(repo: Path, *args: str) -> str:
    result = subprocess.run(["git", *args], cwd=repo, check=True, text=True, capture_output=True)
    return result.stdout.strip()


def repo_pair(tmp_path: Path) -> tuple[Path, Path]:
    origin = tmp_path / "origin.git"
    work = tmp_path / "work"
    subprocess.run(["git", "init", "--bare", str(origin)], check=True, capture_output=True)
    subprocess.run(["git", "init", "-b", "main", str(work)], check=True, capture_output=True)
    git(work, "config", "user.email", "test@example.invalid")
    git(work, "config", "user.name", "Test")
    (work / "tracked.txt").write_text("old\n")
    (work / "tests").mkdir()
    (work / "tests" / "test_candidate_pg_fixture.py").write_text("def test_fixture():\n    assert True\n")
    git(work, "add", "tracked.txt")
    git(work, "add", "tests/test_candidate_pg_fixture.py")
    git(work, "commit", "-m", "old")
    git(work, "remote", "add", "origin", str(origin))
    git(work, "push", "-u", "origin", "main")
    return origin, work


def advance_origin(tmp_path: Path, origin: Path, work: Path) -> str:
    writer = tmp_path / "writer"
    subprocess.run(["git", "clone", "-b", "main", str(origin), str(writer)], check=True, capture_output=True)
    git(writer, "config", "user.email", "test@example.invalid")
    git(writer, "config", "user.name", "Test")
    (writer / "added.txt").write_text("new\n")
    git(writer, "add", "added.txt")
    git(writer, "commit", "-m", "new")
    git(writer, "push", "origin", "main")
    return git(writer, "rev-parse", "HEAD")


def test_s1_is_noop_even_when_origin_advances(tmp_path: Path) -> None:
    origin, work = repo_pair(tmp_path)
    old = git(work, "rev-parse", "HEAD")
    advance_origin(tmp_path, origin, work)
    result = pg_refresh_main.refresh("S1", tmp_path / "out", work)
    assert result["status"] == "NOOP"
    assert git(work, "rev-parse", "HEAD") == old
    assert not (tmp_path / "out" / "code_updates").exists()


def test_s2_refuses_unrelated_tracked_dirty_file(tmp_path: Path) -> None:
    origin, work = repo_pair(tmp_path)
    advance_origin(tmp_path, origin, work)
    (work / "tracked.txt").write_text("local edit\n")
    with pytest.raises(RuntimeError, match="clean tracked working tree"):
        pg_refresh_main.refresh("S2", tmp_path / "out", work)
    assert git(work, "rev-parse", "HEAD") == git(work, "rev-parse", "origin/main")
    assert not (tmp_path / "out" / "code_updates" / "S2_MAIN_REFRESH.json").exists()


def test_s2_validation_failure_is_not_reported_pass_and_retains_outputs(tmp_path: Path) -> None:
    origin, work = repo_pair(tmp_path)
    old = git(work, "rev-parse", "HEAD")
    new = advance_origin(tmp_path, origin, work)
    real_run = subprocess.run

    def failing_runner(args, **kwargs):
        if args[0] == "python" or (len(args) > 1 and args[1] == "scripts/check_repo.py"):
            return subprocess.CompletedProcess(args, 7, "intentional validation failure\n")
        return real_run(args, **kwargs)

    with pytest.raises(RuntimeError, match="validation failed"):
        pg_refresh_main.refresh("S2", tmp_path / "out", work, runner=failing_runner)
    updates = tmp_path / "out" / "code_updates"
    receipt = __import__("json").loads((updates / "S2_MAIN_REFRESH.json").read_text())
    assert receipt["status"] == "FAILED"
    assert receipt["old_sha"] == old and receipt["new_sha"] == new
    assert (updates / "CHECK_REPO.txt").read_text() == "intentional validation failure\n"


def test_s2_fast_forward_records_sha_and_reuses_same_success(tmp_path: Path) -> None:
    origin, work = repo_pair(tmp_path)
    old = git(work, "rev-parse", "HEAD")
    new = advance_origin(tmp_path, origin, work)

    def passing_runner(args, **kwargs):
        if len(args) > 1 and args[1] in {"scripts/check_repo.py", "-m"}:
            return subprocess.CompletedProcess(args, 0, "validation passed\n")
        return subprocess.run(args, **kwargs)

    out = tmp_path / "out"
    receipt = pg_refresh_main.refresh("S2", out, work, runner=passing_runner)
    assert receipt["status"] == "PASS"
    assert receipt["old_sha"] == old and receipt["new_sha"] == new
    assert "1 file changed" in receipt["diffstat"]
    before = (out / "code_updates" / "S2_MAIN_REFRESH.json").read_bytes()
    assert pg_refresh_main.refresh("S2", out, work, runner=passing_runner) == receipt
    assert (out / "code_updates" / "S2_MAIN_REFRESH.json").read_bytes() == before
