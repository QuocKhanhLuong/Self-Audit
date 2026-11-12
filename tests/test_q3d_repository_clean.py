from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest


_CHILD_GUARD = "SELF_AUDIT_Q3D_REPOSITORY_CLEAN_CHILD"
_CHILD_GUARD_VALUE = "1"
_FULL_SUITE_TIMEOUT_SECONDS = 900


def _git(repo: Path, *args: str, env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        cwd=repo,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


def _assert_git_ok(result: subprocess.CompletedProcess[str], action: str) -> None:
    assert result.returncode == 0, (
        f"{action} failed (exit {result.returncode})\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )


def _git_env() -> dict[str, str]:
    env = os.environ.copy()
    # Keep the fresh repository independent of an enclosing worktree, global
    # config, hooks, and any inherited Git index/object directory overrides.
    for name in tuple(env):
        if name.startswith("GIT_"):
            env.pop(name)
    env["GIT_CONFIG_NOSYSTEM"] = "1"
    env["GIT_CONFIG_GLOBAL"] = os.devnull
    return env


def _copy_tracked_working_source(source_root: Path, copy_root: Path) -> None:
    listing = _git(source_root, "ls-files", "-z", env=_git_env())
    _assert_git_ok(listing, "listing tracked source paths")
    relative_paths = {
        Path(os.fsdecode(raw_path))
        for raw_path in listing.stdout.split("\0")
        if raw_path
    }
    # This file is deliberately new and may not be in the source index yet.
    relative_paths.add(Path(__file__).resolve().relative_to(source_root))

    for relative_path in sorted(relative_paths):
        source = source_root / relative_path
        if not source.exists() and not source.is_symlink():
            # A tracked deletion in the active working tree stays deleted in
            # the snapshot; untracked files are never enumerated or copied.
            continue
        destination = copy_root / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        if source.is_symlink():
            destination.symlink_to(os.readlink(source))
        elif source.is_file():
            shutil.copy2(source, destination)


def test_full_pytest_suite_leaves_tracked_working_source_clean(tmp_path: Path) -> None:
    if os.environ.get(_CHILD_GUARD) == _CHILD_GUARD_VALUE:
        pytest.skip("the isolated child suite skips only its own clean-copy launcher")

    source_root = Path(__file__).resolve().parents[1]
    copy_root = tmp_path / "tracked-working-source"
    copy_root.mkdir()
    _copy_tracked_working_source(source_root, copy_root)

    child_temp = tmp_path / "child-temp"
    child_temp.mkdir()
    child_env = _git_env()
    child_env[_CHILD_GUARD] = _CHILD_GUARD_VALUE
    child_env["PYTHONPATH"] = str(copy_root)
    child_env["PYTHONDONTWRITEBYTECODE"] = "1"
    child_env["PYTEST_ADDOPTS"] = ""
    child_env["TMPDIR"] = str(child_temp)
    child_env["TMP"] = str(child_temp)
    child_env["TEMP"] = str(child_temp)
    child_env["COVERAGE_FILE"] = str(child_temp / ".coverage")
    child_env["XDG_CACHE_HOME"] = str(child_temp / "xdg-cache")
    child_env["MPLCONFIGDIR"] = str(child_temp / "matplotlib")
    child_env["TORCH_HOME"] = str(child_temp / "torch")

    init = _git(copy_root, "init", "--quiet", "--template=/dev/null", env=child_env)
    _assert_git_ok(init, "initializing isolated repository")
    for key, value in (
        ("user.name", "Self-Audit Clean Suite Test"),
        ("user.email", "self-audit-clean-suite@example.invalid"),
        ("core.hooksPath", "/dev/null"),
        ("commit.gpgsign", "false"),
    ):
        _assert_git_ok(
            _git(copy_root, "config", "--local", key, value, env=child_env),
            f"configuring isolated repository {key}",
        )
    _assert_git_ok(
        _git(copy_root, "add", "--all", "-f", "--", ".", env=child_env),
        "staging the tracked working-source snapshot",
    )
    _assert_git_ok(
        _git(copy_root, "commit", "--quiet", "-m", "isolated tracked source snapshot", env=child_env),
        "committing the tracked working-source snapshot",
    )
    initial_status = _git(copy_root, "status", "--porcelain", env=child_env)
    _assert_git_ok(initial_status, "checking initial repository status")
    assert initial_status.stdout == "", f"initial snapshot is not clean:\n{initial_status.stdout}"

    # cwd and PYTHONPATH both point at the copy, so project imports resolve to
    # the isolated snapshot. External ResEnc source paths remain inherited.
    pytest_command = [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "-p",
        "no:cacheprovider",
        "--basetemp",
        str(child_temp / "pytest-tmp"),
    ]
    try:
        child = subprocess.run(
            pytest_command,
            cwd=copy_root,
            env=child_env,
            text=True,
            capture_output=True,
            check=False,
            timeout=_FULL_SUITE_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout.decode(errors="replace") if isinstance(exc.stdout, bytes) else (exc.stdout or "")
        stderr = exc.stderr.decode(errors="replace") if isinstance(exc.stderr, bytes) else (exc.stderr or "")
        status = _git(copy_root, "status", "--porcelain", env=child_env)
        pytest.fail(
            f"isolated full pytest suite exceeded {_FULL_SUITE_TIMEOUT_SECONDS}s\n"
            f"git status --porcelain:\n{status.stdout}\n"
            f"stdout tail:\n{stdout[-12000:]}\nstderr tail:\n{stderr[-12000:]}"
        )

    status = _git(copy_root, "status", "--porcelain", env=child_env)
    _assert_git_ok(status, "checking repository status after the full suite")
    details = (
        f"command: {' '.join(pytest_command)}\n"
        f"child return code: {child.returncode}\n"
        f"git status --porcelain:\n{status.stdout}\n"
        f"pytest stdout tail:\n{child.stdout[-12000:]}\n"
        f"pytest stderr tail:\n{child.stderr[-12000:]}"
    )
    assert child.returncode == 0, f"isolated full pytest suite failed:\n{details}"
    assert status.stdout == "", f"pytest left the isolated working tree dirty:\n{details}"
