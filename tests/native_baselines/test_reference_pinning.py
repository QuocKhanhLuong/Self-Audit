"""Fresh-clone pinning regression, using local repositories only (no network)."""
import importlib.util
import json
from pathlib import Path
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("pin_native_references", ROOT / "scripts/pin_native_references.py")
pinning = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pinning)

GIT_IDENTITY = ["-c", "user.name=test", "-c", "user.email=test@example.invalid", "-c", "commit.gpgsign=false"]


def git(*args, cwd):
    return subprocess.run(["git", *GIT_IDENTITY, *args], cwd=cwd, check=True,
                          capture_output=True, text=True).stdout.strip()


@pytest.fixture()
def upstream(tmp_path):
    source = tmp_path / "upstream"
    source.mkdir()
    git("init", "-q", cwd=source)
    (source / "model.py").write_text("VALUE = 1\n")
    git("add", ".", cwd=source)
    git("commit", "-q", "-m", "first", cwd=source)
    pinned = git("rev-parse", "HEAD", cwd=source)
    (source / "model.py").write_text("VALUE = 2\n")
    git("commit", "-q", "-am", "second", cwd=source)
    return source, pinned


def test_fresh_clone_pins_non_tip_commit_without_manual_checkout(tmp_path, upstream):
    source, pinned = upstream
    target = tmp_path / "refs" / "fresh"
    assert pinning.pin_reference(target, str(source), pinned) == pinned
    assert (target / "model.py").read_text() == "VALUE = 1\n"
    assert git("status", "--porcelain", cwd=target) == ""
    # Idempotent on an already-pinned clean checkout.
    assert pinning.pin_reference(target, str(source), pinned) == pinned


def test_leftover_no_checkout_clone_is_recovered(tmp_path, upstream):
    source, pinned = upstream
    target = tmp_path / "leftover"
    subprocess.run(["git", "clone", "-q", "--no-checkout", str(source), str(target)], check=True)
    assert git("status", "--porcelain", cwd=target)  # empty index looks "dirty" to status
    assert pinning.pin_reference(target, str(source), pinned) == pinned


def test_edited_reference_is_refused(tmp_path, upstream):
    source, pinned = upstream
    target = tmp_path / "edited"
    pinning.pin_reference(target, str(source), pinned)
    (target / "model.py").write_text("VALUE = 'local edit'\n")
    with pytest.raises(ValueError, match="dirty reference"):
        pinning.pin_reference(target, str(source), pinned)
    assert (target / "model.py").read_text() == "VALUE = 'local edit'\n"


def test_untracked_content_in_unpopulated_clone_is_refused(tmp_path, upstream):
    source, pinned = upstream
    target = tmp_path / "untracked"
    subprocess.run(["git", "clone", "-q", "--no-checkout", str(source), str(target)], check=True)
    (target / "notes.py").write_text("user content\n")
    with pytest.raises(ValueError, match="dirty reference"):
        pinning.pin_reference(target, str(source), pinned)


def test_wrong_origin_is_refused(tmp_path, upstream):
    source, pinned = upstream
    target = tmp_path / "origin"
    pinning.pin_reference(target, str(source), pinned)
    with pytest.raises(ValueError, match="unexpected origin"):
        pinning.pin_reference(target, str(tmp_path / "elsewhere"), pinned)


def test_pin_references_covers_every_declared_reference(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(pinning, "pin_reference", lambda target, url, commit: calls.append((target, url, commit)))

    pins = pinning.pin_references(tmp_path)

    declared = json.loads((ROOT / "configs/native_baselines/references.json").read_text())
    assert pins == declared
    assert calls == [(tmp_path / pinning.TARGET_NAMES[name], pin["url"], pin["commit"])
                     for name, pin in declared.items()]
    assert any(target == tmp_path / "adnet" for target, _, _ in calls)
