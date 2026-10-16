"""Exercise the documented external environment without pytest path injection."""

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]


def environment():
    return {
        **os.environ,
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONPATH": os.pathsep.join((str(ROOT), str(REPO / "src"))),
    }


def test_documented_standalone_imports(tmp_path):
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "from integration import bank, artifacts, geometry, steps",
        ],
        cwd=tmp_path,
        env=environment(),
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_missing_approval_preflight_stops_before_creating_output(tmp_path):
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "integration.cli",
            "--recipe",
            str(ROOT / "recipe.json"),
            "--approval",
            str(tmp_path / "absent.json"),
            "--expected-approval-sha256",
            "0" * 64,
            "--out",
            str(tmp_path / "preflight"),
        ],
        cwd=tmp_path,
        env=environment(),
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2
    assert "BLOCKED_PRETRAINED" in result.stderr
    assert not (tmp_path / "preflight").exists()
