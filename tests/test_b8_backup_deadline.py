"""An unavailable Drive must not hang the finite server campaign forever."""
import hashlib
import os
from pathlib import Path
import subprocess
import sys
import time

import pytest

from scripts.run_b8_reproduction import DriveBackup, command


def test_command_has_wall_clock_deadline(tmp_path):
    with pytest.raises(subprocess.TimeoutExpired):
        command([sys.executable, "-c", "import time;time.sleep(20)"],
                tmp_path / "log", timeout=0.05)


def test_readback_kills_stalled_process_and_preserves_local_weights(tmp_path, monkeypatch):
    executable = tmp_path / "rclone"
    executable.write_text(f"#!{sys.executable}\nimport time\ntime.sleep(20)\n")
    executable.chmod(0o700)
    monkeypatch.setenv("PATH", str(tmp_path) + os.pathsep + os.environ["PATH"])
    weights = tmp_path / "weights.pt"
    weights.write_bytes(b"retained local weights")
    backup = DriveBackup("KhanhDrive:", "folder", tmp_path / "log", timeout_seconds=0.1)
    start = time.monotonic()
    with pytest.raises(subprocess.TimeoutExpired):
        backup.readback("weights.pt")
    assert time.monotonic() - start < 5
    assert weights.read_bytes() == b"retained local weights"


def test_readback_hashes_actual_stream(tmp_path, monkeypatch):
    executable = tmp_path / "rclone"
    executable.write_text(f"#!{sys.executable}\nimport sys\nsys.stdout.buffer.write(b'payload')\n")
    executable.chmod(0o700)
    monkeypatch.setenv("PATH", str(tmp_path) + os.pathsep + os.environ["PATH"])
    backup = DriveBackup("KhanhDrive:", "folder", tmp_path / "log", timeout_seconds=3)
    assert backup.readback("weights.pt") == (7, hashlib.sha256(b"payload").hexdigest())


@pytest.mark.parametrize("deadline", [0, -1, 3601])
def test_invalid_backup_deadline_fails_closed(tmp_path, deadline):
    with pytest.raises(ValueError, match="deadline"):
        DriveBackup("KhanhDrive:", "folder", tmp_path / "log", timeout_seconds=deadline)
