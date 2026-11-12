"""Command entrypoints must work without an installed package or PYTHONPATH."""

import os
from pathlib import Path
import subprocess
import sys

import pytest


@pytest.mark.parametrize("script", [
    "q3d_export.py", "q3d_infer.py", "q3d_profile.py", "q3d_campaign.py",
    "q3d_preflight.py", "q3d_diagnostics.py", "q3d_physio_cache.py", "smoke_q3d.py",
])
def test_direct_cli_without_pythonpath(script, tmp_path):
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    path = Path(__file__).resolve().parents[1] / "scripts" / script
    result = subprocess.run(
        [sys.executable, str(path), "--help"], cwd=tmp_path, env=environment,
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout
