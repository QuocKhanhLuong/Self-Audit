from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path


def test_cardiac_freeze_validator_rejects_a_bound_artifact_mutation(tmp_path):
    repo_root = Path(__file__).resolve().parents[2]
    source = repo_root / "benchmark_freezes" / "cardiac_benchmark_v1"
    frozen_copy = tmp_path / "cardiac_benchmark_v1"
    shutil.copytree(source, frozen_copy)
    command = [
        sys.executable,
        str(repo_root / "scripts" / "validate_cardiac_benchmark_freeze.py"),
        "--freeze-dir", str(frozen_copy),
        "--repo-root", str(repo_root),
    ]
    assert subprocess.run(command, capture_output=True, text=True).returncode == 0
    contract = frozen_copy / "configs" / "resolved_shared_contract.json"
    contract.write_bytes(contract.read_bytes() + b"\n")
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode != 0
    assert "freeze hash mismatch" in result.stderr
