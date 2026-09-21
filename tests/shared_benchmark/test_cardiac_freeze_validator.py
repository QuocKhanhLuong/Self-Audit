from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.parametrize(
    ("freeze_name", "validator_name"),
    [
        ("cardiac_benchmark_v6", "validate_cardiac_benchmark_v6_freeze.py"),
        ("cardiac_benchmark_v7", "validate_cardiac_benchmark_v7_freeze.py"),
    ],
)
def test_cardiac_freeze_validator_rejects_a_bound_artifact_mutation(tmp_path, freeze_name, validator_name):
    repo_root = Path(__file__).resolve().parents[2]
    source = repo_root / "benchmark_freezes" / freeze_name
    frozen_copy = tmp_path / freeze_name
    shutil.copytree(source, frozen_copy)
    command = [
        sys.executable,
        str(repo_root / "scripts" / validator_name),
        "--freeze-dir", str(frozen_copy),
        "--repo-root", str(repo_root),
    ]
    initial = subprocess.run(command, capture_output=True, text=True)
    if initial.returncode != 0:
        # Historical freezes legitimately drift once the repo evolves without a
        # new published freeze. In that state the validator must still fail
        # closed with a repository-bound mismatch rather than silently pass.
        assert "repository hash mismatch" in initial.stderr
        return
    contract = frozen_copy / "configs" / "resolved_shared_contract.json"
    contract.write_bytes(contract.read_bytes() + b"\n")
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode != 0
    assert "freeze hash mismatch" in result.stderr
