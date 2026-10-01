from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

<<<<<<< HEAD
def test_v6_validator_fail_closes_after_bound_baseline_source_changes(tmp_path):
=======

@pytest.mark.parametrize(
    ("freeze_name", "validator_name"),
    [
        ("cardiac_benchmark_v6", "validate_cardiac_benchmark_v6_freeze.py"),
        ("cardiac_benchmark_v7", "validate_cardiac_benchmark_v7_freeze.py"),
    ],
)
def test_cardiac_freeze_validator_rejects_a_bound_artifact_mutation(tmp_path, freeze_name, validator_name):
>>>>>>> a107b734fee92b9a5b578bdef33a115581dabfcb
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
<<<<<<< HEAD
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode != 0
    assert "repository hash mismatch" in result.stderr


def test_historical_224_freeze_validator_rejects_a_bound_artifact_mutation(tmp_path):
    repo_root = Path(__file__).resolve().parents[2]
    source = repo_root / "benchmark_freezes" / "cardiac_benchmark_v10_historical_224"
    frozen_copy = tmp_path / "cardiac_benchmark_v10_historical_224"
    shutil.copytree(source, frozen_copy)
    command = [
        sys.executable,
        str(repo_root / "scripts" / "validate_cardiac_benchmark_v10_historical_224_freeze.py"),
        "--freeze-dir", str(frozen_copy),
        "--repo-root", str(repo_root),
    ]
    assert subprocess.run(command, capture_output=True, text=True).returncode == 0
    manifest = frozen_copy / "data" / "acdc_shared_manifest.json"
    manifest.write_bytes(manifest.read_bytes() + b"\n")
=======
    initial = subprocess.run(command, capture_output=True, text=True)
    if initial.returncode != 0:
        # Historical freezes legitimately drift once the repo evolves without a
        # new published freeze. In that state the validator must still fail
        # closed with a repository-bound mismatch rather than silently pass.
        assert "repository hash mismatch" in initial.stderr
        return
    contract = frozen_copy / "configs" / "resolved_shared_contract.json"
    contract.write_bytes(contract.read_bytes() + b"\n")
>>>>>>> a107b734fee92b9a5b578bdef33a115581dabfcb
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode != 0
    assert "freeze hash mismatch" in result.stderr
