from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path


def test_v6_validator_fail_closes_after_bound_baseline_source_changes(tmp_path):
    repo_root = Path(__file__).resolve().parents[2]
    source = repo_root / "benchmark_freezes" / "cardiac_benchmark_v6"
    frozen_copy = tmp_path / "cardiac_benchmark_v6"
    shutil.copytree(source, frozen_copy)
    command = [
        sys.executable,
        str(repo_root / "scripts" / "validate_cardiac_benchmark_v6_freeze.py"),
        "--freeze-dir", str(frozen_copy),
        "--repo-root", str(repo_root),
    ]
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
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode != 0
    assert "freeze hash mismatch" in result.stderr
