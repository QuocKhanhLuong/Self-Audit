from __future__ import annotations

import json
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


HISTORICAL_224_FREEZES = [
    ("cardiac_benchmark_v10_historical_224", "validate_cardiac_benchmark_v10_historical_224_freeze.py"),
    ("cardiac_benchmark_v11_historical_224", "validate_cardiac_benchmark_v11_historical_224_freeze.py"),
    ("cardiac_benchmark_v12_historical_224", "validate_cardiac_benchmark_v12_historical_224_freeze.py"),
]
ACTIVE_HISTORICAL_224 = ("cardiac_benchmark_v12_historical_224", "validate_cardiac_benchmark_v12_historical_224_freeze.py")
SUPERSEDED_HISTORICAL_224 = [
    # (label, freeze, validator, immutable source snapshot, freeze ID)
    ("v10", "cardiac_benchmark_v10_historical_224", "validate_cardiac_benchmark_v10_historical_224_freeze.py",
     "8937fe7248a6d4f5bb10530fa0b78817af6a40f8", "cardiac-benchmark-v10-historical-224-7d94c71760a22174"),
    ("v11", "cardiac_benchmark_v11_historical_224", "validate_cardiac_benchmark_v11_historical_224_freeze.py",
     "35ac4bd375a4a0fdc60a2cc9a5f9aece17b72d0b", "cardiac-benchmark-v11-historical-224-29f2c31fb0c2261d"),
]


def _historical_command(repo_root: Path, validator_name: str, freeze_dir: Path, *extra: str) -> list[str]:
    return [
        sys.executable,
        str(repo_root / "scripts" / validator_name),
        "--freeze-dir", str(freeze_dir),
        "--repo-root", str(repo_root),
        *extra,
    ]


@pytest.mark.parametrize(("freeze_name", "validator_name"), HISTORICAL_224_FREEZES)
@pytest.mark.parametrize(
    ("mutated", "message"),
    [
        ("data/acdc_shared_manifest.json", "freeze hash mismatch"),
        ("configs/resolved_shared_contract.json", "freeze hash mismatch"),
        ("FREEZE_MANIFEST.json", "scientific payload hash mismatch"),
    ],
)
def test_historical_224_freeze_validator_rejects_a_bound_artifact_mutation(
    tmp_path, freeze_name, validator_name, mutated, message,
):
    repo_root = Path(__file__).resolve().parents[2]
    frozen_copy = tmp_path / freeze_name
    shutil.copytree(repo_root / "benchmark_freezes" / freeze_name, frozen_copy)
    command = _historical_command(repo_root, validator_name, frozen_copy)
    initial = subprocess.run(command, capture_output=True, text=True)
    assert initial.returncode == 0, initial.stderr
    target = frozen_copy / mutated
    if mutated == "FREEZE_MANIFEST.json":
        document = json.loads(target.read_text())
        document["scientific_payload"]["scientific_readiness"] = "TAMPERED"
        target.write_text(json.dumps(document))
    else:
        target.write_bytes(target.read_bytes() + b"\n")
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode != 0
    assert message in result.stderr


def test_v12_is_the_active_freeze_for_the_current_checkout():
    repo_root = Path(__file__).resolve().parents[2]
    freeze_name, validator = ACTIVE_HISTORICAL_224
    result = subprocess.run(
        _historical_command(repo_root, validator, repo_root / "benchmark_freezes" / freeze_name),
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["freeze_id"].startswith("cardiac-benchmark-v12-historical-224-")
    for runner in ("run_cuts_scientific.py", "run_dfc_scientific.py"):
        text = (repo_root / "scripts" / runner).read_text()
        assert f'"{freeze_name}" / "configs" / "adapter_v2_spec.json"' in text


def test_active_freeze_rejects_a_mutated_bound_repository_source(tmp_path):
    repo_root = Path(__file__).resolve().parents[2]
    freeze_name, validator = ACTIVE_HISTORICAL_224
    freeze = repo_root / "benchmark_freezes" / freeze_name
    payload = json.loads((freeze / "FREEZE_MANIFEST.json").read_text())["scientific_payload"]
    repo_copy = tmp_path / "repo"
    for binding in payload["bound_repository_files"]:
        destination = repo_copy / binding["path"]
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(repo_root / binding["path"], destination)
    bound = repo_copy / "src" / "shared_benchmark" / "spatial.py"
    bound.write_bytes(bound.read_bytes() + b"# tampered\n")
    result = subprocess.run(
        [sys.executable, str(repo_root / "scripts" / validator),
         "--freeze-dir", str(freeze), "--repo-root", str(repo_copy)],
        capture_output=True, text=True,
    )
    assert result.returncode != 0
    assert "repository hash mismatch: src/shared_benchmark/spatial.py" in result.stderr


@pytest.mark.parametrize(("label", "freeze_name", "validator", "snapshot", "freeze_id"), SUPERSEDED_HISTORICAL_224)
def test_superseded_freeze_is_bound_to_its_source_snapshot_not_the_current_checkout(
    label, freeze_name, validator, snapshot, freeze_id,
):
    repo_root = Path(__file__).resolve().parents[2]
    freeze = repo_root / "benchmark_freezes" / freeze_name
    result = subprocess.run(_historical_command(repo_root, validator, freeze), capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert report["freeze_id"] == freeze_id
    assert report["repository_binding_mode"] == "source_snapshot"
    assert report["repository_source_snapshot"] == snapshot
    assert report["validates_current_checkout_sources"] is False
    assert report["status"] == "IMMUTABLE_HISTORICAL_SUPERSEDED"
    # Newer source is never silently accepted as the superseded freeze's source.
    worktree = subprocess.run(
        _historical_command(repo_root, validator, freeze, "--against-worktree"), capture_output=True, text=True,
    )
    assert worktree.returncode != 0
    assert "repository hash mismatch" in worktree.stderr


@pytest.mark.parametrize(("label", "freeze_name", "validator", "snapshot", "freeze_id"), SUPERSEDED_HISTORICAL_224)
def test_superseded_snapshot_validation_fails_closed_without_the_snapshot(
    tmp_path, label, freeze_name, validator, snapshot, freeze_id,
):
    repo_root = Path(__file__).resolve().parents[2]
    not_a_repository = tmp_path / "no_git"
    not_a_repository.mkdir()
    result = subprocess.run(
        [sys.executable, str(repo_root / "scripts" / validator),
         "--freeze-dir", str(repo_root / "benchmark_freezes" / freeze_name), "--repo-root", str(not_a_repository)],
        capture_output=True, text=True,
    )
    assert result.returncode != 0
    assert f"{label} source snapshot unavailable" in result.stderr
