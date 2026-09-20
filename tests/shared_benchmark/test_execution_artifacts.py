from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

from shared_benchmark.artifacts import (
    ArtifactError,
    FAILED,
    RAW_COMPLETE,
    SEMANTIC_ARTIFACT_STAGE,
    GeneratedSample,
    artifact_directory,
    atomic_write_bytes,
    raw_artifact_is_valid,
    run_adapter_after_raw,
    run_generation,
    seal_raw_partition,
    seal_semantic_partition,
    select_manifest_records,
    validate_scientific_execution,
    verify_raw_partition,
    verify_semantic_partition,
)
from shared_benchmark.provenance import sha256_json
from shared_benchmark.semantic_contract import FROZEN_ADAPTER_SPEC_SHA256, adapter_metadata_payload, canonical_metadata_hash
from shared_benchmark.manifest import build_shared_manifest
from shared_benchmark.spatial import build_grid_spec

from helpers import discovered_projection, write_image


ROOT = Path(__file__).resolve().parents[2]
SPEC = json.loads((ROOT / "benchmark_freezes/cardiac_benchmark_v6/configs/adapter_v2_spec.json").read_text(encoding="utf-8"))


@pytest.fixture()
def fixture_manifest(tmp_path: Path):
    write_image(tmp_path, "image.npy", shape=(3, 8, 8), seed=7)
    _, manifest = discovered_projection(tmp_path, target_hw=(8, 8))
    return tmp_path, manifest


def _seal(tmp_path: Path, manifest: dict, record: dict, *, config: str = "cfg", repo: dict | None = None):
    return seal_raw_partition(
        tmp_path / "out", np.arange(64, dtype=np.int32).reshape(8, 8),
        record=record, baseline_name="TEST", baseline_mode="2d",
        manifest_hash=manifest["manifest_hash"], shared_grid_hash=manifest["shared_grid_hash"],
        repository=repo or {"repository_commit_sha": "commit", "working_tree_sha256": "tree"},
        baseline_config_hash=config, seed=42,
        baseline_metadata={"required_provenance": "fixture"},
    )


def _cuts_provenance(checkpoint: str = "checkpoint-a") -> dict:
    return {
        "checkpoint_sha256": checkpoint,
        "checkpoint_training_provenance": {"checkpoint_selection_policy": "final_epoch"},
        "cuts_mode": "CUTS-2D", "phate_configuration": {"n_components": 3},
        "kmeans_configuration": {"n_clusters": 10}, "primary_k": 10, "clustering_seed": 1,
    }


def _dfc_provenance() -> dict:
    return {
        "sample_seed": 42, "seed_derivation": {"seed_derivation_version": "fixture"},
        "update_count": 3, "iteration_count": 3, "final_active_count": 4,
        "minLabels": 3, "maxIter": 1000, "optimizer": {"name": "SGD"},
        "input_mode": "central_slice_2d", "final_forward_semantics": "fixture",
        "fresh_model_optimizer_bn_state_per_sample": True,
    }


def test_raw_hash_stability_and_resume_identity(fixture_manifest):
    root, manifest = fixture_manifest
    record = manifest["records"][0]
    first = _seal(root, manifest, record)
    metadata_hash = first.metadata["scientific_payload_hash"]
    second = verify_raw_partition(first.directory, expected={
        "sample_id": record["sample_id"], "shared_manifest_hash": manifest["manifest_hash"],
        "shared_grid_hash": manifest["shared_grid_hash"], "baseline_config_hash": "cfg",
        "repository_identity": {"repository_commit_sha": "commit", "working_tree_sha256": "tree"}, "seed": 42,
    })
    assert second.metadata["scientific_payload_hash"] == metadata_hash
    assert second.metadata["completion_status"] == RAW_COMPLETE
    assert raw_artifact_is_valid(first.directory)


def test_valid_raw_resume_skips_generator(fixture_manifest):
    root, manifest = fixture_manifest
    record = manifest["records"][0]
    _seal(root, manifest, record)

    def must_not_run(_record):
        raise AssertionError("valid RAW_COMPLETE artifact was recomputed")

    rows = run_generation(
        [record], output_root=root / "out", baseline_name="TEST", baseline_mode="2d",
        manifest_hash=manifest["manifest_hash"], shared_grid_hash=manifest["shared_grid_hash"],
        repository={"repository_commit_sha": "commit", "working_tree_sha256": "tree"},
        baseline_config_hash="cfg", seed_for_record=lambda _r: 42, generate=must_not_run,
        code_identity_value={},
    )
    assert rows[0]["raw_status"] == "SKIPPED_RAW_COMPLETE"


def test_atomic_interruption_never_creates_complete_artifact(tmp_path: Path):
    target = tmp_path / "atomic.bin"

    def interrupt(_temporary: Path):
        raise RuntimeError("simulated interruption")

    with pytest.raises(RuntimeError):
        atomic_write_bytes(target, b"partial", before_replace=interrupt)
    assert not target.exists()


def test_corruption_and_identity_changes_invalidate_cache(fixture_manifest):
    root, manifest = fixture_manifest
    record = manifest["records"][0]
    raw = _seal(root, manifest, record)
    raw_path = raw.directory / raw.metadata["raw_partition_path"]
    raw_path.write_bytes(raw_path.read_bytes() + b"x")
    assert not raw_artifact_is_valid(raw.directory)
    with pytest.raises(ArtifactError):
        verify_raw_partition(raw.directory)
    # A new valid seal is valid only for the exact bound config/repository.
    raw = _seal(root, manifest, record, config="new-config")
    assert not raw_artifact_is_valid(raw.directory, expected={"baseline_config_hash": "old-config"})
    assert not raw_artifact_is_valid(raw.directory, expected={"repository_identity": {"repository_commit_sha": "other", "working_tree_sha256": "tree"}})


def test_adapter_handoff_requires_raw_complete_and_semantic_binds_raw(fixture_manifest):
    root, manifest = fixture_manifest
    record = manifest["records"][0]
    raw = _seal(root, manifest, record)
    handoff = run_adapter_after_raw(
        raw, semantic_root=root / "semantic", record=record,
        central_image=np.zeros((8, 8), dtype=np.float32), adapter_spec=SPEC,
        baseline_name="TEST", baseline_mode="2d",
    )
    assert handoff.directory.parts[-4] == SEMANTIC_ARTIFACT_STAGE
    image = np.zeros((8, 8), dtype=np.float32)
    checked = verify_semantic_partition(handoff.directory, raw_artifact=raw, record=record, central_image=image, adapter_spec=SPEC)
    assert checked.metadata["raw_partition_sha256"] == raw.metadata["raw_partition_sha256"]
    raw_path = raw.directory / raw.metadata["raw_partition_path"]
    changed = np.load(raw_path, allow_pickle=False)
    changed[0, 0] += 1
    np.save(raw_path, changed, allow_pickle=False)
    with pytest.raises(ArtifactError):
        verify_semantic_partition(handoff.directory, raw_artifact=raw, record=record, central_image=image, adapter_spec=SPEC)

    pending = type(raw)(directory=raw.directory, partition=raw.partition, metadata={**raw.metadata, "completion_status": "RUNNING"})
    with pytest.raises(ArtifactError):
        run_adapter_after_raw(
            pending, semantic_root=root / "semantic2", record=record,
            central_image=np.zeros((8, 8), dtype=np.float32), adapter_spec=SPEC,
            baseline_name="TEST", baseline_mode="2d",
        )


@pytest.mark.parametrize("field", ["assignment_reason", "central_image_sha256", "coverage"])
def test_semantic_adapter_metadata_tamper_is_rejected_even_after_rehash(fixture_manifest, field):
    """A forged outer/inner hash cannot replace adapter recomputation."""
    root, manifest = fixture_manifest
    record = manifest["records"][0]
    image = np.zeros((8, 8), dtype=np.float32)
    raw = _seal(root, manifest, record)
    handoff = run_adapter_after_raw(
        raw, semantic_root=root / "semantic", record=record, central_image=image,
        adapter_spec=SPEC, baseline_name="TEST", baseline_mode="2d",
    )
    metadata_path = handoff.directory / "metadata.json"
    state_path = handoff.directory / "state.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    adapter_metadata = metadata["adapter_metadata"]
    if field == "assignment_reason":
        adapter_metadata["assignments"][0]["reason"] = "forged_assignment_reason"
    elif field == "central_image_sha256":
        adapter_metadata["central_image_sha256"] = "0" * 64
    else:
        adapter_metadata["coverage"] = 0.123456
    # Model an attacker who knows both serialization formats and recomputes
    # every seal that was available in the artifact itself.
    adapter_metadata["metadata_sha256"] = canonical_metadata_hash(adapter_metadata_payload(adapter_metadata))
    for key in (
        "assignments", "assignment_reasons", "void_reasons", "role_reasons", "unresolved_reasons",
        "component_graph_digest", "central_image_sha256", "scientific_result_sha256",
    ):
        metadata[key] = adapter_metadata[key]
    metadata["adapter_metadata_sha256"] = adapter_metadata["metadata_sha256"]
    metadata["coverage"] = adapter_metadata["coverage"]
    metadata["scientific_payload_hash"] = sha256_json({key: value for key, value in metadata.items() if key != "scientific_payload_hash"})
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state["artifact_scientific_payload_hash"] = metadata["scientific_payload_hash"]
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    state_path.write_text(json.dumps(state), encoding="utf-8")
    with pytest.raises(ArtifactError):
        verify_semantic_partition(
            handoff.directory, raw_artifact=raw, record=record, central_image=image, adapter_spec=SPEC,
        )


def test_adapter_implementation_identity_includes_real_dependencies(tmp_path: Path):
    from shared_benchmark.artifacts import _adapter_implementation_files, _adapter_implementation_hash

    paths = _adapter_implementation_files(ROOT)
    relative = {path.relative_to(ROOT).as_posix() for path in paths}
    assert {
        "src/shared_benchmark/firewall.py", "src/shared_benchmark/spatial.py",
        "src/shared_benchmark/provenance.py", "src/self_audit_maskfree/data/firewall.py",
    }.issubset(relative)
    for source in paths:
        destination = tmp_path / source.relative_to(ROOT)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    original = _adapter_implementation_hash(tmp_path)
    dependency = tmp_path / "src/shared_benchmark/spatial.py"
    dependency.write_text(dependency.read_text(encoding="utf-8") + "\n# test identity mutation\n", encoding="utf-8")
    assert _adapter_implementation_hash(tmp_path) != original


def test_semantic_failure_does_not_demote_sealed_raw(fixture_manifest):
    root, manifest = fixture_manifest
    record = manifest["records"][0]

    def generate(_record):
        return GeneratedSample(np.zeros((8, 8), dtype=np.int32), np.zeros((8, 8), dtype=np.float32), {})

    rows = run_generation(
        [record], output_root=root / "handoff", baseline_name="TEST", baseline_mode="2d",
        manifest_hash=manifest["manifest_hash"], shared_grid_hash=manifest["shared_grid_hash"],
        repository={"repository_commit_sha": "c", "working_tree_sha256": "t"}, baseline_config_hash="cfg",
        seed_for_record=lambda _r: 42, generate=generate, semantic_root=root / "handoff",
        adapter_spec={"not": "the frozen spec"},
    )
    assert rows[0]["raw_status"] == RAW_COMPLETE and rows[0]["semantic_status"] == FAILED
    raw_dir = artifact_directory(root / "handoff", baseline_name="TEST", baseline_mode="2d", sample_id=record["sample_id"])
    assert verify_raw_partition(raw_dir).metadata["completion_status"] == RAW_COMPLETE


def test_cut_and_dfc_provenance_fields_are_preserved(fixture_manifest):
    root, manifest = fixture_manifest
    record = manifest["records"][0]
    raw = seal_raw_partition(
        root / "out", np.zeros((8, 8), dtype=np.int32), record=record,
        baseline_name="CUTS", baseline_mode="CUTS-2D",
        manifest_hash=manifest["manifest_hash"], shared_grid_hash=manifest["shared_grid_hash"],
        repository={"repository_commit_sha": "c", "working_tree_sha256": "t"},
        baseline_config_hash="cuts-config", seed=42,
        baseline_metadata={"checkpoint_sha256": "ckpt", "checkpoint_training_provenance": {"checkpoint_selection_policy": "final_epoch"}, "cuts_mode": "CUTS-2D", "phate_configuration": {"n_components": 3}, "kmeans_configuration": {"n_clusters": 10}, "primary_k": 10, "clustering_seed": 1},
    )
    assert raw.metadata["baseline_provenance"]["checkpoint_sha256"] == "ckpt"
    raw_dfc = seal_raw_partition(
        root / "out", np.zeros((8, 8), dtype=np.int32), record={**record, "sample_id": record["sample_id"] + "-dfc"},
        baseline_name="DFC", baseline_mode="DFC-Direct-2D-Default-MinL3",
        manifest_hash=manifest["manifest_hash"], shared_grid_hash=manifest["shared_grid_hash"],
        repository={"repository_commit_sha": "c", "working_tree_sha256": "t"}, baseline_config_hash="dfc-config", seed=123,
        baseline_metadata={"sample_seed": 123, "seed_derivation": {"seed_derivation_version": "fixture"}, "update_count": 4, "iteration_count": 4, "final_active_count": 3, "minLabels": 3, "maxIter": 1000, "optimizer": {"name": "SGD"}, "input_mode": "central_slice_2d", "final_forward_semantics": "fixture", "fresh_model_optimizer_bn_state_per_sample": True},
    )
    assert raw_dfc.metadata["baseline_provenance"]["sample_seed"] == 123


def test_cuts_checkpoint_identity_invalidates_resume_and_receipt_is_operational(fixture_manifest):
    root, manifest = fixture_manifest
    record = manifest["records"][0]
    calls: list[str] = []
    checkpoint = ["checkpoint-a"]

    def generate(_record):
        calls.append("generated")
        return GeneratedSample(np.zeros((8, 8), dtype=np.int32), np.zeros((8, 8), dtype=np.float32), _cuts_provenance(checkpoint[0]), {"python_version": "fixture", "cuda_peak_allocated_bytes": None, "cuda_peak_reserved_bytes": None})

    common = dict(
        output_root=root / "cuts", baseline_name="CUTS", baseline_mode="CUTS-2D",
        manifest_hash=manifest["manifest_hash"], shared_grid_hash=manifest["shared_grid_hash"],
        repository={"repository_commit_sha": "c", "working_tree_sha256": "t"}, baseline_config_hash="cfg",
        seed_for_record=lambda _record: 42, generate=generate,
    )
    first = run_generation([record], extra_raw_identity_for_record=lambda _r: {"checkpoint_sha256": "checkpoint-a"}, **common)
    checkpoint[0] = "checkpoint-b"
    second = run_generation([record], extra_raw_identity_for_record=lambda _r: {"checkpoint_sha256": "checkpoint-b"}, **common)
    assert first[0]["raw_status"] == "RAW_COMPLETE"
    assert second[0]["raw_status"] == "RAW_COMPLETE"
    assert calls == ["generated", "generated"]
    assert second[0]["raw_artifact"].metadata["checkpoint_sha256"] == "checkpoint-b"
    receipt = json.loads((second[0]["raw_artifact"].directory / "execution_receipt.json").read_text())
    assert receipt["raw_generation_elapsed_seconds"] is not None
    assert receipt["adapter_elapsed_seconds"] is None
    assert receipt["total_elapsed_seconds"] >= receipt["raw_generation_elapsed_seconds"]
    assert receipt["environment"]["cuda_peak_allocated_bytes"] is None


@pytest.mark.parametrize(
    ("baseline_name", "baseline_mode", "metadata"),
    [("CUTS", "CUTS-2D", {}), ("DFC", "DFC-Direct-2D-Default-MinL3", {})],
)
def test_production_baselines_reject_missing_required_provenance(fixture_manifest, baseline_name, baseline_mode, metadata):
    root, manifest = fixture_manifest
    with pytest.raises(ArtifactError):
        seal_raw_partition(
            root / "out", np.zeros((8, 8), dtype=np.int32), record=manifest["records"][0],
            baseline_name=baseline_name, baseline_mode=baseline_mode,
            manifest_hash=manifest["manifest_hash"], shared_grid_hash=manifest["shared_grid_hash"],
            repository={"repository_commit_sha": "c", "working_tree_sha256": "t"},
            baseline_config_hash="cfg", seed=42, baseline_metadata=metadata,
        )


def test_dfc_execution_receipt_persists_environment(fixture_manifest):
    root, manifest = fixture_manifest
    record = manifest["records"][0]

    def generate(_record):
        return GeneratedSample(
            np.zeros((8, 8), dtype=np.int32), np.zeros((8, 8), dtype=np.float32), _dfc_provenance(),
            {"python_version": "fixture", "pytorch_version": "fixture", "numpy_version": "fixture",
             "cuda_available": False, "device_name": None, "cuda_peak_allocated_bytes": None,
             "cuda_peak_reserved_bytes": None},
        )

    row = run_generation(
        [record], output_root=root / "dfc", baseline_name="DFC", baseline_mode="DFC-Direct-2D-Default-MinL3",
        manifest_hash=manifest["manifest_hash"], shared_grid_hash=manifest["shared_grid_hash"],
        repository={"repository_commit_sha": "c", "working_tree_sha256": "t"}, baseline_config_hash="cfg",
        seed_for_record=lambda _record: 42, generate=generate,
    )[0]
    receipt = json.loads((row["raw_artifact"].directory / "execution_receipt.json").read_text())
    assert receipt["environment"]["pytorch_version"] == "fixture"
    assert receipt["environment"]["cuda_peak_reserved_bytes"] is None


def test_deterministic_limit_selection_and_failure_receipt(fixture_manifest):
    root, manifest = fixture_manifest
    records = select_manifest_records(manifest, split="train", limit=2)
    assert [row["sample_id"] for row in records] == sorted(row["sample_id"] for row in records)[:2]
    calls: list[str] = []

    def fail(record):
        calls.append(record["sample_id"])
        raise RuntimeError("fixture failure")

    results = run_generation(
        records[:1], output_root=root / "failed", baseline_name="TEST", baseline_mode="2d",
        manifest_hash=manifest["manifest_hash"], shared_grid_hash=manifest["shared_grid_hash"],
        repository={"repository_commit_sha": "c", "working_tree_sha256": "t"}, baseline_config_hash="cfg", seed_for_record=lambda _r: 42,
        generate=fail,
    )
    assert results[0]["raw_status"] == FAILED and calls
    state = json.loads((artifact_directory(root / "failed", baseline_name="TEST", baseline_mode="2d", sample_id=records[0]["sample_id"]) / "state.json").read_text())
    assert state["stage"] == FAILED and state["failure"]["exception_type"] == "RuntimeError"


def test_scientific_execution_rejects_fixture_and_source_mismatch(fixture_manifest):
    root, manifest = fixture_manifest
    manifest_path = root / "fixture_manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ArtifactError):
        validate_scientific_execution(manifest_path, root)

    upstream, _ = discovered_projection(root, target_hw=(8, 8))
    scientific = build_shared_manifest(
        upstream, build_grid_spec((8, 8), config_provenance={"source": "test"}),
        fixture=False, scientific=True, local_source_root=root,
    )
    scientific_path = root / "scientific_manifest.json"
    scientific_path.write_text(json.dumps(scientific), encoding="utf-8")
    assert validate_scientific_execution(scientific_path, root)["scientific"] is True
    image = root / "image.npy"
    image.write_bytes(image.read_bytes() + b"changed")
    with pytest.raises(ArtifactError):
        validate_scientific_execution(scientific_path, root)


def test_gt_like_path_and_field_are_rejected(fixture_manifest):
    root, manifest = fixture_manifest
    record = dict(manifest["records"][0])
    record["source"] = dict(record["source"])
    record["source"]["locator"] = "masks/reference.npy"
    with pytest.raises(Exception):
        _seal(root, manifest, record)
