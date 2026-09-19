from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[4]


def _runner():
    spec = importlib.util.spec_from_file_location("dfc_scientific_runner_test", ROOT / "scripts" / "run_dfc_scientific.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_frozen_grid_config_and_cpu_receipt_helpers():
    runner = _runner()
    grid = json.loads((ROOT / "benchmark_freezes" / "cardiac_benchmark_v3" / "configs" / "resolved_shared_contract.json").read_text())
    manifest = {"schema_version": "shared_benchmark_manifest.v1", "split_policy_version": "self_audit.acdc.patient_split.v1", "shared_grid": grid, "shared_grid_hash": runner.FROZEN_SELF_AUDIT_SHARED_GRID_SHA256}
    runner._require_frozen_grid(manifest)
    with pytest.raises(runner.ArtifactError):
        runner._require_frozen_grid({**manifest, "shared_grid_hash": "not-frozen"})
    with pytest.raises(runner.ArtifactError):
        runner._require_expected_config_hash("not-the-computed-hash", "computed")
    receipt = runner._finish_execution_receipt(runner._start_execution_receipt("cpu"), "cpu")
    assert receipt["cuda_peak_allocated_bytes"] is None
    assert receipt["cuda_peak_reserved_bytes"] is None
    assert receipt["scikit_learn_version"]


def test_scientific_runner_rejects_valid_source_checked_8x8_manifest(tmp_path):
    """The production ``run`` path must reject a self-consistent non-frozen grid."""
    runner = _runner()
    from cardiac_benchmark.manifest import make_fixture_manifest
    from shared_benchmark.manifest import manifest_hash

    image_root = tmp_path / "images"
    image_root.mkdir()
    np.save(image_root / "image_0.npy", np.arange(3 * 8 * 8, dtype=np.float32).reshape(3, 8, 8))
    manifest = make_fixture_manifest(image_root)
    # This remains a source-valid shared scientific manifest.  Its only
    # intended violation is the self-consistent, but non-frozen, 8x8 grid.
    manifest["fixture"] = False
    manifest["scientific"] = True
    manifest["manifest_hash"] = manifest_hash(manifest)
    manifest_path = tmp_path / "scientific_8x8.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    args = runner.build_parser().parse_args([
        "--manifest", str(manifest_path), "--image-root", str(image_root),
        "--output-root", str(tmp_path / "outputs"), "--split", "test",
    ])
    with pytest.raises(runner.ArtifactError, match="frozen Self-Audit 256x256 whole-FOV"):
        runner.run(args)


def test_semantic_failure_exits_nonzero():
    runner = _runner()
    with patch.object(runner, "run", return_value=[{"raw_status": "RAW_COMPLETE", "semantic_status": "FAILED"}]):
        assert runner.main(["--manifest", "m", "--image-root", "i", "--output-root", "o", "--split", "test", "--apply-adapter"]) == 1
