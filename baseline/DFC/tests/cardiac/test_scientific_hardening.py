from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from unittest.mock import patch

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
    grid = json.loads((ROOT / "benchmark_freezes" / "cardiac_benchmark_v1" / "configs" / "resolved_shared_contract.json").read_text())
    manifest = {"schema_version": "shared_benchmark_manifest.v1", "shared_grid": grid, "shared_grid_hash": runner.FROZEN_SHARED_GRID_SHA256}
    runner._require_frozen_grid(manifest)
    with pytest.raises(runner.ArtifactError):
        runner._require_frozen_grid({**manifest, "shared_grid_hash": "not-frozen"})
    with pytest.raises(runner.ArtifactError):
        runner._require_expected_config_hash("not-the-computed-hash", "computed")
    receipt = runner._finish_execution_receipt(runner._start_execution_receipt("cpu"), "cpu")
    assert receipt["cuda_peak_allocated_bytes"] is None
    assert receipt["cuda_peak_reserved_bytes"] is None
    assert receipt["scikit_learn_version"]


def test_semantic_failure_exits_nonzero():
    runner = _runner()
    with patch.object(runner, "run", return_value=[{"raw_status": "RAW_COMPLETE", "semantic_status": "FAILED"}]):
        assert runner.main(["--manifest", "m", "--image-root", "i", "--output-root", "o", "--split", "test", "--apply-adapter"]) == 1
