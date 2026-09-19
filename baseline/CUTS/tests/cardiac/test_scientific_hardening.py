from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[4]


def _runner():
    spec = importlib.util.spec_from_file_location("cuts_scientific_runner_test", ROOT / "scripts" / "run_cuts_scientific.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _grid() -> dict:
    return json.loads((ROOT / "benchmark_freezes" / "cardiac_benchmark_v3" / "configs" / "resolved_shared_contract.json").read_text())


def test_frozen_grid_config_and_25d_center_helpers():
    runner = _runner()
    manifest = {"schema_version": "shared_benchmark_manifest.v1", "split_policy_version": "self_audit.acdc.patient_split.v1", "shared_grid": _grid(), "shared_grid_hash": runner.FROZEN_SELF_AUDIT_SHARED_GRID_SHA256}
    runner._require_frozen_grid(manifest)
    bad = {**manifest, "shared_grid": {**manifest["shared_grid"], "target_hw": [8, 8]}}
    with pytest.raises(runner.ArtifactError):
        runner._require_frozen_grid(bad)
    with pytest.raises(runner.ArtifactError):
        runner._require_expected_config_hash("not-the-computed-hash", "computed")
    image = torch.stack((torch.full((3, 3), 10.0), torch.full((3, 3), 20.0), torch.full((3, 3), 30.0)))
    sample = SimpleNamespace(image=image)
    assert np.array_equal(runner._central_image(sample, "CUTS-2D"), image[0].numpy())
    assert np.array_equal(runner._central_image(sample, "CUTS-2.5D"), image[1].numpy())


def test_cpu_receipt_and_semantic_failure_exit_are_observable():
    runner = _runner()
    receipt = runner._finish_execution_receipt(runner._start_execution_receipt("cpu"), "cpu")
    assert receipt["cuda_peak_allocated_bytes"] is None
    assert receipt["cuda_peak_reserved_bytes"] is None
    assert receipt["python_version"] and receipt["pytorch_version"] and receipt["numpy_version"]
    with patch.object(runner, "run", return_value=[{"raw_status": "RAW_COMPLETE", "semantic_status": "FAILED"}]):
        assert runner.main(["--manifest", "m", "--image-root", "i", "--output-root", "o", "--checkpoint", "c", "--split", "test", "--apply-adapter"]) == 1
