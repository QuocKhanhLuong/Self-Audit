from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch

import scripts.context_prediction_worker as worker


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fake_config(tmp_path: Path, image: np.ndarray, masks: dict[str, dict[str, str]]):
    data_dir = tmp_path / "development"
    data_dir.mkdir()
    image_path = data_dir / "patient_eval.npy"
    np.save(image_path, image, allow_pickle=False)
    manifest = {
        "schema": "self_audit.data.v2",
        "role": "development",
        "synthetic": True,
        "patients": [{"id": "eval001", "image": image_path.name, "image_sha256": _sha(image_path), "masks": masks}],
    }
    manifest_path = data_dir / "val.json"
    manifest_path.write_text(json.dumps(manifest, sort_keys=True))
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    payload = model_dir / "weights.pt"
    payload.write_bytes(b"toy-payload")
    model_manifest = {"payload_sha256": _sha(payload), "synthetic": True}
    model_manifest_path = model_dir / "MANIFEST.json"
    model_manifest_path.write_text(json.dumps(model_manifest, sort_keys=True))
    model_spec = {
        "path": str(model_dir),
        "manifest_sha256": _sha(model_manifest_path),
        "payload_sha256": _sha(payload),
        "runtime_root": str(Path(__file__).resolve().parents[1]),
        "runtime_source_sha256": "0" * 64,
    }
    config = {
        "schema": worker.SCHEMA,
        "lambdas": [0, 0.5, 1],
        "device": "cpu",
        "threads": 1,
        "parity": {"logit_atol": 1e-6, "logit_rtol": 1e-5},
        "val_manifest": {"path": str(manifest_path), "sha256": _sha(manifest_path)},
        "teacher": model_spec,
        "student": model_spec,
    }
    return config


class _ToyModel(torch.nn.Module):
    def __init__(self, stateful=False):
        super().__init__()
        self.stateful = stateful
        self.calls = 0

    def forward(self, x):
        self.calls += 1
        b, _, h, w = x.shape
        logits = torch.zeros((b, 4, h, w), dtype=torch.float32)
        logits[:, 0] = x[:, 1]
        if self.stateful and self.calls % 2 == 0:
            logits[:, 1] += 1.0
        return logits


def test_interpolate_context_identity_boundary_and_half():
    x = torch.tensor([[[[1.0]], [[2.0]], [[5.0]]]])
    full = worker.interpolate_context(x, 1.0)
    assert torch.equal(full, x)
    assert full.data_ptr() != x.data_ptr()
    center = worker.interpolate_context(x, 0.0)
    assert torch.equal(center, torch.tensor([[[[2.0]], [[2.0]], [[2.0]]]]))
    half = worker.interpolate_context(x, 0.5)
    assert torch.equal(half, torch.tensor([[[[1.5]], [[2.0]], [[3.5]]]]))


def test_parity_failure_aborts_before_counterfactuals(tmp_path, monkeypatch):
    image = np.arange(2 * 3 * 3 * 4, dtype=np.float32).reshape(2, 3, 3, 4)
    image /= image.max()
    config = _fake_config(tmp_path, image, {"0": {"path": "missing_gt.npy", "sha256": "0" * 64}})
    monkeypatch.setattr(worker, "_load_runtime_model", lambda *args: (_ToyModel(stateful=True), {"synthetic": True}))
    out = tmp_path / "out"
    with pytest.raises(RuntimeError, match="parity"):
        worker.run_worker(config, "teacher", out)
    assert not out.exists()


def test_no_ground_truth_read_and_frozen_coverage_hashes(tmp_path, monkeypatch):
    image = np.zeros((2, 3, 2, 3), dtype=np.float32)
    image[0, 0] = 0.1
    image[0, 1] = 0.4
    image[0, 2] = 0.9
    image[1, 0] = 0.2
    image[1, 1] = 0.5
    image[1, 2] = 0.8
    masks = {
        "0": {"path": "does-not-exist-ground-truth.npy", "sha256": "1" * 64},
        "1": {"path": "also-does-not-exist-ground-truth.npy", "sha256": "2" * 64},
    }
    config = _fake_config(tmp_path, image, masks)
    monkeypatch.setattr(worker, "_load_runtime_model", lambda *args: (_ToyModel(), {"synthetic": True}))
    out = tmp_path / "frozen"
    receipt = worker.run_worker(config, "student", out)
    assert receipt["coverage"] == {"patients": 1, "frames": 2, "slices": 6, "lambdas": [0.0, 0.5, 1.0], "prediction_files": 6}
    records = receipt["records"]
    assert len(records) == 6
    for record in records:
        path = Path(record["path"])
        assert path.is_absolute() and path.is_file()
        assert tuple(record["shape_zyx"]) == (3, 2, 3)
        assert _sha(path) == record["sha256"]
        assert _sha(path) == _sha(path)
        assert record["image_sha256"] == config["val_manifest"]["sha256"][:0] + json.loads(Path(config["val_manifest"]["path"]).read_text())["patients"][0]["image_sha256"]
    saved = json.loads((out / "PREDICTIONS_FROZEN.json").read_text())
    assert saved["role"] == "student"
    assert saved["parity"]["mismatches"] == 0
    assert saved["ground_truth_opened_before_freeze"] is False
    assert saved["protocol_sha256"] == worker.protocol_sha256(saved["protocol"])


def test_protocol_hash_mismatch_rejects_before_model_load(tmp_path, monkeypatch):
    image = np.zeros((1, 2, 2, 2), dtype=np.float32)
    config = _fake_config(tmp_path, image, {"0": {"path": "unopened_gt.npy", "sha256": "3" * 64}})
    called = {"value": False}

    def unexpected_loader(*args):
        called["value"] = True
        raise AssertionError("model loader must not run after protocol mismatch")

    monkeypatch.setattr(worker, "_load_runtime_model", unexpected_loader)
    snapshot = worker.protocol_snapshot("cpu", 1)
    expected = worker.protocol_sha256(snapshot)
    with pytest.raises(ValueError, match="protocol hash mismatch"):
        worker.run_worker(config, "teacher", tmp_path / "out", expected_protocol_sha256="f" * 64)
    assert not called["value"]
    assert not (tmp_path / "out").exists()
    # A current snapshot is accepted when its canonical hash is supplied.
    monkeypatch.setattr(worker, "_load_runtime_model", lambda *args: (_ToyModel(), {"synthetic": True}))
    receipt = worker.run_worker(config, "teacher", tmp_path / "accepted", expected_protocol_sha256=expected)
    assert receipt["protocol_sha256"] == expected


def test_patient_id_path_traversal_rejected(tmp_path, monkeypatch):
    image = np.zeros((1, 2, 2, 2), dtype=np.float32)
    config = _fake_config(tmp_path, image, {"0": {"path": "unopened_gt.npy", "sha256": "4" * 64}})
    manifest_path = Path(config["val_manifest"]["path"])
    manifest = json.loads(manifest_path.read_text())
    manifest["patients"][0]["id"] = "../escape"
    manifest_path.write_text(json.dumps(manifest, sort_keys=True))
    config["val_manifest"]["sha256"] = _sha(manifest_path)
    monkeypatch.setattr(worker, "_load_runtime_model", lambda *args: (_ToyModel(), {"synthetic": True}))
    with pytest.raises(ValueError, match="unsafe patient ID"):
        worker.run_worker(config, "student", tmp_path / "out")
