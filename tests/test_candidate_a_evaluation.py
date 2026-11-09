import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from self_audit.candidate_a import evaluation as ce
from self_audit.data import CLASS_MAPPING, GEOMETRY_MAP_SCHEMA, GEOMETRY_SCHEMA


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def test_surface_metrics_spacing_and_empty_cases():
    empty = np.zeros((8, 8), dtype=bool)
    both = ce._surface_metrics(empty, empty, (2.0, 1.0))
    assert both["status"] == "both_absent"
    one = ce._surface_metrics(empty, np.pad(np.ones((2, 2), bool), 3), (2.0, 1.0))
    assert one["status"] == "one_absent" and one["surface_dice"] == 0.0
    target = np.zeros((8, 8), dtype=bool); target[2:6, 2:6] = True
    same = ce._surface_metrics(target, target, (2.0, 1.0))
    assert same["status"] == "ok" and same["assd_mm"] == 0.0 and same["hd95_mm"] == 0.0


def test_student_only_evaluation_freezes_before_mask_reads(tmp_path, monkeypatch):
    data = tmp_path / "data"; data.mkdir()
    image = np.zeros((1, 1, 8, 8), np.float32); image[0, 0, 2:6, 2:6] = .8
    mask = np.zeros((1, 8, 8), np.uint8); mask[:, 2:6, 2:6] = 1
    image_path, mask_path = data / "p.npy", data / "p_mask.npy"
    np.save(image_path, image); np.save(mask_path, mask)
    native_geometry = {"schema": GEOMETRY_SCHEMA, "array_axes": ["T", "Z", "Y", "X"],
                       "spatial_axes": ["Z", "Y", "X"], "shape_zyx": [1, 8, 8],
                       "spacing_zyx_mm": [2.5, 2.0, 1.0], "spacing_yx_mm": [2.0, 1.0],
                       "affine_zyx": [[2.5, 0., 0., 0.], [0., 2., 0., 0.],
                                       [0., 0., 1., 0.], [0., 0., 0., 1.]],
                       "class_mapping": CLASS_MAPPING}
    manifest = {"schema": "self_audit.data.v2", "role": "development", "synthetic": True,
                "scope": "smoke", "class_mapping": CLASS_MAPPING,
                "geometry": {"schema": GEOMETRY_MAP_SCHEMA, "by_patient": {"p": native_geometry}},
                "patients": [{"id": "p", "image": image_path.name, "image_sha256": _sha(image_path),
                              "masks": {"0": {"path": mask_path.name, "sha256": _sha(mask_path)}}}]}
    manifest_path = data / "manifest.json"; manifest_path.write_text(json.dumps(manifest))

    class Model(torch.nn.Module):
        def infer(self, x, **kwargs):
            probs = torch.zeros((1, 4, 8, 8), dtype=x.dtype, device=x.device)
            probs[:, 0] = .9; probs[:, 1, 2:6, 2:6] = 1.0
            return {"probabilities": probs, "code": {"ids": [0], "values": [0.5]},
                    "trace": {"query_count": 1}}

    class Codec:
        class Quantized:
            levels = torch.tensor([3])

        def quantize_values(self, values, ids, ste=False):
            assert tuple(ids) == (0,)
            return self.Quantized()

        def serialize(self, ids, levels, shape, spacing, budget_bits=None):
            assert tuple(ids) == (0,)
            return b"candidate-code"

        class Packet:
            ids = (0,)

        def deserialize(self, blob, budget_bits=None):
            return self.Packet()

        def dequantize(self, packet, device=None, dtype=torch.float32):
            return torch.tensor([0.5], device=device, dtype=dtype)

    monkeypatch.setattr(ce, "_load_student", lambda *args, **kwargs: (Model(), object(), Codec(),
                                                                        {"synthetic": True,
                                                                         "config": {"rollout": {}}}))
    def fake_infer(*args, **kwargs):
        probs = torch.zeros((1, 4, 8, 8)); probs[:, 0] = .9; probs[:, 1] = .9
        probs[:, 1, 2:6, 2:6] = 1.0
        return probs, {"ids": [0], "values": torch.tensor([[.5]])}, {"query_count": 1}
    monkeypatch.setattr(ce, "_student_infer", fake_infer)
    monkeypatch.setattr(Model, "decode", lambda self, *args, **kwargs:
                            fake_infer()[0], raising=False)
    out = tmp_path / "out"
    result = ce.evaluate("checkpoint", "unused", manifest_path, _sha(manifest_path), out, device="cpu")
    assert result["synthetic"] is True
    assert result["macro_patient"]["RV"]["dice"] == 1.0
    assert result["code"]["total_bytes"] == len(b"candidate-code")
    assert (out / "PREDICTIONS_FROZEN.json").exists()


def test_serialize_rejects_missing_student_values():
    try:
        ce._serialize(object(), {"ids": [0]}, (1.0, 1.0), (2, 2))
    except ValueError as exc:
        assert "values" in str(exc)
    else:
        raise AssertionError("missing code must fail closed")


def _write_invalid_geometry_manifest(tmp_path, *, geometry=True, shape=(1, 4, 5), class_mapping=CLASS_MAPPING):
    image = np.zeros((1, 1, 4, 5), np.float32)
    image_path = tmp_path / "eval.npy"; np.save(image_path, image)
    manifest = {"schema": "self_audit.data.v2", "role": "development", "synthetic": True,
                "class_mapping": class_mapping,
                "patients": [{"id": "eval", "image": image_path.name,
                              "image_sha256": _sha(image_path), "masks": {}}]}
    if geometry:
        manifest["geometry"] = {"schema": GEOMETRY_MAP_SCHEMA, "by_patient": {"eval": {
            "schema": GEOMETRY_SCHEMA, "array_axes": ["T", "Z", "Y", "X"],
            "spatial_axes": ["Z", "Y", "X"], "shape_zyx": list(shape),
            "spacing_zyx_mm": [2.5, 2.0, 1.0], "spacing_yx_mm": [2.0, 1.0],
            "affine_zyx": [[2.5, 0., 0., 0.], [0., 2., 0., 0.],
                            [0., 0., 1., 0.], [0., 0., 0., 1.]],
            "class_mapping": CLASS_MAPPING}}}
    path = tmp_path / "manifest.json"; path.write_text(json.dumps(manifest))
    return path


@pytest.mark.parametrize("kwargs", [
    {"geometry": False},
    {"shape": (2, 4, 5)},
    {"class_mapping": {"0": "wrong"}},
])
def test_evaluation_rejects_untrusted_geometry_before_loading_student(tmp_path, monkeypatch, kwargs):
    manifest = _write_invalid_geometry_manifest(tmp_path, **kwargs)
    monkeypatch.setattr(ce, "_load_student", lambda *args, **kw: (_ for _ in ()).throw(AssertionError("student loaded")))
    with pytest.raises(ValueError):
        ce.evaluate("checkpoint", "unused", manifest, _sha(manifest), tmp_path / "out", device="cpu")


def test_evaluation_rejects_sheared_affine(tmp_path, monkeypatch):
    manifest = _write_invalid_geometry_manifest(tmp_path)
    payload = json.loads(manifest.read_text())
    payload["geometry"]["by_patient"]["eval"]["affine_zyx"] = [
        [2.5, 0., 0., 0.], [0.5, (3.0 ** 0.5) / 2.0, 0., 0.],
        [0., 0., 1., 0.], [0., 0., 0., 1.]]
    manifest.write_text(json.dumps(payload))
    monkeypatch.setattr(ce, "_load_student", lambda *args, **kw: (_ for _ in ()).throw(AssertionError("student loaded")))
    with pytest.raises(ValueError):
        ce.evaluate("checkpoint", "unused", manifest, _sha(manifest), tmp_path / "out", device="cpu")
