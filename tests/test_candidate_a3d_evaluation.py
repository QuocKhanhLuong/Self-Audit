import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from self_audit.candidate_a3d import evaluation as ev
from self_audit.data import CLASS_MAPPING, GEOMETRY_MAP_SCHEMA, GEOMETRY_SCHEMA
from self_audit.model import digest


def _manifest(tmp_path, *, shape=(3, 4, 5), unknown=False, vendor="vendorA"):
    image = np.zeros((1, *shape), dtype=np.float32)
    image[0, 0, 1:3, 1:3] = 0.8
    reference = np.zeros(shape, dtype=np.uint8)
    reference[0, 1:3, 1:3] = 1
    reference[2, 1:3, 3:5] = 2
    if unknown:
        reference[0, 0, 0] = 255
    image_path = tmp_path / "image.npy"
    mask_path = tmp_path / "mask.npy"
    np.save(image_path, image)
    np.save(mask_path, reference)
    spacing = [2.5, 0.7, 1.3]
    geometry = {
        "schema": GEOMETRY_SCHEMA,
        "array_axes": ["T", "Z", "Y", "X"],
        "spatial_axes": ["Z", "Y", "X"],
        "shape_zyx": list(shape),
        "spacing_zyx_mm": spacing,
        "spacing_yx_mm": spacing[1:],
        "affine_zyx": [[spacing[0], 0, 0, 0], [0, spacing[1], 0, 0],
                        [0, 0, spacing[2], 0], [0, 0, 0, 1]],
        "class_mapping": CLASS_MAPPING,
    }
    manifest = {
        "schema": "self_audit.data.v2", "role": "development", "synthetic": True,
        "scope": "smoke", "class_mapping": CLASS_MAPPING,
        "geometry": {"schema": GEOMETRY_MAP_SCHEMA, "by_patient": {"p": geometry}},
        "patient_metadata": {"p": {"vendor": vendor}},
        "patients": [{"id": "p", "image": image_path.name,
                       "image_sha256": digest(image_path),
                       "masks": {"0": {"path": mask_path.name, "sha256": digest(mask_path)}}}],
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    return manifest_path, image, reference


def _perfect_logits(reference):
    logits = torch.full((1, 4, *reference.shape), -2.0)
    logits[:, 0] = 1.0
    for class_id in (1, 2, 3):
        logits[:, class_id][torch.from_numpy(reference == class_id)[None]] = 3.0
    return logits


def test_surface_metrics_use_anisotropic_3d_spacing_and_ignore_unknown():
    reference = np.zeros((4, 5, 6), dtype=np.uint8)
    reference[1:3, 1:4, 2:5] = 1
    same = ev._surface_metrics_3d(reference == 1, reference == 1, (4.0, 0.7, 1.3))
    assert same["status"] == "ok"
    assert same["assd_mm"] == 0.0 and same["hd95_mm"] == 0.0
    reference[0, 0, 0] = 255
    ignored = ev._surface_metrics_for_class(reference, reference, 1, (4.0, 0.7, 1.3))
    assert ignored == {"status": "ignored_reference_voxels", "surface_dice": None,
                       "assd_mm": None, "hd95_mm": None}


def test_teacher_epoch_freezes_both_3d_branches_before_mask_scoring(tmp_path, monkeypatch):
    manifest, _image, reference = _manifest(tmp_path)
    logits = _perfect_logits(reference)

    class Teacher(torch.nn.Module):
        def forward(self, x, *, geometry):
            assert tuple(geometry["shape"]) == reference.shape
            assert tuple(geometry["spacing_zyx"]) == (2.5, 0.7, 1.3)
            assert geometry["context"] == "full_volume"
            branch = logits.to(x.device, x.dtype)
            return {"region_logits": branch, "boundary_logits": branch.clone(),
                    "sdf": torch.zeros_like(branch)}

    output = tmp_path.parent / f"{tmp_path.name}-teacher_epoch"
    original_score = ev._score_volume_predictions

    def assert_frozen(*args, **kwargs):
        assert (output / "PREDICTIONS_FROZEN.json").is_file()
        return original_score(*args, **kwargs)

    monkeypatch.setattr(ev, "_score_volume_predictions", assert_frozen)
    result = ev.evaluate_teacher_epoch(
        Teacher(), manifest, digest(manifest), output, epoch=2, device="cpu",
        train_ids=("train-only",), synthetic=True,
    )
    assert set(result["branches"]) == {"region", "boundary"}
    assert result["branches"]["region"]["per_vendor"]["vendorA"]["RV"]["dice"] == 1.0
    assert result["branches"]["region"]["per_class"]["RV"]["f1"] == 1.0
    assert (output / "TEACHER_EPOCH_METRICS.json").is_file()


def test_student_epoch_uses_packet_roundtrip_and_full_volume_input(tmp_path, monkeypatch):
    manifest, _image, reference = _manifest(tmp_path)
    expected_logits = _perfect_logits(reference)

    class Student(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.zeros(()))
            self.seen_geometry = None

        def decode(self, code_ids, code_values, geometry):
            self.seen_geometry = geometry
            assert tuple(geometry["shape"]) == reference.shape
            assert geometry["context"] == "full_volume"
            assert "features" not in geometry
            return expected_logits.to(self.anchor.device, self.anchor.dtype)

    student = Student()

    def fake_rollout(model, x, bank, codec, geometry, config):
        assert tuple(x.shape) == (1, 1, *reference.shape)
        assert tuple(geometry["shape"]) == reference.shape
        return SimpleNamespace(final_logits=expected_logits, code_ids=(0,),
                               code_values=torch.tensor([0.25]), query_count=1)

    class Codec:
        class Quantized:
            levels = torch.tensor([7], dtype=torch.int64)

        class Packet:
            ids = (0,)
            shape = reference.shape
            spacing_zyx = (2.5, 0.7, 1.3)
            affine_zyx = ((2.5, 0, 0, 0), (0, 0.7, 0, 0),
                          (0, 0, 1.3, 0), (0, 0, 0, 1))

        def quantize_values(self, values, ids, ste=False):
            assert tuple(ids) == (0,)
            assert ste is False
            return self.Quantized()

        def serialize(self, ids, levels, shape, spacing, *, affine_zyx, budget_bits):
            assert tuple(ids) == (0,)
            assert tuple(shape) == reference.shape
            assert tuple(spacing) == (2.5, 0.7, 1.3)
            assert budget_bits == 128
            return b"packet-v1"

        def deserialize(self, blob, budget_bits):
            assert blob == b"packet-v1" and budget_bits == 128
            return self.Packet()

        def dequantize(self, packet, device, dtype):
            return torch.tensor([0.25], device=device, dtype=dtype)

    monkeypatch.setattr(ev, "_student_rollout", fake_rollout)
    output = tmp_path.parent / f"{tmp_path.name}-student_epoch"
    result = ev.evaluate_epoch(
        student, object(), Codec(), SimpleNamespace(bit_budget=128), manifest,
        digest(manifest), output, epoch=3, device="cpu", train_ids=(), synthetic=True,
    )
    assert result["macro_patient"]["RV"]["dice"] == 1.0
    assert result["code"]["total_bytes"] == len(b"packet-v1")
    assert result["profile"]["total_ms"] >= 0.0
    assert student.seen_geometry["context"] == "full_volume"
    assert (output / "PREDICTIONS_FROZEN.json").is_file()
    assert (output / "EPOCH_METRICS.json").is_file()


def test_a3d_manifest_rejects_geometry_shape_before_teacher_forward(tmp_path):
    manifest, _image, _reference = _manifest(tmp_path)
    payload = json.loads(manifest.read_text())
    payload["geometry"]["by_patient"]["p"]["shape_zyx"] = [99, 4, 5]
    manifest.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="geometry"):
        ev.evaluate_teacher_epoch(
            lambda x: {}, manifest, digest(manifest), tmp_path / "bad", epoch=0,
            device="cpu", synthetic=True,
        )


def test_a3d_rejects_unsupported_device_before_model_or_output(tmp_path):
    manifest, _image, _reference = _manifest(tmp_path)

    class UnreachableTeacher(torch.nn.Module):
        def forward(self, *args, **kwargs):
            raise AssertionError("unsupported device must fail before teacher forward")

    output = tmp_path.parent / f"{tmp_path.name}-unsupported-device"
    with pytest.raises(ValueError, match="CPU or CUDA"):
        ev.evaluate_teacher_epoch(
            UnreachableTeacher(), manifest, digest(manifest), output, epoch=0,
            device="mps", synthetic=True,
        )
    assert not output.exists()
