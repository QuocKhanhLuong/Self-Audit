import numpy as np
import pytest
import torch
from torch import nn
from types import ModuleType

from self_audit.cinema_teacher import (
    CLASS_MAP,
    CineMATeacher,
    _resample_to_spacing,
    _tile_starts,
    _axis_support_mask,
    _validate_geometry,
    verify_sha256,
    _import_pinned_get_model,
    _verify_module_origin,
)
import self_audit.cinema_teacher as cinema_teacher_module


class MockCineMA(nn.Module):
    def __init__(self):
        super().__init__()
        self.decoder_dict = nn.ModuleDict({"sax": nn.Identity()})
        self.pred_head_dict = nn.ModuleDict({"sax": nn.Conv3d(32, 4, 1)})
        self.grad_ckpt = True

    def set_grad_ckpt(self, enabled):
        self.grad_ckpt = enabled

    def forward(self, image_dict):
        image = image_dict["sax"]
        features = self.decoder_dict["sax"](image.expand(-1, 32, -1, -1, -1))
        return {"sax": self.pred_head_dict["sax"](features)}


def test_class_map_and_half_overlap_tile_starts():
    assert CLASS_MAP == {0: "BG", 1: "RV", 2: "MYO", 3: "LV"}
    assert _tile_starts(192, 192) == [0]
    assert _tile_starts(193, 192) == [0, 1]
    assert _tile_starts(400, 192) == [0, 96, 192, 208]


def test_direction_aware_grid_origin_handles_oblique_and_flip():
    theta = np.deg2rad(27.0)
    direction = np.array([
        [-np.cos(theta), 0.0, np.sin(theta)],
        [-np.sin(theta), 0.0, -np.cos(theta)],
        [0.0, 1.0, 0.0],
    ])
    old_spacing = np.array([1.6, 1.2, 8.0])
    affine = np.eye(4)
    affine[:3, :3] = direction * old_spacing
    affine[:3, 3] = [13.0, -7.0, 4.0]
    volume = np.arange(8 * 7 * 3, dtype=np.float32).reshape(8, 7, 3)
    _, target_affine, info = _resample_to_spacing(volume, affine, old_spacing, (1, 1, 10))
    expected_shift = direction @ ((np.array([1.0, 1.0, 10.0]) - old_spacing) / 2)
    np.testing.assert_allclose(target_affine[:3, 3] - affine[:3, 3], expected_shift)
    np.testing.assert_allclose(info["direction_xyz"], direction)
    bridge = np.linalg.inv(target_affine) @ affine
    # Physical points computed in either voxel grid agree after the bridge.
    idx_native = np.array([2.0, 3.0, 1.0, 1.0])
    idx_target = bridge @ idx_native
    np.testing.assert_allclose(target_affine @ idx_target, affine @ idx_native)


def test_resampling_support_tracks_extra_ceil_edge_centers():
    volume = np.ones((4, 3, 2), dtype=np.float32)
    affine = np.diag([1.6, 1.0, 10.0, 1.0])
    spacing = np.array([1.6, 1.0, 10.0])
    resampled, target_affine, _ = _resample_to_spacing(volume, affine, spacing, (1, 1, 10))
    support = _axis_support_mask(resampled.shape, np.linalg.inv(affine) @ target_affine, volume.shape)
    assert support.shape == resampled.shape
    assert not support[-1].any()
    native_support = _axis_support_mask(volume.shape, np.linalg.inv(target_affine) @ affine, resampled.shape)
    assert native_support.all()


def test_mock_teacher_native_outputs_feature_tap_and_frozen_inference():
    model = MockCineMA()
    teacher = CineMATeacher(model=model)
    model.train()  # A caller-side mode toggle must not activate dropout.
    model.requires_grad_(True)
    # Small native volume gets end-padded to exactly one model tile.
    volume = np.random.default_rng(4).normal(size=(7, 6, 3)).astype(np.float32)
    affine = np.diag([1.0, 1.0, 10.0, 1.0])
    result = teacher.predict(volume, affine, timepoint="ED")
    assert result.probabilities.shape == (4, 7, 6, 3)
    assert result.features.shape == (32, 7, 6, 3)
    assert result.valid_mask.shape == (7, 6, 3)
    assert result.valid_mask.all()
    assert np.isfinite(result.probabilities).all()
    np.testing.assert_allclose(result.probabilities.sum(axis=0), 1.0, atol=1e-5)
    assert result.geometry["tile_starts_xyz"] == [[0, 0, 0]]
    assert result.lineage["status"] == "SYNTHETIC"
    assert result.lineage["feature_logit_parity_checked"] is True
    assert result.lineage["hook_on_off_parity_checked"] is True
    assert result.lineage["hook_on_off_max_abs_delta"] == 0.0
    assert result.lineage["normalization_deviation"].startswith("image-only full-FOV")
    assert not model.training
    assert model.grad_ckpt is False
    assert all(not p.requires_grad for p in model.parameters())
    assert all(p.grad is None for p in model.parameters())
    assert teacher.synthetic is True


def test_bad_sha_fails_closed(tmp_path):
    asset = tmp_path / "asset.bin"
    asset.write_bytes(b"test")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        verify_sha256(asset, "0" * 64, "fixture")


def test_pinned_loader_uses_strict_state_dict_loading(monkeypatch, tmp_path):
    calls = {}

    class ConfigLoader:
        @staticmethod
        def load(path):
            calls["config_path"] = path
            return {"fixture": True}

    class StrictModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(1))

        def load_state_dict(self, state_dict, strict=True):
            calls["strict"] = strict
            calls["state"] = state_dict

    expected_model = StrictModel()
    monkeypatch.setattr(
        cinema_teacher_module,
        "_import_pinned_get_model",
        lambda _root: (ConfigLoader, lambda _path, device: {"weight": torch.zeros(1)}, lambda _cfg: expected_model),
    )
    loaded = cinema_teacher_module._load_pinned_model(
        tmp_path, tmp_path / "config.yaml", tmp_path / "weights.safetensors"
    )
    assert loaded is expected_model
    assert calls["strict"] is True
    assert calls["state"]["weight"].item() == 0


def test_import_does_not_load_upstream_or_mask_reading_modules():
    import sys

    assert "cinema.segmentation.dataset" not in sys.modules
    assert "cinema.segmentation.mnms" not in sys.modules
    assert "monai.transforms" not in sys.modules


def test_cached_cinema_module_outside_source_root_is_rejected(tmp_path):
    fake = ModuleType("cinema.segmentation.convunetr")
    fake.__file__ = "/tmp/unpinned/convunetr.py"
    with pytest.raises(ValueError, match="outside pinned CineMA source root"):
        _verify_module_origin(tmp_path, "cinema.segmentation.convunetr", fake)


def test_loader_rejects_preloaded_unpinned_cinema_package(tmp_path, monkeypatch):
    module_root = tmp_path / "cinema" / "segmentation"
    module_root.mkdir(parents=True)
    (module_root / "convunetr.py").write_text("# pinned fixture path\n")
    cached = ModuleType("cinema")
    cached.__file__ = "/tmp/unpinned/cinema/__init__.py"
    monkeypatch.setitem(__import__("sys").modules, "cinema", cached)
    with pytest.raises(ValueError, match="outside pinned CineMA source root"):
        _import_pinned_get_model(tmp_path)


def test_nonfinite_decoder_features_fail_closed():
    class InfiniteFeatures(nn.Module):
        def forward(self, image):
            return torch.full((image.shape[0], 32, *image.shape[2:]), float("inf"), device=image.device)

    model = MockCineMA()
    model.decoder_dict["sax"] = InfiniteFeatures()
    teacher = CineMATeacher(model=model)
    with pytest.raises(RuntimeError, match="decoder features contain non-finite"):
        teacher.predict(np.ones((3, 3, 2), dtype=np.float32), np.diag([1, 1, 10, 1]), timepoint="ED")


def test_nonfinite_logits_fail_closed_even_if_allclose_would_pass():
    model = MockCineMA()
    with torch.no_grad():
        model.pred_head_dict["sax"].weight.fill_(float("inf"))
    teacher = CineMATeacher(model=model)
    with pytest.raises(RuntimeError, match="logits contain non-finite"):
        teacher.predict(np.ones((3, 3, 2), dtype=np.float32), np.diag([1, 1, 10, 1]), timepoint="ED")


def test_workspace_preflight_rejects_before_resampling(monkeypatch):
    teacher = CineMATeacher(model=MockCineMA(), max_workspace_bytes=1)
    monkeypatch.setattr(
        cinema_teacher_module,
        "_resample_to_spacing",
        lambda *_args, **_kwargs: pytest.fail("resampling must not run before workspace preflight"),
    )
    with pytest.raises(MemoryError, match="workspace estimate"):
        teacher.predict(np.ones((3, 3, 2), dtype=np.float32), np.diag([1, 1, 10, 1]), timepoint="ED")


def test_support_weighting_preserves_constant_features_at_partial_edge():
    class ConstantFeatures(nn.Module):
        def forward(self, image):
            return torch.ones((image.shape[0], 32, *image.shape[2:]), device=image.device)

    model = MockCineMA()
    model.decoder_dict["sax"] = ConstantFeatures()
    teacher = CineMATeacher(model=model)
    result = teacher.predict(
        np.ones((3, 3, 3), dtype=np.float32), np.diag([1, 1, 8, 1]), timepoint="ED"
    )
    assert result.valid_mask.all()
    np.testing.assert_allclose(result.features, 1.0, atol=1e-6)
    np.testing.assert_allclose(result.probabilities.sum(axis=0), 1.0, atol=1e-6)
    assert result.geometry["native_grid_partial_support_voxels"] > 0


def test_float64_values_that_overflow_float32_are_rejected():
    with pytest.raises(ValueError, match="overflows float32"):
        _validate_geometry(
            np.full((2, 2, 2), np.finfo(np.float64).max),
            np.diag([1, 1, 10, 1]),
        )

