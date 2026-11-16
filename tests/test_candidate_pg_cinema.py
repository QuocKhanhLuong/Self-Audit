"""Software-only tests for Candidate PG S1 CineMA contracts.

Every test injects a tiny source encoder; these fixtures are logic evidence,
not CineMA or patient/GPU evidence.
"""

from __future__ import annotations

import json
import csv
import hashlib
from collections.abc import Sequence
from pathlib import Path

import pytest
import torch
from torch import nn

from self_audit.candidate_pg.cinema_init import (
    HF_FILES,
    HF_REPO,
    _validate_official_config,
    backproject_feature,
    prepare_cinema_sax,
    run_cinema_init,
)
from self_audit.candidate_pg.cinema_init import _feature_to_grid


def _official_raw():
    return {
        "data": {"sax": {"spacing": [1.0, 1.0, 10.0], "patch_size": [192, 192, 16], "in_chans": 1}},
        "model": {
            "size": "base",
            "views": ["sax", "lax_2c", "lax_3c", "lax_4c"],
            "patch_size": [4, 4, 1],
            "scale_factor": [2, 2, 1],
            "enc_conv_chans": [64, 128],
            "enc_conv_n_blocks": 2,
        },
    }


def _geometry(shape=(8, 32, 32)):
    return {
        "shape": list(shape),
        "spacing_zyx": [10.0, 1.0, 1.0],
        "affine_zyx": torch.eye(4).tolist(),
        "context": "full_volume",
    }


class TinyCinema(nn.Module):
    def forward(self, image):
        b = image.shape[0]
        return {
            "conv_1_8": torch.ones((b, 6, 16, 24, 24), device=image.device),
            "tokens_1_16": torch.ones((b, 16 * 12 * 12, 10), device=image.device),
        }


class TinyStudent(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Conv3d(1, 8, 3, padding=1)
        self.stage1 = nn.Conv3d(8, 8, 3, stride=(1, 2, 2), padding=1)
        self.stage2 = nn.Conv3d(8, 16, 3, stride=(1, 2, 2), padding=1)
        self.stage3 = nn.Conv3d(16, 16, 3, stride=(1, 2, 2), padding=1)
        self.stage4 = nn.Conv3d(16, 24, 3, stride=(1, 2, 2), padding=1)

    def forward(self, image, *_args, **_kwargs):
        f0 = torch.nn.functional.gelu(self.stem(image))
        f1 = torch.nn.functional.gelu(self.stage1(f0))
        f2 = torch.nn.functional.gelu(self.stage2(f1))
        f3 = torch.nn.functional.gelu(self.stage3(f2))
        f4 = torch.nn.functional.gelu(self.stage4(f3))
        return {"features": (f0, f1, f2, f3, f4)}


def test_source_architecture_is_derived_from_official_yaml():
    architecture = _validate_official_config(_official_raw())
    assert architecture["skip_strides_zyx"] == [[1, 4, 4], [1, 8, 8]]
    assert architecture["token_stride_zyx"] == [1, 16, 16]
    assert architecture["cinema_conv_1_8_present"] is True


def test_real_source_rejects_non_huggingface_paths():
    from self_audit.candidate_pg.cinema_init import _required_source_settings

    with pytest.raises(ValueError, match="only pretrained/config.yaml"):
        _required_source_settings({"hf_repo": HF_REPO, "files": ("pretrained/config.yaml", "other.safetensors")})
    assert HF_FILES == ("pretrained/cinema.safetensors", "pretrained/config.yaml")


def test_non_sixteen_slices_have_end_padding_outside_valid_support():
    prepared = prepare_cinema_sax(torch.rand(8, 32, 32), _geometry(), synthetic=True)
    assert tuple(prepared.model_input_hwd.shape) == (1, 1, 192, 192, 16)
    assert int(prepared.valid_hwd[..., 8:].sum()) == 0
    assert int(prepared.valid_hwd[..., :8].sum()) == 8 * 32 * 32
    # Native z=7 maps to processed z=7; padded z=8 is not support.
    assert prepared.native_to_processed.shape == (4, 4)


def test_crop_affine_maps_declared_native_crop_origin_to_zero():
    geometry = _geometry((8, 64, 64))
    geometry["cinema_crop"] = {"start_zyx": [2, 4, 6], "size_zyx": [8, 192, 192]}
    prepared = prepare_cinema_sax(torch.rand(8, 64, 64), geometry, synthetic=True)
    origin = torch.tensor([2.0, 4.0, 6.0, 1.0], dtype=torch.float64)
    mapped = torch.as_tensor(prepared.native_to_processed) @ origin
    assert torch.allclose(mapped[:3], torch.zeros(3, dtype=torch.float64), atol=1e-5)


def test_official_token_flatten_hwd_is_restored_to_zyx():
    architecture = {"token_stride_zyx": [1, 16, 16]}
    # H,W,D flatten order used by cinema.vit.patchify_3d.
    tokens = torch.arange(12 * 12 * 16, dtype=torch.float32).reshape(1, -1, 1)
    grid = _feature_to_grid(tokens, feature_name="tokens_1_16", architecture=architecture)
    assert grid is not None and tuple(grid.shape) == (1, 1, 16, 12, 12)
    h, w, d = 3, 5, 7
    expected = float((h * 12 + w) * 16 + d)
    assert float(grid[0, 0, d, h, w]) == expected


def test_backprojection_uses_affine_support_mask():
    prepared = prepare_cinema_sax(torch.rand(8, 32, 32), _geometry(), synthetic=True)
    source = torch.randn(1, 4, 16, 24, 24)
    sampled, valid = backproject_feature(source, prepared, (8, 4, 4), source_stride_zyx=(1, 8, 8))
    assert tuple(sampled.shape) == (1, 4, 8, 4, 4)
    assert tuple(valid.shape) == (1, 1, 8, 4, 4)
    assert valid[:, :, 7].sum() > 0
    assert valid[:, :, 0].sum() > 0


def test_backprojection_convmae_stride_one_centres_are_integer_indices():
    """The z=1 tap must sample source index p, not p-0.5."""

    prepared = prepare_cinema_sax(torch.zeros(5, 32, 32), _geometry((5, 32, 32)), synthetic=True)
    # A coordinate-coded feature has value equal to its source z index.  With
    # ConvMAE kernel=stride=1 the five native z centres map to [0,1,2,3,4].
    source = torch.arange(5, dtype=torch.float32).reshape(1, 1, 5, 1, 1)
    sampled, valid = backproject_feature(
        source,
        prepared,
        (5, 1, 1),
        source_stride_zyx=(1, 8, 8),
        target_stride_zyx=(1, 8, 8),
    )
    assert torch.allclose(sampled[0, 0, :, 0, 0], torch.arange(5, dtype=torch.float32), atol=1e-6)
    assert torch.equal(valid[0, 0, :, 0, 0], torch.ones(5))


def test_backprojection_convmae_stride_eight_and_sixteen_hand_centres():
    """Kernel=stride centre inversion gives .5625 and .53125 at cell one."""

    prepared = prepare_cinema_sax(torch.zeros(5, 32, 32), _geometry((5, 32, 32)), synthetic=True)
    # Values are the source x index, so grid_sample's linear interpolation
    # exposes the exact feature-lattice coordinate returned by the affine map.
    source = torch.arange(3, dtype=torch.float32).reshape(1, 1, 1, 1, 3).expand(1, 1, 5, 1, 3)
    sampled8, valid8 = backproject_feature(
        source,
        prepared,
        (5, 1, 2),
        source_stride_zyx=(1, 8, 8),
        target_stride_zyx=(1, 8, 8),
    )
    sampled16, valid16 = backproject_feature(
        source,
        prepared,
        (5, 1, 2),
        source_stride_zyx=(1, 16, 16),
        target_stride_zyx=(1, 16, 16),
    )
    assert torch.allclose(sampled8[0, 0, 0, 0], torch.tensor([0.0, 0.5625]), atol=1e-6)
    assert torch.allclose(sampled16[0, 0, 0, 0], torch.tensor([0.0, 0.53125]), atol=1e-6)
    assert torch.equal(valid8[0, 0, 0, 0], torch.ones(2))
    assert torch.equal(valid16[0, 0, 0, 0], torch.ones(2))


def test_backprojection_padding_mask_excludes_centres_outside_crop_support():
    prepared = prepare_cinema_sax(torch.zeros(5, 32, 32), _geometry((5, 32, 32)), synthetic=True)
    # Simulate an end-padded processed z extent: source values at z>=3 are
    # still sampled with border mode but must not contribute to the loss.
    prepared.support_processed_bounds_zyx = ((0, 3), (0, 32), (0, 32))
    source = torch.arange(5, dtype=torch.float32).reshape(1, 1, 5, 1, 1)
    _, valid = backproject_feature(
        source,
        prepared,
        (5, 1, 1),
        source_stride_zyx=(1, 8, 8),
        target_stride_zyx=(1, 8, 8),
    )
    assert torch.equal(valid[0, 0, :, 0, 0], torch.tensor([1.0, 1.0, 1.0, 0.0, 0.0]))


def test_synthetic_run_is_train_only_and_discards_projectors(tmp_path):
    train = [{"image": torch.rand(8, 32, 32), "geometry": _geometry(), "split": "train"} for _ in range(3)]
    result = run_cinema_init(
        {"runtime": {"synthetic": True}, "cinema_init": {"synthetic": True, "updates": 2}},
        tmp_path,
        train,
        device="cpu",
        encoder_factory=lambda *args, **kwargs: TinyCinema(),
        student_encoder=TinyStudent(),
    )
    assert result["status"] == "SOFTWARE_ONLY_SYNTHETIC"
    receipt = json.loads(Path(result["receipt_path"]).read_text())
    assert receipt["updates"] == 2
    assert receipt["holdout_count"] == 1
    assert receipt["projectors_discarded"] is True
    payload = torch.load(result["checkpoint_path"], map_location="cpu", weights_only=False)
    assert payload["projectors_discarded"] is True
    assert all("projector" not in key.lower() and "decoder" not in key.lower() for key in payload["model"])
    with (tmp_path / "epochs.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert list(rows[0]) == ["epoch", "update", "cosine_train", "cosine_holdout"]
    assert len(rows) == 1 and rows[0]["update"] == "2"
    assert receipt["data_list_sha256"] == hashlib.sha256((tmp_path / "data_list.json").read_bytes()).hexdigest()


def test_s1_sequence_keeps_images_lazy_and_preserves_decoder(tmp_path):
    class LazyFrames(Sequence):
        identities = [{"patient_id": "p", "frame_index": i, "split": "train"} for i in range(3)]

        def __len__(self):
            return 3

        def __iter__(self):
            raise AssertionError("S1 must index lazily, never list all images")

        def __getitem__(self, index):
            assert isinstance(index, int)
            if index >= 3:
                raise IndexError(index)
            return {"image": torch.full((8, 32, 32), 0.5), "geometry": _geometry(), **self.identities[index]}

    student = TinyStudent()
    student.decoder = nn.Conv3d(24, 4, 1)
    before = {k: v.clone() for k, v in student.decoder.state_dict().items()}
    run_cinema_init({"runtime": {"synthetic": True}, "cinema_init": {"updates": 1}}, tmp_path,
                    LazyFrames(), cinema_encoder=TinyCinema(), student_encoder=student)
    assert all(torch.equal(before[k], v) for k, v in student.decoder.state_dict().items())
    assert all(p.grad is None for p in student.decoder.parameters())
    assert any(p.grad is not None and torch.any(p.grad != 0) for p in student.stage4.parameters())


def test_val_or_test_item_is_rejected_before_training(tmp_path):
    records = [{"image": torch.rand(8, 32, 32), "geometry": _geometry(), "split": "val"} for _ in range(2)]
    with pytest.raises(ValueError, match="TRAIN images only"):
        run_cinema_init(
            {"runtime": {"synthetic": True}, "cinema_init": {"synthetic": True, "updates": 1}},
            tmp_path,
            records,
            encoder_factory=lambda *args, **kwargs: TinyCinema(),
            student_encoder=TinyStudent(),
        )
