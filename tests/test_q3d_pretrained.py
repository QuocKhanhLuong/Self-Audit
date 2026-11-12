"""Software-only contract tests for Q-3D public checkpoint loading."""

from __future__ import annotations

import hashlib

import pytest
import torch
from torch import nn

from self_audit.candidate_a3d.teacher import DualBranchTeacher3D, UPSTREAM_MAIN_SHA
from self_audit.candidate_q3d import pretrained as q3d_pretrained


class _FakeOutput(nn.Module):
    def __init__(self, out_channels: int = 4) -> None:
        super().__init__()
        self.conv_out = nn.Conv3d(2, out_channels, kernel_size=1)


class _FakeMedNeXtState(nn.Module):
    """Tiny state fixture with the pinned model classifier key names."""

    def __init__(self, out_channels: int = 4) -> None:
        super().__init__()
        self.stem = nn.Conv3d(1, 2, kernel_size=1)
        self.out_0 = _FakeOutput(out_channels)


class _FakeEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = _FakeMedNeXtState()
        self.channels = (2, 2)

    def forward_region(self, x: torch.Tensor) -> torch.Tensor:
        return self.model.out_0.conv_out(self.model.stem(x))

    def forward_features(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        feature = self.model.stem(x)
        return feature, feature

    def forward_with_features(self, x: torch.Tensor) -> tuple[tuple[torch.Tensor, ...], torch.Tensor]:
        features = self.forward_features(x)
        return features, self.model.out_0.conv_out(features[0])


def _fake_teacher() -> DualBranchTeacher3D:
    return DualBranchTeacher3D(_FakeEncoder(), backbone_meta={"source_commit": UPSTREAM_MAIN_SHA})


def _config(path, payload_hash: str, *, class_mapping=None):
    return {
        "model": {
            "teacher": {
                "architecture": "mednext_l_k3",
                "source_repo": "/verified/official/MedNeXt",
                "source_commit": UPSTREAM_MAIN_SHA,
                "pretrained": {
                    "weights_path": str(path),
                    "weights_sha256": payload_hash,
                    "weights_source_url": "https://github.com/MIC-DKFZ/MedNeXt/releases/download/public/v1Lk3.pt",
                    "pretraining_dataset": "public_fixture_dataset",
                    "overlap_acdc": False,
                    "overlap_mnms": False,
                    "class_mapping": class_mapping or ["BG", "RV", "MYO", "LV"],
                },
            },
        },
    }


def _write_payload(path, state, *, metadata=None) -> str:
    payload = {"state_dict": state}
    if metadata is not None:
        payload["metadata"] = metadata
    torch.save(payload, path)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _metadata(class_mapping=None):
    return {
        "pretraining_dataset": "public_fixture_dataset",
        "overlap_acdc": False,
        "overlap_mnms": False,
        "class_mapping": class_mapping or ["BG", "RV", "MYO", "LV"],
    }


def _patch_fake_builder(monkeypatch):
    monkeypatch.setattr(
        q3d_pretrained,
        "_construct_mednext_teacher",
        lambda teacher_config: (
            _fake_teacher(),
            {"backbone": {"source_commit": UPSTREAM_MAIN_SHA, "weights_status": "UNTRAINED_RANDOM_INIT"}},
        ),
    )


def test_public_payload_load_is_weights_only_and_inventory_marks_random_boundary(tmp_path, monkeypatch):
    source_model = _FakeMedNeXtState()
    state = {key: torch.full_like(value, 0.125) for key, value in source_model.state_dict().items()}
    checkpoint = tmp_path / "public-mednext.pt"
    payload_hash = _write_payload(checkpoint, state, metadata=_metadata())
    _patch_fake_builder(monkeypatch)

    original_load = torch.load
    observed = {}

    def checked_load(*args, **kwargs):
        observed["weights_only"] = kwargs.get("weights_only")
        return original_load(*args, **kwargs)

    monkeypatch.setattr(q3d_pretrained.torch, "load", checked_load)
    teacher, provenance = q3d_pretrained.build_q3d_teacher(_config(checkpoint, payload_hash))

    assert observed["weights_only"] is True
    assert all(
        torch.equal(teacher.encoder.model.state_dict()[key], value)
        for key, value in state.items()
    )
    inventory = provenance["weight_inventory"]
    assert inventory["loaded_keys"] == sorted(f"encoder.model.{key}" for key in state)
    assert set(inventory["boundary_random_keys"]).issubset(inventory["missing_keys"])
    assert set(inventory["missing_keys"]) == set(inventory["boundary_random_keys"])
    assert inventory["unexpected_keys"] == []
    assert inventory["boundary_random_keys"]
    assert set(inventory["boundary_random_keys"]).issubset(inventory["reinitialized_keys"])
    assert inventory["reinitialized_keys"] == inventory["boundary_random_keys"]
    assert inventory["classifier_reinitialized"] is False
    assert provenance["checkpoint"]["weights_sha256"] == payload_hash
    assert provenance["pretraining_data"]["overlap_acdc"] is False
    assert teacher.provenance["checkpoint"]["weights_source_url"].startswith("https://")


def test_checkpoint_hash_mismatch_fails_before_architecture_construction(tmp_path, monkeypatch):
    checkpoint = tmp_path / "public-mednext.pt"
    payload_hash = _write_payload(checkpoint, _FakeMedNeXtState().state_dict(), metadata=_metadata())
    config = _config(checkpoint, "0" * 64)

    def construction_must_not_run(_teacher_config):
        pytest.fail("architecture was constructed before checkpoint hash verification")

    monkeypatch.setattr(q3d_pretrained, "_construct_mednext_teacher", construction_must_not_run)
    assert len(payload_hash) == 64
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        q3d_pretrained.build_q3d_teacher(config)


def test_incompatible_non_classifier_tensor_is_rejected(tmp_path, monkeypatch):
    state = _FakeMedNeXtState().state_dict()
    state["stem.weight"] = torch.zeros(7, 1, 1, 1, 1)
    checkpoint = tmp_path / "bad-shape.pt"
    payload_hash = _write_payload(checkpoint, state, metadata=_metadata())
    _patch_fake_builder(monkeypatch)

    with pytest.raises(ValueError, match="incompatible MedNeXt tensor shapes"):
        q3d_pretrained.build_q3d_teacher(_config(checkpoint, payload_hash))


def test_classifier_reinitialization_flag_cannot_bypass_class_order_mismatch(tmp_path, monkeypatch):
    state = _FakeMedNeXtState(out_channels=3).state_dict()
    source_classes = ["BG", "MYO", "LV"]
    checkpoint = tmp_path / "three-class-public.pt"
    payload_hash = _write_payload(checkpoint, state, metadata=_metadata(source_classes))
    config = _config(checkpoint, payload_hash, class_mapping=source_classes)
    config["model"]["teacher"]["pretrained"]["classifier_reinitialization"] = {
        "enabled": True,
        "source_class_mapping": source_classes,
        "reason": "public checkpoint classes differ from the fixed Q-3D cardiac mapping",
    }

    def construction_must_not_run(_teacher_config):
        pytest.fail("classifier reset must be rejected before constructing a random region head")

    monkeypatch.setattr(q3d_pretrained, "_construct_mednext_teacher", construction_must_not_run)
    with pytest.raises(ValueError, match="classifier_reinitialization is unsupported"):
        q3d_pretrained.build_q3d_teacher(config)


def test_missing_region_classifier_key_fails_even_if_reset_flag_is_disabled(tmp_path, monkeypatch):
    state = _FakeMedNeXtState().state_dict()
    del state["out_0.conv_out.weight"]
    checkpoint = tmp_path / "missing-region-head.pt"
    payload_hash = _write_payload(checkpoint, state, metadata=_metadata())
    _patch_fake_builder(monkeypatch)

    with pytest.raises(ValueError, match="missing required MedNeXt keys"):
        q3d_pretrained.build_q3d_teacher(_config(checkpoint, payload_hash))


def test_class_order_mismatch_fails_closed_without_reset_option(tmp_path, monkeypatch):
    checkpoint = tmp_path / "wrong-order-public.pt"
    source_classes = ["BG", "MYO", "RV", "LV"]
    payload_hash = _write_payload(
        checkpoint, _FakeMedNeXtState().state_dict(), metadata=_metadata(source_classes)
    )
    _patch_fake_builder(monkeypatch)

    with pytest.raises(ValueError, match="class_mapping must exactly match"):
        q3d_pretrained.build_q3d_teacher(
            _config(checkpoint, payload_hash, class_mapping=source_classes)
        )


def test_missing_overlap_provenance_fails_closed(tmp_path, monkeypatch):
    checkpoint = tmp_path / "raw-state.pt"
    payload_hash = _write_payload(checkpoint, _FakeMedNeXtState().state_dict())
    _patch_fake_builder(monkeypatch)
    config = _config(checkpoint, payload_hash)
    del config["model"]["teacher"]["pretrained"]["overlap_acdc"]

    with pytest.raises(ValueError, match="missing overlap_acdc"):
        q3d_pretrained.build_q3d_teacher(config)


@pytest.mark.parametrize(
    "source_url",
    [
        "https://zenodo.org/records/1234567/files/mednext.pt",
        "https://huggingface.co/MIC-DKFZ/MedNeXt/resolve/main/mednext.pt",
    ],
)
def test_public_https_source_schema_accepts_zenodo_and_huggingface(tmp_path, monkeypatch, source_url):
    checkpoint = tmp_path / "public-mednext.pt"
    payload_hash = _write_payload(checkpoint, _FakeMedNeXtState().state_dict(), metadata=_metadata())
    _patch_fake_builder(monkeypatch)
    config = _config(checkpoint, payload_hash)
    config["model"]["teacher"]["pretrained"]["weights_source_url"] = source_url

    _teacher, provenance = q3d_pretrained.build_q3d_teacher(config)
    assert provenance["checkpoint"]["weights_source_url"] == source_url


def test_non_https_checkpoint_source_is_rejected(tmp_path):
    checkpoint = tmp_path / "public-mednext.pt"
    payload_hash = _write_payload(checkpoint, _FakeMedNeXtState().state_dict(), metadata=_metadata())
    config = _config(checkpoint, payload_hash)
    config["model"]["teacher"]["pretrained"]["weights_source_url"] = "http://example.org/mednext.pt"

    with pytest.raises(ValueError, match="public HTTPS URL"):
        q3d_pretrained.build_q3d_teacher(config)
