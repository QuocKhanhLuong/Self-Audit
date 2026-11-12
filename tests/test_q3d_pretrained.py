"""Software contracts for the Q-3D nnFoundationCNN ResEnc-L teacher."""

from __future__ import annotations

import copy
from pathlib import Path

import pytest
import torch

from self_audit.candidate_q3d.config import canonical_config
from self_audit.candidate_q3d.pretrained import build_q3d_teacher
from self_audit.candidate_q3d.teacher_resenc import Q3DResEncTeacher

CHECKPOINT = Path("/Users/alvinluong/Self-Audit-runs/q3d-resenc-20261112/checkpoint_final.pth")


def _synthetic_config():
    config = canonical_config("v1", k=8, seed=42)
    config["runtime"].update({
        "synthetic": True,
        "smoke_update_overrides": {"T": 1, "Q": 1, "P": 1, "J": 1},
        "smoke_initialization_volumes": 1,
        "synthetic_gate_override": True,
    })
    return config


def _real_config():
    config = canonical_config("v1", k=8, seed=42)
    config["model"]["teacher"]["pretrained"]["weights_path"] = str(CHECKPOINT)
    return config


def test_synthetic_teacher_uses_resenc_class_and_same_seed_state() -> None:
    first, first_provenance = build_q3d_teacher(_synthetic_config(), synthetic=True)
    second, second_provenance = build_q3d_teacher(_synthetic_config(), synthetic=True)

    assert isinstance(first, Q3DResEncTeacher)
    assert isinstance(second, Q3DResEncTeacher)
    assert type(first) is type(second)
    assert first_provenance["architecture"] == "nnfoundation_resenc_l"
    assert first_provenance["weight_inventory"]["source"] == "synthetic_resenc_fixture"
    assert second_provenance["weight_inventory"]["source"] == "synthetic_resenc_fixture"
    assert first.state_dict().keys() == second.state_dict().keys()
    assert all(torch.equal(first.state_dict()[key], second.state_dict()[key]) for key in first.state_dict())


def test_real_hash_mismatch_fails_before_loading() -> None:
    if not CHECKPOINT.is_file():
        pytest.skip("Phase-0 checkpoint is external to the repository")
    config = _real_config()
    config["model"]["teacher"]["pretrained"]["weights_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="Phase-0 checkpoint hash"):
        build_q3d_teacher(config)


def test_real_phase_contract_rejects_unmeasured_stride16_inheritance() -> None:
    config = _real_config()
    config["model"]["teacher"]["phase_offsets"]["encoder_stride8"] = [1.0, 0.0, 0.0]
    with pytest.raises(ValueError, match="phase_offsets"):
        build_q3d_teacher(config)


def test_real_weight_binding_is_required_for_portable_recipe(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("Q3D_NNFOUNDATION_WEIGHTS", raising=False)
    with pytest.raises(ValueError, match="Q3D_NNFOUNDATION_WEIGHTS"):
        build_q3d_teacher(canonical_config("v1", k=8, seed=42))


def test_real_checkpoint_loads_strict_encoder_only_when_phase0_weight_is_present() -> None:
    if not CHECKPOINT.is_file():
        pytest.skip("Phase-0 checkpoint is external to the repository")
    teacher, provenance = build_q3d_teacher(_real_config())

    assert isinstance(teacher, Q3DResEncTeacher)
    assert len(teacher.encoder.state_dict()) == 448
    inventory = provenance["weight_inventory"]
    assert inventory["loaded_key_count"] == 448
    assert inventory["missing_keys"] == []
    assert inventory["unexpected_keys"] == []
    assert provenance["checkpoint"]["hf_revision"] == "f443c09319988e833b3d9589bf95909e4443d7b2"
    assert provenance["backbone"]["encoder_load"]["decoder_loaded"] is False
    assert provenance["backbone"]["encoder_load"]["segmentation_head_loaded"] is False
    assert inventory["loaded_alias_key_count"] == 448
    assert inventory["region_decoder_random_keys"]
    assert all(key.startswith("region_decoder.encoder.") for key in inventory["loaded_alias_keys"])
    assert all(not key.startswith("region_decoder.encoder.") for key in inventory["region_decoder_random_keys"])
    assert provenance["backbone"]["phase_measurement"]["receipt_sha256"] == (
        "2f05888b45ff3bfbf19f90bd91abbd4f0b2ab193b37375c5d875950117d2a14e"
    )


def test_synthetic_toggle_does_not_mutate_real_provenance() -> None:
    config = _synthetic_config()
    before = copy.deepcopy(config["model"]["teacher"]["pretrained"])
    build_q3d_teacher(config, synthetic=True)
    assert config["model"]["teacher"]["pretrained"] == before
