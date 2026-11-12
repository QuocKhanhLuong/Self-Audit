"""Software contracts for the Q-3D nnFoundationCNN ResEnc-L teacher."""

from __future__ import annotations

import copy
from pathlib import Path

import pytest
import torch

from self_audit.candidate_q3d.config import canonical_config
from self_audit.candidate_q3d.pretrained import _require_phase_measurement, build_q3d_teacher
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


def test_synthetic_teacher_keeps_explicit_custom_stride_schedule() -> None:
    config = _synthetic_config()
    custom_strides = [
        [1, 1, 1], [1, 2, 2], [1, 2, 2], [1, 2, 2], [1, 2, 2], [1, 2, 2],
    ]
    config["model"]["teacher"]["encoder_strides"] = custom_strides

    teacher, provenance = build_q3d_teacher(config, synthetic=True)

    assert [list(value) for value in teacher.encoder_strides] == custom_strides
    assert provenance["synthetic"] is True
    assert provenance["weight_inventory"]["evidence_scope"] == "SOFTWARE_ONLY"


def test_real_provenance_binds_geometry_repaired_stride_and_erf_receipt() -> None:
    config = _real_config()
    teacher = config["model"]["teacher"]
    measurement = _require_phase_measurement(teacher)

    assert measurement["receipt_path"] == "reports/q3d-resenc-pixdim-20261112/erf_probe.json"
    assert measurement["receipt_sha256"] == (
        "36f44d7f85cd3872168b01339353fb17787099d309e167ea5d64ffbc1721b72b"
    )
    assert measurement["encoder_stride8"]["measured_mean_offset_zyx"] == [
        0.6737596988677979, 1.7588436603546143, 1.2107672691345215,
    ]

    changed_schedule = copy.deepcopy(teacher)
    changed_schedule["encoder_strides"][1] = [2, 2, 2]
    with pytest.raises(ValueError, match="locked geometry-derived Phase-0 schedule"):
        _require_phase_measurement(changed_schedule)

    stale_measurement = copy.deepcopy(teacher)
    stale_measurement["phase_measurement"]["encoder_stride8"][
        "measured_mean_offset_zyx"
    ] = [0.67213374376297, 1.9470438957214355, 1.722203016281128]
    with pytest.raises(ValueError, match="phase contract"):
        _require_phase_measurement(stale_measurement)


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
        "36f44d7f85cd3872168b01339353fb17787099d309e167ea5d64ffbc1721b72b"
    )


def test_synthetic_toggle_does_not_mutate_real_provenance() -> None:
    config = _synthetic_config()
    before = copy.deepcopy(config["model"]["teacher"]["pretrained"])
    build_q3d_teacher(config, synthetic=True)
    assert config["model"]["teacher"]["pretrained"] == before
