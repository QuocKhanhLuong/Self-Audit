"""Focused ResEnc teacher schedule and reporting contracts."""

from __future__ import annotations

import pytest
import torch

from self_audit.candidate_q3d.config import canonical_config
from self_audit.candidate_q3d.engine import (
    _log_teacher_development,
    _make_teacher_optimizer,
    _teacher_curve_status,
    _teacher_parameter_groups,
    _teacher_stage,
)
from self_audit.candidate_q3d.teacher_resenc import Q3DResEncTeacher


def _config() -> dict:
    config = canonical_config("v1", dataset="acdc", protocol="A", k=8, seed=42)
    config["runtime"]["synthetic"] = True
    config["training"]["teacher_warmup_fraction"] = 0.05
    config["training"]["teacher_encoder_lr"] = 1.0e-4
    config["training"]["teacher_head_lr"] = 1.0e-3
    config["training"]["weight_decay"] = 1.0e-4
    return config


def test_resenc_optimizer_groups_are_disjoint_despite_decoder_encoder_alias() -> None:
    teacher = Q3DResEncTeacher(synthetic=True)
    config = _config()
    groups = _teacher_parameter_groups(teacher, config, multiplier=1.0)

    encoder_ids = {id(parameter) for parameter in teacher.encoder.parameters()}
    group_ids = [{id(parameter) for parameter in group["params"]} for group in groups]
    assert {group["name"] for group in groups} == {"teacher_trunk", "teacher_heads"}
    assert group_ids[0].isdisjoint(group_ids[1])
    assert group_ids[0] == encoder_ids
    assert group_ids[1].isdisjoint(encoder_ids)
    assert {group["name"]: group["lr"] for group in groups} == {
        "teacher_trunk": 1.0e-4,
        "teacher_heads": 1.0e-3,
    }


def test_teacher_a_freezes_encoder_and_keeps_decoder_heads_trainable() -> None:
    teacher = Q3DResEncTeacher(synthetic=True)
    _teacher_stage(teacher, "warmup", freeze=False)
    encoder_ids = {id(parameter) for parameter in teacher.encoder.parameters()}
    assert all(not parameter.requires_grad for parameter in teacher.encoder.parameters())
    head_parameters = [
        parameter for parameter in teacher.parameters() if id(parameter) not in encoder_ids
    ]
    assert head_parameters and all(parameter.requires_grad for parameter in head_parameters)
    # ``region_decoder.encoder`` aliases the encoder, so the recursive train()
    # call must not leave the frozen encoder in training mode.
    assert teacher.region_decoder.training is True
    assert teacher.encoder.training is False

    warmup = _make_teacher_optimizer(teacher, _config(), warmup=True)
    warmup_ids = {id(parameter) for group in warmup.param_groups for parameter in group["params"]}
    assert warmup_ids == {id(parameter) for parameter in head_parameters}
    assert len(warmup.param_groups) == 1
    assert warmup.param_groups[0]["lr"] == 1.0e-3


def test_teacher_b_restores_encoder_and_uses_two_lr_groups() -> None:
    teacher = Q3DResEncTeacher(synthetic=True)
    _teacher_stage(teacher, "warmup", freeze=False)
    _teacher_stage(teacher, "pretrain", freeze=False)
    assert all(parameter.requires_grad for parameter in teacher.parameters())
    assert teacher.encoder.training is True
    groups = _teacher_parameter_groups(teacher, _config(), multiplier=1.0)
    assert {group["name"]: group["lr"] for group in groups} == {
        "teacher_trunk": 1.0e-4,
        "teacher_heads": 1.0e-3,
    }


def test_teacher_b_warmup_fraction_is_part_of_t_source_seal() -> None:
    from self_audit.candidate_q3d.stages import stage_source_config

    config = _config()
    source = stage_source_config(config, "T")
    assert source["teacher_training"]["warmup_fraction"] == 0.05
    changed = {**config, "training": {**config["training"], "warmup_fraction": 0.2}}
    assert stage_source_config(changed, "T") != source


def test_teacher_curve_requires_three_strictly_increasing_actual_points() -> None:
    assert _teacher_curve_status([]) == {"status": "INSUFFICIENT", "value": None, "points": []}
    assert _teacher_curve_status([
        {"update": 400, "score": 0.2}, {"update": 800, "score": 0.3},
    ])["status"] == "INSUFFICIENT"
    measured = _teacher_curve_status([
        {"update": 400, "score": 0.2}, {"update": 800, "score": 0.3},
        {"update": 1200, "score": 0.4}, {"update": 1600, "score": 0.4},
    ])
    assert measured["status"] == "MEASURED"
    assert measured["value"] is False
    assert measured["points"] == [
        {"update": 800, "score": 0.3}, {"update": 1200, "score": 0.4},
        {"update": 1600, "score": 0.4},
    ]


def test_teacher_development_log_is_sparse_and_per_class_on_stderr(capsys) -> None:
    _log_teacher_development(
        {"dataset": {"name": "acdc"}, "protocol": {"variant": "A", "k": 8}, "seed": 42},
        update=400,
        update_budget=8000,
        train_loss=1.25,
        dev={"score": 0.61, "per_class": {"RV": 0.5, "MYO": 0.6, "LV": 0.7}},
    )
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "T update=400/8000" in captured.err
    assert "RV=0.500000 MYO=0.600000 LV=0.700000" in captured.err


def test_teacher_quality_is_sealed_in_t_manifest(tmp_path) -> None:
    from self_audit.candidate_q3d.artifacts import load_stage_manifest, write_stage_manifest

    checkpoint = tmp_path / "T.pt"
    torch.save({
        "schema": "self_audit.candidate_q3d.resume.v3",
        "teacher_dev_fg_dice": 0.61,
        "teacher_dev_per_class": {"RV": 0.5, "MYO": 0.6, "LV": 0.7},
        "teacher_selected_dev_update": 800,
        "teacher_terminal_dev_fg_dice": 0.60,
        "teacher_terminal_dev_per_class": {"RV": 0.49, "MYO": 0.59, "LV": 0.69},
        "dev_curve_still_improving": {"status": "MEASURED", "value": False, "points": []},
    }, checkpoint)
    identity = {"dataset": "acdc", "protocol": "A", "k": 8, "seed": 42}
    source = {"stage": "T", "identity": identity}
    content = {"teacher_state_sha256": "0" * 64}
    manifest_path = tmp_path / "T.artifact.json"
    write_stage_manifest(
        manifest_path,
        stage="T",
        checkpoint_path=checkpoint,
        identity=identity,
        source_config=source,
        dependencies={},
        content_hashes=content,
    )
    loaded = load_stage_manifest(manifest_path, identity=identity, source_config=source)
    assert loaded["teacher_quality"]["teacher_dev_fg_dice"] == 0.61
    assert loaded["content_hashes"]["teacher_quality_sha256"]


def test_published_t_without_quality_fails_closed(tmp_path) -> None:
    from self_audit.candidate_q3d.artifacts import write_stage_manifest

    checkpoint = tmp_path / "published-T.pt"
    torch.save({
        "schema": "self_audit.candidate_q3d.resume.v3",
        "stage_artifact_stage": "T",
    }, checkpoint)
    identity = {"dataset": "acdc", "protocol": "A", "k": 8, "seed": 42}
    with pytest.raises(ValueError, match="teacher-quality fields"):
        write_stage_manifest(
            tmp_path / "T.artifact.json",
            stage="T",
            checkpoint_path=checkpoint,
            identity=identity,
            source_config={"stage": "T", "identity": identity},
            dependencies={},
            content_hashes={"teacher_state_sha256": "0" * 64},
        )

