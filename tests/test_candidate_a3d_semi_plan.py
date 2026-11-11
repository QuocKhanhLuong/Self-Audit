"""A-3D semi-supervised plan and checkpoint provenance contracts."""

import json
from pathlib import Path

import pytest
import torch

from scripts.candidate_a_ablations import create_plan
from scripts.candidate_a_fewshot import create_fewshot_plan
from self_audit.candidate_a3d.config import validate_config
from self_audit.candidate_a3d.data import CandidateA3DData, synthetic_fixture
from self_audit.candidate_a3d.io import export_student, read_checkpoint, save_checkpoint
from self_audit.model import digest


def _semi_file(path: Path) -> Path:
    path.write_text(json.dumps({
        "semi_supervised": {
            "enabled": True,
            "ema_decay": 0.99,
            "teacher_weight": 1.0,
            "student_weight": 1.0,
            "rampup_epochs": 10,
            "confidence_threshold": 0.7,
            "background_weight": 0.1,
            "steps_per_epoch": 0,
            "intensity_gamma": [0.7, 1.5],
            "intensity_noise_std": 0.03,
        }
    }, indent=2) + "\n")
    return path


def test_a3d_plan_pins_one_semi_block_on_every_arm(tmp_path):
    fixture = synthetic_fixture(
        tmp_path / "data", n_labeled=2, n_unlabeled=1, n_evaluation=1,
        shape_tzyx=(1, 2, 4, 5),
    )
    semi = _semi_file(tmp_path / "semi.json")
    plan_path = create_plan(
        track="candidate_a3d", dataset="acdc", train_manifest=fixture["train"],
        development_manifest=fixture["evaluation"], model_config=None,
        out=tmp_path / "plan", seeds=[42], arms=["full", "teacher_free"],
        epochs=1, pretrain_epochs=1, warmup_epochs=1, synthetic=True,
        semi_config=semi,
    )
    plan = json.loads(plan_path.read_text())
    assert plan["semi_config"] == {"path": str(semi.resolve()), "sha256": digest(semi)}
    block = plan["semi_supervised"]
    assert block["enabled"] is True
    for job in plan["jobs"]:
        recipe = json.loads(Path(job["config"]).read_text())
        assert recipe["semi_supervised"] == block


def test_semi_config_is_rejected_for_candidate_a(tmp_path):
    fixture = synthetic_fixture(
        tmp_path / "data", n_labeled=1, n_unlabeled=0, n_evaluation=1,
        shape_tzyx=(1, 2, 4, 5),
    )
    semi = _semi_file(tmp_path / "semi.json")
    with pytest.raises(ValueError, match="candidate_a3d"):
        create_plan(
            track="candidate_a", dataset="acdc", train_manifest=fixture["train"],
            development_manifest=fixture["evaluation"], model_config=None,
            out=tmp_path / "plan", seeds=[42], arms=["full"],
            epochs=1, pretrain_epochs=1, warmup_epochs=1, synthetic=True,
            semi_config=semi,
        )


def test_fewshot_root_and_child_plans_pin_semi_block(tmp_path):
    fixture = synthetic_fixture(
        tmp_path / "data", n_labeled=10, n_unlabeled=1, n_evaluation=1,
        shape_tzyx=(1, 2, 4, 5),
    )
    train = Path(fixture["train"])
    payload = json.loads(train.read_text())
    payload["dataset"] = "acdc"
    train.write_text(json.dumps(payload, indent=2) + "\n")
    semi = _semi_file(tmp_path / "semi.json")
    root_path = create_fewshot_plan(
        dataset="acdc", train_manifest=train, development_manifest=fixture["evaluation"],
        model_config=None, out=tmp_path / "campaign", synthetic=True, semi_config=semi,
    )
    root = json.loads(root_path.read_text())
    assert root["semi_config"]["sha256"] == digest(semi)
    assert root["semi_supervised"]["enabled"] is True
    for child in root["child_plans"]:
        child_plan = json.loads(Path(child["plan"]).read_text())
        assert child_plan["semi_supervised"] == root["semi_supervised"]


def test_checkpoint_keeps_ema_state_but_export_drops_it(tmp_path):
    fixture = synthetic_fixture(
        tmp_path / "data", n_labeled=2, n_unlabeled=1, n_evaluation=1,
        shape_tzyx=(1, 2, 4, 5),
    )
    data = CandidateA3DData(fixture["train"], fixture["train_sha256"], synthetic=True)
    config = json.loads(Path("configs/candidate_a3d/smoke.json").read_text())
    config["semi_supervised"] = {
        "enabled": True,
        "ema_decay": 0.99,
        "teacher_weight": 1.0,
        "student_weight": 1.0,
        "rampup_epochs": 10,
        "confidence_threshold": 0.7,
        "background_weight": 0.1,
        "steps_per_epoch": 0,
        "intensity_gamma": [0.7, 1.5],
        "intensity_noise_std": 0.03,
    }
    config = validate_config(config)
    from self_audit.candidate_a3d.io import build_student
    student, _, _ = build_student(config)
    checkpoint = tmp_path / "checkpoint"
    manifest_sha = save_checkpoint(
        checkpoint, student=student, teacher=None, optimizers=[], config=config,
        data=data, epoch=1, updates=1, provenance={},
        training_state={"student_ema": student.state_dict(), "teacher_ema": None},
    )
    payload, metadata = read_checkpoint(checkpoint, manifest_sha)
    assert set(payload["training_state"]) == {"student_ema", "teacher_ema"}
    assert metadata["supervision"].startswith("dense labeled TRAIN subset plus")
    assert metadata["unlabeled_usage"] == "image_only_consistency_no_masks"
    exported = tmp_path / "export"
    export_student(checkpoint, manifest_sha, exported)
    export_payload = torch.load(exported / "weights.pt", map_location="cpu", weights_only=True)
    assert set(export_payload) == {"student"}

