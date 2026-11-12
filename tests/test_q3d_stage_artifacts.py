from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest


@pytest.fixture
def single_torch_thread():
    import torch

    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(previous)


def _synthetic_config_and_fixture(tmp_path: Path):
    from self_audit.candidate_a3d.data import synthetic_fixture
    from self_audit.candidate_q3d.config import canonical_config, validate_config
    from scripts.smoke_q3d import _manufacture_cine_frames

    fixture = synthetic_fixture(
        tmp_path / "fixture", seed=19, n_labeled=2, n_unlabeled=2, n_evaluation=1,
        shape_tzyx=(2, 8, 32, 32),
    )
    fixture["train_sha256"] = _manufacture_cine_frames(fixture["train"])
    fixture["evaluation_sha256"] = _manufacture_cine_frames(fixture["evaluation"])

    config = canonical_config("v1", dataset="acdc", protocol="A", k=2, seed=19)
    config["runtime"].update({
        "synthetic": True,
        "device": "cpu",
        "smoke_update_overrides": {"T": 1, "Q": 1, "P": 1, "J": 1},
        "smoke_initialization_volumes": 8,
        "synthetic_gate_override": True,
    })
    config["dataset"]["manifests"]["train"] = {
        "path": str(fixture["train"].resolve()),
        "sha256": fixture["train_sha256"],
        "status": "READY",
    }
    config["dataset"]["manifests"]["development"] = {
        "path": str(fixture["evaluation"].resolve()),
        "sha256": fixture["evaluation_sha256"],
        "status": "READY",
    }
    return validate_config(config), fixture


def test_stage_identity_and_source_configs_are_arm_independent(tmp_path):
    import torch

    from self_audit.candidate_q3d.config import canonical_config, validate_config
    from self_audit.candidate_q3d.engine import _data_identity_without_labels, _load_data
    from self_audit.candidate_q3d.stages import stage_identity, stage_source_config

    config, _fixture = _synthetic_config_and_fixture(tmp_path)
    data = _load_data(config, None)
    identity = stage_identity(config, _data_identity_without_labels(data))
    assert identity["dataset"] == "acdc"
    assert identity["protocol"] == "A"
    assert identity["k"] == 2
    assert identity["seed"] == 19
    assert set(identity) == {"dataset", "protocol", "k", "seed"}
    base_t = stage_source_config(config, "T")
    assert base_t["data_identity"]["runtime_synthetic"] is True
    assert base_t["data_identity"]["train_manifest_sha256"] == config["dataset"]["manifests"]["train"]["sha256"]
    assert base_t["data_identity"]["development_manifest_sha256"] == config["dataset"]["manifests"]["development"]["sha256"]

    def arm_config(arm, **kwargs):
        result = canonical_config(arm, dataset="acdc", protocol="A", k=2, seed=19, **kwargs)
        result["runtime"] = copy.deepcopy(config["runtime"])
        if arm == "c1":
            result["runtime"]["smoke_initialization_volumes"] = None
        result["dataset"]["manifests"] = copy.deepcopy(config["dataset"]["manifests"])
        return validate_config(result)

    shared_arms = {
        arm: arm_config(arm)
        for arm in ("v0", "b1", "e1", "c1", "v1", "v1u")
    }
    shared_arms["v1_residual2"] = arm_config("v1", residual2=True)
    for arm_cfg in shared_arms.values():
        assert stage_source_config(config, "T") == stage_source_config(arm_cfg, "T")

    rng_before_source_config = torch.get_rng_state().clone()
    producer_q = stage_source_config(config, "Q")
    assert torch.equal(rng_before_source_config, torch.get_rng_state())
    for arm in ("v0", "b1", "e1", "v1u", "v1_residual2"):
        assert producer_q == stage_source_config(shared_arms[arm], "Q")
    c1_q_consumer = stage_source_config(shared_arms["c1"], "Q")
    c1_q_consumer["tokenizer"].pop("synthetic_initialization_volumes")
    producer_q_without_init = copy.deepcopy(producer_q)
    producer_q_without_init["tokenizer"].pop("synthetic_initialization_volumes")
    assert producer_q_without_init == c1_q_consumer
    for arm in ("b1", "e1", "c1", "v1u", "v1_residual2"):
        assert stage_source_config(config, "P") == stage_source_config(shared_arms[arm], "P")

    o1 = arm_config("o1")
    s1 = arm_config("s1")
    stride4 = arm_config("v1", source_stride=4)
    decoder = arm_config("v1", feature_source="decoder_stride8")
    for arm_cfg in (o1, s1, stride4, decoder):
        assert stage_source_config(config, "T") == stage_source_config(arm_cfg, "T")
        assert stage_source_config(config, "Q") != stage_source_config(arm_cfg, "Q")
        assert stage_source_config(config, "P") != stage_source_config(arm_cfg, "P")

    changed = copy.deepcopy(config)
    changed["runtime"]["synthetic"] = False
    with pytest.raises(ValueError, match="data identity|synthetic"):
        # Dataset byte identities remain bound even when the evidence mode is changed.
        stage_identity(changed, _data_identity_without_labels(data))


def test_tqp_artifacts_are_shared_by_v1_and_v1u_joint_stages(tmp_path, monkeypatch, single_torch_thread):
    import self_audit.candidate_q3d.engine as engine
    from self_audit.candidate_q3d.config import canonical_config, validate_config
    from self_audit.candidate_q3d.stages import import_stage_artifacts, _state_mapping_sha256
    from train import _manufactured_teacher

    config, _fixture = _synthetic_config_and_fixture(tmp_path)

    t_result = engine.run_training(
        config, tmp_path / "source-T", device="cpu", smoke=True,
        teacher=_manufactured_teacher(config), stage="T",
        synthetic_gate_override=True,
    )
    t_artifacts = {"T": t_result["stage_artifacts"]["T"]}
    q_result = engine.run_training(
        config, tmp_path / "source-Q", device="cpu", smoke=True,
        teacher=_manufactured_teacher(config), stage="Q", stage_artifacts=t_artifacts,
        synthetic_gate_override=True,
    )
    tq_artifacts = {
        "T": q_result["stage_artifacts"]["T"],
        "Q": q_result["stage_artifacts"]["Q"],
    }
    p_result = engine.run_training(
        config, tmp_path / "source-P", device="cpu", smoke=True,
        teacher=_manufactured_teacher(config), stage="P", stage_artifacts=tq_artifacts,
        synthetic_gate_override=True,
    )
    shared_artifacts = {
        stage: p_result["stage_artifacts"][stage] for stage in ("T", "Q", "P")
    }
    assert [Path(shared_artifacts[stage]["manifest_path"]).name for stage in ("T", "Q", "P")] == [
        "T.artifact.json", "Q.artifact.json", "P.artifact.json",
    ]

    def unexpected_stage(*_args, **_kwargs):
        pytest.fail("a completed shared T/Q stage was rerun during J")

    monkeypatch.setattr(engine, "_run_teacher_stage", unexpected_stage)
    monkeypatch.setattr(engine, "_run_tokenizer_stage", unexpected_stage)

    joint_results = {}
    joint_teachers = {}
    for arm in ("v1", "v1u", "c1"):
        arm_config = copy.deepcopy(config)
        arm_config["arm"] = arm
        # Canonicalization supplies the arm-specific continuous weighting mode
        # while preserving the exact synthetic manifests and bounded schedule.
        arm_config = canonical_config(arm, dataset="acdc", protocol="A", k=2, seed=19)
        arm_config["runtime"].update(copy.deepcopy(config["runtime"]))
        if arm == "c1":
            arm_config["runtime"]["smoke_initialization_volumes"] = None
        arm_config["dataset"]["manifests"] = copy.deepcopy(config["dataset"]["manifests"])
        arm_config = validate_config(arm_config)
        arm_teacher = _manufactured_teacher(arm_config)
        result = engine.run_training(
            arm_config, tmp_path / f"joint-{arm}", device="cpu", smoke=True,
            teacher=arm_teacher, stage="J",
            stage_artifacts=shared_artifacts, synthetic_gate_override=True,
        )
        assert result["status"] == "COMPLETED"
        assert result["joint_optimizer_updates"] == 1
        joint_results[arm] = result
        joint_teachers[arm] = arm_teacher
    monkeypatch.undo()

    assert joint_results["c1"]["tokenizer"] is None
    assert joint_results["c1"]["stage_artifact_hashes"]["Q"] == joint_results["v1"]["stage_artifact_hashes"]["Q"]

    import torch

    resumed_state = torch.load(joint_results["v1"]["checkpoint"], map_location="cpu", weights_only=False)
    saved_student_state = copy.deepcopy(resumed_state["student_state"])
    saved_completed_stages = copy.deepcopy(resumed_state["completed_stages"])
    engine_data = engine._load_data(config, None)
    import_stage_artifacts(
        config, engine_data, resumed_state, joint_teachers["v1"], None,
        tmp_path / "joint-v1", {"T": shared_artifacts["T"]},
    )
    assert _state_mapping_sha256(resumed_state["student_state"]) == _state_mapping_sha256(saved_student_state)
    assert resumed_state["completed_stages"] == saved_completed_stages
    assert set(resumed_state["stage_artifacts"]) == {"T", "Q", "P"}

    for stage in ("T", "Q", "P"):
        first = joint_results["v1"]["stage_artifacts"][stage]
        second = joint_results["v1u"]["stage_artifacts"][stage]
        assert first["manifest_sha256"] == second["manifest_sha256"]
        assert first["checkpoint_sha256"] == second["checkpoint_sha256"]
        assert first["manifest_path"] == second["manifest_path"]

    # A standalone C1 run has no tokenizer or Q stage, so its physiology
    # artifact is valid with the T dependency alone.
    c1_config = canonical_config("c1", dataset="acdc", protocol="A", k=2, seed=19)
    c1_config["runtime"].update(copy.deepcopy(config["runtime"]))
    c1_config["runtime"]["smoke_initialization_volumes"] = None
    c1_config["dataset"]["manifests"] = copy.deepcopy(config["dataset"]["manifests"])
    c1_config = validate_config(c1_config)
    c1_result = engine.run_training(
        c1_config, tmp_path / "c1-standalone", device="cpu", smoke=True,
        teacher=_manufactured_teacher(c1_config), synthetic_gate_override=True,
    )
    assert c1_result["status"] == "COMPLETED"
    assert c1_result["tokenizer"] is None
    p_document = json.loads(Path(c1_result["stage_artifacts"]["P"]["manifest_path"]).read_text())
    assert set(p_document["dependencies"]) == {"T"}

    data = engine._load_data(config, None)
    fresh_state = {
        "completed_stages": [], "student_state": None, "saved_at": None,
        "stage_artifacts": {}, "stage_artifact_hashes": {},
        "synthetic_gate_override_requested": True,
    }
    with pytest.raises(ValueError, match="requires the matching Q artifact"):
        import_stage_artifacts(
            config, data, fresh_state, _manufactured_teacher(config), None,
            tmp_path / "missing-q-probe", {"T": shared_artifacts["T"], "P": shared_artifacts["P"]},
        )

    # Editing a sealed checkpoint without rebuilding its manifest is rejected
    # before any stage state can be imported or a training stage can run.
    t_checkpoint = Path(shared_artifacts["T"]["checkpoint_path"])
    with t_checkpoint.open("ab") as stream:
        stream.write(b"tampered")
    arm_config = canonical_config("v1", dataset="acdc", protocol="A", k=2, seed=19)
    arm_config["runtime"].update(copy.deepcopy(config["runtime"]))
    arm_config["dataset"]["manifests"] = copy.deepcopy(config["dataset"]["manifests"])
    arm_config = validate_config(arm_config)
    data = engine._load_data(arm_config, None)
    fresh_state = {
        "completed_stages": [], "student_state": None, "saved_at": None,
        "stage_artifacts": {}, "stage_artifact_hashes": {},
        "synthetic_gate_override_requested": True,
    }
    with pytest.raises(ValueError, match="checkpoint file SHA-256 mismatch"):
        import_stage_artifacts(
            arm_config, data, fresh_state, _manufactured_teacher(arm_config), None,
            tmp_path / "tamper-probe", shared_artifacts,
        )


def test_stage_export_recovers_orphans_and_adopts_existing_manifest(tmp_path, single_torch_thread):
    import torch

    import self_audit.candidate_q3d.engine as engine
    from self_audit.candidate_q3d.stages import export_stage_artifact
    from train import _manufactured_teacher

    config, _fixture = _synthetic_config_and_fixture(tmp_path)
    teacher = _manufactured_teacher(config)
    output = tmp_path / "recoverable-T"
    result = engine.run_training(
        config, output, device="cpu", smoke=True, teacher=teacher, stage="T",
        synthetic_gate_override=True,
    )
    data = engine._load_data(config, None)
    state = torch.load(result["checkpoint"], map_location="cpu", weights_only=False)
    # Simulate a crash after the immutable manifest was published but before
    # the root latest.pt checkpoint recorded its reference.
    state["stage_artifacts"] = {}
    state["stage_artifact_hashes"] = {}
    adopted = export_stage_artifact(config, data, state, teacher, None, output, "T")
    assert adopted["manifest_sha256"] == result["stage_artifacts"]["T"]["manifest_sha256"]
    assert adopted["checkpoint_sha256"] == result["stage_artifacts"]["T"]["checkpoint_sha256"]

    # Simulate a crash after checkpoint bytes were saved but before the
    # manifest was created. Recovery publishes a new file and preserves the
    # orphan bytes at their original path.
    orphan_output = tmp_path / "orphan-T"
    orphan_path = orphan_output / "stage_artifacts" / "T" / "T.checkpoint.pt"
    orphan_path.parent.mkdir(parents=True)
    orphan_path.write_bytes(Path(result["stage_artifacts"]["T"]["checkpoint_path"]).read_bytes())
    recovered_state = copy.deepcopy(state)
    recovered_state["stage_artifacts"] = {}
    recovered_state["stage_artifact_hashes"] = {}
    orphan_bytes = orphan_path.read_bytes()
    recovered = export_stage_artifact(config, data, recovered_state, teacher, None, orphan_output, "T")
    assert Path(recovered["checkpoint_path"]) != orphan_path
    assert orphan_path.read_bytes() == orphan_bytes
    assert Path(recovered["manifest_path"]).is_file()
