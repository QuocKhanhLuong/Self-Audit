from __future__ import annotations

import copy
import contextlib
import hashlib
import io
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from scripts import q3d_campaign
from scripts.q3d_campaign import create_campaign_plan, execute_campaign, validate_campaign_plan
from self_audit.candidate_q3d.config import (
    K8_LOCKED_DEV_MANIFEST_SHA256,
    K8_LOCKED_TRAIN_MANIFEST_SHA256,
    NNFOUNDATION_DEFAULT_ENCODER_STRIDES,
    NNFOUNDATION_PHASE_RECEIPT_PATH,
    NNFOUNDATION_PHASE_RECEIPT_SHA256,
    NNFOUNDATION_SIGNED_MEAN_OFFSET_ZYX,
    PRIMARY_ARMS,
    SECONDARY_ARMS,
    canonical_config,
    load_config,
    resolve_stage_updates,
    validate_budget_update_equality,
    validate_config,
    validate_real_teacher_provenance,
)
from self_audit.candidate_a3d.data import CandidateA3DData
from self_audit.candidate_q3d.data import Q3DData


def test_canonical_json_recipes_are_strict_and_preserve_pinned_settings() -> None:
    config_dir = Path("configs/candidate_q3d")
    names = {
        "c0", "c1", "v1", "v1u", "v0", "b1", "e1", "o1", "s1",
        "v1_residual2", "mean_teacher", "a3d_teacher_free", "teacher_supervised",
    }
    assert {path.stem for path in config_dir.glob("*.json")} == names
    for path in sorted(config_dir.glob("*.json")):
        config = load_config(path)
        assert config["schema"] == "self_audit.candidate_q3d.v1"
        assert config["seed"] == 42
        assert config["protocol"]["variant"] == "A"
        assert config["protocol"]["k"] == 5
        assert config["protocol"]["cache_isolation_key"] == "q3d/acdc/protocol-a/k5/seed-42"
        assert config["dataset"]["deferred_cmr_datasets"] == [
            {"name": None, "status": "NOT_RUN", "reason": "dataset identity not specified in Q-3D spec"},
            {"name": None, "status": "NOT_RUN", "reason": "dataset identity not specified in Q-3D spec"},
        ]
        assert [item["dataset"] for item in config["dataset"]["cross_evaluation"][:2]] == ["acdc", "mnms"]
        assert config["model"]["student"]["residual2"]["zero_init"] is True
        assert config["training"]["stage_updates"] == {"T": 8000, "Q": 1000, "P": 1, "J": 8000}
        assert config["training"]["stage_optimizer_updates"] == {"T": 8000, "Q": 1000, "P": 0, "J": 8000}
        assert config["tokenizer"]["initialization_volumes"] == 200
        assert config["augmentation"]["intensity_probability_unlabeled_student"] == 1.0
        if config["model"]["teacher"]["architecture"] == "nnfoundation_resenc_l":
            assert config["model"]["teacher"]["encoder_strides"] == NNFOUNDATION_DEFAULT_ENCODER_STRIDES
            assert config["model"]["teacher"]["phase_offsets"] == {
                "encoder_stride8": [0.0, 0.0, 0.0],
                "encoder_stride16": [0.0, 0.0, 0.0],
            }
            assert config["model"]["teacher"]["dependency_pins"]["dynamic_network_architectures"] == "0.4.4"
            assert config["model"]["teacher"]["pretrained"]["class_mapping"] == ["BG", "RV", "MYO", "LV"]
            assert "f443c09319988e833b3d9589bf95909e4443d7b2" in config["model"]["teacher"]["pretrained"]["weights_source_url"]
            assert config["model"]["teacher"]["pretrained"]["provenance_status"] == "READY"
            assert config["model"]["teacher"]["pretrained"]["weights_path"] == "${Q3D_NNFOUNDATION_WEIGHTS}"
            assert config["model"]["teacher"]["pretrained"]["overlap_acdc"] is False
            assert config["model"]["teacher"]["pretrained"]["overlap_mnms"] is False
            phase = config["model"]["teacher"]["phase_measurement"]
            assert phase["receipt_path"] == NNFOUNDATION_PHASE_RECEIPT_PATH
            assert phase["receipt_sha256"] == NNFOUNDATION_PHASE_RECEIPT_SHA256
            assert phase["encoder_stride8"]["offset_zyx"] == [0.0, 0.0, 0.0]
            assert phase["encoder_stride8"]["measured_mean_offset_zyx"] == NNFOUNDATION_SIGNED_MEAN_OFFSET_ZYX
            assert phase["encoder_stride16"]["status"] == "NOT_MEASURED"
            assert "not_inherited" in phase["encoder_stride16"]["policy"]
    assert load_config(config_dir / "s1.json")["model"]["teacher"]["feature_stride"] == 16
    assert load_config(config_dir / "v1.json")["model"]["teacher"]["feature_stride"] == 8
    assert load_config(config_dir / "v1.json")["tokenizer"]["source_feature"] == "encoder_stride8"
    assert load_config(config_dir / "s1.json")["model"]["teacher"]["feature_source"] == "encoder_stride16"
    assert load_config(config_dir / "v1_residual2.json")["model"]["student"]["residual2"] == {
        "enabled": True, "zero_init": True,
    }
    v1 = load_config(config_dir / "v1.json")
    assert v1["training"]["stage_update_units"]["J"] == "paired_L_U_updates"
    assert v1["training"]["stage_optimizer_updates"]["J"] == 8000
    assert v1["training"]["stage_stream_microbatches"]["J"] == 16000
    assert v1["training"]["gradient_accumulation"] == 2
    assert v1["training"]["gradient_accumulation_unit"] == "stream_microbatches_per_optimizer_step"
    assert v1["training"]["checkpoint_evaluation_interval"] == 400
    assert v1["training"]["checkpoint_evaluation_unit"] == "paired_L_U_updates"
    assert v1["losses"]["labeled_supervised_coefficient"] == 1.0
    assert v1["losses"]["supervised_divide_by_stream_count"] is False
    assert v1["losses"]["l_u_auxiliary_reduction"] == "mean"
    assert v1["checkpoint"]["resume_scope"] == "stage_checkpoint_only"
    c0 = load_config(config_dir / "c0.json")
    assert c0["training"]["stage_update_units"]["J"] == "L_only_updates"
    assert c0["training"]["stage_optimizer_updates"]["J"] == 8000
    assert c0["training"]["stage_stream_microbatches"]["J"] == 8000
    assert c0["training"]["gradient_accumulation"] == 1
    assert c0["losses"]["labeled_supervised_coefficient"] == 1.0
    assert c0["losses"]["supervised_divide_by_stream_count"] is False
    a3d = load_config(config_dir / "a3d_teacher_free.json")
    assert a3d["training"]["stage_update_units"]["J"] == "L_only_updates"
    assert a3d["training"]["stage_optimizer_updates"]["J"] == 8000
    assert a3d["training"]["stage_stream_microbatches"]["J"] == 8000


def test_q3d_feature_source_selection_is_explicit_and_consistent() -> None:
    main = canonical_config("v1")
    stride16 = canonical_config("s1")

    assert main["model"]["teacher"]["feature_source"] == "encoder_stride8"
    assert main["tokenizer"]["source_feature"] == main["model"]["teacher"]["feature_source"]
    assert main["model"]["teacher"]["feature_channels"] == 256
    assert stride16["model"]["teacher"]["feature_source"] == "encoder_stride16"
    assert stride16["tokenizer"]["source_feature"] == "encoder_stride16"
    assert stride16["model"]["teacher"]["feature_channels"] == 320

    mismatched = copy.deepcopy(main)
    mismatched["tokenizer"]["source_feature"] = "encoder_stride16"
    with pytest.raises(ValueError, match="tokenizer.source_feature"):
        validate_config(mismatched)

    unsupported = copy.deepcopy(main)
    unsupported["model"]["teacher"]["feature_source"] = "encoder_stage8_decoder"
    unsupported["tokenizer"]["source_feature"] = "encoder_stage8_decoder"
    with pytest.raises(ValueError, match="feature_source"):
        validate_config(unsupported)

    with pytest.raises(ValueError, match="feature_stride must be 8 or 16"):
        canonical_config("v1", source_stride=4)


def test_real_resenc_schedule_and_phase_receipt_are_locked_but_synthetic_schedule_is_customizable() -> None:
    config = canonical_config("v1")
    assert config["model"]["teacher"]["encoder_strides"] == NNFOUNDATION_DEFAULT_ENCODER_STRIDES

    altered_real = copy.deepcopy(config)
    altered_real["model"]["teacher"]["encoder_strides"][1] = [2, 2, 2]
    with pytest.raises(ValueError, match="locked geometry-derived Phase-0 schedule"):
        validate_config(altered_real)
    with pytest.raises(ValueError, match="locked geometry-derived Phase-0 encoder_strides"):
        validate_real_teacher_provenance(altered_real)

    altered_measurement = copy.deepcopy(config)
    altered_measurement["model"]["teacher"]["phase_measurement"]["encoder_stride8"][
        "measured_mean_offset_zyx"
    ] = [0.0, 0.0, 0.0]
    with pytest.raises(ValueError, match="locked ERF receipt"):
        validate_config(altered_measurement)

    synthetic = copy.deepcopy(config)
    synthetic["runtime"]["synthetic"] = True
    synthetic_strides = [
        [1, 1, 1], [1, 2, 2], [1, 2, 2], [1, 2, 2], [1, 2, 2], [1, 2, 2],
    ]
    synthetic["model"]["teacher"]["encoder_strides"] = synthetic_strides
    assert validate_config(synthetic)["model"]["teacher"]["encoder_strides"] == synthetic_strides


def test_k8_manifest_constants_bind_the_geometry_repaired_receipts() -> None:
    assert K8_LOCKED_TRAIN_MANIFEST_SHA256 == (
        "60dd11747570f3ac183e8636050a3ac3d73651cb43b37deba9ff7422b442a433"
    )
    assert K8_LOCKED_DEV_MANIFEST_SHA256 == (
        "c83031b6deaf275f830401f8da9ba9a0e30530620633106a3a62cc6d492b1f7b"
    )


def test_config_rejects_unknown_root_and_nested_fields() -> None:
    config = canonical_config("v1")
    extra_root = copy.deepcopy(config)
    extra_root["unreviewed"] = True
    with pytest.raises(ValueError, match="exact fields"):
        validate_config(extra_root)
    extra_budget = copy.deepcopy(config)
    extra_budget["training"]["stage_updates"]["unknown"] = 2
    with pytest.raises(ValueError, match="stage_updates"):
        validate_config(extra_budget)


def test_protocol_and_cache_namespaces_are_isolated() -> None:
    protocol_a = canonical_config("v1", dataset="acdc", protocol="A", k=5, seed=42)
    protocol_b = canonical_config("v1", dataset="acdc", protocol="B", k=5, seed=42)
    mnms = canonical_config("v1", dataset="mnms", protocol="A", k=5, seed=42)
    assert protocol_a["protocol"]["cache_isolation_key"] != protocol_b["protocol"]["cache_isolation_key"]
    assert protocol_a["protocol"]["cache_isolation_key"] != mnms["protocol"]["cache_isolation_key"]
    assert protocol_a["checkpoint"]["stage_outputs"]["T"] != protocol_b["checkpoint"]["stage_outputs"]["T"]
    assert protocol_b["protocol"]["teacher_label_budget"] == "all_train"
    assert protocol_b["protocol"]["student_label_budget"] == "k"


def test_budget_equality_holds_for_primary_arms_and_rejects_drift() -> None:
    configs = [canonical_config(arm, k=10) for arm in PRIMARY_ARMS]
    budgets = validate_budget_update_equality(configs)
    assert budgets["J"] == 8000
    changed = copy.deepcopy(configs[-1])
    changed["training"]["stage_updates"]["J"] = 7999
    changed["training"]["stage_optimizer_updates"]["J"] = 7999
    changed["training"]["stage_stream_microbatches"]["J"] = 15998
    with pytest.raises(ValueError, match="same joint student update budget"):
        validate_budget_update_equality(configs[:-1] + [changed])


def test_smoke_override_is_explicit_and_synthetic_only() -> None:
    config = canonical_config("v1")
    config["runtime"]["synthetic"] = True
    config["runtime"]["smoke_update_overrides"] = {"T": 2, "Q": 1, "P": 1, "J": 4}
    config["runtime"]["smoke_initialization_volumes"] = 4
    config["runtime"]["synthetic_gate_override"] = True
    assert resolve_stage_updates(config, smoke=True) == {"T": 2, "Q": 1, "P": 1, "J": 4}
    validated = validate_config(config)
    assert validated["tokenizer"]["initialization_volumes"] == 200
    assert validated["runtime"]["smoke_initialization_volumes"] == 4
    assert validated["runtime"]["synthetic_gate_override"] is True
    assert validated["training"]["stage_optimizer_updates"]["Q"] == 1
    assert validated["training"]["stage_optimizer_updates"]["J"] == 4
    assert validated["training"]["stage_stream_microbatches"]["J"] == 8
    with pytest.raises(ValueError, match="cannot affect a real run"):
        resolve_stage_updates(config, smoke=False)
    config["runtime"]["synthetic"] = False
    config["runtime"]["smoke_initialization_volumes"] = None
    config["runtime"]["synthetic_gate_override"] = False
    with pytest.raises(ValueError, match="cannot affect a real run"):
        validate_config(config)


def test_real_teacher_provenance_requires_verified_source_and_weight_facts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    del tmp_path, monkeypatch
    config = canonical_config("v1")
    weights_path = Path("/Users/alvinluong/Self-Audit-runs/q3d-resenc-20261112/checkpoint_final.pth")
    if not weights_path.is_file():
        pytest.skip("Phase-0 checkpoint is external to the repository")
    config["model"]["teacher"]["pretrained"]["weights_path"] = str(weights_path)
    validate_real_teacher_provenance(config)
    bad_hash = copy.deepcopy(config)
    bad_hash["model"]["teacher"]["pretrained"]["weights_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="SHA-256"):
        validate_real_teacher_provenance(bad_hash)
    del bad_hash
    not_ready = copy.deepcopy(config)
    not_ready["model"]["teacher"]["pretrained"]["provenance_status"] = "NOT_RUN"
    with pytest.raises(ValueError, match="unavailable"):
        validate_real_teacher_provenance(not_ready)
    bad_phase = copy.deepcopy(config)
    bad_phase["model"]["teacher"]["phase_measurement"]["receipt_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="locked ERF receipt"):
        validate_config(bad_phase)
    config = canonical_config("v1")
    config["model"]["teacher"]["pretrained"]["classifier_reinitialization"] = {"enabled": True}
    with pytest.raises(ValueError, match="exact fields"):
        validate_config(config)


def test_load_config_resolves_all_local_manifest_source_and_weight_paths(tmp_path: Path) -> None:
    config_dir = tmp_path / "recipes"
    config_dir.mkdir()
    (config_dir / "data").mkdir()
    manifest = config_dir / "data" / "train.json"
    manifest.write_text("{}\n", encoding="utf-8")
    weights = config_dir / "weights.pt"
    weights.write_bytes(b"config-only-weight-path-fixture")
    config = canonical_config("v1")
    config["dataset"]["manifests"]["train"] = {
        "path": "data/train.json", "sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(), "status": "READY",
    }
    config["model"]["teacher"]["pretrained"]["weights_path"] = "weights.pt"
    recipe = config_dir / "recipe.json"
    recipe.write_text(json.dumps(config), encoding="utf-8")
    loaded = load_config(recipe)
    assert loaded["dataset"]["manifests"]["train"]["path"] == str(manifest.resolve())
    assert loaded["model"]["teacher"]["pretrained"]["weights_path"] == str(weights.resolve())


def test_campaign_plan_orders_stages_settings_and_defers_unknown_datasets(tmp_path: Path) -> None:
    plan_path = create_campaign_plan(
        out=tmp_path / "campaign",
        datasets=("acdc",),
        protocols=("A",),
        ks=(2, 5, 10),
        synthetic=True,
        smoke_updates=2,
        share_stages=False,
    )
    plan = validate_campaign_plan(plan_path)
    experiments = plan["experiments"]
    primary = [item for item in experiments if item["group"] == "primary"]
    secondary = [item for item in experiments if item["group"] == "secondary"]
    baselines = [item for item in experiments if item["group"] == "baseline"]
    assert [item["k"] for item in primary if item["setting"] == "c0"] == [5, 10, 2]
    assert primary and secondary and baselines
    assert max(experiments.index(item) for item in primary) < min(experiments.index(item) for item in secondary)
    assert max(experiments.index(item) for item in secondary) < min(experiments.index(item) for item in baselines)
    assert set(item["setting"] for item in primary if item["k"] == 5) == set(PRIMARY_ARMS)
    assert set(item["setting"] for item in secondary if item["k"] == 5) == set(SECONDARY_ARMS) | {
        "v1_residual2",
    }
    for experiment in experiments:
        jobs = [item for item in plan["stage_jobs"] if item["experiment_id"] == experiment["id"]]
        assert [item["stage"] for item in jobs] == experiment["stages"]
        for previous, current in zip(jobs, jobs[1:]):
            assert current["dependencies"] == [previous["id"]]
            assert previous["stage"] in {"T", "Q", "P"}
        if experiment["setting"] == "v1":
            joint = next(item for item in jobs if item["stage"] == "J")
            assert joint["updates"] == 2
            assert joint["optimizer_updates"] == 2
            assert joint["stream_microbatches"] == 4
            assert joint["update_unit"] == "paired_L_U_updates"
        if experiment["setting"] == "a3d_teacher_free":
            assert experiment["student_joint_updates"] == 2 * experiment["k"]
            assert experiment["student_joint_optimizer_updates"] == 2 * experiment["k"]
            assert experiment["student_joint_stream_microbatches"] == 2 * experiment["k"]
            assert experiment["launch_mapping"]["runner"] == (
                "self_audit.candidate_q3d.baselines.run_a3d_teacher_free"
            )
            assert experiment["launch_mapping"]["update_match"] is True
    unknown = [item for item in plan["cross_evaluation"]
               if item.get("dataset") is None and "target_dataset" not in item]
    assert len(unknown) == 2
    assert all(item["status"] == "NOT_RUN" for item in unknown)
    assert plan["evidence"]["patient_training"] == "NOT_RUN"


def test_campaign_plan_records_manifest_hashes_and_baseline_launches(tmp_path: Path) -> None:
    test_manifest = tmp_path / "mnms-test.json"
    test_manifest.write_text("{}\n", encoding="utf-8")
    plan_path = create_campaign_plan(
        out=tmp_path / "campaign",
        datasets=("acdc",), protocols=("A",), ks=(5,),
        cross_inputs={"mnms": {"test": str(test_manifest)}},
        include_secondary=False, include_baselines=True,
        share_stages=False,
    )
    plan = validate_campaign_plan(plan_path)
    cross = [item for item in plan["cross_evaluation"] if item.get("target_dataset") == "mnms"]
    assert cross
    assert all(item["manifest"]["status"] == "READY" for item in cross)
    a3d = next(item for item in plan["experiments"] if item["setting"] == "a3d_teacher_free")
    assert a3d["launch_mapping"]["runner"] == "self_audit.candidate_q3d.baselines.run_a3d_teacher_free"
    assert a3d["launch_mapping"]["arm"] == "a3d_teacher_free"
    assert a3d["launch_mapping"]["update_match"] is False
    assert plan["baseline_launches"]["mean_teacher"]["arm"] == "mean_teacher"
    assert plan["baseline_launches"]["teacher_supervised"]["arm"] == "teacher_supervised"
    assert plan["baseline_launches"]["teacher_supervised"]["status"] == "IMPLEMENTED"
    teacher_recipe = load_config(next(
        item["config_path"] for item in plan["experiments"] if item["setting"] == "teacher_supervised"
    ))
    assert teacher_recipe["model"]["teacher"]["architecture"] == "nnfoundation_resenc_l"
    teacher_jobs = [item for item in plan["stage_jobs"] if item["setting"] == "teacher_supervised"]
    assert teacher_jobs


def test_teacher_supervised_k_only_protocol_b_readiness_skips_all_train_teacher_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    train_path = tmp_path / "train.json"
    train_path.write_text(json.dumps({
        "schema": "self_audit.data.v2",
        "role": "training",
        "dataset": "acdc",
        "label_budget": 2,
        "fewshot": {"selected_ids": ["patient001", "patient002"]},
        "labeled": [{"id": "patient001"}, {"id": "patient002"}],
    }), encoding="utf-8")
    development_path = tmp_path / "development.json"
    development_path.write_text(json.dumps({
        "schema": "self_audit.data.v2",
        "role": "development",
        "dataset": "acdc",
    }), encoding="utf-8")
    ready_manifest = lambda path: {
        "path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "status": "READY",
    }
    monkeypatch.setattr(q3d_campaign, "validate_real_teacher_provenance", lambda config: None)
    teacher = canonical_config("teacher_supervised", protocol="B", k=2)
    teacher["dataset"]["manifests"]["train"] = ready_manifest(train_path)
    teacher["dataset"]["manifests"]["development"] = ready_manifest(development_path)
    readiness, reason = q3d_campaign._job_readiness(teacher, synthetic=False, stage="T")
    assert (readiness, reason) == ("READY", None)

    student_teacher = canonical_config("v1", protocol="B", k=2)
    student_teacher["dataset"]["manifests"]["train"] = ready_manifest(train_path)
    student_teacher["dataset"]["manifests"]["development"] = ready_manifest(development_path)
    student_teacher["model"]["teacher"]["pretrained"]["provenance_status"] = "READY"
    readiness, reason = q3d_campaign._job_readiness(
        student_teacher, synthetic=False, stage="T",
    )
    assert readiness == "NOT_RUN"
    assert "all-TRAIN" in reason


def test_synthetic_protocol_b_views_share_cohort_images_and_strip_u_masks(tmp_path: Path) -> None:
    plan_path = create_campaign_plan(
        out=tmp_path / "campaign", datasets=("acdc",), protocols=("B",), ks=(2,),
        include_secondary=False, include_baselines=False, synthetic=True, smoke_updates=1,
    )
    plan = validate_campaign_plan(plan_path)
    experiment = next(item for item in plan["experiments"] if item["setting"] == "v1")
    config = load_config(experiment["config_path"])
    assert config["runtime"]["smoke_initialization_volumes"] == 8
    config_path, _train_manifest, manifest_refs = q3d_campaign._prepare_synthetic_run(
        plan_path, experiment, config,
    )
    runtime_config = load_config(config_path)
    student_ref = manifest_refs["train"]
    teacher_ref = runtime_config["dataset"]["manifests"]["teacher_train"]
    student_data = CandidateA3DData(student_ref["path"], student_ref["sha256"], synthetic=True)
    teacher_data = CandidateA3DData(teacher_ref["path"], teacher_ref["sha256"], synthetic=True)
    data = Q3DData(student_data, protocol="B", teacher_data=teacher_data)

    assert len(student_data.lids) == 2
    assert len(student_data.uids) == 2
    assert len(teacher_data.lids) == 4
    assert teacher_data.uids == []
    assert set(data.train_patients) == set(student_data.lids + student_data.uids)
    assert set(data.teacher_patients) == set(data.train_patients)
    assert all("masks" in record for record in teacher_data.labeled)
    assert all(set(record) == {"id", "image", "image_sha256"} for record in student_data.unlabeled)
    student_records = {record["id"]: record for record in student_data.labeled + student_data.unlabeled}
    teacher_records = {record["id"]: record for record in teacher_data.labeled}
    assert all(student_records[patient]["image_sha256"] == teacher_records[patient]["image_sha256"]
               for patient in data.train_patients)
    assert len({teacher_records[patient]["image_sha256"] for patient in data.train_patients}) == len(
        data.train_patients
    )


def test_campaign_adds_confident_teacher_fixture_flag_only_to_teacher_student_arms(tmp_path: Path) -> None:
    plan_path = create_campaign_plan(
        out=tmp_path / "campaign", datasets=("acdc",), protocols=("B",), ks=(2,),
        include_secondary=False, include_baselines=True, synthetic=True, smoke_updates=1,
    )
    plan = validate_campaign_plan(plan_path)
    experiments = {item["setting"]: item for item in plan["experiments"]}
    commands = {}
    for setting in ("v1", "c0", "mean_teacher"):
        experiment = experiments[setting]
        job = next(record for record in plan["stage_jobs"]
                   if record["experiment_id"] == experiment["id"])
        commands[setting] = q3d_campaign._job_command(
            experiment, config_path=Path(experiment["config_path"]),
            train_ref=job["train_manifest"], dev_ref=job["development_manifest"],
            device="cpu", synthetic=True,
        )
    assert "--synthetic-teacher-fixture" in commands["v1"]
    assert "--synthetic-teacher-fixture" not in commands["c0"]
    assert "--synthetic-teacher-fixture" not in commands["mean_teacher"]


def test_campaign_validation_is_not_execution_and_selected_synthetic_setting_executes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan_path = create_campaign_plan(
        out=tmp_path / "campaign", datasets=("acdc",), protocols=("A",), ks=(5,),
        include_secondary=False, include_baselines=False, synthetic=True, smoke_updates=1,
        share_stages=False,
    )
    not_run = execute_campaign(plan_path, setting="c0")
    assert not_run["status"] == "PLAN_VALIDATED_NOT_RUN"
    calls = []

    def fake_prepare(plan_path, experiment, config):
        refs = {name: {"path": None, "sha256": None, "status": "NOT_RUN"}
                for name in ("train", "development")}
        return Path(experiment["config_path"]), {}, refs

    monkeypatch.setattr(q3d_campaign, "_prepare_synthetic_run", fake_prepare)

    def fake_runner(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(returncode=0, stdout=json.dumps({"completed_stages": ["J"]}))

    completed = execute_campaign(plan_path, execute=True, setting="c0", device="cpu", runner=fake_runner)
    assert completed["status"] == "COMPLETED"
    assert completed["patient_training"] == "SYNTHETIC_ONLY"
    assert len(calls) == 1
    assert "--stage" not in calls[0]
    assert "--synthetic" in calls[0]


def test_campaign_enforces_stage_dependencies_and_real_run_gates(tmp_path: Path) -> None:
    synthetic_plan = create_campaign_plan(
        out=tmp_path / "synthetic", datasets=("acdc",), protocols=("A",), ks=(5,),
        include_secondary=False, include_baselines=False, synthetic=True, smoke_updates=1,
    )
    real_plan = create_campaign_plan(
        out=tmp_path / "real", datasets=("acdc",), protocols=("A",), ks=(5,),
        include_secondary=False, include_baselines=False,
    )
    real = validate_campaign_plan(real_plan)
    v1_teacher = next(job for job in real["stage_jobs"] if job["setting"] == "v1" and job["stage"] == "T")
    assert v1_teacher["readiness"] == "NOT_RUN"
    assert "manifest" in v1_teacher["reason"]
    with pytest.raises(ValueError, match="execution blocked"):
        execute_campaign(real_plan, execute=True, setting="v1",
                         runner=lambda *_args, **_kwargs: pytest.fail("must fail before launch"))


def test_all_selection_preflights_missing_real_inputs_before_any_runner_launch(tmp_path: Path) -> None:
    plan_path = create_campaign_plan(
        out=tmp_path / "campaign", datasets=("acdc",), protocols=("A",), ks=(5,),
        include_secondary=False, include_baselines=True,
    )
    launches = []
    with pytest.raises(ValueError, match="execution blocked"):
        execute_campaign(
            plan_path, execute=True, setting="all", device="cuda",
            runner=lambda command, **_kwargs: launches.append(command),
        )
    assert launches == []
    assert not (plan_path.parent / "RUN_STATE.json").exists()


def test_generated_teacher_supervised_baseline_command_dispatches_through_train_cli(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan_path = create_campaign_plan(
        out=tmp_path / "campaign", datasets=("acdc",), protocols=("A",), ks=(5,),
        include_secondary=False, include_baselines=True, synthetic=True, smoke_updates=1,
        share_stages=False,  # Preserve the independent historical baseline launcher contract.
    )
    from self_audit.candidate_q3d import baselines
    from train import main as train_main

    observed = []

    def fake_teacher(config, output_dir, *, device, smoke):
        observed.append((config["arm"], config["training"]["stage_updates"]["T"], smoke, device))
        return {"status": "COMPLETED_SOFTWARE_ONLY", "arm": "teacher_supervised"}

    monkeypatch.setattr(baselines, "run_teacher_supervised", fake_teacher)

    def invoke_generated_command(command, **_kwargs):
        assert command[1].endswith("/train.py")
        assert "--config" in command and "--synthetic" in command
        assert "--allow-full-run" not in command
        assert "--data" not in command and "--validation-data" not in command
        monkeypatch.setattr(sys, "argv", command[1:])
        captured = io.StringIO()
        with contextlib.redirect_stdout(captured):
            train_main()
        return SimpleNamespace(returncode=0, stdout=captured.getvalue())

    result = execute_campaign(
        plan_path, execute=True, setting="teacher_supervised", device="cpu",
        runner=invoke_generated_command,
    )
    assert result["status"] == "COMPLETED"
    assert observed == [("teacher_supervised", 1, True, "cpu")]


def test_k8_is_a_canonical_budget_and_real_manifest_guard_is_fail_closed(tmp_path: Path) -> None:
    config = canonical_config("v1", dataset="acdc", protocol="A", k=8, seed=42)
    assert config["protocol"]["k"] == 8
    assert config["model"]["teacher"]["architecture"] == "nnfoundation_resenc_l"

    train = tmp_path / "train.json"
    development = tmp_path / "development.json"
    train.write_text("{}\n", encoding="utf-8")
    development.write_text("{}\n", encoding="utf-8")
    config["runtime"]["synthetic"] = False
    config["dataset"]["manifests"]["train"] = {
        "path": str(train), "sha256": hashlib.sha256(train.read_bytes()).hexdigest(), "status": "READY",
    }
    config["dataset"]["manifests"]["development"] = {
        "path": str(development), "sha256": hashlib.sha256(development.read_bytes()).hexdigest(), "status": "READY",
    }
    with pytest.raises(ValueError, match="K=8 TRAIN manifest SHA-256"):
        validate_config(config, real_run=True)


def test_non_resenc_teacher_source_fields_are_rejected() -> None:
    config = canonical_config("v1", k=8)
    config["model"]["teacher"]["legacy_source"] = "/tmp/old-backbone"
    with pytest.raises(ValueError, match="exact fields"):
        validate_config(config)


def test_boundary_learning_rate_is_an_explicit_shared_head_alias() -> None:
    config = canonical_config("v1")
    config["training"]["teacher_boundary_lr"] = config["training"]["teacher_head_lr"] / 2
    with pytest.raises(ValueError, match="teacher_boundary_lr must equal teacher_head_lr"):
        validate_config(config)
