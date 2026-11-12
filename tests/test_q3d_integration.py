"""Top-level dispatch contracts for the explicit Q3D runtime."""

from __future__ import annotations

import argparse
import importlib
import json
import sys
import types
from pathlib import Path

import pytest
import torch

import evaluate as evaluation_cli
import train as training_cli
from self_audit.candidate_a3d.data import CandidateA3DData, synthetic_fixture
from self_audit.candidate_q3d.config import (
    arm_settings, canonical_config, resolve_stage_accounting, stages_for_arm, validate_config,
)
from self_audit.candidate_q3d.data import Q3DData


ROOT = Path(__file__).resolve().parents[1]


def _fake_module(name: str, **members):
    module = types.ModuleType(name)
    for key, value in members.items():
        setattr(module, key, value)
    return module


def test_active_pipeline_declares_q3d_default_and_keeps_baselines():
    active = json.loads((ROOT / "ACTIVE_PIPELINE.json").read_text())
    assert active["default_track"] == "candidate_q3d"
    assert active["method"] == "candidate_q3d"
    assert active["tracks"]["candidate_a"]["config_schema"] == "self_audit.candidate_a.v2"
    assert active["tracks"]["candidate_a3d"]["config_schema"] == "self_audit.candidate_a3d.v1"
    assert active["tracks"]["candidate_q3d"]["default_config"] == "configs/candidate_q3d/v1.json"


def test_q3d_configs_are_nested_and_keep_real_provenance_not_run():
    config = json.loads((ROOT / "configs/candidate_q3d/v1.json").read_text())
    validated = validate_config(config)
    assert validated is not config
    assert validated["model"]["student"]["architecture"] == "unetrpp_lite_3d"
    assert validated["runtime"]["synthetic"] is False
    assert validated["model"]["teacher"]["pretrained"]["provenance_status"] == "NOT_FOUND_OFFICIAL"


def test_all_fifteen_recipes_select_declared_engine_effects():
    from self_audit.candidate_q3d.engine import _student_arm

    recipes = sorted((ROOT / "configs/candidate_q3d").glob("*.json"))
    assert len(recipes) == 15
    for path in recipes:
        config = validate_config(json.loads(path.read_text()))
        arm = config["arm"]
        settings = arm_settings(arm)
        assert tuple(stages_for_arm(arm)) == tuple(settings["stages"])
        accounting = resolve_stage_accounting(config)
        assert accounting["optimizer_updates"] == config["training"]["stage_optimizer_updates"]
        assert accounting["stream_microbatches"] == config["training"]["stage_stream_microbatches"]
        if "J" in settings["stages"]:
            expected_streams = 1 if arm in {"c0", "a3d_teacher_free"} else 2
            assert config["training"]["stage_stream_microbatches"]["J"] == (
                expected_streams * config["training"]["stage_updates"]["J"]
            )
            if expected_streams == 1:
                assert config["training"]["stage_update_units"]["J"] == "L_only_updates"
        if "P" in settings["stages"]:
            assert accounting["optimizer_updates"]["P"] == 0
            assert accounting["stream_microbatches"]["P"] is None
            assert config["training"]["stage_update_units"]["P"] == "full_U_patient_cine_passes"
        if settings["tokenizer"]:
            assert config["tokenizer"]["initialization_volumes"] == 200
            assert config["training"]["stage_update_units"]["Q"] == "optimizer_updates_after_separate_init_pass"
        if arm in {"a3d_teacher_free", "mednext_supervised"}:
            assert config["model"]["student"]["architecture"] in {
                "candidate_a3d", "mednext_l_k3_supervised",
            }
            if arm == "a3d_teacher_free":
                assert config["baselines"][arm]["method"] == "candidate_q3d"
            else:
                assert settings["stages"] == ("T",)
                assert config["model"]["student"]["architecture"] == "mednext_l_k3_supervised"
        else:
            assert _student_arm(arm) in {"C0", "C1", "V0", "V1", "V1-U", "B1"}
            assert config["tokenizer"]["enabled"] is settings["tokenizer"]
            assert config["physiology"]["enabled"] is settings["physiology"]
        if path.stem == "v1_residual2":
            assert config["model"]["student"]["residual2"]["enabled"] is True
        if path.stem == "v1_stride4":
            assert config["model"]["teacher"]["feature_stride"] == 4


def test_train_dispatch_selects_only_declared_q3d_engine(monkeypatch):
    sentinel_train = lambda *args, **kwargs: {"status": "called"}
    sentinel_validate = lambda config, **kwargs: config
    monkeypatch.setitem(
        sys.modules,
        "self_audit.candidate_q3d.engine",
        _fake_module("self_audit.candidate_q3d.engine", train=sentinel_train),
    )
    monkeypatch.setitem(
        sys.modules,
        "self_audit.candidate_q3d.config",
        _fake_module("self_audit.candidate_q3d.config", validate_config=sentinel_validate),
    )
    trainer, validator = training_cli._training_api("candidate_q3d")
    assert trainer is sentinel_train
    assert validator is sentinel_validate


@pytest.mark.parametrize("pinned", [False, True])
def test_q3d_cli_handoff_passes_validated_nested_config_without_patient_inputs(monkeypatch, tmp_path, pinned):
    normalized = {
        "method": "candidate_q3d",
        "arm": "v1",
        "runtime": {"synthetic": True, "synthetic_gate_override": False},
        "training": {"stage_updates": {"T": 1, "Q": 1, "P": 1, "J": 1}},
    }
    received = {}

    def validate(config, **kwargs):
        received["config_in"] = config
        received["validation_options"] = kwargs
        return normalized

    def run(config, out, *, device, teacher, synthetic_gate_override, stage=None, stage_artifacts=None):
        if pinned:
            assert stage == "J"
            assert stage_artifacts == {"T": {"manifest_path": "/tmp/T.artifact.json", "manifest_sha256": "a" * 64}}
        else:
            assert stage is None and stage_artifacts is None
        received.update(config=config, out=out, device=device,
                        teacher=teacher, synthetic_gate_override=synthetic_gate_override)
        return {"status": "SYNTHETIC_DISPATCHED"}

    monkeypatch.setitem(
        sys.modules,
        "self_audit.candidate_q3d.config",
        _fake_module(
            "self_audit.candidate_q3d.config",
            validate_config=validate,
            validate_real_teacher_provenance=lambda config: pytest.fail(
                "synthetic dispatch must not invoke real teacher provenance"
            ),
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "self_audit.candidate_q3d.engine",
        _fake_module("self_audit.candidate_q3d.engine", train=run),
    )
    args = argparse.Namespace(
        model_config=None,
        teacher_checkpoint=None,
        teacher_checkpoint_sha256=None,
        data=None,
        data_sha256=None,
        validation_data=None,
        validation_data_sha256=None,
        out=str(tmp_path / "run"),
        synthetic=True,
        allow_full_run=False,
        device="cpu",
        q3d_stage="J" if pinned else None,
        q3d_stage_artifact=["T=/tmp/T.artifact.json"] if pinned else None,
        q3d_stage_artifact_sha=["T=" + "a" * 64] if pinned else None,
    )
    parser = argparse.ArgumentParser()
    result = training_cli._train_q3d(args, parser, {
        "method": "candidate_q3d", "runtime": {"synthetic": True},
    })
    assert result == {"status": "SYNTHETIC_DISPATCHED"}
    assert received["config_in"] == {
        "method": "candidate_q3d", "runtime": {"synthetic": True},
    }
    assert received["validation_options"] == {"real_run": False}
    assert received["config"] is normalized
    assert received["out"] == args.out
    assert received["device"] == "cpu"
    assert received["teacher"] is None
    assert received["synthetic_gate_override"] is False


def test_q3d_main_loads_and_normalizes_nested_config(monkeypatch, tmp_path, capsys):
    config_path = tmp_path / "nested" / "recipe.json"
    config_path.parent.mkdir()
    config_path.write_text(json.dumps({
        "method": "candidate_q3d", "runtime": {"synthetic": True},
    }))
    expected = {"method": "candidate_q3d", "runtime": {"synthetic": True}}
    observed = {}

    def load_config(path, *, real_run):
        observed["path"] = path
        observed["real_run"] = real_run
        return expected

    def dispatch(args, parser, config, *, synthetic_teacher_fixture=False):
        observed["config"] = config
        observed["synthetic_teacher_fixture"] = synthetic_teacher_fixture
        return {"status": "NORMALIZED_Q3D_CONFIG"}

    module = importlib.import_module("self_audit.candidate_q3d.config")
    monkeypatch.setattr(module, "load_config", load_config)
    monkeypatch.setattr(training_cli, "_train_q3d", dispatch)
    monkeypatch.setattr(sys, "argv", ["train.py", "--config", str(config_path)])
    training_cli.main()
    assert observed == {
        "path": str(config_path), "real_run": False, "config": expected,
        "synthetic_teacher_fixture": False,
    }
    assert json.loads(capsys.readouterr().out) == {"status": "NORMALIZED_Q3D_CONFIG"}


def test_real_q3d_config_validation_fails_closed_before_engine_dispatch(monkeypatch, tmp_path):
    calls = []

    def validate(config, *, real_run):
        calls.append(("validate", real_run))
        raise ValueError("real Q3D manifests or teacher provenance are not READY")

    def forbidden_train(*args, **kwargs):
        pytest.fail("engine must not run with incomplete real provenance")

    monkeypatch.setitem(
        sys.modules, "self_audit.candidate_q3d.config",
        _fake_module(
            "self_audit.candidate_q3d.config",
            validate_config=validate,
            validate_real_teacher_provenance=lambda config: pytest.fail(
                "invalid real manifests must fail before provenance revalidation"
            ),
        ),
    )
    monkeypatch.setitem(
        sys.modules, "self_audit.candidate_q3d.engine",
        _fake_module("self_audit.candidate_q3d.engine", train=forbidden_train),
    )
    args = argparse.Namespace(
        model_config=None, teacher_checkpoint=None, teacher_checkpoint_sha256=None,
        data=None, data_sha256=None, validation_data=None, validation_data_sha256=None,
        out=str(tmp_path / "real-run"), synthetic=False, allow_full_run=True, device="cpu",
    )
    with pytest.raises(SystemExit):
        training_cli._train_q3d(
            args, argparse.ArgumentParser(),
            {"method": "candidate_q3d", "runtime": {"synthetic": False}},
        )
    assert calls == [("validate", True)]


def test_synthetic_teacher_fixture_is_refused_for_real_config(monkeypatch, tmp_path):
    normalized = {
        "method": "candidate_q3d", "arm": "v1",
        "runtime": {"synthetic": False, "synthetic_gate_override": False},
    }
    monkeypatch.setitem(
        sys.modules, "self_audit.candidate_q3d.config",
        _fake_module(
            "self_audit.candidate_q3d.config",
            validate_config=lambda config, *, real_run: normalized,
            validate_real_teacher_provenance=lambda config: pytest.fail(
                "synthetic fixture selector must reject before any real teacher setup"
            ),
        ),
    )
    args = argparse.Namespace(
        model_config=None, teacher_checkpoint=None, teacher_checkpoint_sha256=None,
        data=None, data_sha256=None, validation_data=None, validation_data_sha256=None,
        out=str(tmp_path / "real-run"), synthetic=False, allow_full_run=True, device="cpu",
    )
    with pytest.raises(SystemExit):
        training_cli._train_q3d(
            args, argparse.ArgumentParser(),
            {"method": "candidate_q3d", "runtime": {"synthetic": False}},
            synthetic_teacher_fixture=True,
        )


@pytest.mark.parametrize(
    ("arm", "runner_name"),
    [
        ("a3d_teacher_free", "run_a3d_teacher_free"),
        ("mednext_supervised", "run_mednext_supervised"),
    ],
)
def test_nested_baseline_arms_dispatch_to_separate_runner(monkeypatch, tmp_path, arm, runner_name):
    normalized = {
        "method": "candidate_q3d", "arm": arm, "runtime": {"synthetic": True},
    }
    received = {}

    def validate(config, *, real_run):
        received["real_run"] = real_run
        return normalized

    def run(config, out, *, device, smoke):
        received.update(config=config, out=out, device=device, smoke=smoke)
        return {"baseline_arm": arm, "status": "SOFTWARE_ONLY_SYNTHETIC"}

    monkeypatch.setitem(
        sys.modules, "self_audit.candidate_q3d.config",
        _fake_module(
            "self_audit.candidate_q3d.config", validate_config=validate,
            validate_real_teacher_provenance=lambda config: pytest.fail(
                "synthetic baseline dispatch must not require real teacher weights"
            ),
        ),
    )
    baselines = importlib.import_module("self_audit.candidate_q3d.baselines")
    monkeypatch.setattr(baselines, runner_name, run)
    args = argparse.Namespace(
        model_config=None, teacher_checkpoint=None, teacher_checkpoint_sha256=None,
        data=None, data_sha256=None, validation_data=None, validation_data_sha256=None,
        out=str(tmp_path / arm), synthetic=True, allow_full_run=False, device="cpu",
    )
    result = training_cli._train_q3d(
        args, argparse.ArgumentParser(),
        {"method": "candidate_q3d", "arm": arm, "runtime": {"synthetic": True}},
    )
    assert result["baseline_arm"] == arm
    assert received == {
        "real_run": False, "config": normalized, "out": args.out,
        "device": "cpu", "smoke": True,
    }


def test_q3d_export_schema_uses_student_only_path_and_unknown_schema_fails():
    selected = evaluation_cli.evaluator_for_schema("q3d.student_export.v1")
    assert selected is evaluation_cli._evaluate_q3d_export
    with pytest.raises(ValueError, match="unsupported checkpoint schema"):
        evaluation_cli.evaluator_for_schema("unversioned-or-unknown")


def test_q3d_evaluation_hash_pins_export_and_train_ids(monkeypatch, tmp_path):
    artifact = tmp_path / "student.q3d"
    artifact.write_bytes(b"student artifact bytes")
    artifact_sha = evaluation_cli.digest(artifact)
    artifact.with_suffix(artifact.suffix + ".manifest.json").write_text(json.dumps({
        "schema": "q3d.export_manifest.v1",
        "artifact": artifact.name,
        "artifact_sha256": artifact_sha,
        "artifact_bytes": artifact.stat().st_size,
    }))
    train_manifest = tmp_path / "train.json"
    train_manifest.write_text(json.dumps({
        "schema": "self_audit.data.v2",
        "role": "training",
        "labeled": [{"id": "L0"}],
        "unlabeled": [{"id": "U0"}],
    }))
    train_sha = evaluation_cli.digest(train_manifest)
    observed = {}
    model = object()

    def load_export(path, *, expected_sha256, device):
        observed["load"] = (path, expected_sha256, device)
        return model, {}

    def evaluate_manifest(received_model, manifest, sha, out, **kwargs):
        observed["evaluate"] = (received_model, manifest, sha, out, kwargs)
        return {"patient_macro": {"fg_dice": 0.4}}

    monkeypatch.setitem(sys.modules, "self_audit.candidate_q3d.export",
                        _fake_module("self_audit.candidate_q3d.export", load_export=load_export))
    monkeypatch.setitem(sys.modules, "self_audit.candidate_q3d.evaluation",
                        _fake_module("self_audit.candidate_q3d.evaluation", evaluate_manifest=evaluate_manifest))
    result = evaluation_cli._evaluate_q3d_export(
        str(artifact), artifact_sha, "dev.json", "dev-sha", "out", "cpu",
        train_manifest=str(train_manifest), train_sha=train_sha, split="dev",
    )
    assert result == {"patient_macro": {"fg_dice": 0.4}}
    assert observed["load"] == (artifact.resolve(), artifact_sha, "cpu")
    assert observed["evaluate"] == (
        model, "dev.json", "dev-sha", "out",
        {"device": "cpu", "split": "dev", "train_patient_ids": ("L0", "U0")},
    )


def test_q3d_evaluation_refuses_to_skip_training_overlap_manifest(tmp_path):
    artifact = tmp_path / "student.q3d"
    artifact.write_bytes(b"student artifact bytes")
    sha = evaluation_cli.digest(artifact)
    artifact.with_suffix(artifact.suffix + ".manifest.json").write_text(json.dumps({
        "schema": "q3d.export_manifest.v1", "artifact": artifact.name,
        "artifact_sha256": sha, "artifact_bytes": artifact.stat().st_size,
    }))
    with pytest.raises(ValueError, match="requires --train-data"):
        evaluation_cli._evaluate_q3d_export(str(artifact), sha, "dev", "dev-sha", "out")


def test_q3d_evaluate_cli_accepts_internal_test_split(monkeypatch, tmp_path, capsys):
    artifact = tmp_path / "student.q3d"
    artifact.write_bytes(b"student artifact bytes")
    artifact_sha = evaluation_cli.digest(artifact)
    train_manifest = tmp_path / "train.json"
    train_manifest.write_text(json.dumps({
        "schema": "self_audit.data.v2", "role": "training",
        "labeled": [{"id": "L0"}], "unlabeled": [{"id": "U0"}],
    }))
    received = {}

    def run(*args, **kwargs):
        received["args"] = args
        received["kwargs"] = kwargs
        return {"patient_macro": {"fg_dice": 0.5}}

    monkeypatch.setattr(evaluation_cli, "evaluator_for_schema", lambda schema: run)
    monkeypatch.setattr(sys, "argv", [
        "evaluate.py", "--checkpoint", str(artifact), "--checkpoint-sha256", artifact_sha,
        "--data", "test.json", "--data-sha256", "test-sha", "--out", "evaluation-out",
        "--train-data", str(train_manifest), "--train-data-sha256", evaluation_cli.digest(train_manifest),
        "--split", "test",
    ])
    evaluation_cli.main()
    assert received["kwargs"]["split"] == "test"
    assert json.loads(capsys.readouterr().out) == {"fg_dice": 0.5}


def test_mednext_research_evaluation_routes_through_baseline_owner(monkeypatch, tmp_path, capsys):
    checkpoint = tmp_path / "latest.pt"
    checkpoint.write_bytes(b"synthetic resume bytes")
    observed = {}

    def evaluate(checkpoint_path, checkpoint_sha, config_path, manifest, data_sha, out,
                 device, split):
        observed.update(
            checkpoint_path=checkpoint_path, checkpoint_sha=checkpoint_sha,
            config_path=config_path, manifest=manifest, data_sha=data_sha,
            out=out, device=device, split=split,
        )
        return {"artifact_kind": "heavy_mednext_region_baseline", "student_export": False}

    monkeypatch.setattr(evaluation_cli, "_evaluate_configured_baseline", evaluate)
    config = tmp_path / "mednext.json"
    config.write_text("{}")
    monkeypatch.setattr(sys, "argv", [
        "evaluate.py", "--checkpoint", str(checkpoint), "--checkpoint-sha256", "abc",
        "--config", str(config), "--data", "test.json", "--data-sha256", "def",
        "--out", "eval-out", "--device", "cpu", "--split", "test",
    ])
    evaluation_cli.main()
    assert observed == {
        "checkpoint_path": str(checkpoint), "checkpoint_sha": "abc",
        "config_path": str(config), "manifest": "test.json", "data_sha": "def",
        "out": "eval-out", "device": "cpu", "split": "test",
    }
    assert json.loads(capsys.readouterr().out) == {
        "artifact_kind": "heavy_mednext_region_baseline", "student_export": False,
    }


def test_q3d_labeled_image_loader_never_opens_mask(monkeypatch, tmp_path):
    fixture = synthetic_fixture(
        tmp_path / "fixture", n_labeled=2, n_unlabeled=1, n_evaluation=1,
        shape_tzyx=(2, 4, 8, 9),
    )
    candidate_a3d = CandidateA3DData(
        fixture["train"], fixture["train_sha256"], synthetic=True,
    )
    data = Q3DData(candidate_a3d, protocol="A")
    record = candidate_a3d.labeled[0]
    accessed = []
    original_array = candidate_a3d.base.array

    def forbid_mask(path, sha256):
        accessed.append(path)
        assert "mask" not in str(path).lower()
        return original_array(path, sha256)

    monkeypatch.setattr(candidate_a3d.base, "array", forbid_mask)
    monkeypatch.setattr(
        candidate_a3d, "load_labeled_frame",
        lambda key: pytest.fail("Q-stage image loading must not open a labeled mask"),
    )
    identity = data.identity()
    assert identity["protocol"] == "A"
    image_keys = data.labeled_image_frame_keys()
    assert image_keys == [(0, 0), (0, 1), (1, 0), (1, 1)]
    from self_audit.candidate_q3d.engine import _load_labeled_image_only

    image, geometry, metadata = _load_labeled_image_only(data, (0, 1))
    assert image.shape == (1, 1, 4, 8, 9)
    assert image.dtype == torch.float32
    assert geometry["shape"] == (4, 8, 9)
    assert metadata == {"patient": record["id"], "frame": 1}
    permitted_images = {record["image"] for record in candidate_a3d.labeled + candidate_a3d.unlabeled}
    assert accessed
    assert set(accessed).issubset(permitted_images)
    assert candidate_a3d.label_reads == set()


def test_c0_sampler_avoids_u_and_keeps_the_paired_labeled_rng_stream(monkeypatch, tmp_path):
    from self_audit.candidate_q3d.data import Q3DJointSampler

    fixture = synthetic_fixture(
        tmp_path / "paired", seed=41, n_labeled=2, n_unlabeled=1, n_evaluation=1,
        shape_tzyx=(2, 4, 8, 9),
    )
    candidate = CandidateA3DData(fixture["train"], fixture["train_sha256"], synthetic=True)
    data = Q3DData(candidate, protocol="A")
    paired = Q3DJointSampler(data, seed=53, require_unlabeled=True)
    paired_updates = [paired.sample_update() for _ in range(12)]

    u_paths = {record["image"] for record in candidate.unlabeled}
    original_array = candidate.base.array

    def forbid_u(path, sha256):
        assert path not in u_paths, "C0 must not inspect image-only U arrays"
        return original_array(path, sha256)

    monkeypatch.setattr(candidate.base, "array", forbid_u)
    c0 = Q3DJointSampler(data, seed=53, require_unlabeled=False)
    c0_updates = [c0.sample_update() for _ in range(12)]
    assert c0.STATE_SCHEMA == "self_audit.candidate_q3d.joint_sampler.v3"
    assert all("unlabeled_patient" not in row and "unlabeled_key" not in row for row in c0_updates)
    assert [(row["labeled_patient"], row["labeled_key"]) for row in c0_updates] == [
        (row["labeled_patient"], row["labeled_key"]) for row in paired_updates
    ]


def test_c0_sampler_accepts_zero_u_and_sampler_v2_restores_both_rngs(tmp_path):
    from self_audit.candidate_q3d.data import Q3DJointSampler

    no_u_fixture = synthetic_fixture(
        tmp_path / "no-u", seed=47, n_labeled=2, n_unlabeled=0, n_evaluation=1,
        shape_tzyx=(2, 4, 8, 9),
    )
    no_u_candidate = CandidateA3DData(
        no_u_fixture["train"], no_u_fixture["train_sha256"], synthetic=True,
    )
    no_u_data = Q3DData(no_u_candidate, protocol="A")
    no_u_sampler = Q3DJointSampler(no_u_data, seed=59, require_unlabeled=False)
    update = no_u_sampler.sample_update()
    assert "unlabeled_patient" not in update and "unlabeled_key" not in update
    assert no_u_sampler.state_dict()["require_unlabeled"] is False
    with pytest.raises(ValueError, match="requires image-only U"):
        Q3DJointSampler(no_u_data, seed=59, require_unlabeled=True)

    paired_fixture = synthetic_fixture(
        tmp_path / "paired", seed=61, n_labeled=2, n_unlabeled=2, n_evaluation=1,
        shape_tzyx=(2, 4, 8, 9),
    )
    paired_candidate = CandidateA3DData(
        paired_fixture["train"], paired_fixture["train_sha256"], synthetic=True,
    )
    paired_data = Q3DData(paired_candidate, protocol="A")
    original = Q3DJointSampler(paired_data, seed=67)
    for _ in range(5):
        original.sample_update()
    state = original.state_dict()
    expected = [original.sample_update() for _ in range(8)]
    restored = Q3DJointSampler(paired_data, seed=67)
    restored.load_state_dict(state)
    assert [restored.sample_update() for _ in range(8)] == expected
    l_only = Q3DJointSampler(paired_data, seed=67, require_unlabeled=False)
    with pytest.raises(ValueError, match="L/U mode"):
        l_only.load_state_dict(state)
    tampered = dict(state)
    tampered["schema"] = "self_audit.candidate_q3d.joint_sampler.v1"
    with pytest.raises(ValueError, match="schema mismatch"):
        restored.load_state_dict(tampered)


def test_synthetic_update_limit_resume_matches_uninterrupted_c0(tmp_path):
    import torch

    from self_audit.candidate_q3d.engine import train

    torch.set_num_threads(1)
    fixture = synthetic_fixture(
        tmp_path / "fixture", seed=31, n_labeled=2, n_unlabeled=1, n_evaluation=1,
        shape_tzyx=(2, 8, 32, 32),
    )
    config = canonical_config("c0", dataset="acdc", protocol="A", k=2, seed=31)
    config["runtime"].update({"synthetic": True, "device": "cpu"})
    config["runtime"]["smoke_update_overrides"] = {"T": 1, "Q": 1, "P": 1, "J": 4}
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
    config = validate_config(config)
    dev = lambda model, manifest, output_kind, device: {"score": 0.25, "patient_count": 1}

    interrupted_dir = tmp_path / "interrupted"
    first = train(config, interrupted_dir, device="cpu", smoke=True,
                  update_limit=2, dev_evaluator=dev)
    assert first["status"] == "PAUSED"
    resumed = train(config, interrupted_dir, device="cpu", smoke=True,
                    resume_from=first["checkpoint"], update_limit=2, dev_evaluator=dev)
    uninterrupted = train(config, tmp_path / "uninterrupted", device="cpu", smoke=True,
                          dev_evaluator=dev)
    assert resumed["status"] == uninterrupted["status"] == "COMPLETED"
    assert resumed["completed_stages"] == uninterrupted["completed_stages"] == ["J"]
    assert resumed["completed_updates"] == uninterrupted["completed_updates"] == 4
    assert resumed["joint_optimizer_updates"] == uninterrupted["joint_optimizer_updates"] == 4
    assert resumed["student_state_sha256"] == uninterrupted["student_state_sha256"]
    interrupted_updates = [json.loads(line) for line in (interrupted_dir / "metrics.jsonl").read_text().splitlines()]
    uninterrupted_updates = [json.loads(line) for line in (tmp_path / "uninterrupted" / "metrics.jsonl").read_text().splitlines()]
    interrupted_joint = [row for row in interrupted_updates if row.get("stage") == "J"
                         and "streams_per_update" in row]
    uninterrupted_joint = [row for row in uninterrupted_updates if row.get("stage") == "J"
                           and "streams_per_update" in row]
    assert len(interrupted_joint) == len(uninterrupted_joint) == 4
    assert [row["streams_per_update"] for row in interrupted_joint] == [1] * 4
    assert [row["streams_per_update"] for row in uninterrupted_joint] == [1] * 4
    assert [row["optimizer_updates"] for row in interrupted_joint] == [1, 2, 3, 4]


def test_mean_teacher_keeps_labeled_supervision_coefficient_one(monkeypatch, tmp_path):
    import torch
    from torch import nn

    import self_audit.candidate_q3d.engine as engine

    torch.set_num_threads(1)
    fixture = synthetic_fixture(
        tmp_path / "fixture", seed=37, n_labeled=2, n_unlabeled=1, n_evaluation=1,
        shape_tzyx=(2, 4, 8, 9),
    )
    config = canonical_config("mean_teacher", dataset="acdc", protocol="A", k=2, seed=37)
    config["runtime"].update({"synthetic": True, "device": "cpu"})
    config["runtime"]["smoke_update_overrides"] = {"T": 1, "Q": 1, "P": 1, "J": 1}
    config["dataset"]["manifests"]["train"] = {
        "path": str(fixture["train"].resolve()), "sha256": fixture["train_sha256"], "status": "READY",
    }
    config["dataset"]["manifests"]["development"] = {
        "path": str(fixture["evaluation"].resolve()),
        "sha256": fixture["evaluation_sha256"], "status": "READY",
    }
    config = validate_config(config)

    class TinyStudent(nn.Module):
        def __init__(self):
            super().__init__()
            self.scale = nn.Parameter(torch.tensor(1.0))
            self.register_buffer("codebook", torch.empty(0))

        def forward(self, image, geometry=None):
            shape = (1, 4, *image.shape[-3:])
            value = self.scale.expand(shape)
            return {"logits": value, "sdf": value}

    student = TinyStudent()

    def supervised_loss(*args, **kwargs):
        loss = 0.1 * student.scale.square()
        return loss, {"segmentation": loss.detach(), "sdf_near_far": loss.detach()}

    monkeypatch.setattr(engine, "student_supervised_loss", supervised_loss)
    monkeypatch.setattr(
        engine, "mean_teacher_consistency_loss",
        lambda student_logits, teacher_logits: student.scale * 0.0,
    )
    monkeypatch.setattr(
        engine, "_make_optimizer",
        lambda parameters, *, lr, weight_decay: torch.optim.SGD(parameters, lr=lr),
    )
    result = engine.train(
        config, tmp_path / "mean-teacher", device="cpu", smoke=True,
        student=student,
        dev_evaluator=lambda *args: {"score": 0.5, "patient_count": 1},
    )
    expected = 1.0 - 2.0 * 0.1 * config["training"]["student_lr"]
    assert result["status"] == "COMPLETED"
    assert result["completed_updates"] == result["joint_optimizer_updates"] == 1
    assert student.scale.item() == pytest.approx(expected, abs=1e-7)
    joint_record = next(
        row for row in map(json.loads, (tmp_path / "mean-teacher" / "metrics.jsonl").read_text().splitlines())
        if row.get("stage") == "J" and "streams_per_update" in row
    )
    assert joint_record["streams_per_update"] == 2
    assert joint_record["optimizer_updates"] == 1


def test_baseline_training_dispatch_remains_separate_from_q3d():
    trainer, validator = training_cli._training_api("candidate_a3d")
    assert trainer.__module__ == "self_audit.candidate_a3d.engine"
    assert validator.__module__ == "self_audit.candidate_a3d.config"


def test_q3d_main_describe_uses_active_pipeline(capsys, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["train.py", "--describe"])
    training_cli.main()
    described = json.loads(capsys.readouterr().out)
    assert described["method"] == "candidate_q3d"
