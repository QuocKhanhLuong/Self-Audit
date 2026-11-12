"""SOFTWARE_ONLY: mock stage runner, no training/campaign subprocesses."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from self_audit.candidate_q3d.artifacts import file_sha256, write_stage_manifest
from self_audit.candidate_q3d.campaign_runtime import (
    collect_campaign_results,
    execute_shared_campaign,
    select_campaign_experiments,
    write_campaign_results,
)


def _fixture(tmp_path, *, ks=(2,)):
    jobs, experiments = [], []
    multi_k = len(ks) > 1
    for k in ks:
        identity = {"dataset": "acdc", "protocol": "A", "k": k, "seed": 19}
        producer_ids = {stage: f"{stage}{k}" if multi_k else stage for stage in ("T", "Q", "P")}
        for stage in ("T", "Q", "P"):
            dependencies = [] if stage == "T" else (
                [producer_ids["T"]] if stage == "Q" else [producer_ids["T"], producer_ids["Q"]]
            )
            job_id = producer_ids[stage]
            output = tmp_path / job_id
            jobs.append({
                "id": job_id, "stage": stage, "dependencies": dependencies,
                "experiment_id": f"v1@{k}" if multi_k else "v1",
                "method": "candidate_q3d", "readiness": "READY", "reason": None,
                "output": str(output), "artifact_identity": identity,
                "train_manifest": {}, "development_manifest": {},
                "artifact_output": {"manifest_path": str(output / f"{stage}.artifact.json")},
                "stage_artifact_inputs": {
                    parent[0]: {
                        "producer_job_id": parent[1],
                        "path": str(tmp_path / parent[1] / f"{parent[0]}.artifact.json"),
                    }
                    for parent in (("T", producer_ids["T"]), ("Q", producer_ids["Q"]))
                    if parent[1] in dependencies
                },
            })
        for setting in ("v1", "v1u"):
            experiment_id = f"{setting}@{k}" if multi_k else setting
            joint_id = f"{experiment_id}.J"
            config_path = tmp_path / f"{experiment_id}.json"
            config_path.write_text(json.dumps({"method": "candidate_q3d", "arm": setting}))
            stage_job_ids = [*producer_ids.values(), joint_id]
            experiments.append({
                "id": experiment_id, "setting": setting, "arm": setting, **identity,
                "stages": ["T", "Q", "P", "J"], "stage_job_ids": stage_job_ids,
                "stage_artifact_producers": {
                    **{stage: producer_ids[stage] for stage in ("T", "Q", "P")},
                    "J": joint_id,
                },
                "config_path": str(config_path), "output": str(tmp_path / experiment_id),
            })
            jobs.append({
                "id": joint_id, "stage": "J", "dependencies": list(producer_ids.values()),
                "experiment_id": experiment_id, "method": "candidate_q3d", "readiness": "READY", "reason": None,
                "output": str(tmp_path / experiment_id), "train_manifest": {}, "development_manifest": {},
                "stage_artifact_inputs": {
                    stage: {
                        "producer_job_id": producer_ids[stage],
                        "path": str(tmp_path / producer_ids[stage] / f"{stage}.artifact.json"),
                    }
                    for stage in ("T", "Q", "P")
                },
            })
    plan = {"selection": {"synthetic": False}, "experiments": experiments, "stage_jobs": jobs}
    plan_path = tmp_path / "PLAN.json"
    plan_path.write_text(json.dumps(plan))

    def write_json(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))

    def read_state(path, plan, digest):
        return json.loads(path.read_text()) if path.exists() else {
            "plan_sha256": digest,
            "jobs": {job["id"]: {"status": "NOT_RUN"} for job in plan["stage_jobs"]},
            "experiments": {item["id"]: {"status": "NOT_RUN"} for item in plan["experiments"]},
        }

    def job_command(experiment, *, stage, stage_artifacts, **_kwargs):
        command = ["mock-train", "--stage", stage, "--out", experiment["output"]]
        for key, value in stage_artifacts.items():
            command.extend(["--artifact", f"{key}={value}"])
        return command

    api = SimpleNamespace(
        ROOT=tmp_path, _read_state=read_state, _sha256=file_sha256,
        _write_json=write_json, _job_command=job_command,
        load_config=lambda path, **_: json.loads(Path(path).read_text()),
    )
    commands = []

    jobs_by_output = {job["output"]: job for job in jobs}

    def runner(command, **_kwargs):
        commands.append(command)
        stage = command[command.index("--stage") + 1]
        output = Path(command[command.index("--out") + 1])
        job = jobs_by_output[str(output)]
        output.mkdir()
        if stage in {"T", "Q", "P"}:
            checkpoint = output / "checkpoint.pt"
            checkpoint.write_bytes(f"manufactured {stage} checkpoint".encode())
            dependencies = {}
            for parent_stage, reference in job["stage_artifact_inputs"].items():
                path = Path(reference["path"])
                dependencies[parent_stage] = {"path": str(path), "sha256": file_sha256(path)}
            hashes = {"teacher_state_sha256": "a" * 64}
            if stage == "Q":
                hashes["codebook_sha256"] = "b" * 64
                hashes["sampling_phase_sha256"] = "d" * 64
            if stage == "P":
                hashes["cache_sha256"] = "c" * 64
            write_stage_manifest(output / f"{stage}.artifact.json", stage=stage,
                                 checkpoint_path=checkpoint, identity=job["artifact_identity"], source_config={},
                                 dependencies=dependencies, content_hashes=hashes)
        return SimpleNamespace(returncode=0, stdout=json.dumps({"status": "COMPLETED", "completed_stages": [stage]}))

    return plan_path, plan, api, runner, commands


def test_mock_shared_runner_reuses_tqp_and_pins_identical_manifests(tmp_path):
    path, plan, api, runner, commands = _fixture(tmp_path)
    result = execute_shared_campaign(path, plan, setting="v1", device="cuda", runner=runner, api=api)
    assert set(result["results"]) == {"json", "csv", "markdown"}
    assert all(Path(value).is_file() for value in result["results"].values())
    execute_shared_campaign(path, plan, setting="v1u", device="cuda", runner=runner, api=api)
    assert [cmd[cmd.index("--stage") + 1] for cmd in commands] == ["T", "Q", "P", "J", "J"]
    joint = commands[-2:]
    pins = [[cmd[i + 1] for i, value in enumerate(cmd) if value == "--q3d-stage-artifact-sha"] for cmd in joint]
    assert pins[0] == pins[1]
    assert len(pins[0]) == 3
    state = json.loads((tmp_path / "RUN_STATE.json").read_text())
    assert state["experiments"]["v1"]["stage_artifacts"] == state["experiments"]["v1u"]["stage_artifacts"]


def test_mock_shared_runner_selects_k_roots_before_dependency_closure_and_reuses_artifacts(tmp_path):
    path, plan, api, runner, commands = _fixture(tmp_path, ks=(5, 8, 10))
    teacher_only = execute_shared_campaign(
        path, plan, setting="v1", k=(8,), through_stage="T",
        device="cuda", runner=runner, api=api,
    )
    assert teacher_only["status"] == "PARTIAL"
    assert teacher_only["selected_k"] == [8]
    assert teacher_only["through_stage"] == "T"
    assert [cmd[cmd.index("--out") + 1].rsplit("/", 1)[-1] for cmd in commands] == ["T8"]
    state = json.loads((tmp_path / "RUN_STATE.json").read_text())
    assert all(state["jobs"][job_id]["status"] == "NOT_RUN" for job_id in ("T5", "Q5", "P5", "T10", "Q10", "P10"))
    assert state["jobs"]["v1@5.J"]["status"] == "NOT_RUN"
    assert state["jobs"]["v1@10.J"]["status"] == "NOT_RUN"

    producers = execute_shared_campaign(
        path, plan, setting="v1", k=(8,), through_stage="P",
        device="cuda", runner=runner, api=api,
    )
    assert producers["status"] == "PARTIAL"
    assert producers["selected_k"] == [8]
    assert producers["through_stage"] == "P"
    assert [cmd[cmd.index("--out") + 1].rsplit("/", 1)[-1] for cmd in commands] == [
        "T8", "Q8", "P8",
    ]

    completed_v1 = execute_shared_campaign(
        path, plan, setting="v1", k=(8,), device="cuda", runner=runner, api=api,
    )
    assert completed_v1["status"] == "COMPLETED"
    assert completed_v1["selected_k"] == [8]
    assert completed_v1["through_stage"] == "J"
    assert [cmd[cmd.index("--out") + 1].rsplit("/", 1)[-1] for cmd in commands] == [
        "T8", "Q8", "P8", "v1@8",
    ]

    second = execute_shared_campaign(
        path, plan, setting="v1u", k=(8,), device="cuda", runner=runner, api=api,
    )
    assert second["selected_k"] == [8]
    assert [cmd[cmd.index("--out") + 1].rsplit("/", 1)[-1] for cmd in commands] == [
        "T8", "Q8", "P8", "v1@8", "v1u@8",
    ]
    joint_commands = commands[-2:]
    pins = [[cmd[i + 1] for i, value in enumerate(cmd) if value == "--q3d-stage-artifact-sha"]
            for cmd in joint_commands]
    assert pins[0] == pins[1]
    assert len(pins[0]) == 3
    state = json.loads((tmp_path / "RUN_STATE.json").read_text())
    assert state["experiments"]["v1@8"]["stage_artifacts"] == state["experiments"]["v1u@8"]["stage_artifacts"]


def test_k_selector_supports_multiple_values_and_rejects_empty_or_unmatched_selection(tmp_path):
    path, plan, api, runner, commands = _fixture(tmp_path, ks=(5, 8, 10))
    selected, selected_ks = select_campaign_experiments(plan["experiments"], setting="v1", k=(10, 5))
    assert [item["id"] for item in selected] == ["v1@5", "v1@10"]
    assert selected_ks == [5, 10]

    with pytest.raises(ValueError, match="cannot be empty"):
        execute_shared_campaign(path, plan, setting="v1", k=(), device="cuda", runner=runner, api=api)
    with pytest.raises(ValueError, match="no campaign experiments match"):
        execute_shared_campaign(path, plan, setting="v1", k=(2,), device="cuda", runner=runner, api=api)
    assert not (tmp_path / "RUN_STATE.json").exists()
    assert commands == []


def test_run_cli_parses_one_or_more_k_values(monkeypatch, tmp_path):
    from scripts import q3d_campaign

    called = {}

    def execute_campaign(path, **kwargs):
        called.update(kwargs)
        return {"status": "PLAN_VALIDATED_NOT_RUN"}

    monkeypatch.setattr(q3d_campaign, "execute_campaign", execute_campaign)
    q3d_campaign.main([
        "run", "--plan", str(tmp_path / "PLAN.json"), "--setting", "v1", "--k", "8", "10",
        "--through-stage", "P",
    ])
    assert called["setting"] == "v1"
    assert called["k"] == (8, 10)
    assert called["through_stage"] == "P"


def test_mock_shared_runner_rejects_changed_producer_before_next_arm(tmp_path):
    path, plan, api, runner, commands = _fixture(tmp_path)
    execute_shared_campaign(path, plan, setting="v1", device="cuda", runner=runner, api=api)
    manifest = tmp_path / "T" / "T.artifact.json"
    manifest.write_text(manifest.read_text() + "\n")
    with pytest.raises(ValueError, match="changed after production"):
        execute_shared_campaign(path, plan, setting="v1u", device="cuda", runner=runner, api=api)
    assert len(commands) == 4


def test_real_execution_rejects_synthetic_receipt(tmp_path):
    path, plan, api, runner, _ = _fixture(tmp_path)

    def synthetic_runner(command, **kwargs):
        result = runner(command, **kwargs)
        stage = command[command.index("--stage") + 1]
        result.stdout = json.dumps({
            "status": "COMPLETED", "completed_stages": [stage],
            "runtime_synthetic": True,
        })
        return result

    with pytest.raises(RuntimeError, match="synthetic receipt"):
        execute_shared_campaign(path, plan, setting="v1", device="cuda",
                                runner=synthetic_runner, api=api)


def test_result_collector_joins_selected_teacher_quality_and_c0_dev_score(tmp_path):
    """SOFTWARE_ONLY: manufactured sealed T and J artifacts, no training."""
    import torch

    identity = {"dataset": "acdc", "protocol": "A", "k": 8, "seed": 42}
    teacher_dir = tmp_path / "teacher"
    teacher_dir.mkdir()
    teacher_checkpoint = teacher_dir / "latest.pt"
    torch.save({
        "teacher_dev_fg_dice": 0.81,
        "teacher_terminal_dev_fg_dice": 0.91,
        "teacher_dev_per_class": {"RV": 0.75, "MYO": 0.80, "LV": 0.88},
    }, teacher_checkpoint)
    teacher_manifest = teacher_dir / "T.artifact.json"
    write_stage_manifest(
        teacher_manifest, stage="T", checkpoint_path=teacher_checkpoint,
        identity=identity, source_config={}, dependencies={},
        content_hashes={"teacher_state_sha256": "a" * 64},
    )

    c0_dir = tmp_path / "c0"
    c0_dir.mkdir()
    torch.save({"best_student_score": 0.66}, c0_dir / "latest.pt")
    plan = {
        "stage_jobs": [
            {
                "id": "T", "stage": "T", "artifact_identity": identity,
                "artifact_output": {"manifest_path": str(teacher_manifest)},
            },
            {"id": "v1.J", "stage": "J", "output": str(tmp_path / "v1")},
            {"id": "c0.J", "stage": "J", "output": str(c0_dir)},
        ],
        "experiments": [
            {
                "id": "v1", "setting": "v1", **identity,
                "stage_artifact_producers": {"T": "T", "J": "v1.J"},
            },
            {
                "id": "c0", "setting": "c0", **identity,
                "stage_artifact_producers": {"J": "c0.J"},
            },
        ],
    }
    state = {
        "jobs": {
            "T": {"status": "COMPLETED", "artifact": {
                "path": str(teacher_manifest), "sha256": file_sha256(teacher_manifest),
            }},
            "v1.J": {"status": "NOT_RUN"},
            "c0.J": {"status": "COMPLETED"},
        },
        "experiments": {"v1": {"status": "NOT_RUN"}, "c0": {"status": "COMPLETED"}},
    }

    report = collect_campaign_results(tmp_path / "PLAN.json", plan=plan, state=state)
    assert report["status"] == "MEASURED"
    row = report["rows"][0]
    assert row["teacher_dev_fg_dice"] == pytest.approx(0.81)
    assert row["teacher_status"] == "MEASURED"
    assert row["c0_dev_fg_dice"] == pytest.approx(0.66)
    assert row["c0_status"] == "MEASURED"
    assert row["delta_c0_minus_teacher"] == pytest.approx(-0.15)
    assert row["teacher_dev_per_class"]["LV"] == pytest.approx(0.88)

    paths = write_campaign_results(report, tmp_path / "results")
    assert set(paths) == {"json", "csv", "markdown"}
    assert "teacher DEV Dice@K" in Path(paths["markdown"]).read_text(encoding="utf-8")
    assert "c0_dev_fg_dice" in Path(paths["csv"]).read_text(encoding="utf-8")


def test_result_collector_keeps_missing_quality_not_run(tmp_path):
    identity = {"dataset": "acdc", "protocol": "A", "k": 5, "seed": 42}
    teacher_dir = tmp_path / "teacher"
    teacher_dir.mkdir()
    checkpoint = teacher_dir / "latest.pt"
    checkpoint.write_bytes(b"fixture without selected quality")
    manifest = teacher_dir / "T.artifact.json"
    write_stage_manifest(
        manifest, stage="T", checkpoint_path=checkpoint, identity=identity,
        source_config={}, dependencies={}, content_hashes={"teacher_state_sha256": "a" * 64},
    )
    plan = {
        "stage_jobs": [{
            "id": "T", "stage": "T", "artifact_identity": identity,
            "artifact_output": {"manifest_path": str(manifest)},
        }],
        "experiments": [{
            "id": "v1", "setting": "v1", **identity,
            "stage_artifact_producers": {"T": "T"},
        }],
    }
    state = {"jobs": {"T": {"status": "COMPLETED", "artifact": {
        "path": str(manifest), "sha256": file_sha256(manifest),
    }}}, "experiments": {"v1": {"status": "COMPLETED"}}}
    report = collect_campaign_results(tmp_path / "PLAN.json", plan=plan, state=state)
    row = report["rows"][0]
    assert report["status"] == "NOT_RUN"
    assert row["teacher_dev_fg_dice"] is None
    assert row["teacher_status"] == "NOT_RUN"
    assert row["c0_dev_fg_dice"] is None
    assert row["c0_status"] == "NOT_RUN"


def test_result_collector_does_not_skip_missing_teacher_checkpoint(tmp_path):
    identity = {"dataset": "acdc", "protocol": "A", "k": 5, "seed": 42}
    teacher_dir = tmp_path / "teacher"
    teacher_dir.mkdir()
    checkpoint = teacher_dir / "latest.pt"
    checkpoint.write_bytes(b"checkpoint")
    manifest = teacher_dir / "T.artifact.json"
    write_stage_manifest(
        manifest, stage="T", checkpoint_path=checkpoint, identity=identity,
        source_config={}, dependencies={}, content_hashes={"teacher_state_sha256": "a" * 64},
    )
    checkpoint.unlink()
    plan = {
        "stage_jobs": [{
            "id": "T", "stage": "T", "artifact_identity": identity,
            "artifact_output": {"manifest_path": str(manifest)},
        }],
        "experiments": [{
            "id": "v1", "setting": "v1", **identity,
            "stage_artifact_producers": {"T": "T"},
        }],
    }
    state = {"jobs": {"T": {"status": "COMPLETED", "artifact": {
        "path": str(manifest), "sha256": file_sha256(manifest),
    }}}, "experiments": {"v1": {"status": "COMPLETED"}}}
    with pytest.raises(ValueError, match="missing checkpoint"):
        collect_campaign_results(tmp_path / "PLAN.json", plan=plan, state=state)
