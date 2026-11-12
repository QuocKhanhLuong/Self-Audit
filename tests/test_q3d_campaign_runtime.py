"""SOFTWARE_ONLY: mock stage runner, no training/campaign subprocesses."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from self_audit.candidate_q3d.artifacts import file_sha256, write_stage_manifest
from self_audit.candidate_q3d.campaign_runtime import execute_shared_campaign


def _fixture(tmp_path):
    identity = {"dataset": "acdc", "protocol": "A", "k": 2, "seed": 19}
    jobs, experiments = [], []
    for stage in ("T", "Q", "P"):
        dependencies = [] if stage == "T" else (["T"] if stage == "Q" else ["T", "Q"])
        jobs.append({
            "id": stage, "stage": stage, "dependencies": dependencies,
            "experiment_id": "v1", "method": "candidate_q3d", "readiness": "READY", "reason": None,
            "output": str(tmp_path / stage), "artifact_identity": identity,
            "train_manifest": {}, "development_manifest": {},
            "artifact_output": {"manifest_path": str(tmp_path / stage / f"{stage}.artifact.json")},
            "stage_artifact_inputs": {dep: {"producer_job_id": dep, "path": str(tmp_path / dep / f"{dep}.artifact.json")}
                                      for dep in dependencies},
        })
    for setting in ("v1", "v1u"):
        config_path = tmp_path / f"{setting}.json"
        config_path.write_text(json.dumps({"method": "candidate_q3d", "arm": setting}))
        experiments.append({
            "id": setting, "setting": setting, "arm": setting, **identity,
            "stages": ["T", "Q", "P", "J"], "stage_job_ids": ["T", "Q", "P", f"{setting}.J"],
            "stage_artifact_producers": {"T": "T", "Q": "Q", "P": "P", "J": f"{setting}.J"},
            "config_path": str(config_path), "output": str(tmp_path / setting),
        })
        jobs.append({
            "id": f"{setting}.J", "stage": "J", "dependencies": ["T", "Q", "P"],
            "experiment_id": setting, "method": "candidate_q3d", "readiness": "READY", "reason": None,
            "output": str(tmp_path / setting), "train_manifest": {}, "development_manifest": {},
            "stage_artifact_inputs": {stage: {"producer_job_id": stage, "path": str(tmp_path / stage / f"{stage}.artifact.json")}
                                      for stage in ("T", "Q", "P")},
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

    def runner(command, **_kwargs):
        commands.append(command)
        stage = command[command.index("--stage") + 1]
        output = Path(command[command.index("--out") + 1])
        output.mkdir()
        if stage in {"T", "Q", "P"}:
            checkpoint = output / "checkpoint.pt"
            checkpoint.write_bytes(f"manufactured {stage} checkpoint".encode())
            parents = [] if stage == "T" else (["T"] if stage == "Q" else ["T", "Q"])
            dependencies = {}
            for parent in parents:
                path = tmp_path / parent / f"{parent}.artifact.json"
                dependencies[parent] = {"path": str(path), "sha256": file_sha256(path)}
            hashes = {"teacher_state_sha256": "a" * 64}
            if stage == "Q":
                hashes["codebook_sha256"] = "b" * 64
                hashes["sampling_phase_sha256"] = "d" * 64
            if stage == "P":
                hashes["cache_sha256"] = "c" * 64
            write_stage_manifest(output / f"{stage}.artifact.json", stage=stage,
                                 checkpoint_path=checkpoint, identity=identity, source_config={},
                                 dependencies=dependencies, content_hashes=hashes)
        return SimpleNamespace(returncode=0, stdout=json.dumps({"status": "COMPLETED", "completed_stages": [stage]}))

    return plan_path, plan, api, runner, commands


def test_mock_shared_runner_reuses_tqp_and_pins_identical_manifests(tmp_path):
    path, plan, api, runner, commands = _fixture(tmp_path)
    execute_shared_campaign(path, plan, setting="v1", device="cuda", runner=runner, api=api)
    execute_shared_campaign(path, plan, setting="v1u", device="cuda", runner=runner, api=api)
    assert [cmd[cmd.index("--stage") + 1] for cmd in commands] == ["T", "Q", "P", "J", "J"]
    joint = commands[-2:]
    pins = [[cmd[i + 1] for i, value in enumerate(cmd) if value == "--q3d-stage-artifact-sha"] for cmd in joint]
    assert pins[0] == pins[1]
    assert len(pins[0]) == 3
    state = json.loads((tmp_path / "RUN_STATE.json").read_text())
    assert state["experiments"]["v1"]["stage_artifacts"] == state["experiments"]["v1u"]["stage_artifacts"]


def test_mock_shared_runner_rejects_changed_producer_before_next_arm(tmp_path):
    path, plan, api, runner, commands = _fixture(tmp_path)
    execute_shared_campaign(path, plan, setting="v1", device="cuda", runner=runner, api=api)
    manifest = tmp_path / "T" / "T.artifact.json"
    manifest.write_text(manifest.read_text() + "\n")
    with pytest.raises(ValueError, match="changed after production"):
        execute_shared_campaign(path, plan, setting="v1u", device="cuda", runner=runner, api=api)
    assert len(commands) == 4
