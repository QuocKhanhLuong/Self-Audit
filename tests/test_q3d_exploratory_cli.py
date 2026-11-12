"""SOFTWARE_ONLY: explicit exploratory quality-waiver CLI/API wiring.

These tests use manufactured trainers/runners only. They check that the
default strict path is unchanged and that an explicit waiver reason is
forwarded, preserved, required again on reuse, and never relabels gates.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

import train as train_cli
from scripts import q3d_campaign
from self_audit.candidate_q3d import campaign_runtime, profiling
from self_audit.candidate_q3d.campaign_runtime import (
    EXPLORATORY_WAIVER_FLAG,
    EXPLORATORY_WAIVER_SCHEMA,
    execute_shared_campaign,
    exploratory_marker,
    exploratory_waiver_policy,
)
from tests.test_q3d_campaign_runtime import _fixture
from tests.test_q3d_profiling import FakeClock, FakeCuda, FakeData, FakeOptimizer, FakeSampler, FakeTorch, _inputs

ROOT = Path(__file__).resolve().parents[1]
REASON = "overnight exploratory run authorized by user"


# ---------------------------------------------------------------- policy


def test_policy_shape_and_strict_default():
    assert exploratory_waiver_policy(None) is None
    policy = exploratory_waiver_policy(f"  {REASON} ")
    assert policy == {
        "schema": EXPLORATORY_WAIVER_SCHEMA,
        "reason": REASON,
        "scope": ["tokenizer_gate", "physiology_support", "kd_support"],
        "exploratory": True,
    }
    for bad in ("", "   ", 3, b"x"):
        with pytest.raises(ValueError, match="nonempty REASON"):
            exploratory_waiver_policy(bad)


def test_marker_reads_declarations_and_fails_closed_on_contradictions():
    policy = exploratory_waiver_policy(REASON)
    assert exploratory_marker({"status": "COMPLETED"}) is None
    assert exploratory_marker({"exploratory": False}) is None
    digest = campaign_runtime.waiver_policy_sha256(policy)
    assert exploratory_marker({"exploratory": True}) == {
        "exploratory": True, "policy": None, "policy_sha256": None,
    }
    assert exploratory_marker({"exploratory": True, "exploratory_quality_waiver": policy}) == {
        "exploratory": True, "policy": policy, "policy_sha256": digest,
    }
    # Stage manifests seal the checkpoint policy only through content_hashes.
    sealed = {"stage": "Q", "content_hashes": {"exploratory_quality_waiver_sha256": digest}}
    assert exploratory_marker(sealed) == {"exploratory": True, "policy": None, "policy_sha256": digest}
    with pytest.raises(ValueError, match="sealed content hash"):
        exploratory_marker({"exploratory": True, "exploratory_quality_waiver": policy,
                            "content_hashes": {"exploratory_quality_waiver_sha256": "0" * 64}})
    with pytest.raises(ValueError, match="exploratory=false"):
        exploratory_marker({"exploratory": False, "exploratory_quality_waiver": policy})
    with pytest.raises(ValueError, match="malformed"):
        exploratory_marker({"exploratory": True, "exploratory_quality_waiver": {"reason": ""}})
    with pytest.raises(ValueError, match="malformed"):
        exploratory_marker({"exploratory": True, "exploratory_quality_waiver": {**policy, "reason": " x "}})
    with pytest.raises(ValueError, match="boolean"):
        exploratory_marker({"exploratory": "yes"})


# ---------------------------------------------------------------- train.py


def _q3d_args(**overrides):
    values = dict(
        model_config=None, teacher_checkpoint=None, teacher_checkpoint_sha256=None,
        data=None, data_sha256=None, validation_data=None, validation_data_sha256=None,
        out="/tmp/unused-q3d-out", synthetic=True, allow_full_run=False, device="cpu",
        q3d_stage="J", q3d_stage_artifact=None, q3d_stage_artifact_sha=None,
        q3d_exploratory_quality_waiver=None,
    )
    values.update(overrides)
    return argparse.Namespace(**values)


@pytest.fixture
def fake_q3d_train(monkeypatch):
    from self_audit.candidate_q3d import config as q3d_config
    from self_audit.candidate_q3d import engine

    calls = []
    validated = {"arm": "v1", "runtime": {"synthetic": True, "synthetic_gate_override": False}}
    monkeypatch.setattr(q3d_config, "validate_config", lambda config, **_kwargs: validated)

    def fake_train(config, output_dir, **kwargs):
        calls.append(kwargs)
        return {"status": "COMPLETED"}

    monkeypatch.setattr(engine, "train", fake_train)
    return calls


def test_train_default_call_does_not_forward_waiver(fake_q3d_train):
    parser = argparse.ArgumentParser()
    train_cli._train_q3d(_q3d_args(), parser, {"method": "candidate_q3d"})
    assert "exploratory_quality_waiver" not in fake_q3d_train[0]


def test_train_forwards_explicit_reason_to_engine(fake_q3d_train):
    parser = argparse.ArgumentParser()
    train_cli._train_q3d(
        _q3d_args(q3d_exploratory_quality_waiver=f" {REASON} "), parser, {"method": "candidate_q3d"},
    )
    assert fake_q3d_train[0]["exploratory_quality_waiver"] == REASON
    assert fake_q3d_train[0]["synthetic_gate_override"] is False


def test_train_rejects_blank_reason(fake_q3d_train):
    parser = argparse.ArgumentParser()
    with pytest.raises(SystemExit):
        train_cli._train_q3d(
            _q3d_args(q3d_exploratory_quality_waiver="   "), parser, {"method": "candidate_q3d"},
        )
    assert fake_q3d_train == []


@pytest.mark.parametrize("script", ["train.py", "scripts/q3d_profile_gpu.py"])
def test_cli_help_exposes_waiver_flag(script, tmp_path):
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    result = subprocess.run(
        [sys.executable, str(ROOT / script), "--help"], cwd=tmp_path, env=environment,
        capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, result.stderr
    assert EXPLORATORY_WAIVER_FLAG in result.stdout


def test_campaign_run_help_exposes_waiver_flag(capsys):
    with pytest.raises(SystemExit):
        q3d_campaign.main(["run", "--help"])
    assert EXPLORATORY_WAIVER_FLAG in capsys.readouterr().out


# ---------------------------------------------------------------- campaign


def _experiment():
    return {"output": "/tmp/q3d-out", "arm": "v1"}


def test_job_command_default_unchanged_and_waiver_appends_exact_flag(tmp_path):
    kwargs = dict(config_path=tmp_path / "c.json", train_ref={}, dev_ref={}, device="cuda",
                  synthetic=False, stage="J", stage_artifacts={"Q": "/tmp/Q.json"})
    strict = q3d_campaign._job_command(_experiment(), **kwargs)
    assert EXPLORATORY_WAIVER_FLAG not in strict
    explicit_none = q3d_campaign._job_command(_experiment(), exploratory_quality_waiver=None, **kwargs)
    assert explicit_none == strict
    waived = q3d_campaign._job_command(_experiment(), exploratory_quality_waiver=REASON, **kwargs)
    assert waived == [*strict, EXPLORATORY_WAIVER_FLAG, REASON]
    with pytest.raises(ValueError, match="nonempty"):
        q3d_campaign._job_command(_experiment(), exploratory_quality_waiver=" ", **kwargs)


def test_execute_campaign_forwards_waiver_to_shared_runtime(monkeypatch, tmp_path):
    calls = []
    plan = {"selection": {"shared_stage_dag": True, "synthetic": False}, "experiments": [
        {"id": "v1", "setting": "v1", "k": 8}], "stage_jobs": []}
    monkeypatch.setattr(q3d_campaign, "validate_campaign_plan", lambda _path: plan)

    def fake_shared(*_args, **kwargs):
        calls.append(kwargs)
        return {"status": "COMPLETED"}

    monkeypatch.setattr(campaign_runtime, "execute_shared_campaign", fake_shared)
    q3d_campaign.execute_campaign(tmp_path / "PLAN.json", execute=True, setting="v1")
    q3d_campaign.execute_campaign(tmp_path / "PLAN.json", execute=True, setting="v1",
                                  exploratory_quality_waiver=REASON)
    assert "exploratory_quality_waiver" not in calls[0]
    assert calls[1]["exploratory_quality_waiver"] == REASON
    validated = q3d_campaign.execute_campaign(tmp_path / "PLAN.json", setting="v1",
                                              exploratory_quality_waiver=REASON)
    assert validated["status"] == "PLAN_VALIDATED_NOT_RUN"
    assert validated["exploratory_quality_waiver"]["reason"] == REASON


def _waived_fixture(tmp_path, *, exploratory_receipt=True):
    path, plan, api, runner, commands = _fixture(tmp_path)
    command_kwargs = []
    original_command = api._job_command

    def recording_command(experiment, **kwargs):
        command_kwargs.append(kwargs)
        command = original_command(experiment, **kwargs)
        if kwargs.get("exploratory_quality_waiver") is not None:
            command.extend([EXPLORATORY_WAIVER_FLAG, kwargs["exploratory_quality_waiver"]])
        return command

    api._job_command = recording_command

    def waived_runner(command, **kwargs):
        result = runner(command, **kwargs)
        receipt = json.loads(result.stdout)
        stage = receipt["completed_stages"][0]
        if exploratory_receipt and stage in {"Q", "J"}:
            receipt.update({
                "evidence_status": "TRAIN_EXECUTION",
                "runtime_synthetic": False,
                "tokenizer_gate": {"passed": False, "status": "NO_GO"},
                "exploratory": True,
                "exploratory_quality_waiver": exploratory_waiver_policy(REASON),
            })
        if stage == "J":
            receipt["student_dev_fg_dice"] = 0.5
        result.stdout = json.dumps(receipt)
        return result

    return path, plan, api, waived_runner, commands, command_kwargs


def test_shared_campaign_default_is_strict(tmp_path):
    path, plan, api, runner, _commands, command_kwargs = _waived_fixture(tmp_path, exploratory_receipt=False)
    result = execute_shared_campaign(path, plan, setting="v1", device="cuda", runner=runner, api=api)
    assert all("exploratory_quality_waiver" not in kwargs for kwargs in command_kwargs)
    assert "exploratory" not in result
    header = Path(result["results"]["csv"]).read_text().splitlines()[0]
    assert header.split(",")[-1] == "teacher_dev_per_class"


def test_shared_campaign_rejects_exploratory_receipt_without_waiver(tmp_path):
    path, plan, api, runner, _commands, _kwargs = _waived_fixture(tmp_path)
    with pytest.raises(RuntimeError, match="without an explicit waiver"):
        execute_shared_campaign(path, plan, setting="v1", device="cuda", runner=runner, api=api)


def test_shared_campaign_waiver_forwards_preserves_and_labels(tmp_path):
    path, plan, api, runner, commands, command_kwargs = _waived_fixture(tmp_path)
    result = execute_shared_campaign(path, plan, setting="v1", device="cuda", runner=runner, api=api,
                                     exploratory_quality_waiver=REASON)
    assert [kwargs["exploratory_quality_waiver"] for kwargs in command_kwargs] == [REASON] * 4
    assert all(cmd[cmd.index(EXPLORATORY_WAIVER_FLAG) + 1] == REASON for cmd in commands)
    assert result["exploratory"] is True
    assert result["exploratory_quality_waiver"]["reason"] == REASON
    state = json.loads(Path(result["run_state"]).read_text())
    q_record = state["jobs"]["Q"]
    assert q_record["exploratory"] is True
    assert q_record["exploratory_quality_waiver"]["schema"] == EXPLORATORY_WAIVER_SCHEMA
    # The measured gate stays NO_GO and real evidence stays TRAIN_EXECUTION.
    assert q_record["receipt"]["tokenizer_gate"] == {"passed": False, "status": "NO_GO"}
    assert q_record["receipt"]["evidence_status"] == "TRAIN_EXECUTION"
    assert "exploratory" not in state["jobs"]["T"]
    assert state["experiments"]["v1"]["exploratory"] is True
    assert set(state["experiments"]["v1"]["exploratory_jobs"]) == {"Q", "v1.J"}
    report = json.loads(Path(result["results"]["json"]).read_text())
    assert report["exploratory"] is True
    assert report["rows"][0]["exploratory"] is True
    assert report["rows"][0]["teacher_exploratory"] is False
    assert report["rows"][0]["c0_status"] == "NOT_RUN"
    assert {source["job_id"] for source in report["exploratory_sources"]} == {"Q", "v1.J"}
    assert "exploratory" in Path(result["results"]["csv"]).read_text().splitlines()[0]


def test_shared_campaign_rejects_receipt_with_different_reason(tmp_path):
    path, plan, api, runner, _commands, _kwargs = _waived_fixture(tmp_path)
    with pytest.raises(ValueError, match="differs"):
        execute_shared_campaign(path, plan, setting="v1", device="cuda", runner=runner, api=api,
                                exploratory_quality_waiver="different supplied reason")
    state = json.loads((path.parent / "RUN_STATE.json").read_text())
    assert state["jobs"]["Q"]["status"] == "FAILED"


def test_shared_campaign_reuse_of_waived_producers_requires_reason_again(tmp_path):
    path, plan, api, runner, _commands, _kwargs = _waived_fixture(tmp_path)
    execute_shared_campaign(path, plan, setting="v1", device="cuda", runner=runner, api=api,
                            exploratory_quality_waiver=REASON)
    with pytest.raises(ValueError, match=EXPLORATORY_WAIVER_FLAG):
        execute_shared_campaign(path, plan, setting="v1u", device="cuda", runner=runner, api=api)
    with pytest.raises(ValueError, match="differs"):
        execute_shared_campaign(path, plan, setting="v1u", device="cuda", runner=runner, api=api,
                                exploratory_quality_waiver="a different reason")
    result = execute_shared_campaign(path, plan, setting="v1u", device="cuda", runner=runner, api=api,
                                     exploratory_quality_waiver=REASON)
    assert result["status"] == "COMPLETED"


def test_results_report_marks_exploratory_rows(tmp_path):
    report = {
        "schema": "self_audit.candidate_q3d.campaign_results.v1", "status": "MEASURED",
        "join_key": ["dataset", "protocol", "k", "seed"], "exploratory": True,
        "rows": [{"dataset": "acdc", "protocol": "A", "k": 8, "seed": 42,
                  "teacher_dev_fg_dice": None, "teacher_status": "NOT_RUN",
                  "c0_dev_fg_dice": 0.4, "c0_status": "MEASURED", "delta_c0_minus_teacher": None,
                  "teacher_dev_per_class": None, "teacher_exploratory": False,
                  "c0_exploratory": True, "exploratory": True}],
    }
    paths = campaign_runtime.write_campaign_results(report, tmp_path)
    with open(paths["csv"], newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["exploratory"] == "True" and rows[0]["c0_exploratory"] == "True"
    markdown = Path(paths["markdown"]).read_text()
    assert "EXPLORATORY" in markdown and "| evidence |" in markdown


def test_collector_labels_rows_from_state(tmp_path):
    identity = {"dataset": "acdc", "protocol": "A", "k": 8, "seed": 42}
    policy = exploratory_waiver_policy(REASON)
    plan = {
        "stage_jobs": [{"id": "c0.J", "stage": "J", "output": str(tmp_path / "c0")}],
        "experiments": [{"id": "c0", "setting": "c0", **identity,
                         "stage_artifact_producers": {"J": "c0.J"}}],
    }
    state = {"jobs": {"c0.J": {"status": "COMPLETED", "exploratory": True,
                               "exploratory_quality_waiver": policy,
                               "receipt": {"status": "COMPLETED", "student_dev_fg_dice": 0.4,
                                           "exploratory": True,
                                           "exploratory_quality_waiver": policy}}},
             "experiments": {"c0": {"status": "COMPLETED"}}}
    report = campaign_runtime.collect_campaign_results(tmp_path / "PLAN.json", plan=plan, state=state)
    assert report["exploratory"] is True
    assert report["rows"][0]["c0_exploratory"] is True and report["rows"][0]["exploratory"] is True
    assert report["exploratory_sources"][0]["job_id"] == "c0.J"
    state["jobs"]["c0.J"] = {"status": "COMPLETED", "receipt": {"status": "COMPLETED",
                                                                "student_dev_fg_dice": 0.4}}
    strict = campaign_runtime.collect_campaign_results(tmp_path / "PLAN.json", plan=plan, state=state)
    assert strict["exploratory"] is False and "exploratory_sources" not in strict


# ---------------------------------------------------------------- profiling


def _bindings(tmp_path, *, exploratory_q):
    paths, hashes = [], []
    for stage in ("T", "Q", "P"):
        document = {"stage": stage}
        if stage == "Q" and exploratory_q:
            digest = campaign_runtime.waiver_policy_sha256(exploratory_waiver_policy(REASON))
            document["content_hashes"] = {"exploratory_quality_waiver_sha256": digest}
        path = tmp_path / f"{stage}.json"
        path.write_text(json.dumps(document))
        paths.append(f"{stage}={path}")
        hashes.append(f"{stage}={profiling.sha256_file(path)}")
    return paths, hashes


def test_profile_bindings_require_reason_for_waived_artifacts(tmp_path):
    paths, hashes = _bindings(tmp_path, exploratory_q=True)
    with pytest.raises(profiling.ProfilePrerequisiteError, match=EXPLORATORY_WAIVER_FLAG):
        profiling.parse_stage_artifact_bindings(paths, hashes)
    with pytest.raises(profiling.ProfilePrerequisiteError, match="different"):
        profiling.parse_stage_artifact_bindings(paths, hashes, exploratory_quality_waiver="other reason")
    parsed = profiling.parse_stage_artifact_bindings(paths, hashes, exploratory_quality_waiver=REASON)
    assert parsed["Q"].exploratory is True
    assert parsed["Q"].as_dict()["exploratory"] is True
    assert parsed["Q"].as_dict()["exploratory_quality_waiver_sha256"] == (
        campaign_runtime.waiver_policy_sha256(exploratory_waiver_policy(REASON))
    )
    assert set(parsed["T"].as_dict()) == {"manifest_path", "manifest_sha256"}


def test_profile_strict_bindings_unchanged(tmp_path):
    paths, hashes = _bindings(tmp_path, exploratory_q=False)
    parsed = profiling.parse_stage_artifact_bindings(paths, hashes)
    assert all(not artifact.exploratory for artifact in parsed.values())


def _fake_engine(calls):
    data = FakeData()

    class FakeEngine:
        @staticmethod
        def _teacher_frame_choice(_data, _rng):
            return (0, 0)

        @staticmethod
        def _make_q3d_sampler(_data, _seed, _state, *, require_unlabeled=True):
            return FakeSampler()

        @staticmethod
        def train(config, output_dir, **kwargs):
            calls.append(kwargs)
            sampler = FakeEngine._make_q3d_sampler(data, 42, {}, require_unlabeled=True)
            for _ in range(kwargs["update_limit"]):
                sampler.sample_update()
                FakeOptimizer().step()
            return {"runtime_synthetic": False, "status": "PAUSED"}

    return FakeEngine


def test_profile_engine_stage_forwards_reason_only_when_explicit(tmp_path):
    from dataclasses import replace

    calls = []
    engine = _fake_engine(calls)
    strict_inputs = _inputs()
    measurement = profiling.UpdateMeasurement("cuda:0", cuda_api=FakeCuda(), clock=FakeClock())
    profiling.run_engine_stage(strict_inputs, "J", tmp_path / "strict", measurement, device="cuda:0",
                               engine_module=engine, optimizer_type=FakeOptimizer)
    assert "exploratory_quality_waiver" not in calls[-1]

    artifacts = dict(strict_inputs.stage_artifacts)
    artifacts["Q"] = replace(artifacts["Q"], exploratory=True,
                             exploratory_quality_waiver=exploratory_waiver_policy(REASON))
    unwaived = replace(strict_inputs, stage_artifacts=artifacts)
    measurement = profiling.UpdateMeasurement("cuda:0", cuda_api=FakeCuda(), clock=FakeClock())
    with pytest.raises(profiling.ProfilePrerequisiteError, match=EXPLORATORY_WAIVER_FLAG):
        profiling.run_engine_stage(unwaived, "J", tmp_path / "unwaived", measurement, device="cuda:0",
                                   engine_module=engine, optimizer_type=FakeOptimizer)

    waived = replace(unwaived, exploratory_quality_waiver=exploratory_waiver_policy(REASON))
    measurement = profiling.UpdateMeasurement("cuda:0", cuda_api=FakeCuda(), clock=FakeClock())
    profiling.run_engine_stage(waived, "J", tmp_path / "waived", measurement, device="cuda:0",
                               engine_module=engine, optimizer_type=FakeOptimizer)
    assert calls[-1]["exploratory_quality_waiver"] == REASON
    assert calls[-1]["synthetic_gate_override"] is False
    assert calls[-1]["smoke"] is False


def test_profile_rejects_in_memory_artifact_policy_with_different_reason():
    from dataclasses import replace

    artifacts = dict(_inputs().stage_artifacts)
    artifacts["Q"] = replace(artifacts["Q"], exploratory=True,
                             exploratory_quality_waiver=exploratory_waiver_policy(REASON))
    with pytest.raises(profiling.ProfilePrerequisiteError, match="different"):
        profiling._require_artifact_waivers(artifacts, exploratory_waiver_policy("different reason"))


def test_profile_inputs_require_same_explicit_reason(tmp_path):
    from dataclasses import replace

    inputs = replace(_inputs(), exploratory_quality_waiver=exploratory_waiver_policy(REASON))
    report = profiling.run_profile_software_test(
        inputs=inputs, report_path=tmp_path / "profile.json", scratch_root=tmp_path / "scratch",
        stage_runner=lambda *_args, **_kwargs: pytest.fail("must reject before execution"),
        cuda_api=FakeCuda(), torch_module=FakeTorch(FakeCuda()),
        exploratory_quality_waiver="different reason",
    )
    assert report["status"] == "SOFTWARE_ONLY_FAILURE"
    assert "different exploratory waiver REASON" in str(report["failure"])
