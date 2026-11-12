"""SOFTWARE_ONLY: realized U permutation weights, including interrupted resume."""

import copy
import math

import pytest

from self_audit.candidate_q3d import CHECKPOINT_SCHEMA, engine
from tests.test_q3d_engine import _execute_joint_stage
from tests.test_q3d_engine_weighting import (
    _case_with_frozen_rows, _controlled_confidence_targets, _resume_cache_state,
)


def _cycle_events(state):
    return [row for row in state["metrics"]
            if row.get("event") == "physiology_weighting_full_u_cycle"]


@pytest.mark.parametrize("arm", ["v1", "v1u", "c1"])
def test_engine_logs_actual_permutation_mean_and_resumes_mid_cycle(tmp_path, monkeypatch, arm):
    _controlled_confidence_targets(monkeypatch)
    case, modules, initial = _case_with_frozen_rows(tmp_path, arm, n_unlabeled=3)
    case["config"]["training"]["stage_updates"]["J"] = 6
    rows = [
        {"id": patient, "w_pat": row["w_pat"], "protocol": "A",
         "teacher_sha256": engine._state_sha256(modules["teacher"]), "codebook_sha256": None}
        for patient, row in initial["physiology_rows"].items()
    ]
    _, initial, _ = _resume_cache_state(tmp_path, case, rows)
    uninterrupted = _execute_joint_stage(
        case, tmp_path / "whole", copy.deepcopy(modules), state=copy.deepcopy(initial),
    )
    interrupted = _execute_joint_stage(
        case, tmp_path / "resumed", copy.deepcopy(modules),
        state=copy.deepcopy(initial), update_limit=2,
    )
    assert len(interrupted["j_u_patient_cycle"]["observations"]) == 2
    assert not _cycle_events(interrupted)
    # Round-trip the production checkpoint, not just a Python dict copy.
    loaded = engine._load_checkpoint(
        tmp_path / "resumed" / "latest.pt",
        config_sha256=interrupted["config_sha256"], data_identity=interrupted["data_identity"],
    )
    resumed = _execute_joint_stage(
        case, tmp_path / "resumed", copy.deepcopy(modules), state=loaded, update_limit=4,
    )
    assert _cycle_events(uninterrupted) == _cycle_events(resumed)
    assert uninterrupted["j_u_patient_cycle"] == resumed["j_u_patient_cycle"] == {
        "observations": [], "completed_cycles": 2,
    }
    assert uninterrupted["sampler_state"] == resumed["sampler_state"]
    updates = [row for row in resumed["metrics"] if "u_patient_omega" in row]
    events = _cycle_events(resumed)
    assert len(events) == 2
    for cycle, event in enumerate(events):
        consumed = updates[cycle * 3:(cycle + 1) * 3]
        assert {row["sampler"]["unlabeled_patient"] for row in consumed} == set(case["data"].unlabeled_patients)
        actual_mean = math.fsum(row["u_patient_omega"] for row in consumed) / 3
        assert event["realized_mean_omega"] == actual_mean
        assert abs(actual_mean - 1.0) <= 1e-12
        assert event["patient_count"] == 3
        assert event["cycle"] == cycle + 1


def test_realized_mean_uses_observations_across_online_cache_refresh():
    state = {"physiology_cache": {"sha256": "old"}}
    sample = {"unlabeled_cycle": 1, "unlabeled_cycle_position": 1,
              "unlabeled_cycle_complete": False}
    engine._update_u_patient_cycle(state, "a", ["a", "b"], 1.8, sample)
    state["physiology_cache"] = {"sha256": "new"}
    state["physiology_cohort_weighting"] = {"mean_omega": 1.0}
    result = engine._update_u_patient_cycle(
        state, "b", ["a", "b"], 0.8,
        {**sample, "unlabeled_cycle_position": 2, "unlabeled_cycle_complete": True},
    )
    assert result["realized_mean_omega"] == 1.3
    assert result["snapshot"] is None
    assert result["snapshots"] == ["old", "new"]


def test_cycle_rejects_repeated_patient_instead_of_hiding_it():
    state = {}
    sample = {"unlabeled_cycle": 1, "unlabeled_cycle_position": 1,
              "unlabeled_cycle_complete": False}
    engine._update_u_patient_cycle(state, "a", ["a", "b"], 1.0, sample)
    with pytest.raises(ValueError, match="repeated"):
        engine._update_u_patient_cycle(
            state, "a", ["a", "b"], 1.0,
            {**sample, "unlabeled_cycle_position": 2, "unlabeled_cycle_complete": True},
        )


def test_engine_rejects_resume_v2_with_explicit_stream_error(tmp_path):
    assert engine.STATE_SCHEMA == CHECKPOINT_SCHEMA == "self_audit.candidate_q3d.resume.v3"
    state = engine._initial_run_state("config", {})
    state["schema"] = "self_audit.candidate_q3d.resume.v2"
    path = tmp_path / "retired.pt"
    engine._atomic_torch_save(state, path)
    with pytest.raises(ValueError, match=r"resume.v2.*retired U sampling stream"):
        engine._load_checkpoint(path, config_sha256="config", data_identity={})
