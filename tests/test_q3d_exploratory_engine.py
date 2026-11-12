"""Explicit exploratory quality waiver: engine contracts (synthetic, CPU only).

These are SOFTWARE_ONLY checks.  Default (no waiver) behavior stays strict.
"""

from __future__ import annotations

import copy
import math

import pytest
import torch

from self_audit.candidate_q3d.quality import (
    WAIVER_EVENTS_KEY,
    WAIVER_SCHEMA,
    WAIVER_SCOPE,
    WAIVER_STATE_KEY,
    bind_quality_waiver,
    exploratory_relative_omega,
    quality_waiver,
    validate_quality_waiver,
    waiver_covers,
)
from tests.test_q3d_engine import _execute_joint_stage, _joint_stage_case

REASON = "overnight exploratory full pipeline"


# ---------------------------------------------------------------- helpers


def test_quality_waiver_policy_is_explicit_and_canonical():
    assert quality_waiver(None) is None
    policy = quality_waiver(f"  {REASON}  ")
    assert policy == {
        "schema": WAIVER_SCHEMA, "reason": REASON,
        "scope": list(WAIVER_SCOPE), "exploratory": True,
    }
    assert WAIVER_SCHEMA == "q3d.exploratory_quality_waiver.v1"
    assert WAIVER_SCOPE == ("tokenizer_gate", "physiology_support", "kd_support")
    for bad in ("", "   "):
        with pytest.raises(ValueError):
            quality_waiver(bad)
    with pytest.raises(TypeError):
        quality_waiver(True)  # type: ignore[arg-type]
    assert validate_quality_waiver(policy) == policy
    forged = dict(policy, exploratory=False)
    with pytest.raises(ValueError):
        validate_quality_waiver(forged)
    with pytest.raises(ValueError):
        validate_quality_waiver(dict(policy, scope=["tokenizer_gate"]))
    with pytest.raises(ValueError):
        validate_quality_waiver({**policy, "extra": 1})
    assert waiver_covers(policy, "kd_support")
    assert not waiver_covers({"schema": "self_audit.candidate_q3d.resume.v3"}, "kd_support")
    assert not waiver_covers({WAIVER_STATE_KEY: None}, "tokenizer_gate")


def test_bind_requires_explicit_matching_reason_on_waived_state():
    strict_state: dict = {}
    assert bind_quality_waiver(strict_state, None, context="t") is None
    assert WAIVER_STATE_KEY not in strict_state and WAIVER_EVENTS_KEY not in strict_state

    state: dict = {}
    bind_quality_waiver(state, quality_waiver(REASON), context="start")
    assert state[WAIVER_STATE_KEY]["reason"] == REASON
    assert [event["event"] for event in state[WAIVER_EVENTS_KEY]] == ["exploratory_quality_waiver_bound"]
    with pytest.raises(RuntimeError, match="requires the same explicit"):
        bind_quality_waiver(copy.deepcopy(state), None, context="resume")
    with pytest.raises(RuntimeError, match="differs"):
        bind_quality_waiver(copy.deepcopy(state), quality_waiver("other reason"), context="resume")
    again = copy.deepcopy(state)
    bind_quality_waiver(again, quality_waiver(REASON), context="resume")
    assert len(again[WAIVER_EVENTS_KEY]) == 1  # re-binding the same policy is not a new event


def test_exploratory_omega_fallbacks_are_finite():
    low_ess = {"a": 1.0, "b": 0.0, "c": 0.0, "d": 0.0, "e": 0.0}
    result = exploratory_relative_omega(low_ess, "V1")
    assert result["fallback"] == "low_ess_relative_normalized"
    assert result["omega"] == {"a": 5.0, "b": 0.0, "c": 0.0, "d": 0.0, "e": 0.0}
    assert math.isclose(result["mean_omega"], 1.0, abs_tol=1e-12)
    zero = exploratory_relative_omega({"a": 0.0, "b": 0.0}, "C1")
    assert zero["fallback"] == "zero_mean_uniform_unit_omega"
    assert zero["omega"] == {"a": 1.0, "b": 1.0}
    v1u = exploratory_relative_omega(low_ess, "v1u")
    assert v1u["fallback"] == "v1u_unit_omega"
    assert set(v1u["omega"].values()) == {1.0}
    for item in (result, zero, v1u):
        assert all(math.isfinite(value) and value >= 0.0 for value in item["omega"].values())
    with pytest.raises(ValueError):
        exploratory_relative_omega({"a": float("nan")}, "V1")


# ---------------------------------------------------------- engine stages


def _waived_state(case, reason=REASON):
    import self_audit.candidate_q3d.engine as engine

    state = engine._initial_run_state(
        engine._object_sha256(case["config"]), engine._data_identity_without_labels(case["data"]),
    )
    bind_quality_waiver(state, quality_waiver(reason) if reason else None, context="test")
    return state


def _modules(case):
    return {name: case[name] for name in ("teacher", "student", "tokenizer", "ema")}


def test_c1_allzero_u_support_continues_with_explicit_uniform_fallback(tmp_path):
    case = _joint_stage_case(tmp_path, "c1")
    case["config"]["training"]["stage_updates"]["J"] = 1
    case["config"]["losses"]["w_mode"] = "physiology"
    strict = _waived_state(case, reason=None)
    strict["physiology_rows"] = {p: {"w_pat": 0.0} for p in case["data"].unlabeled_patients}
    with pytest.raises(RuntimeError, match="zero support for every TRAIN U patient"):
        _execute_joint_stage(case, tmp_path / "strict", copy.deepcopy(_modules(case)), state=strict)

    state = _waived_state(case)
    state["physiology_rows"] = {p: {"w_pat": 0.0} for p in case["data"].unlabeled_patients}
    state = _execute_joint_stage(case, tmp_path / "waived", _modules(case), state=state)
    assert state["j_optimizer_updates"] == 1
    gate = state["physiology_support_gate"]
    assert gate["status"] == "NO_GO_ALL_U_WEIGHTS_ZERO" and gate["passed"] is False
    assert gate["override_scope"] == "exploratory_quality_waiver"
    weighting = state["physiology_cohort_weighting"]
    assert weighting["gate_passed"] is False
    assert weighting["override_policy"] == "exploratory_zero_mean_uniform_unit_omega"
    assert set(weighting["omega"].values()) == {1.0}
    assert state.get("synthetic_gate_override_used") is not True
    events = [event["event"] for event in state[WAIVER_EVENTS_KEY]]
    assert "physiology_support_waived" in events
    for record in state["metrics"]:
        for value in record.get("losses", {}).values():
            if isinstance(value, float):
                assert math.isfinite(value)


def test_c1_low_ess_normalizes_relative_weights_without_stop(tmp_path):
    case = _joint_stage_case(tmp_path, "c1", n_unlabeled=5)
    case["config"]["training"]["stage_updates"]["J"] = 1
    case["config"]["losses"]["w_mode"] = "physiology"
    patients = sorted(case["data"].unlabeled_patients)
    rows = {p: {"w_pat": 1.0 if index == 0 else 0.0} for index, p in enumerate(patients)}

    strict = _waived_state(case, reason=None)
    strict["physiology_rows"] = copy.deepcopy(rows)
    with pytest.raises(RuntimeError):
        _execute_joint_stage(case, tmp_path / "strict", copy.deepcopy(_modules(case)), state=strict)
    assert strict["physiology_support_gate"]["status"] == "NO_GO_U_ESS_GATE"

    state = _waived_state(case)
    state["physiology_rows"] = copy.deepcopy(rows)
    state = _execute_joint_stage(case, tmp_path / "waived", _modules(case), state=state)
    gate = state["physiology_support_gate"]
    assert gate["status"] == "NO_GO_U_ESS_GATE" and gate["passed"] is False
    weighting = state["physiology_cohort_weighting"]
    assert weighting["override_policy"] == "exploratory_low_ess_relative_normalized"
    assert weighting["omega"][patients[0]] == 5.0
    assert all(weighting["omega"][p] == 0.0 for p in patients[1:])
    assert state["j_optimizer_updates"] == 1


def test_global_zero_kd_support_stops_strict_and_logs_zero_distillation_when_waived(
    tmp_path, monkeypatch,
):
    def no_confidence(config, confidence, patient_id, run_state, *, labeled):
        return torch.zeros_like(confidence)

    monkeypatch.setattr("self_audit.candidate_q3d.engine._cell_weights", no_confidence)
    case = _joint_stage_case(tmp_path, "c1")
    case["config"]["training"]["stage_updates"]["J"] = 1
    case["config"]["losses"]["continuous_feature_kd"] = 1.0
    with pytest.raises(RuntimeError, match="global KD collapse"):
        _execute_joint_stage(
            case, tmp_path / "strict", copy.deepcopy(_modules(case)),
            state=_waived_state(case, reason=None),
        )
    state = _execute_joint_stage(case, tmp_path / "waived", _modules(case), state=_waived_state(case))
    assert state["j_kd_supported_branches"] == 0
    assert state["exploratory_zero_distillation"] is True
    assert "J" in state["completed_stages"]
    assert any(event["event"] == "kd_support_waived_zero_distillation"
               for event in state[WAIVER_EVENTS_KEY])
    assert any(record.get("event") == "kd_support_waived_zero_distillation"
               and record.get("exploratory") is True for record in state["metrics"])


def test_exploratory_freeze_rejects_pass_or_uninitialized_and_keeps_measured_gate():
    from self_audit.candidate_q3d.engine import _freeze_tokenizer_for_exploratory_waiver

    class _Tokenizer(torch.nn.Module):
        def __init__(self, initialized):
            super().__init__()
            self.codebook_initialized = initialized
            self.register_buffer("is_frozen", torch.zeros((), dtype=torch.bool))
            self.reconstructor = torch.nn.Linear(1, 1)
            self.anatomy_head = torch.nn.Linear(1, 1)

        def artifact_metadata(self):
            return {"sha256": "x", "frozen": bool(self.is_frozen), "initialized": self.codebook_initialized}

    policy = quality_waiver(REASON)
    failed = {"status": "NO_GO", "passed": False, "perplexity": 3.0}
    with pytest.raises(ValueError):
        _freeze_tokenizer_for_exploratory_waiver(_Tokenizer(True), {"status": "PASS", "passed": True}, policy)
    with pytest.raises(RuntimeError):
        _freeze_tokenizer_for_exploratory_waiver(_Tokenizer(False), failed, policy)
    tokenizer = _Tokenizer(True)
    result = _freeze_tokenizer_for_exploratory_waiver(tokenizer, failed, policy)
    assert result["gate"] == failed and result["gate"]["passed"] is False
    assert result["exploratory"] is True and result[WAIVER_STATE_KEY] == policy
    assert result["frozen"] is True and "synthetic_gate_override" not in result
    assert tokenizer.reconstructor is None and tokenizer.anatomy_head is None


# -------------------------------------------------------------- run_training


def _c0_synthetic(tmp_path):
    from tests.test_q3d_engine import _ResumeProbeStudent
    from self_audit.candidate_a3d.data import CandidateA3DData, synthetic_fixture
    from self_audit.candidate_q3d.config import canonical_config
    from self_audit.candidate_q3d.data import Q3DData

    fixture = synthetic_fixture(
        tmp_path / "fixture", seed=19, n_labeled=2, n_unlabeled=2, n_evaluation=1,
        shape_tzyx=(2, 4, 16, 17),
    )
    data = Q3DData(CandidateA3DData(fixture["train"], fixture["train_sha256"], synthetic=True), protocol="A")
    config = canonical_config("c0", dataset="acdc", protocol="A", k=2, seed=42)
    config["runtime"].update({"synthetic": True, "device": "cpu"})
    config["runtime"]["smoke_update_overrides"] = {"T": None, "Q": None, "P": None, "J": 2}
    config["dataset"]["manifests"]["train"] = {
        "path": str(fixture["train"].resolve()), "sha256": fixture["train_sha256"], "status": "READY",
    }
    config["dataset"]["manifests"]["development"] = {
        "path": str(fixture["evaluation"].resolve()),
        "sha256": fixture["evaluation_sha256"], "status": "READY",
    }

    def evaluate(_model, _manifest, _kind, _device):
        return {"score": 0.5, "patient_count": 1}

    return config, data, evaluate, _ResumeProbeStudent


def test_resume_of_waived_checkpoint_requires_the_same_explicit_reason(tmp_path):
    from self_audit.candidate_q3d.engine import train

    config, data, evaluate, student_cls = _c0_synthetic(tmp_path)
    output = tmp_path / "run"
    torch.manual_seed(3)
    first = train(
        config, output, device="cpu", data=data, student=student_cls(), smoke=True,
        update_limit=1, dev_evaluator=evaluate, exploratory_quality_waiver=REASON,
    )
    assert first["status"] == "PAUSED"
    assert first["exploratory"] is True
    assert first["evidence_status"] == "SOFTWARE_ONLY_SYNTHETIC"
    assert first["exploratory_quality_waiver"]["reason"] == REASON
    with pytest.raises(RuntimeError, match="requires the same explicit"):
        train(config, output, device="cpu", data=data, student=student_cls(), smoke=True,
              update_limit=1, dev_evaluator=evaluate)
    with pytest.raises(RuntimeError, match="differs"):
        train(config, output, device="cpu", data=data, student=student_cls(), smoke=True,
              update_limit=1, dev_evaluator=evaluate, exploratory_quality_waiver="another reason")
    second = train(
        config, output, device="cpu", data=data, student=student_cls(), smoke=True,
        update_limit=1, dev_evaluator=evaluate, exploratory_quality_waiver=REASON,
    )
    assert second["status"] == "COMPLETED" and second["exploratory"] is True
    assert [e["event"] for e in second["exploratory_quality_waiver_events"]] == [
        "exploratory_quality_waiver_bound",
    ]


def test_default_run_records_no_waiver(tmp_path):
    from self_audit.candidate_q3d.engine import train

    config, data, evaluate, student_cls = _c0_synthetic(tmp_path)
    result = train(config, tmp_path / "strict", device="cpu", data=data, student=student_cls(),
                   smoke=True, dev_evaluator=evaluate)
    assert result["exploratory"] is False
    assert result["exploratory_quality_waiver"] is None
    assert result["exploratory_quality_waiver_events"] == []
    state = torch.load(result["checkpoint"], map_location="cpu", weights_only=False)
    assert WAIVER_STATE_KEY not in state
    with pytest.raises(ValueError):
        train(config, tmp_path / "bad", device="cpu", data=data, student=student_cls(),
              smoke=True, dev_evaluator=evaluate, exploratory_quality_waiver="  ")


# ------------------------------------------------- full synthetic T/Q/P/J


def _full_v1_config(tmp_path):
    from self_audit.candidate_q3d.config import validate_config
    from tests.test_q3d_stage_artifacts import _synthetic_config_and_fixture

    config, _fixture = _synthetic_config_and_fixture(tmp_path)
    config = copy.deepcopy(config)
    config["runtime"]["synthetic_gate_override"] = False
    return validate_config(config)


def _force_w_pat(monkeypatch, value):
    import self_audit.candidate_q3d.physiology as physiology

    original = physiology.physiology_weight_for_patient

    def patched(*args, **kwargs):
        row = original(*args, **kwargs)
        row["w_pat"] = value
        return row

    monkeypatch.setattr(physiology, "physiology_weight_for_patient", patched)


def test_full_pipeline_failed_q_allzero_p_strict_stops_and_waiver_reaches_j(tmp_path, monkeypatch):
    from self_audit.candidate_q3d import engine
    from train import _manufactured_teacher

    _force_w_pat(monkeypatch, 0.0)
    config = _full_v1_config(tmp_path)
    strict_output = tmp_path / "strict"
    with pytest.raises(RuntimeError, match="tokenizer gate failed"):
        engine.run_training(config, strict_output, device="cpu", smoke=True,
                            teacher=_manufactured_teacher(config))

    result = engine.run_training(
        config, tmp_path / "waived", device="cpu", smoke=True,
        teacher=_manufactured_teacher(config), exploratory_quality_waiver=REASON,
    )
    assert result["status"] == "COMPLETED"
    assert result["completed_stages"][-3:] == ["Q", "P", "J"]
    assert result["exploratory"] is True
    assert result["tokenizer_gate"]["passed"] is False
    assert result["tokenizer_gate"]["status"] != "PASS"
    assert result["synthetic_gate_override_used"] is False
    assert result["exploratory_tokenizer_waiver_used"] is True
    assert result["physiology_support_gate"]["passed"] is False
    assert result["physiology_cohort_weighting"]["override_policy"] == (
        "exploratory_zero_mean_uniform_unit_omega"
    )
    events = {event["event"] for event in result["exploratory_quality_waiver_events"]}
    assert {"tokenizer_gate_waived", "physiology_support_waived"} <= events
    assert set(result["stage_artifacts"]) >= {"T", "Q", "P"}

    # The key operational path: a strict run stopped at the Q gate resumes
    # from its own checkpoint with an explicit waiver and no extra Q updates.
    stopped = torch.load(strict_output / "latest.pt", map_location="cpu", weights_only=False)
    q_cursor = int(stopped["stage_cursor"])
    resumed = engine.run_training(
        config, strict_output, device="cpu", smoke=True,
        teacher=_manufactured_teacher(config), exploratory_quality_waiver=REASON,
    )
    assert resumed["status"] == "COMPLETED"
    assert resumed["tokenizer_gate"]["passed"] is False
    q_updates = [r for r in resumed.get("exploratory_quality_waiver_events", [])
                 if r["event"] == "tokenizer_gate_waived"]
    assert len(q_updates) == 1
    final = torch.load(resumed["checkpoint"], map_location="cpu", weights_only=False)
    assert final["q_optimizer_updates"] == stopped["q_optimizer_updates"]
    assert q_cursor == stopped["stage_cursor"]
