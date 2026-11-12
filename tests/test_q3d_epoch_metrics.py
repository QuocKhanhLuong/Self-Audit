"""SOFTWARE_ONLY: epoch observations do not alter selection or training state."""
import copy
import random

import numpy as np
import pytest
import torch

from self_audit.candidate_q3d import engine
from tests.test_q3d_engine import _execute_joint_stage, _joint_stage_case

DEV = {"score": 0.5, "per_class": {"RV": 0.4, "MYO": 0.5, "LV": 0.6}, "patient_count": 1}


@pytest.mark.parametrize("fails", [False, True])
def test_extra_evaluation_restores_rng_and_every_submodule_mode(fails):
    model = torch.nn.Sequential(torch.nn.Dropout(), torch.nn.Linear(2, 2))
    model.train()
    model[1].eval()
    before = engine._rng_state()
    modes = [m.training for m in model.modules()]

    def evaluate(*args):
        random.random()
        np.random.rand()
        torch.rand(2)
        model.eval()
        if fails:
            raise RuntimeError("probe")
        return DEV

    kwargs = dict(data=None, config=None, device=None, output_kind="teacher", evaluator=evaluate)
    if fails:
        with pytest.raises(RuntimeError, match="probe"):
            engine._observational_development(model, {}, **kwargs)
    else:
        assert engine._observational_development(model, {}, **kwargs) == DEV
    after = engine._rng_state()
    assert engine._same_state_value(before, after)
    assert [m.training for m in model.modules()] == modes


def _teacher_run(case, output, modules):
    spatial_rng, intensity_generator = engine._seed_all(42)
    state = engine._initial_run_state(engine._object_sha256(case["config"]), {})

    def evaluate(*args):
        # An attractive intermediate score cannot select a checkpoint.
        torch.rand(1)
        return {**DEV, "score": 0.99 if state["stage_cursor"] == 4 else 0.5}

    engine._run_teacher_stage(
        case["config"], data=case["data"], teacher=modules["teacher"], student=modules["student"],
        tokenizer=None, ema_model=modules["ema"], state=state, output_dir=output,
        device=torch.device("cpu"), spatial_rng=spatial_rng,
        intensity_generator=intensity_generator, stage_rng=random.Random(89), sampler=None,
        smoke=True, update_limit_active=False, dev_evaluator=evaluate, update_budget=8,
    )
    return state


def test_teacher_epoch_cadence_preserves_selection_and_training_rng(tmp_path, monkeypatch):
    case = _joint_stage_case(tmp_path, "c1")
    case["config"]["training"]["checkpoint_evaluation_interval"] = 6
    modules = {key: copy.deepcopy(case[key]) for key in ("teacher", "student", "ema")}
    measured = _teacher_run(case, tmp_path / "measured", copy.deepcopy(modules))
    events = [r for r in measured["metrics"] if r.get("event") == "development_epoch"]
    assert [(r["epoch"], r["update"]) for r in events] == [(1, 4), (2, 8)]
    assert all(r["per_class"] == DEV["per_class"] for r in events)
    assert measured["best_teacher_score"] == 0.5
    assert [r["update"] for r in measured["teacher_dev_history"]] == [6, 8]
    monkeypatch.setattr(engine, "_observational_development", lambda *a, **kw: {**DEV, "score": 0.99})
    control = _teacher_run(case, tmp_path / "control", copy.deepcopy(modules))
    for key in ("teacher_state", "optimizer_states", "stage_rng_state", "intensity_rng_state", "rng"):
        assert engine._same_state_value(measured[key], control[key]), key


def test_joint_epoch_is_exact_full_u_cycle_and_survives_resume(tmp_path):
    case = _joint_stage_case(tmp_path, "v1", n_unlabeled=3)
    case["config"]["training"]["stage_updates"]["J"] = 6
    case["config"]["training"]["checkpoint_evaluation_interval"] = 400
    modules = {key: copy.deepcopy(case[key]) for key in ("teacher", "student", "tokenizer", "ema")}
    evaluate = lambda *args: copy.deepcopy(DEV)
    whole = _execute_joint_stage(case, tmp_path / "whole", copy.deepcopy(modules), dev_evaluator=evaluate)
    first = _execute_joint_stage(case, tmp_path / "resume", copy.deepcopy(modules),
                                 update_limit=2, dev_evaluator=evaluate)
    assert not any(r.get("event") == "development_epoch" for r in first["metrics"])
    loaded = engine._load_checkpoint(tmp_path / "resume/latest.pt",
        config_sha256=first["config_sha256"], data_identity=first["data_identity"])
    resumed = _execute_joint_stage(case, tmp_path / "resume", copy.deepcopy(modules),
                                   state=loaded, update_limit=4, dev_evaluator=evaluate)
    events = lambda state: [r for r in state["metrics"] if r.get("event") == "development_epoch"]
    assert events(resumed) == events(whole)
    assert [(r["epoch"], r["update"]) for r in events(whole)] == [(1, 3), (2, 6)]
    assert all(r["per_class"] == DEV["per_class"] for r in events(whole))
    selections = [r for r in whole["metrics"] if r.get("event") == "development_selection"]
    assert [r["update"] for r in selections] == [6]
    assert engine._same_state_value(whole["student_state"], resumed["student_state"])
