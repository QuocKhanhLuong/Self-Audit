"""Concrete, hash-pinned ablations rather than hand-copied drifting recipes."""
import json
from pathlib import Path
import pytest
from scripts.candidate_a_ablations import create_plan, run_plan
from self_audit.candidate_a.data import synthetic_candidate_a_fixture


@pytest.mark.parametrize("dataset", ["acdc", "mnms"])
def test_all_component_configs_are_resolved_and_pinned(tmp_path, dataset):
    data = synthetic_candidate_a_fixture(tmp_path / "data", n_labeled=2,
        n_unlabeled=0, n_evaluation=1, shape_tzyx=(1, 1, 8, 9))
    path = create_plan(dataset=dataset, train_manifest=data["train"],
        development_manifest=data["evaluation"], model_config=None, out=tmp_path / "plan",
        seeds=[7], epochs=1, pretrain_epochs=1, warmup_epochs=1, synthetic=True)
    plan = json.loads(path.read_text())
    assert len(plan["jobs"]) == 21  # 19 arms plus two shared teacher stages.
    assert run_plan(path)["status"] == "PLAN_VALIDATED_NOT_RUN"
    assert not (path.parent / "runs").exists()
    configs = {j["id"].removeprefix("s7_"): json.loads(Path(j["config"]).read_text()) for j in plan["jobs"]}
    assert all(c["dataset"]["name"] == dataset and c["epochs"] == 1 for c in configs.values())
    assert configs["teacher_free"]["use_teacher"] is False
    assert configs["fixed_queries"]["rollout"]["policy_enabled"] is False
    assert configs["teacher_region_proposals"]["rollout"]["teacher_proposal_types"] == ["REGION"]
    assert configs["boundary_only"]["bank"]["types"] == ["BOUNDARY"]
    assert configs["no_measurement_supervision"]["loss"]["region"] == 0
    assert configs["no_measurement_supervision"]["loss"]["boundary"] == 0
    assert configs["decoder_half_width"]["student"]["decoder_dim"] * 2 == configs["full"]["student"]["decoder_dim"]
    corrupted = Path(plan["jobs"][-1]["config"])
    value = json.loads(corrupted.read_text())
    value["seed"] = 8
    corrupted.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="recipe changed"):
        run_plan(path)


def test_teacher_free_plan_never_schedules_teacher(tmp_path):
    data = synthetic_candidate_a_fixture(tmp_path / "data", n_labeled=2,
        n_unlabeled=0, n_evaluation=1, shape_tzyx=(1, 1, 8, 9))
    path = create_plan(dataset="acdc", train_manifest=data["train"], development_manifest=None,
        model_config=None, out=tmp_path / "plan", seeds=[0], arms=["teacher_free"],
        epochs=1, synthetic=True)
    plan = json.loads(path.read_text())
    assert len(plan["jobs"]) == 1 and plan["jobs"][0]["teacher_from"] is None
    config = json.loads(Path(plan["jobs"][0]["config"]).read_text())
    assert config["validation_every"] == 0
    result = run_plan(path, execute=True, device="cpu")
    report = json.loads((Path(result["runs"]["s0_teacher_free"]) / "REPORT.json").read_text())
    assert report["teacher_updates"] == 0 and report["student_updates"] == 2
    assert report["epochs"] == 1
