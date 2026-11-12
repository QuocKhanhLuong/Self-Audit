"""A-3D recipe/planner contracts; all fixtures are manufactured only."""

import json
from pathlib import Path

import pytest

from scripts.candidate_a_ablations import create_plan, main, run_plan
from self_audit.candidate_a3d.data import synthetic_fixture
from self_audit.candidate_a3d.config import validate_config


def test_a3d_recipes_have_native_volume_and_explicit_untrained_reference():
    for path in sorted(Path("configs/candidate_a3d").glob("*.json")):
        config = validate_config(json.loads(path.read_text()))
        assert config["dataset"]["protocol"] == "native_3d"
        assert config["student"]["in_channels"] == 1
        assert config["student"]["local_queries"] is True
        assert config["bank"]["region_grid"] == [2, 2, 2]
        assert config["bank"]["boundary_grid"] == [2, 2, 2]
        assert config["rollout"]["baseline_ids"] == [0, 8, 16]
        if config["model"]["weights"] is None:
            assert config["model"]["backbone"] == "tiny_test3d" or path.name != "smoke.json"


@pytest.mark.parametrize("dataset", ["acdc", "mnms"])
def test_a3d_cli_defaults_to_one_seed_and_four_main_comparisons(tmp_path, dataset):
    fixture = synthetic_fixture(
        tmp_path / "data", n_labeled=2, n_unlabeled=0, n_evaluation=1,
        shape_tzyx=(1, 3, 6, 7),
    )
    out = tmp_path / "plan"
    main(["plan", "--track", "candidate_a3d", "--dataset", dataset,
          "--train", str(fixture["train"]), "--development", str(fixture["evaluation"]),
          "--out", str(out), "--synthetic", "--epochs", "1",
          "--pretrain-epochs", "1", "--warmup-epochs", "1"])
    plan = json.loads((out / "PLAN.json").read_text())
    assert plan["training_seeds"] == [42]
    assert plan["ablation_arms"] == ["full", "teacher_free", "fixed_queries", "global_only_queries"]
    jobs = {job["id"]: job for job in plan["jobs"]}
    assert set(jobs) == {"s42_pretrain", "s42_warmup", "s42_full", "s42_teacher_free",
                         "s42_fixed_queries", "s42_global_only_queries"}
    assert jobs["s42_teacher_free"]["teacher_from"] is None
    assert jobs["s42_warmup"]["teacher_from"] == "s42_pretrain"
    for name in ("full", "fixed_queries", "global_only_queries"):
        assert jobs[f"s42_{name}"]["teacher_from"] == "s42_warmup"
    configs = {name: json.loads(Path(job["config"]).read_text()) for name, job in jobs.items()}
    assert len({config["rollout"]["bit_budget"] for config in configs.values()}) == 1
    assert all(config["seed"] == 42 and config["epochs"] == 1 for config in configs.values())
    assert run_plan(out / "PLAN.json")["jobs"] == 6
    assert not (out / "runs").exists()


def test_a3d_planner_pins_explicit_diagnostic_arms_without_executing(tmp_path):
    data_module = pytest.importorskip("self_audit.candidate_a3d.data")
    fixture = data_module.synthetic_fixture(
        tmp_path / "data", n_labeled=2, n_unlabeled=0, n_evaluation=1,
        shape_tzyx=(1, 3, 6, 7),
    )
    plan_path = create_plan(
        track="candidate_a3d", dataset="acdc", train_manifest=fixture["train"],
        development_manifest=fixture["evaluation"], model_config=None,
        out=tmp_path / "plan", seeds=[7], epochs=1, pretrain_epochs=1,
        warmup_epochs=1, synthetic=True,
        arms=[arm["name"] for arm in json.loads(Path("configs/ablations/candidate_a3d.json").read_text())["arms"]],
    )
    plan = json.loads(plan_path.read_text())
    assert plan["schema"] == "self_audit.candidate_a3d.plan.v1"
    assert plan["track"] == "candidate_a3d"
    assert len(plan["jobs"]) == 23  # 21 explicit arms plus two shared teacher stages.
    assert run_plan(plan_path)["status"] == "PLAN_VALIDATED_NOT_RUN"
    assert not (plan_path.parent / "runs").exists()
    recipes = {job["id"]: json.loads(Path(job["config"]).read_text()) for job in plan["jobs"]}
    assert recipes["s7_budget_2048bit"]["rollout"]["bit_budget"] == 2048
    assert recipes["s7_budget_8192bit"]["rollout"]["bit_budget"] == 8192
    assert recipes["s7_global_only_queries"]["student"]["local_queries"] is False
    assert recipes["s7_teacher_free"]["use_teacher"] is False
    assert recipes["s7_teacher_free"]["teacher_init"]["kind"] == "none"


def test_real_data_teacher_free_plan_does_not_require_backbone_config(tmp_path):
    fixture = synthetic_fixture(
        tmp_path / "data", n_labeled=2, n_unlabeled=0, n_evaluation=1,
        shape_tzyx=(1, 3, 6, 7),
    )
    for manifest_path, role in ((fixture["train"], "training"), (fixture["evaluation"], "development")):
        manifest = json.loads(manifest_path.read_text())
        manifest["synthetic"] = False
        manifest["dataset"] = "acdc"
        manifest["role"] = role
        manifest_path.write_text(json.dumps(manifest))
    path = create_plan(
        track="candidate_a3d", dataset="acdc", train_manifest=fixture["train"],
        development_manifest=fixture["evaluation"], model_config=None,
        out=tmp_path / "plan", seeds=[3], arms=["teacher_free"], epochs=1,
        pretrain_epochs=1, warmup_epochs=1, synthetic=False,
    )
    plan = json.loads(path.read_text())
    assert len(plan["jobs"]) == 1
    config = json.loads(Path(plan["jobs"][0]["config"]).read_text())
    assert config["use_teacher"] is False
    assert config["model"]["backbone"] == "mednext_l_k3"
