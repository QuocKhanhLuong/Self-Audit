"""Few-shot campaign planning contracts; no patient training is run here."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.candidate_a_fewshot import create_fewshot_plan, run_fewshot_plan
from self_audit.candidate_a3d.data import synthetic_fixture


def _fixtures(tmp_path: Path, dataset: str) -> tuple[Path, Path]:
    fixture = synthetic_fixture(tmp_path / dataset, n_labeled=10, n_unlabeled=2,
                                n_evaluation=1, shape_tzyx=(1, 2, 4, 5))
    train = Path(fixture["train"])
    data = json.loads(train.read_text())
    data["dataset"] = dataset
    train.write_text(json.dumps(data, indent=2) + "\n")
    return train, Path(fixture["evaluation"])


@pytest.mark.parametrize("dataset", ["acdc", "mnms"])
def test_default_plan_has_21_arms_and_92_jobs(tmp_path, dataset):
    train, development = _fixtures(tmp_path, dataset)
    plan_path = create_fewshot_plan(dataset=dataset, train_manifest=train,
        development_manifest=development, model_config=None, out=tmp_path / "campaign",
        synthetic=True)
    plan = json.loads(plan_path.read_text())
    assert plan["shots"] == [2, 5, 8, 10]
    assert plan["seed"] == 42
    assert len(plan["arms"]) == 21
    assert plan["total_jobs"] == 92
    assert plan["config"]["path"].endswith("configs/fewshot/candidate_a3d.json")
    assert len(plan["config"]["sha256"]) == 64
    assert plan["checkpoint_every_epochs"] == 0
    for item in plan["child_plans"]:
        child = json.loads(Path(item["plan"]).read_text())
        assert len(child["jobs"]) == 23
        assert child["training_seeds"] == [42]
        assert child["ablation_arms"] == plan["arms"]
    assert run_fewshot_plan(plan_path)["status"] == "PLAN_VALIDATED_NOT_RUN"


def test_nested_manifests_share_dev_and_use_nested_ids(tmp_path):
    train, development = _fixtures(tmp_path, "acdc")
    plan_path = create_fewshot_plan(dataset="acdc", train_manifest=train,
        development_manifest=development, model_config=None, out=tmp_path / "campaign",
        synthetic=True)
    plan = json.loads(plan_path.read_text())
    selected = []
    development_ids = {record["id"] for record in json.loads(development.read_text())["patients"]}
    for item in plan["manifests"]:
        manifest = json.loads(Path(item["path"]).read_text())
        selected.append(set(manifest["fewshot"]["selected_ids"]))
        assert manifest["label_seed"] == 42
        assert manifest["label_budget"] == item["k"]
        assert development_ids.issubset(set(manifest["evaluation_ids"]))
        assert all("masks" in record for record in manifest["labeled"])
        assert all("masks" not in record for record in manifest["unlabeled"])
    assert selected[0] < selected[1] < selected[2] < selected[3]


def test_non_null_external_weights_rejected_before_campaign_creation(tmp_path):
    train, development = _fixtures(tmp_path, "acdc")
    model = tmp_path / "model.json"
    model.write_text(json.dumps({"backbone": "mednext_l_k3", "source_repo": "/src",
                                 "source_commit": "abc", "weights": "/weights.pt",
                                 "weights_sha256": "deadbeef"}))
    with pytest.raises(ValueError, match="cannot be reused"):
        create_fewshot_plan(dataset="acdc", train_manifest=train,
            development_manifest=development, model_config=model,
            out=tmp_path / "campaign", synthetic=True)
    assert not (tmp_path / "campaign").exists()


def test_profiled_runtime_overrides_are_pinned_in_every_child_recipe(tmp_path):
    train, development = _fixtures(tmp_path, "acdc")
    path = create_fewshot_plan(
        dataset="acdc", train_manifest=train, development_manifest=development,
        model_config=None, out=tmp_path / "campaign", synthetic=True,
        checkpoint_every_epochs=2, decoder_chunk_size=7,
    )
    plan = json.loads(path.read_text())
    assert plan["checkpoint_every_epochs"] == 2
    assert plan["decoder_chunk_size"] == 7
    for item in plan["child_plans"]:
        child = json.loads(Path(item["plan"]).read_text())
        assert child["runtime_overrides"] == {
            "checkpoint_every_epochs": 2,
            "decoder_chunk_size": 7,
        }
        for job in child["jobs"]:
            recipe = json.loads(Path(job["config"]).read_text())
            assert recipe["checkpoint_every_epochs"] == 2
            assert recipe["student"]["chunk_size"] == 7
    assert run_fewshot_plan(path)["status"] == "PLAN_VALIDATED_NOT_RUN"


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("checkpoint_every_epochs", -1, "checkpoint_every_epochs"),
        ("decoder_chunk_size", 0, "decoder_chunk_size"),
    ],
)
def test_invalid_profiled_runtime_override_is_rejected_before_output(
    tmp_path, field, value, message
):
    train, development = _fixtures(tmp_path, "acdc")
    kwargs = {field: value}
    with pytest.raises(ValueError, match=message):
        create_fewshot_plan(
            dataset="acdc", train_manifest=train, development_manifest=development,
            model_config=None, out=tmp_path / "campaign", synthetic=True, **kwargs,
        )
    assert not (tmp_path / "campaign").exists()


def test_changed_child_plan_is_rejected(tmp_path):
    train, development = _fixtures(tmp_path, "acdc")
    path = create_fewshot_plan(dataset="acdc", train_manifest=train,
        development_manifest=development, model_config=None, out=tmp_path / "campaign",
        synthetic=True)
    plan = json.loads(path.read_text())
    child_path = Path(plan["child_plans"][0]["plan"])
    child = json.loads(child_path.read_text())
    child["training_seeds"] = [43]
    child_path.write_text(json.dumps(child, indent=2) + "\n")
    with pytest.raises(ValueError, match="child plan changed"):
        run_fewshot_plan(path)
