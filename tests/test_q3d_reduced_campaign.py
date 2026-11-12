from __future__ import annotations

from pathlib import Path

import pytest

from scripts.q3d_campaign import (
    DEFAULT_REDUCED_RUNS,
    create_campaign_plan,
    main,
    validate_campaign_plan,
)


def _reduced_plan(tmp_path: Path) -> tuple[Path, dict]:
    path = create_campaign_plan(
        out=tmp_path / "reduced",
        datasets=("acdc",),
        protocols=("A",),
        runs=DEFAULT_REDUCED_RUNS,
        synthetic=True,
        smoke_updates=1,
    )
    return path, validate_campaign_plan(path)


def test_reduced_catalogue_has_exact_shared_producers_and_six_joints(tmp_path: Path) -> None:
    _, plan = _reduced_plan(tmp_path)
    assert plan["selection"]["runs"] == [
        ["v1", 8], ["v1", 5], ["v1", 10], ["v1u", 8], ["c1", 8], ["c0", 8],
    ]
    assert plan["selection"]["k_order"] == [8, 5, 10]

    primary = [item for item in plan["experiments"] if item["group"] == "primary"]
    assert [(item["setting"], item["k"]) for item in primary] == [
        ("v1", 8), ("v1", 5), ("v1", 10), ("v1u", 8), ("c1", 8), ("c0", 8),
    ]
    dag = plan["shared_stage_dag"]
    assert dag["stage_job_counts"] == {"T": 3, "Q": 3, "P": 3, "J": 6}
    assert dag["shared_producer_counts"] == {"T": 3, "Q": 3, "P": 3}
    assert len(plan["stage_jobs"]) == 15

    by_key = {(item["setting"], item["k"]): item for item in primary}
    k8_refs = by_key["v1", 8]["stage_artifact_producers"]
    for setting in ("v1u", "c1"):
        refs = by_key[setting, 8]["stage_artifact_producers"]
        assert {stage: refs[stage] for stage in ("T", "Q", "P")} == {
            stage: k8_refs[stage] for stage in ("T", "Q", "P")
        }
    assert set(by_key["c0", 8]["stage_artifact_producers"]) == {"J"}
    assert by_key["v1", 5]["stage_artifact_producers"]["T"] != k8_refs["T"]
    assert by_key["v1", 10]["stage_artifact_producers"]["T"] != k8_refs["T"]

    assert plan["selection"]["baseline_settings"] == ["teacher_supervised"]
    baseline = next(item for item in plan["experiments"] if item["setting"] == "teacher_supervised")
    assert baseline["stage_artifact_producers"]["T"] == k8_refs["T"]
    assert not any(job.get("setting") == "teacher_supervised" for job in plan["stage_jobs"])
    assert len(plan["evaluation_jobs"]) == 1
    evaluation = plan["evaluation_jobs"][0]
    assert evaluation["source_stage_job_id"] == k8_refs["T"]
    assert evaluation["student_joint_updates"] == 0
    assert evaluation["independent_baseline"] is False
    assert evaluation["status"] == "NOT_RUN"

    assert not any(item["setting"] == "a3d_teacher_free" for item in plan["experiments"])
    assert not any(job["setting"] == "a3d_teacher_free" for job in plan["stage_jobs"])
    assert len(plan["external_references"]) == 1
    assert plan["external_references"][0]["status"] == "NOT_RUN"


@pytest.mark.parametrize(
    "runs, message",
    [
        (("v1@8", "v1@8"), "duplicate run pair"),
        (("v1@7",), "unsupported run K"),
        (("does_not_exist@8",), "unsupported run setting"),
        (("v1",), "SETTING@K syntax"),
    ],
)
def test_explicit_run_catalogue_rejects_ambiguous_pairs(tmp_path: Path, runs, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        create_campaign_plan(
            out=tmp_path / "invalid",
            datasets=("acdc",), protocols=("A",), runs=runs,
            synthetic=True, smoke_updates=1,
        )


def test_explicit_runs_conflict_with_legacy_selection_flags(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="cannot be combined"):
        create_campaign_plan(
            out=tmp_path / "conflict",
            datasets=("acdc",), protocols=("A",), runs=("v1@8",),
            include_secondary=False, synthetic=True, smoke_updates=1,
        )


def test_explicit_k8_rejects_a_nonlocked_train_manifest(tmp_path: Path) -> None:
    bad_manifest = tmp_path / "wrong-k8-train.json"
    bad_manifest.write_text('{"schema": "self_audit.data.v2"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="locked campaign input"):
        create_campaign_plan(
            out=tmp_path / "wrong-k8",
            datasets=("acdc",), protocols=("A",), runs=("v1@8",),
            manifest_inputs={"acdc": {"k8:train": str(bad_manifest)}},
        )


def test_incompatible_manifest_schema_is_pinned_but_not_ready(tmp_path: Path) -> None:
    legacy_manifest = tmp_path / "legacy-train.json"
    legacy_manifest.write_text(
        '{"schema": "self_audit.manifest.v1", "role": "training", "label_budget": 5}\n',
        encoding="utf-8",
    )
    path = create_campaign_plan(
        out=tmp_path / "legacy-schema",
        datasets=("acdc",), protocols=("A",), runs=("v1@5",),
        manifest_inputs={"acdc": {"k5:train": str(legacy_manifest)}},
    )
    plan = validate_campaign_plan(path)
    assert plan["lineage_audit"]["schema_loader_status"] == "NOT_RUN"
    contract = plan["lineage_audit"]["manifest_contracts"][0]
    assert contract["schema"] == "self_audit.manifest.v1"
    assert contract["status"] == "NOT_RUN"
    t_job = next(job for job in plan["stage_jobs"] if job["stage"] == "T")
    assert t_job["readiness"] == "NOT_RUN"
    assert "schema/role" in t_job["reason"]


def test_explicit_runs_without_k8_do_not_advertise_teacher_baseline(tmp_path: Path) -> None:
    path = create_campaign_plan(
        out=tmp_path / "no-k8",
        datasets=("acdc",), protocols=("A",), runs=("v1@5",),
        synthetic=True, smoke_updates=1,
    )
    plan = validate_campaign_plan(path)
    assert plan["selection"]["baseline_settings"] == []
    assert plan["evaluation_jobs"] == []


def test_nested_lineage_checks_manifest_labeled_ids(tmp_path: Path) -> None:
    manifest = tmp_path / "mismatched-lineage.json"
    manifest.write_text(
        '{"schema":"self_audit.data.v2","role":"training","dataset":"acdc",'
        '"label_budget":5,"labeled":[{"id":"patient001"},{"id":"patient003"},'
        '{"id":"patient004"},{"id":"patient005"},{"id":"patient006"}],'
        '"fewshot":{"k":5,"seed":42,"selected_ids":["patient002","patient003",'
        '"patient004","patient005","patient006"],'
        '"source_sha256":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"}}\n',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="labeled IDs"):
        create_campaign_plan(
            out=tmp_path / "bad-lineage",
            datasets=("acdc",), protocols=("A",), runs=("v1@5",),
            manifest_inputs={"acdc": {"k5:train": str(manifest)}},
            synthetic=True, smoke_updates=1,
        )


def test_reduced_development_must_reuse_locked_bytes(tmp_path: Path) -> None:
    development = tmp_path / "development.json"
    development.write_text('{"schema":"self_audit.data.v2","role":"development"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="locked K=8 manifest"):
        create_campaign_plan(
            out=tmp_path / "bad-development",
            datasets=("acdc",), protocols=("A",), runs=("v1@5", "v1@8", "v1@10"),
            manifest_inputs={"acdc": {"development": str(development)}},
            synthetic=True, smoke_updates=1,
        )


def test_bare_cli_uses_the_six_run_catalogue(tmp_path: Path, capsys) -> None:
    main(["plan", "--out", str(tmp_path / "cli-default")])
    output = capsys.readouterr().out
    assert '"status": "NOT_RUN"' in output
    plan = validate_campaign_plan(tmp_path / "cli-default" / "PLAN.json")
    assert plan["selection"]["runs"] == [
        ["v1", 8], ["v1", 5], ["v1", 10], ["v1u", 8], ["c1", 8], ["c0", 8],
    ]
    assert plan["selection"]["datasets"] == ["acdc"]
    assert plan["selection"]["protocols"] == ["A"]
