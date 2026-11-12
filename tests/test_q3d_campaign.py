from __future__ import annotations

import json
from pathlib import Path

from scripts.q3d_campaign import create_campaign_plan, validate_campaign_plan


def _shared_plan(tmp_path: Path) -> tuple[Path, dict]:
    path = create_campaign_plan(
        out=tmp_path / "shared-plan",
        datasets=("acdc",),
        protocols=("A",),
        ks=(5,),
        include_secondary=True,
        include_baselines=True,
    )
    return path, validate_campaign_plan(path)


def test_shared_campaign_builds_one_tqp_source_per_feature_variant(tmp_path: Path) -> None:
    _, plan = _shared_plan(tmp_path)
    dag = plan["shared_stage_dag"]
    assert len(plan["experiments"]) == 15
    assert dag["stage_job_counts"] == {"T": 1, "Q": 5, "P": 5, "J": 14}
    assert dag["shared_producer_counts"] == {"T": 1, "Q": 5, "P": 5}
    assert len(plan["stage_jobs"]) == 25

    jobs = {job["id"]: job for job in plan["stage_jobs"]}
    assert sum(job["stage"] == "T" for job in plan["stage_jobs"]) == 1
    assert all(job["status"] == "NOT_RUN" for job in plan["stage_jobs"])
    for job in plan["stage_jobs"]:
        if job["kind"] == "shared_stage_producer":
            output = job["artifact_output"]
            assert Path(output["manifest_path"]).name == f"{job['stage']}.artifact.json"
            assert output["manifest_sha256"] is None
        if job["stage"] == "J":
            assert all(ref["sha256"] is None for ref in job["stage_artifact_inputs"].values())
            assert all(ref["status"] == "NOT_RUN" for ref in job["stage_artifact_inputs"].values())
            assert set(job["dependencies"]) == {
                ref["producer_job_id"] for ref in job["stage_artifact_inputs"].values()
            }
    assert all(reference["producer_job_id"] in jobs
               for reference in plan["experiments"][0]["expected_stage_artifacts"].values())


def test_v1_v1u_share_exact_refs_and_source_ablations_split_qp_only(tmp_path: Path) -> None:
    _, plan = _shared_plan(tmp_path)
    experiments = {
        item["setting"]: item for item in plan["experiments"]
    }
    v1 = experiments["v1"]["stage_artifact_producers"]
    v1u = experiments["v1u"]["stage_artifact_producers"]
    assert {stage: v1[stage] for stage in ("T", "Q", "P")} == {
        stage: v1u[stage] for stage in ("T", "Q", "P")
    }
    for setting in ("c1", "e1"):
        refs = experiments[setting]["stage_artifact_producers"]
        assert {stage: refs[stage] for stage in ("T", "Q", "P")} == {
            stage: v1[stage] for stage in ("T", "Q", "P")
        }
    assert experiments["c1"]["expected_stage_artifacts"]["Q"]["producer_job_id"] == v1["Q"]

    for setting in ("o1", "s1", "v1_stride4", "v1_decoder_stride8"):
        refs = experiments[setting]["stage_artifact_producers"]
        assert refs["T"] == v1["T"]
        assert refs["Q"] != v1["Q"]
        assert refs["P"] != v1["P"]

    residual2 = experiments["v1_residual2"]["stage_artifact_producers"]
    assert {stage: residual2[stage] for stage in ("T", "Q", "P")} == {
        stage: v1[stage] for stage in ("T", "Q", "P")
    }
    assert len(plan["shared_stage_dag"]["expected_v1_v1u_shared_references"]) == 1
    assert plan["shared_stage_dag"]["expected_v1_v1u_shared_reference_count"] == 1
    assert plan["shared_stage_dag"]["shared_consumer_references"][0]["consumer_settings"] == [
        "c1", "v1", "v1u", "e1",
    ]
    assert len(plan["shared_stage_dag"]["source_ablation_references"]) == 4


def test_campaign_plan_default_is_shared_and_reports_full_catalogue_counts(tmp_path: Path) -> None:
    path = create_campaign_plan(out=tmp_path / "default-shared")
    plan = validate_campaign_plan(path)
    dag = plan["shared_stage_dag"]
    assert plan["selection"]["shared_stage_dag"] is True
    assert dag["stage_job_counts"] == {"T": 3, "Q": 15, "P": 15, "J": 42}
    assert dag["expected_v1_v1u_shared_reference_count"] == 3
    assert len(dag["expected_v1_v1u_shared_references"]) == 3


def test_real_unresolved_phase_sources_are_blocked_and_synthetic_proxy_stays_ready(
    tmp_path: Path,
) -> None:
    real_path = create_campaign_plan(
        out=tmp_path / "real-plan", datasets=("acdc",), protocols=("A",), ks=(5,),
        include_baselines=False,
    )
    real = validate_campaign_plan(real_path)
    real_jobs = {job["id"]: job for job in real["stage_jobs"]}
    real_experiments = {item["setting"]: item for item in real["experiments"]}
    for setting in ("v1_stride4", "v1_decoder_stride8"):
        experiment = real_experiments[setting]
        for stage in ("Q", "P", "J"):
            job = real_jobs[experiment["stage_artifact_producers"][stage]]
            assert job["readiness"] == "NOT_RUN"
            assert "BLOCKED: OPEN feature-grid phase contract" in job["reason"]
        teacher_job = real_jobs[experiment["stage_artifact_producers"]["T"]]
        assert "OPEN feature-grid phase contract" not in (teacher_job["reason"] or "")
    for setting in ("v1", "s1"):
        experiment = real_experiments[setting]
        for stage in ("Q", "P", "J"):
            job = real_jobs[experiment["stage_artifact_producers"][stage]]
            assert "OPEN feature-grid phase contract" not in (job["reason"] or "")

    synthetic_path = create_campaign_plan(
        out=tmp_path / "synthetic-plan", datasets=("acdc",), protocols=("A",), ks=(5,),
        include_baselines=False, synthetic=True, smoke_updates=1,
    )
    synthetic = validate_campaign_plan(synthetic_path)
    synthetic_jobs = {job["id"]: job for job in synthetic["stage_jobs"]}
    synthetic_experiments = {item["setting"]: item for item in synthetic["experiments"]}
    for setting in ("v1_stride4", "v1_decoder_stride8"):
        experiment = synthetic_experiments[setting]
        for stage in ("Q", "P", "J"):
            job = synthetic_jobs[experiment["stage_artifact_producers"][stage]]
            assert job["readiness"] == "SMOKE_READY"
            assert "OPEN feature-grid phase contract" not in (job["reason"] or "")


def test_mednext_supervised_baseline_reuses_t_without_retraining(tmp_path: Path) -> None:
    _, plan = _shared_plan(tmp_path)
    baseline = next(item for item in plan["experiments"] if item["setting"] == "mednext_supervised")
    t_producer = next(job for job in plan["stage_jobs"] if job["stage"] == "T")
    assert baseline["stage_artifact_producers"]["T"] == t_producer["id"]
    assert baseline["baseline_reuse"]["independent_teacher_training"] == "NOT_RUN"
    assert baseline["baseline_reuse"]["independent_baseline"] == "NOT_RUN"
    assert baseline["launch_mapping"]["status"] == "REUSE_SHARED_T_NOT_RUN"
    assert len([job for job in plan["stage_jobs"] if job["stage"] == "T"]) == 1


def test_shared_real_plan_keeps_producer_manifests_unresolved(tmp_path: Path) -> None:
    path, plan = _shared_plan(tmp_path)
    raw = json.loads(path.read_text(encoding="utf-8"))
    assert raw["status"] == "PLAN_ONLY_NOT_RUN"
    assert raw["evidence"]["patient_training"] == "NOT_RUN"
    for job in plan["stage_jobs"]:
        if job.get("artifact_output"):
            assert job["artifact_output"]["manifest_sha256"] is None
            assert job["artifact_output"]["status"] == "NOT_RUN"
