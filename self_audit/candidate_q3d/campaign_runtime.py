"""Execute an explicitly authorized shared-stage DAG with durable SHA pins.

Planning and validation never call this module. Tests use a manufactured runner;
this delivery does not execute a patient or synthetic campaign.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Mapping

from .artifacts import file_sha256, load_stage_manifest


def _identity_matches(actual: Mapping[str, Any], expected: Mapping[str, Any]) -> bool:
    return all(actual.get(key) == expected.get(key) for key in ("dataset", "protocol", "k", "seed"))


def _verified_output(job: Mapping[str, Any], *, pinned_sha: str | None = None) -> dict[str, Any]:
    path = Path(job["artifact_output"]["manifest_path"]).resolve(strict=True)
    digest = file_sha256(path)
    if pinned_sha is not None and digest != pinned_sha:
        raise ValueError(f"shared artifact changed after production: {job['id']}")
    raw = json.loads(path.read_text(encoding="utf-8"))
    identity = raw.get("identity", {})
    if not _identity_matches(identity, job["artifact_identity"]):
        raise ValueError(f"shared artifact has a different dataset/protocol/K/seed: {job['id']}")
    manifest = load_stage_manifest(path, identity=identity)
    if manifest["stage"] != job["stage"]:
        raise ValueError("shared producer manifest stage mismatch")
    return {"path": str(path), "sha256": digest, "content_hashes": manifest["content_hashes"]}


def execute_shared_campaign(
    plan_path: Path,
    plan: Mapping[str, Any],
    *,
    setting: str,
    device: str,
    runner: Any,
    api: Any,
) -> dict[str, Any]:
    """Run each needed producer once, then launch only each selected J stage.

    ``api`` supplies the script's existing config/fixture/CLI helpers. Completed
    producers can be reused across later setting selections only after their
    stored manifest SHA and every recursive checkpoint dependency revalidate.
    Interrupted/failed jobs require an explicit recovery outside this launcher.
    """
    synthetic = bool(plan["selection"]["synthetic"])
    if not synthetic and device != "cuda":
        raise ValueError("real campaign execution requires explicit CUDA")
    jobs = {job["id"]: job for job in plan["stage_jobs"]}
    experiments = {item["id"]: item for item in plan["experiments"]}
    selected = [item for item in experiments.values() if setting == "all" or item["setting"] == setting]
    if not selected:
        raise ValueError("selection did not match any campaign settings")
    state_path = plan_path.parent / "RUN_STATE.json"
    state = api._read_state(state_path, plan, api._sha256(plan_path))

    needed: set[str] = set()
    visiting: set[str] = set()
    ordered: list[str] = []

    def visit(job_id: str) -> None:
        if job_id in needed:
            return
        if job_id in visiting or job_id not in jobs:
            raise ValueError("shared campaign dependency cycle or missing producer")
        visiting.add(job_id)
        for dependency in jobs[job_id]["dependencies"]:
            visit(dependency)
        visiting.remove(job_id)
        needed.add(job_id)
        ordered.append(job_id)

    for experiment in selected:
        if state["experiments"][experiment["id"]]["status"] != "NOT_RUN":
            raise ValueError(f"setting already has a run-state record: {experiment['id']}")
        for job_id in experiment["stage_job_ids"]:
            visit(job_id)
    # Validate the entire dependency closure before launching anything.
    for job_id in ordered:
        job = jobs[job_id]
        status = state["jobs"][job_id]["status"]
        if status not in {"NOT_RUN", "COMPLETED"}:
            raise ValueError(f"no implicit resume of {status} shared stage: {job_id}")
        if job["readiness"] != ("SMOKE_READY" if synthetic else "READY"):
            raise ValueError(f"execution blocked for {job_id}: {job['reason']}")
        if status == "COMPLETED" and job.get("artifact_output"):
            _verified_output(job, pinned_sha=state["jobs"][job_id]["artifact"]["sha256"])

    runtime_configs: dict[str, Path] = {}
    fixture_refs = state.setdefault("synthetic_cohorts", {})

    def runtime_config(experiment: Mapping[str, Any]) -> Path:
        experiment_id = experiment["id"]
        if experiment_id in runtime_configs:
            return runtime_configs[experiment_id]
        path = Path(experiment["config_path"])
        config = api.load_config(path)
        if synthetic:
            cohort_key = ":".join(str(experiment[key]) for key in ("dataset", "protocol", "k", "seed"))
            if cohort_key not in fixture_refs:
                created_path, _, _ = api._prepare_synthetic_run(plan_path, experiment, config)
                created = api.load_config(created_path)
                fixture_refs[cohort_key] = copy.deepcopy(created["dataset"]["manifests"])
                api._write_json(state_path, state)
            config["dataset"]["manifests"] = copy.deepcopy(fixture_refs[cohort_key])
            api.validate_config(config)
            path = plan_path.parent / "runtime_configs" / f"{experiment_id}.json"
            api._write_json(path, config)
        elif config.get("method") == "candidate_q3d":
            api.load_config(path, real_run=True)
        runtime_configs[experiment_id] = path
        return path

    for job_id in ordered:
        job = jobs[job_id]
        if state["jobs"][job_id]["status"] == "COMPLETED":
            continue
        experiment = dict(experiments[job["experiment_id"]])
        experiment["output"] = job["output"]
        config_path = runtime_config(experiment)
        inputs: dict[str, str] = {}
        pins: dict[str, str] = {}
        for stage, reference in job.get("stage_artifact_inputs", {}).items():
            upstream_id = reference["producer_job_id"]
            upstream_state = state["jobs"][upstream_id]
            if upstream_state["status"] != "COMPLETED":
                raise ValueError("shared dependency has not completed")
            artifact = _verified_output(jobs[upstream_id], pinned_sha=upstream_state["artifact"]["sha256"])
            if Path(reference["path"]).resolve() != Path(artifact["path"]):
                raise ValueError("planned shared dependency path changed")
            inputs[stage], pins[stage] = artifact["path"], artifact["sha256"]
        command = api._job_command(
            experiment, config_path=config_path,
            train_ref=job["train_manifest"], dev_ref=job["development_manifest"],
            device=device, synthetic=synthetic,
            stage=job["stage"] if job["method"] == "candidate_q3d" else None,
            stage_artifacts=inputs,
        )
        for stage, digest in sorted(pins.items()):
            command.extend(["--q3d-stage-artifact-sha", f"{stage}={digest}"])
        output = Path(job["output"])
        if output.exists() and any(output.iterdir()):
            raise FileExistsError(f"shared job output is nonempty: {output}")
        state["jobs"][job_id] = {
            "status": "RUNNING", "command": command, "exit_code": None,
            "input_manifest_sha256": pins,
        }
        api._write_json(state_path, state)
        try:
            result = runner(command, cwd=api.ROOT, check=False, capture_output=True, text=True)
            code = getattr(result, "returncode", result if isinstance(result, int) else 0)
            stdout = getattr(result, "stdout", "") or ""
            if code != 0:
                raise RuntimeError(f"shared stage failed ({code}): {job_id}")
            receipt = json.loads(stdout)
            if receipt.get("status") not in {"COMPLETED", "COMPLETED_SOFTWARE_ONLY"}:
                raise RuntimeError(f"trainer returned an incomplete stage: {job_id}")
            reported = receipt.get("completed_stages", [])
            if experiment.get("arm") == "a3d_teacher_free" or job["method"] != "candidate_q3d":
                reported = reported or ["J"]
            if job["stage"] not in reported:
                raise RuntimeError(f"trainer did not complete planned stage: {job_id}")
            record = state["jobs"][job_id]
            record.update({"status": "COMPLETED", "exit_code": 0, "stdout_tail": stdout[-4000:]})
            if job.get("artifact_output"):
                record["artifact"] = _verified_output(job)
        except Exception as exc:
            state["jobs"][job_id].update({"status": "FAILED", "error": str(exc)})
            api._write_json(state_path, state)
            raise
        api._write_json(state_path, state)

    for experiment in selected:
        artifact_refs = {
            stage: state["jobs"][producer]["artifact"]
            for stage, producer in experiment["stage_artifact_producers"].items()
            if stage in {"T", "Q", "P"}
        }
        state["experiments"][experiment["id"]] = {
            "status": "COMPLETED", "exit_code": 0,
            "completed_stages": experiment["stages"], "stage_artifacts": artifact_refs,
            "evaluation": "NOT_RUN", "baseline_reuse": experiment.get("baseline_reuse"),
        }
    api._write_json(state_path, state)
    return {
        "status": "COMPLETED", "plan": str(plan_path), "run_state": str(state_path),
        "completed_settings": [item["id"] for item in selected],
        "patient_training": "SYNTHETIC_ONLY" if synthetic else "EXECUTED",
    }
