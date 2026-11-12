"""Execute an explicitly authorized shared-stage DAG with durable SHA pins.

Planning and validation never call this module. Tests use a manufactured runner;
this delivery does not execute a patient or synthetic campaign.
"""

from __future__ import annotations

import copy
import csv
import hashlib
import math
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .artifacts import file_sha256, load_stage_manifest
from .quality import WAIVER_SCHEMA, WAIVER_SCOPE, WAIVER_STATE_KEY, quality_waiver, validate_quality_waiver

EXPLORATORY_WAIVER_SCHEMA = WAIVER_SCHEMA
EXPLORATORY_WAIVER_SCOPE = WAIVER_SCOPE
EXPLORATORY_WAIVER_FLAG = "--q3d-exploratory-quality-waiver"
# Stage manifests seal a waived checkpoint's policy under this content hash.
EXPLORATORY_WAIVER_CONTENT_HASH = "exploratory_quality_waiver_sha256"


def exploratory_waiver_policy(reason: Any) -> dict[str, Any] | None:
    """Return the shared explicit-opt-in waiver policy for ``reason``.

    ``None`` keeps the default strict behaviour. The policy is the engine's
    canonical :func:`quality.quality_waiver`; it never changes a gate
    measurement, only records that an operator explicitly continued past it.
    """

    try:
        return quality_waiver(reason)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{EXPLORATORY_WAIVER_FLAG} requires a nonempty REASON string: {exc}") from exc


def waiver_policy_sha256(policy: Mapping[str, Any]) -> str:
    """Canonical policy digest, identical to the stage-manifest content hash."""

    payload = json.dumps(dict(policy), sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def exploratory_marker(document: Any) -> dict[str, Any] | None:
    """Read an artifact/receipt exploratory declaration without trusting it as PASS.

    Returns ``None`` for normal documents and ``{"exploratory": True,
    "policy": <waiver or None>, "policy_sha256": <digest or None>}`` for
    explicitly waived ones. Stage manifests declare a waiver only through
    ``content_hashes.exploratory_quality_waiver_sha256``. Contradictory
    declarations fail closed instead of being silently treated as normal.
    """

    if not isinstance(document, Mapping):
        return None
    flag = document.get("exploratory")
    try:
        waiver = validate_quality_waiver(document.get(WAIVER_STATE_KEY))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"exploratory_quality_waiver metadata is malformed: {exc}") from exc
    content_hashes = document.get("content_hashes")
    sealed = content_hashes.get(EXPLORATORY_WAIVER_CONTENT_HASH) if isinstance(content_hashes, Mapping) else None
    if sealed is None:
        sealed = document.get(EXPLORATORY_WAIVER_CONTENT_HASH)
    if sealed is not None and (not isinstance(sealed, str) or len(sealed) != 64):
        raise ValueError("exploratory waiver content hash is malformed")
    if flag is not None and type(flag) is not bool:
        raise ValueError("exploratory marker must be a boolean")
    if flag is None and waiver is None and sealed is None:
        return None
    if flag is False:
        if waiver is not None or sealed is not None:
            raise ValueError("document declares an exploratory waiver but exploratory=false")
        return None
    if waiver is not None and sealed is not None and waiver_policy_sha256(waiver) != sealed:
        raise ValueError("exploratory waiver policy does not match its sealed content hash")
    return {
        "exploratory": True,
        "policy": waiver,
        "policy_sha256": waiver_policy_sha256(waiver) if waiver is not None else sealed,
    }


def _require_waiver_for(marker: dict[str, Any] | None, policy: dict[str, Any] | None, label: str) -> None:
    """Waived state is reusable only with the same explicit reason (engine contract)."""

    if marker is None:
        return
    if policy is None:
        raise ValueError(
            f"{label} is exploratory (quality gates waived); resume/import requires the same explicit "
            f"{EXPLORATORY_WAIVER_FLAG} REASON"
        )
    recorded = marker.get("policy_sha256")
    if recorded is not None and recorded != waiver_policy_sha256(policy):
        raise ValueError(f"{label} exploratory waiver differs from the explicitly supplied REASON")


def _identity_matches(actual: Mapping[str, Any], expected: Mapping[str, Any]) -> bool:
    return all(actual.get(key) == expected.get(key) for key in ("dataset", "protocol", "k", "seed"))


def select_campaign_experiments(
    experiments: Iterable[Mapping[str, Any]],
    *,
    setting: str,
    k: Sequence[int] | None = None,
) -> tuple[list[Mapping[str, Any]], list[int]]:
    """Select campaign roots by setting and optional K before DAG closure."""
    candidates = [
        item for item in experiments
        if setting == "all" or item.get("setting") == setting
    ]
    if not candidates:
        raise ValueError(f"selection did not match campaign experiments for setting={setting!r}")

    requested: tuple[int, ...] | None = None
    if k is not None:
        values = tuple(k)
        if not values:
            raise ValueError("K selection cannot be empty; omit --k to select every K")
        if any(not isinstance(value, int) or isinstance(value, bool) for value in values):
            raise ValueError("K selection values must be integers")
        requested = tuple(sorted(set(values)))
        available = {
            item.get("k") for item in candidates
            if isinstance(item.get("k"), int) and not isinstance(item.get("k"), bool)
        }
        unmatched = [value for value in requested if value not in available]
        if unmatched:
            raise ValueError(
                f"no campaign experiments match setting={setting!r} for requested K={unmatched}; "
                f"available K={sorted(available)}"
            )
        candidates = [item for item in candidates if item.get("k") in requested]

    selected_ks = sorted({
        item["k"] for item in candidates
        if isinstance(item.get("k"), int) and not isinstance(item.get("k"), bool)
    })
    if not selected_ks:
        raise ValueError("selected campaign experiments do not declare integer K values")
    return candidates, selected_ks


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
    verified = {"path": str(path), "sha256": digest, "content_hashes": manifest["content_hashes"]}
    marker = exploratory_marker(manifest) or exploratory_marker(raw)
    if marker is not None:
        verified["exploratory"] = True
        verified["exploratory_quality_waiver"] = marker["policy"]
        verified[EXPLORATORY_WAIVER_CONTENT_HASH] = marker["policy_sha256"]
    if "teacher_quality" in manifest:
        verified["teacher_quality"] = copy.deepcopy(manifest["teacher_quality"])
    return verified


def _finite_score(value: Any) -> float | None:
    """Normalize a reporting score without turning missing data into zero."""

    try:
        score = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(score) or not 0.0 <= score <= 1.0:
        return None
    return score


def _receipt_is_synthetic(receipt: Mapping[str, Any]) -> bool:
    return (
        receipt.get("status") == "COMPLETED_SOFTWARE_ONLY"
        or receipt.get("runtime_synthetic") is True
        or receipt.get("evidence_status") == "SOFTWARE_ONLY_SYNTHETIC"
        or receipt.get("evidence") == "SOFTWARE_ONLY"
    )


def _teacher_quality_for_experiment(
    experiment: Mapping[str, Any],
    jobs: Mapping[str, Mapping[str, Any]],
    state: Mapping[str, Any],
    *, synthetic: bool = False,
) -> tuple[float | None, dict[str, Any] | None, str]:
    producer_id = experiment.get("stage_artifact_producers", {}).get("T")
    if not isinstance(producer_id, str) or producer_id not in jobs:
        return None, None, "NOT_RUN"
    job = jobs[producer_id]
    job_state = state.get("jobs", {}).get(producer_id, {})
    if job_state.get("status") != "COMPLETED":
        return None, None, "NOT_RUN"
    receipt = job_state.get("receipt")
    if receipt is not None:
        if not isinstance(receipt, Mapping) or receipt.get("status") not in {
            "COMPLETED", "COMPLETED_SOFTWARE_ONLY",
        }:
            raise ValueError(f"teacher stage has a failed or malformed receipt: {producer_id}")
        if not synthetic and _receipt_is_synthetic(receipt):
            raise ValueError(f"real result collection rejected a synthetic teacher receipt: {producer_id}")
    artifact_state = job_state.get("artifact") if isinstance(job_state, Mapping) else None
    pinned_sha = artifact_state.get("sha256") if isinstance(artifact_state, Mapping) else None
    manifest_path = job.get("artifact_output", {}).get("manifest_path")
    if not isinstance(manifest_path, str) or not Path(manifest_path).is_file():
        return None, None, "NOT_RUN"
    if not isinstance(pinned_sha, str):
        raise ValueError(f"completed teacher stage has no pinned artifact SHA-256: {producer_id}")
    try:
        verified = _verified_output(job, pinned_sha=pinned_sha)
    except FileNotFoundError as exc:
        raise ValueError(f"completed teacher stage references a missing checkpoint: {producer_id}") from exc
    quality = verified.get("teacher_quality")
    if not isinstance(quality, Mapping):
        return None, None, "NOT_RUN"
    score = _finite_score(quality.get("teacher_dev_fg_dice"))
    if score is None:
        return None, dict(quality), "NOT_RUN"
    return score, dict(quality), "MEASURED"


def _score_from_mapping(value: Mapping[str, Any]) -> float | None:
    """Read only explicit student VAL selection fields from a trainer receipt."""

    for key in ("student_dev_fg_dice", "best_student_score", "development_patient_macro_fg_dice"):
        score = _finite_score(value.get(key))
        if score is not None:
            return score
    development = value.get("development")
    if isinstance(development, Mapping):
        metric = development.get("metric")
        if metric in {None, "development_patient_macro_fg_dice"}:
            score = _finite_score(development.get("score"))
            if score is not None:
                return score
    return None


def _selected_c0_score(
    experiment: Mapping[str, Any],
    jobs: Mapping[str, Mapping[str, Any]],
    state: Mapping[str, Any],
    *, synthetic: bool = False,
) -> float | None:
    producer_id = experiment.get("stage_artifact_producers", {}).get("J")
    if not isinstance(producer_id, str) or producer_id not in jobs:
        return None
    job = jobs[producer_id]
    job_state = state.get("jobs", {}).get(producer_id, {})
    if job_state.get("status") != "COMPLETED":
        return None
    for source in (job_state, state.get("experiments", {}).get(experiment.get("id"), {})):
        if isinstance(source, Mapping):
            receipt = source.get("receipt")
            if isinstance(receipt, Mapping):
                if not synthetic and _receipt_is_synthetic(receipt):
                    raise ValueError(
                        f"real result collection rejected a synthetic student receipt: {producer_id}"
                    )
                score = _score_from_mapping(receipt)
                if score is not None:
                    return score
            score = _score_from_mapping(source)
            if score is not None:
                return score
    output = job.get("output")
    if not isinstance(output, str):
        return None
    output_dir = Path(output).expanduser().resolve()
    checkpoint = output_dir / "latest.pt"
    if checkpoint.is_file():
        try:
            import torch

            try:
                payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
            except TypeError:  # pragma: no cover - older torch compatibility
                payload = torch.load(checkpoint, map_location="cpu")
        except Exception:  # noqa: BLE001 - an absent/legacy checkpoint is NOT_RUN
            payload = None
        if isinstance(payload, Mapping):
            score = _score_from_mapping(payload)
            if score is not None:
                return score
    metrics_path = output_dir / "metrics.jsonl"
    if metrics_path.is_file():
        last_score: float | None = None
        for line in metrics_path.read_text(encoding="utf-8").splitlines():
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(record, Mapping) and record.get("stage") == "J" \
                    and record.get("event") == "development_selection":
                score = _score_from_mapping(record)
                if score is not None:
                    last_score = score
        return last_score
    return None


def _job_exploratory_policy(job_id: Any, state: Mapping[str, Any]) -> dict[str, Any] | None:
    """Return the exploratory marker recorded for one completed job, if any."""

    if not isinstance(job_id, str):
        return None
    job_state = state.get("jobs", {}).get(job_id, {})
    if not isinstance(job_state, Mapping) or job_state.get("status") != "COMPLETED":
        return None
    for document in (job_state, job_state.get("receipt"), job_state.get("artifact")):
        marker = exploratory_marker(document)
        if marker is not None:
            return marker
    return None


def _experiment_exploratory(experiment: Mapping[str, Any], state: Mapping[str, Any],
                            stages: Sequence[str]) -> list[dict[str, Any]]:
    producers = experiment.get("stage_artifact_producers", {})
    markers = []
    if isinstance(producers, Mapping):
        for stage in stages:
            marker = _job_exploratory_policy(producers.get(stage), state)
            if marker is not None:
                markers.append({"stage": stage, "job_id": producers.get(stage), **marker})
    return markers


def collect_campaign_results(
    plan_path: str | Path,
    *,
    plan: Mapping[str, Any] | None = None,
    state: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Collect teacher VAL quality beside the selected C0 VAL score.

    The join key is exactly ``(dataset, protocol, k, seed)``.  Teacher quality
    is read only from the verified shared T manifest, while C0 quality is read
    from its completed J receipt/checkpoint.  Missing or unselected artifacts
    stay ``None`` with ``NOT_RUN`` status; no score is inferred from another K
    or from a different arm.
    """

    resolved_plan = Path(plan_path).expanduser().resolve()
    if plan is None:
        plan = json.loads(resolved_plan.read_text(encoding="utf-8"))
    if state is None:
        state_path = resolved_plan.parent / "RUN_STATE.json"
        state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.is_file() else {"jobs": {}, "experiments": {}}
    jobs = {job["id"]: job for job in plan.get("stage_jobs", []) if isinstance(job, Mapping) and isinstance(job.get("id"), str)}
    experiments = [item for item in plan.get("experiments", []) if isinstance(item, Mapping)]
    identities: dict[tuple[Any, ...], dict[str, Any]] = {}
    exploratory_sources: list[dict[str, Any]] = []
    for experiment in experiments:
        key = tuple(experiment.get(name) for name in ("dataset", "protocol", "k", "seed"))
        if any(value is None for value in key):
            continue
        row = identities.setdefault(key, {
            "dataset": key[0], "protocol": key[1], "k": key[2], "seed": key[3],
            "teacher_dev_fg_dice": None, "teacher_dev_per_class": None,
            "teacher_status": "NOT_RUN", "c0_dev_fg_dice": None,
            "c0_status": "NOT_RUN", "delta_c0_minus_teacher": None,
            "teacher_exploratory": False, "c0_exploratory": False, "exploratory": False,
        })
        # A waived V1/C1 run still makes this campaign row exploratory when
        # its reused T is strict and C0 has not run yet.
        experiment_markers = _experiment_exploratory(experiment, state, ("T", "Q", "P", "J"))
        row["exploratory"] = bool(row["exploratory"] or experiment_markers)
        exploratory_sources.extend(experiment_markers)
        if experiment.get("setting") == "c0":
            score = _selected_c0_score(
                experiment, jobs, state,
                synthetic=bool(plan.get("selection", {}).get("synthetic", False)),
            )
            if score is not None:
                row["c0_dev_fg_dice"] = score
                row["c0_status"] = "MEASURED"
                markers = _experiment_exploratory(experiment, state, ("T", "Q", "P", "J"))
                if markers:
                    row["c0_exploratory"] = True
                    exploratory_sources.extend(markers)
        if experiment.get("setting") in {"v1", "v1u", "c1", "teacher_supervised"}:
            score, quality, status = _teacher_quality_for_experiment(
                experiment, jobs, state,
                synthetic=bool(plan.get("selection", {}).get("synthetic", False)),
            )
            if status == "MEASURED" or row["teacher_status"] == "NOT_RUN":
                row["teacher_dev_fg_dice"] = score
                row["teacher_dev_per_class"] = quality.get("teacher_dev_per_class") if quality else None
                row["teacher_status"] = status
                markers = _experiment_exploratory(experiment, state, ("T",)) if status == "MEASURED" else []
                row["teacher_exploratory"] = bool(markers)
                exploratory_sources.extend(markers)
    rows = []
    for row in identities.values():
        row["exploratory"] = bool(row["exploratory"] or row["teacher_exploratory"] or row["c0_exploratory"])
        if row["teacher_status"] == "MEASURED" and row["c0_status"] == "MEASURED":
            row["delta_c0_minus_teacher"] = row["c0_dev_fg_dice"] - row["teacher_dev_fg_dice"]
        rows.append(row)
    rows.sort(key=lambda row: (str(row["dataset"]), str(row["protocol"]), int(row["k"]), int(row["seed"])))
    report = {
        "schema": "self_audit.candidate_q3d.campaign_results.v1",
        "status": "MEASURED" if any(row["teacher_status"] == "MEASURED" or row["c0_status"] == "MEASURED" for row in rows) else "NOT_RUN",
        "join_key": ["dataset", "protocol", "k", "seed"],
        "rows": rows,
        "exploratory": any(row["exploratory"] for row in rows),
    }
    if report["exploratory"]:
        unique: dict[str, dict[str, Any]] = {}
        for item in exploratory_sources:
            unique.setdefault(str(item["job_id"]), item)
        report["exploratory_policy"] = (
            "EXPLORATORY_QUALITY_GATES_WAIVED: gate measurements remain NO_GO/false; "
            "scores are not normal contract evidence"
        )
        report["exploratory_sources"] = [unique[key] for key in sorted(unique)]
    return report


def write_campaign_results(report: Mapping[str, Any], out_dir: str | Path) -> dict[str, str]:
    """Write a compact JSON, CSV, and Markdown view of a collected report."""

    output = Path(out_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    json_path = output / "RESULTS.json"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    fields = ["dataset", "protocol", "k", "seed", "teacher_dev_fg_dice", "teacher_status",
              "c0_dev_fg_dice", "c0_status", "delta_c0_minus_teacher", "teacher_dev_per_class"]
    exploratory = bool(report.get("exploratory")) or any(
        isinstance(row, Mapping) and row.get("exploratory") for row in report.get("rows", [])
    )
    if exploratory:
        # Strict reports keep the historical header byte-for-byte; waived
        # reports append explicit markers so no row can be read as normal.
        fields = [*fields, "exploratory", "teacher_exploratory", "c0_exploratory"]
    csv_path = output / "RESULTS.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in report.get("rows", []):
            serial = dict(row)
            serial["teacher_dev_per_class"] = json.dumps(serial.get("teacher_dev_per_class"), sort_keys=True)
            writer.writerow({field: serial.get(field) for field in fields})
    md_path = output / "RESULTS.md"
    lines = [
        "# Q3D campaign results", "",
        "Scores are selected validation metrics only; missing artifacts remain `NOT_RUN`.", "",
        *([
            "**EXPLORATORY: quality gates were explicitly waived. Gate measurements remain NO_GO; "
            "rows marked `exploratory` are not normal contract evidence.**", "",
        ] if exploratory else []),
        "| dataset | protocol | K | seed | teacher val FG Dice@K | teacher status | C0 val FG Dice@K | C0 status | C0 - teacher |",
        "|---|---:|---:|---:|---:|---|---:|---|---:|",
    ]
    if exploratory:
        lines[-2] = lines[-2][:-2] + " | evidence |"
        lines[-1] = lines[-1] + "---|"
    for row in report.get("rows", []):
        values = [row.get("dataset"), row.get("protocol"), row.get("k"), row.get("seed"),
                  row.get("teacher_dev_fg_dice"), row.get("teacher_status"),
                  row.get("c0_dev_fg_dice"), row.get("c0_status"), row.get("delta_c0_minus_teacher")]
        if exploratory:
            values.append("EXPLORATORY" if row.get("exploratory") else "")
        lines.append("| " + " | ".join("" if value is None else str(value) for value in values) + " |")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"json": str(json_path), "csv": str(csv_path), "markdown": str(md_path)}


def execute_shared_campaign(
    plan_path: Path,
    plan: Mapping[str, Any],
    *,
    setting: str,
    k: Sequence[int] | None = None,
    through_stage: str = "J",
    device: str,
    runner: Any,
    api: Any,
    exploratory_quality_waiver: str | None = None,
) -> dict[str, Any]:
    """Run each needed producer once, then launch only each selected J stage.

    ``api`` supplies the script's existing config/fixture/CLI helpers. Completed
    producers can be reused across later setting selections only after their
    stored manifest SHA and every recursive checkpoint dependency revalidate.
    Interrupted/failed jobs require an explicit recovery outside this launcher.

    ``exploratory_quality_waiver`` (default ``None``: strict) forwards the
    explicit operator reason to every launched trainer command. Reusing or
    importing any exploratory producer artifact requires that reason again.
    """
    waiver_policy = exploratory_waiver_policy(exploratory_quality_waiver)
    synthetic = bool(plan["selection"]["synthetic"])
    if not synthetic and device != "cuda":
        raise ValueError("real campaign execution requires explicit CUDA")
    stage_order = ("T", "Q", "P", "J")
    if through_stage not in stage_order:
        raise ValueError(f"through-stage must be one of {stage_order}")
    jobs = {job["id"]: job for job in plan["stage_jobs"]}
    experiments = {item["id"]: item for item in plan["experiments"]}
    selected, selected_ks = select_campaign_experiments(
        experiments.values(), setting=setting, k=k,
    )
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
        if state["experiments"][experiment["id"]]["status"] not in {"NOT_RUN", "PARTIAL"}:
            raise ValueError(f"setting already has a completed run-state record: {experiment['id']}")
        root_id = experiment.get("stage_artifact_producers", {}).get("J")
        if root_id is None:
            root_id = next(
                (job_id for job_id in experiment.get("stage_job_ids", [])
                 if job_id in jobs and jobs[job_id]["stage"] == "J"),
                None,
            )
        if root_id is None:
            # Explicit comparison baselines (for example teacher_supervised)
            # may end at T. Preserve their declared producer roots.
            roots = experiment.get("stage_job_ids", [])
            if not roots:
                raise ValueError(f"campaign experiment has no stage roots: {experiment['id']}")
            for job_id in roots:
                visit(job_id)
        else:
            visit(root_id)
    stage_rank = {stage: index for index, stage in enumerate(stage_order)}
    ordered = [job_id for job_id in ordered if stage_rank[jobs[job_id]["stage"]] <= stage_rank[through_stage]]
    if not ordered:
        raise ValueError(f"no campaign stages are available through {through_stage} for the selected roots")
    # Validate the entire dependency closure before launching anything.
    for job_id in ordered:
        job = jobs[job_id]
        status = state["jobs"][job_id]["status"]
        if status not in {"NOT_RUN", "COMPLETED"}:
            raise ValueError(f"no implicit resume of {status} shared stage: {job_id}")
        if job["readiness"] != ("SMOKE_READY" if synthetic else "READY"):
            raise ValueError(f"execution blocked for {job_id}: {job['reason']}")
        if status == "COMPLETED" and job.get("artifact_output"):
            reused = _verified_output(job, pinned_sha=state["jobs"][job_id]["artifact"]["sha256"])
            _require_waiver_for(exploratory_marker(reused), waiver_policy, f"completed shared stage {job_id}")
        if status == "COMPLETED":
            _require_waiver_for(_job_exploratory_policy(job_id, state), waiver_policy,
                                f"completed shared stage {job_id}")

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
            _require_waiver_for(exploratory_marker(artifact), waiver_policy, f"imported stage artifact {upstream_id}")
            if Path(reference["path"]).resolve() != Path(artifact["path"]):
                raise ValueError("planned shared dependency path changed")
            inputs[stage], pins[stage] = artifact["path"], artifact["sha256"]
        waiver_kwargs = (
            {"exploratory_quality_waiver": waiver_policy["reason"]}
            if waiver_policy is not None and job["method"] == "candidate_q3d" else {}
        )
        command = api._job_command(
            experiment, config_path=config_path,
            train_ref=job["train_manifest"], dev_ref=job["development_manifest"],
            device=device, synthetic=synthetic,
            stage=job["stage"] if job["method"] == "candidate_q3d" else None,
            stage_artifacts=inputs,
            **waiver_kwargs,
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
        if waiver_kwargs:
            state["jobs"][job_id]["exploratory_quality_waiver_requested"] = copy.deepcopy(waiver_policy)
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
            synthetic_receipt = (
                receipt.get("status") == "COMPLETED_SOFTWARE_ONLY"
                or receipt.get("runtime_synthetic") is True
                or receipt.get("evidence_status") == "SOFTWARE_ONLY_SYNTHETIC"
                or receipt.get("evidence") == "SOFTWARE_ONLY"
            )
            if not synthetic and synthetic_receipt:
                raise RuntimeError(
                    f"real campaign rejected a synthetic receipt: {job_id}"
                )
            receipt_marker = exploratory_marker(receipt)
            if receipt_marker is not None and not waiver_kwargs:
                raise RuntimeError(
                    f"trainer returned an exploratory receipt without an explicit waiver: {job_id}"
                )
            _require_waiver_for(receipt_marker, waiver_policy, f"trainer receipt {job_id}")
            reported = receipt.get("completed_stages", [])
            if experiment.get("arm") == "a3d_teacher_free" or job["method"] != "candidate_q3d":
                reported = reported or ["J"]
            if job["stage"] not in reported:
                raise RuntimeError(f"trainer did not complete planned stage: {job_id}")
            record = state["jobs"][job_id]
            record.update({
                "status": "COMPLETED", "exit_code": 0, "stdout_tail": stdout[-4000:],
                "receipt": copy.deepcopy(receipt),
            })
            if job.get("artifact_output"):
                record["artifact"] = _verified_output(job)
                artifact_marker = exploratory_marker(record["artifact"])
                if artifact_marker is not None and not waiver_kwargs:
                    raise RuntimeError(
                        f"trainer produced an exploratory artifact without an explicit waiver: {job_id}"
                    )
                _require_waiver_for(artifact_marker, waiver_policy, f"produced artifact {job_id}")
                receipt_marker = receipt_marker or artifact_marker
            if receipt_marker is not None:
                record["exploratory"] = True
                record["exploratory_quality_waiver"] = receipt_marker["policy"] or copy.deepcopy(waiver_policy)
        except Exception as exc:
            state["jobs"][job_id].update({"status": "FAILED", "error": str(exc)})
            api._write_json(state_path, state)
            raise
        api._write_json(state_path, state)

    completed_settings = []
    for experiment in selected:
        experiment_jobs = [jobs[job_id] for job_id in experiment["stage_job_ids"]]
        completed_stages = [
            stage for stage in experiment.get("stages", stage_order)
            if any(job["stage"] == stage and state["jobs"][job["id"]]["status"] == "COMPLETED"
                   for job in experiment_jobs)
        ]
        is_complete = all(state["jobs"][job["id"]]["status"] == "COMPLETED" for job in experiment_jobs)
        if is_complete:
            completed_settings.append(experiment["id"])
        artifact_refs = {}
        for stage, producer in experiment.get("stage_artifact_producers", {}).items():
            if stage in {"T", "Q", "P"} and state["jobs"].get(producer, {}).get("status") == "COMPLETED":
                artifact = state["jobs"][producer].get("artifact")
                if artifact is not None:
                    artifact_refs[stage] = artifact
        state["experiments"][experiment["id"]] = {
            "status": "COMPLETED" if is_complete else "PARTIAL",
            "exit_code": 0 if is_complete else None,
            "completed_stages": completed_stages, "stage_artifacts": artifact_refs,
            "evaluation": "NOT_RUN", "baseline_reuse": experiment.get("baseline_reuse"),
        }
        experiment_markers = [
            job["id"] for job in experiment_jobs if _job_exploratory_policy(job["id"], state) is not None
        ]
        if experiment_markers:
            state["experiments"][experiment["id"]]["exploratory"] = True
            state["experiments"][experiment["id"]]["exploratory_jobs"] = experiment_markers
    api._write_json(state_path, state)
    results = collect_campaign_results(plan_path, plan=plan, state=state)
    result_paths = write_campaign_results(results, plan_path.parent / "results")
    summary = {
        "status": "COMPLETED" if len(completed_settings) == len(selected) else "PARTIAL",
        "plan": str(plan_path), "run_state": str(state_path),
        "selected_settings": [item["id"] for item in selected],
        "completed_settings": completed_settings,
        "selected_k": selected_ks,
        "through_stage": through_stage,
        "results": result_paths,
        "patient_training": "SYNTHETIC_ONLY" if synthetic else "EXECUTED",
    }
    if waiver_policy is not None or results.get("exploratory"):
        summary["exploratory"] = True
        summary["exploratory_quality_waiver"] = copy.deepcopy(waiver_policy)
    return summary
