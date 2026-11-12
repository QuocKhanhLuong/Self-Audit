#!/usr/bin/env python3
"""Run the Q3D T/Q/P/J path on a temporary manufactured native-volume fixture."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import tempfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_run_state(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    import torch

    sidecar = path.with_suffix(path.suffix + ".sha256")
    if not sidecar.is_file():
        raise ValueError("synthetic smoke research checkpoint is missing its SHA-256 sidecar")
    sidecar_parts = sidecar.read_text(encoding="ascii").split()
    if len(sidecar_parts) != 2 or sidecar_parts[1] != path.name or sidecar_parts[0] != _sha256(path):
        raise ValueError("synthetic smoke research checkpoint SHA-256 sidecar mismatch")
    value = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(value, dict) or value.get("schema") != "self_audit.candidate_q3d.resume.v2":
        raise ValueError("synthetic smoke requires a Q3D resume.v2 research checkpoint")
    return dict(value)


def _manifest_record(path: Path, sha256: str) -> dict[str, Any]:
    return {"path": str(path.resolve()), "sha256": sha256, "status": "READY"}


def _manufacture_cine_frames(manifest_path: Path) -> str:
    """Give each two-frame synthetic cine a known ED/ES intensity cue."""
    import numpy as np

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    records = manifest.get("patients", []) if manifest.get("role") == "development" else (
        list(manifest.get("labeled", [])) + list(manifest.get("unlabeled", []))
    )
    ordered_records = sorted(records, key=lambda record: str(record["id"]))
    patient_offsets = {
        record["id"]: (index - (len(ordered_records) - 1) / 2.0) * 0.002
        for index, record in enumerate(ordered_records)
    }
    for record in records:
        image_path = manifest_path.parent / record["image"]
        cine = np.load(image_path, allow_pickle=False)
        if cine.ndim != 4 or cine.shape[0] != 2:
            raise ValueError("Q3D synthetic smoke requires two native cine frames per patient")
        patient_offset = patient_offsets[record["id"]]
        cine[0].fill(0.2 + patient_offset)
        cine[1].fill(0.8 + patient_offset)
        np.save(image_path, cine, allow_pickle=False)
        record["image_sha256"] = _sha256(image_path)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return _sha256(manifest_path)


def _manufactured_teacher(config: dict[str, Any]):
    """Shared software-only teacher fixture exported by the train dispatcher."""
    from train import _manufactured_teacher as make_teacher

    return make_teacher(config)


def _tokenizer_initialization_count(state: dict[str, Any] | None) -> int | None:
    tokenizer_state = (state or {}).get("tokenizer_state")
    if not isinstance(tokenizer_state, dict):
        return None
    value = tokenizer_state.get("initialization_volume_count")
    if hasattr(value, "item"):
        return int(value.item())
    return int(value) if isinstance(value, int) else None


_ALL_RECIPES = (
    "a3d_teacher_free", "b1", "c0", "c1", "e1", "mednext_supervised",
    "mean_teacher", "o1", "s1", "v0", "v1", "v1_decoder_stride8",
    "v1_residual2", "v1_stride4", "v1u",
)


def _recipe_config(name: str, canonical_config):
    kwargs: dict[str, Any] = {"dataset": "acdc", "protocol": "A", "k": 2, "seed": 42}
    if name == "v1_stride4":
        return canonical_config("v1", source_stride=4, **kwargs)
    if name == "v1_decoder_stride8":
        return canonical_config("v1", feature_source="decoder_stride8", **kwargs)
    if name == "v1_residual2":
        return canonical_config("v1", residual2=True, **kwargs)
    return canonical_config(name, **kwargs)


def _run_recipe(
    name: str,
    output_dir: Path,
    fixture: dict[str, Any],
    args,
    *,
    stage: str | None = None,
    stage_artifacts: dict[str, str] | None = None,
) -> dict[str, Any]:
    import argparse as argparse_module
    import torch

    from self_audit.candidate_q3d.config import (
        arm_settings, canonical_config, stages_for_arm, validate_config,
    )
    from self_audit.candidate_q3d.engine import run_training

    config = _recipe_config(name, canonical_config)
    settings = arm_settings(config["arm"])
    has_tokenizer = bool(settings["tokenizer"])
    config["runtime"].update({
        "synthetic": True,
        "device": "cpu",
        "smoke_initialization_volumes": args.init_volumes if has_tokenizer else None,
        "synthetic_gate_override": has_tokenizer,
        "smoke_update_overrides": {"T": 1, "Q": args.q_updates, "P": 1, "J": 4},
    })
    config["dataset"]["manifests"]["train"] = _manifest_record(
        fixture["train"], fixture["train_sha256"],
    )
    config["dataset"]["manifests"]["development"] = _manifest_record(
        fixture["evaluation"], fixture["evaluation_sha256"],
    )
    validated = validate_config(config)
    expected_stages = list(stages_for_arm(validated["arm"]))
    if stage is None:
        expected_completed_stages = expected_stages
    elif stage in expected_stages:
        # The engine only executes the requested stage, but its returned
        # completed_stages also records the verified imported lineage.
        expected_completed_stages = expected_stages[:expected_stages.index(stage) + 1]
    else:
        raise ValueError(f"stage {stage!r} is not part of recipe {name!r}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    config_path = output_dir.with_name(output_dir.name + ".config.json")
    config_path.write_text(
        json.dumps(validated, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    config_sha256 = _sha256(config_path)

    engine_result: dict[str, Any] | None = None
    engine_error: str | None = None
    if name in {"a3d_teacher_free", "mednext_supervised"}:
        try:
            from train import _train_q3d

            cli_args = argparse_module.Namespace(
                model_config=None, teacher_checkpoint=None, teacher_checkpoint_sha256=None,
                data=None, data_sha256=None, validation_data=None, validation_data_sha256=None,
                out=str(output_dir), synthetic=True, allow_full_run=False, device="cpu",
            )
            result = _train_q3d(cli_args, argparse_module.ArgumentParser(), validated)
            engine_result = dict(result)
        except Exception as exc:
            engine_error = f"{type(exc).__name__}: {exc}"
    else:
        try:
            torch.manual_seed(19)
            teacher = _manufactured_teacher(validated) if "T" in expected_stages else None
            result = run_training(
                validated, str(output_dir), device="cpu", smoke=True, teacher=teacher,
                synthetic_gate_override=validated["runtime"]["synthetic_gate_override"],
                stage=stage, stage_artifacts=stage_artifacts,
            )
            engine_result = dict(result) if isinstance(result, dict) else {"result": result}
        except Exception as exc:
            engine_error = f"{type(exc).__name__}: {exc}"

    state = None if name == "a3d_teacher_free" else _load_run_state(output_dir / "latest.pt")
    gate_record = (state or {}).get("gate")
    if engine_result:
        gate_record = engine_result.get("tokenizer_gate", engine_result.get("gate", gate_record))
    if gate_record is None and engine_error and "tokenizer gate failed" in engine_error.lower():
        gate_record = {"status": "NO_GO_DETAILS_IN_ENGINE_ERROR", "passed": False,
                       "detail": engine_error}
    if gate_record is None and engine_error and "did not reach k-means++ initialization" in engine_error:
        gate_record = {"status": "NOT_READY", "passed": False, "detail": engine_error}
    if gate_record is None and has_tokenizer:
        gate_record = {
            "status": "NOT_RECORDED" if state is not None else "NOT_RUN",
            "passed": None,
            "detail": engine_error or "tokenizer stage did not record its gate result",
        }
    override_used = bool((engine_result or {}).get(
        "synthetic_gate_override_used", (engine_result or {}).get("synthetic_gate_override", False),
    ))
    actual_gate = gate_record
    if isinstance(gate_record, dict) and gate_record.get("status") == "SYNTHETIC_TEST_OVERRIDE":
        override_used = True
        actual_gate = gate_record.get("original_gate")

    completed = list((state or {}).get("completed_stages", []))
    if engine_result:
        completed = list(engine_result.get("completed_stages", engine_result.get("stages", completed)))
    if name == "a3d_teacher_free" and (engine_result or {}).get("status", "").startswith("COMPLETED"):
        completed = expected_completed_stages
    stage_artifact_hashes = (engine_result or {}).get("stage_artifact_hashes", {})
    if not isinstance(stage_artifact_hashes, dict):
        stage_artifact_hashes = {}
    gate_required = has_tokenizer and "Q" in expected_completed_stages
    gate_missing = gate_required and (actual_gate is None or actual_gate.get("passed") is None)
    gate_failed = actual_gate is not None and actual_gate.get("passed") is False
    if gate_missing:
        status = "FAILED"
    elif gate_failed and not override_used:
        status = "STOPPED_AT_TOKENIZER_GATE"
    elif engine_error:
        status = "FAILED"
    elif completed != expected_completed_stages:
        status = "FAILED"
    elif gate_failed and override_used:
        status = "COMPLETED_WITH_SYNTHETIC_GATE_OVERRIDE"
    else:
        status = "COMPLETED_DECLARED_SYNTHETIC_STAGES"

    report = {
        "schema": "self_audit.candidate_q3d.smoke.v1",
        "recipe": name,
        "arm": validated["arm"],
        "status": status,
        "evidence_status": "SOFTWARE_ONLY_SYNTHETIC",
        "runtime_synthetic": True,
        "device": "cpu",
        "fixture_train_manifest_sha256": fixture["train_sha256"],
        "fixture_development_manifest_sha256": fixture["evaluation_sha256"],
        "config_path": str(config_path.resolve()),
        "config_sha256": config_sha256,
        "configured_stage_updates": validated["runtime"]["smoke_update_overrides"],
        "tokenizer_initialization_volumes": _tokenizer_initialization_count(state),
        "tokenizer_initialization_required_volumes": validated["tokenizer"]["initialization_volumes"],
        "synthetic_initialization_volume_override": validated["runtime"]["smoke_initialization_volumes"],
        "synthetic_gate_override_requested": validated["runtime"]["synthetic_gate_override"],
        "synthetic_teacher_fixture": ({
            "kind": "manufactured_coordinate_partition",
            "class_mapping": ["BG", "RV", "MYO", "LV"],
            "frame_intensity_cue": {"frame_0": 0.2, "frame_1": 0.8},
            "confidence_logits": [10.0, -10.0],
            "physiology_design": "frame_0 LV/RV ED exceeds frame_1 by equal volumes; MYO is constant",
        } if "T" in expected_stages else None),
        "completed_stages": completed,
        "expected_stages": expected_stages,
        "expected_completed_stages": expected_completed_stages,
        "requested_stage": stage,
        "tokenizer_gate": actual_gate or {"status": "NOT_APPLICABLE", "passed": None},
        "tokenizer_gate_engine_record": gate_record,
        "synthetic_gate_override_used": override_used,
        "engine_error": engine_error,
        "engine_result": engine_result,
        "physiology_cohort_weighting": (engine_result or {}).get("physiology_cohort_weighting"),
        "stage_artifact_inputs": {
            key: {
                "path": str(Path(path).resolve()),
                "manifest_sha256": _sha256(Path(path)) if Path(path).is_file() else None,
            }
            for key, path in sorted((stage_artifacts or {}).items())
        },
        "stage_artifact_hashes": {
            key: value
            for key, value in sorted(stage_artifact_hashes.items())
            if key in (stage_artifacts or {})
        },
        "patient_metrics_claimed": False,
        "gpu_evidence": "NOT_RUN",
        "real_data_evidence": "NOT_RUN",
    }
    report_path = output_dir.with_name(output_dir.name + ".smoke.json")
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")
    return report


def _run_shared_v1_tqp(output_dir: Path, fixture: dict[str, Any], args) -> dict[str, Any]:
    """Produce one synthetic V1 T/Q/P artifact chain for V1 and V1-u J."""
    producer_root = output_dir / "_shared_v1_tqp"
    artifact_inputs: dict[str, str] = {}
    stage_reports: dict[str, dict[str, Any]] = {}
    for stage in ("T", "Q", "P"):
        stage_output = producer_root / stage
        report = _run_recipe(
            "v1", stage_output, fixture, args, stage=stage,
            stage_artifacts=artifact_inputs or None,
        )
        stage_reports[stage] = report
        result = report.get("engine_result") or {}
        refs = result.get("stage_artifacts", {})
        reference = refs.get(stage) if isinstance(refs, dict) else None
        manifest_path = None
        if isinstance(reference, dict):
            manifest_path = reference.get("manifest_path", reference.get("path"))
        artifact_path = Path(manifest_path) if isinstance(manifest_path, str) else (
            stage_output / f"{stage}.artifact.json"
        )
        artifact_inputs[stage] = str(artifact_path.resolve())

    artifact_records = {}
    for stage, path_text in artifact_inputs.items():
        path = Path(path_text)
        result = stage_reports[stage].get("engine_result") or {}
        content_hashes = result.get("stage_artifact_hashes", {})
        artifact_records[stage] = {
            "path": path_text,
            "manifest_sha256": _sha256(path) if path.is_file() else None,
            "stage_artifact_hashes": content_hashes.get(stage) if isinstance(content_hashes, dict) else None,
            "producer_status": stage_reports[stage]["status"],
            "producer_requested_stage": stage_reports[stage]["requested_stage"],
            "producer_active_stages": (stage_reports[stage].get("engine_result") or {}).get("active_stages"),
            "producer_completed_stages": stage_reports[stage]["completed_stages"],
        }
    return {
        "producer": "v1",
        "producer_stage_order": ["T", "Q", "P"],
        "artifacts": artifact_records,
        "stage_reports": {
            stage: str((producer_root / stage).with_name(f"{stage}.smoke.json").resolve())
            for stage in ("T", "Q", "P")
        },
        "all_producer_stages_completed": all(
            stage_reports[stage]["status"].startswith("COMPLETED")
            and stage_reports[stage]["requested_stage"] == stage
            and (stage_reports[stage].get("engine_result") or {}).get("active_stages") == [stage]
            for stage in ("T", "Q", "P")
        ),
        "gate": stage_reports["Q"]["tokenizer_gate"],
        "synthetic_gate_override_requested": stage_reports["Q"]["synthetic_gate_override_requested"],
        "synthetic_gate_override_used": any(
            stage_reports[stage]["synthetic_gate_override_used"]
            for stage in ("T", "Q", "P")
        ),
    }


def _validate_v1_v1u_omega(run_reports: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Check the synthetic physiology cache reaches both J arms without a gate bypass."""
    records: dict[str, dict[str, Any]] = {}
    for name in ("v1", "v1u"):
        report = run_reports[name]
        weighting = report.get("physiology_cohort_weighting")
        omega_raw = weighting.get("omega") if isinstance(weighting, dict) else None
        try:
            omega = {str(key): float(value) for key, value in omega_raw.items()} if isinstance(omega_raw, dict) else {}
        except (TypeError, ValueError):
            omega = {}
        computed_mean = sum(omega.values()) / len(omega) if omega else None
        reported_mean = weighting.get("mean_omega") if isinstance(weighting, dict) else None
        records[name] = {
            "omega": omega,
            "computed_mean_omega": computed_mean,
            "reported_mean_omega": reported_mean,
            "gate_passed": bool(weighting.get("gate_passed")) if isinstance(weighting, dict) else False,
            "override_used": bool(weighting.get("override_used")) if isinstance(weighting, dict) else True,
        }

    v1 = records["v1"]
    v1u = records["v1u"]
    shared_patients = bool(v1["omega"]) and set(v1["omega"]) == set(v1u["omega"])
    omega_differs = shared_patients and any(
        abs(v1["omega"][patient] - v1u["omega"][patient]) > 1.0e-6
        for patient in v1["omega"]
    )
    v1_is_nonuniform = bool(v1["omega"]) and any(
        not math.isclose(value, 1.0, rel_tol=0.0, abs_tol=1.0e-6)
        for value in v1["omega"].values()
    )
    v1u_is_uniform = bool(v1u["omega"]) and all(
        math.isclose(value, 1.0, rel_tol=0.0, abs_tol=1.0e-6)
        for value in v1u["omega"].values()
    )
    means_are_one = all(
        record["computed_mean_omega"] is not None
        and math.isclose(record["computed_mean_omega"], 1.0, rel_tol=0.0, abs_tol=1.0e-6)
        and isinstance(record["reported_mean_omega"], (int, float))
        and math.isclose(record["reported_mean_omega"], 1.0, rel_tol=0.0, abs_tol=1.0e-6)
        for record in records.values()
    )
    gates_pass_without_override = all(
        record["gate_passed"] and not record["override_used"]
        for record in records.values()
    )
    return {
        "v1": records["v1"],
        "v1u": records["v1u"],
        "shared_patient_ids": shared_patients,
        "omega_differs": omega_differs,
        "v1_is_nonuniform": v1_is_nonuniform,
        "v1u_is_uniform": v1u_is_uniform,
        "means_are_one": means_are_one,
        "gates_passed_without_override": gates_pass_without_override,
        "passed": (
            shared_patients and omega_differs and v1_is_nonuniform and v1u_is_uniform
            and means_are_one and gates_pass_without_override
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, help="New output directory; must not exist")
    parser.add_argument("--all-arms", action="store_true", help="Run all 15 Q3D arm/variant recipes")
    parser.add_argument("--recipe", choices=_ALL_RECIPES, default="v1",
                        help="One bounded recipe to run when --all-arms is absent")
    parser.add_argument("--q-updates", type=int, default=1,
                        help="Synthetic tokenizer optimizer updates after the separate initialization pass")
    parser.add_argument("--init-volumes", type=int, default=32,
                        help="Explicit synthetic-only bound for tokenizer initialization volume visits")
    args = parser.parse_args()
    output_dir = Path(args.out).resolve()
    report_path = output_dir.with_name(output_dir.name + ".smoke.json")
    if output_dir.exists() or report_path.exists():
        parser.error("--out and its smoke report path must not already exist")
    recipe_names = _ALL_RECIPES if args.all_arms else (args.recipe,)
    config_paths = [
        (output_dir / name).with_name(name + ".config.json") if args.all_arms
        else output_dir.with_name(output_dir.name + ".config.json")
        for name in recipe_names
    ]
    if any(path.exists() for path in config_paths):
        parser.error("a synthetic recipe config artifact already exists beside --out")
    if args.q_updates < 1:
        parser.error("--q-updates must be a positive optimizer-update count")
    if args.init_volumes < 1 or args.init_volumes >= 200:
        parser.error("--init-volumes must be in [1,199] for a bounded synthetic override")

    from self_audit.candidate_a3d.data import synthetic_fixture
    import torch

    torch.set_num_threads(1)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="q3d-smoke-") as temporary:
        fixture = synthetic_fixture(
            Path(temporary) / "fixture", seed=19, n_labeled=2, n_unlabeled=2, n_evaluation=1,
            shape_tzyx=(2, 8, 32, 32),
        )
        fixture["train_sha256"] = _manufacture_cine_frames(fixture["train"])
        fixture["evaluation_sha256"] = _manufacture_cine_frames(fixture["evaluation"])
        recipes = _ALL_RECIPES if args.all_arms else (args.recipe,)
        run_reports = {}
        shared_v1_tqp = _run_shared_v1_tqp(output_dir, fixture, args) if args.all_arms else None
        for recipe in recipes:
            run_output = output_dir / recipe if args.all_arms else output_dir
            if args.all_arms and recipe in {"v1", "v1u"}:
                shared_artifacts = {
                    stage: shared_v1_tqp["artifacts"][stage]["path"]
                    for stage in ("T", "Q", "P")
                }
                report = _run_recipe(
                    recipe, run_output, fixture, args, stage="J",
                    stage_artifacts=shared_artifacts,
                )
            else:
                report = _run_recipe(recipe, run_output, fixture, args)
            run_reports[recipe] = report
            print(json.dumps({
                "recipe": recipe, "status": report["status"],
                "completed_stages": report["completed_stages"],
                "tokenizer_gate": report["tokenizer_gate"],
            }, sort_keys=True))

    if args.all_arms:
        statuses = {name: report["status"] for name, report in run_reports.items()}
        assert shared_v1_tqp is not None
        expected_manifest_hashes = {
            stage: shared_v1_tqp["artifacts"][stage]["manifest_sha256"]
            for stage in ("T", "Q", "P")
        }
        expected_artifact_hashes = {
            stage: shared_v1_tqp["artifacts"][stage]["stage_artifact_hashes"]
            for stage in ("T", "Q", "P")
        }
        consumers = {}
        for name in ("v1", "v1u"):
            report = run_reports[name]
            result_hashes = (report.get("engine_result") or {}).get("stage_artifact_hashes", {})
            content_hashes = {
                stage: result_hashes.get(stage) if isinstance(result_hashes, dict) else None
                for stage in ("T", "Q", "P")
            }
            consumers[name] = {
                "requested_stage": report["requested_stage"],
                "engine_active_stages": (report.get("engine_result") or {}).get("active_stages"),
                "stage_artifact_inputs": report["stage_artifact_inputs"],
                "stage_artifact_hashes": content_hashes,
            }
        shared_manifest_identity_passed = shared_v1_tqp["all_producer_stages_completed"] and all(
            run_reports[name]["requested_stage"] == "J"
            and consumers[name]["engine_active_stages"] == ["J"]
            and all(expected_manifest_hashes[stage] is not None for stage in ("T", "Q", "P"))
            and all(
                isinstance(expected_artifact_hashes[stage], dict)
                and expected_artifact_hashes[stage].get("manifest_sha256") == expected_manifest_hashes[stage]
                for stage in ("T", "Q", "P")
            )
            and {
                stage: run_reports[name]["stage_artifact_inputs"].get(stage, {}).get("manifest_sha256")
                for stage in ("T", "Q", "P")
            } == expected_manifest_hashes
            and {
                stage: run_reports[name]["stage_artifact_inputs"].get(stage, {}).get("path")
                for stage in ("T", "Q", "P")
            } == {
                stage: shared_v1_tqp["artifacts"][stage]["path"]
                for stage in ("T", "Q", "P")
            }
            for name in ("v1", "v1u")
        )
        shared_content_identity_passed = (
            all(
                isinstance(value, dict)
                and all(isinstance(value.get(field), str) and value[field] for field in (
                    "manifest_sha256", "checkpoint_sha256",
                ))
                for value in expected_artifact_hashes.values()
            )
            and all(
                consumers[name]["stage_artifact_hashes"] == expected_artifact_hashes
                for name in ("v1", "v1u")
            )
        )
        shared_identity_passed = (
            shared_v1_tqp["all_producer_stages_completed"]
            and shared_manifest_identity_passed and shared_content_identity_passed
        )
        omega_validation = _validate_v1_v1u_omega(run_reports)
        status = "COMPLETED_ALL_SYNTHETIC_RECIPES" if all(
            value.startswith("COMPLETED") for value in statuses.values()
        ) and shared_identity_passed and omega_validation["passed"] else "FAILED_SYNTHETIC_RECIPES"
        summary = {
            "schema": "self_audit.candidate_q3d.smoke_all.v1",
            "status": status,
            "evidence_status": "SOFTWARE_ONLY_SYNTHETIC",
            "recipe_statuses": statuses,
            "shared_v1_v1u_stages": {
                **shared_v1_tqp,
                "consumer_stages": {name: "J" for name in ("v1", "v1u")},
                "consumers": consumers,
                "manifest_identity_passed": shared_manifest_identity_passed,
                "content_identity_passed": shared_content_identity_passed,
                "identity_passed": shared_identity_passed,
                "physiology_omega_validation": omega_validation,
            },
            "runs": {name: {
                "report": str(output_dir / f"{name}.smoke.json"),
                "config": report["config_path"],
                "config_sha256": report["config_sha256"],
                "completed_stages": report["completed_stages"],
                "expected_stages": report["expected_stages"],
                "expected_completed_stages": report["expected_completed_stages"],
                "requested_stage": report["requested_stage"],
                "engine_active_stages": (report.get("engine_result") or {}).get("active_stages"),
                "stage_artifact_inputs": report["stage_artifact_inputs"],
                "stage_artifact_hashes": report["stage_artifact_hashes"],
                "tokenizer_gate": report["tokenizer_gate"],
                "synthetic_gate_override_requested": report["synthetic_gate_override_requested"],
                "synthetic_gate_override_used": report["synthetic_gate_override_used"],
                "physiology_cohort_weighting": report["physiology_cohort_weighting"],
                "engine_error": report["engine_error"],
            } for name, report in run_reports.items()},
            "patient_metrics_claimed": False,
            "gpu_evidence": "NOT_RUN",
            "real_data_evidence": "NOT_RUN",
        }
        report_path.write_text(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n")
    else:
        report = run_reports[args.recipe]
        summary = {"status": report["status"], "completed_stages": report["completed_stages"],
                   "tokenizer_gate": report["tokenizer_gate"]}
        report_path = output_dir.with_name(output_dir.name + ".smoke.json")
    print(json.dumps({**summary, "report": str(report_path)}, indent=2))
    return 0 if str(summary["status"]).startswith("COMPLETED") else 1


if __name__ == "__main__":
    raise SystemExit(main())
