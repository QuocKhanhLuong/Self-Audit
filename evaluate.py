#!/usr/bin/env python3
"""Evaluate a frozen student from an explicit Candidate checkpoint schema."""

from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
from typing import Any

from self_audit.model import digest


_EVALUATORS = {
    "self_audit.candidate_a.checkpoint.v2": "self_audit.candidate_a.evaluation",
    "self_audit.candidate_a.student.v2": "self_audit.candidate_a.evaluation",
    "self_audit.candidate_a3d.checkpoint.v1": "self_audit.candidate_a3d.evaluation",
    "self_audit.candidate_a3d.student.v1": "self_audit.candidate_a3d.evaluation",
}
_Q3D_EXPORT_SCHEMA = "q3d.student_export.v1"
_Q3D_EXPORT_MANIFEST_SCHEMA = "q3d.export_manifest.v1"


def _q3d_training_patient_ids(path: str | Path, expected_sha256: str) -> tuple[str, ...]:
    manifest_path = Path(path).resolve()
    if not manifest_path.is_file() or digest(manifest_path) != expected_sha256:
        raise ValueError("Q3D TRAIN manifest is missing or its SHA-256 does not match")
    try:
        payload = json.loads(manifest_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("Q3D TRAIN manifest is not valid JSON") from exc
    if payload.get("schema") != "self_audit.data.v2" or payload.get("role") != "training":
        raise ValueError("Q3D evaluation requires a hash-pinned self_audit.data.v2 TRAIN manifest")
    ids: list[str] = []
    for field in ("labeled", "unlabeled"):
        records = payload.get(field)
        if not isinstance(records, list):
            raise ValueError(f"Q3D TRAIN manifest {field} records are invalid")
        for record in records:
            patient_id = record.get("id") if isinstance(record, dict) else None
            if not isinstance(patient_id, str) or not patient_id:
                raise ValueError("Q3D TRAIN patient IDs must be nonempty strings")
            ids.append(patient_id)
    if not ids or len(ids) != len(set(ids)):
        raise ValueError("Q3D TRAIN manifest patient IDs must be nonempty and unique")
    return tuple(ids)


def _evaluate_q3d_export(checkpoint: str, checkpoint_sha: str, data_manifest: str,
                         data_sha: str, out: str, device: str = "cuda",
                         threshold: float | None = None, *, train_manifest: str | None = None,
                         train_sha: str | None = None, split: str | None = None) -> dict[str, Any]:
    if threshold is not None:
        raise ValueError("Q3D evaluation uses four-class argmax and does not accept --threshold")
    if not train_manifest or not train_sha:
        raise ValueError("Q3D evaluation requires --train-data and --train-data-sha256 for overlap checks")
    artifact_path = Path(checkpoint).resolve()
    if not artifact_path.is_file() or digest(artifact_path) != checkpoint_sha:
        raise ValueError("Q3D student export is missing or its SHA-256 does not match")
    sidecar = artifact_path.with_suffix(artifact_path.suffix + ".manifest.json")
    try:
        export_manifest = json.loads(sidecar.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("Q3D student export is missing its valid sidecar manifest") from exc
    if (export_manifest.get("schema") != _Q3D_EXPORT_MANIFEST_SCHEMA
            or export_manifest.get("artifact") != artifact_path.name
            or export_manifest.get("artifact_sha256") != checkpoint_sha
            or export_manifest.get("artifact_bytes") != artifact_path.stat().st_size):
        raise ValueError("Q3D export sidecar does not identify the requested artifact bytes")

    from self_audit.candidate_q3d.export import load_export
    from self_audit.candidate_q3d.evaluation import evaluate_manifest

    train_ids = _q3d_training_patient_ids(train_manifest, train_sha)
    model, _metadata = load_export(artifact_path, expected_sha256=checkpoint_sha, device=device)
    return evaluate_manifest(
        model, data_manifest, data_sha, out, device=device, split=split,
        train_patient_ids=train_ids,
    )


def _evaluate_configured_baseline(checkpoint: str, checkpoint_sha: str, config_path: str,
                                  data_manifest: str, data_sha: str, out: str,
                                  device: str = "cpu", split: str | None = None) -> dict[str, Any]:
    """Route nested Q3D baseline configs through their owning evaluation APIs."""
    from self_audit.candidate_q3d.config import load_config

    config = load_config(config_path, real_run=False)
    arm = config.get("arm")
    if config.get("method") != "candidate_q3d":
        raise ValueError("--config for a nested baseline checkpoint must declare method=candidate_q3d")
    if arm == "teacher_supervised":
        from self_audit.candidate_q3d.baselines import evaluate_teacher_supervised_baseline
        if not Path(checkpoint).is_file():
            raise ValueError("teacher_supervised evaluation requires its resume checkpoint file")
        if split == "external_test":
            raise ValueError("teacher-supervised evaluation supports only --split dev or test")
        return evaluate_teacher_supervised_baseline(
            checkpoint, checkpoint_sha, config, data_manifest, data_sha, out,
            device=device, split=split or "dev",
        )
    if arm == "a3d_teacher_free":
        from self_audit.candidate_q3d.baselines import evaluate_a3d_teacher_free_baseline
        if not Path(checkpoint).is_dir():
            raise ValueError("a3d_teacher_free evaluation requires a checkpoint or student-export directory")
        if split == "external_test":
            raise ValueError("A3D Q3D evaluation supports only --split dev or test")
        return evaluate_a3d_teacher_free_baseline(
            checkpoint, checkpoint_sha, config, data_manifest, data_sha, out,
            device=device, split=split or "test",
        )
    raise ValueError("--config supports only Q3D arms a3d_teacher_free or teacher_supervised")


def evaluator_for_schema(schema: str) -> Any:
    """Resolve a checkpoint only through its exact declared track schema."""

    if schema == _Q3D_EXPORT_SCHEMA:
        return _evaluate_q3d_export
    if not isinstance(schema, str) or schema not in _EVALUATORS:
        raise ValueError(f"unsupported checkpoint schema: {schema!r}")
    return importlib.import_module(_EVALUATORS[schema]).evaluate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--data-sha256", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--threshold", type=float)
    parser.add_argument("--config", help="Nested Q3D baseline config for A3D or teacher-supervised checkpoint evaluation")
    parser.add_argument("--train-data", help="Q3D-only hash-pinned TRAIN manifest for overlap checks")
    parser.add_argument("--train-data-sha256", help="SHA-256 of the Q3D TRAIN manifest")
    parser.add_argument("--split", choices=["dev", "test", "external_test"],
                        help="Optional expected Q3D manifest role")
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)
    if args.config:
        if args.threshold is not None or args.train_data or args.train_data_sha256:
            parser.error("--threshold and --train-data options cannot be combined with Q3D baseline --config")
        try:
            result = _evaluate_configured_baseline(
                args.checkpoint, args.checkpoint_sha256, args.config,
                args.data, args.data_sha256, args.out, args.device, args.split,
            )
        except (OSError, TypeError, ValueError) as exc:
            parser.error(str(exc))
        print(json.dumps(result, indent=2))
        return

    if checkpoint_path.is_file():
        try:
            evaluate = evaluator_for_schema(_Q3D_EXPORT_SCHEMA)
            result = evaluate(
                args.checkpoint, args.checkpoint_sha256, args.data, args.data_sha256,
                args.out, args.device, args.threshold, train_manifest=args.train_data,
                train_sha=args.train_data_sha256, split=args.split,
            )
        except (OSError, TypeError, ValueError) as exc:
            parser.error(str(exc))
        print(json.dumps(result["patient_macro"], indent=2))
        return

    if args.train_data or args.train_data_sha256 or args.split:
        parser.error("--train-data and --split are Q3D-only options")
    manifest = Path(args.checkpoint) / "MANIFEST.json"
    if not manifest.is_file():
        parser.error("checkpoint has no MANIFEST.json")
    if digest(manifest) != args.checkpoint_sha256:
        parser.error("checkpoint manifest hash mismatch")
    try:
        manifest_payload = json.loads(manifest.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        parser.error(f"checkpoint manifest is not valid JSON: {exc}")
    try:
        evaluate = evaluator_for_schema(manifest_payload.get("schema"))
    except ValueError as exc:
        parser.error(str(exc))
    if manifest_payload.get("schema", "").startswith("self_audit.candidate_q3d.") and args.threshold is not None:
        parser.error("Q3D evaluation uses four-class argmax and does not accept --threshold")
    result = evaluate(
        args.checkpoint,
        args.checkpoint_sha256,
        args.data,
        args.data_sha256,
        args.out,
        args.device,
        args.threshold,
    )
    print(json.dumps(result["macro_patient"], indent=2))


if __name__ == "__main__":
    main()
