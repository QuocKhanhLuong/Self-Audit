#!/usr/bin/env python3
"""Canonical trainer dispatcher for Candidate A, A-3D, and Q3D."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import MethodType
from typing import Any


def _training_api(method: str):
    """Return the isolated trainer and validator declared for one method."""

    if method == "candidate_a":
        from self_audit.candidate_a.engine import train
        from self_audit.candidate_a.config import validate_config
        return train, validate_config
    if method == "candidate_a3d":
        from self_audit.candidate_a3d.engine import train
        from self_audit.candidate_a3d.config import validate_config
        return train, validate_config
    if method == "candidate_q3d":
        from self_audit.candidate_q3d.engine import train
        from self_audit.candidate_q3d.config import validate_config
        return train, validate_config
    raise ValueError("config.method must be candidate_a, candidate_a3d, or candidate_q3d")


def _manufactured_teacher(config: dict[str, Any]):
    """Build a reproducible, explicitly synthetic teacher fixture for software smoke."""
    import torch

    from self_audit.candidate_q3d.engine import _build_teacher

    teacher, _provenance = _build_teacher(config, synthetic=True)
    original_forward = teacher.forward

    def forward(self, image, geometry=None, *, return_features=False):
        output = dict(original_forward(
            image, geometry=geometry, return_features=return_features,
        ))
        batch, _channels, depth, height, width = image.shape
        x = torch.arange(width, device=image.device).view(1, 1, 1, width)
        frame_mean = image.mean(dim=(1, 2, 3, 4))
        low_frame = frame_mean <= 0.5
        low = torch.where(x < 4, 0, torch.where(x < 10, 1, torch.where(x < 12, 2, 3)))
        high = torch.where(x < 12, 0, torch.where(x < 14, 1, torch.where(x < 16, 2, 3)))
        classes = torch.where(low_frame.view(batch, 1, 1, 1), low, high)
        classes = classes.expand(batch, depth, height, width)
        logits = image.new_full((batch, 4, depth, height, width), -10.0)
        logits.scatter_(1, classes.unsqueeze(1), 10.0)
        # The smoke fixture encodes a small patient-specific intensity offset.
        # It varies predicted RV stroke volume without changing the locked
        # physiology thresholds, making V1/V1-u's relative weighting observable.
        mixture = (0.05 + 50.0 * (frame_mean - 0.8)).clamp(0.0, 0.2)
        high_rv = (~low_frame).view(batch, 1, 1, 1) & (classes == 1)
        logits[:, 1] = torch.where(
            high_rv, (1.0 - mixture).log().view(batch, 1, 1, 1), logits[:, 1],
        )
        logits[:, 0] = torch.where(
            high_rv, mixture.clamp_min(1e-8).log().view(batch, 1, 1, 1), logits[:, 0],
        )
        output["region_logits"] = logits
        return output

    teacher.forward = MethodType(forward, teacher)
    return teacher


def _train_q3d(args: argparse.Namespace, parser: argparse.ArgumentParser,
               config: dict[str, Any], *, synthetic_teacher_fixture: bool = False):
    from self_audit.candidate_q3d.config import (
        validate_config,
        validate_real_teacher_provenance,
    )

    if any((args.model_config, args.teacher_checkpoint, args.teacher_checkpoint_sha256,
            args.data, args.data_sha256, args.validation_data, args.validation_data_sha256)):
        parser.error("Q3D manifests and teacher provenance are declared in the nested config")
    if not args.out:
        parser.error("Q3D requires --out")
    try:
        runtime = config.get("runtime")
        synthetic_requested = isinstance(runtime, dict) and runtime.get("synthetic") is True
        validated = validate_config(config, real_run=not synthetic_requested)
    except (OSError, TypeError, ValueError) as exc:
        parser.error(f"invalid Q3D config: {exc}")
    configured_synthetic = validated["runtime"]["synthetic"]
    if bool(args.synthetic) != configured_synthetic:
        parser.error("Q3D --synthetic must match the explicit runtime.synthetic config value")
    if type(synthetic_teacher_fixture) is not bool:
        parser.error("synthetic teacher fixture selector must be a bool")
    if synthetic_teacher_fixture and not configured_synthetic:
        parser.error("--synthetic-teacher-fixture requires runtime.synthetic=true")
    if not configured_synthetic:
        if not args.allow_full_run:
            parser.error("real Q3D execution requires --allow-full-run")
        try:
            validate_real_teacher_provenance(validated)
        except (TypeError, ValueError, FileNotFoundError) as exc:
            parser.error(f"real Q3D teacher provenance is incomplete: {exc}")

    baseline_runners = {
        "a3d_teacher_free": "run_a3d_teacher_free",
        "teacher_supervised": "run_teacher_supervised",
    }
    if validated["arm"] in baseline_runners:
        if synthetic_teacher_fixture:
            parser.error("--synthetic-teacher-fixture applies only to Q3D student arms")
        try:
            from self_audit.candidate_q3d import baselines
        except ImportError as exc:
            parser.error(f"Q3D baseline runners are unavailable: {exc}")
        runner = getattr(baselines, baseline_runners[validated["arm"]], None)
        if not callable(runner):
            parser.error(f"Q3D baseline runner is missing for arm {validated['arm']!r}")
        return runner(
            validated, args.out, device=args.device, smoke=bool(args.synthetic),
        )

    from self_audit.candidate_q3d.engine import train

    teacher = _manufactured_teacher(validated) if synthetic_teacher_fixture else None
    artifact_inputs: dict[str, Any] = {}
    for binding in getattr(args, "q3d_stage_artifact", None) or []:
        key, separator, value = binding.partition("=")
        if separator != "=" or key not in {"T", "Q", "P"} or not value or key in artifact_inputs:
            parser.error("--q3d-stage-artifact requires unique T=path, Q=path, or P=path bindings")
        artifact_inputs[key] = value
    pinned_stages: set[str] = set()
    for binding in getattr(args, "q3d_stage_artifact_sha", None) or []:
        key, separator, value = binding.partition("=")
        if (separator != "=" or key not in artifact_inputs or key in pinned_stages
                or len(value) != 64 or any(char not in "0123456789abcdef" for char in value)):
            parser.error("--q3d-stage-artifact-sha requires a unique STAGE=SHA256 matching a manifest binding")
        artifact_inputs[key] = {"manifest_path": artifact_inputs[key], "manifest_sha256": value}
        pinned_stages.add(key)

    return train(
        validated,
        args.out,
        device=args.device,
        teacher=teacher,
        synthetic_gate_override=validated["runtime"]["synthetic_gate_override"],
        stage=getattr(args, "q3d_stage", None),
        stage_artifacts=artifact_inputs or None,
    )


def _train_baseline(args: argparse.Namespace, parser: argparse.ArgumentParser,
                    config: dict[str, Any], method: str):
    if not all((args.data, args.data_sha256, args.out)):
        parser.error("baseline tracks require --data, --data-sha256 and --out")
    if bool(args.validation_data) != bool(args.validation_data_sha256):
        parser.error("validation requires both manifest and SHA256")
    train, validate_config = _training_api(method)
    if args.model_config:
        config["model"] = json.loads(Path(args.model_config).read_text())
    if args.teacher_checkpoint or args.teacher_checkpoint_sha256:
        if not (args.teacher_checkpoint and args.teacher_checkpoint_sha256):
            parser.error("teacher initialization requires both path and manifest SHA256")
        config["teacher_init"] = {
            "kind": args.teacher_checkpoint_kind or method,
            "path": args.teacher_checkpoint,
            "sha256": args.teacher_checkpoint_sha256,
        }
    validated = validate_config(config)
    from self_audit.model import digest

    if digest(args.data) != args.data_sha256:
        parser.error("training manifest hash mismatch")
    data = json.loads(Path(args.data).read_text())
    geometry = data.get("geometry", {}).get("by_patient", {})
    updates = validated["epochs"] * sum(
        len(record["masks"]) * (
            1 if method == "candidate_a3d"
            else geometry[record["id"]]["shape_zyx"][0]
        )
        for record in data["labeled"]
    )
    if updates > 128 and not args.allow_full_run:
        parser.error("more than 128 updates requires --allow-full-run")
    return train(
        validated,
        args.data,
        args.data_sha256,
        args.out,
        device=args.device,
        synthetic=args.synthetic,
        validation_manifest=args.validation_data,
        validation_sha256=args.validation_data_sha256,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--describe", action="store_true")
    parser.add_argument("--config", default="configs/candidate_q3d/v1.json")
    parser.add_argument("--q3d-stage", choices=("T", "Q", "P", "J"))
    parser.add_argument("--q3d-stage-artifact", action="append", metavar="STAGE=MANIFEST")
    parser.add_argument("--q3d-stage-artifact-sha", action="append", metavar="STAGE=SHA256")
    parser.add_argument("--model-config", help="Baseline-only local backbone source and weight provenance JSON")
    parser.add_argument("--teacher-checkpoint", help="Baseline-only verified initialization folder")
    parser.add_argument("--teacher-checkpoint-sha256", help="Baseline-only MANIFEST.json SHA256")
    parser.add_argument("--teacher-checkpoint-kind", choices=["candidate_a", "candidate_a3d", "native_v2"])
    parser.add_argument("--data", help="Baseline-only frozen TRAIN manifest")
    parser.add_argument("--data-sha256")
    parser.add_argument("--out")
    parser.add_argument("--validation-data", help="Baseline-only disjoint development manifest")
    parser.add_argument("--validation-data-sha256")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--allow-full-run", action="store_true")
    parser.add_argument("--synthetic", action="store_true",
                        help="Manufactured CPU check; must match Q3D runtime.synthetic")
    parser.add_argument("--synthetic-teacher-fixture", action="store_true",
                        help="Inject a fixed teacher only for a synthetic Q3D software run")
    args = parser.parse_args()
    if args.describe:
        active = Path(__file__).with_name("ACTIVE_PIPELINE.json")
        print(active.read_text())
        return
    try:
        config = json.loads(Path(args.config).read_text())
    except (OSError, json.JSONDecodeError) as exc:
        parser.error(f"cannot read config {args.config!r}: {exc}")
    if not isinstance(config, dict):
        parser.error("config root must be an object")
    method = config.get("method")
    if method == "candidate_q3d":
        runtime = config.get("runtime")
        synthetic_requested = isinstance(runtime, dict) and runtime.get("synthetic") is True
        try:
            from self_audit.candidate_q3d.config import load_config
            config = load_config(args.config, real_run=not synthetic_requested)
        except (OSError, TypeError, ValueError) as exc:
            parser.error(f"invalid Q3D config: {exc}")
        result = _train_q3d(
            args, parser, config,
            synthetic_teacher_fixture=bool(args.synthetic_teacher_fixture),
        )
    elif method in {"candidate_a", "candidate_a3d"}:
        result = _train_baseline(args, parser, config, method)
    else:
        parser.error("config.method must be candidate_a, candidate_a3d, or candidate_q3d")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
