#!/usr/bin/env python3
"""Run a short synthetic forward/backward check for one registry arm.

The smoke is deliberately bounded and mask-free: it creates an in-memory
phantom image and labels, builds the configured volume model, checks the
contracted ``B,3,Z,H,W -> B,4,Z,H,W`` path, and performs at most ten tiny
optimization iterations.  It never opens a dataset path or starts the
historical 10k-step recipe.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np

try:  # direct script execution
    from volume_preflight import (bounded_steps, load_config, load_registry, make_receipt,
                                  synthetic_runtime_config, validate_config, validate_runtime_bounds)
except ImportError:  # package-style import from repository root
    from scripts.volume_preflight import (bounded_steps, load_config, load_registry, make_receipt,
                                          synthetic_runtime_config, validate_config, validate_runtime_bounds)

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _builder(config: Mapping[str, Any]):
    try:
        from self_audit.volume.models import build_volume_model
    except Exception as exc:  # pragma: no cover - surfaced in receipt
        raise RuntimeError("self_audit.volume.models.build_volume_model is unavailable") from exc
    return build_volume_model(config, synthetic=True)


def _move_batch(batch: Mapping[str, Any], device: Any) -> dict[str, Any]:
    return {key: (value.to(device) if hasattr(value, "to") else value) for key, value in batch.items()}


def _json_safe(value: Any) -> Any:
    if isinstance(value, float) and not np.isfinite(value):
        return "NOT_AVAILABLE"
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _synchronized_elapsed(start: float, *, device: Any | None = None, torch_module: Any | None = None) -> float:
    """Return elapsed wall time after draining asynchronous device work.

    CUDA launches are cooperative with this guard: a kernel already in flight
    cannot be interrupted, but synchronizing before the read means the receipt
    includes that work instead of reporting a premature CPU timestamp.
    """
    if (device is not None and getattr(device, "type", None) == "cuda"
            and torch_module is not None):
        torch_module.cuda.synchronize(device)
    return float(time.perf_counter() - start)


def _check_deadline(start: float, max_seconds: float, *, stage: str, device: Any | None = None,
                    torch_module: Any | None = None) -> float:
    """Check the cooperative smoke deadline and return the measured elapsed time."""
    elapsed = _synchronized_elapsed(start, device=device, torch_module=torch_module)
    if elapsed > max_seconds:
        raise TimeoutError(
            f"synthetic smoke exceeded {max_seconds:.1f}s during {stage} "
            f"(elapsed {elapsed:.3f}s)"
        )
    return elapsed


_CORRECTION_PROTOCOLS = {"automatic_only", "interactive_extra_supervision"}
_CORRECTION_MODES = {"manual_only", "geodesic_3d", "geodesic_2d", "image_guided"}
_CORRECTION_ROLES = {"TRAIN", "FIXTURE"}


def _declared_correction_budget(protocol: str) -> int:
    """Return the bounded fixture click budget implied by a declared protocol."""
    if protocol == "automatic_only":
        return 0
    if protocol == "interactive_extra_supervision":
        # The synthetic witness uses one explicit fixture click.  This is a
        # declared protocol budget, not an algorithm inferred from a filename.
        return 1
    raise ValueError("correction protocol must be automatic_only or interactive_extra_supervision")


def _load_correction_config(config: Mapping[str, Any], config_id: str,
                            correction_config_path: Path | None) -> dict[str, Any] | None:
    """Validate an explicit correction witness before constructing the model.

    I00/I01 declare a correction protocol in the row config and therefore must
    receive a matching sidecar.  Other rows (for example V04) may opt into an
    additional witness, but its protocol/mode/role/budget still has to be
    explicit and is recorded in coverage.  No path or filename is interpreted
    as an algorithm choice.
    """
    declared_raw = config.get("correction")
    declared = dict(declared_raw) if isinstance(declared_raw, Mapping) else None
    declared_protocol = None if declared is None else declared.get("protocol")
    declared_active = declared_protocol in _CORRECTION_PROTOCOLS
    if correction_config_path is None:
        if declared_active:
            raise ValueError(
                f"{config_id} declares correction.protocol={declared_protocol!r}; "
                "an explicit matching --correction-config is required"
            )
        return None

    path = Path(correction_config_path)
    try:
        raw = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read correction config before model construction: {path}") from exc
    if not isinstance(raw, Mapping):
        raise ValueError("correction config must be an object")
    correction = dict(raw)
    if correction.get("schema") != "self_audit.volume.correction_config.v1":
        raise ValueError("correction config schema mismatch")
    required = ("protocol", "mode", "dataset_role")
    missing = [key for key in required if key not in correction]
    if missing:
        raise ValueError(f"correction config missing declared fields: {', '.join(missing)}")
    protocol = correction["protocol"]
    mode = correction["mode"]
    role = correction["dataset_role"]
    if not isinstance(protocol, str) or protocol not in _CORRECTION_PROTOCOLS:
        raise ValueError("correction config protocol must be automatic_only or interactive_extra_supervision")
    if not isinstance(mode, str) or mode not in _CORRECTION_MODES:
        raise ValueError(f"correction config mode is invalid: {mode!r}")
    if not isinstance(role, str) or role not in _CORRECTION_ROLES:
        raise ValueError(f"correction config dataset_role is invalid: {role!r}")
    if role != "FIXTURE":
        raise ValueError("synthetic smoke correction config must declare dataset_role='FIXTURE'")

    expected_budget = _declared_correction_budget(str(protocol))
    # Accept either spelling for compatibility with correction APIs, but if a
    # sidecar declares a budget it must agree with the protocol's fixture
    # witness.  Missing optional fields use the protocol-derived budget.
    budget_values = []
    for key in ("click_budget", "supervision_budget"):
        if key in correction:
            value = correction[key]
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"correction config {key} must be a nonnegative integer")
            budget_values.append((key, value))
            if value != expected_budget:
                raise ValueError(
                    f"correction config {key}={value} mismatches declared {protocol} budget {expected_budget}"
                )
    if len({value for _, value in budget_values}) > 1:
        raise ValueError("correction config click_budget and supervision_budget disagree")
    correction["supervision_budget"] = expected_budget
    if declared_active:
        for key in ("protocol", "mode", "dataset_role"):
            if correction.get(key) != declared.get(key):
                raise ValueError(
                    f"correction config {key}={correction.get(key)!r} mismatches "
                    f"{config_id} declared {declared.get(key)!r}"
                )
        declared_budget = _declared_correction_budget(str(declared_protocol))
        if expected_budget != declared_budget:
            raise ValueError(f"correction config supervision budget mismatches {config_id} declaration")
    return correction


def _correction_witness(image_zyx: np.ndarray, output_dir: Path, correction: Mapping[str, Any], affine: Any,
                        spacing_xyz_mm: Any, prior_probabilities: np.ndarray | None = None) -> dict[str, Any]:
    from self_audit.volume.correction import PointPrompt, VolumeCorrector

    z, h, w = image_zyx.shape
    protocol = str(correction.get("protocol", "unspecified"))
    if protocol == "automatic_only":
        prompts: list[PointPrompt] = []
    elif protocol == "interactive_extra_supervision":
        prompts = [PointPrompt(coord=(w // 2, h // 2, z // 2), target_class=1,
                               coord_system="native_voxel_xyz", prompt_id="fixture_center")]
    else:
        raise ValueError("correction protocol must explicitly be automatic_only or interactive_extra_supervision")
    frozen_prior = None if prior_probabilities is None else np.asarray(prior_probabilities, dtype=np.float32)
    # VolumeCorrector's strict probability path clips and normalizes its prior;
    # hash that exact frozen baseline so automatic I00 can prove no refinement
    # occurred while still exercising the real correction/export workflow.
    baseline_prior = None
    if frozen_prior is not None:
        baseline_prior = np.clip(frozen_prior, 1e-6, 1.0)
        baseline_prior = baseline_prior / np.sum(baseline_prior, axis=0, keepdims=True)
    result = VolumeCorrector(
        num_classes=int(correction.get("num_classes", 4)),
        propagation_mode=str(correction.get("mode", "image_guided")),
        sigma_mm=float(correction.get("sigma_mm", 6.0)),
        gamma_intensity=float(correction.get("gamma_intensity", 8.0)),
        prompt_weight=float(correction.get("prompt_weight", 0.9)),
    ).correct(image_zyx, prompts, prior_probabilities=baseline_prior,
              affine=np.asarray(affine, dtype=float),
              spacing_xyz_mm=tuple(float(v) for v in spacing_xyz_mm), patient_id="synthetic_smoke")
    labels_path = output_dir / "correction_labels.nii.gz"
    provenance_path = output_dir / "correction_provenance.nii.gz"
    probs_path = output_dir / "correction_probabilities.nii.gz"
    json_path = output_dir / "correction.json"
    result.export_nifti(labels_path, provenance_path, probs_path)
    result.export_json(json_path)
    result_hash = hashlib.sha256(result.probabilities.tobytes()).hexdigest()
    prior_hash = None if baseline_prior is None else hashlib.sha256(baseline_prior.tobytes()).hexdigest()
    return {"protocol": str(correction.get("protocol", "unspecified")),
            "mode": str(correction.get("mode", "unspecified")),
            "dataset_role": str(correction.get("dataset_role", "unspecified")),
            "supervision_budget": int(correction.get("supervision_budget", len(prompts))),
            "prompt_count": len(prompts), "manual_voxels": int(result.summary["manual_voxels"]),
            "propagated_voxels": int(result.summary["propagated_voxels"]),
            "frozen_student_probability_sha256": prior_hash,
            "result_probability_sha256": result_hash,
            "prior_unchanged": bool(baseline_prior is not None and np.allclose(result.probabilities, baseline_prior,
                                                                               rtol=1e-6, atol=1e-6)),
            "labels_export": labels_path.exists(), "provenance_export": provenance_path.exists(),
            "probabilities_export": probs_path.exists(), "json_export": json_path.exists()}


def run_smoke(config_path: Path, *, requested_steps: int | None = None, device_name: str = "cpu", max_seconds: float = 120.0,
              registry_path: Path | None = None, crop_size: int | list[int] | tuple[int, int] = 64,
              batch_size: int | None = None, correction_config_path: Path | None = None) -> dict[str, Any]:
    if not isinstance(max_seconds, (int, float)) or isinstance(max_seconds, bool) or not math.isfinite(float(max_seconds)) or not 0 < float(max_seconds) <= 600:
        raise ValueError("max_seconds must be finite and in (0,600]")
    config = load_config(config_path)
    validate_config(config)
    registry, entries = load_registry(registry_path)
    config_id = config_path.stem
    entry = entries.get(config_id)
    if entry is None:
        raise ValueError(f"config filename stem {config_id!r} is not in the native-volume registry")
    if bool(entry.get("historical_reference", False)):
        # V00 is a provenance witness only.  Never instantiate its old model or
        # reinterpret it through the v1 trainer, and never run 10k steps.
        receipt = make_receipt("smoke", config_path, config,
                               status="HISTORICAL_REFERENCE_ONLY",
                               historical_reference=True,
                               historical_config_path=entry.get("historical_a0_source"),
                               historical_config_sha256=entry.get("historical_a0_sha256"),
                               steps=0, optimizer_steps=0, data_fixture="NOT_RUN",
                               coverage="provenance_only", synthetic=False)
        return receipt
    # Validate any explicit correction sidecar before importing/building the
    # model.  I00/I01 are fail-closed when the required matching witness is
    # omitted; V04 and other rows may opt in explicitly and get coverage
    # fields recording the sidecar's declared protocol.
    correction_config = _load_correction_config(config, config_id, correction_config_path)
    validate_runtime_bounds(config, max_seconds=max_seconds)
    try:
        import torch
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("PyTorch is required for volume smoke") from exc
    if device_name.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("requested CUDA device is unavailable; GPU smoke is NOT_RUN")
    device = torch.device(device_name)
    seed = int(config["training"].get("seed", 0))
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    from self_audit.volume.config import validate_volume_config
    steps = bounded_steps(config, requested_steps)
    requested_microbatch = None if batch_size is None else int(batch_size)
    if requested_microbatch is not None and requested_microbatch < 1:
        raise ValueError("batch_size must be positive")
    runtime_config, runtime_overrides = synthetic_runtime_config(
        config, steps=steps, crop_size=crop_size, microbatch=requested_microbatch
    )
    validate_runtime_bounds(runtime_config, max_seconds=max_seconds)
    runtime_config = validate_volume_config(runtime_config, allow_full_run=False)
    # Force acceptance in the synthetic U fixture only.  The receipt records
    # this explicit override; real TRAIN runs retain the configured threshold.
    method = str(runtime_config["training"]["method"])
    if method in {"ema_full", "ema_full_c01"}:
        runtime_overrides["training.pseudo_threshold"] = {
            "from": runtime_config["training"].get("pseudo_threshold"), "to": 0.0,
            "reason": "bounded synthetic U fixture"
        }
        runtime_overrides["training.pseudo_delay"] = {
            "from": runtime_config["training"].get("pseudo_delay"), "to": 0,
            "reason": "bounded synthetic U fixture"
        }
        runtime_config["training"]["pseudo_threshold"] = 0.0
        runtime_config["training"]["pseudo_delay"] = 0
    start = time.perf_counter()
    built = _builder(runtime_config)
    if not isinstance(built, tuple) or len(built) != 2:
        raise RuntimeError("build_volume_model must return (module, provenance)")
    model, provenance = built
    if not isinstance(model, torch.nn.Module):
        raise RuntimeError("build_volume_model returned a non-module")
    model = model.to(device)
    model.train()
    parameters = [p for p in model.parameters() if p.requires_grad]
    if not parameters:
        raise RuntimeError("configured model has no trainable parameters")
    microbatch = int(runtime_config["training"].get("microbatch", 1))
    accumulation = int(runtime_config["training"].get("accumulation", 1))
    b = microbatch * accumulation
    z = int(runtime_config["geometry"]["slab_depth"])
    crop = runtime_config["geometry"]["crop_size"]
    if isinstance(crop, int):
        h = w = crop
    else:
        h, w = (int(x) for x in crop)
    # Keep a configured shape but bound pathological hand-edited configs.
    if z > 16 or h > 256 or w > 256:
        raise ValueError("smoke shape exceeds bounded synthetic limit (Z<=16,H/W<=256)")
    # All actual transactions below come from the deterministic L/U fixture;
    # no synthetic labels are fabricated in this command.
    from self_audit.volume.data import VolumeTrainingData
    from self_audit.volume.engine import VolumeTrainer
    from self_audit.volume.fixtures import make_volume_fixture
    region_manager = None
    if runtime_config["regions"].get("mode") != "none" and float(runtime_config["regions"].get("weight", 0.0)) > 0:
        from self_audit.volume.regions import RegionManager
        region_manager = RegionManager(runtime_config["regions"])
    output_shape: list[int] | None = None
    loss_value = float("nan")
    loss_metrics: dict[str, Any] = {}
    transaction_records: list[dict[str, Any]] = []
    labeled_geo = labeled_valid = unlabeled_geo = unlabeled_valid = 0
    with tempfile.TemporaryDirectory(prefix="native_volume_smoke_") as fixture_dir:
        manifest, manifest_sha, geometry, geometry_sha = make_volume_fixture(fixture_dir, shape_tzyx=(2, max(5, z), 17, 19))
        data = VolumeTrainingData(manifest, manifest_sha, geometry, geometry_sha, synthetic=True)
        rng = np.random.default_rng(int(runtime_config["training"]["seed"]))
        trainer = VolumeTrainer(model, runtime_config, region_manager=region_manager)
        for step in range(steps):
            _check_deadline(start, max_seconds, stage=f"before transaction {step}",
                            device=device, torch_module=torch)
            lb = data.sample_batch(step % len(data.labeled), rng, b, labeled=True, slab_depth=z,
                                   crop_size=crop, spacing_xyz_mm=runtime_config["geometry"]["spacing_xyz_mm"],
                                   fov_xyz_mm=runtime_config["geometry"]["fov_xyz_mm"], flip=bool(runtime_config["geometry"]["flip"]))
            ub = data.sample_batch(step % len(data.unlabeled), rng, b, labeled=False, slab_depth=z,
                                   crop_size=crop, spacing_xyz_mm=runtime_config["geometry"]["spacing_xyz_mm"],
                                   fov_xyz_mm=runtime_config["geometry"]["fov_xyz_mm"], flip=bool(runtime_config["geometry"]["flip"]))
            if runtime_config["temporal"].get("mode") != "none" and float(runtime_config["temporal"].get("weight", 0.0)) > 0:
                pairs = [data.sample_temporal_pair(step % len(data.unlabeled), rng, labeled=False, slab_depth=z, crop_size=crop,
                                                   spacing_xyz_mm=runtime_config["geometry"]["spacing_xyz_mm"],
                                                   fov_xyz_mm=runtime_config["geometry"]["fov_xyz_mm"], flip=bool(runtime_config["geometry"]["flip"])) for _ in range(b)]
                ub["image"] = torch.stack([pair["image"] for pair in pairs])
                ub["adjacent_image"] = torch.stack([pair["adjacent_image"] for pair in pairs])
                ub["geometric_valid"] = torch.stack([pair["geometric_valid"] for pair in pairs])
                ub["adjacent_geometric_valid"] = torch.stack([pair["adjacent_geometric_valid"] for pair in pairs])
                ub["metadata"] = [pair["metadata"] for pair in pairs]
            lb = _move_batch(lb, device); ub = _move_batch(ub, device)
            labeled_geo += int(lb["geometric_valid"].sum().item())
            labeled_valid += int(lb["label_valid"].sum().item())
            if method not in {"supervised", "supervised_extra"}:
                unlabeled_geo += int(ub["geometric_valid"].sum().item())
            extra_batch = None
            if method == "supervised_extra":
                # C01 is an explicitly named extra supervised update-budget
                # control, not a pseudo-target transaction.  Pass it through
                # the engine's single transaction so EMA remains exactly once.
                extra_lb = data.sample_batch((step + 1) % len(data.labeled), rng, b, labeled=True, slab_depth=z,
                                             crop_size=crop, spacing_xyz_mm=runtime_config["geometry"]["spacing_xyz_mm"],
                                             fov_xyz_mm=runtime_config["geometry"]["fov_xyz_mm"], flip=bool(runtime_config["geometry"]["flip"]))
                extra_batch = _move_batch(extra_lb, device)
            record = trainer.transaction(lb, None if method in {"supervised", "supervised_extra"} else ub,
                                         iteration=step, extra_labeled_batch=extra_batch)
            transaction_records.append(record)
            _check_deadline(start, max_seconds, stage=f"after transaction {step}",
                            device=device, torch_module=torch)
        optimizer_steps = trainer.supervised_updates + trainer.supervised_extra_updates + trainer.pseudo_updates
        if not transaction_records:
            raise RuntimeError("fixture smoke produced no trainer transaction")
        # Verify the model contract on the final L batch without replacing the
        # transaction loss with a model-only witness.
        with torch.no_grad():
            logits = model(lb["image"], valid_mask=lb["geometric_valid"])
            student_probabilities = torch.softmax(logits[0], dim=0).detach().cpu().numpy()
        _check_deadline(start, max_seconds, stage="final functional forward",
                        device=device, torch_module=torch)
        output_shape = list(logits.shape)
        loss_value = float(transaction_records[-1]["supervised_loss"])
        loss_metrics = _json_safe(dict(transaction_records[-1]))
        coverage: dict[str, Any] = {"trainer": True, "fixture_manifest_sha256": manifest_sha,
                                    "fixture_geometry_sha256": geometry_sha,
                                    "labeled_label_reads": sorted(data.label_reads),
                                    "u_records_mask_free": True,
                                    "exposure": {
                                        "labeled_geometric_voxels": labeled_geo,
                                        "labeled_valid_voxels": labeled_valid,
                                        "unlabeled_geometric_voxels": unlabeled_geo,
                                        "unlabeled_valid_voxels": unlabeled_valid,
                                    },
                                    "transaction_records": _json_safe(transaction_records)}
        if region_manager is not None:
            region_records = [r for r in transaction_records if "region_loss" in r]
            if not region_records:
                raise RuntimeError("region-enabled arm did not execute region consistency loss")
            coverage["region_witness"] = {"region_loss": float(region_records[-1]["region_loss"]),
                                           "region_cache_hash": getattr(region_manager, "_cache_hash", "")}
        if runtime_config["temporal"].get("mode") != "none" and float(runtime_config["temporal"].get("weight", 0.0)) > 0:
            temporal_records = [r for r in transaction_records if "temporal_loss" in r]
            if not temporal_records:
                raise RuntimeError("temporal-enabled arm did not execute temporal consistency loss")
            coverage["temporal_witness"] = {"temporal_loss": float(temporal_records[-1]["temporal_loss"]),
                                              "temporal_valid_voxels": float(temporal_records[-1].get("temporal_valid_voxels", 0.0))}
        if correction_config_path is not None:
            _check_deadline(start, max_seconds, stage="before correction witness allocation",
                            device=device, torch_module=torch)
            if correction_config is None:  # pragma: no cover - guarded above
                raise RuntimeError("correction witness sidecar was not validated")
            coverage["correction_witness"] = _correction_witness(
                lb["image"][0, 1].detach().cpu().numpy(), Path(fixture_dir), correction_config,
                lb["metadata"][0]["native_affine_xyz_to_world_mm"],
                lb["metadata"][0]["spacing_xyz_mm"],
                student_probabilities,
            )
            witness = coverage["correction_witness"]
            if not all(witness[key] for key in ("labels_export", "provenance_export", "probabilities_export", "json_export")):
                raise RuntimeError("correction/export witness incomplete")
            declared = config.get("correction")
            declared = dict(declared) if isinstance(declared, Mapping) else {}
            witness["declared_protocol"] = declared.get("protocol")
            witness["declared_mode"] = declared.get("mode")
            witness["declared_dataset_role"] = declared.get("dataset_role")
            witness["declared_supervision_budget"] = (
                _declared_correction_budget(str(declared["protocol"]))
                if declared.get("protocol") in _CORRECTION_PROTOCOLS else None
            )
            _check_deadline(start, max_seconds, stage="after correction witness",
                            device=device, torch_module=torch)
        if method == "supervised_extra":
            coverage["extra_supervised_updates"] = trainer.supervised_extra_updates
    # Verify that an independent teacher copy can be made without creating a
    # second update stream.  It is never used as truth in this synthetic check.
    _check_deadline(start, max_seconds, stage="before teacher witness allocation",
                    device=device, torch_module=torch)
    teacher = copy.deepcopy(model).eval()
    with torch.no_grad():
        teacher_logits = teacher(lb["image"], valid_mask=lb["geometric_valid"])
    teacher_finite = bool(torch.isfinite(teacher_logits).all())
    del teacher
    # This is the final witness check and the sole receipt timestamp: it is
    # intentionally after teacher forward and CUDA synchronization so the
    # reported duration covers every smoke operation.
    elapsed = _check_deadline(start, max_seconds, stage="after teacher witness",
                              device=device, torch_module=torch)
    return make_receipt(
        "smoke",
        config_path,
        config,
        steps=steps,
        optimizer_steps=optimizer_steps,
        optimizer_updates=trainer.optimizer_updates,
        supervised_updates=trainer.supervised_updates,
        supervised_extra_updates=trainer.supervised_extra_updates,
        pseudo_updates=trainer.pseudo_updates,
        ema_updates=trainer.ema_updates,
        elapsed_seconds=elapsed,
        output_shape=output_shape,
        loss=loss_value,
        loss_metrics=loss_metrics,
        model_parameters=sum(p.numel() for p in model.parameters()),
        model_provenance=provenance if isinstance(provenance, Mapping) else str(provenance),
        teacher_copy_finite=teacher_finite,
        device=str(device),
        coverage=coverage,
        runtime_overrides=runtime_overrides,
        logical_batch_size=b,
        microbatch=microbatch,
        accumulation=accumulation,
        pseudo_threshold_override=(0.0 if method in {"ema_full", "ema_full_c01"} else None),
        pseudo_delay_override=(0 if method in {"ema_full", "ema_full_c01"} else None),
        synthetic=True,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--registry", type=Path, default=None)
    parser.add_argument("--steps", type=int, default=None, help="bounded synthetic steps (never exceeds 10)")
    parser.add_argument("--device", default="cpu", help="cpu or an explicitly available cuda device")
    parser.add_argument("--max-seconds", type=float, default=120.0)
    parser.add_argument("--crop-size", type=int, default=64)
    parser.add_argument("--batch", type=int, default=None, help="explicit synthetic logical microbatch override")
    parser.add_argument("--correction-config", type=Path, default=None,
                        help="explicit correction protocol JSON; never inferred from filename")
    parser.add_argument("--output", type=Path, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        receipt = run_smoke(args.config, requested_steps=args.steps, device_name=args.device,
                            max_seconds=args.max_seconds, registry_path=args.registry,
                            crop_size=args.crop_size, batch_size=args.batch,
                            correction_config_path=args.correction_config)
    except Exception as exc:
        receipt = {"schema": "self_audit.volume.tooling_receipt.v1", "command": "smoke", "status": "FAIL", "error": str(exc), "real_train_smoke": "NOT_RUN", "gpu": "NOT_RUN"}
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(receipt, indent=2) + "\n")
        print(json.dumps(receipt, indent=2))
        return 2
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
