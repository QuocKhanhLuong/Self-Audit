"""Fail-closed training runner for the opt-in Astra student.

The runner intentionally has no relationship with the historical
``GuidanceStudent`` trainer.  It consumes a hash-pinned ``FrozenTrainingBank``,
constructs the approved DINOv3-backed student, and writes only new checkpoint
artifacts.  A missing approval, bank, or checkpoint writer is an error; there
is no random-backbone or historical-checkpoint fallback.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from integration.artifacts import canonical_hash, construct_student, sha256
from integration.geometry import require_digest
from layer_upgrade.losses import soft_ce
from torch import Tensor, nn
from torch.nn import functional as F

from .checkpoint import save_checkpoint
from .config import constructor_recipe, source_hashes, validate_recipe
from .data import FrozenTrainingBank
from .data import prepare_targets as prepare_bank_targets


def _json_dump(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def _is_inside(path: Path, directory: Path) -> bool:
    try:
        path.relative_to(directory)
    except ValueError:
        return False
    return True


def _load_approval(
    path: str | Path, expected_sha256: str
) -> tuple[dict[str, Any], Path]:
    expected = require_digest(expected_sha256)
    supplied_path = Path(path).expanduser()
    approval_path = supplied_path.resolve()
    if supplied_path.is_symlink() or not approval_path.is_file():
        raise FileNotFoundError("approved DINOv3 approval JSON is required")
    if sha256(approval_path) != expected:
        raise ValueError("backbone approval digest differs from the locked input")
    try:
        approval = json.loads(approval_path.read_text())
    except json.JSONDecodeError as exc:
        raise ValueError("backbone approval JSON is invalid") from exc
    if not isinstance(approval, dict):
        raise ValueError("backbone approval JSON must be an object")
    required = {
        "schema",
        "approved",
        "model",
        "lineage",
        "repo",
        "weights",
        "weights_sha256",
        "source_commit",
        "review_record",
    }
    missing = sorted(required - set(approval))
    if missing:
        raise ValueError(f"backbone approval missing fields: {', '.join(missing)}")
    if approval.get("approved") is not True:
        raise ValueError("BLOCKED_PRETRAINED: approval is not affirmative")
    if approval.get("model") != "dinov3_convnext_tiny":
        raise ValueError("BLOCKED_PRETRAINED: official DINOv3 ConvNeXt-Tiny required")
    if approval.get("lineage") != "official_self_supervised_dinov3":
        raise ValueError("BLOCKED_PRETRAINED: unsupported backbone lineage")
    repo = Path(approval["repo"]).expanduser().resolve()
    weights = Path(approval["weights"]).expanduser().resolve()
    if not repo.is_dir() or repo.is_symlink() or not (repo / "hubconf.py").is_file():
        raise FileNotFoundError("approved local DINOv3 source checkout is required")
    if weights.is_symlink() or not weights.is_file():
        raise FileNotFoundError("approved local DINOv3 weights are required")
    if sha256(weights) != require_digest(approval["weights_sha256"]):
        raise ValueError("approved DINOv3 weights digest mismatch")
    return {**approval, "repo": str(repo), "weights": str(weights)}, approval_path


def _validate_bank(bank: str | Path, expected_sha256: str) -> FrozenTrainingBank:
    expected = require_digest(expected_sha256)
    if not isinstance(bank, (str, Path)):
        raise TypeError(
            "bank must be an on-disk path; unverified bank objects are not accepted"
        )
    bank = FrozenTrainingBank(bank, expected)
    if not getattr(bank, "sha256", None) == expected:
        raise ValueError("bank manifest differs from the locked input")
    patients = tuple(getattr(bank, "patients", ()))
    if not patients or len(patients) != len(set(patients)):
        raise ValueError("audited TRAIN bank must contain nonempty unique patients")
    manifest = getattr(bank, "manifest", None)
    if not isinstance(manifest, Mapping) or manifest.get("role") != "guidance_bank":
        raise ValueError("audited guidance-bank manifest is required")
    if manifest.get("split") != "train":
        raise ValueError("training runner accepts only an audited TRAIN bank")
    if not callable(getattr(bank, "load_patient", None)) or not callable(
        getattr(bank, "batch", None)
    ):
        raise ValueError("FrozenTrainingBank patient/batch interface is required")
    if not callable(getattr(bank, "reverify", None)):
        raise ValueError("FrozenTrainingBank final reverify interface is required")
    allowed = getattr(bank, "allowed_classes", None)
    if allowed is not None and len(tuple(allowed)) != 4:
        raise ValueError("audited bank must declare the four-class permission policy")
    return bank


def _validate_output_location(
    out: Path, bank: Any, approval: Mapping[str, Any]
) -> None:
    if out.exists():
        raise FileExistsError("training output must be a fresh, nonexistent directory")
    forbidden = [Path(getattr(bank, "path", "")).resolve()]
    forbidden.extend(
        [Path(approval["repo"]).resolve(), Path(approval["weights"]).resolve()]
    )
    if any(
        _is_inside(out, directory)
        for directory in forbidden
        if str(directory) not in {".", ""}
    ):
        raise ValueError(
            "training output cannot be inside bank/source/weights directories"
        )


def _seed_everything(seed: int) -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def _validate_model(
    model: Any, recipe: Mapping[str, Any], approval: Mapping[str, Any]
) -> dict[str, Any]:
    if not isinstance(model, nn.Module):
        raise TypeError("approved constructor must return a torch.nn.Module")
    identity = getattr(model, "integration_identity", None)
    if not isinstance(identity, Mapping):
        raise ValueError("student lacks approved integration identity")
    if identity.get("recipe", {}).get("architecture") != recipe["architecture"]:
        raise ValueError(
            "constructed student architecture does not match training recipe"
        )
    if model.__class__.__name__ == "GuidanceStudent":
        raise ValueError("historical GuidanceStudent cannot run in Astra training")
    backbone = identity.get("backbone", {})
    if (
        not isinstance(backbone, Mapping)
        or backbone.get("model") != "dinov3_convnext_tiny"
    ):
        raise ValueError(
            "constructed student does not carry the approved DINOv3 identity"
        )
    if backbone.get("verified_weights_sha256") != approval["weights_sha256"]:
        raise ValueError("constructed student weights do not match approved weights")
    params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if params <= 0 or trainable <= 0:
        raise ValueError("constructed student has no trainable parameters")
    return {
        "class": model.__class__.__name__,
        "architecture": recipe["architecture"],
        "parameters": params,
        "trainable_parameters": trainable,
        "pretrained_loaded": True,
        "backbone": backbone.get("model"),
        "backbone_weights_sha256": backbone.get("verified_weights_sha256"),
        "source_hashes": dict(source_hashes()),
        "source_hash": canonical_hash(source_hashes()),
    }


def _feature_loss(outputs: Mapping[str, Any], prepared: Mapping[str, Any]) -> Tensor:
    student = outputs.get("features")
    target = prepared.get("feature_target", prepared.get("feature_targets"))
    valid = prepared.get("feature_valid")
    if (
        not isinstance(student, Tensor)
        or not isinstance(target, Tensor)
        or not isinstance(valid, Tensor)
    ):
        raise ValueError(
            "student and full native stride-16 teacher features are required"
        )
    if (
        student.shape != target.shape
        or valid.shape != student.shape[:1] + student.shape[2:]
    ):
        raise ValueError("native stride-16 feature shapes are not aligned")
    if not torch.isfinite(student).all() or not torch.isfinite(target).all():
        raise FloatingPointError("nonfinite feature tensor")
    valid = valid.detach().bool() & (target.square().sum(1) > 1e-12)
    if not valid.any():
        raise ValueError("audited bank supplied no nonzero native stride-16 features")
    cosine = F.cosine_similarity(student, target.detach(), dim=1, eps=1e-8)
    value = (1.0 - cosine)[valid].mean()
    if not torch.isfinite(value):
        raise FloatingPointError("nonfinite feature loss")
    return value


def _semantic_gate(batch: Mapping[str, Any], prepared: Mapping[str, Any]) -> bool:
    raw = batch.get("semantic_active_mask")
    if raw is None:
        active = bool(batch.get("semantic_active", False))
    else:
        if not isinstance(raw, Tensor) or raw.dtype is not torch.bool:
            raise ValueError("historical semantic active mask must be boolean")
        active = bool(raw.any().item())
        if "semantic_active" in batch and bool(batch["semantic_active"]) != active:
            raise ValueError("semantic active gate changed before transport")
    if "semantic_active" in prepared and prepared["semantic_active"] is not None:
        if bool(prepared["semantic_active"]) != active:
            raise ValueError("semantic active gate changed during target transport")
    return active


def _lineage(
    bank: Any, optimizer_steps: int, semantic_updates: int = 0, used_patients=None
) -> dict[str, Any]:
    manifest = bank.manifest
    teacher = dict(manifest.get("teacher", {}))
    if not teacher.get("source"):
        provider = teacher.get("provider")
        commit = teacher.get("source_commit_expected", teacher.get("source_commit"))
        if (
            not isinstance(provider, str)
            or not provider.strip()
            or not isinstance(commit, str)
            or not commit.strip()
        ):
            raise ValueError(
                "teacher lineage requires source or provider@source_commit_expected"
            )
        teacher["source"] = f"{provider}@{commit}"
    splits = manifest.get("locked_splits")
    if not isinstance(splits, Mapping) or set(splits) != {"train", "val", "test"}:
        raise ValueError("locked train/val/test splits are required")
    return {
        "bank_manifest_sha256": bank.sha256,
        "teacher": teacher,
        "locked_splits": {
            name: list(splits[name]) for name in ("train", "val", "test")
        },
        "train_patients": list(bank.patients)
        if used_patients is None
        else sorted(used_patients),
        "planned_train_patients": list(bank.patients),
        "optimizer_steps": int(optimizer_steps),
        "semantic_updates": int(semantic_updates),
        "allowed_classes": [bool(v) for v in bank.allowed_classes],
    }


def _save(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    folder: Path,
    config: Mapping[str, Any],
    lineage: Mapping[str, Any],
    state: Mapping[str, Any],
) -> str:
    result = save_checkpoint(
        model, optimizer, folder, config, dict(lineage), training_state=dict(state)
    )
    if not isinstance(result, str) or len(result) != 64:
        raise ValueError("checkpoint writer must return a manifest SHA256")
    return result


def _indices_from_manifest(bank: Any, patient_id: str) -> list[tuple[int, int]]:
    record = bank.manifest.get("image_records", {}).get(patient_id, {})
    shape = record.get("shape_xyzt")
    if not isinstance(shape, (list, tuple)) or len(shape) != 4:
        raise ValueError(f"{patient_id}: image_records.shape_xyzt is required")
    width, height, slices, frames = (int(v) for v in shape)
    if min(width, height, slices, frames) < 1:
        raise ValueError(f"{patient_id}: image_records shape is empty")
    return [(t, z) for t in range(frames) for z in range(slices)]


def _stable_patient_seed(patient_id: str) -> int:
    """Return a process-independent seed for per-patient shuffling."""
    return int.from_bytes(
        hashlib.sha256(patient_id.encode("utf-8")).digest()[:8], "big"
    )


def run_training(
    *,
    bank: str | Path,
    expected_bank_sha256: str,
    approval: str | Path,
    expected_approval_sha256: str,
    config: Mapping[str, Any] | str | Path,
    out: str | Path,
    device: str | torch.device = "cuda",
) -> dict[str, Any]:
    """Run the approved Astra student on TRAIN contexts only.

    The function performs every immutable-input check before creating ``out``.
    ``max_steps`` is a deliberate pilot cap; ``None`` requests the declared
    epoch budget.  Resume, VAL/TEST evaluation, native-detail supervision,
    guard/refiner steps, and export are outside this runner's scope.
    """
    if isinstance(config, (str, Path)):
        config = json.loads(Path(config).read_text())
    config = validate_recipe(dict(config))
    approval_data, approval_path = _load_approval(approval, expected_approval_sha256)
    bank_obj = _validate_bank(bank, expected_bank_sha256)
    out_path = Path(out).expanduser().resolve()
    _validate_output_location(out_path, bank_obj, approval_data)
    selected_device = torch.device(device)
    if selected_device.type not in {"cpu", "cuda"} or (
        selected_device.type == "cuda" and not torch.cuda.is_available()
    ):
        raise RuntimeError(
            "requested training device is unavailable; no device fallback"
        )
    _lineage(
        bank_obj, 0
    )  # Validate the real teacher metadata before allocating a model/output.
    _seed_everything(config["seed"])
    if config.get("threads"):
        torch.set_num_threads(config["threads"])
    model, provenance = construct_student(
        constructor_recipe(config), approval_data, device=str(selected_device)
    )
    startup = _validate_model(model, config, approval_data)
    if provenance.get("verified_weights_sha256") != approval_data["weights_sha256"]:
        raise ValueError("constructor provenance does not match approval receipt")

    # The first write happens only after all input, model, and device checks.
    out_path.mkdir(parents=True, exist_ok=False)
    (out_path / "checkpoints").mkdir()
    _json_dump(out_path / "CONFIG.json", dict(config))
    startup.update(
        {
            "device": str(selected_device),
            "approval_json": str(approval_path),
            "approval_sha256": require_digest(expected_approval_sha256),
            "bank_manifest_sha256": require_digest(expected_bank_sha256),
            "config_sha256": canonical_hash(config),
            "bank_orientation_contract": getattr(
                bank_obj, "orientation_contract", None
            ),
            "torch_version": str(torch.__version__),
            "cuda_runtime": torch.version.cuda,
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "device_name": torch.cuda.get_device_name(selected_device)
            if selected_device.type == "cuda"
            else "CPU",
            "detail_branch": {
                "native_objective": "none",
                "status": "NOT_TRAINED_NATIVE_DETAIL",
                "reason": "feature plus coarse objectives only",
            },
        }
    )
    _json_dump(out_path / "STARTUP.json", startup)
    print("[ASTRA_STARTUP] " + json.dumps(startup, sort_keys=True), flush=True)

    model.train()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )
    writer = out_path / "metrics.jsonl"
    metric_rows = 0
    optimizer_steps = 0
    semantic_updates = 0
    checkpoints: list[dict[str, Any]] = []
    stop = False
    epoch_completed = 0
    used_patients = set()
    started = time.perf_counter()
    if selected_device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(selected_device)
    batches_per_epoch = sum(
        (len(_indices_from_manifest(bank_obj, pid)) + config["batch_size"] - 1)
        // config["batch_size"]
        for pid in bank_obj.patients
    )

    with writer.open("w", encoding="utf-8") as metrics_file:
        for epoch in range(1, config["epochs"] + 1):
            steps_this_epoch = 0
            patient_order = list(bank_obj.patients)
            random.Random(config["seed"] + epoch).shuffle(patient_order)
            for patient_id in patient_order:
                patient = bank_obj.load_patient(patient_id)
                indices = _indices_from_manifest(bank_obj, patient_id)
                random.Random(
                    config["seed"] * 1009
                    + epoch * 9176
                    + _stable_patient_seed(patient_id)
                ).shuffle(indices)
                for start in range(0, len(indices), config["batch_size"]):
                    if (
                        config["max_steps"] is not None
                        and optimizer_steps >= config["max_steps"]
                    ):
                        stop = True
                        break
                    batch_indices = indices[start : start + config["batch_size"]]
                    batch = bank_obj.batch(
                        patient, batch_indices, device=selected_device
                    )
                    target_builder = getattr(bank_obj, "prepare_targets", None)
                    prepared = (
                        target_builder(batch)
                        if callable(target_builder)
                        else prepare_bank_targets(batch)
                    )
                    outputs = model(batch["images"])
                    feature_loss = _feature_loss(outputs, prepared)
                    active = _semantic_gate(batch, prepared)
                    semantic_loss = None
                    target_entropy = None
                    if active:
                        coarse_target = prepared.get(
                            "coarse_target", prepared.get("target")
                        )
                        coarse_weights = prepared.get(
                            "coarse_weights", prepared.get("weights")
                        )
                        if not isinstance(coarse_target, Tensor) or not isinstance(
                            coarse_weights, Tensor
                        ):
                            raise ValueError(
                                "active semantic batch lacks transported Q/reliability targets"
                            )
                        if coarse_weights.sum() > 0:
                            semantic_loss = soft_ce(
                                outputs["coarse_logits"], coarse_target, coarse_weights
                            )
                            entropy_map = -(
                                coarse_target
                                * coarse_target.clamp_min(
                                    torch.finfo(coarse_target.dtype).tiny
                                ).log()
                            ).sum(1)
                            target_entropy = (
                                entropy_map * coarse_weights
                            ).sum() / coarse_weights.sum().clamp_min(1e-8)
                    loss = feature_loss * float(config["feature_weight"])
                    if semantic_loss is not None:
                        loss = loss + semantic_loss * float(config["semantic_weight"])
                    if not torch.isfinite(loss):
                        raise FloatingPointError("nonfinite Astra training loss")
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    grads = [p.grad for p in model.parameters() if p.grad is not None]
                    if (
                        not grads
                        or not torch.stack(
                            [torch.isfinite(g).all() for g in grads]
                        ).all()
                    ):
                        raise FloatingPointError("nonfinite Astra training gradient")
                    gradient_norm = torch.nn.utils.clip_grad_norm_(
                        model.parameters(),
                        float(config["gradient_clip_norm"]),
                        error_if_nonfinite=True,
                    )
                    grads = [p.grad for p in model.parameters() if p.grad is not None]
                    if not torch.stack([torch.isfinite(g).all() for g in grads]).all():
                        raise FloatingPointError(
                            "nonfinite clipped Astra training gradient"
                        )
                    optimizer.step()
                    optimizer_steps += 1
                    used_patients.add(patient_id)
                    steps_this_epoch += 1
                    semantic_updates += int(semantic_loss is not None)
                    row = {
                        "step": optimizer_steps,
                        "epoch": epoch,
                        "patient_id": patient_id,
                        "batch_size": len(batch_indices),
                        "feature_loss": float(feature_loss.detach()),
                        "semantic_loss": None
                        if semantic_loss is None
                        else float(semantic_loss.detach()),
                        "loss": float(loss.detach()),
                        "semantic_update": semantic_loss is not None,
                        "semantic_updates": semantic_updates,
                        "target_entropy": None
                        if target_entropy is None
                        else float(target_entropy.detach()),
                        "semantic_kl": None
                        if target_entropy is None
                        else float((semantic_loss - target_entropy).detach()),
                        "gradient_norm_before_clip": float(gradient_norm),
                        "elapsed_seconds": time.perf_counter() - started,
                        "peak_allocated_bytes": torch.cuda.max_memory_allocated(
                            selected_device
                        )
                        if selected_device.type == "cuda"
                        else None,
                    }
                    metric_rows += 1
                    print(
                        "[ASTRA_METRIC] " + json.dumps(row, sort_keys=True), flush=True
                    )
                    metrics_file.write(
                        json.dumps(row, sort_keys=True, allow_nan=False) + "\n"
                    )
                    metrics_file.flush()
                    if (
                        config["max_steps"] is not None
                        and optimizer_steps >= config["max_steps"]
                    ):
                        stop = True
                        break
                del patient
                if stop:
                    break
            partial = steps_this_epoch < batches_per_epoch
            epoch_completed = epoch - int(partial)
            lineage = _lineage(
                bank_obj, optimizer_steps, semantic_updates, used_patients
            )
            checkpoint_folder = out_path / "checkpoints" / f"epoch_{epoch:04d}"
            checkpoints.append(
                {
                    "epoch": epoch,
                    "partial": partial,
                    "manifest_sha256": _save(
                        model,
                        optimizer,
                        checkpoint_folder,
                        config,
                        lineage,
                        {
                            "epoch": epoch,
                            "step": optimizer_steps,
                            "semantic_updates": semantic_updates,
                            "partial": partial,
                        },
                    ),
                }
            )
            if stop:
                break

    bank_obj.reverify()
    if sha256(approval_path) != expected_approval_sha256:
        raise ValueError("backbone approval changed during training")
    if source_hashes() != startup["source_hashes"]:
        raise ValueError("training source changed during run")
    lineage = _lineage(bank_obj, optimizer_steps, semantic_updates, used_patients)
    final_sha = _save(
        model,
        optimizer,
        out_path / "final",
        config,
        lineage,
        {
            "epoch": epoch,
            "step": optimizer_steps,
            "semantic_updates": semantic_updates,
            "partial": partial,
        },
    )
    status = "COMPLETED_MAX_STEPS" if stop else "COMPLETED_EPOCH_BUDGET"
    report = {
        "schema": "layer_guided.training_report.v1",
        "status": status,
        "receipt_status": "SUCCESS",
        "config": dict(config),
        "startup": startup,
        "optimizer_steps": optimizer_steps,
        "semantic_updates": semantic_updates,
        "epochs_completed": epoch_completed,
        "epochs_started": epoch,
        "max_steps": config["max_steps"],
        "checkpoints": checkpoints,
        "final_manifest_sha256": final_sha,
        "metrics_rows": metric_rows,
        "elapsed_seconds": time.perf_counter() - started,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(selected_device)
        if selected_device.type == "cuda"
        else None,
        "val_test": "NOT_RUN",
        "native_detail": "NOT_TRAINED_NATIVE_DETAIL",
        "resume": "NOT_IMPLEMENTED",
    }
    _json_dump(out_path / "REPORT.json", report)
    return report


__all__ = ["run_training"]
