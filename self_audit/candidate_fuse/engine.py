"""Training engine for the locked native-volume Candidate FUSE recipe."""
from __future__ import annotations

import copy
import gc
import hashlib
import json
import math
import os
import random
import tempfile
import time
from contextlib import nullcontext
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from torch import Tensor

from self_audit.losses import supervised_loss
from self_audit.candidate_q3d.augmentation import apply_spatial_transform, sample_spatial_transform
from .config import SEMI_SCHEMA, TRANSFER_SCHEMA, config_sha256, validate_config
from .evaluation import choose_deployment, evaluate_modes
from .export import export_model
from .student import FuseStudent, sample_keep_mask
from .teacher import (FuseTeacher, build_teacher, configure_teacher_step,
                      teacher_loss, teacher_parameter_groups)


def _soft_ce(logits: Tensor, teacher_logits: Tensor) -> Tensor:
    if logits.shape != teacher_logits.shape or logits.ndim != 5 or logits.shape[1] != 4:
        raise ValueError("soft pseudo CE requires matching [B,4,D,H,W] logits")
    target = torch.softmax(teacher_logits.detach().float(), dim=1)
    return -(target * torch.log_softmax(logits.float(), dim=1)).sum(1).mean()


def _unwrap_teacher(value: Any) -> tuple[Mapping[str, Tensor], tuple[Tensor, ...]]:
    if isinstance(value, tuple) and len(value) == 2:
        output, features = value
    elif isinstance(value, Mapping):
        output, features = value, value.get("features")
    else:
        raise TypeError("teacher forward must return output mapping and features")
    if not isinstance(output, Mapping) or not isinstance(output.get("region_logits"), Tensor):
        raise TypeError("teacher output requires region_logits")
    if not isinstance(features, (tuple, list)) or len(features) != 5:
        raise ValueError("teacher output requires five feature maps")
    return output, tuple(features)


def _frame_tensor(frame: Any, device: torch.device) -> tuple[Tensor, Tensor, Mapping[str, Any]]:
    image = torch.as_tensor(frame.image, dtype=torch.float32, device=device)
    target = getattr(frame, "target", None)
    if target is None:
        raise ValueError("labeled frame has no target")
    target = torch.as_tensor(target, device=device)
    if target.ndim == 3:
        target = target.unsqueeze(0)
    if target.ndim == 5 and target.shape[1] == 1:
        target = target[:, 0]
    if image.ndim == 4:
        image = image.unsqueeze(0)
    if image.ndim != 5 or tuple(image.shape[:2]) != (1, 1) or target.shape != image.shape[:1] + image.shape[-3:]:
        raise ValueError("frame tensors must be image [1,1,D,H,W] and target [1,D,H,W]")
    if target.dtype.is_floating_point or bool(((target < 0) | (target > 3)).any()):
        raise ValueError("FUSE training target must contain integer classes 0 through 3")
    return image, target.long(), frame.geometry


def _frame_image(frame: Any, device: torch.device) -> tuple[Tensor, Mapping[str, Any]]:
    image = torch.as_tensor(frame.image, dtype=torch.float32, device=device)
    if image.ndim == 4:
        image = image.unsqueeze(0)
    if image.ndim != 5 or tuple(image.shape[:2]) != (1, 1):
        raise ValueError("unlabeled frame image must be [1,1,D,H,W]")
    return image, frame.geometry


def _transform(image: Tensor, target: Tensor | None, geometry: Mapping[str, Any], rng: random.Random):
    # Runtime geometry primitives are strict about their four-key schema;
    # manifests may retain provenance alongside those keys. Keep that metadata
    # on the returned geometry while letting the canonical helper validate the
    # spatial fields it actually transforms.
    runtime = {key: geometry[key] for key in ("shape", "spacing_zyx", "affine_zyx", "context")}
    transform = sample_spatial_transform(runtime, rng=rng)
    image, target, transformed = apply_spatial_transform(image, target, runtime, **transform)
    transformed.update({key: value for key, value in geometry.items() if key not in transformed})
    return image, target, transformed


def _teacher_batch_loss(outputs: Mapping[str, Any], target: Tensor,
                        geometry: Mapping[str, Any]) -> Tensor:
    """Mean per-frame objective; each frame retains its own Dice reduction."""
    batch = int(target.shape[0])
    if batch == 1:
        return teacher_loss(outputs, target, geometry)
    losses=[]
    for bi in range(batch):
        sliced={name:(value[bi:bi+1] if isinstance(value,Tensor) and value.ndim>0 and value.shape[0]==batch else value)
                for name,value in outputs.items()}
        losses.append(teacher_loss(sliced,target[bi:bi+1],geometry))
    return torch.stack(losses).mean()


def _autocast(device: torch.device, config: Mapping[str, Any]):
    if device.type == "cuda" and config["runtime"]["amp_dtype"] == "bfloat16":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def _lambda_u(update: int, total_updates: int, fraction: float) -> float:
    ramp = max(1, int(math.ceil(fraction * total_updates)))
    return min(1.0, update / ramp)


def joint_loss(student: FuseStudent, teacher: FuseTeacher, labeled: Any, unlabeled: Any,
               *, update: int, total_updates: int, config: Mapping[str, Any],
               generator: torch.Generator | None = None) -> tuple[Tensor, dict[str, Any]]:
    """Exact §5 paired objective; teacher remains a detached frozen target."""
    if total_updates <= 0 or not 0 <= update < total_updates:
        raise ValueError("joint update must be inside a positive update budget")
    if any(p.requires_grad for p in teacher.parameters()) or teacher.training:
        raise RuntimeError("teacher must be frozen and eval before joint_loss")
    device = next(student.parameters()).device
    image_l, target_l, geom_l = _frame_tensor(labeled, device)
    image_u, geom_u = _frame_image(unlabeled, device)
    # L and U belong to different patients and can have different native
    # shapes and spacings. Their transforms are therefore sampled separately;
    # each frame still has one shared transformed input for its T and S paths.
    runtime_l = {key: geom_l[key] for key in ("shape", "spacing_zyx", "affine_zyx", "context")}
    runtime_u = {key: geom_u[key] for key in ("shape", "spacing_zyx", "affine_zyx", "context")}
    transform_l = sample_spatial_transform(runtime_l, rng=_joint_py_rng)
    transform_u = sample_spatial_transform(runtime_u, rng=_joint_py_rng)
    image_l, target_l, geom_l = apply_spatial_transform(image_l, target_l, runtime_l, **transform_l)
    image_u, _, geom_u = apply_spatial_transform(image_u, None, runtime_u, **transform_u)
    geom_l.update({key: value for key, value in labeled.geometry.items() if key not in geom_l})
    geom_u.update({key: value for key, value in unlabeled.geometry.items() if key not in geom_u})
    keep_l = sample_keep_mask(1, update, total_updates, device=device, generator=generator)
    keep_u = sample_keep_mask(1, update, total_updates, device=device, generator=generator)
    with torch.no_grad():
        teacher_l, features_l = _unwrap_teacher(teacher(image_l, geom_l, return_features=True))
        teacher_u, features_u = _unwrap_teacher(teacher(image_u, geom_u, return_features=True))
        pseudo_u = teacher_u["region_logits"].detach()
    out_l = student(image_l, geom_l, teacher_features=features_l, keep_mask=keep_l)
    out_u = student(image_u, geom_u, teacher_features=features_u, keep_mask=keep_u)
    supervised = supervised_loss(out_l["logits"], target_l)
    unsupervised = _soft_ce(out_u["logits"], pseudo_u)
    mimic_l, mimic_u = out_l["mimic_loss"], out_u["mimic_loss"]
    ramp = _lambda_u(update, total_updates, float(config["training"]["unlabeled_ramp_fraction"]))
    loss = supervised + ramp * unsupervised + float(config["training"]["mimic_weight"]) * (mimic_l + mimic_u)
    if not torch.isfinite(loss):
        raise FloatingPointError("nonfinite FUSE joint loss")
    stats = {"loss": float(loss.detach().float()), "supervised": float(supervised.detach().float()),
             "soft_ce": float(unsupervised.detach().float()), "lambda_u": ramp,
             "mimic_l": float(mimic_l.detach().float()), "mimic_u": float(mimic_u.detach().float()),
             "keep_fraction_l": float(keep_l.detach().mean()),
             "keep_fraction_u": float(keep_u.detach().mean()),
             "gate_means_l": out_l["gate_means"], "gate_means_u": out_u["gate_means"]}
    return loss, stats


_joint_py_rng = random.Random(42)


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_torch_save(value: Any, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(fd)
    temporary = Path(name)
    try:
        torch.save(value, temporary)
        with temporary.open("rb") as stream:
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return _sha(path)


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n")


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, sort_keys=True, indent=2, allow_nan=False).encode() + b"\n"
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(fd); temporary = Path(name)
    try:
        with temporary.open("wb") as stream:
            stream.write(payload); stream.flush(); os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try: os.fsync(directory_fd)
        finally: os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)


def _state(path: Path, *, status: str, stage: str, epoch: int, update: int,
           checkpoint: Path | None = None, checkpoint_sha: str | None = None) -> None:
    _write_json(path / "STATE.json", {"schema": "self_audit.candidate_fuse.state.v1", "status": status,
        "stage": stage, "epoch": epoch, "update": update,
        "updated_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint_path": str(checkpoint) if checkpoint else None,
        "checkpoint_sha256": checkpoint_sha})


def _metric(path: Path, row: Mapping[str, Any]) -> None:
    with (path / "metrics.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(dict(row), sort_keys=True, allow_nan=False) + "\n")
        stream.flush(); os.fsync(stream.fileno())


def _metrics_receipt(out: Path) -> dict[str, Any]:
    data = (out / "metrics.jsonl").read_bytes()
    return {"offset": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def _checkpoint_due(update: int, total_updates: int, *, synthetic: bool) -> bool:
    if update < 1 or total_updates < update:
        raise ValueError("checkpoint update must be within the committed update budget")
    return bool(synthetic or update % 20 == 0 or update == total_updates)


def _transfer_teacher_budget(epoch_size: int, epochs: int, *, batch_size: int = 2,
                             max_optimizer_steps: int | None = None) -> dict[str, int]:
    """Return Track B's paired optimizer-step and labeled-exposure budgets."""
    if any(type(value) is not int or value < 1 for value in (epoch_size, epochs, batch_size)):
        raise ValueError("Track B T budget dimensions must be positive integers")
    steps = math.ceil(epoch_size / batch_size) * epochs
    exposures = epoch_size * epochs
    if max_optimizer_steps is not None:
        if type(max_optimizer_steps) is not int or max_optimizer_steps < 1:
            raise ValueError("Track B max_updates must be a positive optimizer-step count")
        steps = min(steps, max_optimizer_steps)
        exposures = min(exposures, steps * batch_size)
    return {"optimizer_steps": steps, "examples_seen": exposures}


def _recover_metrics(out: Path, receipt: Mapping[str, Any]) -> str | None:
    """Restore the checkpoint's committed JSONL prefix and archive its suffix."""
    path = out / "metrics.jsonl"
    data = path.read_bytes()
    offset, expected = receipt.get("offset"), receipt.get("sha256")
    if type(offset) is not int or offset < 0 or len(data) < offset:
        raise ValueError("resume metrics log is shorter than its committed checkpoint offset")
    prefix = data[:offset]
    if hashlib.sha256(prefix).hexdigest() != expected:
        raise ValueError("resume metrics committed prefix hash mismatch")
    suffix = data[offset:]
    archived = None
    if suffix:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
        archive = out / f"metrics.uncommitted.{stamp}.jsonl"
        archive.write_bytes(suffix)
        with archive.open("rb") as stream: os.fsync(stream.fileno())
        archived = str(archive)
    with path.open("wb") as stream:
        stream.write(prefix); stream.flush(); os.fsync(stream.fileno())
    return archived


def _rng_state(generator: torch.Generator, py_rng: random.Random) -> dict[str, Any]:
    return {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            "generator": generator.get_state(), "augmentation": py_rng.getstate()}


def _restore_rng(state: Mapping[str, Any], generator: torch.Generator, py_rng: random.Random) -> None:
    # Resume payloads are loaded with map_location=target_device. RNG state
    # byte tensors remain CPU-owned even when model/optimizer tensors map to CUDA.
    random.setstate(state["python"]); np.random.set_state(state["numpy"])
    torch.set_rng_state(torch.as_tensor(state["torch"], dtype=torch.uint8, device="cpu"))
    cuda_states=state.get("cuda")
    if cuda_states is not None:
        torch.cuda.set_rng_state_all([torch.as_tensor(value,dtype=torch.uint8,device="cpu") for value in cuda_states])
    generator.set_state(torch.as_tensor(state["generator"],dtype=torch.uint8,device="cpu"))
    py_rng.setstate(state["augmentation"])


def _data(config: Mapping[str, Any], role: str, synthetic: bool):
    from .data import open_fuse_data
    name = "train" if role == "train" else "val" if role == "development" else "test"
    return open_fuse_data(config["data"][f"{name}_manifest"], config["data"][f"{name}_manifest_sha256"],
                          role=role, synthetic=synthetic)


def _identity(config: Mapping[str, Any], train: Any, val: Any) -> dict[str, Any]:
    return {"config_sha256": config_sha256(config), "train_manifest_sha256": train.manifest_sha256,
            "val_manifest_sha256": val.manifest_sha256}


def _validate_semi_track_a(config: Mapping[str, Any], train: Any, val: Any) -> None:
    """Fail closed on Track A identity, L8 masks, and disjoint evaluation data."""
    if config["schema"] != SEMI_SCHEMA:
        return
    manifest = train.manifest
    expected_dataset = "synthetic" if config["runtime"]["synthetic"] else "acdc"
    if manifest.get("schema") != "self_audit.candidate_fuse.data.v2" or manifest.get("dataset") != expected_dataset:
        raise ValueError("FUSE experiment A requires matching data.v2 TRAIN manifest")
    # FuseData's data.v2 loader verifies L/source IDs, compact-list SHA, exact
    # source-manifest bytes, and each source image/mask identity before this
    # orchestration gate is reached.
    contract = manifest.get("labeled_contract")
    ids = contract["patient_ids"]
    if len(ids) != 8 or not set(ids).issubset(train.patients):
        raise ValueError("ACDC L8 must be a subset of the U TRAIN patient inventory")
    if set(train.patients) & set(val.patients):
        raise ValueError("ACDC TRAIN and validation patients overlap")
    if {str(record["id"]) for record in manifest.get("labeled", [])} != set(ids):
        raise ValueError("ACDC TRAIN masks may be declared only for L8")
    labeled_keys = train.labeled_keys()
    if len(labeled_keys) != 16 or {pid for pid, _ in labeled_keys} != set(ids):
        raise ValueError("ACDC L8 must expose exactly 16 labeled ED/ES frames")


def _validate_transfer_track_b(config: Mapping[str, Any], train: Any, val: Any) -> None:
    if config["schema"] != TRANSFER_SCHEMA:
        return
    if config["runtime"]["synthetic"]:
        if train.manifest.get("synthetic") is not True or val.manifest.get("synthetic") is not True:
            raise ValueError("synthetic Track B requires synthetic TRAIN and validation fixtures")
        return
    if (train.manifest.get("schema") != "self_audit.candidate_fuse.data.v3"
            or val.manifest.get("schema") != "self_audit.candidate_fuse.data.v3"
            or train.manifest.get("dataset_release") != "original_2020_open345"
            or val.manifest.get("dataset_release") != "original_2020_open345"
            or len(train.patients) != 175 or len(val.patients) != 34
            or len(train.labeled_keys()) != 16
            or set(train.patients) & set(val.patients)):
        raise ValueError("real Track B requires the locked M&Ms TRAIN175/VAL34 manifests and target L8")


def _attach_gate_means(metrics: dict[str, Any], teacher: FuseTeacher, student: FuseStudent,
                       data: Any, device: torch.device) -> None:
    """Record actual validation G means without affecting selection or RNG."""
    flags = teacher.training, student.training
    sums = np.zeros(5, dtype=np.float64); count = 0
    try:
        teacher.eval(); student.eval()
        with torch.inference_mode():
            for key in data.labeled_keys():
                frame = data.load(key, labeled=True)
                image = torch.as_tensor(frame.image, dtype=torch.float32, device=device)
                if image.ndim == 4: image = image.unsqueeze(0)
                output, features = _unwrap_teacher(teacher(image, frame.geometry, return_features=True))
                del output
                result = student(image, frame.geometry, teacher_features=features,
                                 keep_mask=torch.ones((1,1,1,1,1), device=device))
                sums += np.asarray(result["gate_means"], dtype=np.float64); count += 1
    finally:
        teacher.train(flags[0]); student.train(flags[1])
    means = (sums / count).tolist() if count else [None] * 5
    for mode, record in metrics.items():
        record["gate_means"] = means if mode == "fusion" else [0.0] * 5


def _save_resume(out: Path, payload: dict[str, Any], stage: str, epoch: int, update: int) -> str:
    path = out / "resume.pt"
    payload["metrics_receipt"] = _metrics_receipt(out)
    digest = _atomic_torch_save(payload, path)
    _state(out, status="RUNNING", stage=stage, epoch=epoch, update=update,
           checkpoint=path, checkpoint_sha=digest)
    _write_json(out / "resume.sha256.json", {"path": path.name, "sha256": digest})
    return digest


def _clear_and_freeze_teacher(teacher: FuseTeacher) -> bool:
    """Erase T-stage autograd buffers before locking the selected snapshot."""
    for parameter in teacher.parameters():
        parameter.grad = None
    teacher.freeze_for_joint()
    clean = all(parameter.grad is None and not parameter.requires_grad
                for parameter in teacher.parameters()) and not teacher.training
    if not clean:
        raise RuntimeError("FUSE teacher did not enter a clean frozen J state")
    return clean


def _assert_teacher_clean(teacher: FuseTeacher) -> bool:
    clean = (not teacher.training and
             all(parameter.grad is None and not parameter.requires_grad
                 for parameter in teacher.parameters()))
    if not clean:
        raise RuntimeError("FUSE teacher acquired gradients or left eval mode during J")
    return clean


def _export_receipts(out: Path, teacher: FuseTeacher, student: FuseStudent,
                     config: Mapping[str, Any], identity: Mapping[str, Any],
                     validation: Mapping[str, Any], deployment: Mapping[str, Any], *,
                     zero_shot: Mapping[str, Any] | None = None) -> tuple[dict[str, Any], bool, dict[str, Any]]:
    export_dir = out / "exports"
    export_dir.mkdir(parents=True, exist_ok=True)
    result = export_model(None, teacher, config, export_dir / "teacher.pt", mode="teacher",
                          metadata={"identity": dict(identity), "validation": validation})
    teacher_receipt = {"path": str(Path(result["path"]).resolve()),
                       "sha256": result["sha256"], "mode": "teacher",
                       "manifest_path": str(Path(result["manifest_path"]).resolve())}
    receipts: dict[str, Any] = {"teacher": teacher_receipt}
    for mode in ("student", "fusion"):
        result = export_model(student, teacher, config, export_dir / f"{mode}.pt",
            mode=mode, metadata={"identity": dict(identity), "validation": validation,
                                 "deployment_mode": deployment["mode"]})
        receipts[mode] = {"path": str(Path(result["path"]).resolve()),
                          "sha256": result["sha256"], "mode": mode,
                          "manifest_path": str(Path(result["manifest_path"]).resolve())}
    if zero_shot is not None:
        # The report-only source comparison is frozen before T; bind the
        # original bytes into the final B bundle without re-exporting T.
        source = dict(zero_shot["source_teacher"])
        source["configured_source"] = copy.deepcopy(config["experiment"]["source_teacher"])
        source.update(zero_shot.get("source_teacher_verification", {}))
        verified = _verify_zero_shot_artifacts(out, identity=identity,
            source=source, expected=zero_shot)
        receipts["teacher_zero_shot"] = {"path": verified["path"],
            "sha256": verified["sha256"], "mode": "teacher_zero_shot",
            "manifest_path": verified["manifest_path"]}
    # Verify the exact emitted bytes and exported module inventories. This is
    # a measured check, not a claim inferred from how export_model is called.
    teacher_payload = torch.load(receipts["teacher"]["path"], map_location="cpu", weights_only=True)
    student_payload = torch.load(receipts["student"]["path"], map_location="cpu", weights_only=True)
    fusion_payload = torch.load(receipts["fusion"]["path"], map_location="cpu", weights_only=True)
    teacher_clean = (teacher_payload.get("schema") == "self_audit.candidate_fuse.export.v1" and
                     teacher_payload.get("mode") == "teacher" and
                     bool(teacher_payload.get("teacher_state")) and
                     not teacher_payload.get("student_state"))
    student_keys = set(student_payload.get("student_state", {}))
    fusion_keys = set(fusion_payload.get("student_state", {}))
    student_clean = (student_payload.get("schema") == "self_audit.candidate_fuse.export.v1" and
                     student_payload.get("mode") == "student" and bool(student_keys) and
                     not any(key.startswith(("fusions.", "mimic_heads.")) for key in student_keys) and
                     "teacher_state" not in student_payload)
    fusion_clean = (fusion_payload.get("schema") == "self_audit.candidate_fuse.export.v1" and
                    fusion_payload.get("mode") == "fusion" and bool(fusion_payload.get("teacher_state")) and
                    any(key.startswith("fusions.") for key in fusion_keys) and
                    not any(key.startswith("mimic_heads.") for key in fusion_keys))
    receipts_clean = all(Path(r["path"]).is_file() and _sha(Path(r["path"])) == r["sha256"]
                         for r in receipts.values())
    sidecars_clean = all(Path(r["manifest_path"]).is_file() and
                         json.loads(Path(r["manifest_path"]).read_text()).get("artifact_sha256")==r["sha256"]
                         for r in receipts.values())
    clean = bool(teacher_clean and student_clean and fusion_clean and receipts_clean and sidecars_clean)
    if not clean:
        raise RuntimeError("FUSE comparison export inventory or SHA verification failed")
    selected = receipts[str(deployment["mode"])]
    deployment_receipt={**dict(deployment), "path": selected["path"],
                        "sha256": selected["sha256"]}
    return receipts, clean, deployment_receipt


def _load_resume(out: Path, identity: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    path = out / "resume.pt"
    sha_path = out / "resume.sha256.json"
    if not path.is_file() or not sha_path.is_file():
        raise FileNotFoundError("resume requested but atomic resume.pt and SHA receipt are required")
    receipt = json.loads(sha_path.read_text())
    if receipt.get("sha256") != _sha(path):
        raise ValueError("resume checkpoint SHA-256 mismatch")
    payload = torch.load(path, map_location=device, weights_only=False)
    if payload.get("identity") != dict(identity):
        raise ValueError("resume config or manifest identity differs from checkpoint")
    if not isinstance(payload.get("metrics_receipt"), Mapping):
        raise ValueError("resume checkpoint lacks its committed metrics receipt")
    return payload


def _load_teacher_reference(reference: Mapping[str, Any], identity: Mapping[str, Any],
                            teacher: FuseTeacher, device: torch.device, *, profile: bool = False,
                            supplied_path: str | Path | None = None) -> None:
    validated = _teacher_reference(Path(reference.get("path", "")), identity, profile=profile)
    if validated != dict(reference):
        raise ValueError("J resume checkpoint teacher reference differs from its verified artifact")
    payload = torch.load(validated["path"], map_location=device, weights_only=False)
    if supplied_path is not None:
        supplied = _teacher_reference(Path(supplied_path), identity, profile=profile)
        supplied_payload = torch.load(supplied["path"], map_location="cpu", weights_only=False)
        pinned_payload = torch.load(validated["path"], map_location="cpu", weights_only=False)
        left, right = supplied_payload.get("model"), pinned_payload.get("model")
        if not isinstance(left, Mapping) or not isinstance(right, Mapping) or set(left) != set(right) or any(
                not torch.equal(left[key].detach().cpu(), right[key].detach().cpu()) for key in left):
            raise ValueError("supplied teacher artifact weights differ from the J checkpoint pin")
    teacher.load_state_dict(payload["model"], strict=True)


def _transfer_source_reference(config: Mapping[str, Any]) -> dict[str, Any]:
    """Verify Track B's source artifact and metadata before any deserialization."""
    source = config["experiment"]["source_teacher"]
    raw_path = source.get("path")
    if not isinstance(raw_path, str) or not raw_path.strip():
        raise ValueError("Track B requires a concrete selected source teacher path")
    path = Path(raw_path).expanduser().resolve()
    metadata_path = Path(str(path) + ".meta.json")
    if not path.is_file() or not metadata_path.is_file():
        raise FileNotFoundError("Track B selected source teacher or metadata is missing")
    digest = _sha(path)
    metadata_digest = _sha(metadata_path)
    if digest != source["sha256"]:
        raise ValueError("Track B source teacher SHA256 mismatch")
    if metadata_digest != source["metadata_sha256"]:
        raise ValueError("Track B source teacher metadata SHA256 mismatch")
    metadata = json.loads(metadata_path.read_text())
    if metadata.get("status") != "SELECTED":
        raise ValueError("Track B source teacher metadata must have SELECTED status")
    if metadata.get("sha256") != digest or metadata.get("identity") != source["identity"]:
        raise ValueError("Track B source teacher metadata identity/SHA mismatch")
    return {"path": str(path), "sha256": digest,
            "metadata_path": str(metadata_path), "metadata_sha256": metadata_digest,
            "identity": dict(source["identity"]), "status": metadata["status"]}


def _import_transfer_source(reference: Mapping[str, Any], teacher: FuseTeacher) -> None:
    # The bytes and sidecar were verified by _transfer_source_reference before
    # this weights_only load. Q3D's alias-aware strict loader checks every
    # encoder, decoder and region/boundary head tensor.
    payload = torch.load(reference["path"], map_location="cpu", weights_only=True)
    if not isinstance(payload, Mapping) or set(payload) != {"model"} or not isinstance(payload["model"], Mapping):
        raise ValueError("Track B source payload must contain exactly the selected model state")
    try:
        teacher.load_state_dict(payload["model"], strict=True)
    except (RuntimeError, ValueError, TypeError) as exc:
        raise ValueError("Track B source teacher failed strict full-model loading") from exc
    del payload


def _verify_zero_shot_artifacts(out: Path, *, identity: Mapping[str, Any],
                                source: Mapping[str, Any], expected: Mapping[str, Any] | None = None) -> dict[str, Any]:
    artifact = out / "exports" / "teacher_zero_shot.pt"
    manifest = Path(str(artifact) + ".manifest.json")
    report = out / "ZERO_SHOT_VAL.json"
    if not artifact.is_file() or not manifest.is_file() or not report.is_file():
        raise ValueError("Track B requires the committed zero-shot teacher export and VAL receipt")
    artifact_sha = _sha(artifact)
    val_sha = _sha(report)
    sidecar = json.loads(manifest.read_text())
    result = json.loads(report.read_text())
    if (sidecar.get("schema") != "self_audit.candidate_fuse.export.v1"
            or sidecar.get("mode") != "teacher" or sidecar.get("artifact_sha256") != artifact_sha):
        raise ValueError("Track B zero-shot export SHA/sidecar mismatch")
    configured_source = source.get("configured_source", source)
    if "configured_source" not in source:
        # Direct callers/tests may pass the verifier result; engine callers
        # always provide the canonical config mapping explicitly.
        configured_source = {key: source[key] for key in
                             ("path", "sha256", "metadata_sha256", "identity")}
    expected_verification = source.get("source_teacher_verification")
    if expected_verification is None:
        expected_verification = {key: source[key] for key in
                                 ("path", "sha256", "metadata_sha256", "identity", "status")}
        if "metadata_path" in source:
            expected_verification["metadata_path"] = source["metadata_path"]
    if (result.get("source_teacher") != configured_source
            or result.get("source_teacher_verification") != expected_verification
            or result.get("identity") != dict(identity)):
        raise ValueError("Track B zero-shot VAL does not match the configured source and target identity")
    if result.get("export_sha256") != artifact_sha or "teacher_zero_shot" not in result.get("metrics", {}):
        raise ValueError("Track B zero-shot VAL receipt does not bind its export")
    export_payload = torch.load(artifact, map_location="cpu", weights_only=True)
    export_metadata = export_payload.get("metadata", {}) if isinstance(export_payload, Mapping) else {}
    if (not isinstance(export_payload, Mapping)
            or export_payload.get("schema") != "self_audit.candidate_fuse.export.v1"
            or export_payload.get("mode") != "teacher"
            or not isinstance(export_payload.get("teacher_state"), Mapping)
            or export_payload.get("student_state")
            or export_metadata.get("comparison_label") != "teacher_zero_shot"
            or export_metadata.get("source_teacher") != configured_source
            or export_metadata.get("source_teacher_verification") != expected_verification):
        raise ValueError("Track B zero-shot export payload is not the pinned source-only teacher")
    del export_payload
    if expected is not None and (expected.get("sha256") != artifact_sha
                                 or expected.get("val_sha256") != val_sha
                                 or expected.get("source_teacher") != configured_source):
        raise ValueError("Track B resume zero-shot pins differ from committed artifacts")
    return {"path": str(artifact.resolve()), "sha256": artifact_sha,
            "manifest_path": str(manifest.resolve()), "mode": "teacher_zero_shot",
            "val_path": str(report.resolve()), "val_sha256": val_sha,
            "source_teacher": configured_source,
            "source_teacher_verification": expected_verification}


def _create_zero_shot_artifacts(out: Path, *, config: Mapping[str, Any], identity: Mapping[str, Any],
                                source: Mapping[str, Any], teacher: FuseTeacher,
                                val_data: Any, device: torch.device) -> dict[str, Any]:
    artifact = out / "exports" / "teacher_zero_shot.pt"
    report = out / "ZERO_SHOT_VAL.json"
    if artifact.exists() or Path(str(artifact) + ".manifest.json").exists() or report.exists():
        return _verify_zero_shot_artifacts(out, identity=identity,
            source=dict(source, configured_source=copy.deepcopy(config["experiment"]["source_teacher"])))
    # The export contract pins the exact configured source_teacher mapping.
    # Keep the resolved path/status used during verification separately so
    # provenance is both exact and auditable without changing v3 config data.
    source_public = copy.deepcopy(config["experiment"]["source_teacher"])
    source_verification = {key: source[key] for key in
                           ("path", "sha256", "metadata_path", "metadata_sha256", "identity", "status")}
    teacher.eval()
    exported = export_model(None, teacher, config, artifact, mode="teacher",
        metadata={"identity": dict(identity), "source_teacher": source_public,
                  "source_teacher_verification": source_verification,
                  "comparison_label": "teacher_zero_shot", "selection_role": "report_only"})
    metrics = evaluate_modes(teacher, None, val_data, device=device, modes=("teacher",))
    _atomic_json(report, {"schema": "self_audit.candidate_fuse.zero_shot_val.v1",
        "comparison_label": "teacher_zero_shot", "selection_role": "report_only",
        "identity": dict(identity), "source_teacher": source_public,
        "source_teacher_verification": source_verification,
        "export_sha256": exported["sha256"], "metrics": {"teacher_zero_shot": metrics["teacher"]}})
    verification = dict(source, configured_source=source_public)
    return _verify_zero_shot_artifacts(out, identity=identity, source=verification)


def _teacher_reference(path: Path, identity: Mapping[str, Any], *, profile: bool = False) -> dict[str, Any]:
    path = path.resolve()
    meta_path = Path(str(path) + ".meta.json")
    if not path.is_file() or not meta_path.is_file():
        raise ValueError("selected FUSE teacher artifact or metadata is missing")
    meta = json.loads(meta_path.read_text())
    digest = _sha(path)
    if meta.get("identity") != dict(identity) or meta.get("sha256") != digest:
        raise ValueError("selected FUSE teacher artifact identity or SHA is inconsistent")
    if meta.get("status") not in {"SELECTED", "PROFILE"}:
        raise ValueError("selected FUSE teacher artifact status is invalid")
    if meta.get("status") == "PROFILE" and not profile:
        raise ValueError("PROFILE teacher cannot be imported or resumed for production J")
    return {"path": str(path), "sha256": digest, "identity": dict(identity)}


def _restore_sampler(sampler: Any, state: Mapping[str, Any], *, continue_teacher: bool) -> None:
    """Restore valid initial-T sampler states with an empty lazy J L-stream.

    FuseSampler.load_state_dict currently validates l_stream as a full
    permutation even though its fresh state is intentionally empty. Pass a
    temporary valid permutation through that validator, then restore the lazy
    initial state when T is continuing. For a T-to-J transition, advance the
    saved Python RNG exactly as next_pair's first lazy shuffle would.
    """
    restored = copy.deepcopy(dict(state))
    original_rng = restored.get("py_rng")
    empty_l = not restored.get("l_stream") and int(restored.get("l_cursor", 0)) == 0
    if empty_l:
        rng = random.Random()
        rng.setstate(original_rng)
        l_stream = list(sampler.lkeys)
        rng.shuffle(l_stream)
        restored["l_stream"] = l_stream
        if not continue_teacher:
            restored["py_rng"] = rng.getstate()
    sampler.load_state_dict(restored)
    if empty_l and continue_teacher:
        sampler.py_rng.setstate(original_rng)
        sampler._l_stream = []
        sampler._l_cursor = 0


def _new_or_append_metrics(out: Path, resume: bool) -> None:
    path = out / "metrics.jsonl"
    if path.exists() and not resume and path.stat().st_size:
        raise FileExistsError("output already contains metrics; use resume or a new run directory")
    path.touch(exist_ok=True)


def run_training(config: Mapping[str, Any], out: str | Path, *, device: str = "cpu", stage: str = "all",
                 resume: bool = False, max_updates: int | None = None, teacher_artifact: str | Path | None = None,
                 profile: bool = False) -> dict[str, Any]:
    """Train T then J by epoch, checkpointing every committed optimizer update."""
    cfg = validate_config(config, real_run=not bool(config.get("runtime", {}).get("synthetic")))
    if stage not in {"all", "T", "J"}: raise ValueError("stage must be all, T or J")
    if max_updates is not None and max_updates <= 0: raise ValueError("max_updates must be positive")
    synthetic = bool(cfg["runtime"]["synthetic"])
    if profile and max_updates is None: raise ValueError("profile runs require an explicit bounded max_updates")
    if max_updates is not None and not (synthetic or profile):
        raise ValueError("bounded real FUSE runs are profile-only")
    deterministic=bool(cfg["runtime"]["deterministic"])
    target_device = torch.device(device)
    cuda_requested = target_device.type == "cuda"
    if deterministic and cuda_requested:
        # Set before any CUDA availability/context query or seed call.
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.use_deterministic_algorithms(deterministic, warn_only=(deterministic and cuda_requested))
    determinism_status = "BEST_EFFORT_GPU_WARN_ONLY" if deterministic and cuda_requested else (
        "STRICT_CPU" if deterministic else "DISABLED")
    determinism_note = ("CUDA determinism is best-effort warn-only for kernels without deterministic backward implementations; numerical, finite-value, and data-contract errors still fail closed."
                        if deterministic and cuda_requested else None)
    if deterministic and cuda_requested:
        import warnings
        warnings.warn("FUSE CUDA determinism is best-effort warn-only because some 3D interpolation/EPA backward kernels lack deterministic implementations; this does not suppress numerical, finite-value, or data-contract errors.", RuntimeWarning)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic=deterministic
        torch.backends.cudnn.benchmark=not deterministic
    if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
        torch.backends.cuda.matmul.allow_tf32=not deterministic
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.allow_tf32=not deterministic
    if target_device.type == "cuda" and not torch.cuda.is_available(): raise RuntimeError("CUDA requested but unavailable")
    if target_device.type == "cpu": torch.set_num_threads(cfg["runtime"]["num_threads"])
    if target_device.type == "cuda": torch.cuda.reset_peak_memory_stats(target_device)
    run_started = time.perf_counter()
    torch.manual_seed(cfg["seed"]); np.random.seed(cfg["seed"]); random.seed(cfg["seed"])
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(cfg["seed"])
    out_path = Path(out).resolve(); out_path.mkdir(parents=True, exist_ok=True)
    _new_or_append_metrics(out_path, resume)
    train_data = _data(cfg, "train", synthetic); val_data = _data(cfg, "development", synthetic)
    _validate_semi_track_a(cfg, train_data, val_data)
    _validate_transfer_track_b(cfg, train_data, val_data)
    identity = _identity(cfg, train_data, val_data)
    transfer = cfg["schema"] == TRANSFER_SCHEMA
    source_reference = _transfer_source_reference(cfg) if transfer else None
    py_rng = random.Random(cfg["seed"]); rng_generator = torch.Generator(device=target_device).manual_seed(cfg["seed"])
    teacher, teacher_provenance = build_teacher(cfg, synthetic=synthetic)
    teacher.to(target_device)
    student = FuseStudent(cfg, teacher_channels=teacher.feature_channels, teacher_phases=teacher.feature_phases).to(target_device)
    teacher_epoch_size = len(train_data.labeled_keys())
    teacher_batch_size = int(cfg["training"]["teacher_batch_size"])
    teacher_updates = teacher_epoch_size * int(cfg["training"]["teacher_epochs"])
    transfer_budget = (_transfer_teacher_budget(teacher_epoch_size,
        int(cfg["training"]["teacher_epochs"]), batch_size=teacher_batch_size) if transfer else None)
    teacher_step_budget = (transfer_budget["optimizer_steps"] if transfer_budget is not None else
        math.ceil(teacher_epoch_size / teacher_batch_size) * int(cfg["training"]["teacher_epochs"]))
    joint_updates = len(train_data.patients) * int(cfg["training"]["joint_epochs"])
    if max_updates is not None:
        if transfer:
            if stage in {"all", "T"}:
                transfer_budget = _transfer_teacher_budget(teacher_epoch_size,
                    int(cfg["training"]["teacher_epochs"]), batch_size=teacher_batch_size,
                    max_optimizer_steps=max_updates)
                teacher_step_budget = transfer_budget["optimizer_steps"]
                teacher_updates = transfer_budget["examples_seen"]
        else:
            teacher_updates = min(teacher_updates, max_updates) if stage in {"all", "T"} else teacher_updates
        joint_updates = min(joint_updates, max_updates) if stage in {"all", "J"} else joint_updates
    start_stage = stage
    teacher_ref = None
    transition_t_to_j = False
    resume_payload = None
    metrics = None
    zero_shot_info = None
    if resume:
        payload = _load_resume(out_path, identity, target_device)
        resume_payload = payload
        archived_suffix = _recover_metrics(out_path, payload["metrics_receipt"])
        start_stage = payload["stage"]
        transition_t_to_j = stage == "J" and start_stage == "T"
        if transfer:
            if not isinstance(payload.get("zero_shot_teacher"), Mapping):
                raise ValueError("Track B resume checkpoint lacks the committed zero-shot teacher pins")
            zero_shot_info = _verify_zero_shot_artifacts(out_path, identity=identity,
                source=source_reference, expected=payload.get("zero_shot_teacher"))
        if start_stage == "J":
            teacher_ref = payload.get("teacher_artifact")
            if not isinstance(teacher_ref, Mapping):
                raise ValueError("J resume checkpoint has no selected teacher artifact reference")
            _load_teacher_reference(teacher_ref, identity, teacher, target_device,
                                    profile=profile, supplied_path=teacher_artifact)
        elif transition_t_to_j:
            if not payload.get("selected_teacher"):
                raise ValueError("T checkpoint has no selected teacher for the T-to-J transition")
            teacher_ref = _teacher_reference(out_path / "teacher.pt", identity, profile=profile)
            _load_teacher_reference(teacher_ref, identity, teacher, target_device,
                                    profile=profile, supplied_path=teacher_artifact)
        elif payload.get("teacher") is not None:
            teacher.load_state_dict(payload["teacher"], strict=True)
        if payload.get("student") is not None: student.load_state_dict(payload["student"], strict=True)
        _restore_rng(payload["rng"], rng_generator, py_rng)
    elif stage == "J":
        if teacher_artifact is None: raise ValueError("J stage requires selected --fuse-teacher-artifact")
        artifact = Path(teacher_artifact)
        source_ref = _teacher_reference(artifact, identity, profile=profile)
        _load_teacher_reference(source_ref, identity, teacher, target_device, profile=profile)
        imported_teacher=torch.load(source_ref["path"], map_location=target_device, weights_only=False)
        copied_sha=_atomic_torch_save(imported_teacher,out_path/"teacher.pt")
        meta=json.loads(Path(str(artifact.resolve())+".meta.json").read_text())
        meta["sha256"]=copied_sha
        _write_json(out_path/"teacher.pt.meta.json",meta)
        teacher_ref=_teacher_reference(out_path/"teacher.pt",identity,profile=profile)
    elif stage == "all" and (out_path / "resume.pt").exists() and not resume:
        raise FileExistsError("existing progress requires --fuse-resume")
    elif transfer:
        _import_transfer_source(source_reference, teacher)
        zero_shot_info = _create_zero_shot_artifacts(out_path, config=cfg, identity=identity,
            source=source_reference, teacher=teacher, val_data=val_data, device=target_device)
    if transfer and zero_shot_info is None:
        # Stage J begins from the selected adapted teacher, while retaining and
        # verifying the original source comparison created before T.
        zero_shot_info = _verify_zero_shot_artifacts(out_path, identity=identity,
            source=dict(source_reference,
                        configured_source=copy.deepcopy(cfg["experiment"]["source_teacher"])))
    sampler = None
    from .data import FuseSampler
    sampler = FuseSampler(train_data, seed=cfg["seed"])
    need_teacher_optimizer = not (resume and start_stage == "J") and stage != "J"
    teacher_opt = (torch.optim.AdamW(teacher_parameter_groups(teacher, cfg),
        weight_decay=float(cfg["training"]["weight_decay"])) if need_teacher_optimizer else None)
    student_opt = torch.optim.AdamW(student.parameters(), lr=float(cfg["training"]["student_lr"]),
                                    weight_decay=float(cfg["training"]["weight_decay"]))
    current_epoch = 0; current_update = 0; best_t = -float("inf"); best_j = -float("inf")
    failure_stage = start_stage
    selected_teacher = False; teacher_state = None
    if resume:
        if start_stage == "T" and not transition_t_to_j:
            if teacher_opt is None or "teacher_optimizer" not in payload:
                raise ValueError("T resume checkpoint lacks teacher optimizer state")
            teacher_opt.load_state_dict(payload["teacher_optimizer"])
        if not transition_t_to_j:
            student_opt.load_state_dict(payload["student_optimizer"])
        _restore_sampler(sampler,payload["sampler"],continue_teacher=(start_stage=="T" and not transition_t_to_j))
        current_epoch=payload["epoch"]; current_update=payload["update"]
        best_t=payload.get("best_t",best_t); best_j=payload.get("best_j",best_j)
        selected_teacher=payload.get("selected_teacher",False)
        if transition_t_to_j:
            # T is complete: retain its RNG, sampler cursors and initialized
            # student weights, while beginning a fresh J optimizer/schedule.
            current_epoch = 0
            current_update = 0
    try:
        do_teacher = stage in {"all", "T"} and not (resume and start_stage == "J")
        if do_teacher:
            failure_stage = "T"
            t_epochs = int(cfg["training"]["teacher_epochs"])
            epoch_size = sampler.teacher_epoch_size
            t_budget = min(teacher_updates, epoch_size*t_epochs)
            if transfer:
                t_step_budget = min(teacher_step_budget, math.ceil(epoch_size / teacher_batch_size) * t_epochs)
                t_budget = min(t_budget, t_step_budget * teacher_batch_size)
            else:
                t_step_budget = None
            t_a_boundary = 0 if transfer else max(1, round(t_budget * 0.05))
            t_update = current_update if resume and start_stage == "T" else 0
            t_start_epoch = (current_epoch if resume and start_stage == "T" else 0)
            if resume and start_stage == "T" and t_update % epoch_size:
                t_start_epoch = t_update // epoch_size
            t_optimizer_steps = int(payload.get("teacher_optimizer_steps", t_update)) if resume and start_stage == "T" else 0
            metrics = payload.get("last_validation") if resume and start_stage == "T" else None
            for epoch in range(t_start_epoch, t_epochs):
                # A committed epoch boundary may already sit exactly at a
                # bounded profile's final budget. Do not repeat its VAL or
                # replace the selected snapshot when resuming that same cap.
                if t_update >= t_budget or (t_step_budget is not None and t_optimizer_steps >= t_step_budget):
                    break
                teacher.train(True); losses=[]
                in_epoch = t_update % epoch_size
                remaining = min(epoch_size - in_epoch if in_epoch else epoch_size, t_budget-t_update)
                epoch_end_update = t_update + max(0, remaining)
                while t_update < epoch_end_update and (t_step_budget is None or t_optimizer_steps < t_step_budget):
                    configured_batch = int(cfg["training"]["teacher_batch_size"])
                    # Track B profiles/budgets count optimizer updates; the
                    # historical A tracks keep their frame-exposure units.
                    checkpoint_cursor = t_optimizer_steps if transfer else t_update
                    checkpoint_quantum = configured_batch if transfer else 1
                    until_checkpoint = (20 - (checkpoint_cursor % 20)) * checkpoint_quantum
                    until_phase = (configured_batch if transfer or t_update >= t_a_boundary
                                   else t_a_boundary-t_update)
                    take = min(configured_batch, epoch_end_update-t_update, t_budget-t_update,
                               until_checkpoint, until_phase)
                    keys=sampler.next_labeled_batch(take)
                    frames=[train_data.load(key,labeled=True) for key in keys]
                    samples=[_frame_tensor(frame,target_device) for frame in frames]
                    if len(samples)==1:
                        image,target,geometry=_transform(*samples[0],py_rng)
                    else:
                        first_image,first_target,geometry=samples[0]
                        for image_i,target_i,geometry_i in samples[1:]:
                            if image_i.shape != first_image.shape or target_i.shape != first_target.shape or any(geometry_i[k] != geometry[k] for k in ("shape","spacing_zyx","affine_zyx","context")):
                                raise ValueError("same-patient teacher batch must have identical native shape and geometry")
                        runtime={key:geometry[key] for key in ("shape","spacing_zyx","affine_zyx","context")}
                        transform=sample_spatial_transform(runtime,rng=py_rng)
                        transformed=[]
                        for image_i,target_i,geometry_i in samples:
                            xi,yi,gi=apply_spatial_transform(image_i,target_i,runtime,**transform)
                            gi.update({k:v for k,v in geometry_i.items() if k not in gi})
                            transformed.append((xi,yi,gi))
                        image=torch.cat([v[0] for v in transformed],dim=0)
                        target=torch.cat([v[1] for v in transformed],dim=0)
                        geometry=transformed[0][2]
                    configure_teacher_step(teacher,teacher_opt,t_update,t_budget,cfg)
                    teacher_opt.zero_grad(set_to_none=True)
                    with _autocast(target_device,cfg):
                        outputs=teacher(image,geometry,return_features=False)
                        loss=_teacher_batch_loss(outputs,target,geometry)
                    if not torch.isfinite(loss): raise FloatingPointError("nonfinite FUSE teacher loss")
                    loss.backward(); torch.nn.utils.clip_grad_norm_(teacher.parameters(),float(cfg["training"]["gradient_clip_norm"]))
                    teacher_opt.step(); losses.append(float(loss.detach().float())); t_update+=len(keys); t_optimizer_steps+=1
                    _metric(out_path,{"type":"train_update","stage":"T","epoch":epoch,"update":t_update,"optimizer_steps":t_optimizer_steps,"batch_size":len(keys),"examples_seen":t_update,"loss":losses[-1]})
                    checkpoint_cursor = t_optimizer_steps if transfer else t_update
                    checkpoint_budget = t_step_budget if transfer else t_budget
                    checkpoint_due = (_checkpoint_due(checkpoint_cursor, checkpoint_budget, synthetic=synthetic)
                                      if checkpoint_cursor > 0 else False)
                    if checkpoint_due and checkpoint_cursor < checkpoint_budget and t_update % epoch_size != 0:
                        payload={"identity":identity,"stage":"T","epoch":epoch,"update":t_update,
                            "teacher":teacher.state_dict(),"student":student.state_dict(),
                            "teacher_optimizer":teacher_opt.state_dict(),"student_optimizer":student_opt.state_dict(),
                            "sampler":sampler.state_dict(),"rng":_rng_state(rng_generator,py_rng),
                            "best_t":best_t,"best_j":best_j,"selected_teacher":selected_teacher,
                            "teacher_optimizer_steps":t_optimizer_steps,
                            **({"zero_shot_teacher":zero_shot_info} if transfer else {})}
                        _save_resume(out_path,payload,"T",epoch,t_update)
                        current_epoch=epoch; current_update=t_update
                    if t_update>=t_budget: break
                metrics=evaluate_modes(teacher,None,val_data,device=target_device,modes=("teacher",))
                score=float(metrics["teacher"]["fg_dice"])
                if score>best_t:
                    best_t=score; teacher_state=copy.deepcopy(teacher.state_dict()); selected_teacher=True
                    meta_status="PROFILE" if profile else "SELECTED"
                    tpath=out_path/"teacher.pt"; digest=_atomic_torch_save({"model":teacher_state},tpath)
                    _write_json(Path(str(tpath)+".meta.json"),{"status":meta_status,"sha256":digest,"identity":identity,"provenance":teacher_provenance})
                    teacher_ref=_teacher_reference(tpath,identity,profile=profile)
                # Commit validation selection and exact RNG/sampler cursors
                # before appending the corresponding metric event.
                current_epoch=epoch+1; current_update=t_update
                boundary={"identity":identity,"stage":"T","epoch":epoch+1,"update":t_update,
                    "teacher":teacher.state_dict(),"student":student.state_dict(),
                    "teacher_optimizer":teacher_opt.state_dict(),"student_optimizer":student_opt.state_dict(),
                    "sampler":sampler.state_dict(),"rng":_rng_state(rng_generator,py_rng),
                    "best_t":best_t,"best_j":best_j,"selected_teacher":selected_teacher,
                    "teacher_optimizer_steps":t_optimizer_steps,
                    **({"zero_shot_teacher":zero_shot_info} if transfer else {}),
                    "last_validation":metrics}
                _metric(out_path,{"type":"validation","stage":"T","epoch":epoch+1,"update":t_update,"modes":metrics})
                _save_resume(out_path,boundary,"T",epoch+1,t_update)
                print(f"FUSE T epoch={epoch+1} update={t_update} loss={float(np.mean(losses)) if losses else float('nan'):.6g} val_fg={score:.6f}",flush=True)
                current_epoch=epoch+1; current_update=t_update
                if t_update>=t_budget or (t_step_budget is not None and t_optimizer_steps>=t_step_budget): break
            if teacher_state is None:
                if (out_path/"teacher.pt").is_file(): teacher_state=torch.load(out_path/"teacher.pt",map_location=target_device,weights_only=False)["model"]
                else: raise RuntimeError("teacher stage ended without a selected validation snapshot")
            teacher.load_state_dict(teacher_state,strict=True); _clear_and_freeze_teacher(teacher)
            if stage == "T":
                if metrics is None:
                    raise RuntimeError("completed T checkpoint is missing its committed validation metrics")
                teacher_result=export_model(None,teacher,cfg,out_path/"exports"/"teacher.pt",mode="teacher",
                    metadata={"identity":identity,"validation":metrics,
                              "status":"PROFILE" if profile else "SELECTED"})
                teacher_receipt={"path":str(Path(teacher_result["path"]).resolve()),
                    "sha256":teacher_result["sha256"],"mode":"teacher",
                    "manifest_path":str(Path(teacher_result["manifest_path"]).resolve())}
                teacher_exports={"teacher":teacher_receipt}
                if transfer:
                    teacher_exports["teacher_zero_shot"]={"path":zero_shot_info["path"],
                        "sha256":zero_shot_info["sha256"],"mode":"teacher_zero_shot",
                        "manifest_path":zero_shot_info["manifest_path"]}
                teacher_payload=torch.load(teacher_receipt["path"],map_location="cpu",weights_only=True)
                teacher_export_clean=(teacher_payload.get("schema")=="self_audit.candidate_fuse.export.v1" and
                    teacher_payload.get("mode")=="teacher" and bool(teacher_payload.get("teacher_state")) and
                    not teacher_payload.get("student_state") and
                    _sha(Path(teacher_receipt["path"]))==teacher_receipt["sha256"] and
                    json.loads(Path(teacher_receipt["manifest_path"]).read_text()).get("artifact_sha256")==teacher_receipt["sha256"])
                if not teacher_export_clean: raise RuntimeError("teacher-only export inventory or SHA check failed")
                result={"status":"PAUSED","stage":"T","update":t_update,"update_unit":"examples_seen",
                    "optimizer_steps":t_optimizer_steps,"batch_size":int(cfg["training"]["teacher_batch_size"]),
                    "examples_seen":t_update,"profile":"PROFILE" if profile else None}
                result.update({"teacher_artifact":str((out_path/"teacher.pt").resolve()),
                    "joint_checkpoint":None,"exports":teacher_exports,
                    "deployment":None,"teacher_has_no_grads_after_J":False,"export_clean":teacher_export_clean,
                    "determinism_status":determinism_status,"determinism_note":determinism_note})
                if profile:
                    if target_device.type == "cuda": torch.cuda.synchronize(target_device)
                    result["profile_metrics"]={"elapsed_seconds":time.perf_counter()-run_started,
                        "peak_memory_bytes":torch.cuda.max_memory_allocated(target_device) if target_device.type=="cuda" else None,
                        "dtype":cfg["runtime"]["amp_dtype"] if target_device.type=="cuda" else "float32",
                        "device":str(target_device),"updates":t_update,"update_unit":"examples_seen",
                        "optimizer_steps":t_optimizer_steps,"examples_seen":t_update}
                _atomic_json(out_path/"RESULT.json",result)
                _state(out_path,status="COMPLETED",stage="T",epoch=current_epoch,update=t_update,
                       checkpoint=out_path/"resume.pt" if (out_path/"resume.pt").is_file() else None,
                       checkpoint_sha=_sha(out_path/"resume.pt") if (out_path/"resume.pt").is_file() else None)
                return result
            # J consumes only the selected frozen T artifact. Release the
            # large optimizer state before moving into student training.
            if teacher_opt is not None:
                teacher_opt.state.clear()
                del teacher_opt
                teacher_opt=None
                gc.collect()
                if target_device.type=="cuda": torch.cuda.empty_cache()
        elif resume and teacher_ref is None and (out_path/"teacher.pt").is_file():
            teacher.load_state_dict(torch.load(out_path/"teacher.pt",map_location=target_device,weights_only=False)["model"],strict=True)
            teacher_ref=_teacher_reference(out_path/"teacher.pt",identity)
        if not getattr(teacher, "_joint_frozen", False):
            _clear_and_freeze_teacher(teacher)
        if stage in {"all","J"}:
            failure_stage = "J"
            if stage == "all" and not selected_teacher:
                raise RuntimeError("J requires a teacher selected by the T validation gate")
            if teacher_ref is None:
                teacher_ref=_teacher_reference(out_path/"teacher.pt",identity)
            if teacher_opt is not None:
                raise RuntimeError("teacher optimizer state must be released before J")
            student.train(True); total=joint_updates; epoch_size=sampler.joint_epoch_size
            start_j_epoch=(current_update // epoch_size) if resume and start_stage=="J" else 0
            j_update=current_update if resume and start_stage=="J" else 0
            max_epochs=int(cfg["training"]["joint_epochs"]); warmup=max(1,int(math.ceil(float(cfg["training"]["warmup_fraction"])*total)))
            for epoch in range(start_j_epoch,max_epochs):
                _assert_teacher_clean(teacher)
                losses=[]; in_epoch=j_update % epoch_size
                remaining=min(epoch_size-in_epoch if in_epoch else epoch_size,total-j_update)
                for _ in range(max(0,remaining)):
                    key_l,key_u=sampler.next_pair(); frame_l=train_data.load(key_l,labeled=True); frame_u=train_data.load(key_u,labeled=False)
                    student_opt.zero_grad(set_to_none=True)
                    for group in student_opt.param_groups:
                        if j_update<warmup: scale=(j_update+1)/warmup
                        else:
                            q=min(1.,(j_update-warmup)/max(1,total-warmup)); scale=.5*(1+math.cos(math.pi*q))
                        group["lr"]=float(cfg["training"]["student_lr"])*scale
                    global _joint_py_rng
                    _joint_py_rng=py_rng
                    with _autocast(target_device,cfg):
                        loss,stats=joint_loss(student,teacher,frame_l,frame_u,update=j_update,total_updates=total,config=cfg,generator=rng_generator)
                    _assert_teacher_clean(teacher)
                    loss.backward(); torch.nn.utils.clip_grad_norm_(student.parameters(),float(cfg["training"]["gradient_clip_norm"])); student_opt.step()
                    _assert_teacher_clean(teacher)
                    j_update+=1; losses.append(stats["loss"])
                    _metric(out_path,{"type":"train_update","stage":"J","epoch":epoch,"update":j_update,"losses":stats})
                    if (_checkpoint_due(j_update,total,synthetic=synthetic)
                            and j_update < total and j_update % epoch_size != 0):
                        payload={"identity":identity,"stage":"J","epoch":epoch,"update":j_update,
                            "teacher_artifact":teacher_ref,"student":student.state_dict(),
                            "student_optimizer":student_opt.state_dict(),"sampler":sampler.state_dict(),
                            "rng":_rng_state(rng_generator,py_rng),"best_t":best_t,"best_j":best_j,
                            "selected_teacher":True,
                            **({"zero_shot_teacher":zero_shot_info} if transfer else {})}
                        _save_resume(out_path,payload,"J",epoch,j_update)
                        current_epoch=epoch; current_update=j_update
                    if j_update>=total: break
                modes=evaluate_modes(teacher,student,val_data,device=target_device,modes=("teacher","student","fusion"))
                _assert_teacher_clean(teacher)
                _attach_gate_means(modes,teacher,student,val_data,target_device)
                score=float(modes["fusion"]["fg_dice"])
                if score>best_j:
                    best_j=score
                    digest=_atomic_torch_save({"student":copy.deepcopy(student.state_dict()),"epoch":epoch+1,"update":j_update,"identity":identity},out_path/"joint_best.pt")
                    _write_json(out_path/"joint_best.sha256.json",{"sha256":digest,"epoch":epoch+1,"update":j_update})
                current_epoch=epoch+1; current_update=j_update
                boundary={"identity":identity,"stage":"J","epoch":epoch+1,"update":j_update,
                    "teacher_artifact":teacher_ref,"student":student.state_dict(),
                    "student_optimizer":student_opt.state_dict(),
                    "sampler":sampler.state_dict(),"rng":_rng_state(rng_generator,py_rng),
                    "best_t":best_t,"best_j":best_j,"selected_teacher":True,
                    **({"zero_shot_teacher":zero_shot_info} if transfer else {})}
                _metric(out_path,{"type":"validation","stage":"J","epoch":epoch+1,"update":j_update,"modes":modes})
                _save_resume(out_path,boundary,"J",epoch+1,j_update)
                print(f"FUSE J epoch={epoch+1} update={j_update} loss={float(np.mean(losses)) if losses else float('nan'):.6g} val_fusion_fg={score:.6f}",flush=True)
                current_epoch=epoch+1; current_update=j_update
                if j_update>=total: break
            best_path=out_path/"joint_best.pt"
            if not best_path.is_file(): raise RuntimeError("J completed without selected joint snapshot")
            if _sha(best_path)!=json.loads((out_path/"joint_best.sha256.json").read_text())["sha256"]: raise ValueError("joint_best SHA mismatch")
            best=torch.load(best_path,map_location=target_device,weights_only=False); student.load_state_dict(best["student"],strict=True)
            metrics=evaluate_modes(teacher,student,val_data,device=target_device,modes=("teacher","student","fusion"))
            _assert_teacher_clean(teacher)
            _attach_gate_means(metrics,teacher,student,val_data,target_device)
            decision=choose_deployment(metrics,margin=.01)
            receipts,export_clean,deployment_receipt=_export_receipts(out_path,teacher,student,cfg,identity,metrics,decision,
                zero_shot=zero_shot_info if transfer else None)
            teacher_has_no_grads_after_j=_assert_teacher_clean(teacher)
            _metric(out_path,{"type":"validation_selected","stage":"J","epoch":best["epoch"],"update":best["update"],"modes":metrics,"deployment":decision})
        result={"status":"COMPLETED","stage":stage,"epoch":current_epoch,"update":current_update,
                "validation":metrics if stage!="T" else None,"deployment":decision if stage!="T" else None}
        if stage != "T":
            result.update({"teacher_artifact":str((out_path/"teacher.pt").resolve()),
                "joint_checkpoint":str((out_path/"joint_best.pt").resolve()),"exports":receipts,
                "deployment":deployment_receipt,
                "export_path":deployment_receipt["path"],
                "export_sha256":deployment_receipt["sha256"],
                "stages":["T","J","VAL3","deploy_export"],
                "teacher_has_no_grads_after_J":teacher_has_no_grads_after_j,
                "export_clean":export_clean,"determinism_status":determinism_status,
                "determinism_note":determinism_note})
        if profile:
            if target_device.type == "cuda": torch.cuda.synchronize(target_device)
            result["profile"]={"elapsed_seconds":time.perf_counter()-run_started,
                "peak_memory_bytes":torch.cuda.max_memory_allocated(target_device) if target_device.type=="cuda" else None,
                "dtype":cfg["runtime"]["amp_dtype"] if target_device.type=="cuda" else "float32",
                "device":str(target_device),"updates":current_update}
        _atomic_json(out_path/"RESULT.json",result)
        _state(out_path,status="COMPLETED",stage="J" if stage!="T" else "T",epoch=current_epoch,update=current_update)
        return result
    except Exception:
        _state(out_path,status="FAILED",stage=failure_stage,epoch=current_epoch,update=current_update,
               checkpoint=out_path/"resume.pt" if (out_path/"resume.pt").exists() else None,
               checkpoint_sha=_sha(out_path/"resume.pt") if (out_path/"resume.pt").exists() else None)
        raise


__all__ = ["joint_loss", "run_training"]
