"""Locked native-volume validation for the FUSE teacher, student and fusion."""
from __future__ import annotations

import hashlib
import json
import datetime as dt
import fcntl
import os
import random
from collections import defaultdict
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch

from self_audit.candidate_q3d.evaluation import frame_class_dice, surface_distances_3d

MODES = ("teacher", "student", "fusion")
TRANSFER_MODES = ("teacher_zero_shot", "teacher", "student", "fusion")
CLASS_NAMES = ("RV", "MYO", "LV")


def _logits(value: Any) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value
    if isinstance(value, Mapping):
        for key in ("region_logits", "logits"):
            if isinstance(value.get(key), torch.Tensor):
                return value[key]
    raise TypeError("model output must contain region_logits or logits Tensor")


def _snapshot_rng():
    state = (random.getstate(), np.random.get_state(), torch.get_rng_state())
    cuda = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    return (*state, cuda)


def _restore_rng(state):
    random.setstate(state[0]); np.random.set_state(state[1]); torch.set_rng_state(state[2])
    if state[3] is not None:
        torch.cuda.set_rng_state_all(state[3])


def _spacing(geometry: Mapping[str, Any]) -> tuple[float, float, float]:
    value = geometry.get("spacing_zyx", geometry.get("spacing_zyx_mm"))
    if value is None or len(value) != 3:
        raise ValueError("native geometry requires spacing_zyx in mm")
    spacing = tuple(float(v) for v in value)
    if any(not np.isfinite(v) or v <= 0 for v in spacing):
        raise ValueError("native geometry spacing must be finite and positive")
    return spacing


def _one_prediction(mode, teacher, student, image, geometry, device):
    image = torch.as_tensor(image, dtype=torch.float32, device=device)
    if image.ndim == 3:
        image = image[None, None]
    if image.ndim != 5 or tuple(image.shape[:2]) != (1, 1):
        raise ValueError("FUSE evaluation images must be [1,1,D,H,W]")
    if mode == "teacher":
        return _logits(teacher(image, geometry, return_features=False))
    if mode == "student":
        return _logits(student(image, geometry, teacher_features=None))
    output = teacher(image, geometry, return_features=True)
    if isinstance(output, tuple) and len(output) == 2:
        teacher_output, features = output
    elif isinstance(output, Mapping):
        teacher_output, features = output, output.get("features")
    else:
        raise TypeError("teacher fusion output must provide features")
    if features is None:
        raise ValueError("teacher did not provide fusion features")
    return _logits(student(image, geometry, teacher_features=features,
                           keep_mask=torch.ones((1, 1, 1, 1, 1), device=device)))


def evaluate_modes(teacher, student, data, *, device="cpu",
                   modes=("teacher", "student", "fusion"),
                   fusion_teacher=None, fusion_student=None) -> dict[str, Any]:
    """Evaluate modes on identical labeled frames; aggregate ED/ES within patient."""
    selected = tuple(modes)
    if not selected or len(set(selected)) != len(selected) or set(selected) - set(MODES):
        raise ValueError("modes must be a nonempty unique subset of teacher/student/fusion")
    if "teacher" in selected or "fusion" in selected:
        if teacher is None:
            raise ValueError("teacher is required for teacher/fusion evaluation")
    if "student" in selected and student is None:
        raise ValueError("student is required for student evaluation")
    models = [m for m in (teacher, student, fusion_teacher, fusion_student) if m is not None]
    seen = set()
    modules = []
    for model in models:
        for module in model.modules():
            if id(module) not in seen:
                seen.add(id(module))
                modules.append((module, module.training))
    rng = _snapshot_rng()
    rows = {mode: defaultdict(list) for mode in selected}
    try:
        # Fusion-only still requires both networks to run in evaluation mode.
        # Evaluating every supplied model also covers the three independently
        # exported checkpoints used by evaluate_exports.
        for model in models:
            model.eval()
        keys = data.labeled_keys()
        if not keys:
            raise ValueError("FUSE evaluation dataset has no labeled frames")
        with torch.inference_mode():
            for key in keys:
                frame = data.load(key, labeled=True)
                if frame.target is None:
                    raise ValueError("labeled evaluation frame has no target")
                reference = torch.as_tensor(frame.target).cpu().numpy().astype(np.uint8, copy=False)
                if reference.ndim == 4 and reference.shape[0] == 1:
                    reference = reference[0]
                image = frame.image
                patient = str(frame.patient_id)
                spacing = _spacing(frame.geometry)
                for mode in selected:
                    active_teacher = fusion_teacher if mode == "fusion" and fusion_teacher is not None else teacher
                    active_student = fusion_student if mode == "fusion" and fusion_student is not None else student
                    logits = _one_prediction(mode, active_teacher, active_student, image, frame.geometry, device)
                    prediction = logits.argmax(1)[0].detach().cpu().numpy().astype(np.uint8)
                    if prediction.shape != reference.shape:
                        raise ValueError("FUSE prediction and target native shapes differ")
                    dice = frame_class_dice(reference, prediction)
                    h95 = {}
                    for class_id, name in enumerate(CLASS_NAMES, 1):
                        h95[name] = surface_distances_3d(
                            reference == class_id, prediction == class_id, spacing
                        )["hd95_mm"]
                    rows[mode][patient].append((dice, h95))
    finally:
        # Calling train(flag) recursively would overwrite intentionally mixed
        # submodule states. Restore each captured bit directly.
        for module, flag in modules:
            module.training = flag
        _restore_rng(rng)
    result = {}
    for mode in selected:
        patient_metrics = {}
        for patient, frame_rows in rows[mode].items():
            per_class = {}
            for name in CLASS_NAMES:
                # ED/ES frames are macro averaged within patient. A class with
                # no reference support in any frame is explicitly absent.
                dvals = [float(d[name]["dice"]) for d, _ in frame_rows
                         if d[name]["reference_present"] and d[name]["dice"] is not None]
                hvals = [float(h[name]) for _, h in frame_rows if h[name] is not None]
                per_class[name] = {"dice": float(np.mean(dvals)) if dvals else None,
                                   "hd95_mm": float(np.mean(hvals)) if hvals else None,
                                   "frames_with_reference": len(dvals)}
            fg = [per_class[n]["dice"] for n in CLASS_NAMES if per_class[n]["dice"] is not None]
            patient_metrics[patient] = {"per_class": per_class,
                                        "fg_dice": float(np.mean(fg)) if fg else None}
        per_class = {}
        for name in CLASS_NAMES:
            ds = [p["per_class"][name]["dice"] for p in patient_metrics.values()
                  if p["per_class"][name]["dice"] is not None]
            hs = [p["per_class"][name]["hd95_mm"] for p in patient_metrics.values()
                  if p["per_class"][name]["hd95_mm"] is not None]
            per_class[name] = {"dice": float(np.mean(ds)) if ds else None,
                               "hd95_mm": float(np.mean(hs)) if hs else None,
                               "patient_count": len(ds), "hd95_patient_count": len(hs)}
        fg = [p["fg_dice"] for p in patient_metrics.values() if p["fg_dice"] is not None]
        result[mode] = {"fg_dice": float(np.mean(fg)) if fg else None,
                        "per_class": per_class,
                        "patient_count": len(patient_metrics),
                        "per_patient": patient_metrics}
    return result


def choose_deployment(metrics: Mapping[str, Any], *, margin: float = 0.01) -> dict[str, Any]:
    if margin != 0.01:
        raise ValueError("FUSE deploy margin is locked at 0.01")
    if not isinstance(metrics, Mapping) or not {"student", "fusion"} <= set(metrics):
        raise ValueError("deployment requires student and fusion validation metrics")
    student = metrics["student"].get("fg_dice")
    fusion = metrics["fusion"].get("fg_dice")
    if student is None or fusion is None or not np.isfinite(student) or not np.isfinite(fusion):
        raise ValueError("student and fusion FG Dice must be finite for deployment")
    mode = "student" if float(student) >= float(fusion) - margin else "fusion"
    return {"mode": mode, "student_fg_dice": float(student),
            "fusion_fg_dice": float(fusion), "margin": margin,
            "rule": "student_if_student_fg_dice_ge_fusion_minus_0.01"}


def evaluate_export(path, manifest, manifest_sha256, *cli_args, role, device="cpu", out=None):
    """Canonical evaluate.py adapter for an exported FUSE inference artifact."""
    artifact = Path(path).resolve()
    if role == "dev":
        role = "development"
    # Canonical evaluate.py dispatch retains its cross-track positional form:
    # (artifact, artifact_sha, data_manifest, data_sha, out). The public FUSE
    # API is the compact (artifact, data_manifest, data_sha, *, out=...) form.
    artifact_sha_arg = None
    if cli_args:
        if len(cli_args) != 2 or not isinstance(manifest, str) or len(manifest) != 64:
            raise TypeError("invalid canonical evaluate.py FUSE arguments")
        artifact_sha_arg = manifest
        manifest, manifest_sha256, out = manifest_sha256, cli_args[0], cli_args[1]
    if not artifact.is_file():
        raise ValueError("FUSE export artifact is missing")
    digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
    if artifact_sha_arg is not None and digest != artifact_sha_arg:
        raise ValueError("FUSE export SHA-256 mismatch")
    if role == "test":
        raise ValueError(
            "single-mode FUSE TEST evaluation is disabled; use evaluate_exports "
            "with the locked three-mode TEST transaction and ledger"
        )
    if role != "development":
        raise ValueError("FUSE export evaluation role must be development")
    sidecar_path = artifact.with_suffix(artifact.suffix + ".manifest.json")
    try:
        sidecar = json.loads(sidecar_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("FUSE export sidecar manifest is missing or invalid") from exc
    if (sidecar.get("schema") != "self_audit.candidate_fuse.export.v1"
            or sidecar.get("artifact") != artifact.name
            or sidecar.get("artifact_sha256") != digest
            or sidecar.get("artifact_bytes") != artifact.stat().st_size):
        raise ValueError("FUSE export sidecar does not identify the requested artifact bytes")
    from .data import FuseData
    from .export import load_export

    try:
        manifest_payload = json.loads(Path(manifest).read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("FUSE evaluation manifest is missing or invalid") from exc
    manifest_digest = hashlib.sha256(Path(manifest).read_bytes()).hexdigest()
    if manifest_digest != manifest_sha256:
        raise ValueError("FUSE evaluation manifest SHA-256 mismatch")
    data = FuseData(manifest, expected_sha256=manifest_sha256, role=role,
                    synthetic=bool(manifest_payload.get("synthetic", False)))
    model = load_export(artifact, device=device)
    mode = sidecar.get("mode")
    if mode not in {"teacher", "student", "fusion"}:
        raise ValueError("FUSE export sidecar has an invalid selected mode")
    result = evaluate_modes(model.teacher, model.student, data, device=device, modes=(mode,))
    payload = {"schema": "candidate_fuse.evaluation.v1", "role": role,
               "mode": mode, "artifact_sha256": digest,
               "manifest_sha256": manifest_sha256, "metrics": result[mode]}
    if out is not None:
        destination = Path(out)
        if destination.suffix.lower() == ".json":
            destination.parent.mkdir(parents=True, exist_ok=True)
        else:
            destination.mkdir(parents=True, exist_ok=True)
            destination = destination / f"FUSE_{role.upper()}_{mode.upper()}_METRICS.json"
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")
        temporary.replace(destination)
        payload["metrics_path"] = str(destination)
    return payload


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    os.replace(temporary, path)


def _same_state(left, right, label: str, *, allow_extra: bool = False) -> None:
    a, b = left.state_dict(), right.state_dict()
    if (not set(a).issubset(b) if allow_extra else set(a) != set(b)):
        raise ValueError(f"FUSE {label} export state keys differ")
    if any(not torch.equal(a[key].detach().cpu(), b[key].detach().cpu()) for key in a):
        raise ValueError(f"FUSE {label} export states differ across transaction bundle")


def evaluate_exports(bundle: dict, data_manifest, out, device, role, expected_sha256):
    """Run the locked three-mode or Track B four-mode transaction.

    ``bundle`` maps the three comparison modes, or all four Track B comparison
    labels, to export receipts ``{"path": ..., "sha256": ...}``. Track B's
    ``teacher_zero_shot`` receipt intentionally has payload/sidecar mode
    ``teacher``; its transaction label is report-only. ``expected_sha256`` is
    either the pinned data-manifest SHA string, or a mapping with ``manifest``
    and optional per-mode SHA values. ``data_manifest`` may be a path or a
    ``{"path": ..., "sha256": ...}`` mapping. A single atomic ledger is
    created under ``out``; an existing ledger locks that role against reruns.
    """
    selected_role = "development" if role == "dev" else role
    if selected_role not in {"development", "test"}:
        raise ValueError("FUSE export transaction role must be development or test")
    if not isinstance(bundle, Mapping) or set(bundle) not in (set(MODES), set(TRANSFER_MODES)):
        raise ValueError("FUSE export bundle must contain exactly three comparison receipts or the four Track B receipts")
    transfer = set(bundle) == set(TRANSFER_MODES)
    if transfer and selected_role != "test":
        raise ValueError("Track B four-mode evaluation is reserved for its one-shot TEST transaction")
    selected_modes = TRANSFER_MODES if transfer else MODES
    if isinstance(data_manifest, Mapping):
        manifest_path = Path(data_manifest["path"]).resolve()
        embedded_manifest_sha = data_manifest.get("sha256")
    else:
        manifest_path = Path(data_manifest).resolve()
        embedded_manifest_sha = None
    if isinstance(expected_sha256, Mapping):
        expected_manifest_sha = expected_sha256.get("manifest", expected_sha256.get("data_manifest"))
        expected_modes = {mode: expected_sha256.get(mode) for mode in selected_modes}
    else:
        expected_manifest_sha = expected_sha256
        expected_modes = {}
    if embedded_manifest_sha is not None:
        if expected_manifest_sha is not None and embedded_manifest_sha != expected_manifest_sha:
            raise ValueError("FUSE data manifest SHA declarations disagree")
        expected_manifest_sha = embedded_manifest_sha
    if not isinstance(expected_manifest_sha, str) or len(expected_manifest_sha) != 64:
        raise ValueError("FUSE transaction requires an expected data-manifest SHA-256")
    # Pin every artifact, sidecar and manifest before constructing FUSE data,
    # whose role-checked validation reads the ED/ES masks.
    from .export import _sha, _validate_sidecar, load_export
    pinned = {}
    sidecars = {}
    for mode in selected_modes:
        receipt = bundle[mode]
        if not isinstance(receipt, Mapping) or not isinstance(receipt.get("path"), (str, Path)):
            raise ValueError(f"FUSE {mode} receipt requires an artifact path")
        path = Path(receipt["path"]).resolve()
        expected = expected_modes.get(mode) or receipt.get("sha256")
        if not isinstance(expected, str) or len(expected) != 64 or _sha(path) != expected:
            raise ValueError(f"FUSE {mode} artifact SHA-256 mismatch")
        if receipt.get("sha256") not in (None, expected):
            raise ValueError(f"FUSE {mode} receipt SHA differs from the pinned SHA")
        expected_payload_mode = "teacher" if mode == "teacher_zero_shot" else mode
        if receipt.get("mode") not in (None, mode, expected_payload_mode):
            raise ValueError(f"FUSE {mode} receipt declares a different export mode")
        sidecar = _validate_sidecar(path, expected)
        if sidecar.get("mode") != expected_payload_mode:
            raise ValueError(f"FUSE {mode} receipt points to a different export mode")
        pinned[mode] = {"path": str(path), "sha256": expected}
        sidecars[mode] = sidecar
    if not manifest_path.is_file() or _sha(manifest_path) != expected_manifest_sha:
        raise ValueError("FUSE transaction data-manifest SHA-256 mismatch")

    # Read export metadata and weights-only payloads before constructing the
    # role-specific data reader, whose TEST load path can open target masks.
    transfer_config = None
    config_sha = None
    manifest_doc = None
    if transfer:
        payloads = {}
        for mode, receipt in pinned.items():
            payload = torch.load(receipt["path"], map_location="cpu", weights_only=True)
            if not isinstance(payload, Mapping) or payload.get("schema") != "self_audit.candidate_fuse.export.v1":
                raise ValueError(f"FUSE {mode} artifact is not a valid weights-only export")
            if not isinstance(payload.get("config"), Mapping):
                raise ValueError(f"FUSE {mode} export lacks its embedded config")
            payloads[mode] = payload
        from .config import TRANSFER_SCHEMA, config_sha256, validate_config
        try:
            transfer_config = validate_config(payloads["teacher"]["config"])
        except (TypeError, ValueError) as exc:
            raise ValueError("Track B transaction exports do not carry a valid v3 config") from exc
        if (transfer_config.get("schema") != TRANSFER_SCHEMA
                or transfer_config.get("experiment", {}).get("track") != "B"
                or transfer_config.get("experiment", {}).get("dataset") != "MNMS"):
            raise ValueError("four-mode FUSE transaction requires the embedded Track B v3 config")
        config_sha = config_sha256(transfer_config)
        configured_test_path = transfer_config["data"].get("test_manifest")
        configured_test_sha = transfer_config["data"].get("test_manifest_sha256")
        if (not isinstance(configured_test_path, str)
                or Path(configured_test_path).resolve() != manifest_path
                or configured_test_sha != expected_manifest_sha):
            raise ValueError("Track B config does not pin this exact target TEST manifest")
        source_teacher = transfer_config["experiment"]["source_teacher"]
        expected_data_identity = {
            key: transfer_config["data"].get(key) for key in
            ("train_manifest_sha256", "val_manifest_sha256", "test_manifest_sha256")
        }
        for mode in selected_modes:
            if payloads[mode].get("config") != transfer_config:
                raise ValueError("Track B exports do not embed the same complete v3 config")
            metadata = sidecars[mode].get("metadata")
            if (not isinstance(metadata, Mapping)
                    or metadata.get("config_sha256") != config_sha
                    or metadata.get("data_manifest_sha256") != expected_data_identity):
                raise ValueError(f"Track B {mode} export metadata differs from its config/data identity")
        zero_metadata = sidecars["teacher_zero_shot"].get("metadata", {})
        declared_source = zero_metadata.get("source_teacher") if isinstance(zero_metadata, Mapping) else None
        if declared_source != source_teacher:
            raise ValueError("zero-shot export source provenance differs from the Track B config")
        manifest_doc = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (manifest_doc.get("schema") != "self_audit.candidate_fuse.data.v3"
                or manifest_doc.get("role") != "test"
                or manifest_doc.get("dataset_release") != "original_2020_open345"):
            raise ValueError("Track B TEST manifest is not the locked M&Ms data.v3 target split")
        # Verify all source-container bytes and ZIP directory metadata before
        # creating the irreversible one-shot TEST transaction. No member data
        # is read or decompressed by this metadata-only pass.
        from .mnms_data import MNMSFuseData
        preflight = MNMSFuseData(manifest_path, expected_manifest_sha,
                                 role="test", synthetic=bool(manifest_doc.get("synthetic", False)))
        preflight.verify_sources()

    root = Path(out).resolve()
    root.mkdir(parents=True, exist_ok=True)
    ledger_path = root / ("TEST_TRANSACTION.json" if selected_role == "test"
                          else "DEVELOPMENT_TRANSACTION.json")
    lock_path = root / (ledger_path.name + ".lock")
    with lock_path.open("a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        if ledger_path.exists():
            raise ValueError(f"FUSE {selected_role} evaluation transaction already exists; refusing rerun")
        started = dt.datetime.now(dt.timezone.utc).isoformat()
        ledger = {"schema": "self_audit.candidate_fuse.evaluation_transaction.v1",
                  "status": "RUNNING", "role": selected_role, "started_utc": started,
                  "manifest": str(manifest_path), "manifest_sha256": expected_manifest_sha,
                  "artifacts": pinned}
        if transfer:
            ledger["config_sha256"] = config_sha
        _atomic_json(ledger_path, ledger)
        try:
            exports = {mode: load_export(value["path"], device=device,
                                          expected_sha256=value["sha256"])
                       for mode, value in pinned.items()}
            _same_state(exports["teacher"].teacher.model, exports["fusion"].teacher.model, "teacher")
            _same_state(exports["student"].student, exports["fusion"].student,
                        "student", allow_extra=True)
            if manifest_doc is None:
                manifest_doc = json.loads(manifest_path.read_text())
            from .data import open_fuse_data
            data = open_fuse_data(manifest_path, expected_sha256=expected_manifest_sha,
                                  role=selected_role,
                                  synthetic=bool(manifest_doc.get("synthetic", False)))
            metrics = evaluate_modes(
                exports["teacher"].teacher, exports["student"].student, data,
                device=device, modes=MODES,
                fusion_teacher=exports["fusion"].teacher,
                fusion_student=exports["fusion"].student,
            )
            if transfer:
                # This source-only comparison is report-only and cannot enter
                # validation selection or the student/fusion deploy rule.
                metrics["teacher_zero_shot"] = evaluate_modes(
                    exports["teacher_zero_shot"].teacher, None, data,
                    device=device, modes=("teacher",),
                )["teacher"]
            ledger.update({"status": "COMMITTED", "completed_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                           "metrics": metrics})
            _atomic_json(ledger_path, ledger)
            return ledger
        except Exception as exc:
            ledger.update({"status": "FAILED_LOCKED", "failed_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                           "error": f"{type(exc).__name__}: {exc}"})
            _atomic_json(ledger_path, ledger)
            raise
