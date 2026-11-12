"""DEV-only Q-3D codebook and physiology diagnostics.

This module is deliberately read-only: DEV images are evaluated before labels
are opened, and labels are used only for the reported teacher RV metrics. No
result from this module is a checkpoint-selection or training signal.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from ..candidate_a3d.evaluation import (
    _validated_manifest,
    _volume_input,
)
from ..candidate_a3d.geometry import VolumeGeometry3D
from .alignment import align_teacher_to_student4
from .physiology import patient_physiology_from_probabilities

_HEX = re.compile(r"^[0-9a-f]{64}$")
_CLASSES = ("BG", "RV", "MYO", "LV")


def _file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _state_sha256(module: nn.Module) -> str:
    from .engine import _state_sha256 as engine_state_sha256

    return engine_state_sha256(module)


def _require_digest(value: str, name: str) -> str:
    if not isinstance(value, str) or not _HEX.fullmatch(value):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _freeze(module: nn.Module, name: str) -> None:
    module.eval()
    module.requires_grad_(False)
    if any(parameter.requires_grad for parameter in module.parameters()):
        raise ValueError(f"{name} must be frozen for Q-3D diagnostics")


def _phase_frame(record: Mapping[str, Any], phase: str) -> int:
    """Resolve an explicitly annotated ED/ES frame without guessing."""
    aliases = (phase.lower(), phase.upper(), f"{phase.lower()}_frame", f"{phase.upper()}_frame")
    for key in aliases:
        value = record.get(key)
        if isinstance(value, int) and not isinstance(value, bool):
            return value
    phase_map = record.get("phases")
    if isinstance(phase_map, Mapping):
        for key in aliases:
            value = phase_map.get(key)
            if isinstance(value, int) and not isinstance(value, bool):
                return value
    masks = record.get("masks")
    if isinstance(masks, Mapping):
        matches = []
        for key, value in masks.items():
            if isinstance(value, Mapping) and str(value.get("phase", "")).upper() == phase:
                matches.append(int(key))
        if len(matches) == 1:
            return matches[0]
    raise ValueError(f"DEV patient {record.get('id')} must declare its {phase} mask frame")


def _pooled_teacher_outputs(
    logits: torch.Tensor, target_hw: tuple[int, int]
) -> tuple[torch.Tensor, torch.Tensor]:
    """Mirror engine S4 target pooling for class probabilities and confidence."""
    from torch.nn import functional as F

    probs = torch.softmax(logits.float(), dim=1)
    confidence = probs.max(dim=1, keepdim=True).values
    pad_h, pad_w = (-probs.shape[-2]) % 16, (-probs.shape[-1]) % 16
    if pad_h or pad_w:
        probs = F.pad(probs, (0, pad_w, 0, pad_h, 0, 0), mode="replicate")
        confidence = F.pad(confidence, (0, pad_w, 0, pad_h, 0, 0), mode="replicate")
    probs = F.avg_pool3d(probs, kernel_size=(1, 16, 16), stride=(1, 16, 16))
    confidence = F.avg_pool3d(confidence, kernel_size=(1, 16, 16), stride=(1, 16, 16))
    return probs[..., :target_hw[0], :target_hw[1]], confidence[..., :target_hw[0], :target_hw[1]]


def _teacher_with_source(teacher: nn.Module, image: torch.Tensor, geometry: Mapping[str, Any],
                         *, source: str, synthetic: bool) -> dict[str, torch.Tensor]:
    from .engine import _teacher_outputs

    # Use the same exact one-pass feature path as T/Q/J, including synthetic
    # hook padding and the verified MedNeXt decoder-block hook for real runs.
    return _teacher_outputs(
        teacher, image, geometry, with_feature=True, source=source, synthetic=synthetic,
    )


def _codebook_diagnostics(
    codes: Sequence[torch.Tensor], class_probs: Sequence[torch.Tensor], size: int,
    *, min_perplexity: float, max_dead_fraction: float, min_class_codes: int,
    dead_threshold: float,
) -> dict[str, Any]:
    if len(codes) != len(class_probs):
        raise ValueError("teacher code and class-probability frame counts differ")
    counts = torch.zeros(size, dtype=torch.float64)
    class_mass = torch.zeros((size, 4), dtype=torch.float64)
    for code, probabilities in zip(codes, class_probs):
        flat_codes = code.detach().cpu().reshape(-1).long()
        flat_probs = probabilities.detach().cpu().float().permute(0, 2, 3, 4, 1).reshape(-1, 4)
        if flat_codes.numel() != flat_probs.shape[0]:
            raise ValueError("teacher code lattice does not match pooled DEV class probabilities")
        counts += torch.bincount(flat_codes, minlength=size).to(torch.float64)
        class_mass.index_add_(0, flat_codes, flat_probs.to(torch.float64))
    total = float(counts.sum())
    if total <= 0:
        raise ValueError("DEV produced no teacher code assignments")
    usage = counts / total
    entropy = -(usage[usage > 0] * usage[usage > 0].log()).sum()
    perplexity = float(entropy.exp())
    dead_fraction = float((usage < dead_threshold).double().mean())
    mass_totals = class_mass.sum(dim=1)
    ownership_share = class_mass.max(dim=1).values / mass_totals.clamp_min(1e-12)
    ownership = torch.where(ownership_share > 0.5, class_mass.argmax(dim=1), -1)
    class_coverage = {
        _CLASSES[class_id]: int((ownership == class_id).sum())
        for class_id in (1, 2, 3)
    }
    gate = {
        "status": "PASS" if (
            perplexity >= min_perplexity
            and dead_fraction <= max_dead_fraction
            and all(value >= min_class_codes for value in class_coverage.values())
        ) else "FAIL",
        "perplexity_minimum": float(min_perplexity),
        "perplexity": perplexity,
        "dead_code_fraction_threshold": float(dead_threshold),
        "dead_code_fraction": dead_fraction,
        "dead_code_count": int((usage < dead_threshold).sum()),
        "class_code_minimum": int(min_class_codes),
        "class_code_ownership_threshold_strictly_greater_than": 0.5,
        "class_code_coverage": class_coverage,
        "assignment_count": int(total),
        "usage_source": "all_native_DEV_frames_teacher_hard_codes",
        "class_ownership_source": "teacher_soft_class_probability_mass_on_DEV_S4",
    }
    return gate


def run_dev_diagnostics(
    teacher: nn.Module,
    tokenizer: nn.Module | None,
    student: nn.Module | None,
    *,
    config: Mapping[str, Any],
    development_manifest: str | Path,
    development_sha256: str,
    train_patient_ids: Sequence[str],
    protocol: str,
    device: str | torch.device = "cpu",
    synthetic: bool = False,
) -> dict[str, Any]:
    """Run immutable full-native DEV diagnostics and report-only GT correlations.

    Image inference and physiology happen in a complete first pass. The DEV
    mask paths are not opened until that pass finishes. ED/ES masks then serve
    only to compute teacher RV Dice and HD95 for the Spearman report.
    """
    if protocol not in {"A", "B"}:
        raise ValueError("Q-3D protocol must be A or B")
    if config.get("method") != "candidate_q3d" or config.get("schema") != "self_audit.candidate_q3d.v1":
        raise ValueError("diagnostics require a hash-bound Q-3D v1 config")
    if config.get("protocol", {}).get("variant") != protocol:
        raise ValueError("diagnostic protocol differs from config identity")
    if config.get("runtime", {}).get("synthetic") is not bool(synthetic):
        raise ValueError("diagnostic real/synthetic setting differs from config")
    arm = str(config.get("arm", "")).lower()
    physiology_only = arm == "c1" and tokenizer is None
    if not physiology_only and tokenizer is None:
        raise ValueError("Q-3D code diagnostics require a frozen tokenizer; only C1 supports physiology-only diagnostics")
    if tokenizer is not None and not bool(config.get("tokenizer", {}).get("enabled", False)):
        raise ValueError("config disables the tokenizer but a tokenizer module was supplied")
    if physiology_only and not bool(config.get("physiology", {}).get("enabled", False)):
        raise ValueError("C1 physiology-only diagnostics require physiology.enabled=true")
    development_sha256 = _require_digest(development_sha256, "development_sha256")
    if not train_patient_ids or len(set(train_patient_ids)) != len(train_patient_ids):
        raise ValueError("checkpoint identity must declare unique TRAIN patient IDs")
    if tokenizer is not None and (not tokenizer.codebook_initialized or not tokenizer.codebook_frozen):
        raise ValueError("DEV diagnostics require the initialized frozen Q-stage tokenizer")

    modules = [(teacher, "teacher")]
    if tokenizer is not None:
        modules.append((tokenizer, "tokenizer"))
    if student is not None:
        modules.append((student, "student"))
    for module, name in modules:
        if any(parameter.requires_grad for parameter in module.parameters()):
            raise ValueError(f"{name} parameters must be frozen before read-only diagnostics")
        _freeze(module, name)
    before = {name: _state_sha256(module) for module, name in modules}

    path, manifest, patients, patient_ids, geometries, resolve = _validated_manifest(
        development_manifest, development_sha256, train_ids=train_patient_ids, synthetic=synthetic,
    )
    if manifest.get("role") != "development":
        raise ValueError("Q-3D diagnostics require role=development; TEST is forbidden")
    target_device = torch.device(device)
    teacher.to(target_device).eval()
    if tokenizer is not None:
        tokenizer.to(target_device).eval()
    if student is not None:
        student.to(target_device).eval()
    teacher_sha = before["teacher"]
    codebook_sha = getattr(tokenizer, "codebook_sha256", None) if tokenizer is not None else None
    protocol_config = config.get("physiology", {})

    patient_rows: list[dict[str, Any]] = []
    all_teacher_codes: list[torch.Tensor] = []
    all_class_probs: list[torch.Tensor] = []
    all_agreements: list[float] = []
    prediction_cache: dict[str, tuple[np.ndarray, dict[str, Any], list[np.ndarray]]] = {}

    with torch.inference_mode():
        for record, patient_id in zip(patients, patient_ids):
            image_path = resolve(record["image"])
            image = np.load(image_path, mmap_mode="r", allow_pickle=False)
            geometry = VolumeGeometry3D.from_cache_geometry(
                geometries[patient_id], expected_shape=image.shape[1:],
            ).as_dict()
            frames_probabilities: list[torch.Tensor] = []
            patient_agreement: list[float] = []
            frame_predictions: list[np.ndarray] = []
            for frame in range(int(image.shape[0])):
                x = _volume_input(image, frame, target_device)
                if physiology_only:
                    teacher_outputs = teacher(x, geometry=geometry)
                else:
                    feature_source = f"stride{config['model']['teacher']['feature_stride']}"
                    teacher_outputs = _teacher_with_source(
                        teacher, x, geometry, source=feature_source, synthetic=synthetic,
                    )
                region_logits = teacher_outputs["region_logits"]
                frame_predictions.append(region_logits.argmax(1)[0].detach().cpu().numpy())
                frames_probabilities.append(torch.softmax(region_logits.float(), dim=1)[0].cpu())
                if not physiology_only:
                    assert tokenizer is not None and student is not None
                    aligned = align_teacher_to_student4(
                        teacher_outputs["feature"], x.shape[-3:],
                        teacher_stride=int(config["model"]["teacher"]["feature_stride"]),
                    )
                    teacher_token = tokenizer(aligned)
                    student_output = student(x, geometry=geometry)
                    teacher_codes = teacher_token.get("codes")
                    student_codes = student_output.get("codes")
                    if teacher_codes is None or student_codes is None:
                        raise ValueError("Q-3D VQ diagnostics require teacher and student hard code indices")
                    if teacher_codes.shape != student_codes.shape:
                        raise ValueError("teacher/student code grids differ after exact S4 alignment")
                    patient_agreement.append(float((teacher_codes == student_codes).float().mean().cpu()))
                    probabilities, _confidence = _pooled_teacher_outputs(
                        region_logits, (int(teacher_codes.shape[-2]), int(teacher_codes.shape[-1])),
                    )
                    all_teacher_codes.append(teacher_codes.detach().cpu())
                    all_class_probs.append(probabilities.detach().cpu())
            physiology = patient_physiology_from_probabilities(
                torch.stack(frames_probabilities), geometry, patient_id=patient_id,
                teacher_sha256=teacher_sha, codebook_sha256=codebook_sha,
                protocol=protocol, config=protocol_config,
            )
            patient_rows.append({
                "patient_id": patient_id,
                "w_pat": physiology["w_pat"],
                "physiology": physiology,
                "teacher_student_hard_code_agreement": (
                    float(np.mean(patient_agreement)) if patient_agreement else None
                ),
            })
            all_agreements.extend(patient_agreement)
            prediction_cache[patient_id] = (image, geometry, frame_predictions)

    tokenizer_gate = (
        {"status": "NOT_APPLICABLE", "reason": "C1_has_no_tokenizer_or_codebook"}
        if physiology_only else
        _codebook_diagnostics(
            all_teacher_codes, all_class_probs, int(tokenizer.codebook_size),
            min_perplexity=float(config["tokenizer"]["minimum_perplexity"]),
            max_dead_fraction=float(config["tokenizer"]["maximum_dead_code_fraction"]),
            min_class_codes=int(config["tokenizer"]["minimum_class_codes"]),
            dead_threshold=float(config["tokenizer"]["dead_code_fraction"]),
        )
    )

    # Second pass: the only label reads in this tool. No metric is returned to
    # training or checkpoint-selection code.
    from .evaluation import _dice, surface_distances_3d

    diagnostic_metrics: list[dict[str, Any]] = []
    for record, patient_id in zip(patients, patient_ids):
        image, geometry, frame_predictions = prediction_cache[patient_id]
        mask_records = record.get("masks")
        if not isinstance(mask_records, Mapping):
            raise ValueError(f"DEV patient {patient_id} has no ED/ES mask mapping")
        phases: dict[str, Any] = {}
        for phase in ("ED", "ES"):
            frame = _phase_frame(record, phase)
            mask_record = mask_records.get(str(frame))
            if (not isinstance(mask_record, Mapping)
                    or set(mask_record) - {"path", "sha256", "phase"}
                    or not {"path", "sha256"}.issubset(mask_record)):
                raise ValueError(f"DEV {phase} mask must be declared with path and SHA-256")
            mask_path = resolve(mask_record["path"])
            if _file_sha256(mask_path) != mask_record["sha256"]:
                raise ValueError(f"DEV {phase} mask SHA-256 mismatch for {patient_id}")
            reference = np.load(mask_path, allow_pickle=False)
            if reference.shape != image.shape[1:] or not np.isin(reference, (0, 1, 2, 3)).all():
                raise ValueError("DEV ED/ES mask must cover the native BG/RV/MYO/LV volume")
            if not 0 <= frame < len(frame_predictions):
                raise ValueError(f"DEV {phase} frame is outside cached first-pass predictions")
            prediction = frame_predictions[frame]
            rv_dice = _dice(reference, prediction, 1)
            rv_surface = surface_distances_3d(
                reference == 1, prediction == 1, geometry["spacing_zyx"],
            )
            rv_hd95 = rv_surface["hd95_mm"]
            exclusion_reason = None
            if rv_hd95 is None:
                reference_rv_present = bool(np.any(reference == 1))
                teacher_rv_present = bool(np.any(prediction == 1))
                if not reference_rv_present and not teacher_rv_present:
                    exclusion_reason = "both_reference_and_teacher_rv_absent"
                elif not reference_rv_present:
                    exclusion_reason = "reference_rv_absent"
                elif not teacher_rv_present:
                    exclusion_reason = "teacher_rv_absent"
                else:
                    exclusion_reason = f"surface_metric_{rv_surface['status']}"
            phases[phase] = {
                "rv_dice": rv_dice,
                "rv_hd95_mm": rv_hd95,
                "rv_hd95_exclusion_reason": exclusion_reason,
            }
        diagnostic_metrics.append({
            "patient_id": patient_id,
            "w_pat": next(row["w_pat"] for row in patient_rows if row["patient_id"] == patient_id),
            "teacher_rv_dice": float(np.mean([phases[p]["rv_dice"] for p in phases if phases[p]["rv_dice"] is not None]))
            if any(phases[p]["rv_dice"] is not None for p in phases) else None,
            "teacher_rv_hd95_mm": float(np.mean([phases[p]["rv_hd95_mm"] for p in phases if phases[p]["rv_hd95_mm"] is not None]))
            if any(phases[p]["rv_hd95_mm"] is not None for p in phases) else None,
            "by_phase": phases,
        })

    usable = [row for row in diagnostic_metrics if row["teacher_rv_dice"] is not None]
    corr_rows = [{"w_pat": row["w_pat"], "rv_dice": row["teacher_rv_dice"]} for row in usable]
    dice_correlation = _spearman_report(corr_rows)
    h95_rows = [row for row in diagnostic_metrics if row["teacher_rv_hd95_mm"] is not None]
    hd95 = _spearman_report([
        {"w_pat": row["w_pat"], "rv_dice": row["teacher_rv_hd95_mm"]} for row in h95_rows
    ])
    hd95_exclusions: dict[str, int] = {}
    for row in diagnostic_metrics:
        for phase in ("ED", "ES"):
            reason = row["by_phase"][phase]["rv_hd95_exclusion_reason"]
            if reason is not None:
                hd95_exclusions[reason] = hd95_exclusions.get(reason, 0) + 1
    after = {name: _state_sha256(module) for module, name in modules}
    if before != after:
        raise RuntimeError("Q-3D diagnostic changed frozen teacher, tokenizer, or student state")

    return {
        "schema": "self_audit.candidate_q3d.dev_diagnostics.v1",
        "status": "COMPLETED",
        "scope": "development_only",
        "protocol": protocol,
        "dataset": manifest.get("dataset"),
        "development_manifest_sha256": development_sha256,
        "train_patient_ids_sha256": hashlib.sha256("\n".join(sorted(train_patient_ids)).encode()).hexdigest(),
        "teacher_state_sha256": before["teacher"],
        "tokenizer_state_sha256": before.get("tokenizer"),
        "student_state_sha256": before.get("student"),
        "teacher_student_hard_code_agreement": ({
            "status": "NOT_APPLICABLE",
            "reason": "C1_physiology_only_arm_has_no_VQ_codes",
        } if physiology_only else {
            "status": "COMPUTED",
            "mean": float(np.mean(all_agreements)) if all_agreements else None,
            "frame_count": len(all_agreements),
            "alignment": "teacher_features_hooked_then_engine_align_teacher_to_student4",
        }),
        "tokenizer_gate": tokenizer_gate,
        "physiology": {
            "source": "full_native_DEV_cine_teacher_probabilities_without_GT_or_ED_ES_metadata",
            "patients": patient_rows,
        },
        "teacher_rv_metric_correlations": {
            "split": "development",
            "purpose": "report_only_not_checkpoint_selection_or_training",
            "w_pat_vs_teacher_rv_dice": dice_correlation,
            "w_pat_vs_teacher_rv_hd95_mm": hd95,
            "w_pat_vs_teacher_rv_hd95_exclusions": {
                "excluded_phase_count": sum(hd95_exclusions.values()),
                "by_reason": hd95_exclusions,
            },
            "patient_metrics": diagnostic_metrics,
        },
        "teacher_predictions_reused_from_label_free_first_pass": True,
        "test_masks_opened": False,
        "training_or_checkpoint_selection_influence": False,
        "synthetic": bool(synthetic),
        "evidence_status": "SOFTWARE_ONLY_SYNTHETIC" if synthetic else "DEV_DIAGNOSTIC_ONLY",
    }


def diagnose_checkpoint(
    config_path: str | Path,
    checkpoint_path: str | Path,
    *,
    device: str | torch.device = "cpu",
    expected_checkpoint_sha256: str | None = None,
) -> dict[str, Any]:
    """Load a hash-bound research resume checkpoint and run DEV diagnostics."""
    config_path = Path(config_path).expanduser().resolve()
    checkpoint_path = Path(checkpoint_path).expanduser().resolve()
    from .config import load_config
    from .engine import (
        _build_models, _data_identity_without_labels, _load_checkpoint, _load_data, _object_sha256,
    )

    config = load_config(config_path, real_run=False)
    if config["runtime"]["synthetic"]:
        if device == "cuda":
            raise ValueError("synthetic diagnostics default to CPU; select another device explicitly")
    else:
        if config["runtime"].get("allow_unverified_teacher"):
            raise ValueError("diagnostic real config cannot allow an unverified teacher")
    sidecar = checkpoint_path.with_suffix(checkpoint_path.suffix + ".sha256")
    if not sidecar.is_file():
        raise ValueError("Q-3D diagnostic checkpoint requires its .sha256 sidecar")
    try:
        sidecar_fields = sidecar.read_text(encoding="ascii").split()
    except (OSError, UnicodeDecodeError) as exc:
        raise ValueError("Q-3D diagnostic checkpoint SHA-256 sidecar cannot be read") from exc
    if len(sidecar_fields) != 2 or sidecar_fields[1] != checkpoint_path.name:
        raise ValueError("Q-3D diagnostic checkpoint SHA-256 sidecar is malformed")
    sidecar_sha = sidecar_fields[0]
    _require_digest(sidecar_sha, "checkpoint sidecar digest")
    actual_sha = _file_sha256(checkpoint_path)
    if sidecar_sha != actual_sha:
        raise ValueError("Q-3D research checkpoint SHA-256 sidecar mismatch")
    if expected_checkpoint_sha256 is not None and _require_digest(expected_checkpoint_sha256, "expected_checkpoint_sha256") != actual_sha:
        raise ValueError("Q-3D checkpoint SHA-256 differs from the requested identity")
    config_sha = _object_sha256(config)
    data = _load_data(config, None)
    data_identity = _data_identity_without_labels(data)
    try:
        payload = _load_checkpoint(
            checkpoint_path, config_sha256=config_sha, data_identity=data_identity,
        )
    except ValueError as exc:
        if "schema mismatch" in str(exc):
            raise ValueError(
                "diagnostics require a Q-3D research resume checkpoint (schema resume.v2); "
                "student deployment exports are rejected"
            ) from exc
        raise
    train_ids = tuple(data.train_patients)
    completed = set(payload.get("completed_stages", ()))
    required_stages = {"J"}
    if config["model"]["teacher"]["architecture"] == "mednext_l_k3":
        required_stages.add("T")
    if config["tokenizer"]["enabled"]:
        required_stages.add("Q")
    if config["physiology"]["enabled"]:
        required_stages.add("P")
    if not required_stages.issubset(completed):
        raise ValueError(
            f"DEV diagnostic requires completed stages {sorted(required_stages)}; "
            f"checkpoint has {sorted(completed)}"
        )
    dev = config["dataset"]["manifests"]["development"]
    if not isinstance(dev, Mapping) or not dev.get("path") or not dev.get("sha256"):
        raise ValueError("Q-3D config must pin the development manifest path and SHA-256")
    device_obj = torch.device(device)
    teacher, student, tokenizer, _provenance = _build_models(
        config, device=device_obj, teacher=None, student=None, tokenizer=None,
    )
    if teacher is None:
        raise ValueError("Q-3D diagnostic requires a trained Q-3D teacher")
    if student is None:
        raise ValueError("Q-3D diagnostic checkpoint is missing the student model")
    if not config["tokenizer"]["enabled"] and str(config["arm"]).lower() != "c1":
        raise ValueError("physiology-only diagnostics without Q are supported only for C1")
    teacher_source = "teacher_state" if str(config["arm"]).lower() == "o1" else "best_teacher_state"
    teacher_state = payload.get(teacher_source)
    if teacher_state is None and teacher_source == "best_teacher_state":
        teacher_state = payload.get("teacher_state")
    student_state = payload.get("best_student_state") or payload.get("student_state")
    tokenizer_state = payload.get("tokenizer_state") if tokenizer is not None else None
    if not isinstance(teacher_state, Mapping) or not isinstance(student_state, Mapping):
        raise ValueError("research checkpoint lacks teacher/student states")
    if tokenizer is not None and not isinstance(tokenizer_state, Mapping):
        raise ValueError("research checkpoint lacks its Q-stage tokenizer state")
    teacher.load_state_dict(teacher_state, strict=True)
    student.load_state_dict(student_state, strict=True)
    if tokenizer is not None:
        tokenizer.load_state_dict(tokenizer_state, strict=True)
    # Checkpoint module loading is read-only and does not restore optimizers,
    # RNGs, sampler, or any mutable training engine state.
    if tokenizer is not None and not tokenizer.codebook_frozen:
        raise ValueError("research checkpoint tokenizer did not pass and freeze the Q gate")
    student_codebook = getattr(student, "codebook", None)
    if tokenizer is not None and isinstance(student_codebook, torch.Tensor) and not torch.equal(
        student_codebook.detach().cpu(), tokenizer.C0.detach().cpu(),
    ):
        raise ValueError("student interface codebook differs from the frozen teacher tokenizer C0")
    for module in (teacher, tokenizer, student):
        if module is None:
            continue
        module.eval().requires_grad_(False)
    return run_dev_diagnostics(
        teacher, tokenizer, student, config=config,
        development_manifest=dev["path"], development_sha256=dev["sha256"],
        train_patient_ids=train_ids, protocol=config["protocol"]["variant"],
        device=device_obj, synthetic=bool(config["runtime"]["synthetic"]),
    ) | {
        "checkpoint_sha256": actual_sha,
        "config_sha256": config_sha,
        "teacher_state_source": teacher_source,
    }


def _spearman_report(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Use no rho when fewer than three patients or either variable is constant."""
    values = [(float(row["w_pat"]), float(row["rv_dice"])) for row in rows
              if row.get("w_pat") is not None and row.get("rv_dice") is not None
              and math.isfinite(float(row["w_pat"])) and math.isfinite(float(row["rv_dice"]))]
    if len(values) < 3:
        return {"status": "NOT_RUN_INSUFFICIENT_PATIENTS", "n": len(values), "rho": None, "p_value": None}
    x, y = np.asarray(values, dtype=np.float64).T
    if np.unique(x).size < 2 or np.unique(y).size < 2:
        return {"status": "NOT_RUN_CONSTANT_INPUT", "n": len(values), "rho": None, "p_value": None}
    from scipy import stats

    result = stats.spearmanr(x, y)
    return {"status": "COMPUTED", "n": len(values),
            "rho": float(result.statistic), "p_value": float(result.pvalue)}
