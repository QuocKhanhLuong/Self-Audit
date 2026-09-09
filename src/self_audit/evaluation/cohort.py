"""Reusable patient-volume cohort aggregation for diagnostic and external runs."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch

from ..audit.semantics import METRIC_SPACE_VOLUME_RESIZED, macro_mean
from .volume_inference import COMPARISON_MODES, evaluate_comparison_modes, resolve_class_names, split_cases_by_phase


def _cohort_block(
    patient_blocks: Sequence[Mapping[str, Any]],
    *,
    class_names: Mapping[int, str],
) -> dict[str, Any]:
    """Aggregate per-patient Dice blocks into one cohort block, nan-aware."""

    if not patient_blocks:
        return {
            "patient_count": 0,
            "per_class_dice": {},
            "per_class_dice_named": {},
            "macro_dice": float("nan"),
            "macro_of_patient_macros": float("nan"),
        }
    per_class: dict[int, float] = {}
    for cls, name in class_names.items():
        values = [
            float(block["per_class_dice"].get(cls, block["per_class_dice"].get(str(cls), float("nan"))))
            for block in patient_blocks
        ]
        per_class[int(cls)] = macro_mean(values)
    patient_macros = [float(block["macro_dice"]) for block in patient_blocks]
    return {
        "patient_count": len(patient_blocks),
        "per_class_dice": {int(cls): value for cls, value in per_class.items()},
        "per_class_dice_named": {class_names[cls]: value for cls, value in per_class.items()},
        "macro_dice": macro_mean(list(per_class.values())),
        "macro_of_patient_macros": macro_mean(patient_macros),
    }


@torch.no_grad()
def evaluate_volume_cohort(
    model: torch.nn.Module,
    dataset: Any,
    device: torch.device,
    *,
    tau_accept: float,
    t_max: int,
    neutral_margin: float,
    empty_policy: str,
    image_size: int = 256,
    num_classes: int = 4,
    batch_size: int = 8,
    max_volumes: int | None = None,
    geometry_by_case: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Evaluate all comparison modes and aggregate by case, patient, and phase.

    Phase grouping is delegated to :func:`split_cases_by_phase`; callers that
    have no authoritative ED/ES metadata should use its ``unknown`` and
    ``all`` cohorts and must not relabel timepoints heuristically.
    """

    case_ids = [record.case_id for record in getattr(dataset, "records", [])]
    if not case_ids:
        raise ValueError("Volume evaluation requires a dataset exposing .records with case ids")
    phases = split_cases_by_phase(case_ids)
    names = resolve_class_names(None, num_classes=num_classes)
    geometry_by_case = dict(geometry_by_case or {})

    ordered = list(case_ids)
    if max_volumes is not None:
        ordered = ordered[: int(max_volumes)]
    selected = set(ordered)

    patients: dict[str, dict[str, Any]] = {}
    for case_id in ordered:
        volume, mask, spacing = dataset.get_volume(case_id)
        geometry = dict(geometry_by_case.get(case_id, {}))
        if spacing is not None and "dataset_spacing" not in geometry:
            geometry["dataset_spacing"] = [float(value) for value in spacing]
        results = evaluate_comparison_modes(
            model,
            volume,
            mask,
            image_size=int(image_size),
            depth_axis=0,
            tau_accept=float(tau_accept),
            t_max=int(t_max),
            device=device,
            batch_size=int(batch_size),
            empty_policy=empty_policy,
            neutral_margin=float(neutral_margin),
            num_classes=int(num_classes),
            geometry=geometry or None,
        )
        patients[case_id] = {
            "metric_space": METRIC_SPACE_VOLUME_RESIZED,
            "geometry_declared": geometry or None,
            "modes": {
                mode: {
                    "per_class_dice": results[mode]["volume_dice"]["per_class_dice"],
                    "per_class_dice_named": results[mode]["volume_dice"]["per_class_dice_named"],
                    "macro_dice": results[mode]["volume_dice"]["macro_dice"],
                    "excluded_classes": results[mode]["volume_dice"]["excluded_classes"],
                    "macro_dice_delta_vs_initial": results[mode]["macro_dice_delta_vs_initial"],
                }
                for mode in COMPARISON_MODES
            },
            "evaluation_meta": results["evaluation_meta"],
        }

    cohorts: dict[str, Any] = {}
    for phase_name in ("ED", "ES", "unknown", "all"):
        if phase_name == "all":
            members = [case for case in ordered if case in selected]
        else:
            members = [case for case in phases.get(phase_name, []) if case in selected]
        cohorts[phase_name] = {
            "case_ids": members,
            "modes": {
                mode: _cohort_block(
                    [patients[case]["modes"][mode] for case in members],
                    class_names=names,
                )
                for mode in COMPARISON_MODES
            },
        }

    return {
        "metric_space": METRIC_SPACE_VOLUME_RESIZED,
        "native_dice_available": False,
        "native_dice_note": (
            "volume_native Dice is NOT derivable from preprocessed_data/: masks are already resized "
            "to the network grid destructively. Use raw labels with evaluate_volume_native for native metrics."
        ),
        "class_names": {int(cls): name for cls, name in names.items()},
        "empty_policy": empty_policy,
        "tau_accept": float(tau_accept),
        "t_max": int(t_max),
        "volumes_evaluated": len(patients),
        "volumes_available": len(case_ids),
        "phase_counts": {name: len(values) for name, values in phases.items()},
        "patients": patients,
        "cohorts": cohorts,
    }


__all__ = ["evaluate_volume_cohort"]
