"""Subject-level split utilities shared by all cardiac dataset adapters."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping
from typing import Any

import numpy as np


def _normalise_case_subject(case_to_subject: Mapping[Any, Any]) -> dict[str, str]:
    if not isinstance(case_to_subject, Mapping) or not case_to_subject:
        raise ValueError("case_to_subject must be a non-empty mapping")
    normalised: dict[str, str] = {}
    for raw_case, raw_subject in case_to_subject.items():
        case_id = str(raw_case).strip()
        subject_id = str(raw_subject).strip()
        if not case_id or not subject_id:
            raise ValueError("case and subject IDs must not be blank")
        normalised[case_id] = subject_id
    return normalised


def subject_level_split(
    case_to_subject: Mapping[Any, Any],
    *,
    train_fraction: float = 0.8,
    val_fraction: float = 0.1,
    seed: int = 42,
) -> dict[str, list[str]]:
    """Split complete cases by subject with deterministic membership."""

    if not 0.0 < float(train_fraction) < 1.0:
        raise ValueError("train_fraction must be between zero and one")
    if not 0.0 <= float(val_fraction) < 1.0:
        raise ValueError("val_fraction must be between zero and one")
    if float(train_fraction) + float(val_fraction) >= 1.0:
        raise ValueError("train_fraction + val_fraction must be less than one")

    mapping = _normalise_case_subject(case_to_subject)
    grouped: dict[str, list[str]] = defaultdict(list)
    for case_id, subject_id in sorted(mapping.items()):
        grouped[subject_id].append(case_id)
    subjects = sorted(grouped)
    if len(subjects) < 3:
        raise ValueError("subject-level train/val/test split requires at least three subjects")

    rng = np.random.default_rng(int(seed))
    shuffled = list(subjects)
    rng.shuffle(shuffled)
    n_train = min(max(int(round(len(subjects) * float(train_fraction))), 1), len(subjects) - 2)
    n_val = min(
        max(int(round(len(subjects) * float(val_fraction))), 1),
        len(subjects) - n_train - 1,
    )
    subject_splits = {
        "train": shuffled[:n_train],
        "val": shuffled[n_train : n_train + n_val],
        "test": shuffled[n_train + n_val :],
    }
    result = {
        split: sorted(case_id for subject in split_subjects for case_id in grouped[subject])
        for split, split_subjects in subject_splits.items()
    }
    validate_subject_split(result, mapping)
    return result


def validate_subject_split(
    case_ids_by_split: Mapping[str, Iterable[Any]],
    case_to_subject: Mapping[Any, Any],
) -> None:
    """Validate case membership and prove subject disjointness."""

    mapping = _normalise_case_subject(case_to_subject)
    seen_cases: dict[str, str] = {}
    seen_subjects: dict[str, str] = {}
    for raw_split, raw_case_ids in case_ids_by_split.items():
        split = str(raw_split)
        for raw_case in raw_case_ids:
            case_id = str(raw_case)
            if case_id not in mapping:
                raise ValueError(f"case {case_id!r} is not present in case_to_subject")
            previous_case_split = seen_cases.get(case_id)
            if previous_case_split is not None and previous_case_split != split:
                raise ValueError(
                    f"case {case_id!r} appears in multiple splits: "
                    f"{previous_case_split!r} and {split!r}"
                )
            seen_cases[case_id] = split
            subject_id = mapping[case_id]
            previous_subject_split = seen_subjects.get(subject_id)
            if previous_subject_split is not None and previous_subject_split != split:
                raise ValueError(
                    f"subject {subject_id!r} appears in multiple splits: "
                    f"{previous_subject_split!r} and {split!r}"
                )
            seen_subjects[subject_id] = split
