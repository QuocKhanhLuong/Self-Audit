from __future__ import annotations

import pytest

try:
    from self_audit.data.splits import subject_level_split, validate_subject_split
except ImportError:
    from src.self_audit.data.splits import subject_level_split, validate_subject_split


def _cases() -> dict[str, str]:
    return {
        "P001_ED": "P001",
        "P001_ES": "P001",
        "P002_ED": "P002",
        "P003_ED": "P003",
        "P004_ED": "P004",
        "P005_ED": "P005",
        "P006_ED": "P006",
    }


def test_subject_level_split_is_deterministic_and_keeps_subject_cases_together() -> None:
    mapping = _cases()
    first = subject_level_split(mapping, train_fraction=0.5, val_fraction=0.25, seed=42)
    second = subject_level_split(mapping, train_fraction=0.5, val_fraction=0.25, seed=42)
    assert first == second
    validate_subject_split(first, mapping)
    for split_cases in first.values():
        subjects = {mapping[case_id] for case_id in split_cases}
        assert sum(case_id in split_cases for case_id in ("P001_ED", "P001_ES")) in (0, 2)
        assert subjects == {mapping[case_id] for case_id in split_cases}


def test_subject_level_split_rejects_too_few_subjects() -> None:
    with pytest.raises(ValueError, match="at least three subjects"):
        subject_level_split({"a": "P001", "b": "P002"}, seed=1)


def test_subject_level_split_rejects_duplicate_or_unknown_membership() -> None:
    mapping = _cases()
    with pytest.raises(ValueError, match="multiple splits"):
        validate_subject_split(
            {"train": ["P001_ED"], "val": ["P001_ES"], "test": []},
            mapping,
        )
    with pytest.raises(ValueError, match="not present"):
        validate_subject_split(
            {"train": ["P999"], "val": [], "test": []},
            mapping,
        )
