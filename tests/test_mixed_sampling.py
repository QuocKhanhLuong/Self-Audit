from __future__ import annotations

from collections import Counter

import torch
from torch.utils.data import Dataset

try:
    from self_audit.data.mixed import DatasetSubjectBalancedSampler, MixedCardiacDataset
except ImportError:
    from src.self_audit.data.mixed import DatasetSubjectBalancedSampler, MixedCardiacDataset


class _Samples(Dataset):
    def __init__(self, dataset: str, subjects: list[str]) -> None:
        self.samples = [
            {
                "image": torch.zeros(3, 2, 2),
                "mask": torch.zeros(2, 2, dtype=torch.long),
                "dataset": dataset,
                "case_id": f"{subject}-case-{index}",
                "subject_id": subject,
            }
            for index, subject in enumerate(subjects)
        ]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        return self.samples[index]


def test_mixed_dataset_namespaces_source_and_subject_ids() -> None:
    dataset = MixedCardiacDataset(
        {"acdc": _Samples("acdc", ["P1"]), "mnms": _Samples("mnms", ["P2"])}
    )
    sample = dataset[0]
    assert sample["case_id"].startswith("acdc:")
    assert sample["subject_id"] == "acdc:P1"
    assert sample["source_case_id"] == "P1-case-0"


def test_subject_dataset_sampler_is_deterministic_and_balanced() -> None:
    dataset = MixedCardiacDataset(
        {
            "small": _Samples("small", ["S1", "S2"]),
            "large": _Samples("large", ["L1"] * 20),
        }
    )
    sampler = DatasetSubjectBalancedSampler(dataset, seed=7, epoch_size=400)
    first = list(iter(sampler))
    second = list(iter(DatasetSubjectBalancedSampler(dataset, seed=7, epoch_size=400)))
    assert first == second
    sources = Counter(dataset[index]["dataset"] for index in first)
    assert abs(sources["small"] - sources["large"]) < 80
    selected_subjects = Counter(dataset[index]["subject_id"] for index in first)
    assert selected_subjects["large:L1"] > 0
