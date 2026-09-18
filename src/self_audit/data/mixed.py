"""Dataset-balanced and subject-balanced composition for four cardiac sources."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterator, Mapping, Sequence
from typing import Any

import numpy as np
from torch.utils.data import Dataset, Sampler


class MixedCardiacDataset(Dataset):
    """Concatenate source datasets while preserving source/subject identity."""

    def __init__(self, datasets: Mapping[str, Dataset]) -> None:
        if not datasets:
            raise ValueError("MixedCardiacDataset requires at least one source dataset")
        self.datasets = {str(name): dataset for name, dataset in datasets.items()}
        self._sources = list(self.datasets)
        self._offsets: dict[str, int] = {}
        offset = 0
        for name, dataset in self.datasets.items():
            self._offsets[name] = offset
            offset += len(dataset)
        self._length = offset
        if self._length == 0:
            raise ValueError("MixedCardiacDataset sources contain no samples")
        self.training_sampler: Sampler[int] | None = None

    def __len__(self) -> int:
        return self._length

    def _locate(self, index: int) -> tuple[str, int]:
        value = int(index)
        if value < 0 or value >= self._length:
            raise IndexError(value)
        for name in self._sources:
            start = self._offsets[name]
            stop = start + len(self.datasets[name])
            if start <= value < stop:
                return name, value - start
        raise IndexError(value)

    def __getitem__(self, index: int) -> dict[str, Any]:
        source, local_index = self._locate(index)
        sample = dict(self.datasets[source][local_index])
        raw_case = str(sample.get("case_id", local_index))
        raw_subject = str(sample.get("subject_id", sample.get("patient_id", "")))
        sample["dataset"] = source
        sample["source_dataset"] = source
        sample["case_id"] = f"{source}:{raw_case}"
        sample["subject_id"] = f"{source}:{raw_subject}"
        sample["patient_id"] = sample["subject_id"]
        sample["source_case_id"] = raw_case
        return sample

    def subject_groups(self) -> dict[tuple[str, str], list[int]]:
        groups: dict[tuple[str, str], list[int]] = defaultdict(list)
        for source, dataset in self.datasets.items():
            for local_index in range(len(dataset)):
                sample = dataset[local_index]
                subject = str(sample.get("subject_id", sample.get("patient_id", "")))
                groups[(source, subject)].append(self._offsets[source] + local_index)
        return dict(groups)


class DatasetSubjectBalancedSampler(Sampler[int]):
    """Draw equal mass per dataset, then per subject, then per slice sample."""

    def __init__(
        self,
        dataset: MixedCardiacDataset,
        *,
        seed: int = 42,
        epoch_size: int | None = None,
    ) -> None:
        self.dataset = dataset
        self.seed = int(seed)
        self.epoch_size = int(epoch_size or len(dataset))
        if self.epoch_size < 1:
            raise ValueError("epoch_size must be positive")
        groups = dataset.subject_groups()
        self.groups: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
        for (source, subject), indices in groups.items():
            self.groups[source][subject].extend(indices)
        self.sources = sorted(self.groups)
        if not self.sources:
            raise ValueError("cannot sample a mixed dataset with no subject groups")
        self._epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self._epoch = int(epoch)

    def __len__(self) -> int:
        return self.epoch_size

    def __iter__(self) -> Iterator[int]:
        rng = np.random.default_rng(self.seed + self._epoch)
        for _ in range(self.epoch_size):
            source = str(rng.choice(self.sources))
            subjects = sorted(self.groups[source])
            subject = str(rng.choice(subjects))
            indices = self.groups[source][subject]
            yield int(rng.choice(indices))
