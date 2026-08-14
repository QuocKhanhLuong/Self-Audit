"""M&Ms external-domain adapter with an explicit semantic class mapping."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np

from .acdc import _strip_archive_suffix
from .common import (
    CLASS_NAMES,
    NUM_CLASSES,
    VolumeRecord,
    VolumeSliceDataset,
    load_array,
    patient_id_from_case_id,
)


# M&Ms releases commonly encode LV/MYO/RV as 1/2/3, whereas the locked ACDC
# contract is RV/MYO/LV as 1/2/3.  This is deliberately a named, inspectable
# mapping rather than an implicit assumption in the dataset implementation.
DEFAULT_MNMS_TO_ACDC = {0: 0, 1: 3, 2: 2, 3: 1}


@dataclass(frozen=True)
class MNMSClassMapping:
    raw_to_acdc: Mapping[int, int]
    source_name: str = "M&Ms explicit LV/MYO/RV mapping"

    def __post_init__(self) -> None:
        mapping = {int(key): int(value) for key, value in self.raw_to_acdc.items()}
        if 0 not in mapping or mapping[0] != 0:
            raise ValueError("M&Ms class mapping must explicitly map raw background 0 to ACDC background 0")
        values = set(mapping.values())
        if not values.issubset(set(range(NUM_CLASSES))):
            raise ValueError(f"M&Ms mapping targets must be in 0..{NUM_CLASSES - 1}, got {values}")
        object.__setattr__(self, "raw_to_acdc", mapping)

    def apply(self, labels: np.ndarray) -> np.ndarray:
        labels = np.asarray(labels)
        unique = set(int(value) for value in np.unique(labels))
        unknown = sorted(unique - set(self.raw_to_acdc))
        if unknown:
            raise ValueError(
                f"M&Ms mask contains raw labels without an explicit class mapping: {unknown}. "
                f"Configured mapping: {dict(self.raw_to_acdc)}"
            )
        mapped = np.zeros_like(labels, dtype=np.int64)
        for raw, target in self.raw_to_acdc.items():
            mapped[labels == int(raw)] = int(target)
        return mapped


def map_mnms_labels(labels: np.ndarray, raw_to_acdc: Mapping[int, int] | MNMSClassMapping) -> np.ndarray:
    mapping = raw_to_acdc if isinstance(raw_to_acdc, MNMSClassMapping) else MNMSClassMapping(raw_to_acdc)
    return mapping.apply(labels)


def _find_pair_dirs(root: Path) -> list[tuple[Path, Path]]:
    pairs: list[tuple[Path, Path]] = []
    image_names = ("images", "image", "volumes", "volume")
    mask_names = ("masks", "mask", "labels", "label", "segmentations", "segs")
    for image_name in image_names:
        image_dir = root / image_name
        if not image_dir.is_dir():
            continue
        for mask_name in mask_names:
            mask_dir = root / mask_name
            if mask_dir.is_dir():
                pairs.append((image_dir, mask_dir))
    return pairs


def _supported_volume_file(path: Path) -> bool:
    name = path.name.lower()
    return name.endswith((".npy", ".npz", ".nii", ".nii.gz"))


def _volume_key(path: Path) -> str:
    return _strip_archive_suffix(path)


def discover_mnms_records(data_root: str | Path, split: str | None = None) -> list[VolumeRecord]:
    root = Path(data_root)
    if not root.exists():
        raise FileNotFoundError(
            f"M&Ms data root does not exist: {root}. Provide the external dataset explicitly; "
            "the domain-shift protocol never substitutes ACDC data."
        )
    roots = [root]
    if split is not None:
        aliases = {"train": ("train", "training"), "val": ("val", "validation"), "test": ("test", "testing")}.get(str(split).lower(), (str(split),))
        roots = [root / alias for alias in aliases if (root / alias).exists()]
        if not roots:
            raise FileNotFoundError(f"M&Ms split directory {split!r} not found under {root}")
    records: list[VolumeRecord] = []
    seen: set[str] = set()
    for candidate in roots:
        for image_dir, mask_dir in _find_pair_dirs(candidate):
            image_files = {
                _volume_key(p): p
                for p in image_dir.iterdir()
                if p.is_file() and _supported_volume_file(p)
            }
            mask_files = {
                _volume_key(p): p
                for p in mask_dir.iterdir()
                if p.is_file() and _supported_volume_file(p)
            }
            for image_key, image_path in sorted(image_files.items()):
                candidates = [image_key, f"{image_key}_gt", f"{image_key}_label", f"{image_key}_seg"]
                mask_path = next((mask_files[key] for key in candidates if key in mask_files), None)
                if mask_path is None:
                    continue
                case_id = _strip_archive_suffix(image_path)
                if case_id in seen:
                    continue
                seen.add(case_id)
                records.append(VolumeRecord(case_id, patient_id_from_case_id(case_id), image_path, mask_path, split or "", source_format="nifti" if image_path.name.lower().endswith((".nii", ".nii.gz")) else "npy"))
        if not records:
            masks = sorted(candidate.rglob("*_gt.nii")) + sorted(candidate.rglob("*_gt.nii.gz"))
            for mask_path in masks:
                image_name = mask_path.name.replace("_gt.nii.gz", ".nii.gz").replace("_gt.nii", ".nii")
                image_path = mask_path.with_name(image_name)
                if not image_path.exists():
                    raise FileNotFoundError(f"M&Ms mask has no matching image: {mask_path}")
                case_id = _strip_archive_suffix(image_path)
                records.append(VolumeRecord(case_id, patient_id_from_case_id(case_id), image_path, mask_path, split or "", source_format="nifti"))
    if not records:
        raise FileNotFoundError(
            f"No paired M&Ms image/mask volumes found under {root}. Expected image(s)/mask(s) or paired NIfTI files."
        )
    return sorted(records, key=lambda record: record.case_id)


def discover_mnms_cases(data_root: str | Path, split: str | None = None) -> list[str]:
    return [record.case_id for record in discover_mnms_records(data_root, split=split)]


class MNMSDataset(VolumeSliceDataset):
    """M&Ms samples normalized to the ACDC-compatible four-class contract."""

    class_names = CLASS_NAMES
    num_classes = NUM_CLASSES

    def __init__(
        self,
        data_root: str | Path = "data/MnMs",
        *,
        split: str | None = None,
        case_ids: Iterable[str] | None = None,
        records: Sequence[VolumeRecord] | None = None,
        raw_to_acdc: Mapping[int, int] | None = None,
        class_mapping: MNMSClassMapping | None = None,
        image_size: int = 256,
        augment: bool = False,
        transform: object | None = None,
        foreground_only: bool = False,
        depth_axis: int | None = None,
        expected_slices: int | None = None,
        **kwargs: object,
    ) -> None:
        all_records = list(records) if records is not None else discover_mnms_records(data_root, split=split)
        wanted = set(str(value) for value in case_ids) if case_ids is not None else None
        if wanted is not None:
            missing = sorted(wanted - {record.case_id for record in all_records})
            if missing:
                raise FileNotFoundError(f"Requested M&Ms cases are missing: {missing[:5]}")
            all_records = [record for record in all_records if record.case_id in wanted]
        if not all_records:
            raise ValueError("M&Ms selection contains zero cases")
        self.class_mapping = class_mapping or MNMSClassMapping(raw_to_acdc or DEFAULT_MNMS_TO_ACDC)
        self.raw_to_acdc = dict(self.class_mapping.raw_to_acdc)
        super().__init__(
            all_records,
            image_size=image_size,
            augment=augment,
            transform=transform,
            foreground_only=foreground_only,
            depth_axis=depth_axis,
            expected_slices=expected_slices,
            **{
                key: value
                for key, value in kwargs.items()
                if key in {"lower_percentile", "upper_percentile", "max_cache"}
            },
        )

    def _load(self, record_index: int):
        volume, raw_mask, spacing = super()._load(record_index)
        return volume, self.class_mapping.apply(raw_mask), spacing

    def get_volume(self, case_id: str):
        for index, record in enumerate(self.records):
            if record.case_id == case_id:
                return self._load(index)
        raise KeyError(f"Unknown M&Ms case_id: {case_id}")


MNMsDataset = MNMSDataset
