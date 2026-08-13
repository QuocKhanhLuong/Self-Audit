"""ACDC adapter for the locked 2.5-D center-slice contract."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from .common import (
    CLASS_NAMES,
    NUM_CLASSES,
    VolumeRecord,
    VolumeSliceDataset,
    patient_id_from_case_id,
    patient_level_split,
    read_split_manifest,
    validate_patient_split,
)


def _strip_archive_suffix(path: Path) -> str:
    name = path.name
    for suffix in (".nii.gz", ".nii", ".npy", ".npz"):
        if name.lower().endswith(suffix):
            return name[: -len(suffix)]
    return path.stem


def _split_aliases(split: str | None) -> tuple[str, ...]:
    if split is None:
        return ()
    normalized = str(split).lower()
    return {
        "train": ("train", "training"),
        "training": ("training", "train"),
        "val": ("val", "validation", "valid"),
        "validation": ("validation", "val", "valid"),
        "test": ("test", "testing"),
        "testing": ("testing", "test"),
    }.get(normalized, (normalized,))


def _paired_files(root: Path) -> list[tuple[Path, Path, str]]:
    pairs: list[tuple[Path, Path, str]] = []
    for volume_dir in (root / "volumes", root / "images", root / "image"):
        if not volume_dir.is_dir():
            continue
        mask_dir = next((root / name for name in ("masks", "labels", "segmentations", "mask") if (root / name).is_dir()), None)
        if mask_dir is None:
            continue
        volume_files = {p.stem: p for p in volume_dir.iterdir() if p.is_file() and p.suffix.lower() in {".npy", ".npz"}}
        mask_files = {p.stem: p for p in mask_dir.iterdir() if p.is_file() and p.suffix.lower() in {".npy", ".npz"}}
        missing = sorted(set(volume_files) ^ set(mask_files))
        if missing:
            raise FileNotFoundError(f"ACDC volume/mask pairing mismatch under {root}: {missing[:5]}")
        pairs.extend((volume_files[key], mask_files[key], key) for key in sorted(volume_files))
    return pairs


def _raw_nifti_pairs(root: Path) -> list[tuple[Path, Path, str]]:
    pairs: list[tuple[Path, Path, str]] = []
    masks = sorted(root.rglob("*_gt.nii")) + sorted(root.rglob("*_gt.nii.gz"))
    for mask in masks:
        image_name = mask.name.replace("_gt.nii.gz", ".nii.gz").replace("_gt.nii", ".nii")
        image = mask.with_name(image_name)
        if not image.exists():
            raise FileNotFoundError(f"ACDC mask has no matching image: {mask}")
        pairs.append((image, mask, _strip_archive_suffix(image)))
    return pairs


def _metadata_spacing(root: Path) -> dict[str, tuple[float, ...]]:
    metadata_path = root / "metadata.json"
    if not metadata_path.exists():
        return {}
    try:
        with open(metadata_path, encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, ValueError):
        return {}
    values: dict[str, tuple[float, ...]] = {}
    for case_id, info in (payload.get("volume_info", {}) or {}).items():
        if not isinstance(info, dict):
            continue
        spacing = info.get("effective_spacing") or info.get("orig_spacing")
        if isinstance(spacing, (list, tuple)) and len(spacing) >= 3:
            values[str(case_id)] = tuple(float(value) for value in spacing[:3])
    return values


def discover_acdc_records(data_root: str | Path, split: str | None = None) -> list[VolumeRecord]:
    """Discover paired ACDC records without slicing across patient boundaries."""

    root = Path(data_root)
    if not root.exists():
        raise FileNotFoundError(
            f"ACDC data root does not exist: {root}. Provide raw ACDC or a preprocessed "
            "root containing paired volumes/ and masks/."
        )
    candidate_roots: list[tuple[Path, str]] = []
    aliases = _split_aliases(split)
    if aliases:
        candidate_roots.extend((root / alias, alias) for alias in aliases if (root / alias).exists())
    if not candidate_roots:
        candidate_roots.append((root, ""))
        if split is None:
            for child in sorted(root.iterdir()):
                if child.is_dir() and child.name.lower() in {"train", "training", "val", "validation", "test", "testing"}:
                    candidate_roots.append((child, child.name.lower()))

    records: list[VolumeRecord] = []
    seen: set[str] = set()
    for candidate, split_name in candidate_roots:
        spacing_by_case = _metadata_spacing(candidate)
        pairs = _paired_files(candidate)
        if not pairs:
            pairs = _raw_nifti_pairs(candidate)
        for image_path, mask_path, case_id in pairs:
            if case_id in seen:
                continue
            seen.add(case_id)
            records.append(
                VolumeRecord(
                    case_id=case_id,
                    patient_id=patient_id_from_case_id(case_id),
                    image_path=image_path,
                    mask_path=mask_path,
                    split=split_name,
                    spacing=spacing_by_case.get(case_id),
                    source_format="nifti" if image_path.name.lower().endswith((".nii", ".nii.gz")) else "npy",
                )
            )
    if not records:
        raise FileNotFoundError(
            f"No paired ACDC volumes found under {root}. Expected volumes/ + masks/ "
            "or raw patient*/ *_gt.nii(.gz) files."
        )
    return sorted(records, key=lambda record: record.case_id)


def discover_acdc_cases(data_root: str | Path, split: str | None = None) -> list[str]:
    return [record.case_id for record in discover_acdc_records(data_root, split=split)]


def resolve_acdc_records(
    records: Sequence[VolumeRecord],
    *,
    split: str | None,
    split_manifest: str | Path | None = "splits/acdc_patient_split_seed42.json",
    seed: int = 42,
) -> list[VolumeRecord]:
    if split is None:
        return list(records)
    normalized = {"training": "train", "validation": "val", "testing": "test"}.get(str(split).lower(), str(split).lower())
    explicit = [record for record in records if record.split and record.split.lower() in _split_aliases(normalized)]
    if explicit:
        return explicit
    manifest_path = Path(split_manifest) if split_manifest is not None else None
    if manifest_path is not None and manifest_path.exists():
        manifest = read_split_manifest(manifest_path)
        if normalized not in manifest:
            raise ValueError(f"ACDC split manifest has no {normalized!r} split: {manifest_path}")
        wanted = set(manifest[normalized])
        selected = [record for record in records if record.case_id in wanted]
        if not selected:
            # A checked-in manifest may belong to the repository's real data
            # root while a caller is running a synthetic or alternate root.
            # Do not fabricate membership from it; fall through to a fresh
            # deterministic patient split for the discovered records.
            pass
        else:
            return selected
    split_ids = patient_level_split([record.case_id for record in records], seed=seed)
    if normalized not in split_ids:
        raise ValueError(f"Unknown ACDC split {split!r}")
    wanted = set(split_ids[normalized])
    return [record for record in records if record.case_id in wanted]


class ACDCDataset(VolumeSliceDataset):
    """Expose ACDC as ``image [3,H,W]`` and ``mask [H,W]`` samples."""

    class_names = CLASS_NAMES
    num_classes = NUM_CLASSES

    def __init__(
        self,
        data_root: str | Path = "preprocessed_data/ACDC",
        *,
        split: str | None = None,
        case_ids: Iterable[str] | None = None,
        records: Sequence[VolumeRecord] | None = None,
        split_manifest: str | Path | None = "splits/acdc_patient_split_seed42.json",
        seed: int = 42,
        image_size: int = 256,
        augment: bool = False,
        transform: object | None = None,
        foreground_only: bool = False,
        **kwargs: object,
    ) -> None:
        all_records = list(records) if records is not None else discover_acdc_records(data_root)
        selected = resolve_acdc_records(all_records, split=split, split_manifest=split_manifest, seed=seed)
        if case_ids is not None:
            wanted = set(str(value) for value in case_ids)
            missing = sorted(wanted - {record.case_id for record in all_records})
            if missing:
                raise FileNotFoundError(f"Requested ACDC cases are missing: {missing[:5]}")
            selected = [record for record in selected if record.case_id in wanted]
        if not selected:
            raise ValueError(f"ACDC split {split!r} selected zero cases")
        super().__init__(
            selected,
            image_size=image_size,
            augment=augment,
            transform=transform,
            foreground_only=foreground_only,
            **{key: value for key, value in kwargs.items() if key in {"lower_percentile", "upper_percentile", "max_cache"}},
        )

        # Keep an explicit identity mapping even though ACDC already uses the
        # target ids.  This makes the contract inspectable by training code.
        self.raw_to_acdc = {0: 0, 1: 1, 2: 2, 3: 3}

    def get_volume(self, case_id: str) -> tuple[object, object, tuple[float, ...] | None]:
        for index, record in enumerate(self.records):
            if record.case_id == case_id:
                return self._load(index)
        raise KeyError(f"Unknown ACDC case_id: {case_id}")

    def _load(self, record_index: int):
        volume, mask, spacing = super()._load(record_index)
        unique = set(int(value) for value in np.unique(mask))
        unknown = sorted(unique - {0, 1, 2, 3})
        if unknown:
            raise ValueError(f"ACDC mask contains labels outside the locked 0..3 contract: {unknown}")
        return volume, mask, spacing


ACDCSelfAuditDataset = ACDCDataset
