"""ACDC adapter for the locked 2.5-D center-slice contract."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from .common import (
    CANONICAL_SPLITS,
    CLASS_NAMES,
    NUM_CLASSES,
    SUPPORTED_SPLIT_ALIASES,
    VolumeRecord,
    VolumeSliceDataset,
    canonicalize_split_alias,
    compute_split_signature,
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
        volume_files: dict[str, Path] = {}
        for p in volume_dir.iterdir():
            if p.is_file() and p.suffix.lower() in {".npy", ".npz"}:
                stem = p.stem
                if stem in volume_files:
                    raise ValueError(
                        f"Collision in {volume_dir}: multiple files with identical stem {stem!r}: "
                        f"{volume_files[stem].name} and {p.name}"
                    )
                volume_files[stem] = p

        mask_files: dict[str, Path] = {}
        for p in mask_dir.iterdir():
            if p.is_file() and p.suffix.lower() in {".npy", ".npz"}:
                stem = p.stem
                if stem in mask_files:
                    raise ValueError(
                        f"Collision in {mask_dir}: multiple files with identical stem {stem!r}: "
                        f"{mask_files[stem].name} and {p.name}"
                    )
                mask_files[stem] = p

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
        effective_spacing = info.get("effective_spacing")
        original_spacing = info.get("orig_spacing")
        # The checked-in preprocessing script stores resized NPY volumes as
        # [H,W,Z].  Its effective spacing is already [Z,Y,X], while the raw
        # NIfTI header's original spacing is [X,Y,Z] and needs reordering.
        if isinstance(effective_spacing, (list, tuple)) and len(effective_spacing) >= 3:
            values[str(case_id)] = tuple(float(value) for value in effective_spacing[:3])
        elif isinstance(original_spacing, (list, tuple)) and len(original_spacing) >= 3:
            values[str(case_id)] = (
                float(original_spacing[2]),
                float(original_spacing[0]),
                float(original_spacing[1]),
            )
    return values


@dataclass(frozen=True)
class EffectiveSplits:
    """Immutable container for authoritative effective splits and deterministic signature."""

    splits: dict[str, list[VolumeRecord]]
    signature: str
    strategy: str  # "manifest", "explicit_tags", "patient_level_fallback"
    manifest_path: Path | None = None

    def __getitem__(self, split: str) -> list[VolumeRecord]:
        canon = canonicalize_split_alias(split)
        if canon not in self.splits:
            raise KeyError(f"Split {split!r} ({canon!r}) not in effective splits: {sorted(self.splits.keys())}")
        return self.splits[canon]

    def get(self, split: str, default: Any = None) -> list[VolumeRecord] | Any:
        try:
            canon = canonicalize_split_alias(split)
            return self.splits.get(canon, default)
        except ValueError:
            return default


def discover_acdc_records(data_root: str | Path, split: str | None = None) -> list[VolumeRecord]:
    """Discover paired ACDC records without slicing across patient boundaries.

    Distinguishes multiple traversals of the exact same image/mask file pair
    (e.g. from recursive rglob at root plus child split directory roots) from
    real collisions (distinct file paths with the same case_id). Preserves the
    most specific consistent split tag and fails on conflicting ownership.
    """

    root = Path(data_root)
    if not root.exists():
        raise FileNotFoundError(
            f"ACDC data root does not exist: {root}. Provide raw ACDC or a preprocessed "
            "root containing paired volumes/ and masks/."
        )
    candidate_roots: list[tuple[Path, str]] = []
    if split is not None:
        canon_req = canonicalize_split_alias(split)
        aliases = _split_aliases(split)
        if aliases:
            candidate_roots.extend((root / alias, alias) for alias in aliases if (root / alias).exists())
    else:
        canon_req = None

    if not candidate_roots:
        candidate_roots.append((root, ""))
        if split is None:
            for child in sorted(root.iterdir()):
                if child.is_dir() and child.name.lower() in {
                    "train", "training", "val", "validation", "valid", "test", "testing"
                }:
                    candidate_roots.append((child, child.name.lower()))

    # Map: case_id -> (VolumeRecord, (resolved_image_path, resolved_mask_path))
    records_by_case: dict[str, tuple[VolumeRecord, tuple[Path, Path]]] = {}

    for candidate, split_name in candidate_roots:
        spacing_by_case = _metadata_spacing(candidate)
        pairs = _paired_files(candidate)
        if not pairs:
            pairs = _raw_nifti_pairs(candidate)

        norm_tag = canonicalize_split_alias(split_name) if split_name else ""

        for image_path, mask_path, case_id in pairs:
            pair_identity = (image_path.resolve(), mask_path.resolve())

            if case_id not in records_by_case:
                record = VolumeRecord(
                    case_id=case_id,
                    patient_id=patient_id_from_case_id(case_id),
                    image_path=image_path,
                    mask_path=mask_path,
                    split=norm_tag,
                    spacing=spacing_by_case.get(case_id),
                    source_format="nifti" if image_path.name.lower().endswith((".nii", ".nii.gz")) else "npy",
                )
                records_by_case[case_id] = (record, pair_identity)
            else:
                existing_rec, existing_identity = records_by_case[case_id]
                if pair_identity == existing_identity:
                    # Traversal of the identical resolved file pair
                    if not existing_rec.split and norm_tag:
                        # Upgrade untagged record with specific tag
                        updated_rec = VolumeRecord(
                            case_id=existing_rec.case_id,
                            patient_id=existing_rec.patient_id,
                            image_path=existing_rec.image_path,
                            mask_path=existing_rec.mask_path,
                            split=norm_tag,
                            spacing=existing_rec.spacing or spacing_by_case.get(case_id),
                            source_format=existing_rec.source_format,
                        )
                        records_by_case[case_id] = (updated_rec, pair_identity)
                    elif existing_rec.split and norm_tag:
                        if existing_rec.split != norm_tag:
                            raise ValueError(
                                f"Conflicting split tags discovered for case {case_id!r}: "
                                f"{existing_rec.split!r} vs {norm_tag!r}"
                            )
                else:
                    # Distinct files sharing the same case ID -> real collision
                    raise ValueError(
                        f"Collision: duplicate case {case_id!r} discovered at distinct file paths: "
                        f"{existing_identity[0]} vs {pair_identity[0]}"
                    )

    if not records_by_case:
        raise FileNotFoundError(
            f"No paired ACDC volumes found under {root}. Expected volumes/ + masks/ "
            "or raw patient*/ *_gt.nii(.gz) files."
        )

    all_records = sorted([rec for rec, _ in records_by_case.values()], key=lambda r: r.case_id)

    if canon_req is not None:
        tagged_for_split = [r for r in all_records if r.split == canon_req]
        if tagged_for_split:
            return tagged_for_split
        elif any(r.split for r in all_records):
            return []
        return all_records

    return all_records


def discover_acdc_cases(data_root: str | Path, split: str | None = None) -> list[str]:
    return [record.case_id for record in discover_acdc_records(data_root, split=split)]


def resolve_effective_acdc_splits(
    records: Sequence[VolumeRecord],
    *,
    split_manifest: str | Path | None = None,
    seed: int = 42,
    train_fraction: float = 0.8,
    val_fraction: float = 0.1,
    train_split: str = "train",
    val_split: str = "val",
    test_split: str = "test",
) -> EffectiveSplits:
    """One authoritative resolver producing all effective split record lists and a deterministic signature.

    Dataset loaders and preflight/entrypoint validators call this resolver;
    neither reimplements split resolution or validation independently.
    """
    if not records:
        raise ValueError("Cannot resolve splits from an empty record list")

    # Validate aliases
    req_train = canonicalize_split_alias(train_split)
    req_val = canonicalize_split_alias(val_split)
    req_test = canonicalize_split_alias(test_split)
    if req_train != "train":
        raise ValueError(f"Configured train_split must alias 'train', got {train_split!r}")
    if req_val != "val":
        raise ValueError(f"Configured val_split must alias 'val', got {val_split!r}")
    if req_test != "test":
        raise ValueError(f"Configured test_split must alias 'test', got {test_split!r}")

    # Check for duplicate case records in input
    seen_cases: dict[str, tuple[VolumeRecord, tuple[Path, Path]]] = {}
    for r in records:
        cid = r.case_id
        identity = (r.image_path.resolve(), r.mask_path.resolve())
        if cid in seen_cases:
            prev_rec, prev_id = seen_cases[cid]
            if prev_id != identity:
                raise ValueError(
                    f"Collision: duplicate case {cid!r} with distinct file paths: "
                    f"{prev_id[0]} vs {identity[0]}"
                )
            else:
                raise ValueError(f"Duplicate record entry for case {cid!r} in input records")
        seen_cases[cid] = (r, identity)

    discovered_by_case = {r.case_id: r for r in records}
    discovered_cases = set(discovered_by_case.keys())

    manifest_path = Path(split_manifest) if split_manifest is not None else None

    # Strategy 1: Explicit Manifest governs membership
    if manifest_path is not None:
        if not manifest_path.exists():
            raise FileNotFoundError(f"Configured ACDC split manifest does not exist: {manifest_path}")
        manifest = read_split_manifest(manifest_path)

        # Check that required train and val splits exist in manifest
        if "train" not in manifest or not manifest["train"]:
            raise ValueError(f"Configured split manifest missing required 'train' split: {manifest_path}")
        if "val" not in manifest or not manifest["val"]:
            raise ValueError(f"Configured split manifest missing required 'val' split: {manifest_path}")

        # Check alignment between discovered cases and manifest cases
        all_manifest_cases = set().union(*(set(cases) for cases in manifest.values()))
        missing = sorted(all_manifest_cases - discovered_cases)
        if missing:
            raise ValueError(
                f"Configured ACDC split manifest does not match discovered cases: missing={missing[:5]}"
            )
        extra = sorted(discovered_cases - all_manifest_cases)
        if extra:
            raise ValueError(
                f"Configured ACDC split manifest does not match discovered cases: extra={extra[:5]}"
            )

        # Conflicting directory tags fail clearly rather than override the manifest
        for split_name, case_ids in manifest.items():
            for cid in case_ids:
                rec = discovered_by_case[cid]
                if rec.split:
                    norm_tag = canonicalize_split_alias(rec.split)
                    if norm_tag != split_name:
                        raise ValueError(
                            f"Manifest-vs-tag conflict for case {cid!r}: manifest assigned to {split_name!r}, "
                            f"but record has directory tag {rec.split!r} ({norm_tag!r})"
                        )

        effective_splits: dict[str, list[VolumeRecord]] = {}
        for split_name in ("train", "val", "test"):
            if split_name in manifest:
                effective_splits[split_name] = sorted(
                    [discovered_by_case[cid] for cid in manifest[split_name]],
                    key=lambda r: r.case_id,
                )

        validate_patient_split({s: [r.case_id for r in recs] for s, recs in effective_splits.items()})

        sig = compute_split_signature(effective_splits)
        return EffectiveSplits(
            splits=effective_splits,
            signature=sig,
            strategy="manifest",
            manifest_path=manifest_path,
        )

    # Strategy 2: Explicit tags without manifest
    tagged_records: dict[str, list[VolumeRecord]] = {}
    untagged_records: list[VolumeRecord] = []
    for rec in records:
        if rec.split and str(rec.split).strip():
            norm_tag = canonicalize_split_alias(rec.split)
            tagged_records.setdefault(norm_tag, []).append(rec)
        else:
            untagged_records.append(rec)

    if tagged_records and untagged_records:
        raise ValueError(
            f"Mixed tagged and untagged records found ({len(tagged_records)} tag groups, "
            f"{len(untagged_records)} untagged records). Split assignment is ambiguous: "
            "all records must be tagged or all untagged."
        )

    if tagged_records:
        if "train" not in tagged_records or not tagged_records["train"]:
            raise ValueError("Explicitly tagged records missing required 'train' split")
        if "val" not in tagged_records or not tagged_records["val"]:
            raise ValueError("Explicitly tagged records missing required 'val' split")

        validate_patient_split({s: [r.case_id for r in recs] for s, recs in tagged_records.items()})

        effective_splits = {
            s: sorted(recs, key=lambda r: r.case_id)
            for s, recs in sorted(tagged_records.items())
        }
        sig = compute_split_signature(effective_splits)
        return EffectiveSplits(
            splits=effective_splits,
            signature=sig,
            strategy="explicit_tags",
            manifest_path=None,
        )

    # Strategy 3: Deterministic patient-level fallback for untagged records
    split_ids = patient_level_split(
        [r.case_id for r in records],
        train_fraction=train_fraction,
        val_fraction=val_fraction,
        seed=seed,
    )
    effective_splits = {
        split_name: sorted(
            [discovered_by_case[cid] for cid in case_ids],
            key=lambda r: r.case_id,
        )
        for split_name, case_ids in split_ids.items()
    }
    validate_patient_split({s: [r.case_id for r in recs] for s, recs in effective_splits.items()})
    sig = compute_split_signature(effective_splits)
    return EffectiveSplits(
        splits=effective_splits,
        signature=sig,
        strategy="patient_level_fallback",
        manifest_path=None,
    )


def resolve_acdc_records(
    records: Sequence[VolumeRecord],
    *,
    split: str | None,
    split_manifest: str | Path | None = None,
    seed: int = 42,
    train_fraction: float = 0.8,
    val_fraction: float = 0.1,
) -> list[VolumeRecord]:
    """Resolve ACDC volume records for a requested split using the authoritative resolver."""
    if split is None:
        if split_manifest is not None:
            effective = resolve_effective_acdc_splits(
                records,
                split_manifest=split_manifest,
                seed=seed,
                train_fraction=train_fraction,
                val_fraction=val_fraction,
            )
            all_recs: list[VolumeRecord] = []
            for s_recs in effective.splits.values():
                all_recs.extend(s_recs)
            return sorted(all_recs, key=lambda r: r.case_id)
        return sorted(records, key=lambda r: r.case_id)

    effective = resolve_effective_acdc_splits(
        records,
        split_manifest=split_manifest,
        seed=seed,
        train_fraction=train_fraction,
        val_fraction=val_fraction,
    )
    canon = canonicalize_split_alias(split)
    if canon not in effective.splits:
        raise ValueError(f"Requested ACDC split {split!r} ({canon!r}) not in effective splits: {sorted(effective.splits.keys())}")
    return effective.splits[canon]


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
        split_manifest: str | Path | None = None,
        seed: int = 42,
        image_size: int = 256,
        augment: bool = False,
        transform: object | None = None,
        foreground_only: bool = False,
        depth_axis: int | None = None,
        expected_slices: int | None = None,
        **kwargs: object,
    ) -> None:
        all_records = list(records) if records is not None else discover_acdc_records(data_root)
        if split is not None or split_manifest is not None:
            effective = resolve_effective_acdc_splits(
                all_records,
                split_manifest=split_manifest,
                seed=seed,
            )
            self.split_signature = effective.signature
            if split is not None:
                canon = canonicalize_split_alias(split)
                if canon not in effective.splits:
                    raise ValueError(f"ACDC split {split!r} ({canon!r}) not in effective splits: {sorted(effective.splits.keys())}")
                selected = effective.splits[canon]
            else:
                all_recs: list[VolumeRecord] = []
                for s_recs in effective.splits.values():
                    all_recs.extend(s_recs)
                selected = sorted(all_recs, key=lambda r: r.case_id)
        else:
            selected = sorted(all_records, key=lambda r: r.case_id)
            self.split_signature = compute_split_signature({"all": selected})

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
            depth_axis=depth_axis,
            expected_slices=expected_slices,
            **{
                key: value
                for key, value in kwargs.items()
                if key in {"lower_percentile", "upper_percentile", "max_cache"}
            },
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
