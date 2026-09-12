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

# Scientific protocol labels
NATIVE_MNMS_PROTOCOL = "mnms_unified_native_v1"
EXTERNAL_MNMS_PROTOCOL = "acdc_frozen_to_mnms_external_v1"



@dataclass(frozen=True)
class MNMSClassMapping:
    raw_to_acdc: Mapping[int, int]
    source_name: str = "M&Ms explicit LV/MYO/RV mapping"

    def __post_init__(self) -> None:
        if not isinstance(self.raw_to_acdc, Mapping):
            raise TypeError(f"raw_to_acdc must be a mapping, got {type(self.raw_to_acdc).__name__}")
        mapping: dict[int, int] = {}
        for raw_key, raw_value in self.raw_to_acdc.items():
            if isinstance(raw_key, bool) or isinstance(raw_value, bool):
                raise TypeError("M&Ms class mapping keys and values cannot be booleans")
            if isinstance(raw_key, (float, np.floating)):
                if not np.isfinite(raw_key) or not float(raw_key).is_integer():
                    raise ValueError(f"M&Ms class mapping key cannot be fractional: {raw_key}")
                int_key = int(raw_key)
            elif isinstance(raw_key, (int, np.integer)):
                int_key = int(raw_key)
            elif isinstance(raw_key, str):
                try:
                    f = float(raw_key)
                    if not np.isfinite(f) or not f.is_integer():
                        raise ValueError(f"M&Ms class mapping key cannot be fractional: {raw_key}")
                    int_key = int(raw_key)
                except ValueError as exc:
                    raise ValueError(f"M&Ms class mapping key must be an integer: {raw_key}") from exc
            else:
                raise TypeError(f"M&Ms class mapping key must be an integer, got {type(raw_key).__name__}")

            if isinstance(raw_value, (float, np.floating)):
                if not np.isfinite(raw_value) or not float(raw_value).is_integer():
                    raise ValueError(f"M&Ms class mapping value cannot be fractional: {raw_value}")
                int_val = int(raw_value)
            elif isinstance(raw_value, (int, np.integer)):
                int_val = int(raw_value)
            elif isinstance(raw_value, str):
                try:
                    f = float(raw_value)
                    if not np.isfinite(f) or not f.is_integer():
                        raise ValueError(f"M&Ms class mapping value cannot be fractional: {raw_value}")
                    int_val = int(raw_value)
                except ValueError as exc:
                    raise ValueError(f"M&Ms class mapping value must be an integer: {raw_value}") from exc
            else:
                raise TypeError(f"M&Ms class mapping value must be an integer, got {type(raw_value).__name__}")

            mapping[int_key] = int_val
        expected = set(range(NUM_CLASSES))
        if 0 not in mapping or mapping[0] != 0:
            raise ValueError("M&Ms class mapping must explicitly map raw background 0 to ACDC background 0")
        if set(mapping) != expected:
            raise ValueError(
                "M&Ms class mapping must explicitly map raw classes 0..3; "
                f"got raw keys {sorted(mapping)}"
            )
        values = set(mapping.values())
        if values != expected:
            raise ValueError(
                f"M&Ms mapping must be a one-to-one four-class mapping to 0..{NUM_CLASSES - 1}, got {values}"
            )
        object.__setattr__(self, "raw_to_acdc", mapping)

    def apply(self, labels: np.ndarray) -> np.ndarray:
        labels = np.asarray(labels)
        if not np.isfinite(labels).all() or not np.equal(labels, np.floor(labels)).all():
            raise ValueError("M&Ms mask labels must be finite integers")
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


# Mask filenames may repeat the image key verbatim or add one of these
# suffixes.  The order is the matching precedence and is deliberately fixed so
# discovery is deterministic when several of them exist for one image.
MASK_KEY_SUFFIXES = ("_gt", "_label", "_seg")


def _mask_key_candidates(image_key: str) -> list[str]:
    return [image_key, *(f"{image_key}{suffix}" for suffix in MASK_KEY_SUFFIXES)]


def _supported_volume_file(path: Path) -> bool:
    name = path.name.lower()
    return name.endswith((".npy", ".npz", ".nii", ".nii.gz"))


def _group_volume_files(directory: Path) -> dict[str, list[Path]]:
    """Group the supported volume files of one directory by normalized key.

    Several distinct files can normalize to the same key (``case.npy`` beside
    ``case.npz``).  Grouping keeps every one of them visible instead of letting
    a dictionary comprehension silently drop all but the last, so the strict
    caller can refuse to choose.

    Directory order is preserved deliberately: the tolerant caller takes the
    LAST entry of a group, which is the file the previous dictionary
    comprehension over ``iterdir()`` would have kept.  External and default
    discovery therefore selects exactly what it selected before.
    """

    groups: dict[str, list[Path]] = {}
    for path in directory.iterdir():
        if path.is_file() and _supported_volume_file(path):
            groups.setdefault(_volume_key(path), []).append(path)
    return groups


def _volume_key(path: Path) -> str:
    return _strip_archive_suffix(path)


def is_mnms_binary_path(path: str | Path) -> bool:
    """Check if a path references the known mnm_binary component.

    Targets the specific 'mnm_binary' component rather than arbitrary parent
    directories containing the substring 'binary' (e.g. '/tmp/binary_test/mnm' is benign).
    """
    for part in Path(path).parts:
        if part.lower() in {"mnm_binary", "mnms_binary"}:
            return True
    return False


def discover_mnms_records(
    data_root: str | Path,
    split: str | None = None,
    *,
    strict_pairing: bool = False,
) -> list[VolumeRecord]:
    """Discover paired M&Ms image/mask volumes under ``data_root``.

    ``strict_pairing`` controls what happens when the layout is not an exact
    one-to-one pairing.  The default, ``False``, keeps its historical tolerant
    behaviour — an unpartnered volume is skipped and the discovered cohort is
    returned — which is what external evaluation over a tree that intentionally
    carries unlabelled volumes needs.

    ``True`` raises instead, so a supervised training cohort can neither
    silently shrink nor be guessed at. It rejects an image with no mask, a mask
    with no image, several files normalizing to one key (``case.npy`` beside
    ``case.npz``), an image matching more than one candidate mask, and one mask
    claimed by two different images (``p`` and ``p_gt`` both taking ``p_gt``).
    """

    root = Path(data_root)
    if is_mnms_binary_path(root):
        raise ValueError(
            f"M&Ms binary derivative dataset is incompatible with four-class contract: {root}"
        )
    if not root.exists():
        raise FileNotFoundError(
            f"M&Ms data root does not exist: {root}. Provide the external dataset explicitly; "
            "the domain-shift protocol never substitutes ACDC data."
        )
    roots = [root]
    if split is not None:
        aliases = {
            "train": ("train", "training"),
            "training": ("train", "training"),
            "val": ("val", "validation"),
            "validation": ("val", "validation"),
            "test": ("test", "testing"),
            "testing": ("test", "testing"),
        }.get(str(split).lower(), (str(split),))
        roots = [root / alias for alias in aliases if (root / alias).exists()]
        if not roots:
            raise FileNotFoundError(f"M&Ms split directory {split!r} not found under {root}")
    records: list[VolumeRecord] = []
    seen: set[str] = set()
    images_without_mask: list[str] = []
    masks_without_image: list[str] = []
    colliding_keys: list[str] = []
    ambiguous_masks: list[str] = []
    shared_masks: list[str] = []
    duplicate_cases: list[str] = []
    for candidate in roots:
        for image_dir, mask_dir in _find_pair_dirs(candidate):
            image_groups = _group_volume_files(image_dir)
            mask_groups = _group_volume_files(mask_dir)
            # Two files that normalise to one key (``case.npy`` and
            # ``case.npz``) used to collapse into a single dictionary entry,
            # hiding one of them.  Record the collision; strict mode refuses to
            # guess which file was meant.
            for role, groups in (("image", image_groups), ("mask", mask_groups)):
                for key, paths in sorted(groups.items()):
                    if len(paths) > 1:
                        colliding_keys.append(
                            f"{role} key {key!r}: {sorted(str(p) for p in paths)}"
                        )
            # The last entry of a group is what the previous dictionary
            # comprehension kept; tolerant selection is unchanged.
            image_files = {key: paths[-1] for key, paths in image_groups.items()}
            mask_files = {key: paths[-1] for key, paths in mask_groups.items()}
            mask_claims: dict[str, list[str]] = {}
            for image_key, image_path in sorted(image_files.items()):
                matching_keys = [
                    key for key in _mask_key_candidates(image_key) if key in mask_files
                ]
                if not matching_keys:
                    images_without_mask.append(str(image_path))
                    continue
                if len(matching_keys) > 1:
                    ambiguous_masks.append(
                        f"{image_path}: {[str(mask_files[key]) for key in matching_keys]}"
                    )
                mask_key = matching_keys[0]
                mask_claims.setdefault(mask_key, []).append(image_key)
                mask_path = mask_files[mask_key]
                case_id = _strip_archive_suffix(image_path)
                if case_id in seen:
                    # Two pair directories or two split aliases offered the same
                    # case.  Tolerant discovery keeps the first and moves on;
                    # strict discovery will not pick one silently.
                    duplicate_cases.append(f"{case_id}: {image_path}")
                    continue
                seen.add(case_id)
                records.append(VolumeRecord(case_id, patient_id_from_case_id(case_id), image_path, mask_path, split or "", source_format="nifti" if image_path.name.lower().endswith((".nii", ".nii.gz")) else "npy"))
            # One mask cannot be the ground truth of two different images
            # (``p`` matching ``p_gt`` while ``p_gt`` itself is also an image).
            for mask_key, image_keys in sorted(mask_claims.items()):
                if len(image_keys) > 1:
                    shared_masks.append(
                        f"{mask_files[mask_key]} claimed by {sorted(image_keys)}"
                    )
            masks_without_image.extend(
                str(mask_files[key]) for key in sorted(set(mask_files) - set(mask_claims))
            )
        if not records:
            masks = sorted(candidate.rglob("*_gt.nii")) + sorted(candidate.rglob("*_gt.nii.gz"))
            for mask_path in masks:
                image_name = mask_path.name.replace("_gt.nii.gz", ".nii.gz").replace("_gt.nii", ".nii")
                image_path = mask_path.with_name(image_name)
                if not image_path.exists():
                    raise FileNotFoundError(f"M&Ms mask has no matching image: {mask_path}")
                case_id = _strip_archive_suffix(image_path)
                records.append(VolumeRecord(case_id, patient_id_from_case_id(case_id), image_path, mask_path, split or "", source_format="nifti"))
    if strict_pairing:
        details: list[str] = []
        if images_without_mask:
            details.append(f"images with no mask: {sorted(images_without_mask)}")
        if masks_without_image:
            details.append(f"masks with no image: {sorted(masks_without_image)}")
        if colliding_keys:
            details.append(f"files sharing one normalized key: {sorted(colliding_keys)}")
        if ambiguous_masks:
            details.append(f"images matching several masks: {sorted(ambiguous_masks)}")
        if shared_masks:
            details.append(f"masks claimed by several images: {sorted(shared_masks)}")
        if duplicate_cases:
            details.append(
                f"case ids discovered more than once: {sorted(duplicate_cases)}"
            )
        if details:
            raise FileNotFoundError(
                f"M&Ms cohort under {root} (split={split!r}) is not unambiguously paired; "
                "a supervised cohort must not silently shrink or guess. "
                + "; ".join(details)
                + f". Accepted mask key suffixes: {list(MASK_KEY_SUFFIXES)}."
            )
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
    scientific_protocol_label = NATIVE_MNMS_PROTOCOL
    external_protocol_label = EXTERNAL_MNMS_PROTOCOL

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
        root_path = Path(data_root)
        if is_mnms_binary_path(root_path):
            raise ValueError(
                f"M&Ms binary derivative dataset is incompatible with four-class contract: {root_path}"
            )
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
