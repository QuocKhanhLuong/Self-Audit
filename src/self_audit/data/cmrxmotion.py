"""Audited adapter for CMRxMotion ED/ES SAX volumes and ZIP archives."""

from __future__ import annotations

import gzip
import io
import re
import zipfile
from collections import OrderedDict
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .cardiac35d import Cardiac35DAdapter, CardiacUnit, LoadedCardiacUnit
from .label_schema import remap_labels


CMRXMOTION_RAW_TO_COMMON = {0: 0, 1: 3, 2: 2, 3: 1}
_NIFTI_RE = re.compile(
    r"^(?P<subject>[^/\\-]+)-(?P<acquisition>[^/\\-]+)-"
    r"(?P<phase>ED|ES)(?P<label>-label)?$",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class CMRxMotionName:
    subject_id: str
    acquisition_id: str
    phase: str
    is_label: bool
    case_id: str


@dataclass(frozen=True)
class ArchiveMemberRef:
    """Reference to an extracted NIfTI or a member inside a ZIP archive."""

    path: Path | None = None
    archive_path: Path | None = None
    member_name: str = ""

    @property
    def display_path(self) -> str:
        if self.archive_path is not None:
            return f"{self.archive_path}!{self.member_name}"
        return str(self.path)


@dataclass(frozen=True)
class LoadedNifti:
    array: np.ndarray
    affine: np.ndarray
    zooms: tuple[float, ...]
    dtype: str


@dataclass(frozen=True)
class CMRxMotionCase:
    subject_id: str
    acquisition_id: str
    phase: str
    case_id: str
    image_ref: ArchiveMemberRef
    mask_ref: ArchiveMemberRef | None


@dataclass(frozen=True)
class CMRxMotionAuditRecord:
    subject_id: str
    acquisition_id: str
    phase: str
    case_id: str
    image_path: str
    mask_path: str
    has_mask: bool
    image_shape: tuple[int, ...]
    mask_shape: tuple[int, ...]
    image_dtype: str
    mask_dtype: str
    spacing: tuple[float, ...]
    unique_labels: tuple[int, ...]
    affine_equal: bool | None
    error: str = ""


def _strip_nifti_suffix(name: str | Path) -> str:
    value = Path(str(name)).name
    lower = value.lower()
    for suffix in (".nii.gz", ".nii"):
        if lower.endswith(suffix):
            return value[: -len(suffix)]
    return Path(value).stem


def parse_cmrxmotion_filename(name: str | Path) -> CMRxMotionName:
    """Parse the verified subject-acquisition-phase naming convention."""

    stem = _strip_nifti_suffix(name)
    match = _NIFTI_RE.fullmatch(stem)
    if match is None:
        raise ValueError(f"Unrecognized CMRxMotion filename: {name!s}")
    subject = match.group("subject")
    acquisition = match.group("acquisition")
    phase = match.group("phase").upper()
    return CMRxMotionName(
        subject_id=subject,
        acquisition_id=acquisition,
        phase=phase,
        is_label=bool(match.group("label")),
        case_id=f"{subject}-{acquisition}-{phase}",
    )


def _read_reference(ref: ArchiveMemberRef) -> bytes:
    if ref.archive_path is not None:
        with zipfile.ZipFile(ref.archive_path) as archive:
            payload = archive.read(ref.member_name)
    elif ref.path is not None:
        payload = ref.path.read_bytes()
    else:
        raise ValueError("NIfTI reference has neither path nor archive member")
    if ref.member_name.lower().endswith(".gz") or (ref.path is not None and ref.path.name.lower().endswith(".gz")):
        return gzip.decompress(payload)
    return payload


def load_nifti_reference(ref: ArchiveMemberRef) -> LoadedNifti:
    """Load a NIfTI reference without extracting or modifying source archives."""

    try:
        import nibabel as nib
    except ImportError as exc:  # pragma: no cover
        raise ImportError("CMRxMotion NIfTI loading requires nibabel") from exc
    payload = _read_reference(ref)
    try:
        image = nib.Nifti1Image.from_bytes(payload)
    except Exception:
        image = nib.load(io.BytesIO(payload))
    array = np.asanyarray(image.dataobj)
    return LoadedNifti(
        array=np.asarray(array),
        affine=np.asarray(image.affine),
        zooms=tuple(float(value) for value in image.header.get_zooms()),
        dtype=str(array.dtype),
    )


def _squeeze_image_axis(array: np.ndarray, *, case_id: str) -> np.ndarray:
    value = np.asarray(array)
    if value.ndim == 3:
        return value
    if value.ndim != 4:
        raise ValueError(f"CMRxMotion {case_id} must be 3-D or singleton-4-D, got {value.shape}")
    singleton_axes = [axis for axis, size in enumerate(value.shape) if int(size) == 1]
    if len(singleton_axes) != 1:
        raise ValueError(
            f"CMRxMotion {case_id} has no unique singleton image axis: {value.shape}"
        )
    return np.squeeze(value, axis=singleton_axes[0])


def _iter_references(root: Path) -> list[ArchiveMemberRef]:
    refs: list[ArchiveMemberRef] = []
    if root.is_file() and root.suffix.lower() == ".zip":
        archives = [root]
        extracted_root = None
    else:
        archives = sorted(root.rglob("*.zip")) if root.exists() else []
        extracted_root = root if root.is_dir() else None
    for archive_path in archives:
        with zipfile.ZipFile(archive_path) as archive:
            refs.extend(
                ArchiveMemberRef(archive_path=archive_path, member_name=name)
                for name in archive.namelist()
                if not name.endswith("/")
                and name.lower().endswith((".nii", ".nii.gz"))
            )
    if extracted_root is not None:
        refs.extend(
            ArchiveMemberRef(path=path)
            for path in sorted(extracted_root.rglob("*"))
            if path.is_file()
            and path.suffix.lower() == ".nii"
            or path.is_file() and path.name.lower().endswith(".nii.gz")
        )
    return refs


class CMRxMotionAdapter(Cardiac35DAdapter):
    dataset_name = "cmr_motion"

    def __init__(
        self,
        data_root: str | Path,
        *,
        allow_affine_mismatch: bool = False,
        max_array_cache: int = 2,
    ) -> None:
        self.data_root = Path(data_root)
        self.allow_affine_mismatch = bool(allow_affine_mismatch)
        self.max_array_cache = max(int(max_array_cache), 1)
        self.cases = self._discover_cases()
        self._loaded_cache: OrderedDict[str, tuple[LoadedNifti, LoadedNifti | None, bool | None]] = OrderedDict()
        self._audit_cache: dict[str, CMRxMotionAuditRecord] = {}

    def _discover_cases(self) -> list[CMRxMotionCase]:
        grouped: dict[str, dict[str, ArchiveMemberRef | None]] = {}
        metadata: dict[str, CMRxMotionName] = {}
        seen_refs: dict[str, str] = {}
        for ref in _iter_references(self.data_root):
            parsed = parse_cmrxmotion_filename(ref.member_name or ref.path.name)
            key = parsed.case_id
            kind = "mask" if parsed.is_label else "image"
            if kind == "image" and key in grouped and grouped[key].get("image") is not None:
                raise ValueError(f"duplicate CMRxMotion image case: {key}")
            if kind == "mask" and key in grouped and grouped[key].get("mask") is not None:
                raise ValueError(f"duplicate CMRxMotion mask case: {key}")
            grouped.setdefault(key, {})[kind] = ref
            metadata[key] = parsed
            seen_refs[key] = ref.display_path
        cases: list[CMRxMotionCase] = []
        for case_id in sorted(grouped):
            values = grouped[case_id]
            parsed = metadata[case_id]
            image_ref = values.get("image")
            if image_ref is None:
                raise ValueError(f"CMRxMotion case {case_id} has a label but no image")
            cases.append(
                CMRxMotionCase(
                    subject_id=parsed.subject_id,
                    acquisition_id=parsed.acquisition_id,
                    phase=parsed.phase,
                    case_id=case_id,
                    image_ref=image_ref,
                    mask_ref=values.get("mask"),
                )
            )
        return cases

    @staticmethod
    def _normalise_loaded(case: CMRxMotionCase) -> tuple[LoadedNifti, LoadedNifti | None, bool | None]:
        image = load_nifti_reference(case.image_ref)
        image_array = _squeeze_image_axis(image.array, case_id=case.case_id)
        image = LoadedNifti(image_array, image.affine, image.zooms, image.dtype)
        if case.mask_ref is None:
            return image, None, None
        mask = load_nifti_reference(case.mask_ref)
        mask_array = _squeeze_image_axis(mask.array, case_id=case.case_id)
        mask = LoadedNifti(mask_array, mask.affine, mask.zooms, mask.dtype)
        if image.array.shape != mask.array.shape:
            raise ValueError(
                f"CMRxMotion {case.case_id} image/mask shape mismatch: "
                f"{image.array.shape} vs {mask.array.shape}"
            )
        return image, mask, bool(np.allclose(image.affine, mask.affine, rtol=1e-5, atol=1e-5))

    def _load_case(self, case: CMRxMotionCase) -> tuple[LoadedNifti, LoadedNifti | None, bool | None]:
        cached = self._loaded_cache.get(case.case_id)
        if cached is not None:
            self._loaded_cache.move_to_end(case.case_id)
            return cached
        value = self._normalise_loaded(case)
        self._loaded_cache[case.case_id] = value
        self._loaded_cache.move_to_end(case.case_id)
        while len(self._loaded_cache) > self.max_array_cache:
            self._loaded_cache.popitem(last=False)
        return value

    def audit(self) -> list[CMRxMotionAuditRecord]:
        records: list[CMRxMotionAuditRecord] = []
        for case in self.cases:
            try:
                image, mask, affine_equal = self._load_case(case)
                labels = tuple(int(value) for value in np.unique(mask.array)) if mask is not None else ()
                record = CMRxMotionAuditRecord(
                    subject_id=case.subject_id,
                    acquisition_id=case.acquisition_id,
                    phase=case.phase,
                    case_id=case.case_id,
                    image_path=case.image_ref.display_path,
                    mask_path=case.mask_ref.display_path if case.mask_ref else "",
                    has_mask=mask is not None,
                    image_shape=tuple(int(value) for value in image.array.shape),
                    mask_shape=tuple(int(value) for value in mask.array.shape) if mask is not None else (),
                    image_dtype=image.dtype,
                    mask_dtype=mask.dtype if mask is not None else "",
                    spacing=tuple(float(value) for value in image.zooms[:3]),
                    unique_labels=labels,
                    affine_equal=affine_equal,
                )
            except Exception as exc:
                record = CMRxMotionAuditRecord(
                    subject_id=case.subject_id,
                    acquisition_id=case.acquisition_id,
                    phase=case.phase,
                    case_id=case.case_id,
                    image_path=case.image_ref.display_path,
                    mask_path=case.mask_ref.display_path if case.mask_ref else "",
                    has_mask=case.mask_ref is not None,
                    image_shape=(),
                    mask_shape=(),
                    image_dtype="",
                    mask_dtype="",
                    spacing=(),
                    unique_labels=(),
                    affine_equal=None,
                    error=str(exc),
                )
            records.append(record)
            self._audit_cache[case.case_id] = record
        return records

    def discover_units(
        self,
        *,
        split: str | None = None,
        supervised_only: bool = True,
    ) -> list[CardiacUnit]:
        del split
        units: list[CardiacUnit] = []
        for case in self.cases:
            record = self._audit_cache.get(case.case_id)
            if record is None:
                self.audit()
                record = self._audit_cache[case.case_id]
            if supervised_only and not record.has_mask:
                continue
            if record.error:
                continue
            if supervised_only and record.affine_equal is False and not self.allow_affine_mismatch:
                continue
            shape = tuple(reversed(record.image_shape)) if len(record.image_shape) == 3 else ()
            units.append(
                CardiacUnit(
                    dataset=self.dataset_name,
                    case_id=case.case_id,
                    subject_id=case.subject_id,
                    image_ref=case,
                    mask_ref=case.mask_ref,
                    phase=case.phase,
                    acquisition_id=case.acquisition_id,
                    metadata={
                        "shape": shape,
                        "affine_equal": record.affine_equal,
                        "has_mask": record.has_mask,
                        "source_spacing": record.spacing,
                    },
                )
            )
        return units

    def load_unit(self, unit: CardiacUnit) -> LoadedCardiacUnit:
        case = unit.image_ref if isinstance(unit.image_ref, CMRxMotionCase) else next(
            item for item in self.cases if item.case_id == unit.case_id
        )
        image, mask, affine_equal = self._load_case(case)
        if mask is None:
            raise ValueError(f"CMRxMotion case {case.case_id} has no ground-truth mask")
        if image.array.shape != mask.array.shape:
            raise ValueError(f"CMRxMotion case {case.case_id} image/mask shape mismatch")
        if affine_equal is False and not self.allow_affine_mismatch:
            raise ValueError(
                f"CMRxMotion case {case.case_id} has an affine mismatch; "
                "enable allow_affine_mismatch only after visual QC"
            )
        image_zhw = np.transpose(image.array, (2, 1, 0)).astype(np.float32, copy=False)
        mask_zhw = np.transpose(mask.array, (2, 1, 0))
        return LoadedCardiacUnit(
            image_zhw=image_zhw,
            mask_zhw=self.remap_labels(mask_zhw),
            spacing=(
                float(image.zooms[2]),
                float(image.zooms[1]),
                float(image.zooms[0]),
            ),
            metadata={
                "phase": case.phase,
                "acquisition_id": case.acquisition_id,
                "affine_equal": affine_equal,
                "native_orientation": affine_equal is False,
                "source_spacing": tuple(float(value) for value in image.zooms[:3]),
            },
        )

    def remap_labels(self, mask: np.ndarray) -> np.ndarray:
        return remap_labels(mask, CMRXMOTION_RAW_TO_COMMON)

    @property
    def case_to_subject(self) -> dict[str, str]:
        return {case.case_id: case.subject_id for case in self.cases}

    @property
    def subjects(self) -> tuple[str, ...]:
        return tuple(sorted({case.subject_id for case in self.cases}))
