"""Audited adapter for the flattened SAX CMR-MULTI cine volumes."""

from __future__ import annotations

import math
import re
import zipfile
from collections import OrderedDict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from xml.etree import ElementTree

import numpy as np

from .cardiac35d import Cardiac35DAdapter, CardiacUnit, LoadedCardiacUnit
from .label_schema import remap_labels


CMR_MULTI_RAW_TO_COMMON = {0: 0, 1: 2, 2: 3, 3: 1}
_NIFTI_SUFFIXES = (".nii.gz", ".nii")


class ZTInferenceError(ValueError):
    """Raised when a flattened CMR-MULTI candidate cannot be accepted."""


@dataclass(frozen=True)
class ZTInference:
    n: int
    z: int
    t: int
    score: float
    temporal_dice: float
    wrap_dice: float
    spatial_dice: float
    confidence: str
    status: str
    reason: str
    top2_score: float | None = None
    derived_z_spacing: float | None = None

    @property
    def score_gap(self) -> float:
        if self.top2_score is None:
            return math.inf
        return float(self.score - self.top2_score)

    @property
    def accepted(self) -> bool:
        return self.status == "accepted"


@dataclass(frozen=True)
class CMRMultiWorkbookRecord:
    case_id: str
    patient_id: str
    image_path: str = ""
    mask_path: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CMRMultiPair:
    case_id: str
    subject_id: str
    image_path: Path
    mask_path: Path
    workbook: CMRMultiWorkbookRecord | None = None


@dataclass(frozen=True)
class CMRMultiAuditRecord:
    case_id: str
    subject_id: str
    image_path: str
    mask_path: str
    shape: tuple[int, ...]
    image_dtype: str
    mask_dtype: str
    spacing: tuple[float, ...] = ()
    unique_labels: tuple[int, ...] = ()
    inference: ZTInference | None = None
    affine_equal: bool | None = None
    error: str = ""


@dataclass(frozen=True)
class _ZTScore:
    z: int
    t: int
    score: float
    temporal: float
    wrap: float
    spatial: float


def _strip_nifti_suffix(name: str | Path) -> str:
    value = Path(str(name)).name
    for suffix in _NIFTI_SUFFIXES:
        if value.lower().endswith(suffix):
            return value[: -len(suffix)]
    return Path(value).stem


def _dice_foreground(left: np.ndarray, right: np.ndarray) -> float:
    lhs = np.asarray(left) > 0
    rhs = np.asarray(right) > 0
    denominator = int(np.count_nonzero(lhs)) + int(np.count_nonzero(rhs))
    if denominator == 0:
        return 1.0
    return float(2 * np.count_nonzero(lhs & rhs) / denominator)


def _candidate_score(mask_zt: np.ndarray, z: int, t: int) -> _ZTScore:
    candidate = np.asarray(mask_zt).reshape(mask_zt.shape[0], mask_zt.shape[1], z, t, order="C")
    temporal_values = [
        _dice_foreground(candidate[:, :, z_index, time], candidate[:, :, z_index, time + 1])
        for z_index in range(z)
        for time in range(t - 1)
    ]
    wrap_values = [
        _dice_foreground(candidate[:, :, z_index, t - 1], candidate[:, :, z_index, 0])
        for z_index in range(z)
    ]
    spatial_values = [
        _dice_foreground(candidate[:, :, z_index, time], candidate[:, :, z_index + 1, time])
        for time in range(t)
        for z_index in range(z - 1)
    ]
    temporal = float(np.mean(temporal_values)) if temporal_values else 0.0
    wrap = float(np.mean(wrap_values)) if wrap_values else 0.0
    spatial = float(np.mean(spatial_values)) if spatial_values else 0.0
    score = temporal + wrap + max(temporal - spatial, 0.0)
    return _ZTScore(z=z, t=t, score=score, temporal=temporal, wrap=wrap, spatial=spatial)


def infer_zt(
    flattened_mask: np.ndarray,
    *,
    min_t: int = 18,
    max_t: int = 35,
    min_z: int = 6,
    max_z: int = 24,
) -> ZTInference:
    """Infer [Z,T] and reject candidates that fail continuity confidence gates."""

    array = np.asarray(flattened_mask)
    if array.ndim != 3:
        raise ValueError(f"CMR-MULTI flattened masks must be [X,Y,N], got {array.shape}")
    n = int(array.shape[2])
    candidates: list[_ZTScore] = []
    for t in range(int(min_t), int(max_t) + 1):
        if n % t:
            continue
        z = n // t
        if int(min_z) <= z <= int(max_z):
            candidates.append(_candidate_score(array, z, t))
    if not candidates:
        raise ZTInferenceError(
            f"no Z/T candidates for N={n} within T=[{min_t},{max_t}] and Z=[{min_z},{max_z}]"
        )
    candidates.sort(key=lambda item: (-item.score, item.t, item.z))
    best = candidates[0]
    top2 = candidates[1].score if len(candidates) > 1 else None
    gap = math.inf if top2 is None else best.score - top2
    gates = {
        "temporal_dice": best.temporal >= 0.90,
        "wrap_dice": best.wrap >= 0.80,
        "temporal_minus_spatial": best.temporal - best.spatial >= 0.03,
        "top1_top2_gap": gap >= 0.02,
    }
    accepted = all(gates.values())
    failed = ", ".join(name for name, passed in gates.items() if not passed)
    return ZTInference(
        n=n,
        z=best.z,
        t=best.t,
        score=float(best.score),
        temporal_dice=float(best.temporal),
        wrap_dice=float(best.wrap),
        spatial_dice=float(best.spatial),
        confidence="high" if accepted and gap >= 0.05 else ("medium" if accepted else "low"),
        status="accepted" if accepted else "manual_review",
        reason="all confidence gates passed" if accepted else f"failed gates: {failed}",
        top2_score=float(top2) if top2 is not None else None,
        derived_z_spacing=None,
    )


def reconstruct_zt(
    flattened: np.ndarray,
    inference: ZTInference,
    *,
    require_accepted: bool = False,
) -> np.ndarray:
    """Reshape [X,Y,N] into [X,Y,Z,T] with time as the fastest dimension."""

    array = np.asarray(flattened)
    if array.ndim != 3:
        raise ValueError(f"flattened CMR-MULTI arrays must be [X,Y,N], got {array.shape}")
    if require_accepted and not inference.accepted:
        raise ZTInferenceError(
            f"Z/T inference for N={inference.n} is not accepted: {inference.reason}"
        )
    if int(array.shape[2]) != int(inference.n):
        raise ValueError(
            f"flattened array N={array.shape[2]} does not match inference N={inference.n}"
        )
    if int(inference.z) * int(inference.t) != int(inference.n):
        raise ValueError("Z*T must equal the original flattened third-axis length")
    return array.reshape(array.shape[0], array.shape[1], inference.z, inference.t, order="C")


def _local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1].lower()


def _column_index(reference: str) -> int:
    letters = re.match(r"[A-Za-z]+", reference or "")
    if not letters:
        return 0
    result = 0
    for char in letters.group(0).upper():
        result = result * 26 + ord(char) - ord("A") + 1
    return result - 1


def _xlsx_shared_strings(archive: zipfile.ZipFile) -> list[str]:
    try:
        payload = archive.read("xl/sharedStrings.xml")
    except KeyError:
        return []
    root = ElementTree.fromstring(payload)
    values: list[str] = []
    for item in root.iter():
        if _local_name(item.tag) == "si":
            values.append("".join(node.text or "" for node in item.iter() if _local_name(node.tag) == "t"))
    return values


def _xlsx_cell_value(cell: ElementTree.Element, shared: list[str]) -> str:
    value = next((node.text or "" for node in cell.iter() if _local_name(node.tag) == "v"), "")
    cell_type = cell.attrib.get("t", "")
    if cell_type == "s" and value:
        return shared[int(float(value))]
    if cell_type == "inlineStr":
        return "".join(node.text or "" for node in cell.iter() if _local_name(node.tag) == "t")
    return value


def _xlsx_rows(archive: zipfile.ZipFile, sheet_name: str | None = None) -> list[dict[str, str]]:
    shared = _xlsx_shared_strings(archive)
    sheet_paths = sorted(name for name in archive.namelist() if re.fullmatch(r"xl/worksheets/sheet\d+\.xml", name))
    selected: list[dict[str, str]] = []
    for sheet_path in sheet_paths:
        root = ElementTree.fromstring(archive.read(sheet_path))
        rows: list[list[tuple[int, str]]] = []
        for row in root.iter():
            if _local_name(row.tag) != "row":
                continue
            cells = []
            for cell in row:
                if _local_name(cell.tag) == "c":
                    cells.append((_column_index(cell.attrib.get("r", "")), _xlsx_cell_value(cell, shared)))
            rows.append(cells)
        if not rows:
            continue
        headers = {index: value.strip().lower() for index, value in rows[0] if value.strip()}
        header_text = set(headers.values())
        if "patient_id" not in header_text and "image_path" not in header_text:
            continue
        records = []
        for cells in rows[1:]:
            row = {headers[index]: value.strip() for index, value in cells if index in headers}
            if row:
                records.append(row)
        selected.extend(records)
        if sheet_name:
            break
    return selected


def parse_cmr_multi_workbook(path: str | Path) -> dict[str, CMRMultiWorkbookRecord]:
    """Read SAX metadata from the dataset workbook using only stdlib XLSX parsing."""

    workbook_path = Path(path)
    if not workbook_path.is_file():
        raise FileNotFoundError(workbook_path)
    with zipfile.ZipFile(workbook_path) as archive:
        rows = _xlsx_rows(archive, sheet_name="SAX")
    result: dict[str, CMRMultiWorkbookRecord] = {}
    for row in rows:
        raw_image = row.get("image_path", "")
        raw_anno = row.get("anno_path", row.get("annotation_path", ""))
        case_source = raw_image or raw_anno or row.get("patient_id", "")
        case_id = _strip_nifti_suffix(case_source)
        if not case_id:
            continue
        subject_id = row.get("patient_id", "") or case_id
        result[case_id] = CMRMultiWorkbookRecord(
            case_id=case_id,
            patient_id=subject_id,
            image_path=raw_image,
            mask_path=raw_anno,
            metadata=dict(row),
        )
    return result


def _find_workbook(root: Path) -> Path | None:
    candidates = sorted(root.rglob("*.xlsx"))
    return candidates[0] if candidates else None


def _native_nifti(path: Path) -> tuple[np.ndarray, tuple[float, ...], np.ndarray]:
    try:
        import nibabel as nib
    except ImportError as exc:  # pragma: no cover
        raise ImportError("CMR-MULTI NIfTI loading requires nibabel") from exc
    image = nib.load(str(path))
    array = np.asanyarray(image.dataobj)
    zooms = tuple(float(value) for value in image.header.get_zooms()[:3])
    return array, zooms, np.asarray(image.affine)


class CMRMultiAdapter(Cardiac35DAdapter):
    dataset_name = "cmr_multi"

    def __init__(
        self,
        data_root: str | Path,
        *,
        require_confident: bool = True,
        max_array_cache: int = 2,
    ) -> None:
        self.data_root = Path(data_root)
        self.require_confident = bool(require_confident)
        self.max_array_cache = max(int(max_array_cache), 1)
        self._inference_cache: dict[str, ZTInference] = {}
        self._array_cache: OrderedDict[str, tuple[np.ndarray, np.ndarray, tuple[float, ...], np.ndarray, np.ndarray]] = OrderedDict()
        workbook_path = _find_workbook(self.data_root)
        self.workbook_records = parse_cmr_multi_workbook(workbook_path) if workbook_path else {}
        self.pairs = self._discover_pairs()

    def _discover_pairs(self) -> list[CMRMultiPair]:
        image_paths = sorted(
            path for path in self.data_root.rglob("*")
            if path.is_file()
            and path.name.lower().endswith(_NIFTI_SUFFIXES)
            and path.parent.name.lower() == "image"
        )
        mask_paths = sorted(
            path for path in self.data_root.rglob("*")
            if path.is_file()
            and path.name.lower().endswith(_NIFTI_SUFFIXES)
            and path.parent.name.lower() in {"anno", "annotation", "annotations", "mask", "masks"}
        )
        image_by_case = {_strip_nifti_suffix(path.name): path for path in image_paths}
        mask_by_case = {_strip_nifti_suffix(path.name): path for path in mask_paths}
        if len(image_by_case) != len(image_paths) or len(mask_by_case) != len(mask_paths):
            raise ValueError("CMR-MULTI pairing has duplicate case IDs")
        image_cases = set(image_by_case)
        mask_cases = set(mask_by_case)
        if image_cases != mask_cases:
            missing_masks = sorted(image_cases - mask_cases)
            missing_images = sorted(mask_cases - image_cases)
            raise ValueError(
                "CMR-MULTI pairing is not one-to-one: "
                f"missing_masks={missing_masks[:5]}, missing_images={missing_images[:5]}"
            )
        pairs = []
        for case_id in sorted(image_cases):
            workbook = self.workbook_records.get(case_id)
            subject_id = workbook.patient_id if workbook else case_id
            pairs.append(CMRMultiPair(case_id, subject_id, image_by_case[case_id], mask_by_case[case_id], workbook))
        return pairs

    def _load_pair_arrays(self, pair: CMRMultiPair) -> tuple[np.ndarray, np.ndarray, tuple[float, ...], np.ndarray, np.ndarray]:
        cached = self._array_cache.get(pair.case_id)
        if cached is not None:
            self._array_cache.move_to_end(pair.case_id)
            return cached
        image, spacing, image_affine = _native_nifti(pair.image_path)
        mask, mask_spacing, mask_affine = _native_nifti(pair.mask_path)
        if image.ndim != 3 or mask.ndim != 3:
            raise ValueError(f"CMR-MULTI {pair.case_id} must be [X,Y,N], got {image.shape} and {mask.shape}")
        if image.shape != mask.shape:
            raise ValueError(f"CMR-MULTI {pair.case_id} image/mask shape mismatch: {image.shape} vs {mask.shape}")
        if not np.allclose(spacing, mask_spacing, rtol=1e-5, atol=1e-5):
            raise ValueError(f"CMR-MULTI {pair.case_id} image/mask spacing mismatch")
        value = (np.asarray(image), np.asarray(mask), spacing, image_affine, mask_affine)
        self._array_cache[pair.case_id] = value
        self._array_cache.move_to_end(pair.case_id)
        while len(self._array_cache) > self.max_array_cache:
            self._array_cache.popitem(last=False)
        return value

    def infer_case(self, pair: CMRMultiPair) -> ZTInference:
        cached = self._inference_cache.get(pair.case_id)
        if cached is None:
            _, mask, _, _, _ = self._load_pair_arrays(pair)
            cached = infer_zt(mask)
            self._inference_cache[pair.case_id] = cached
        return cached

    def audit(self) -> list[CMRMultiAuditRecord]:
        records: list[CMRMultiAuditRecord] = []
        for pair in self.pairs:
            try:
                image, mask, _, image_affine, mask_affine = self._load_pair_arrays(pair)
                inference = self.infer_case(pair)
                records.append(
                    CMRMultiAuditRecord(
                        case_id=pair.case_id,
                        subject_id=pair.subject_id,
                        image_path=str(pair.image_path),
                        mask_path=str(pair.mask_path),
                        shape=tuple(int(value) for value in image.shape),
                        image_dtype=str(image.dtype),
                        mask_dtype=str(mask.dtype),
                        spacing=tuple(float(value) for value in self._load_pair_arrays(pair)[2]),
                        unique_labels=tuple(int(value) for value in np.unique(mask)),
                        inference=inference,
                        affine_equal=bool(np.allclose(image_affine, mask_affine, rtol=1e-5, atol=1e-5)),
                    )
                )
            except Exception as exc:
                image_shape: tuple[int, ...] = ()
                image_dtype = ""
                mask_dtype = ""
                labels: tuple[int, ...] = ()
                try:
                    image, mask, _, _, _ = self._load_pair_arrays(pair)
                    image_shape = tuple(int(value) for value in image.shape)
                    image_dtype = str(image.dtype)
                    mask_dtype = str(mask.dtype)
                    labels = tuple(int(value) for value in np.unique(mask))
                except Exception:
                    pass
                records.append(
                    CMRMultiAuditRecord(
                        case_id=pair.case_id,
                        subject_id=pair.subject_id,
                        image_path=str(pair.image_path),
                        mask_path=str(pair.mask_path),
                        shape=image_shape,
                        image_dtype=image_dtype,
                        mask_dtype=mask_dtype,
                        unique_labels=labels,
                        inference=None,
                        error=str(exc),
                    )
                )
        return records

    def discover_units(
        self,
        *,
        split: str | None = None,
        supervised_only: bool = True,
    ) -> list[CardiacUnit]:
        del supervised_only
        units: list[CardiacUnit] = []
        for pair in self.pairs:
            if split and pair.workbook and pair.workbook.metadata.get("split") not in {None, "", split}:
                continue
            try:
                inference = self.infer_case(pair)
            except Exception:
                if self.require_confident:
                    continue
                raise
            if self.require_confident and not inference.accepted:
                continue
            for time in range(inference.t):
                units.append(
                    CardiacUnit(
                        dataset=self.dataset_name,
                        case_id=f"{pair.case_id}:t{time:02d}",
                        subject_id=pair.subject_id,
                        image_ref=pair,
                        mask_ref=pair,
                        split=split or "",
                        time_index=time,
                        metadata={
                            "parent_case_id": pair.case_id,
                            "shape": (inference.z, 0, 0),
                            "zt_confidence": inference.confidence,
                            "zt_status": inference.status,
                            "zt_score": inference.score,
                        },
                    )
                )
        return units

    def load_unit(self, unit: CardiacUnit) -> LoadedCardiacUnit:
        pair = unit.image_ref if isinstance(unit.image_ref, CMRMultiPair) else next(
            item for item in self.pairs if item.case_id == str(unit.metadata["parent_case_id"])
        )
        inference = self.infer_case(pair)
        if self.require_confident and not inference.accepted:
            raise ZTInferenceError(f"{pair.case_id} Z/T inference is not accepted")
        image, mask, spacing, _, _ = self._load_pair_arrays(pair)
        image_xyzt = reconstruct_zt(image, inference, require_accepted=self.require_confident)
        mask_xyzt = reconstruct_zt(mask, inference, require_accepted=self.require_confident)
        time = int(unit.time_index)
        if time < 0 or time >= inference.t:
            raise IndexError(f"time index {time} outside [0,{inference.t})")
        image_zhw = np.transpose(image_xyzt[:, :, :, time], (2, 1, 0)).astype(np.float32, copy=False)
        mask_zhw = np.transpose(mask_xyzt[:, :, :, time], (2, 1, 0))
        return LoadedCardiacUnit(
            image_zhw=image_zhw,
            mask_zhw=self.remap_labels(mask_zhw),
            spacing=None,
            metadata={
                "source_spacing": tuple(float(value) for value in spacing),
                "zt": (inference.z, inference.t),
                "time_index": time,
                "native_orientation": True,
            },
        )

    def remap_labels(self, mask: np.ndarray) -> np.ndarray:
        return remap_labels(mask, CMR_MULTI_RAW_TO_COMMON)
