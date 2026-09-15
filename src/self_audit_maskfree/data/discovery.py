"""Image-only discovery and manifest construction for the mask-free pipeline.

The manifest has two different identities by design:

``patient_id``
    The only image-derived identity used for a train/dev/test split. Source,
    frame and slice identifiers never participate in the split hash.

``volume_id`` / ``unit_id``
    A stable source-file-plus-frame identity and its individual Z slice. Every
    acquired frame and every acquired slice is enumerated, including frames of
    a genuine 4-D cine. The current observation model still resolves such a
    source to the weaker ``spatial_predictive`` protocol until a fit-time
    temporal transport implementation exists.

Discovery never opens annotation paths. It fingerprints each selected image
file once for exact-resume provenance and computes a separate frame-content
fingerprint only to identify geometry-and-image-identical re-exports. Neither
fingerprint seeds a partition or a patient split; the latter are determined
only by the declared study grid, patient identity and global seed.
"""
from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np

from .firewall import assert_image_only, forbidden_reason, has_allowed_suffix
from .geometry import read_geometry
from .partition import partition_identity
from ..progress import current_progress

SCHEMA_VERSION = "maskfree150.data.v2"
SUPPORTED_DATASETS = ("acdc", "mnms")
SPLITS = ("train", "dev", "test")

SPLIT_RATIOS = {"train": 0.70, "dev": 0.15, "test": 0.15}
MIN_CINE_FRAMES = 8
IMPLEMENTED_PROTOCOLS = ("spatial_predictive",)
# A source header may be rounded differently by a writer as long as the
# resulting native voxel-coordinate map is effectively unchanged.  This is a
# coordinate tolerance, not a permission to resample or to ignore a grid.
HEADER_GRID_TOLERANCE_VOXELS = 1e-4

_OFFICIAL_TEST_DIRS = {"testing", "test"}
_OFFICIAL_TRAIN_DIRS = {"training", "train"}
_PREPROCESSED_DIR_TOKENS = {"preprocessed_data", "preprocessed", "volumes"}

_FRAME_SUFFIX = re.compile(r"_(?:frame|time)\d+$", re.IGNORECASE)
_MNMS_TIME_SUFFIX = re.compile(r"_t\d+$", re.IGNORECASE)
_SERIES_SUFFIX = re.compile(r"_(4d|sa|la|ed|es|cine)$", re.IGNORECASE)


class DataRootError(FileNotFoundError):
    """Raised when the requested dataset root does not exist locally."""


class ProtocolUnavailableError(ValueError):
    """Raised when a protocol is requested that this repository cannot honour."""


class MixedStudyGeometryError(ValueError):
    """Raised when one study cannot share one observation-grid role map.

    ``details`` is intentionally JSON-serializable so the preparation CLI can
    persist a failed inventory without turning a geometry conflict into an
    opaque traceback.  ``diagnostic`` is retained as an alias for callers that
    used the earlier review terminology.
    """

    def __init__(
        self,
        message: str,
        *,
        details: dict[str, Any] | None = None,
        diagnostic: dict[str, Any] | None = None,
    ) -> None:
        if details is None:
            details = diagnostic if diagnostic is not None else {}
        self.details = details
        self.diagnostic = details
        super().__init__(message)


def _strip_suffixes(stem: str) -> str:
    """Strip only known frame/series suffixes from a file stem.

    M&Ms exports commonly carry ``_tXX``. Removing it here is essential: if
    ``case001_t01`` and ``case001_t02`` become two split identities, the same
    patient can enter both train and test. Unknown suffixes remain part of the
    identity rather than being guessed away.
    """
    name = stem
    for _ in range(8):
        updated = _FRAME_SUFFIX.sub("", name)
        updated = _MNMS_TIME_SUFFIX.sub("", updated)
        updated = _SERIES_SUFFIX.sub("", updated)
        if updated == name:
            break
        name = updated
    return name


def _file_stem(path: Path) -> str:
    name = path.name
    for suffix in (".nii.gz", ".nii", ".npy"):
        if name.lower().endswith(suffix):
            return name[: -len(suffix)]
    return path.stem


def patient_id_from_path(path: str | Path) -> str:
    """Derive a patient identity from file naming only."""
    p = Path(path)
    stem = _file_stem(p)
    base = _strip_suffixes(stem)
    return base or stem


def official_split_from_path(path: str | Path) -> str | None:
    """Return ``test``/``train`` when a folder explicitly declares a cohort."""
    parts = [part.lower() for part in Path(path).parts]
    if any(part in _OFFICIAL_TEST_DIRS for part in parts):
        return "test"
    if any(part in _OFFICIAL_TRAIN_DIRS for part in parts):
        return "train"
    return None


def _is_preprocessed(path: Path) -> bool:
    return any(part.lower() in _PREPROCESSED_DIR_TOKENS for part in path.parts)


def iter_image_files(root: Path) -> Iterable[Path]:
    """Yield selected image containers while skipping annotation trees."""
    for path in sorted(root.rglob("*")):
        if path.is_dir() or not has_allowed_suffix(path):
            continue
        if forbidden_reason(path) is not None:
            continue
        yield path


def _spatial_shape(shape: Iterable[int], depth_axis: int) -> list[int]:
    dims = [int(s) for s in shape]
    if len(dims) not in (3, 4):
        raise ValueError(f"expected rank-3 or rank-4 image, got shape {dims}")
    if depth_axis < 0 or depth_axis >= 3:
        raise ValueError(f"depth_axis must be in 0..2, got {depth_axis}")
    if dims[depth_axis] <= 0:
        raise ValueError(f"depth axis {depth_axis} has non-positive extent in {dims}")
    spatial = [dims[index] for index in range(3) if index != depth_axis]
    if any(size <= 0 for size in spatial):
        raise ValueError(f"spatial axes have a non-positive extent in {dims}")
    return spatial


def _frame_metadata(geometry: dict[str, Any], depth_axis: int) -> dict[str, Any]:
    """Describe every frame without treating ED/ES exports as a cine."""
    shape = [int(s) for s in geometry["shape"]]
    if len(shape) == 4:
        frame_axis = 3
        n_frames = int(shape[frame_axis])
        if n_frames <= 0:
            raise ValueError(f"frame axis has non-positive extent in {shape}")
        zooms = geometry.get("zooms") or []
        frame_duration = float(zooms[frame_axis]) if len(zooms) > frame_axis else 0.0
        units = geometry.get("xyzt_units") or []
        time_unit = str(units[1]).lower() if len(units) >= 2 and units[1] else "unknown"
        times_reliable = bool(
            np.isfinite(frame_duration)
            and frame_duration > 0.0
            and time_unit not in ("unknown", "", "0", "none")
        )
        return {
            "is_multi_frame": True,
            "frame_axis": frame_axis,
            "n_frames": n_frames,
            "frame_indices": list(range(n_frames)),
            "frame_duration": frame_duration if times_reliable else None,
            "frame_times": [float(index * frame_duration) for index in range(n_frames)]
            if times_reliable
            else None,
            "time_unit": time_unit,
            "times_reliable": times_reliable,
        }
    return {
        "is_multi_frame": False,
        "frame_axis": None,
        "n_frames": 1,
        # A single-frame source still has an explicit frame identity. This
        # makes volume_id uniformly source+frame for 3-D and 4-D inputs.
        "frame_indices": [0],
        "frame_duration": None,
        "frame_times": [0.0],
        "time_unit": "unknown",
        "times_reliable": False,
    }


def _cine_eligible(frames: dict[str, Any], geometry: dict[str, Any]) -> tuple[bool, str]:
    if not frames["is_multi_frame"]:
        return False, "single-frame acquisition"
    if frames["n_frames"] < MIN_CINE_FRAMES:
        return False, f"only {frames['n_frames']} frames, need >= {MIN_CINE_FRAMES}"
    if not frames["times_reliable"]:
        return False, "frame times or time units are not reliable in the header"
    if geometry["native_geometry"] != "available":
        return False, geometry["native_geometry_reason"] or "native geometry unavailable"
    return True, "multi-frame acquisition with reliable times and geometry"


def _hash_fraction(*parts: str) -> float:
    digest = hashlib.blake2b("|".join(parts).encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") / float(2**64)


def assign_splits(
    patients: list[str], *, dataset: str, seed: int, official: dict[str, str]
) -> tuple[dict[str, str], dict[str, Any]]:
    """Assign patient-disjoint splits from patient identity and the seed only."""
    assignment: dict[str, str] = {}
    official_test = sorted(p for p in patients if official.get(p) == "test")
    for patient in official_test:
        assignment[patient] = "test"

    remaining = sorted(p for p in patients if p not in assignment)
    # The source/frame and full-content fingerprints are deliberately absent.
    ranked = sorted(remaining, key=lambda p: (_hash_fraction(dataset, str(seed), p), p))

    if official_test:
        total = SPLIT_RATIOS["train"] + SPLIT_RATIOS["dev"]
        targets = [("dev", SPLIT_RATIOS["dev"] / total)]
        rule = "official_test_folder_plus_rank_hashed_train_dev"
    else:
        targets = [("test", SPLIT_RATIOS["test"]), ("dev", SPLIT_RATIOS["dev"])]
        rule = "rank_hashed_patient_id_70_15_15"

    cursor = 0
    n = len(ranked)
    for name, ratio in targets:
        if n == 0:
            break
        count = round(ratio * n)
        if n >= len(targets) + 1:
            count = max(1, count)
        count = min(count, max(0, n - cursor - 1))
        for patient in ranked[cursor : cursor + count]:
            assignment[patient] = name
        cursor += count
    for patient in ranked[cursor:]:
        assignment[patient] = "train"

    provenance = {
        "rule": rule,
        "seed": int(seed),
        "ratios": dict(SPLIT_RATIOS),
        "official_test_membership": bool(official_test),
        "official_test_patients": len(official_test),
        "patient_level_disjoint": True,
        "split_identity": "patient_id",
        "selection_inputs": ["patient_id", "seed"],
        "annotation_inputs": [],
        "content_fingerprint_used": False,
    }
    return assignment, provenance


def _source_fingerprint(path: Path) -> str:
    """Hash source bytes once for exact-resume provenance."""
    digest = hashlib.sha256()
    with assert_image_only(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_frame(path: Path, frame_index: int) -> np.ndarray:
    """Read one image frame for duplicate detection through the image firewall."""
    p = assert_image_only(path)
    name = p.name.lower()
    if name.endswith((".nii", ".nii.gz")):
        import nibabel as nib

        data = nib.load(str(p)).dataobj
    elif name.endswith(".npy"):
        data = np.load(p, mmap_mode="r")
    else:  # pragma: no cover - iter_image_files filters this
        raise ValueError(f"unsupported image container {p}")
    if len(data.shape) == 4:
        if not 0 <= frame_index < int(data.shape[3]):
            raise ValueError(f"frame {frame_index} outside source shape {data.shape}")
        array = np.asarray(data[..., frame_index])
    elif len(data.shape) == 3:
        if frame_index != 0:
            raise ValueError(f"single-frame source does not have frame {frame_index}")
        array = np.asarray(data)
    else:
        raise ValueError(f"expected rank-3 or rank-4 source, got {data.shape}")
    if not np.issubdtype(array.dtype, np.number):
        raise ValueError(f"image frame has non-numeric dtype {array.dtype}")
    # Duplicate detection must compare the stored frame representation.  In
    # particular, do not convert float64 to float32 or replace non-finite
    # values: either operation could collapse two different stored frames.
    return np.ascontiguousarray(array)


def _frame_fingerprint(array: np.ndarray) -> str:
    digest = hashlib.blake2b(digest_size=20)
    array = np.ascontiguousarray(array)
    digest.update(
        json.dumps(
            {"shape": list(array.shape), "dtype": np.dtype(array.dtype).str},
            sort_keys=True,
        ).encode()
    )
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _frame_stored_shape(entry: dict[str, Any]) -> tuple[int, ...]:
    """Return the exact rank-3 shape of one stored frame.

    A rank-4 cine and its rank-3 frame export deliberately have different
    *source* shapes but the same extracted frame shape.  The latter is the
    shape that participates in the exact image-content duplicate proof.
    """
    geometry = entry["geometry"]
    shape = [int(s) for s in geometry["shape"]]
    if len(shape) == 4:
        frame_axis = int(entry["frames"].get("frame_axis", 3))
        if frame_axis < 0 or frame_axis >= len(shape):
            raise ValueError(f"frame axis {frame_axis} is outside source shape {shape}")
        shape.pop(frame_axis)
    return tuple(shape)


def _geometry_descriptor(entry: dict[str, Any]) -> dict[str, Any]:
    """Return a complete, JSON-safe geometry descriptor for diagnostics."""
    geometry = entry["geometry"]
    depth_axis = int(geometry.get("depth_axis", 2))
    shape = [int(s) for s in geometry.get("shape", [])]
    return {
        "source_path": entry.get("path"),
        "relative_path": entry.get("relative_path"),
        "source_format": geometry.get("source_format"),
        "shape": shape,
        "full_shape": list(shape),
        "stored_frame_shape": list(_frame_stored_shape(entry)),
        "spatial_shape": _spatial_shape(shape, depth_axis),
        "depth_axis": depth_axis,
        "native_geometry": geometry.get("native_geometry"),
        "native_geometry_reason": geometry.get("native_geometry_reason"),
        "native_affine": geometry.get("native_affine"),
        "orientation": geometry.get("orientation"),
        "zooms": geometry.get("zooms"),
        "xyzt_units": geometry.get("xyzt_units"),
        "spatial_unit": geometry.get("spatial_unit"),
        "spacing_mm": geometry.get("spacing_mm"),
    }


def _affine_voxel_displacement(
    left: dict[str, Any], right: dict[str, Any], source_shape: list[int]
) -> float | None:
    """Measure the largest corner displacement in voxel coordinates.

    World coordinates of every corner are compared, then expressed in both
    source voxel bases.  Taking the larger L2 displacement is conservative for
    small scale/shear changes while remaining independent of the units label.
    No resampling is performed.
    """
    left_affine = np.asarray(left.get("native_affine"), dtype=np.float64)
    right_affine = np.asarray(right.get("native_affine"), dtype=np.float64)
    if left_affine.shape != (4, 4) or right_affine.shape != (4, 4):
        return None
    if not np.all(np.isfinite(left_affine)) or not np.all(np.isfinite(right_affine)):
        return None
    left_linear = left_affine[:3, :3]
    right_linear = right_affine[:3, :3]
    if abs(float(np.linalg.det(left_linear))) <= 0.0:
        return None
    if abs(float(np.linalg.det(right_linear))) <= 0.0:
        return None
    if len(source_shape) < 3:
        return None
    spatial_shape = [int(size) for size in source_shape[:3]]
    if any(size <= 0 for size in spatial_shape):
        return None
    corners = np.asarray(
        [
            [x, y, z]
            for x in (0, spatial_shape[0] - 1)
            for y in (0, spatial_shape[1] - 1)
            for z in (0, spatial_shape[2] - 1)
        ],
        dtype=np.float64,
    )
    left_world = corners @ left_linear.T + left_affine[:3, 3]
    right_world = corners @ right_linear.T + right_affine[:3, 3]
    try:
        # Compare in each source's voxel coordinates.  The maximum of both
        # directions is robust to a small scale change near a long corner.
        left_delta = np.linalg.solve(
            left_linear, (right_world - left_world).T
        ).T
        right_delta = np.linalg.solve(
            right_linear, (left_world - right_world).T
        ).T
    except np.linalg.LinAlgError:
        return None
    displacement = np.concatenate(
        [np.linalg.norm(left_delta, axis=1), np.linalg.norm(right_delta, axis=1)]
    )
    if not np.all(np.isfinite(displacement)):
        return None
    return float(np.max(displacement))


def _geometry_key(geometry: dict[str, Any], depth_axis: int) -> str:
    """Exact geometry identity for duplicate detection and study-grid checks."""
    payload = {
        "source_format": geometry.get("source_format"),
        "spatial_shape": _spatial_shape(geometry.get("shape", []), depth_axis),
        "depth_axis": int(depth_axis),
        "native_geometry": geometry.get("native_geometry"),
        "native_affine": geometry.get("native_affine"),
        "orientation": geometry.get("orientation"),
        "zooms": (geometry.get("zooms") or [])[:3],
        "spatial_unit": geometry.get("spatial_unit"),
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)


def _study_geometry_key(geometry: dict[str, Any], depth_axis: int) -> str:
    """Grid key used to ensure one exact study-scoped role map."""
    payload = {
        "source_format": geometry.get("source_format"),
        "spatial_shape": _spatial_shape(geometry.get("shape", []), depth_axis),
        "depth_axis": int(depth_axis),
        "native_geometry": geometry.get("native_geometry"),
        "native_affine": geometry.get("native_affine"),
        "orientation": geometry.get("orientation"),
        "spatial_unit": geometry.get("spatial_unit"),
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)


def _volume_id(dataset: str, relative_path: str, frame_index: int) -> str:
    """Stable, path-safe source-plus-frame identity."""
    source_digest = hashlib.blake2b(
        f"{dataset}|{relative_path}".encode(), digest_size=12
    ).hexdigest()
    return f"{dataset}:vol-{source_digest}:t{int(frame_index):04d}"


def _prepare_entry(
    path: Path,
    root_path: Path,
    geometry: dict[str, Any],
    depth_axis: int,
    dataset: str,
    *,
    progress: Any | None = None,
    file_index: int | None = None,
    files_total: int | None = None,
) -> dict[str, Any]:
    progress = current_progress() if progress is None else progress
    progress_details = {"path": str(path)}
    if file_index is not None:
        progress_details["file_index"] = int(file_index)
    if files_total is not None:
        progress_details["files_total"] = int(files_total)
    frames = _frame_metadata(geometry, depth_axis)
    with progress.stage("discovery.hash", **progress_details):
        source_fingerprint = _source_fingerprint(path)
        progress.event(
            "discovery.source_hash",
            **progress_details,
            source_fingerprint=source_fingerprint,
        )
    frame_fingerprints: dict[int, str] = {}
    with progress.stage(
        "discovery.frame_fingerprint",
        **progress_details,
        frame_count=len(frames["frame_indices"]),
    ):
        for frame in frames["frame_indices"]:
            frame = int(frame)
            # Set the live context before decompression/hash work so a
            # heartbeat identifies the frame currently being read.
            progress.update(**progress_details, frame_index=frame)
            progress.event(
                "discovery.frame_fingerprint_start",
                **progress_details,
                frame_index=frame,
            )
            fingerprint = _frame_fingerprint(_load_frame(path, frame))
            frame_fingerprints[frame] = fingerprint
            progress.event(
                "discovery.frame_fingerprint",
                **progress_details,
                frame_index=frame,
                stored_frame_shape=list(_frame_stored_shape({"geometry": geometry, "frames": frames})),
                frame_fingerprint=fingerprint,
            )
    relative_path = str(path.relative_to(root_path))
    patient_id = patient_id_from_path(path)
    return {
        "source_id": f"{path.parent.name}:{path.name}",
        "study_id": f"{dataset}:{patient_id}",
        "patient_id": patient_id,
        "dataset": dataset,
        "path": str(path),
        "relative_path": relative_path,
        "geometry": geometry,
        "frames": frames,
        "source_fingerprint": source_fingerprint,
        "frame_fingerprints": frame_fingerprints,
        "geometry_key": _geometry_key(geometry, depth_axis),
        "official_folder": official_split_from_path(path),
        "preprocessed_root": _is_preprocessed(path),
    }


def _dedup(
    entries: list[dict[str, Any]], *, dataset: str, progress: Any | None = None
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Collapse only cross-file, stored-shape-and-image-identical frame exports.

    All frames from one source file remain counted, even when two adjacent cine
    frames happen to contain identical pixels. A single-frame export is
    collapsed only when its extracted frame has exactly the same stored shape
    and content as a frame from another source. Header-only geometry changes do
    not disprove this content identity (and are recorded in the duplicate
    evidence); unrelated non-identical series from the same patient survive.
    """
    progress = current_progress() if progress is None else progress
    refs: list[dict[str, Any]] = []
    for entry in entries:
        for frame_index in entry["frames"]["frame_indices"]:
            frame_index = int(frame_index)
            refs.append(
                {
                    "entry": entry,
                    "frame_index": frame_index,
                    "volume_id": _volume_id(dataset, entry["relative_path"], frame_index),
                    "frame_fingerprint": entry["frame_fingerprints"][frame_index],
                    "counted": True,
                    "duplicate_of": None,
                }
            )

    groups: dict[tuple[str, tuple[int, ...], str], list[dict[str, Any]]] = {}
    for ref in refs:
        entry = ref["entry"]
        # A frame fingerprint already contains shape, but retaining the shape
        # explicitly makes the proof visible and prevents future hash changes
        # from accidentally broadening a duplicate group.
        key = (
            entry["patient_id"],
            _frame_stored_shape(entry),
            ref["frame_fingerprint"],
        )
        groups.setdefault(key, []).append(ref)

    duplicates: list[dict[str, Any]] = []
    with progress.stage("discovery.dedup", frame_candidates=len(refs)):
        for group in groups.values():
            source_paths = {ref["entry"]["path"] for ref in group}
            if len(source_paths) <= 1:
                continue

            # A matching frame from one 4-D cine and one 3-D export is the
            # only case in which a header-different geometry can be called a
            # re-export.  Exact-geometry frame exports remain eligible, while
            # two 4-D series are retained even if one frame happens to match;
            # that frame equality does not prove the acquisitions are one
            # series.
            ordered = sorted(
                group,
                key=lambda ref: (
                    0 if ref["entry"]["frames"]["is_multi_frame"] else 1,
                    ref["entry"]["relative_path"],
                    ref["frame_index"],
                ),
            )
            canonical: list[dict[str, Any]] = []
            for ref in ordered:
                entry = ref["entry"]
                candidates = [
                    primary
                    for primary in canonical
                    if primary["entry"]["path"] != entry["path"]
                    and not (
                        primary["entry"]["frames"]["is_multi_frame"]
                        and entry["frames"]["is_multi_frame"]
                    )
                    and (
                        primary["entry"]["geometry_key"] == entry["geometry_key"]
                        or (
                            primary["entry"]["frames"]["is_multi_frame"]
                            != entry["frames"]["is_multi_frame"]
                            and primary["entry"]["geometry"].get("source_format")
                            == "nifti"
                            and entry["geometry"].get("source_format") == "nifti"
                            and primary["entry"]["geometry"].get("native_geometry")
                            == "available"
                            and entry["geometry"].get("native_geometry") == "available"
                        )
                    )
                ]
                if not candidates:
                    canonical.append(ref)
                    continue
                primary_ref = min(
                    candidates,
                    key=lambda candidate: (
                        0 if candidate["entry"]["frames"]["is_multi_frame"] else 1,
                        candidate["entry"]["relative_path"],
                        candidate["frame_index"],
                    ),
                )
                ref["counted"] = False
                ref["duplicate_of"] = primary_ref["volume_id"]
                primary_entry = primary_ref["entry"]
                duplicate_entry = ref["entry"]
                geometry_match = (
                    primary_entry["geometry_key"] == duplicate_entry["geometry_key"]
                )
                cross_rank_export = (
                    primary_entry["frames"]["is_multi_frame"]
                    != duplicate_entry["frames"]["is_multi_frame"]
                )
                rationale = (
                    "stored frame shape and content are byte-identical after exact "
                    "dtype-preserving extraction; rank-4 cine is preferred over its "
                    "rank-3 frame re-export despite header geometry differences"
                    if cross_rank_export and not geometry_match
                    else "stored frame shape and content are byte-identical and the "
                    "source geometries also match"
                )
                duplicate = {
                    "volume_id": ref["volume_id"],
                    "source_path": ref["entry"]["path"],
                    "relative_path": ref["entry"]["relative_path"],
                    "frame_index": ref["frame_index"],
                    "duplicate_of": primary_ref["volume_id"],
                    # Keep a stable broad reason for old inventory readers;
                    # ``rationale`` records exactly what was proven.
                    "reason": "image_identical_frame_export",
                    "rationale": rationale,
                    "stored_frame_shape": list(_frame_stored_shape(duplicate_entry)),
                    "stored_frame_fingerprint": ref["frame_fingerprint"],
                    "geometry_match": bool(geometry_match),
                    "cross_rank_export": bool(cross_rank_export),
                    "primary_source_path": primary_entry["path"],
                    "primary_relative_path": primary_entry["relative_path"],
                    "primary_frame_index": primary_ref["frame_index"],
                    "primary_geometry": _geometry_descriptor(primary_entry),
                    "duplicate_geometry": _geometry_descriptor(duplicate_entry),
                }
                duplicates.append(duplicate)
                progress.event("discovery.duplicate", **duplicate)
    return [ref for ref in refs if ref["counted"]], sorted(duplicates, key=lambda x: x["volume_id"])


def _grid_comparison(
    left: dict[str, Any],
    right: dict[str, Any],
    *,
    depth_axis: int,
    tolerance_voxels: float,
) -> dict[str, Any]:
    """Compare two source grids without modifying either source affine."""
    left_geometry = left["geometry"]
    right_geometry = right["geometry"]
    left_shape = [int(s) for s in left_geometry["shape"]]
    right_shape = [int(s) for s in right_geometry["shape"]]
    left_stored_shape = list(_frame_stored_shape(left))
    right_stored_shape = list(_frame_stored_shape(right))
    left_spatial = _spatial_shape(left_shape, depth_axis)
    right_spatial = _spatial_shape(right_shape, depth_axis)
    left_orientation = left_geometry.get("orientation")
    right_orientation = right_geometry.get("orientation")
    left_status = left_geometry.get("native_geometry")
    right_status = right_geometry.get("native_geometry")

    blocking: list[dict[str, Any]] = []
    header_differences: list[dict[str, Any]] = []

    if left_spatial != right_spatial:
        blocking.append(
            {
                "code": "spatial_shape_difference",
                "left": list(left_spatial),
                "right": list(right_spatial),
            }
        )
    if left_stored_shape != right_stored_shape:
        blocking.append(
            {
                "code": "stored_frame_shape_difference",
                "left": left_stored_shape,
                "right": right_stored_shape,
            }
        )
    left_depth = int(left_shape[depth_axis])
    right_depth = int(right_shape[depth_axis])
    if left_depth != right_depth:
        blocking.append(
            {
                "code": "depth_extent_difference",
                "left": left_depth,
                "right": right_depth,
            }
        )
    if int(left_geometry.get("depth_axis", depth_axis)) != int(
        right_geometry.get("depth_axis", depth_axis)
    ):
        blocking.append(
            {
                "code": "depth_axis_difference",
                "left": int(left_geometry.get("depth_axis", depth_axis)),
                "right": int(right_geometry.get("depth_axis", depth_axis)),
            }
        )
    if left_orientation != right_orientation:
        blocking.append(
            {
                "code": "orientation_difference",
                "left": left_orientation,
                "right": right_orientation,
            }
        )
    if left_geometry.get("source_format") != right_geometry.get("source_format"):
        # A known NIfTI affine and an affine-less NPY source cannot be proven
        # to share the same role-coordinate map, even when their array shape
        # happens to agree.
        blocking.append(
            {
                "code": "source_format_difference",
                "left": left_geometry.get("source_format"),
                "right": right_geometry.get("source_format"),
            }
        )
    left_unit = left_geometry.get("spatial_unit")
    right_unit = right_geometry.get("spatial_unit")
    if left_unit != right_unit:
        unit_difference = {
            "code": "spatial_unit_difference",
            "left": left_unit,
            "right": right_unit,
        }
        # A missing/unknown declaration is also not proof of a shared physical
        # map.  Two sources must make the same spatial-unit claim; this keeps
        # millimetre metrics and native role coordinates fail-closed.
        blocking.append(unit_difference)

    max_displacement: float | None = None
    if left_status != right_status:
        blocking.append(
            {
                "code": "native_geometry_availability_difference",
                "left": left_status,
                "right": right_status,
            }
        )
    elif left_status == "available":
        max_displacement = _affine_voxel_displacement(
            left_geometry,
            right_geometry,
            left_shape[:3],
        )
        if max_displacement is None:
            blocking.append(
                {
                    "code": "affine_comparison_unavailable",
                    "left": left_geometry.get("native_affine"),
                    "right": right_geometry.get("native_affine"),
                }
            )
        elif max_displacement > tolerance_voxels:
            blocking.append(
                {
                    "code": "affine_voxel_displacement_exceeds_tolerance",
                    "measured_max_voxel_displacement": max_displacement,
                    "tolerance_voxels": tolerance_voxels,
                }
            )

    # Header units/zooms are retained as provenance.  They do not alter the
    # index-to-index role map when the affine corner displacement is within
    # tolerance, but the difference must remain inspectable for physical-metric
    # review.
    if left_geometry.get("xyzt_units") != right_geometry.get("xyzt_units"):
        header_differences.append(
            {
                "code": "xyzt_units_difference",
                "left": left_geometry.get("xyzt_units"),
                "right": right_geometry.get("xyzt_units"),
            }
        )
    if left_geometry.get("zooms") != right_geometry.get("zooms"):
        header_differences.append(
            {
                "code": "header_zooms_difference",
                "left": left_geometry.get("zooms"),
                "right": right_geometry.get("zooms"),
            }
        )

    return {
        "left_path": left["path"],
        "right_path": right["path"],
        "left_relative_path": left.get("relative_path"),
        "right_relative_path": right.get("relative_path"),
        "left_shape": list(left_shape),
        "right_shape": list(right_shape),
        "left_stored_frame_shape": left_stored_shape,
        "right_stored_frame_shape": right_stored_shape,
        "left_spatial_shape": list(left_spatial),
        "right_spatial_shape": list(right_spatial),
        "left_depth_axis": int(left_geometry.get("depth_axis", depth_axis)),
        "right_depth_axis": int(right_geometry.get("depth_axis", depth_axis)),
        "left_orientation": left_orientation,
        "right_orientation": right_orientation,
        "left_native_geometry": left_status,
        "right_native_geometry": right_status,
        "left_native_affine": left_geometry.get("native_affine"),
        "right_native_affine": right_geometry.get("native_affine"),
        "left_xyzt_units": left_geometry.get("xyzt_units"),
        "right_xyzt_units": right_geometry.get("xyzt_units"),
        "left_spatial_unit": left_geometry.get("spatial_unit"),
        "right_spatial_unit": right_geometry.get("spatial_unit"),
        "max_voxel_displacement": max_displacement,
        "measured_max_voxel_displacement": max_displacement,
        "tolerance_voxels": tolerance_voxels,
        "displacement_metric": "max_l2_over_8_spatial_corners",
        "differences": blocking,
        "header_differences": header_differences,
        "compatible": not blocking,
    }


def _ensure_study_grids(
    counted: list[dict[str, Any]],
    *,
    progress: Any | None = None,
    tolerance_voxels: float = HEADER_GRID_TOLERANCE_VOXELS,
) -> dict[str, Any]:
    """Validate one shared grid map per study before role partitioning.

    Exact duplicate frame exports are removed before this function runs.  For
    the remaining sources, only a bounded affine-header precision difference is
    accepted.  Spatial shape, depth axis, orientation and affine availability
    remain hard requirements; no resampling or per-unit partition is created.
    """
    progress = current_progress() if progress is None else progress
    by_study: dict[str, dict[str, dict[str, Any]]] = {}
    for ref in counted:
        entry = ref["entry"]
        by_study.setdefault(entry["study_id"], {})[entry["path"]] = entry

    checks: dict[str, Any] = {
        "status": "passed",
        "tolerance_voxels": float(tolerance_voxels),
        "displacement_metric": "max_l2_over_8_spatial_corners",
        "no_resampling": True,
        "studies": {},
    }
    with progress.stage(
        "discovery.grid_check",
        study_count=len(by_study),
        source_count=sum(len(entries) for entries in by_study.values()),
        tolerance_voxels=float(tolerance_voxels),
    ):
        for study_id in sorted(by_study):
            entries = sorted(
                by_study[study_id].values(), key=lambda entry: entry["relative_path"]
            )
            comparisons: list[dict[str, Any]] = []
            incompatible: list[dict[str, Any]] = []
            for index, left in enumerate(entries):
                for right in entries[index + 1 :]:
                    comparison = _grid_comparison(
                        left,
                        right,
                        depth_axis=int(left["geometry"].get("depth_axis", 2)),
                        tolerance_voxels=float(tolerance_voxels),
                    )
                    comparisons.append(comparison)
                    progress.event("discovery.grid_comparison", **comparison)
                    if not comparison["compatible"]:
                        incompatible.append(comparison)

            if incompatible:
                details = {
                    "code": "mixed_study_native_grid",
                    "study_id": study_id,
                    "tolerance_voxels": float(tolerance_voxels),
                    "displacement_metric": "max_l2_over_8_spatial_corners",
                    "no_resampling": True,
                    "geometries": [_geometry_descriptor(entry) for entry in entries],
                    "differences": incompatible,
                    "mismatch_explanation": (
                        "The study contains source geometries whose spatial shape, "
                        "depth axis, orientation, native-affine availability, or "
                        "corner displacement cannot share one exact role map."
                    ),
                }
                message = (
                    f"study {study_id!r} has mixed native grid/affine/orientation "
                    f"across {len(entries)} source geometries; refusing discovery "
                    "because a single study-scoped observation partition cannot be "
                    f"mapped within {tolerance_voxels:g} voxel ({json.dumps(details, sort_keys=True, default=str)})"
                )
                progress.event("discovery.grid_conflict", **details)
                raise MixedStudyGeometryError(message, details=details)

            reference = entries[0] if entries else None
            study_check = {
                "status": "compatible",
                "source_paths": [entry["path"] for entry in entries],
                "n_source_geometries": len(entries),
                "comparisons": comparisons,
                "tolerance_voxels": float(tolerance_voxels),
                "displacement_metric": "max_l2_over_8_spatial_corners",
                "no_resampling": True,
            }
            checks["studies"][study_id] = study_check

            for entry in entries:
                own_comparisons = [
                    comparison
                    for comparison in comparisons
                    if comparison["left_path"] == entry["path"]
                    or comparison["right_path"] == entry["path"]
                ]
                displacements = [
                    float(comparison["max_voxel_displacement"])
                    for comparison in own_comparisons
                    if comparison["max_voxel_displacement"] is not None
                ]
                measured = max(displacements, default=(0.0 if entry["geometry"].get("native_geometry") == "available" else None))
                unit_differences = [
                    difference
                    for comparison in own_comparisons
                    for difference in comparison["header_differences"]
                ]
                if entry["geometry"].get("native_geometry") != "available":
                    rationale = (
                        "same stored-grid source format, spatial shape, depth axis and "
                        "orientation; no native affine comparison was available; no "
                        "resampling performed"
                    )
                elif measured == 0.0:
                    rationale = (
                        "same native spatial shape, depth axis and orientation; affine "
                        "corner displacement is 0 voxel; original source affine retained"
                    )
                else:
                    rationale = (
                        "same native spatial shape, depth axis and orientation; measured "
                        f"header corner displacement {measured:.12g} voxel is within "
                        f"the {tolerance_voxels:.12g}-voxel tolerance; no resampling "
                        "performed; original source affine retained per record"
                    )
                if unit_differences:
                    rationale += (
                        "; declared header units/zooms differ and remain recorded as "
                        "physical-spacing provenance"
                    )
                entry["study_grid_compatibility"] = {
                    "status": "compatible",
                    "reference_source_path": reference["path"] if reference else None,
                    "tolerance_voxels": float(tolerance_voxels),
                    "measured_max_voxel_displacement": measured,
                    "max_voxel_displacement": measured,
                    "displacement_metric": "max_l2_over_8_spatial_corners",
                    "rationale": rationale,
                    "header_differences": unit_differences,
                    "comparison_count": len(own_comparisons),
                    "no_resampling": True,
                }
    return checks


def discover_dataset(
    root: str | Path,
    dataset: str,
    *,
    seed: int = 42,
    protocol: str = "auto",
    depth_axis: int = 2,
) -> dict[str, Any]:
    """Build an image-only manifest enumerating every source frame and Z slice."""
    dataset = str(dataset).lower()
    if dataset not in SUPPORTED_DATASETS:
        raise ValueError(f"dataset must be one of {SUPPORTED_DATASETS}, got {dataset!r}")
    if depth_axis not in (0, 1, 2):
        raise ValueError(f"depth_axis must be 0, 1 or 2, got {depth_axis!r}")
    root_path = Path(root).expanduser()
    if not root_path.exists():
        raise DataRootError(f"dataset root {root_path} does not exist on this machine")
    if not root_path.is_dir():
        raise DataRootError(f"dataset root {root_path} is not a directory")

    entries: list[dict[str, Any]] = []
    official: dict[str, str] = {}
    preprocessed_roots = False
    skipped_unreadable: list[dict[str, str]] = []
    progress = current_progress()

    with progress.stage("discovery.list_files", root=str(root_path), dataset=dataset):
        image_paths = list(iter_image_files(root_path))
        progress.event(
            "discovery.files_listed",
            root=str(root_path),
            dataset=dataset,
            files_total=len(image_paths),
        )
    progress.update(
        operation="discovery",
        dataset=dataset,
        root=str(root_path),
        files_total=len(image_paths),
    )

    for file_index, path in enumerate(image_paths, start=1):
        progress.update(
            operation="discovery",
            item=str(path),
            file_index=file_index,
            files_total=len(image_paths),
        )
        try:
            with progress.stage(
                "discovery.header",
                path=str(path),
                file_index=file_index,
                files_total=len(image_paths),
            ):
                geometry = read_geometry(path)
                geometry["dataset"] = dataset
                geometry["depth_axis"] = int(depth_axis)
                shape = [int(s) for s in geometry["shape"]]
                if len(shape) not in (3, 4):
                    raise ValueError(f"unsupported array rank {len(shape)}")
                _spatial_shape(shape, depth_axis)
                progress.event(
                    "discovery.header_read",
                    path=str(path),
                    file_index=file_index,
                    files_total=len(image_paths),
                    source_format=geometry.get("source_format"),
                    shape=shape,
                    native_affine=geometry.get("native_affine"),
                    orientation=geometry.get("orientation"),
                    xyzt_units=geometry.get("xyzt_units"),
                    zooms=geometry.get("zooms"),
                )
                entry = _prepare_entry(
                    path,
                    root_path,
                    geometry,
                    depth_axis,
                    dataset,
                    progress=progress,
                    file_index=file_index,
                    files_total=len(image_paths),
                )
        except Exception as exc:
            skipped = {"path": str(path), "reason": f"{type(exc).__name__}: {exc}"}
            skipped_unreadable.append(skipped)
            progress.event("discovery.file_skipped", **skipped)
            continue
        entries.append(entry)
        patient_id = entry["patient_id"]
        if entry["official_folder"] == "test":
            official[patient_id] = "test"
        if entry["preprocessed_root"]:
            preprocessed_roots = True

    counted, duplicates = _dedup(entries, dataset=dataset, progress=progress)
    geometry_checks = _ensure_study_grids(counted, progress=progress)

    patients = sorted({ref["entry"]["patient_id"] for ref in counted})
    split_map, split_provenance = assign_splits(
        patients, dataset=dataset, seed=seed, official=official
    )

    records: list[dict[str, Any]] = []
    cine_eligible_volumes = 0
    for ref in sorted(
        counted,
        key=lambda item: (item["entry"]["relative_path"], item["frame_index"]),
    ):
        entry = ref["entry"]
        geometry = entry["geometry"]
        frames = entry["frames"]
        shape = [int(s) for s in geometry["shape"]]
        spatial_hw = _spatial_shape(shape, depth_axis)
        depth = int(shape[depth_axis])
        frame_index = int(ref["frame_index"])
        cine_eligible, cine_reason = _cine_eligible(frames, geometry)
        cine_eligible_volumes += int(cine_eligible)
        volume_id = ref["volume_id"]
        frame_time = None
        if frames.get("frame_times") is not None:
            frame_time = frames["frame_times"][frame_index]
        record_base = {
            "volume_id": volume_id,
            "unit_id": None,
            "study_id": entry["study_id"],
            "patient_id": entry["patient_id"],
            "dataset": dataset,
            "split": split_map[entry["patient_id"]],
            "path": entry["path"],
            "source_path": entry["path"],
            "relative_path": entry["relative_path"],
            "source_format": geometry["source_format"],
            "shape": shape,
            # W7 receives the three spatial native axes per frame. The full
            # source shape (including T when present) remains in shape/frames.
            "native_shape": [int(s) for s in shape[:3]],
            "dtype": geometry["dtype"],
            # native_hw is the two axes other than the declared depth axis.
            "native_hw": spatial_hw,
            "depth": depth,
            "num_slices": depth,
            "depth_axis": int(depth_axis),
            "frame_axis": frames.get("frame_axis"),
            "slice_index": None,
            "frame_index": frame_index,
            "frame_time": frame_time,
            "frame_selection_rule": (
                "all_acquired_frame_indices_image_only"
                if frames["is_multi_frame"]
                else "single_acquired_frame_index_0_image_only"
            ),
            "native_geometry": geometry["native_geometry"],
            "native_geometry_reason": geometry["native_geometry_reason"],
            "native_affine": geometry["native_affine"],
            "orientation": geometry["orientation"],
            "zooms": geometry["zooms"],
            "xyzt_units": geometry.get("xyzt_units"),
            "spatial_unit": geometry.get("spatial_unit"),
            "spacing_mm": geometry.get("spacing_mm"),
            "spacing_reason": geometry.get("spacing_reason"),
            "spacing_valid": geometry["spacing_valid"],
            "study_grid_compatibility": entry.get("study_grid_compatibility"),
            "export_grid": geometry["export_grid"],
            "native_grid_export": geometry["native_grid_export"],
            "frames": frames,
            "cine_eligible": cine_eligible,
            "cine_reason": cine_reason,
            "acquisition_id": volume_id,
            "source_fingerprint": entry["source_fingerprint"],
            "source_hash": entry["source_fingerprint"],
            "frame_fingerprint": ref["frame_fingerprint"],
            "duplicate_of": None,
            "cohort_provenance": {
                "official_folder": entry["official_folder"],
                "official_test_membership": split_provenance["official_test_membership"],
                "preprocessed_root": entry["preprocessed_root"],
                "annotation_selected_cohort": entry["preprocessed_root"],
                "split_rule": split_provenance["rule"],
                "split_identity": "patient_id",
                "mask_inputs_used": False,
            },
            # This is keyed by study and native grid, never by content hash.
            "partition_id": partition_identity(
                study_id=entry["study_id"],
                height=int(spatial_hw[0]),
                width=int(spatial_hw[1]),
                seed=seed,
            ),
        }
        for slice_index in range(depth):
            record = dict(record_base)
            record["slice_index"] = int(slice_index)
            record["unit_id"] = f"{volume_id}:z{slice_index:04d}"
            records.append(record)

    resolved_protocol, protocol_reason = _resolve_protocol(protocol, cine_eligible_volumes)
    limitations = _limitations(
        records=records,
        preprocessed_roots=preprocessed_roots,
        cine_eligible_total=cine_eligible_volumes,
        split_provenance=split_provenance,
        skipped_unreadable=skipped_unreadable,
    )
    volume_ids = sorted({record["volume_id"] for record in records})
    readiness = {
        "inventory_status": "INSPECTED" if records else "EMPTY",
        "root": str(root_path),
        "n_image_files_seen": len(entries),
        "n_source_files": len(entries),
        "n_volume_candidates": sum(int(e["frames"]["n_frames"]) for e in entries),
        "n_volumes": len(volume_ids),
        "n_records": len(records),
        "n_units": len(records),
        "n_frames_enumerated": sum(int(e["frames"]["n_frames"]) for e in entries),
        "n_slices_enumerated": len(records),
        "n_duplicates_collapsed": len(duplicates),
        "n_patients": len(patients),
        "per_split_patients": {
            name: len({p for p, split in split_map.items() if split == name}) for name in SPLITS
        },
        "per_split_records": {
            name: sum(1 for r in records if r["split"] == name) for name in SPLITS
        },
        "per_split_units": {
            name: sum(1 for r in records if r["split"] == name) for name in SPLITS
        },
        "source_formats": {
            fmt: sum(1 for e in entries if e["geometry"]["source_format"] == fmt)
            for fmt in ("nifti", "npy")
        },
        "native_geometry_available": sum(
            1 for r in records if r["native_geometry"] == "available"
        ),
        "spacing_valid_records": sum(1 for r in records if r["spacing_valid"]),
        "cine_eligible_records": sum(1 for r in records if r["cine_eligible"]),
        "cine_eligible_volumes": cine_eligible_volumes,
        "skipped_unreadable": skipped_unreadable,
        "mask_files_read": 0,
        "annotation_sidecars_read": 0,
        "source_fingerprints": "one_sha256_per_selected_source_file",
        "usable_for_training": bool(
            records and sum(1 for r in records if r["split"] == "train") > 0
        ),
    }
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "contract_version": "maskfree150.v1",
        "dataset": dataset,
        "root": str(root_path),
        "seed": int(seed),
        "requested_protocol": protocol,
        "resolved_protocol": resolved_protocol,
        "protocol_reason": protocol_reason,
        "depth_axis": int(depth_axis),
        "discovery_contract": {
            "version": "maskfree150.data.discovery.v2",
            "source_fingerprint": "sha256_source_file_bytes_once",
            "frame_fingerprint": "blake2b160_shape_dtype_stored_frame_bytes",
            "duplicate_proof": (
                "exact_stored_frame_shape_and_content; exact_geometry_cross_file_exports "
                "or_native_nifti_rank4_cine_to_rank3_frame_export"
            ),
            "grid_check": "pairwise_native_affine_corner_voxel_displacement",
            "grid_check_tolerance_voxels": float(HEADER_GRID_TOLERANCE_VOXELS),
            "no_resampling": True,
        },
        "split_provenance": split_provenance,
        "records": records,
        "duplicates": duplicates,
        "geometry_checks": geometry_checks,
        "readiness": readiness,
        "limitations": limitations,
    }
    manifest["manifest_id"] = manifest_identity(manifest)
    progress.update(
        operation="discovery",
        dataset=dataset,
        files_total=len(image_paths),
        files_read=len(entries),
        files_skipped=len(skipped_unreadable),
        frames_enumerated=readiness["n_frames_enumerated"],
        units_enumerated=readiness["n_units"],
        duplicates_collapsed=readiness["n_duplicates_collapsed"],
        studies_checked=len(geometry_checks["studies"]),
    )
    return manifest


def _resolve_protocol(requested: str, cine_eligible_total: int) -> tuple[str, str]:
    if requested not in ("auto", "spatial_predictive", "cine_predictive"):
        raise ValueError(f"unknown protocol {requested!r}")
    if requested == "cine_predictive":
        raise ProtocolUnavailableError(
            "cine_predictive requires a fit-time temporal transport model; the "
            "observation model implements spatial_predictive only, so the explicit "
            "cine request cannot be silently relabelled"
        )
    if cine_eligible_total > 0:
        reason = (
            f"{cine_eligible_total} volume frame(s) are cine-eligible, but no fit-time "
            "temporal transport model is implemented; resolved to the explicit weaker "
            "spatial appearance experiment rather than claiming cine_predictive"
        )
    else:
        reason = (
            "no source satisfies multi-frame, >= 8 frames, reliable frame times and "
            "valid native geometry; spatial_predictive is the only honest protocol"
        )
    return "spatial_predictive", reason


def _limitations(
    *,
    records: list[dict[str, Any]],
    preprocessed_roots: bool,
    cine_eligible_total: int,
    split_provenance: dict[str, Any],
    skipped_unreadable: list[dict[str, str]],
) -> list[dict[str, str]]:
    items: list[dict[str, str]] = []
    if preprocessed_roots:
        items.append(
            {
                "code": "annotation_selected_cohort",
                "detail": (
                    "Records come from a preprocessed root whose ED/ES frame selection "
                    "required manual masks. Images are used image-only here, but the "
                    "cohort itself is annotation-selected."
                ),
            }
        )
    if any(not r["native_grid_export"] for r in records):
        items.append(
            {
                "code": "stored_grid_export_only",
                "detail": (
                    "Some records have no native affine (for example an NPY container). "
                    "Exports for those records are stored-grid only; native-grid NIfTI "
                    "is unavailable, not approximated."
                ),
            }
        )
    if any(not r["spacing_valid"] for r in records):
        items.append(
            {
                "code": "millimetre_metrics_unavailable",
                "detail": (
                    "Millimetre spacing is available only when a NIfTI header declares "
                    "a recognised spatial unit and positive spatial zooms."
                ),
            }
        )
    if any(r["orientation"] is None for r in records):
        items.append(
            {
                "code": "orientation_unknown",
                "detail": "Orientation codes are unavailable for some records; semantic assignment may remain unresolved.",
            }
        )
    if cine_eligible_total > 0:
        items.append(
            {
                "code": "cine_detected_temporal_transport_unimplemented",
                "detail": (
                    "Genuine multi-frame cine was detected and all frames are enumerated, "
                    "but no fit-time temporal transport model exists; the temporal "
                    "experiment is not performed."
                ),
            }
        )
    else:
        items.append(
            {
                "code": "no_cine_available",
                "detail": "No cine acquisition satisfies the frame/time/geometry requirement; cine is not manufactured from exports.",
            }
        )
    if not split_provenance["official_test_membership"]:
        items.append(
            {
                "code": "synthetic_split_not_a_fresh_test_set",
                "detail": (
                    "No official test folder was found; test is a frozen deterministic "
                    "patient split from image identity only and is not an official or "
                    "previously untouched cohort."
                ),
            }
        )
    if skipped_unreadable:
        items.append(
            {
                "code": "unreadable_files_skipped",
                "detail": f"{len(skipped_unreadable)} selected image file(s) could not be probed; see readiness.",
            }
        )
    return items


def manifest_identity(manifest: dict[str, Any]) -> str:
    """Stable cohort/content identity; no full-content value seeds partitions."""
    payload = {
        "schema_version": manifest["schema_version"],
        "dataset": manifest["dataset"],
        "seed": manifest["seed"],
        "resolved_protocol": manifest["resolved_protocol"],
        "depth_axis": manifest["depth_axis"],
        "split_rule": manifest["split_provenance"]["rule"],
        "records": [
            [
                r["unit_id"],
                r["volume_id"],
                r["patient_id"],
                r["split"],
                r["partition_id"],
                r["relative_path"],
                r["frame_index"],
                r["slice_index"],
                r["shape"],
                r.get("source_fingerprint"),
                r.get("frame_fingerprint"),
            ]
            for r in manifest["records"]
        ],
    }
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.blake2b(blob.encode("utf-8"), digest_size=16).hexdigest()


def save_manifest(manifest: dict[str, Any], path: str | Path) -> Path:
    """Atomically write a manifest."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(target.name + ".tmp")
    tmp.write_text(json.dumps(manifest, indent=2, sort_keys=False), encoding="utf-8")
    tmp.replace(target)
    return target


def load_manifest(path: str | Path) -> dict[str, Any]:
    manifest_path = Path(path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") not in (SCHEMA_VERSION, "maskfree150.data.v1"):
        raise ValueError(
            f"manifest schema {manifest.get('schema_version')!r} is not a supported mask-free schema"
        )
    return manifest
